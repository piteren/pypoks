import copy
import math
from pypaq.lipytools.printout import stamp, ProgBar
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.files import list_dir, prep_folder
from pypaq.mpython.mptools import Que, QMessage
from pypaq.mpython.mpdecor import proc
from torchness.tbwr import TBwr
import random
import statistics
import time
from typing import Dict, List, Tuple, Optional

from envy import DMK_MODELS_FD, PyPoksException
from pologic.potable import QPTable, StepQPTable
from pologic.game_config import GameConfig
from pologic.hand_history import states2HHtexts
from podecide.dmk import RanDMK, FolDMK, HumanDMK
from podecide.tools.update_sync import UpdSync
from podecide.tools.devices_monitor import DEVMonitor
from gui.human_game_gui import HumanGameGUI


def stddev_with_none(values) -> Optional[float]:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def separated_factor(
        a_wonH: Optional[float],
        a_wonH_mean_stddev: Optional[float],
        b_wonH: Optional[float],
        b_wonH_mean_stddev: Optional[float],
) -> float:
    """ separated factor for two player results,
    returned 1.0 - means that wonH_IV (mean won after interval) of both players
    are distanced by sum of their wonH_mean_stddev """
    if a_wonH_mean_stddev is None or b_wonH_mean_stddev is None:
        return 0.0
    if a_wonH_mean_stddev + b_wonH_mean_stddev == 0:
        return 1000
    return abs(a_wonH - b_wonH) / (a_wonH_mean_stddev + b_wonH_mean_stddev)


def separation_report(
        dmk_results: Dict,
        n_stddev: float=                            1.0, # base mean stddev for separation calculation
        sep_pairs: Optional[List[Tuple[str,str]]]=  None,
        max_nf: float=                              1.1,
) -> Dict:
    """ separation report for dmk_results """

    sep_nc = 0.0
    sep_nf = 0.0
    sep_pairs_nc = 0.0
    sep_pairs_nf = 0.0
    sep_pairs_stat: List[float] = []

    n_dmk = len(dmk_results)

    # compute separated normalized count & normalized factor
    sep = {}
    for dn_a in dmk_results:
        sep[dn_a] = n_dmk - 1
        for dn_b in dmk_results:
            if dn_a != dn_b:
                sf = separated_factor(
                    a_wonH=             dmk_results[dn_a]['last_wonH_afterIV'],
                    a_wonH_mean_stddev= dmk_results[dn_a]['wonH_IV_mean_stddev'],
                    b_wonH=             dmk_results[dn_b]['last_wonH_afterIV'],
                    b_wonH_mean_stddev= dmk_results[dn_b]['wonH_IV_mean_stddev'])
                if sf < n_stddev:
                    sep[dn_a] -= 1
                sep_nf += min(sf, max_nf*n_stddev)
        sep_nc += sep[dn_a]
    n_max = (n_dmk - 1) * n_dmk
    sep_nc /= n_max
    sep_nf /= n_max

    # same for given pairs
    if sep_pairs:
        for sp in sep_pairs:
            sf = separated_factor(
                a_wonH=             dmk_results[sp[0]]['last_wonH_afterIV'],
                a_wonH_mean_stddev= dmk_results[sp[0]]['wonH_IV_mean_stddev'],
                b_wonH=             dmk_results[sp[1]]['last_wonH_afterIV'],
                b_wonH_mean_stddev= dmk_results[sp[1]]['wonH_IV_mean_stddev'])
            sep_pairs_stat.append(sf)
            if sf >= n_stddev:
                sep_pairs_nc += 1
            sep_pairs_nf += min(sf, max_nf)
        sep_pairs_nc /= len(sep_pairs)
        sep_pairs_nf /= len(sep_pairs)

    return {
        'sep_nc':           sep_nc,         # <0.0;1.0> normalized count of separated
        'sep_nf':           sep_nf,         # <0.0;1.1> normalized factor of separation
        'sep_pairs_nc':     sep_pairs_nc,   # <0.0;1.0> normalized count of separated pairs
        'sep_pairs_nf':     sep_pairs_nf,   # <0.0;1.1> normalized factor of pairs separation
        'sep_pairs_stat':   sep_pairs_stat} # [0,1, ..] each par marked as separated or not


class GameManager:
    """ manages games of DMKs """

    def __init__(
            self,
            dmks_recipe: List[Tuple[type,Dict]],    # list of (type(DMK), dmk_point)
            game_config: GameConfig,
            name: Optional[str]=    None,
            gm_loop: Optional[int]= None,           # if given sets DMKs to publish loop stats
            seed=                   123,
            logger=                 None,
            loglevel=               20,
            debug_dmks=             False,          # sets DMKs logger into debug mode
            debug_tables=           False,          # sets tables logger into debug mode
    ):

        if name is None:
            name = self.__class__.__name__
        self.name = name

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                folder=     DMK_MODELS_FD,
                level=      loglevel)
        self.logger = logger

        self.logger.info(f'*** {self.__class__.__name__} : {self.name} *** starts..')

        self.game_config = game_config
        self.gm_loop = gm_loop
        self._run_game_called = False
        self.seed = seed
        self.tables = None
        self.debug_tables = debug_tables
        self.que_to_gm = Que()  # here GM receives data from DMKs and Tables

        ### build DMKs

        dmk_typeL = [dr[0] for dr in dmks_recipe]
        dmk_pointL = [copy.deepcopy(dr[1]) for dr in dmks_recipe]

        if self.gm_loop is not None:
            for dmk_point in dmk_pointL:
                dmk_point['collect_loop_stats'] = True

        dmk_logger = get_child(logger=self.logger, name='dmks_logger')
        if debug_dmks: dmk_logger.setLevel(10)

        self.dmkD = {
            dmk.name: dmk
            for dmk in [dmk_type(logger=dmk_logger, **dmk_point) for dmk_type,dmk_point in zip(dmk_typeL, dmk_pointL)]}

        # update DMKs ques
        for dmk in self.dmkD.values():
            dmk.que_to_gm = self.que_to_gm

        self.families = set([dmk.family for dmk in self.dmkD.values()])

    def _get_dmk_focus_names(self) -> List[str]:
        """ prepares list of DMK names GM is focused on while preparing dmk_results
        here all DMKs """
        return list(self.dmkD.keys())

    def _put_players_on_tables(self):
        """ creates tables and puts players on (with some baseline policy) """

        self.logger.debug('> puts players on tables..')

        # build dict of lists of players (per family): {family: [(pid, que_to_pl, que_from_pl)]}
        fam_ques: Dict[str, List[Tuple[str,Que,Que]]] = {fam: [] for fam in self.families}
        for dmk in self.dmkD.values():
            for k in dmk.queD_to_player: # {pid: que_to_pl}
                fam_ques[dmk.family].append((k, dmk.queD_to_player[k], dmk.que_from_player))

        # shuffle players in families
        for fam in fam_ques:
            random.shuffle(fam_ques[fam])

        quesLL = [fam_ques[fam] for fam in fam_ques] # convert to list of lists

        ### convert to flat list

        # cut in equal pieces
        min_len = min([len(l) for l in quesLL])
        cut_quesLL = []
        for l in quesLL:
            while len(l) > 1.66*min_len:
                cut_quesLL.append(l[:min_len])
                l = l[min_len:]
            cut_quesLL.append(l)
        quesLL = cut_quesLL
        random.shuffle(quesLL)
        random.shuffle(quesLL)

        quesL = []  # flat list
        qLL_IXL = []
        while quesLL:

            if not qLL_IXL:
                qLL_IXL = list(range(len(quesLL)))      # fill indexes
                random.shuffle(qLL_IXL)                 # shuffle them
            qLL_IX = qLL_IXL.pop()                      # now take last index

            quesL.append(quesLL[qLL_IX].pop())          # add last from list
            if not quesLL[qLL_IX]:
                quesLL.pop(qLL_IX)                      # remove empty list
                qLL_IXL = list(range(len(quesLL)))      # new indexes then
                random.shuffle(qLL_IXL)                 # shuffle them

        num_players = len(quesL)
        if num_players % self.game_config.table_size != 0:
            raise PyPoksException(f"num_players ({num_players}) has to be a multiple of table_size ({self.game_config.table_size})")

        # create table and put players on
        self.tables = []
        table_ques = []
        table_logger = get_child(self.logger, name='table_logger')
        if self.debug_tables: table_logger.setLevel(10)
        while quesL:
            table_ques.append(quesL.pop())
            if len(table_ques) == self.game_config.table_size:
                self.tables.append(QPTable(
                    name=           f'tbl{len(self.tables)}',
                    game_config=    self.game_config,
                    que_to_gm=      self.que_to_gm,
                    pl_ques=        {t[0]: (t[1], t[2]) for t in table_ques},
                    logger=         table_logger))
                table_ques = []

    def _start_tables(self):
        self.logger.debug(f'GM starts {len(self.tables)} tables..')
        for tbl in self.tables:
            tbl.start()
        for _ in self.tables:
            self.que_to_gm.get()
        self.logger.debug(f'> all tables processes started!')

    def _start_dmks_processes(self):
        self.logger.debug(f'GM starts {len(self.dmkD)} DMKs processes..')
        for dmk in self.dmkD.values():
            dmk.start()
        for _ in self.dmkD:
            message = self.que_to_gm.get()
            self.logger.debug(f'> {message.type}: {message.data}')
        self.logger.debug(f'> all DMKs processes started!')

    def _start_dmks_loops(self):
        self.logger.debug(f'GM starts {len(self.dmkD)} DMKs loops..')
        message = QMessage(type='start_dmk_loop', data=None)
        for dmk in self.dmkD.values():
            dmk.que_from_gm.put(message) # synchronizes DMKs a bit..
        for _ in self.dmkD:
            message = self.que_to_gm.get()
            self.logger.debug(f'> {message.type}: {message.data}')
        self.logger.debug(f'> all DMKs loops started!')

    def _save_dmks(self, only_trainable=True):
        self.logger.debug('GM saves DMKs')
        n_saved = 0
        message = QMessage(type='save_dmk', data=None)
        for dmk in self.dmkD.values():
            if (dmk.trainable and only_trainable) or not only_trainable:
                dmk.que_from_gm.put(message)
                n_saved += 1
        for _ in range(n_saved):
            self.que_to_gm.get()
        self.logger.debug('> all DMKs saved!')

    def _stop_tables(self):
        self.logger.debug('GM stops tables loops..')
        message = QMessage(type='stop_table', data=None)
        for table in self.tables:
            table.que_from_gm.put(message)
        for _ in self.tables:
            self.que_to_gm.get()
        self.logger.debug('> all tables loops stopped!')

    def _stop_dmks_loops(self):
        self.logger.debug(f'GM stops {len(self.dmkD)} DMKs loops..')
        message = QMessage(type='stop_dmk_loop', data=None)
        for dmk in self.dmkD.values():
            dmk.que_from_gm.put(message)
        for _ in self.dmkD:
            self.que_to_gm.get()
        self.logger.debug('> all DMKs loops stopped!')

    def _stop_dmks_processes(self):
        self.logger.debug(f'GM stops {len(self.dmkD)} DMKs processes..')
        message = QMessage(type='stop_dmk_process', data=None)
        for dmk in self.dmkD.values():
            dmk.que_from_gm.put(message)
        for _ in self.dmkD:
            self.que_to_gm.get()
        self.logger.debug('> all DMKs processes stopped!')

    def run_game(
            self,
            game_size=                                  10000,  # number of hands for a game (per DMK)
            sleep=                                      10,     # loop sleep (seconds)
            progress_report=                            True,
            publish=                                    True,   # publish GM to TB
            sep_all_break: bool=                        False,  # breaks game when all DMKs are separated
            sep_pairs: Optional[List[Tuple[str,str]]]=  None,   # pairs of DMK names for separation condition
            sep_pairs_factor: float=                    0.9,    # factor of separated pairs needed to break the game
            sep_n_stddev: float=                        1.0,
            sep_min_IV: int=                            10,     # minimal number of IV to enable any sep break
    ) -> Dict[str, Dict]:
        """ runs game, returns DMK results dictionary

        By now, by design run_game() may be called only once,
        cause DMK processes are started and then stopped and process cannot be started twice,
        there is no real need to change this design.
        """

        if self._run_game_called:
            raise PyPoksException('GM cannot run_game() more than once!')
        else:
            self._run_game_called = True

        tbwr = TBwr(logdir=f'{DMK_MODELS_FD}/{self.name}') if publish else None

        # UpdSync will be used if any trainable DMKs found
        got_trainable = any([d.trainable for d in self.dmkD.values()])
        upd_sync = UpdSync(
            dmkL=       list(self.dmkD.values()),
            tb_name=    f'UpdSync_{self.name}' if publish else None,
            logger=     get_child(self.logger)) if got_trainable else None

        dev_monitor = DEVMonitor(tb_name=f'DEVMon_{self.name}') if publish else None

        # save of DMK results + info (for focused only)
        dmk_results = {
            dn: {
                'wonH_IV':              [],     # wonH (won $ / hand) of interval
                'wonH_afterIV':         [],     # wonH (won $ / hand) after interval
                'last_wonH_afterIV':    0,
                'wonH_IV_stddev':       None,
                'wonH_IV_mean_stddev':  None,
                'family':               self.dmkD[dn].family,
                'trainable':            self.dmkD[dn].trainable,
                'global_stats':         None,   # SM.global_stats, will be updated at the end of the game
            } for dn in self._get_dmk_focus_names()}

        # start all subprocesses
        self._put_players_on_tables()
        self._start_tables()
        self._start_dmks_processes()
        self._start_dmks_loops()

        stime = time.time()
        time_last_report = stime
        n_hands_last_report = 0

        self.logger.info(f'> {self.name} starts a game..')
        pb = ProgBar(total=1.0)
        loop_ix = 0
        while True:

            time.sleep(sleep)

            # retrieve and process actual DMK reports
            reports = self._get_reports({dn: len(dmk_results[dn]['wonH_IV']) for dn in dmk_results})
            num_IV = [] # number of wonH_IV, for each DMK
            for dn in reports:
                dmk_results[dn]['wonH_IV'] += reports[dn]['wonH_IV']
                dmk_results[dn]['wonH_afterIV'] += reports[dn]['wonH_afterIV']
                wonH_IV_stddev = stddev_with_none(dmk_results[dn]['wonH_IV'])
                dmk_results[dn]['wonH_IV_stddev'] = wonH_IV_stddev
                dmk_results[dn]['wonH_IV_mean_stddev'] = wonH_IV_stddev / math.sqrt(len(dmk_results[dn]['wonH_IV'])) if wonH_IV_stddev is not None else None
                dmk_results[dn]['last_wonH_afterIV'] = dmk_results[dn]['wonH_afterIV'][-1] if dmk_results[dn]['wonH_afterIV'] else None
                num_IV.append(len(dmk_results[dn]['wonH_IV']))
            min_num_IV = min(num_IV) # lowest number of wonH_IV

            # calculate game factor
            nhL = [reports[dn]['n_hands'] for dn in reports]
            n_hands = sum(nhL)      # total
            n_hands_min = min(nhL)  # minimal
            game_factor = n_hands_min / game_size
            if game_factor >= 1:
                game_factor = 1

            sep_nc, sep_nf, sep_pairs_nc, sep_pairs_nf = 0,0,0,0
            if len(dmk_results) > 1: # cannot compute separation for single DMK

                sr = separation_report(
                    dmk_results=    dmk_results,
                    n_stddev=       sep_n_stddev,
                    sep_pairs=      sep_pairs)
                sep_nc = sr['sep_nc']
                sep_nf = sr['sep_nf']
                sep_pairs_nc = sr['sep_pairs_nc']
                sep_pairs_nf = sr['sep_pairs_nf']

                if tbwr:
                    tbwr.add(value=sep_nc, tag=f'GM/sep_nc', step=loop_ix)
                    tbwr.add(value=sep_nf, tag=f'GM/sep_nf', step=loop_ix)
                    if sep_pairs:
                        tbwr.add(value=sep_pairs_nc, tag=f'GM/sep_pairs_nc', step=loop_ix)
                        tbwr.add(value=sep_pairs_nf, tag=f'GM/sep_pairs_nf', step=loop_ix)

            # INFO: progress relies on reports, and reports may be prepared in custom way (overridden) by diff GMs
            if progress_report:

                # progress
                passed = (time.time()-stime)/60
                left_nfo = ' - '
                if game_factor > 0:
                    full_time = passed / game_factor
                    left = (1-game_factor) * full_time
                    left_nfo = f'{left:.1f}'

                # speed
                hdiff = n_hands-n_hands_last_report # number of hand played by focused DMKs since last report
                n_hands_last_report = n_hands
                hspeed = hdiff / (time.time()-time_last_report)
                time_last_report = time.time()

                sep_report_pairs = f'::{sep_pairs_nc:.2f}[{sep_pairs_nf:.2f}]' if sep_pairs else ''

                pb(current= game_factor,
                   prefix=  f'GM: {passed:.1f}min left:{left_nfo}min',
                   suffix=  f'{int(hspeed)}H/s -- SEP:{sep_nc:.2f}[{sep_nf:.2f}]{sep_report_pairs}')

                if tbwr: tbwr.add(value=hspeed, tag=f'GM/speedH/s', step=loop_ix)

                # not using report here, but triggers TB publish
                if dev_monitor:
                    dev_monitor.get_report()

            # games break - factor condition
            if game_factor == 1:
                fin_condition = 'game factor'
                break

            # games break - all DMKs separation condition
            if sep_all_break and min_num_IV >= sep_min_IV and sep_nc == 1.0:
                fin_condition = f'all DMKs separation, factor:{game_factor:.2f}'
                break

            # games break - pairs separation breaking value condition
            if sep_pairs and min_num_IV >= sep_min_IV and sep_pairs_nc >= sep_pairs_factor:
                fin_condition = f'pairs separation {sep_pairs_factor:.2f}, factor:{game_factor:.2f}'
                break

            loop_ix += 1

        if tbwr:
            tbwr.flush()

        if upd_sync:
            upd_sync.stop()

        if dev_monitor:
            dev_monitor.stop()

        self._stop_tables()
        self._stop_dmks_loops()

        if self.gm_loop is not None:
            dmk_names = list(dmk_results.keys())
            dmk_names.sort(key=lambda x:dmk_results[x]['last_wonH_afterIV'], reverse=True)
            pos = {dn: ix for ix,dn in enumerate(dmk_names)}
            for dn in dmk_results:
                message = QMessage(
                    type=   'publish_loop_stats',
                    data=   {
                        'step':             self.gm_loop,
                        'position':         pos[dn],
                        'wonH':             dmk_results[dn]['last_wonH_afterIV'],
                        'wonH_IV_stddev':   dmk_results[dn]['wonH_IV_stddev']})
                self.dmkD[dn].que_from_gm.put(message)

        message = QMessage(type='send_global_stats', data=None)
        for dn in dmk_results:
            self.dmkD[dn].que_from_gm.put(message)
        for _ in dmk_results:
            message = self.que_to_gm.get()
            data = message.data
            dmk_name = data.pop('dmk_name')
            dmk_results[dmk_name]['global_stats'] = data['global_stats']

        self._save_dmks()
        self._stop_dmks_processes()

        taken_sec = time.time() - stime
        taken_nfo = f'{taken_sec / 60:.1f}min' if taken_sec > 100 else f'{taken_sec:.1f}sec'
        speed = n_hands / taken_sec
        self.logger.info(f'{self.name} finished run_game (condition: {fin_condition}), avg speed: {speed:.1f}H/s, time taken: {taken_nfo}')
        loop_stats = {'speed': speed}

        return {
            'dmk_results':  dmk_results,
            'loop_stats':   loop_stats}

    def _get_reports(
            self,
            dmk_report_IV:Dict[str,int] # {dn: from_IV}
    ) -> Dict[str, Dict]:
        """ asks DMKs to send reports, but only form given IV """
        reports: Dict[str, Dict] = {} # {dn: {n_hands, wonH_IV, wonH_afterIV}}
        for dn,from_IV in dmk_report_IV.items():
            message = QMessage(type='send_dmk_report', data=from_IV)
            self.dmkD[dn].que_from_gm.put(message)
        for _ in dmk_report_IV:
            message = self.que_to_gm.get()
            report = message.data
            dmk_name = report.pop('dmk_name')
            reports[dmk_name] = report
        return reports


class GameManager_PTR(GameManager):
    """ GameManager for "PLay & TRain With Ref" concept for FolDMKs where DMKs may play or train against ref """

    def __init__(
            self,
            game_config: GameConfig,
            dmk_point_refL: Optional[List[Dict]]=   None, # playable reference DMK list
            dmk_point_PLL: Optional[List[Dict]]=    None, # playable DMK list
            dmk_point_TRL: Optional[List[Dict]]=    None, # trainable DMK list
            n_tables: int=                          1000, # desired number of tables, may be adjusted up
            name: Optional[str]=                    None,
            seed=                                   123,
            **kwargs):
        """
        there are 4 possible setups:
        1. only dmk_point_TRL             -> all DMKs are training (against each other)
        2. only dmk_point_PLL             -> all DMKs are playing (against each other)
        3. dmk_point_TRL & dmk_point_refL -> DMKs_TRL are training against DMKs_ref
        4. dmk_point_PLL & dmk_point_refL -> DMKs_PLL are playing against DMKs_ref
        """

        if not (dmk_point_PLL or dmk_point_TRL):
            raise PyPoksException('playing OR training DMKs must be given')

        if dmk_point_PLL and dmk_point_TRL:
            raise PyPoksException('playing AND training DMKs cannot be given')

        if not dmk_point_refL: dmk_point_refL = []
        if not dmk_point_PLL: dmk_point_PLL = []
        if not dmk_point_TRL: dmk_point_TRL = []

        self.dmk_name_refL = [dna['name'] for dna in dmk_point_refL]
        self.dmk_name_PLL = [dna['name'] for dna in dmk_point_PLL]
        self.dmk_name_TRL = [dna['name'] for dna in dmk_point_TRL]

        n_dmk_players, n_tables = self._get_num_players_tables(
            n_tables=   n_tables,
            n_dmk=      len(dmk_point_PLL + dmk_point_TRL),
            table_size= game_config.table_size,
            is_ref=     bool(dmk_point_refL))

        ### update points

        for dmk in dmk_point_TRL:
            dmk.update({
                'n_players': n_dmk_players,
                'trainable': True})

        for dmk in dmk_point_PLL:
            dmk.update({
                'n_players': n_dmk_players,
                'trainable': False})

        # ref_pattern is a list of DMK names from refs that will be used to seat players at tables,
        # each DMK from TRL or PLL will get same tables with ref players seated from this pattern
        self.ref_pattern = []
        if dmk_point_refL:
            n_ref_players_for_one = n_dmk_players * (game_config.table_size-1)                              # number of refs players required by one TRL or PLL player
            ref_pattern_single_dmk = (self.dmk_name_refL * n_ref_players_for_one)[:n_ref_players_for_one]   # first longer, then trim

            random.seed(seed)                                                                               # to have same list randomized always with same seed
            random.shuffle(ref_pattern_single_dmk)                                                          # adds randomness, but repeated since seed fixed

            self.ref_pattern = ref_pattern_single_dmk * (len(dmk_point_TRL + dmk_point_PLL))

            for dmk in dmk_point_refL:
                dmk.update({
                    'n_players': len([nm for nm in self.ref_pattern if nm == dmk['name']]),
                    'trainable': False})

        gc_params = {
            'table_size':       game_config.table_size,
            'table_moves':      game_config.table_moves,
            'table_cash_start': game_config.table_cash_start}
        for dmk_point in dmk_point_refL + dmk_point_PLL + dmk_point_TRL:
            for k in gc_params:
                if k in dmk_point:
                    self.logger.warning(f'key: {k} already present in dmk_point with value: {dmk_point[k]} <- overriding with: {gc_params[k]}')
                dmk_point[k] = gc_params[k]

        nm = 'PL' if self.dmk_name_PLL else 'TR'
        if dmk_point_refL:
            nm += '_ref'
        GameManager.__init__(
            self,
            dmks_recipe=    [(FolDMK,point) for point in dmk_point_refL + dmk_point_PLL + dmk_point_TRL],
            game_config=    game_config,
            name=           name or f'GM_{nm}_{stamp(letters=None)}',
            seed=           seed,
            **kwargs)

        self.logger.info(f'*** GameManager_PTR initialized with (PL:{len(dmk_point_PLL)} TR:{len(dmk_point_TRL)} ref:{len(dmk_point_refL)}) DMKs on {n_tables} tables')
        for dna in dmk_point_PLL + dmk_point_TRL:
            self.logger.debug(f'> {dna["name"]} with {dna["n_players"]} players, trainable: {dna["trainable"]}')

    @staticmethod
    def _get_num_players_tables(n_tables:int, n_dmk:int, table_size:int, is_ref:bool) -> Tuple[int,int]:
        """ computes proper number of players per DMK and number of tables
        for given expected number of tables """

        if is_ref:
            n_dmk_players = math.ceil(n_tables / n_dmk)             # adjusted up number of players per DMK
            n_tables = n_dmk_players * n_dmk                        # adjusted up number of tables
        else:
            n_all_players = n_tables * table_size                   # target number of all players
            lcm = math.lcm(table_size,n_dmk)
            n_all_players = math.ceil(n_all_players / lcm) * lcm    # adjusted up number of all players
            n_dmk_players = int(n_all_players / n_dmk)              # number of players per DMK
            n_tables = int(n_all_players / table_size)              # number of tables

        return n_dmk_players, n_tables

    def _put_players_on_tables(self):
        """ creates new tables & puts players with PTR policy """

        # use original GM policy
        if not self.dmk_name_refL:
            return GameManager._put_players_on_tables(self)

        self.logger.debug('> puts players on tables with PTR policy..')

        ques_refD = {}
        ques_rest = []

        for dmk_name in self.dmkD:

            dmk = self.dmkD[dmk_name]

            # select target
            ques = ques_rest
            if dmk_name in self.dmk_name_refL:
                ques_refD[dmk_name] = []
                ques = ques_refD[dmk_name]

            for k in dmk.queD_to_player: # {pid: que_to_pl}
                ques.append((k, dmk.queD_to_player[k], dmk.que_from_player))

        # put on tables
        self.tables = []
        table_ques =  []
        table_logger = get_child(self.logger, name='table_logger', change_level=-10 if self.debug_tables else 10)
        rp_ix = 0
        while ques_rest:

            table_ques.append(ques_rest.pop())
            while len(table_ques) < self.game_config.table_size:
                table_ques.append(ques_refD[self.ref_pattern[rp_ix]].pop())
                rp_ix += 1

            self.tables.append(QPTable(
                name=           f'tbl{len(self.tables)}',
                game_config=    self.game_config,
                que_to_gm=      self.que_to_gm,
                pl_ques=        {t[0]: (t[1], t[2]) for t in table_ques},
                logger=         table_logger))
            table_ques = []

    def run_game(self, **kwargs) -> Dict:
        """ adds age update to dmk_results """

        # update trainable age - needs to be done before game, cause after the game DMKs are saved
        for dmk in self.dmkD.values():
            if dmk.trainable:
                dmk.age += 1

        rgd = GameManager.run_game(self, **kwargs)

        for dn in rgd['dmk_results']:
            rgd['dmk_results'][dn]['age'] = self.dmkD[dn].age
        return rgd

    def _get_dmk_focus_names(self) -> List[str]:
        """ GamesManager_PTR is focused on TRL or PLL """
        return self.dmk_name_TRL or self.dmk_name_PLL


class HumanGameManager(GameManager):
    """ HumanGameManager runs game of human vs AI agent with GUI """

    HUMAN_NAME = 'human'

    def __init__(
            self,
            dmk_names: List[str],
            game_config: GameConfig,
            save_hands_FD=  'hg_hands',
            **kwargs,
    ):

        if len(dmk_names) != game_config.table_size - 1:
            err = f"number of given DMK names should be equal table_size - 1 ({game_config.table_size-1})"
            self.logger.error(err)
            raise PyPoksException(err)

        self.save_hands_FD = save_hands_FD

        dmk_human_point = {
            'name':             self.HUMAN_NAME,
            'table_size':       game_config.table_size,
            'table_moves':      game_config.table_moves,
            'fwd_stats_step':   10}

        dmk_pointL = [{
            'name':             nm,
            'trainable':        False,
            'n_players':        1,
            'motorch_point':    {'device': None},
            'table_size':       game_config.table_size,
            'table_moves':      game_config.table_moves,
            'table_cash_start': game_config.table_cash_start,
            'fwd_stats_step':   10,
        } for nm in dmk_names]

        GameManager.__init__(
            self,
            dmks_recipe=    [(HumanDMK,dmk_human_point)] + [(FolDMK,point) for point in dmk_pointL],
            game_config=    game_config,
            **kwargs)

        self.gui = HumanGameGUI(
            players=            [self.HUMAN_NAME] + dmk_names,
            game_config=        game_config,
            imgs_FD=            'gui/imgs',
            logger=             get_child(self.logger))

        self.dmkD[self.HUMAN_NAME].gui_queI = self.gui.queI
        self.dmkD[self.HUMAN_NAME].gui_queO = self.gui.queO

    def _put_players_on_tables(self):

        ques = []
        for dmk in self.dmkD.values():
            for k in dmk.queD_to_player:
                ques.append((k, dmk.queD_to_player[k], dmk.que_from_player))

        table_logger = get_child(self.logger, name='table_logger')
        if self.debug_tables: table_logger.setLevel(10)

        self.tables = [StepQPTable(
            name=           'table',
            game_config=    self.game_config,
            que_to_gm=      self.que_to_gm,
            pl_ques=        {t[0]: (t[1], t[2]) for t in ques},
            logger=         table_logger)]

    @proc
    def start_gui(self):
        self.gui.run_loop()
        self.logger.debug('GUI exited')

    def start_game(self, save_hands=True):

        if self._run_game_called: raise PyPoksException('GM cannot run_game() more than once!')
        else: self._run_game_called = True

        if save_hands:
            prep_folder(self.save_hands_FD)

        self.start_gui()

        # start all subprocesses
        self._put_players_on_tables()
        self._start_tables()
        self._start_dmks_processes()
        self._start_dmks_loops()

        hix = 0
        while True:

            message_gm = QMessage(type='run_hand', data=None)
            self.tables[0].que_from_gm.put(message_gm)
            
            message_table = self.que_to_gm.get()
            hh = message_table.data
            print(f'\n{hh}')

            if save_hands:
                hh.save(f'{self.save_hands_FD}/hand_{hix:03}.jsonl')

            message_gui = self.gui.que_to_gm.get()
            if message_gui.type == 'exit':
                break

            hix += 1

            for dn in self.dmkD:
                if dn != self.HUMAN_NAME:
                    message = QMessage(type='reset_fwd_state')
                    self.dmkD[dn].que_from_gm.put(message)
            for dn in self.dmkD:
                if dn != self.HUMAN_NAME:
                    message = self.que_to_gm.get()
                    print(message)

        self._stop_tables()
        self._stop_dmks_loops()
        self._stop_dmks_processes()


class TestGameManager(GameManager):
    """ TestGameManager is an extract of GamesManager needed to run test game vs AI agent
    by now only single agent game is supported """

    def __init__(
            self,
            dmk_agent_name: str,
            game_config: GameConfig,
            **kwargs,
    ):

        self.dmk_agent_name = dmk_agent_name

        dmk_agent_point = {
            'name':             dmk_agent_name,
            'trainable':        False,
            'n_players':        1,
            'motorch_point':    {'device': None},
            'table_size':       game_config.table_size,
            'table_moves':      game_config.table_moves,
            'table_cash_start': game_config.table_cash_start,
            'fwd_stats_step':   10}

        dmk_random_point = {
            'name':             'human',
            'trainable':        False,
            'n_players':        game_config.table_size-1,
            'table_size':       game_config.table_size,
            'table_moves':      game_config.table_moves,
            'fwd_stats_step':   10}

        GameManager.__init__(
            self,
            dmks_recipe=    [(FolDMK,dmk_agent_point),(RanDMK,dmk_random_point)],
            game_config=    game_config,
            **kwargs)

    def _put_players_on_tables(self):

        ques = []
        for dmk in self.dmkD.values():
            for k in dmk.queD_to_player:
                ques.append((k, dmk.queD_to_player[k], dmk.que_from_player))

        table_logger = get_child(self.logger, name='table_logger')
        if self.debug_tables: table_logger.setLevel(10)

        self.tables = [StepQPTable(
                name=           'table',
                game_config=    self.game_config,
                que_to_gm=      self.que_to_gm,
                pl_ques=        {t[0]: (t[1], t[2]) for t in ques},
                logger=         table_logger)]

    def run_test_game(self, test_dir:str='hg_hands'):

        if self._run_game_called: raise PyPoksException('GM cannot run_game() more than once!')
        else: self._run_game_called = True

        # start all subprocesses
        self._put_players_on_tables()
        self._start_tables()
        self._start_dmks_processes()
        self._start_dmks_loops()

        hh_files = list_dir(test_dir)['files']
        hh_files = sorted([fn for fn in hh_files if fn.endswith('.txt')])
        self.logger.debug(f'got {len(hh_files)} HH files')
        prep_folder(f'{test_dir}/out')

        for hh_file in hh_files:

            self.logger.debug(f'running {hh_file}..')

            with open(f'{test_dir}/{hh_file}') as file:
                hh_strL = [l[:-1] for l in file]
            hh_strL = [l.replace('agent', self.dmk_agent_name) for l in hh_strL]

            hh_gcf = hh_strL[0].split()[1]
            if hh_gcf != self.game_config.name:
                raise PyPoksException(f'given HH GCF -> {hh_gcf} != {self.game_config.name} <- current game config')

            message_gm = QMessage(type='run_hand', data=hh_strL)
            self.tables[0].que_from_gm.put(message_gm)

            message_table = self.que_to_gm.get()
            hh_out = message_table.data
            texts = states2HHtexts(hh_out.events, rename={self.dmk_agent_name:'agent'}, add_probs=True)
            with open(f'{test_dir}/out/{hh_file[:-4]}_out.txt', 'w') as file:
                file.write('\n'.join(texts))

            message = QMessage(type='reset_fwd_state')
            self.dmkD[self.dmk_agent_name].que_from_gm.put(message)
            message = self.que_to_gm.get()
            print(message)

        self._stop_tables()
        self._stop_dmks_loops()
        self._stop_dmks_processes()

        self.logger.info('TestGamesManager finished.')