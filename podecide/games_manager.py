import copy
import math
from pypaq.lipytools.printout import stamp, progress_
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.mpython.mptools import Que, QMessage
from torchness.tbwr import TBwr
import random
import statistics
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

from envy import MODELS_FD, DMK_MODELS_FD, N_TABLE_PLAYERS, PyPoksException
from pologic.potable import QPTable
from podecide.dmk import FolDMK, HuDMK
from gui.gui_hdmk import GUI_HDMK



def stdev_with_none(values) -> Optional[float]:
    if len(values) < 2:
        return None
    return statistics.stdev(values)

# separated factor for two results
def separated_factor(
        a_wonH: Optional[float],
        a_wonH_mean_stdev: Optional[float],
        b_wonH: Optional[float],
        b_wonH_mean_stdev: Optional[float],
        n_stdev: float) -> float:
    if a_wonH_mean_stdev is None or b_wonH_mean_stdev is None:
        return 0.0
    if a_wonH_mean_stdev + b_wonH_mean_stdev == 0:
        return 1000
    return abs(a_wonH - b_wonH) / (n_stdev * (a_wonH_mean_stdev + b_wonH_mean_stdev))

# prepares separation report
def separation_report(
        dmk_results: Dict,
        n_stdev: float,
        sep_pairs: Optional[List[Tuple[str,str]]]=  None,
        max_nf: float=                              1.1,
) -> Dict:

    sep_nc = 0.0
    sep_nf = 0.0
    sep_pairs_nc = 0.0
    sep_pairs_nf = 0.0
    sep_pairs_stat = []

    n_dmk = len(dmk_results)

    # compute separated normalized count & normalized factor
    sep = {}
    for dn_a in dmk_results:
        sep[dn_a] = n_dmk - 1
        for dn_b in dmk_results:
            if dn_a != dn_b:
                sf = separated_factor(
                    a_wonH=             dmk_results[dn_a]['last_wonH_afterIV'],
                    a_wonH_mean_stdev=  dmk_results[dn_a]['wonH_IV_mean_stdev'],
                    b_wonH=             dmk_results[dn_b]['last_wonH_afterIV'],
                    b_wonH_mean_stdev=  dmk_results[dn_b]['wonH_IV_mean_stdev'],
                    n_stdev=            n_stdev)
                if sf < 1:
                    sep[dn_a] -= 1
                sep_nf += min(sf, max_nf)
        sep_nc += sep[dn_a]
    n_max = (n_dmk - 1) * n_dmk
    sep_nc /= n_max
    sep_nf /= n_max

    # same for given pairs
    if sep_pairs:
        for sp in sep_pairs:
            sf = separated_factor(
                a_wonH=             dmk_results[sp[0]]['last_wonH_afterIV'],
                a_wonH_mean_stdev=  dmk_results[sp[0]]['wonH_IV_mean_stdev'],
                b_wonH=             dmk_results[sp[1]]['last_wonH_afterIV'],
                b_wonH_mean_stdev=  dmk_results[sp[1]]['wonH_IV_mean_stdev'],
                n_stdev=            n_stdev)
            sep_pairs_stat.append(0 if sf<1 else 1)
            if sf>=1: sep_pairs_nc += 1
            sep_pairs_nf += min(sf, max_nf)
        sep_pairs_nc /= len(sep_pairs)
        sep_pairs_nf /= len(sep_pairs)

    return {
        'sep_nc':           sep_nc,         # <0.0;1.0> normalized count of separated
        'sep_nf':           sep_nf,         # <0.0;1.1> normalized factor of separation
        'sep_pairs_nc':     sep_pairs_nc,   # <0.0;1.0> normalized count of separated pairs
        'sep_pairs_nf':     sep_pairs_nf,   # <0.0;1.1> normalized factor of pairs separation
        'sep_pairs_stat':   sep_pairs_stat} # [0,1, ..] each par marked as separated or not


# manages games of DMKs (at least QueDMKs)
class GamesManager:

    def __init__(
            self,
            dmk_pointL: List[Dict],         # points with eventually added 'dmk_type'
            name: Optional[str]=    None,
            logger=                 None,
            loglevel=               20,
            debug_dmks=             False,
            debug_tables=           False):

        self.name = name or f'GM_{stamp()}'

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                folder=     MODELS_FD,
                level=      loglevel)
        self.logger = logger
        self.debug_tables = debug_tables

        self.logger.info(f'*** GamesManager : {self.name} *** starts..')

        self.que_to_gm = Que()  # here GM receives data from DMKs and Tables

        dmk_pointL = copy.deepcopy(dmk_pointL) # copy to not modify original list
        dmk_types = [point.pop('dmk_type',FolDMK) for point in dmk_pointL]
        dmk_logger = get_child(self.logger, name='dmks_logger', change_level=-10 if debug_dmks else 10)
        dmks = [dmk_type(logger=dmk_logger, **point) for dmk_type,point in zip(dmk_types, dmk_pointL)]
        self.dmkD = {dmk.name: dmk for dmk in dmks} # Dict[str, dmk_type] INFO:is not typed because DMK may have diff types

        for dmk in self.dmkD.values(): dmk.que_to_gm = self.que_to_gm  # DMKs are build from folders, they need que to be updated then
        self.families = set([dmk.family for dmk in self.dmkD.values()])

        self.tbwr = TBwr(logdir=f'{DMK_MODELS_FD}/{self.name}')

        self.tables = None

    # starts DMKs (starts loops)
    def _start_dmks(self):

        self.logger.debug('> starts DMKs..')

        idmk = tqdm(self.dmkD.values()) if self.logger.level<20 else self.dmkD.values()
        for dmk in idmk: dmk.start()
        self.logger.debug('> initializing..')
        idmk = tqdm(self.dmkD) if self.logger.level < 20 else self.dmkD
        for _ in idmk:
            message = self.que_to_gm.get()
            self.logger.debug(f'>> {message}')
        self.logger.debug(f'> initialized {len(self.dmkD)} DMKs!')

        message = QMessage(type='start_dmk_loop', data=None)
        for dmk in self.dmkD.values(): dmk.que_from_gm.put(message) # synchronizes DMKs a bit..
        for _ in self.dmkD:
            message = self.que_to_gm.get()
            self.logger.debug(f'>> {message}')
        self.logger.debug(f'> started {len(self.dmkD)} DMKs!')

    def _save_dmks(self):

        self.logger.debug('> saves DMKs')

        n_saved = 0
        message = QMessage(type='save_dmk', data=None)
        for dmk in self.dmkD.values():
            dmk.que_from_gm.put(message)
            n_saved += 1
        for _ in range(n_saved):
            self.que_to_gm.get()
        self.logger.debug('> all DMKs saved!')

    # stops DMKs loops
    def _stop_dmks_loops(self):
        self.logger.debug('Stopping DMKs loops..')
        message = QMessage(type='stop_dmk_loop', data=None)
        for dmk in self.dmkD.values(): dmk.que_from_gm.put(message)
        idmk = tqdm(self.dmkD) if self.logger.level < 20 else self.dmkD
        for _ in idmk:
            self.que_to_gm.get()
        self.logger.debug('> all DMKs loops stopped!')

    # stops DMKs processes
    def _stop_dmks_processes(self):
        self.logger.debug('Stopping DMKs processes..')
        message = QMessage(type='stop_dmk_process', data=None)
        for dmk in self.dmkD.values(): dmk.que_from_gm.put(message)
        idmk = tqdm(self.dmkD) if self.logger.level < 20 else self.dmkD
        for _ in idmk:
            self.que_to_gm.get()
        self.logger.debug('> all DMKs exited!')

    # creates new tables & puts players with random policy
    def _put_players_on_tables(self):

        self.logger.info('> puts players on tables..')

        # build dict of lists of players (per family): {family: [(pid, que_to_pl, que_from_pl)]}
        fam_ques: Dict[str, List[Tuple[str,Que,Que]]] = {fam: [] for fam in self.families}
        for dmk in self.dmkD.values():
            for k in dmk.queD_to_player: # {pid: que_to_pl}
                fam_ques[dmk.family].append((k, dmk.queD_to_player[k], dmk.que_from_player))

        # shuffle players in families
        for fam in fam_ques:
            random.shuffle(fam_ques[fam])
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
        if num_players % N_TABLE_PLAYERS != 0:
            raise PyPoksException(f'num_players ({num_players}) has to be a multiple of N_TABLE_PLAYERS ({N_TABLE_PLAYERS})')

        # put on tables
        self.tables = []
        table_ques = []
        table_logger = get_child(self.logger, name='table_logger', change_level=-10) if self.debug_tables else None
        while quesL:
            table_ques.append(quesL.pop())
            if len(table_ques) == N_TABLE_PLAYERS:
                self.tables.append(QPTable(
                    name=       f'tbl{len(self.tables)}',
                    que_to_gm=  self.que_to_gm,
                    pl_ques=    {t[0]: (t[1], t[2]) for t in table_ques},
                    logger=     table_logger))
                table_ques = []

    # starts all tables
    def _start_tables(self):
        self.logger.debug('> starts tables..')
        itbl = tqdm(self.tables) if self.logger.level < 20 else self.tables
        for tbl in itbl: tbl.start()
        for _ in itbl:
            self.que_to_gm.get()
        self.logger.debug(f'> tables ({len(self.tables)}) processes started!')

    # stops tables
    def _stop_tables(self):
        self.logger.debug('> stops tables loops..')
        message = QMessage(type='stop_table', data=None)
        for table in self.tables: table.que_from_gm.put(message)
        itbl = tqdm(self.tables) if self.logger.level < 20 else self.tables
        for _ in itbl:
            self.que_to_gm.get()
        # INFO: tables now are just Process objects with target loop stopped
        self.logger.debug('> tables loops stopped!')

    # runs game, returns DMK results dictionary
    def run_game(
            self,
            game_size=                                  10000,  # number of hands for a game (per DMK)
            sleep=                                      10,     # loop sleep (seconds)
            progress_report=                            True,
            publish_GM=                                 False,
            sep_all_break: bool=                        False,  # breaks game when all DMKs are separated
            sep_pairs: Optional[List[Tuple[str,str]]]=  None,   # pairs of DMK names for separation condition
            sep_pairs_factor: float=                    0.9,    # factor of separated pairs needed to break the game
            sep_n_stdev: float=                         2.0,
            sep_min_IV: int=                            10,     # minimal number of IV to enable any sep break
    ) -> Dict[str, Dict]:
        """
        By now, by design run_game() may be called only once,
        cause DMK processes are started and then stopped and process cannot be started twice,
        there is no real need to change this design.
        """

        # save of DMK results + additional DMK info
        dmk_results = {
            dn: {
                'wonH_IV':              [],     # wonH (won $ / hand) of interval
                'wonH_afterIV':         [],     # wonH (won $ / hand) after interval
                'last_wonH_afterIV':    None,
                'wonH_IV_stdev':        None,
                'wonH_IV_mean_stdev':   None,
                'family':               self.dmkD[dn].family,
                'trainable':            self.dmkD[dn].trainable,
                'global_stats':         None,   # SM.global_stats, will be updated at the end of the game
            } for dn in self._get_dmk_focus_names()}

        # starts all subprocesses
        self._put_players_on_tables()
        self._start_tables()
        self._start_dmks()

        stime = time.time()
        time_last_report = stime
        n_hands_last_report = 0

        self.logger.info(f'{self.name} starts a game..')
        loop_ix = 0
        while True:

            time.sleep(sleep)

            reports = self._get_reports({dn: len(dmk_results[dn]['wonH_IV']) for dn in dmk_results}) # actual DMK reports
            num_IV = []
            for dn in reports:
                dmk_results[dn]['wonH_IV'] += reports[dn]['wonH_IV']
                dmk_results[dn]['wonH_afterIV'] += reports[dn]['wonH_afterIV']
                wonH_IV_stdev = stdev_with_none(dmk_results[dn]['wonH_IV'])
                dmk_results[dn]['wonH_IV_stdev'] = wonH_IV_stdev
                dmk_results[dn]['wonH_IV_mean_stdev'] = wonH_IV_stdev / math.sqrt(len(dmk_results[dn]['wonH_IV'])) if wonH_IV_stdev is not None else None
                dmk_results[dn]['last_wonH_afterIV'] = dmk_results[dn]['wonH_afterIV'][-1] if dmk_results[dn]['wonH_afterIV'] else None
                num_IV.append(len(dmk_results[dn]['wonH_IV']))
            num_IV = min(num_IV)

            # calculate game factor
            n_hands = sum([reports[dn]['n_hands'] for dn in reports])
            game_factor = n_hands / len(reports) / game_size
            if game_factor >= 1: game_factor = 1

            sr = separation_report(
                dmk_results=    dmk_results,
                n_stdev=        sep_n_stdev,
                sep_pairs=      sep_pairs)
            sep_nc = sr['sep_nc']
            sep_nf = sr['sep_nf']
            sep_pairs_nc = sr['sep_pairs_nc']
            sep_pairs_nf = sr['sep_pairs_nf']

            if publish_GM:
                self.tbwr.add(value=sep_nc, tag=f'GM/sep_nc', step=loop_ix)
                self.tbwr.add(value=sep_nf, tag=f'GM/sep_nf', step=loop_ix)
                if sep_pairs:
                    self.tbwr.add(value=sep_pairs_nc, tag=f'GM/sep_pairs_nc', step=loop_ix)
                    self.tbwr.add(value=sep_pairs_nf, tag=f'GM/sep_pairs_nf', step=loop_ix)

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
                hdiff = n_hands-n_hands_last_report
                hd_pp = int(hdiff / len(reports))
                spd_report = f'({num_IV}) {int(hdiff / (time.time()-time_last_report))}H/s (+{hd_pp}Hpp)'
                n_hands_last_report = n_hands
                time_last_report = time.time()

                sep_report_pairs = f'::{sep_pairs_nc:.2f}[{sep_pairs_nf:.2f}]' if sep_pairs else ''

                progress_(
                    current=    game_factor,
                    total=      1.0,
                    prefix=     f'GM: {passed:.1f}min left:{left_nfo}min',
                    suffix=     f'{spd_report} -- SEP:{sep_nc:.2f}[{sep_nf:.2f}]{sep_report_pairs}',
                    length=     20)

            # games break - factor condition
            if game_factor == 1:
                self.logger.info('> finished game (game factor condition)')
                break

            # games break - all DMKs separation condition
            if sep_all_break and num_IV >= sep_min_IV and sep_nc == 1.0:
                self.logger.info(f'> finished game (all DMKs separation condition), game factor: {game_factor:.2f})')
                break

            # games break - pairs separation breaking value condition
            if sep_pairs and num_IV >= sep_min_IV and sep_pairs_nc >= sep_pairs_factor:
                self.logger.info(f'> finished game (pairs separation factor: {sep_pairs_factor:.2f}, game factor: {game_factor:.2f})')
                break

            loop_ix += 1

        self.tbwr.flush()

        self._stop_tables()
        self._stop_dmks_loops()

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
        self.logger.info(f'{self.name} finished run_game, avg speed: {speed:.1f}H/s, time taken: {taken_nfo}')
        loop_stats = {'speed': speed}

        return {
            'dmk_results':  dmk_results,
            'loop_stats':   loop_stats}

    # prepares list of DMK names GM is focused on while preparing dmk_results
    def _get_dmk_focus_names(self) -> List[str]:
        return list(self.dmkD.keys())

    # asks DMKs to send reports, but only form given IV
    def _get_reports(
            self,
            dmk_report_IV:Dict[str,int] # {dn: from_IV}
    ) -> Dict[str, Dict]:
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

# GamesManager for "PLay & TRain With Ref" concept for FolDMKs (DMKs may play or train against ref)
class GamesManager_PTR(GamesManager):

    def __init__(
            self,
            dmk_point_ref: Optional[List[Dict]]=    None, # playable reference DMK list
            dmk_point_PLL: Optional[List[Dict]]=    None, # playable DMK list
            dmk_point_TRL: Optional[List[Dict]]=    None, # trainable DMK list
            dmk_n_players: int=                     60,
            name: Optional[str]=                    None,
            **kwargs):

        """
        there are 4 possible setups:
        1. only dmk_point_TRL:
            all DMKs are training (against each other)
        2. only dmk_point_PLL
            all DMKs are playing (against each other)
        3. dmk_point_TRL & dmk_point_ref
            DMKs_TRL are training against DMKs_ref
        4. dmk_point_PLL & dmk_point_ref
            DMKs_TRL are playing against DMKs_ref
        """

        if not dmk_point_ref: dmk_point_ref = []
        if not dmk_point_PLL: dmk_point_PLL = []
        if not dmk_point_TRL: dmk_point_TRL = []

        if not (dmk_point_PLL or dmk_point_TRL):
            raise PyPoksException('playing OR training DMKs must be given')

        if dmk_point_PLL and dmk_point_TRL:
            raise PyPoksException('playing AND training DMKs cannot be given')

        n_tables = (len(dmk_point_TRL) + len(dmk_point_PLL)) * dmk_n_players # default when there are _ref given
        if not dmk_point_ref:
            dmk_dnaL = dmk_point_PLL or dmk_point_TRL
            if (len(dmk_dnaL) * dmk_n_players) % N_TABLE_PLAYERS != 0:
                raise PyPoksException('Please correct number of DMK players: len(DMKs) * dmk_n_players must be multiplication of N_TABLE_PLAYERS')
            n_tables = (len(dmk_dnaL) * dmk_n_players) // N_TABLE_PLAYERS

        ### update points

        for dmk in dmk_point_TRL:
            dmk.update({
                'n_players': dmk_n_players, # TODO: what for???
                'trainable': True})

        for dmk in dmk_point_PLL:
            dmk.update({
                'n_players': dmk_n_players, # TODO: what for???
                'trainable': False})

        self.ref_pattern = []
        if dmk_point_ref:
            n_players_for_one = dmk_n_players * (N_TABLE_PLAYERS-1)
            rest_names = [dna['name'] for dna in dmk_point_ref] * n_players_for_one
            self.ref_pattern = rest_names[:n_players_for_one]
            self.ref_pattern *= (len(dmk_point_TRL) + len(dmk_point_PLL))
            for dmk in dmk_point_ref:
                dmk.update({
                    'n_players': len([nm for nm in self.ref_pattern if nm == dmk['name']]),
                    'trainable': False})

        self.dmk_name_ref = [dna['name'] for dna in dmk_point_ref]
        self.dmk_name_PLL = [dna['name'] for dna in dmk_point_PLL]
        self.dmk_name_TRL = [dna['name'] for dna in dmk_point_TRL]

        nm = 'PL' if self.dmk_name_PLL else 'TR'
        if dmk_point_ref:
            nm += '_ref'
        GamesManager.__init__(
            self,
            dmk_pointL= dmk_point_ref + dmk_point_PLL + dmk_point_TRL,
            name=       name or f'GM_{nm}_{stamp()}',
            **kwargs)

        self.logger.info(f'*** GamesManager_PTR started with (PL:{len(dmk_point_PLL)} TR:{len(dmk_point_TRL)} ref:{len(dmk_point_ref)}) DMKs on {n_tables} tables')
        for dna in dmk_point_PLL + dmk_point_TRL:
            self.logger.debug(f'> {dna["name"]} with {dna["n_players"]} players, trainable: {dna["trainable"]}')

    # creates new tables & puts players with PTR policy
    def _put_players_on_tables(self):

        # use previous policy
        if not self.dmk_name_ref:
            return GamesManager._put_players_on_tables(self)

        self.logger.info('> puts players on tables with PTR policy..')

        ques_refD = {}
        ques_rest = []

        for dmk_name in self.dmkD:

            dmk = self.dmkD[dmk_name]

            ques = ques_rest
            if dmk_name in self.dmk_name_ref:
                ques_refD[dmk_name] = []
                ques = ques_refD[dmk_name]

            for k in dmk.queD_to_player: # {pid: que_to_pl}
                ques.append((k, dmk.queD_to_player[k], dmk.que_from_player))

        # put on tables
        self.tables = []
        table_ques =  []
        table_logger = get_child(self.logger, name='table_logger', change_level=-10) if self.debug_tables else None
        rp_ix = 0
        while ques_rest:

            table_ques.append(ques_rest.pop())
            while len(table_ques) < N_TABLE_PLAYERS:
                table_ques.append(ques_refD[self.ref_pattern[rp_ix]].pop())
                rp_ix += 1

            self.tables.append(QPTable(
                name=       f'tbl{len(self.tables)}',
                que_to_gm=  self.que_to_gm,
                pl_ques=    {t[0]: (t[1], t[2]) for t in table_ques},
                logger=     table_logger))
            table_ques = []

    # adds age update to dmk_results
    def run_game(self, **kwargs) -> Dict:

        # update trainable age - needs to be done before game, cause after game DMKs are saved
        for dmk in self.dmkD.values():
            if dmk.trainable: dmk.age += 1

        rgd = GamesManager.run_game(self, **kwargs)

        for dn in rgd['dmk_results']:
            rgd['dmk_results'][dn]['age'] = self.dmkD[dn].age
        return rgd

    # at GamesManager_PTR we are focused on TRL (or PLL if not)
    def _get_dmk_focus_names(self) -> List[str]:
        return self.dmk_name_TRL or self.dmk_name_PLL

# manages DMKs for human games
class HuGamesManager(GamesManager):

    def __init__(
            self,
            dmk_names: Union[List[str],str],
            logger=     None,
            loglevel=   20):

        if not logger:
            logger = get_pylogger(level=loglevel)

        if N_TABLE_PLAYERS != 3:
            raise PyPoksException('HuGamesManage supports now only 3-handed tables')

        logger.info(f'HuGamesManager starts with given dmk_names: {dmk_names}')

        h_name = 'hm0'

        hdna = {
            'name':             h_name,
            'family':           'h',
            'trainable':        False,
            'n_players':        1,
            #'publish':          False,
            'fwd_stats_step':   10}

        if type(dmk_names) is str: dmk_names = [dmk_names]

        self.tk_gui = GUI_HDMK(players=[h_name]+dmk_names, imgs_FD='gui/imgs')

        hdmk = HuDMK(tk_gui=self.tk_gui, **hdna)

        if len(dmk_names) not in [1,2]:
            raise PyPoksException('Number of given DMK names must be equal 1 or 2')

        ddL = [{
            'name':             nm,
            'trainable':        False,
            'n_players':        N_TABLE_PLAYERS - len(dmk_names),
            #'publish':          False,
            'fwd_stats_step':   10} for nm in dmk_names]

        GamesManager.__init__(self, dmk_pointL=ddL, logger=logger)

        # update/override with HuDMK
        self.dmkD[hdna['name']] = hdmk
        self.families.add(hdna['family'])
        hdmk.que_to_gm = self.que_to_gm

    # starts all subprocesses
    def start_games(self):
        self._put_players_on_tables()
        self._start_tables()
        self._start_dmks()

    # an alternative way of stopping all subprocesses (dmks & tables)
    def kill_games(self):
        self.logger.info('HuGamesManager is killing games..')
        for dmk in self.dmkD.values(): dmk.kill()
        for table in self.tables: table.kill()

    def run_tk(self): self.tk_gui.run_tk()