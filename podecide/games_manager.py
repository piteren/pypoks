import copy
import math
import os
from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.printout import stamp, progress_
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.moving_average import MovAvg
from pypaq.mpython.mptools import Que, QMessage
from torchness.tbwr import TBwr
import random
import statistics
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

from pypoks_base import PyPoksException
from pypoks_envy import MODELS_FD, DMK_MODELS_FD, N_TABLE_PLAYERS
from pologic.potable import QPTable
from podecide.dmk import FolDMK, HuDMK
from gui.gui_hdmk import GUI_HDMK

# keys managed by family settings and FSExc
FAM_KEYS = [
    'trainable',
    'do_GX',
    'pex_max',
    'prob_zero',
    'prob_max',
    'step_min',
    'step_max',
    'iLR']


# separated factor for two results
def separated_factor(
        a_wonH: float,
        a_wonH_mean_stdev: float,
        b_wonH: float,
        b_wonH_mean_stdev: float,
        n_stdev: float) -> float:
    return abs(a_wonH - b_wonH) / (n_stdev * (a_wonH_mean_stdev + b_wonH_mean_stdev))

# prepares separation report
def separation_report(
        dmk_results: Dict,
        n_stdev: float,
        sep_pairs: Optional[List[Tuple[str,str]]]=  None,
) -> Dict:

    n_dmk = len(dmk_results)

    # add wonH_mean_stdev
    for dn in dmk_results:
        stdev = statistics.stdev(dmk_results[dn]['wonH_IV'])
        dmk_results[dn]['wonH_mean_stdev'] = stdev / math.sqrt(len(dmk_results[dn]['wonH_IV']))

    # compute separated normalized count & normalized factor
    sep_nc = 0
    sep_nf = 0
    for dn_a in dmk_results:
        dmk_results[dn_a]['separated'] = n_dmk - 1
        for dn_b in dmk_results:
            if dn_a != dn_b:
                sf = separated_factor(
                    a_wonH=             dmk_results[dn_a]['wonH_afterIV'][-1],
                    a_wonH_mean_stdev=  dmk_results[dn_a]['wonH_mean_stdev'],
                    b_wonH=             dmk_results[dn_b]['wonH_afterIV'][-1],
                    b_wonH_mean_stdev=  dmk_results[dn_b]['wonH_mean_stdev'],
                    n_stdev=            n_stdev)
                if sf < 1:
                    dmk_results[dn_a]['separated'] -= 1
                sep_nf += min(sf,1.1)
        sep_nc += dmk_results[dn_a]['separated']
    n_max = (n_dmk - 1) * n_dmk
    sep_nc /= n_max
    sep_nf /= n_max

    # same for given pairs
    sep_pairs_nc = 0
    sep_pairs_nf = 0
    sep_pairs_stat = []
    if sep_pairs:
        for sp in sep_pairs:
            sf = separated_factor(
                a_wonH=             dmk_results[sp[0]]['wonH_afterIV'][-1],
                a_wonH_mean_stdev=  dmk_results[sp[0]]['wonH_mean_stdev'],
                b_wonH=             dmk_results[sp[1]]['wonH_afterIV'][-1],
                b_wonH_mean_stdev=  dmk_results[sp[1]]['wonH_mean_stdev'],
                n_stdev=            n_stdev)
            sep_pairs_stat.append(0 if sf<1 else 1)
            if sf>=1: sep_pairs_nc += 1
            sep_pairs_nf += min(sf,1.1)
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

        dmk_pointL = copy.deepcopy(dmk_pointL) # copy not to modify original list
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
            sleep=                                      20,     # loop sleep (seconds)
            progress_report=                            True,
            publish_GM=                                 False,
            sep_all_break: bool=                        False,  # breaks game when all DMKs are separated
            sep_pairs: Optional[List[Tuple[str,str]]]=  None,   # pairs of DMK names for separation condition
            sep_pairs_factor: float=                    0.9,    # factor of separated pairs needed to break the game
            sep_n_stdev: float=                         2.0,
    ) -> Dict:
        """
        By now, by design run_game() may be called only once,
        cause DMK processes are started and then stopped and process cannot be started twice,
        there is no real need to change this design.
        """

        dmk_focus_names = self._get_dmk_focus_names()

        # save of DMK results + additional DMK info
        dmk_results = {
            dn: {
                'wonH_IV':      [],     # wonH of interval
                'wonH_afterIV': [],     # wonH after interval
                'family':       self.dmkD[dn].family,
                'trainable':    self.dmkD[dn].trainable,
                'global_stats': None,
            } for dn in dmk_focus_names}
        ixs_IV = 0  # start index for IV reports (where we finished in last iteration)

        # CXange in Rankings variables
        dmk_rank = {dn: [] for dn in dmk_focus_names} # save of rankings for intervals
        cxr = None
        cxr_mavg = MovAvg(factor=0.1)

        sep_nc = None
        sep_nf = None
        sep_pairs_nc = None
        sep_pairs_nf = None

        # starts all subprocesses
        self._put_players_on_tables()
        self._start_tables()
        self._start_dmks()

        stime = time.time()
        time_last_report = stime
        n_hands_last_report = 0

        self.logger.info(f'{self.name} starts a game..')
        while True:

            time.sleep(sleep)

            reports = self._get_reports(ixs_IV) # actual DMK reports

            # calculate game factor
            n_hands = sum([reports[dn]['n_hands'] for dn in reports])
            game_factor = n_hands / len(reports) / game_size
            if game_factor >= 1: game_factor = 1

            n_iv = []
            for dn in reports:
                dmk_results[dn]['wonH_IV'] += reports[dn].pop('wonH_IV')
                dmk_results[dn]['wonH_afterIV'] += reports[dn].pop('wonH_afterIV')
                n_iv.append(len(dmk_results[dn]['wonH_IV']))
            ixe_IV = min(n_iv) # compute end index for IV reports (where we will finish in this iteration)

            # changes in ranking (CXR) & separation (SEP)
            for eix in range(ixs_IV,ixe_IV):

                won_order = [(dn,dmk_results[dn]['wonH_afterIV'][eix]) for dn in dmk_results]
                won_order.sort(key=lambda x:x[1], reverse=True)
                for ix,nw in enumerate(won_order):
                    dmk_rank[nw[0]].append(ix)

                #CXR & SEP needs variance, which is possible for at least 2 values
                if len(dmk_rank[won_order[0][0]]) > 1:
                    cxr = sum([abs(dmk_rank[n][-1]-dmk_rank[n][-2]) for n in dmk_rank])
                    cxr_mavg.upd(cxr)
                    if publish_GM:
                        self.tbwr.add(value=cxr_mavg(), tag=f'GM/cxr_mavg', step=eix)

                    sr = separation_report(
                        dmk_results=    dmk_results,
                        n_stdev=        sep_n_stdev,
                        sep_pairs=      sep_pairs)
                    sep_nc = sr['sep_nc']
                    sep_nf = sr['sep_nf']
                    sep_pairs_nc = sr['sep_pairs_nc']
                    sep_pairs_nf = sr['sep_pairs_nf']
                    if publish_GM:
                        sr.pop('sep_pairs_stat')
                        for k in sr:
                            self.tbwr.add(value=sr[k], tag=f'GM/{k}', step=eix)

            ixs_IV = ixe_IV

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
                spd_report = f'{int(hdiff / (time.time()-time_last_report))}H/s (+{hd_pp}Hpp)'
                n_hands_last_report = n_hands
                time_last_report = time.time()

                cxr_report = f' CXR:{cxr}->{round(cxr_mavg(),1)}' if cxr is not None else ''

                sep_report_all = f'{sep_nc:.2f}[{sep_nf:.2f}]' if sep_nc is not None else ''
                sep_report_pairs = f'::{sep_pairs_nc:.2f}[{sep_pairs_nf:.2f}]' if sep_pairs and sep_pairs_nc is not None else ''
                sep_report = f' SEP:{sep_report_all}{sep_report_pairs}' if sep_report_all else ''

                progress_(
                    current=    game_factor,
                    total=      1.0,
                    prefix=     f'GM: {passed:.1f}min left:{left_nfo}min',
                    suffix=     f'{spd_report} --{cxr_report}{sep_report}',
                    length=     20)

            # games break - factor condition
            if game_factor == 1:
                self.logger.info('> finished game (game factor condition)')
                break

            # games break - all DMKs separation condition
            if sep_all_break and sep_nc == 1:
                self.logger.info(f'> finished game (all DMKs separation condition), game factor: {game_factor:.2f})')
                break

            # games break - pairs separation breaking value condition
            if sep_pairs and sep_pairs_nc >= sep_pairs_factor:
                self.logger.info(f'> finished game (pairs separation factor: {sep_pairs_factor::.2f}, game factor: {game_factor:.2f})')
                break

        self.tbwr.flush()

        self._stop_tables()
        self._stop_dmks_loops()

        message = QMessage(type='send_global_stats', data=None)
        for dn in dmk_focus_names:
            self.dmkD[dn].que_from_gm.put(message)
        for _ in dmk_focus_names:
            message = self.que_to_gm.get()
            data = message.data
            dmk_name = data.pop('dmk_name')
            dmk_results[dmk_name]['global_stats'] = data['global_stats']

        self._save_dmks()
        self._stop_dmks_processes()

        taken_sec = time.time() - stime
        taken_nfo = f'{taken_sec / 60:.1f}min' if taken_sec > 100 else f'{taken_sec:.1f}sec'
        self.logger.info(f'{self.name} finished run_game, avg speed: {n_hands / taken_sec:.1f}H/s, time taken: {taken_nfo}')

        return dmk_results

    # prepares list of DMK names GM is focused on
    def _get_dmk_focus_names(self) -> List[str]:
        return list(self.dmkD.keys())

    # asks DMKs to prepare reports
    def _get_reports(self, from_IV:int) -> Dict[str, Dict]: # {dn: {n_hands, wonH_IV, wonH_afterIV}}
        dmk_names = self._get_dmk_focus_names()
        reports: Dict[str, Dict] = {}
        for dn in dmk_names:
            message = QMessage(type='send_dmk_report', data=from_IV)
            self.dmkD[dn].que_from_gm.put(message)
        for _ in dmk_names:
            message = self.que_to_gm.get()
            report = message.data
            dmk_name = report.pop('dmk_name')
            reports[dmk_name] = report
        return reports

# GamesManager for Play & TRain concept for FolDMKs (some DMKs may play, some DMKs may train)
class GamesManager_PTR(GamesManager):

    def __init__(
            self,
            dmk_point_PLL: Optional[List[Dict]]=    None, # playable DMK list
            dmk_point_TRL: Optional[List[Dict]]=    None, # trainable DMK list
            dmk_n_players: int=                     60,
            name: Optional[str]=                    None,
            **kwargs):

        """
        there are 3 possible scenarios:
        1.playable & trainable:
            dmk_n_players - sets number of players per trainable DMK
            number of players for each playable DMK is calculated
            number of tables == dmk_n_players (each trainable has one table)
        2.only trainable:
            dmk_n_players - sets number of players per trainable DMK
            number of tables = len(dmk)*dmk_n_players / N_TABLE_PLAYERS
        3.only playable
            dmk_n_players - sets number of players per playable DMK
            number of tables = len(dmk)*dmk_n_players / N_TABLE_PLAYERS
        """

        if not dmk_point_PLL: dmk_point_PLL =  []
        if not dmk_point_TRL: dmk_point_TRL = []

        if not (dmk_point_PLL or dmk_point_TRL):
            raise PyPoksException('playing OR training DMKs must be given')

        n_tables = len(dmk_point_TRL) * dmk_n_players # default when there are both playable & trainable
        if not dmk_point_PLL or not dmk_point_TRL:
            dmk_dnaL = dmk_point_PLL or dmk_point_TRL
            if (len(dmk_dnaL) * dmk_n_players) % N_TABLE_PLAYERS != 0:
                raise PyPoksException('Please correct number of DMK players: n DMKs * n players must be multiplication of N_TABLE_PLAYERS')
            n_tables = int((len(dmk_dnaL) * dmk_n_players) / N_TABLE_PLAYERS)

        # override to train (each DMK by default is saved as a trainable - we set also trainable to have this info here for later usage, it needs n_players to be set)
        for dmk in dmk_point_TRL:
            dmk.update({
                'n_players':    dmk_n_players,
                'trainable':    True})

        if dmk_point_PLL:

            # both
            if dmk_point_TRL:
                n_rest_players = n_tables * (N_TABLE_PLAYERS-1)
                rest_names = [dna['name'] for dna in dmk_point_PLL]
                rest_names = random.choices(rest_names, k=n_rest_players)
                for point in dmk_point_PLL:
                    point.update({
                        'n_players': len([nm for nm in rest_names if nm == point['name']]),
                        'trainable': False})

            # only playable
            else:
                play_dna = {
                    'n_players': dmk_n_players,
                    'trainable': False}
                for dmk in dmk_point_PLL:
                    dmk.update(play_dna)

        self.dmk_name_PLL = [dna['name'] for dna in dmk_point_PLL]
        self.dmk_name_TRL = [dna['name'] for dna in dmk_point_TRL]

        nm = 'PL' if self.dmk_name_PLL else 'TR'
        if self.dmk_name_PLL and self.dmk_name_TRL:
            nm = 'TR+PL'
        GamesManager.__init__(
            self,
            dmk_pointL= dmk_point_PLL + dmk_point_TRL,
            name=       name or f'GM_{nm}_{stamp()}',
            **kwargs)

        self.logger.info(f'*** GamesManager_PTR started with (PL:{len(dmk_point_PLL)} TR:{len(dmk_point_TRL)}) DMKs on {n_tables} tables')
        for dna in dmk_point_PLL + dmk_point_TRL:
            self.logger.debug(f'> {dna["name"]} with {dna["n_players"]} players, trainable: {dna["trainable"]}')

    # creates new tables & puts players with PTR policy
    def _put_players_on_tables(self):

        # use previous policy
        if not (self.dmk_name_PLL and self.dmk_name_TRL):
            return GamesManager._put_players_on_tables(self)

        self.logger.info('> puts players on tables with PTR policy..')

        ques_PL = []
        ques_TR = []

        for dmk in self.dmkD.values():
            ques = ques_TR if dmk.trainable else ques_PL
            for k in dmk.queD_to_player: # {pid: que_to_pl}
                ques.append((k, dmk.queD_to_player[k], dmk.que_from_player))

        # shuffle players
        random.shuffle(ques_PL)
        random.shuffle(ques_TR)

        # put on tables
        self.tables = []
        table_ques =  []
        table_logger = get_child(self.logger, name='table_logger', change_level=-10) if self.debug_tables else None
        while ques_TR:
            table_ques.append(ques_TR.pop())
            while len(table_ques) < N_TABLE_PLAYERS: table_ques.append(ques_PL.pop())
            random.shuffle(table_ques)
            self.tables.append(QPTable(
                name=       f'tbl{len(self.tables)}',
                que_to_gm=  self.que_to_gm,
                pl_ques=    {t[0]: (t[1], t[2]) for t in table_ques},
                logger=     table_logger))
            table_ques = []
        assert not ques_PL and not ques_TR

    # adds age update
    def run_game(self, **kwargs) -> Dict:

        # update trainable age - needs to be done before game, cause after game DMKs are saved
        for dmk in self.dmkD.values():
            if dmk.trainable: dmk.age += 1

        dmk_results = GamesManager.run_game(self, **kwargs)

        # put age into res_list
        for dn in dmk_results:
            dmk_results[dn]['age'] = self.dmkD[dn].age

        return dmk_results

    # prepares list of DMK names we are focused in the game (with focus on TRL)
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