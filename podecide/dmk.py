import math
from abc import abstractmethod, ABC
from multiprocessing import Process
import numpy as np
from pypaq.pytypes import NPL
from pypaq.lipytools.stats import mam
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.files import prep_folder
from pypaq.mpython.mptools import Que, QMessage
from pypaq.pms.parasave import ParaSave
from pypaq.pms.base import POINT
import random
import statistics
from torchness.motorch import MOTorch
from torchness.tbwr import TBwr
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from typing import List, Tuple, Optional, Dict, Any

from envy import DMK_MODELS_FD, DMK_POINT_PFX, N_TABLE_PLAYERS, TABLE_CASH_START, TBL_MOV, WON_IV
from pologic.podeck import PDeck
from pologic.hand_history import STATE
from podecide.game_state import GameState
from podecide.dmk_motorch import DMK_MOTorch
from podecide.stats.won_manager import WonMan
from podecide.stats.player_stats import PStatsEx
from gui.gui_hdmk import GUI_HDMK


# ***************************************************************************************************** abstract classes

# Decision MaKer - basic interface to decide for poker players (PPlayer on PTable)
class DMK(ABC):

    def __init__(
            self,
            name: str,                      # name should be unique (@table)
            n_players: int= 100,            # number of players managed by one DMK
            n_moves: int=   len(TBL_MOV),   # number of (all) moves supported by DMK, has to match the table
            family: str=    'a',            # family (type) used to group DMKs and manage together
            save_topdir=    DMK_MODELS_FD,
            logger=         None,
            loglevel=       20):

        self.name = name

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                add_stamp=  False,
                folder=     f'{save_topdir}/{self.name}',
                level=      loglevel)
        self._logger = logger

        self._player_ids = [f'{self.name}_{ix}' for ix in range(n_players)] if n_players>1 else [self.name] # DMK keeps unique ids (str) of players
        self.n_moves = n_moves
        self.family = family
        self.save_topdir = save_topdir

        self._logger.info(f'*** {self.__class__.__name__} (DMK) : {self.name} *** initialized')
        self._logger.debug(f'> {self.n_moves} supported moves')
        self._logger.debug(f'> {len(self._player_ids)} players')

    # takes & stores player states
    @abstractmethod
    def collect_states(
            self,
            player_id: str,
            player_states: List[Any]):
        pass

    # makes decisions [(player_id,move,probs)..] for one/some/all players with possible moves
    @abstractmethod
    def make_decisions(self) -> List[Tuple[str,int,NPL]]:
        pass

# Methods DMK - extends DMK baseline with sub-methods and data structures
class MethDMK(DMK, ABC):

    def __init__(self, **kwargs):

        DMK.__init__(self, **kwargs)

        # variables below store states data
        self._states_new: Dict[str,List[GameState]] = {pid: [] for pid in self._player_ids} # dict of new states lists
        self._n_states_new = 0  # cache, number of all _states_new == sum([len(l) for l in self._states_new.values()])

    # takes player states, encodes and saves
    def collect_states(
            self,
            player_id: str,
            player_states: List[Tuple]):

        encoded_states = self._encode_states(player_id, player_states)

        # save
        if encoded_states:
            self._states_new[player_id] += encoded_states
            self._n_states_new += len(encoded_states)

    # encodes player states into type appropriate for DMK to make decisions
    @abstractmethod
    def _encode_states(
            self,
            player_id: str,
            player_stateL: List[STATE],
    ) -> List[GameState]:
        return [GameState(state_orig_data=value) for value in player_stateL] # wraps into list of GameState

    # takes possible_moves (and their cash) from poker player
    def collect_possible_moves(
            self,
            player_id: str,
            possible_moves :List[bool],
            moves_cash :List[int]):
        last_state = self._states_new[player_id][-1]
        last_state.possible_moves = possible_moves
        last_state.moves_cash = moves_cash

    # makes decisions then flushes states
    def make_decisions(self) -> List[Tuple[str,int,NPL]]:

        decL = self._decisions_from_new_states()  # get decisions list

        # flush new states using decisions list
        for dec in decL:
            pid, _, _ = dec
            self._states_new[pid] = []  # reset

        return decL

    # makes decisions & returns list of decisions using data from _states_new
    def _decisions_from_new_states(self) -> List[Tuple[str,int,NPL]]:
        self._calc_probs()
        decL = self._sample_moves_for_ready_players()
        return decL

    # calculates probabilities for at least some _states_new with possible_moves
    @abstractmethod
    def _calc_probs(self) -> None: pass

    # samples moves for players with ready data (possible_moves and probs)
    def _sample_moves_for_ready_players(self) -> List[Tuple[str,int,NPL]]:
        decL = []
        for pid in self._states_new:
            if self._states_new[pid]: # not empty
                last_state = self._states_new[pid][-1]
                if last_state.possible_moves is not None and last_state.probs is not None:
                    move = self._sample_move(
                        probs=              last_state.probs,
                        possible_moves =    last_state.possible_moves,
                        pid=                pid)
                    last_state.move = move # add move to the last state
                    decL.append((pid, move, last_state.probs))
        return decL

    # samples move form possible_moves using given probabilities
    def _sample_move(
            self,
            probs: np.ndarray,
            possible_moves: List[bool],
            pid: str        # pid to be used by inherited implementations
    ) -> int:

        probs = probs * possible_moves                      # mask probs

        # take mask if no intersection
        if sum(probs) == 0:
            probs = np.asarray(possible_moves).astype(float)

        probs /= sum(probs)                                 # normalize

        return int(np.random.choice(self.n_moves, p=probs)) # choice

# Methods Trainable DMK - adds training methods and data structures, training may be triggered while making decisions
class MeTrainDMK(MethDMK, ABC):

    def __init__(
            self,
            trainable=          True,
            upd_trigger: int=   30000,  # estimated target batch size (number of decisions) for update (updating from half - check selection state policy @UPD - half_rectangle in trapeze)
            upd_step=           0,      # updates counter (clock)
            **kwargs):

        MethDMK.__init__(self, **kwargs)

        self.trainable =    trainable
        self.upd_trigger =  upd_trigger
        self.upd_step =     upd_step

        # variables below store moves/decisions data
        self._states_dec: Dict[str,List[GameState]] = {pid: [] for pid in self._player_ids}  # dict with decided states lists (for update)
        self._n_states_dec = 0  # cache, number of all _states_dec == sum([len(l) for l in self._states_dec.values()])

    # makes decisions > moves states > trains policy - overrides MethDMK method, which does not train
    def make_decisions(self) -> List[Tuple[str,int,NPL]]:
        decL = self._decisions_from_new_states()    # get decisions list
        self.__move_states(decL)                    # move states from _states_new to _states_dec
        self.__train_policy()                       # learn/train policy,..it is called here but triggered inside

        decr = f'\n{self.name} made decisions:\n'
        for dec in decL:
            decr += f'> {dec}\n'
        self._logger.debug(decr)

        return decL

    # moves states (from _new to _dec) having decisions list
    def __move_states(self, decL :List[Tuple[str,int,NPL]]):
        for dec in decL:
            pid, move, _ = dec
            states = self._states_new[pid]      # take all new states
            self._states_new[pid] = []          # reset
            self._n_states_new -= len(states)
            self._states_dec[pid] += states     # put states to dec
            self._n_states_dec += len(states)

    # trains DMK (policy)
    def __train_policy(self) -> None:
        if self._n_states_dec > self.upd_trigger: # only when trigger fires
            ust_details = self._training_core()
            self._flush_states_dec(ust_details)
            self.upd_step += 1

    # trains (updates self policy) with _states_dec (cache of taken moves & received rewards)
    def _training_core(self): return None

    # flushes _states_dec using information from ust_details
    def _flush_states_dec(self, ust_details) -> None:
        self._states_dec = {pid: [] for pid in self._player_ids} # flushes all _states_dec (here baseline, not uses ust_details)
        self._n_states_dec = 0

# Qued DMK - extends DMK with Ques and Process (to act on QPTable with QPPlayer and to be managed by GamesManager)
class QueDMK(MeTrainDMK, ABC):

    def __init__(
            self,
            name: str,
            save_topdir=    DMK_MODELS_FD,
            publish_more=   True,           # allows to publish advanced FWD process & UPD process stats
            **kwargs):

        MeTrainDMK.__init__(
            self,
            name=           name,
            save_topdir=    save_topdir,
            **kwargs)

        self._process = Process(name=f'QueDMK_process:{name}', target=self.__dmk_proc)

        self._que_to_gm = None # here QuedDMK sends data to GamesManager, data is in form (name, command, data)
        self._que_from_gm = Que() # here QuedDMK receives data from GamesManager, data is in form (command, data)

        self._running_process = False  # flag for running process loop
        self._running_game = False  # flag for running game loop

        # stats of the qued process, check __reset for details
        self._processFWD_stats = {}
        self.__reset_processFWD_stats()

        # every DMK creates part of DMK-Player Ques network
        self._que_from_player = Que() # here player puts data for DMK
        self._queD_to_player = {pid: Que() for pid in self._player_ids} # dict with ques where DMK puts data for each player

        self.publish_more = publish_more
        self._tbwr = None

    # starts Process
    def start(self):
        self._process.start()

    # process method (target of Process)
    def __dmk_proc(self):

        # first call _pre_process()
        self._pre_process()
        message = QMessage(
            type=   'dmk_status',
            data=   f'{self.name} (DMK) _pre_process() done')
        self.que_to_gm.put(message)

        # here wait for start loop message (or other after)
        self._running_process = True
        while self._running_process:
            gm_data = self.que_from_gm.get()
            self._do_what_GM_says(gm_data)

        # being here means QueDMK finishes its process..
        message = QMessage(
            type=   'dmk_status',
            data=   f'{self.name} (DMK) process finished')
        self.que_to_gm.put(message)

    # method called BEFORE process loop, builds objects that HAVE to be built in process memory scope
    def _pre_process(self) -> None:
        # build TBwr here, inside a process
        fd = f'{self.save_topdir}/{self.name}'
        prep_folder(fd)
        self._tbwr = TBwr(logdir=fd)

    # processes GM messages
    def _do_what_GM_says(self, message: QMessage):

        if message.type == 'start_dmk_loop':
            self._running_game = True
            self.__decisions_loop()

        if message.type == 'stop_dmk_loop':
            self._running_game = False

        if message.type == 'stop_dmk_process':
            self._running_process = False

        if message.type == 'save_dmk':
            self.save()
            dmk_message = QMessage(type='dmk_saved', data=self.name)
            self.que_to_gm.put(dmk_message)

    # loop of the game, processes incoming data (from players or GM) and sends back decisions (to players) or other data (to GM)
    # it is core method that call other important - previously implemented, like:
    #  - collect_states
    #  - collect_possible_moves
    #  - make_decisions, which triggers training_core
    def __decisions_loop(self):

        message = QMessage(
            type=   'dmk_status',
            data=   f'{self.name} (DMK) decisions_loop started')
        self.que_to_gm.put(message)

        n_waiting = 0 # num players (-> tables) waiting for decision
        while self._running_game:

            n_players = len(self._player_ids)

            # 'flush' the que of data from players
            pmL = []
            while True:
                player_message = self.que_from_player.get(block=False)
                if player_message: pmL.append(player_message)
                else: break
            self._processFWD_stats['0.messages'].append(len(pmL) / n_players)

            for player_message in pmL:

                data = player_message.data

                if player_message.type == 'state_changes':
                    self.collect_states(
                        player_id=      data['id'],
                        player_states=  data['state_changes'])

                # if player sends message to make decision it blocks the table
                if player_message.type == 'make_decision':
                    self.collect_possible_moves(
                        player_id=      data['id'],
                        possible_moves= data['possible_moves'],
                        moves_cash=     data['moves_cash'])
                    n_waiting += 1

            # if got any waiting >> make decisions and put them to players
            if n_waiting:
                self._processFWD_stats['1.waiting'].append(n_waiting / n_players)
                decL = self.make_decisions()
                self._processFWD_stats['2.decisions'].append(len(decL) / n_players)
                self._processFWD_stats['3.unlocked'].append(len(decL) / n_waiting)
                n_waiting -= len(decL)
                for d in decL:
                    pid, move, probs = d
                    message = QMessage(
                        type=   'move',
                        data=   {
                            'selected_move':    move,
                            'probs':            probs})
                    self.queD_to_player[pid].put(message)

            # eventually get data from GM, ..way to exit game_loop
            gm_message = self.que_from_gm.get(block=False)
            if gm_message:
                self._do_what_GM_says(gm_message)

        message = QMessage(
            type=   'dmk_status',
            data=   f'{self.name} (DMK) decisions_loop stopped')
        self.que_to_gm.put(message)

    # adds histogram
    def _decisions_from_new_states(self) -> List[Tuple[str,int,NPL]]:

        ### add histogram data to _process_stats

        nd = {}
        for pid in self._states_new:
            l = len(self._states_new[pid])
            if l not in nd: nd[l] = 0
            nd[l] += 1

        hist_nfo = ''
        for k in sorted(list(nd.keys())):
            hist_nfo += f'{k:d}:{nd[k]:d} '
        self._processFWD_stats['new_states_hist'].append(hist_nfo[:-1])

        return super()._decisions_from_new_states()

    # publishes value to TB
    def tb_add(
            self,
            value: Optional,
            tag: str,
            histogram: Optional=    None,
            step: Optional[int]=    None):

        if step is None:
            step = self.upd_step

        if histogram is not None:
            self._tbwr.add_histogram(values=histogram, tag=tag, step=step)
        else:
            self._tbwr.add(value=value, tag=tag, step=step)

    # publishes processFWD stats, resets
    def _publish_FWD_stats(self, step) -> None:

        hd = []
        for str_upd in self._processFWD_stats.pop('new_states_hist'): # looks like: ['0:207 2:93', '0:232 3:63 4:5',..], it says that 207 players had 0 new states, 93 had 2, in next iteration 232 had 0..
            sn = str_upd.split(' ')
            for sns in sn:
                snss = sns.split(':')
                val = int(snss[0])
                if val: # remove 0
                    hd += [val] * int(snss[1])
        self.tb_add(value=None, histogram=np.asarray(hd), tag='process.FWD.new_states', step=step)

        for k in self._processFWD_stats:
            st = self._processFWD_stats[k]
            val = sum(st) / len(st) if len(st) else 0
            self.tb_add(value=val, tag=f'process.FWD/{k}', step=step)

        self.__reset_processFWD_stats()

    # resets process stats
    def __reset_processFWD_stats(self) -> None:
        self._processFWD_stats = {
            '0.messages':       [], # List[float] - factor of players that send messages in one loop
            '1.waiting':        [], # List[float] - factor of players waiting for a decision
            '2.decisions':      [], # List[float] - factor of players that has been given decisions
            '3.unlocked':       [], # List[float] - factor of waiting players that got unlocked by decision made
            '4.n_rows':         [], # List[int]   - number of rows processed to get possible moves
            'new_states_hist':  []} # List[str]   - histogram of num new states (while calculating probs & making decisions)


    def save(self): pass


    def kill(self):
        self._process.terminate()

    @property
    def pid(self):
        return self._process.pid

    @property
    def que_to_gm(self):
        return self._que_to_gm

    @que_to_gm.setter
    def que_to_gm(self, que:Que):
        self._que_to_gm = que

    @property
    def que_from_gm(self):
        return self._que_from_gm

    @property
    def que_from_player(self):
        return self._que_from_player

    @property
    def queD_to_player(self):
        return self._queD_to_player

# Stats Manager DMK adds WonMan & PStatsEx objects, communicates with GamesManager (GM)
class StaMaDMK(QueDMK, ABC):

    def __init__(
            self,
            fwd_stats_step=         0,      # FWD stats step
            won_iv=                 WON_IV,
            use_villain_stats=      True,
            publish_player_stats=   True,
            **kwargs):

        QueDMK.__init__(self, **kwargs)

        self.fwd_stats_step = fwd_stats_step
        self.won_iv = won_iv

        self.use_villain_stats = use_villain_stats
        self.publish_player_stats = publish_player_stats

    # prepares macro-averaged stats of players -> DMK stats
    def _accumulate_global_stats(self) -> Dict[str,float]:
        my_statsL = [self._players_stats[pid][0].player_stats for pid in self._player_ids]
        my_stats = {k: 0.0 for k in my_statsL[0]}
        nk = len(self._player_ids)
        for e in my_statsL:
            for k in my_stats:
                my_stats[k] += e[k]
        for k in my_stats:
            my_stats[k] /= nk
        return my_stats

    # adds stats management
    def _encode_states(
            self,
            player_id,
            player_stateL: List[STATE]) -> List[GameState]:

        for i in (range(N_TABLE_PLAYERS) if self.use_villain_stats else [0]):
            self._players_stats[player_id][i].process_states(player_stateL)

        wonD = self._wm.process_states(player_stateL) # send states to WonMan
        if wonD:

            self._wonH_IV.append(wonD['wonH'])
            self._wonH_afterIV.append(sum(self._wonH_IV) / len(self._wonH_IV))

            if self.publish_player_stats:

                my_stats = self._accumulate_global_stats()
                for l,k in zip('abcdefghijklmnoprs'[:len(my_stats)], my_stats):
                    self.tb_add(
                        value=  my_stats[k],
                        tag=    f'player_stats/{l}.{k}',
                        step=   self.fwd_stats_step)

                self.tb_add(
                    value=  self._wonH_IV[-1],
                    tag=    f'player_won/wonH_IV',
                    step=   self.fwd_stats_step)
                self.tb_add(
                    value=  self._wonH_afterIV[-1],
                    tag=    f'player_won/wonH_afterIV',
                    step=   self.fwd_stats_step)

                if len(self._wonH_IV) > 1:
                    wonH_IVstd = statistics.stdev(self._wonH_IV)
                    self.tb_add(
                        value=  wonH_IVstd, # wonH_IV stdev
                        tag=    f'player_won/wonH_IV_std',
                        step=   self.fwd_stats_step)
                    self.tb_add(
                        value=  wonH_IVstd / math.sqrt(len(self._wonH_IV)), # wonH_IV mean stdev, 12.37: https://pl.wikibooks.org/wiki/Statystyka_matematyczna/Twierdzenie_o_rozk%C5%82adzie_normalnym_jednowymiarowym
                        tag=    f'player_won/wonH_IV_mean_std',
                        step=   self.fwd_stats_step)

            if self.publish_more:
                self._publish_FWD_stats(step=self.fwd_stats_step)

            if self.publish_player_stats or self.publish_more:
                self.fwd_stats_step += 1

        return super()._encode_states(player_id, player_stateL)

    # SM & PStatsEx have to be build here, inside process method
    def _pre_process(self) -> None:

        self._wm = WonMan(won_iv=self.won_iv)
        self._wonH_IV = []      # my wonH of interval (DMK_STATS_IV), computed by WonMan
        self._wonH_afterIV = [] # my wonH AFTER interval, sum(wonH_IV)/len(wonH_IV)

        ps_logger = get_child(
            logger=         self._logger,
            name=           'pstatsex',
            change_level=   10)

        # PStatsEx for each table player, 0-my, 1-1st villain, ..
        self._players_stats = {
            pid: {ix: PStatsEx(
                player=         ix,
                use_initial=    False,
                upd_freq=       10,
                logger=         ps_logger,
            ) for ix in (range(N_TABLE_PLAYERS) if self.use_villain_stats else [0])} for pid in self._player_ids}

        super()._pre_process()

    def _do_what_GM_says(self, message: QMessage):

        super()._do_what_GM_says(message)

        if message.type == 'send_dmk_report':
            self.que_to_gm.put(QMessage(
                type=   'dmk_report',
                data=   {
                    'dmk_name':     self.name,
                    'n_hands':      self._wm.get_global_nhands(),           # current number of hands (since init)
                    'wonH_IV':      self._wonH_IV[message.data:],           # wonH of intervals GM is asking for
                    'wonH_afterIV': self._wonH_afterIV[message.data:]}))    # wonH AFTER intervals GM is asking for

        if message.type == 'send_global_stats':
            self.que_to_gm.put(QMessage(
                type=   'global_stats',
                data=   {
                    'dmk_name':     self.name,
                    'global_stats': self._accumulate_global_stats()}))

# Exploring Advanced DMK - implements Policy of EXploring (PEX) while making a decision (active while training only)
class ExaDMK(StaMaDMK, ABC):

    def __init__(
            self,
            enable_pex: bool=           True,   # enables/disables PEX
            pex_max: float=             0.05,   # maximal pex value
            prob_zero: float=           0.2,    # prob of setting: pex = 0
            prob_max: float=            0.2,    # prob of setting: pex = pex_max
            step_min: int=              1000,   # minimal step count to choose new pex
            step_max: int=              100000, # maximal step count to choose new pex
            pid_pex_fraction: float=    1.0,    # performs pex only on fraction <0.0-1.0> of players
            publish_pex=                True,   # publish pex to TB
            **kwargs):

        StaMaDMK.__init__(self, **kwargs)
        self.enable_pex = enable_pex
        self.pex_max = pex_max
        self.prob_max = prob_max
        self.prob_zero = prob_zero
        self.step_min = step_min
        self.step_max = step_max
        self.pid_pex_fraction = pid_pex_fraction
        self.publish_pex = publish_pex

        self._pid_pex = {pid: False for pid in self._player_ids} # enable pex for a player
        self._pex = 0.0  # probability of exploring >> probability of choosing exploring move
        self._step = 0   # step counter - for this number of steps pex will be fixed (0 for sampling in the first step)

    # random probs forced by pex-advanced - keeps pex for n steps, then samples new value
    def __pex_probs(
            self,
            probs: np.ndarray,
            pid: str) -> np.ndarray:

        # eventually set new pex
        if self._step == 0:
            self._step = random.randint(self.step_min, self.step_max) # set new next step counter
            # set factor
            if random.random() < self.prob_max+self.prob_zero:
                if random.random() < self.prob_max/(self.prob_max+self.prob_zero): factor = 1
                else:                                                              factor = 0
            else:                                                                  factor = random.random()
            self._pex = factor * self.pex_max
            self._pid_pex = {pid: random.random() < self.pid_pex_fraction for pid in self._player_ids}
        else: self._step -= 1

        # choose exploring move, encode it into probs
        if self._pid_pex[pid] and random.random()<self._pex:
            move_ix = np.random.choice(self.n_moves)
            probs = np.zeros(shape=self.n_moves)
            probs[move_ix] = 1
        return probs

    # adds sampling with PEX
    def _sample_move(
            self,
            probs: np.ndarray,
            possible_moves :List[bool],
            pid: str) -> int:
        # if PEX is enabled and is trainable - it is only trainable policy
        if self.enable_pex and self.trainable:
            probs = self.__pex_probs(probs, pid)
        return super()._sample_move(probs, possible_moves, pid)

    # adds pex to TB
    def _publish_FWD_stats(self, step):
        super()._publish_FWD_stats(step)
        if self.enable_pex and self.publish_pex:
            self.tb_add(value=self._pex, tag='process.FWD/pex', step=step)

# ***************************************************************************************** NOT abstract implementations

# Random DMK implements baseline/equal (random decision) probs
class RanDMK(StaMaDMK):

    #def __init__(self, **kwargs):
    #    StaMaDMK.__init__(self, **kwargs)

    # calculates probabilities - baseline: sets equal for ALL new states of ALL players with possible moves
    def _calc_probs(self) -> None:
        for pid in self._states_new:
            if self._states_new[pid]:
                if self._states_new[pid][-1].possible_moves:
                    self._states_new[pid][-1].probs = np.asarray([1 / self.n_moves] * self.n_moves) # equal probs

    def save(self): pass

# Neural DMK, with NN (MOTorch) as a deciding model, MOTorch is a sub-object - it is initialized inside a Process
class NeurDMK(ExaDMK):

    def __init__(
            self,
            motorch_point: Optional[POINT]= None,
            reward_share: Optional[int]=    None, # reward sharing (between states) policy, for None every state gets reward/len(moves), for int gets reward/N
            publish_update=                 True,
            **kwargs):
        ExaDMK.__init__(self, **kwargs)
        self._mdl = None
        self.motorch_point = motorch_point or {}
        self.reward_share = reward_share
        self.publish_update = publish_update

    # encodes selection of HH states data into a form accepted by NN input
    # returns only selected states (used by NN)
    def _encode_states(
            self,
            player_id,
            player_stateL: List[STATE],
    ) -> List[GameState]:

        es = super()._encode_states(player_id, player_stateL)
        es_sel = [] # selected states
        ser = f'\nstates encoding report for {player_id}\n'
        for s in es:

            val = s.state_orig_data
            ser += f'> {val}\n'

            # update table cash
            if val[0] == 'T$$':
                self._table_cash[player_id] = [float(val[1][0]), float(val[1][1]), float(val[1][2])]

            # update my cards with my hand
            if val[0] == 'PLH' and val[1][0] == 0:
                self._my_cards[player_id] = [PDeck.cti(c) for c in val[1][1:]]

            # update my cards with table cards
            if val[0] == 'TCD':
                self._my_cards[player_id] += [PDeck.cti(c) for c in val[1]]

            # players POS or MOV
            if val[0] in ['POS','MOV']:

                if val[0] == 'POS':
                    event_id = 0
                    mv_cash = 0.0
                    pl_cash = [float(val[1][2]), 0.0, 0.0]
                    self._pos[player_id][val[1][0]] = val[1][1]  # save POS for next MOVes
                else:
                    event_id = 1 + val[1][1] # 1 + mov_id (index in TBL_MOV)
                    mv_cash = float(val[1][2])
                    pl_cash = [float(val[1][3][0]), float(val[1][3][1]), float(val[1][3][2])]

                nval = {
                    'cards':        [] + self._my_cards[player_id],     # List[7 x int] copy of my cards: 0-52
                    'event_id':     event_id,                           # int: O-(1+len(TBL_MOV))
                    'cash':         [mv_cash, *pl_cash, *self._table_cash[player_id]], # List[7 x float] move cash, pl.cash, pl.cash_ch, pl.cash_cr, table.pot, table.cash_cr, table.cash_tc
                    'pl_id':        val[1][0],                          # int player ID: 0-N_TABLE_PLAYERS
                    'pl_pos':       self._pos[player_id][val[1][0]],    # int player pos: 0-N_TABLE_PLAYERS
                    'pl_stats':     list(self._players_stats[player_id][val[1][0]].player_stats.values()), # List[float,..] 0.0-1.0
                }

                es_sel.append(GameState(nval))
                ser += f'---> {nval}\n'

            if val[0] == 'PRS' and val[1][0] == 0: # my result
                if self._states_dec[player_id]:
                    self._states_dec[player_id][-1].reward = val[1][1] # we can append reward to last state here

                # reset
                self._my_cards[player_id] = []
                self._table_cash[player_id] = (0,0,0)

        self._logger.debug(ser)

        return es_sel

    # add probabilities for at least some states with possible_moves (called by _decisions_from_new_states)
    def _calc_probs(self) -> None:

        n_rows = 0
        got_probs_for_possible = False
        while not got_probs_for_possible:

            vals_row = []
            for pid in self._states_new:
                if self._states_new[pid]:
                    for s in self._states_new[pid]:
                        if s.probs is None:
                            vals_row.append((pid,s.state_orig_data))
                            break

            if not vals_row: break # it is possible, that all probs are done (e.g. possible moves appeared after probs calculated)
            else:
                probs_row = self.__calc_probs_vr(vals_row)
                n_rows += 1
                for pr in probs_row:
                    pid, probs = pr
                    for s in self._states_new[pid]:
                        if s.probs is None:
                            s.probs = probs
                            if s.possible_moves: got_probs_for_possible = True
                            break

        self._processFWD_stats['4.n_rows'].append(n_rows)

    # calculate probs for a row
    def __calc_probs_vr(
            self,
            vals_row :List[tuple]):

        # build batches
        batches = {
            k: [] for k in
            ['cards','event_id','cash','pl_id','pl_pos','pl_stats','enc_cnn_state']}
        pids = []
        for vr in vals_row:

            pid, val = vr
            pids.append(pid) # save list of pid

            # padded cards wrapped (in seq axis)
            cards = val['cards']
            cards += [52] * (7 - len(cards))
            batches['cards'].append([cards])

            # single wrap (in seq axis)
            for k in ['event_id','cash','pl_id','pl_pos','pl_stats']:
                batches[k].append([val[k]])

            batches['enc_cnn_state'].append(self._last_fwd_state[pid])

        # convert batches
        for k in ['cards','event_id','cash','pl_id','pl_pos','pl_stats']:
            batches[k] = self._mdl.convert(batches[k])
        batches['enc_cnn_state'] = self._mdl.convert(np.asarray(batches['enc_cnn_state'])) # from list of np.ndarrays

        out = self._mdl(**batches)
        probs = out['probs']
        fin_states = out['fin_state']
        probs = np.squeeze(probs, axis=1) # remove sequence axis (1)

        probs_row = []
        for ix in range(fin_states.shape[0]):
            pid = pids[ix]
            # TODO: refactor data types
            probs_row.append((pid, probs[ix].cpu().detach().numpy()))
            # TODO: refactor data types, ..here we store history as np.ndarray
            self._last_fwd_state[pid] = fin_states[ix].cpu().detach().numpy() # save fwd states

        return probs_row

    # NN update
    def _training_core(self):

        if self.trainable:

            pidL = [] + self._player_ids

            # for every player: move rewards down to (last) moves (and build rewards dict)
            rewards = {} # {pid: [[99,95,92][85,81,77,74]..]} # indexes of moves, first is always rewarded
            for pid in pidL:
                rewards[pid] = []
                reward = None
                passed_first_reward = False # we need to skip some last(first from the reversed) moves that do not have rewards yet
                move_ixL = [] # list of move index
                for ix in reversed(range(len(self._states_dec[pid]))):

                    st = self._states_dec[pid][ix]

                    if st.reward is not None:
                        passed_first_reward = True

                        # got previous reward without a move, add it here
                        if reward is not None:
                            reward += st.reward
                        else:
                            reward = st.reward

                        st.reward = None

                    if st.move is not None and passed_first_reward: # got move here and it will share some reward
                        if reward is not None: # put that reward here
                            st.reward = reward
                            reward = None
                            # got previous list of mL
                            if move_ixL:
                                rewards[pid].append(move_ixL)
                                move_ixL = []
                        move_ixL.append(ix) # always add cause passed first reward

                if move_ixL: rewards[pid].append(move_ixL) # finally add last

            # remove not rewarded players (rare, but possible)
            pid_not_rewarded = []
            for pid in pidL:
                if not rewards[pid]:
                    rewards.pop(pid)
                    pid_not_rewarded.append(pid)
            if pid_not_rewarded:
                self._logger.debug(f'got not rewarded players: {pid_not_rewarded}')
                for pid in pid_not_rewarded:
                    pidL.remove(pid)

            # share (down) rewards:
            for pid in pidL:
                for move_ixL in rewards[pid]:
                    rIX = move_ixL[0] # index of reward
                    # only when already not shared (..from previous update)
                    if self._states_dec[pid][rIX].reward_sh is None:

                        if self.reward_share is None:
                            rew_sh = self._states_dec[pid][rIX].reward / len(move_ixL)
                        else:
                            rew_sh = self._states_dec[pid][rIX].reward / self.reward_share

                        for mIX in move_ixL:
                            self._states_dec[pid][mIX].reward_sh = rew_sh
                    else:
                        break

            last_rewarded_move = [(pid,rewards[pid][0][0]) for pid in pidL] # [(pid, index of state)]
            last_rewarded_move = sorted(last_rewarded_move, key=lambda x: x[1], reverse=True) # sort decreasing

            half_players = len(self._states_dec) // 2
            if len(last_rewarded_move) < half_players:
                half_players = len(last_rewarded_move)

            last_rewarded_move = last_rewarded_move[:half_players]          # trim
            n_moves_upd = last_rewarded_move[-1][1] + 1                     # n states to use for update (+1 since moves are indexed from 0)
            upd_pid = [e[0] for e in last_rewarded_move]                    # extract pid to update

            # publish UPD process stats (sts, mov, rew)
            if self.publish_more:

                # num of done states
                n_sts = mam([len(self._states_dec[pid]) for pid in self._player_ids])
                self.tb_add(value=n_sts[0], tag='process.UPD/0.n_sts_min')
                self.tb_add(value=n_sts[2], tag='process.UPD/1.n_sts_max')

                # num of states with moves
                n_mov = mam([sum([len(ml) for ml in rewards[pid]]) for pid in pidL])
                self.tb_add(value=n_mov[0], tag='process.UPD/2.n_mov_min')
                self.tb_add(value=n_mov[2], tag='process.UPD/3.n_mov_max')

                # num of states with rewards
                n_rew = mam([len(rewards[pid]) for pid in pidL])
                self.tb_add(value=n_rew[0], tag='process.UPD/4.n_rew_min')
                self.tb_add(value=n_rew[2], tag='process.UPD/5.n_rew_max')

                self.tb_add(value=n_sts[1] / n_mov[1], tag='process.UPD/6.n_sts/mov')
                self.tb_add(value=n_sts[1] / n_rew[1], tag='process.UPD/7.n_sts/rew')
                self.tb_add(value=n_mov[1] / n_rew[1], tag='process.UPD/8.n_mov/rew')

            # build batches
            batches = {
                k: [] for k in
                ['cards','event_id','cash','pl_id','pl_pos','pl_stats','move','reward','enc_cnn_state']}
            for pid in upd_pid:

                # build seqs
                seqs = {
                    k: [] for k in
                    ['cards','event_id','cash','pl_id','pl_pos','pl_stats','move','reward','enc_cnn_state']}
                for state in self._states_dec[pid][:n_moves_upd]:

                    val = state.state_orig_data

                    # padded cards
                    cards = val['cards']
                    cards += [52]*(7-len(cards)) # pad cards
                    seqs['cards'].append(cards)

                    for k in ['event_id','cash','pl_id','pl_pos','pl_stats']:
                        seqs[k].append(val[k])

                    seqs['move'].append(state.move if state.move is not None else 0)
                    seqs['reward'].append(state.reward_sh / TABLE_CASH_START if state.reward_sh is not None else 0)

                for k in ['cards','event_id','cash','pl_id','pl_pos','pl_stats','move','reward']:
                    batches[k].append(seqs[k])
                batches['enc_cnn_state'].append(self._last_upd_state[pid])

            # convert batches
            for k in ['cards','event_id','cash','pl_id','pl_pos','pl_stats','move','reward']:
                batches[k] = self._mdl.convert(batches[k])
            batches['enc_cnn_state'] = self._mdl.convert(np.asarray(batches['enc_cnn_state'])) # from list of np.ndarrays

            out = self._mdl.backward(empty_cuda_cache=True, **batches) # INFO: empty_cuda_cache allows to train bigger models on limited GPU mem
            fin_states = out['fin_state']

            # save upd states
            for ix in range(fin_states.shape[0]):
                self._last_upd_state[upd_pid[ix]] = fin_states[ix].cpu().detach().numpy()

            # publish NN backprop stats
            if self.publish_update:
                self._ze_pro_enc.process(out['zsL_enc'], self.upd_step)
                self._ze_pro_cnn.process(out['zsL_cnn'], self.upd_step)
                for ix,k in enumerate([
                    'loss',
                    'currentLR',
                    'gg_norm',
                    'gg_norm_clip',
                    'min_probs_mean',
                    'max_probs_mean',
                ]):
                    self.tb_add(value=out[k], tag=f'nn/{ix}.{k}')
                self.tb_add(value=n_moves_upd * len(upd_pid), tag='nn/6.batchsize')

            return n_moves_upd, upd_pid

        return None

    # flush properly
    def _flush_states_dec(self, ust_details) -> None:
        if ust_details is None: super()._flush_states_dec(ust_details) # to remove all while not learning
        # leave only not used
        else:
            n_moves_upd, upd_pid = ust_details # unpack
            for pid in upd_pid:
                self._states_dec[pid] = self._states_dec[pid][n_moves_upd:]
            self._n_states_dec -= n_moves_upd * len(upd_pid)

    # overrides StaMaDMK with MOTorch, StatsManager_TB and 2x ZeroesProcessor
    def _pre_process(self):

        super()._pre_process()

        self._mdl = DMK_MOTorch(
            name=           self.name,
            save_topdir=    self.save_topdir,
            logger=         self._logger,
            tbwr=           self._tbwr, # INFO: probably not used by this MOTorch..
            **self.motorch_point)

        # TODO: refactor data types, ..here we store history as np.ndarray
        zero_state = self._mdl.module.enc_cnn.get_zero_history().cpu().detach().numpy()

        self._last_fwd_state =   {pa: zero_state for pa in self._player_ids}  # NN state after last fwd
        self._last_upd_state =   {pa: zero_state for pa in self._player_ids}  # NN state after last upd

        # properties below are updated while encoding states
        self._my_cards =         {pa: []      for pa in self._player_ids}  # current cards of player
        self._table_cash =       {pa: (0,0,0) for pa in self._player_ids}  # current (before player move) table cash (from T$$ state)
        self._pos =              {pa: {}      for pa in self._player_ids}  # current players positions {pl_id:pos}

        self._ze_pro_enc = ZeroesProcessor(
            intervals=      (5,20),
            tag_pfx=        'nane_enc',
            tbwr=           self._tbwr) if self.publish_update else None
        self._ze_pro_cnn = ZeroesProcessor(
            intervals=      (5,20),
            tag_pfx=        'nane_cnn',
            tbwr=           self._tbwr) if self.publish_update else None

    def _do_what_GM_says(self, message: QMessage):

        super()._do_what_GM_says(message)

        # reloads NN model checkpoint
        if message.type == 'reload_model':
            self.__reload_model()
            dmk_message = QMessage(type='dmk_model_reloaded', data=self.name)
            self.que_to_gm.put(dmk_message)

    # reloads model checkpoint (after GX)
    def __reload_model(self):
        self._mdl.load_ckpt()

    # saves MOTorch
    def save(self):
        if self._mdl: self._mdl.save()
        else: self._logger.warning('NN model (@NeurDMK) has not been saved, probably not initialized yet')

    @staticmethod
    def dmk_motorch_from_point(
            motorch_point: POINT,
            logger=     None,
            loglevel=   20) -> None:
        model = DMK_MOTorch(**motorch_point, logger=logger, loglevel=loglevel)
        model.save()

# Foldered DMK - ParaSave (POINT) + NeurDMK (MOTorch) --> saving / serialization / genetic crossing (GX) / version management
class FolDMK(ParaSave, NeurDMK):

    SAVE_TOPDIR = DMK_MODELS_FD
    SAVE_FN_PFX = DMK_POINT_PFX

    def __init__(
            self,
            name: str,
            age: int=                       0,              # FolDMK age, updated by GM (number of TR games)
            save_topdir=                    SAVE_TOPDIR,
            save_fn_pfx=                    SAVE_FN_PFX,
            motorch_point: Optional[POINT]= None,
            **kwargs):

        # load point from folder
        point_saved = ParaSave.load_point(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

        point = {
            'name':         name,
            'age':          age,
            'save_topdir':  save_topdir,
            'save_fn_pfx':  save_fn_pfx}
        point.update(point_saved)
        point['motorch_point'] = motorch_point or {} # always overwrite saved motorch_point
        point.update(kwargs)

        point_neurdmk = {}
        point_neurdmk.update(point)

        # remove ParaSave params
        for k in ['save_fn_pfx','psdd','gxable','age','parents']:
            if k in point_neurdmk:
                point_neurdmk.pop(k)

        NeurDMK.__init__(self, **point_neurdmk)

        # eventually remove DMK logger (given with kwargs)
        if 'logger' in point:
            point.pop('logger')

        parasave_logger = get_child(
            logger=         self._logger,
            name=           'parasave',
            change_level=   10)
        ParaSave.__init__(self, logger=parasave_logger, **point)


    def _do_what_GM_says(self, message: QMessage):

        super()._do_what_GM_says(message)

        # updates self (DMK) params and iLR of NN model
        if message.type == 'reload_dmk_settings':
            self.update(message.data)
            if 'iLR' in message.data: self._mdl.update_LR(message.data['iLR'])
            dmk_message = QMessage(
                type=   'dmk_settings_accepted',
                data=   f'{self.name} accepted new settings: {message.data}')
            self.que_to_gm.put(dmk_message)

    # saves FolDMK, INFO: NN saved only when trainable
    def save(self):
        ParaSave.save_point(self)
        if self.trainable:
            NeurDMK.save(self)

    # copies saved FolDMK
    @staticmethod
    def copy_saved(
            name_src: str,
            name_trg: str,
            save_topdir_src: str=           DMK_MODELS_FD,
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: str=               DMK_POINT_PFX,
            logger=                         None,
            loglevel=                       30):

        ParaSave.copy_saved_point(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            save_fn_pfx=        save_fn_pfx,
            logger=             logger,
            loglevel=           loglevel)

        MOTorch.copy_saved(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            logger=             logger,
            loglevel=           loglevel)

    # performs GX on saved FolDMK (without even building child objects)
    @classmethod
    def gx_saved(
            cls,
            name_parent_main: str,
            name_parent_scnd: Optional[str],                # if not given makes GX only with main parent
            name_child: str,
            save_topdir_parent_main: Optional[str]= None,
            save_topdir_parent_scnd: Optional[str]= None,
            save_topdir_child: Optional[str]=       None,
            save_fn_pfx: Optional[str]=             None,
            do_gx_ckpt=                             True,
            ratio=                                  0.5,
            noise=                                  0.03,
            logger=                                 None,
            loglevel=                               20,
    ) -> None:

        if not save_topdir_parent_main: save_topdir_parent_main = cls.SAVE_TOPDIR
        if not save_topdir_parent_scnd: save_topdir_parent_scnd = save_topdir_parent_main
        if not save_topdir_child: save_topdir_child = save_topdir_parent_main
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        cls.gx_saved_point(
            name_parent_main=           name_parent_main,
            name_parent_scnd=           name_parent_scnd,
            name_child=                 name_child,
            save_topdir_parent_main=    save_topdir_parent_main,
            save_topdir_parent_scnd=    save_topdir_parent_scnd,
            save_topdir_child=          save_topdir_child,
            save_fn_pfx=                save_fn_pfx,
            logger=                     logger,
            loglevel=                   loglevel)

        # set proper age
        child_age = 0
        if do_gx_ckpt:
            age_pm = cls.load_point(
                name=           name_parent_main,
                save_topdir=    save_topdir_parent_main,
                save_fn_pfx=    save_fn_pfx)['age']
            age_ps = cls.load_point(
                name=           name_parent_scnd,
                save_topdir=    save_topdir_parent_scnd,
                save_fn_pfx=    save_fn_pfx)['age']
            child_age = max(age_pm,age_ps)
        cls.oversave_point(
            name=                       name_child,
            save_topdir=                save_topdir_child,
            save_fn_pfx=                save_fn_pfx,
            age=                        child_age)

        MOTorch.gx_saved(
            name_parent_main=           name_parent_main,
            name_parent_scnd=           name_parent_scnd,
            name_child=                 name_child,
            save_topdir_parent_main=    save_topdir_parent_main,
            save_topdir_parent_scnd=    save_topdir_parent_scnd,
            save_topdir_child=          save_topdir_child,
            save_fn_pfx=                MOTorch.SAVE_FN_PFX,
            do_gx_ckpt=                 do_gx_ckpt,
            ratio=                      ratio,
            noise=                      noise,
            logger=                     logger,
            loglevel=                   loglevel)

    @staticmethod
    def from_points(
            foldmk_point: POINT,
            motorch_point: POINT,
            logger=     None
    ) -> None:
        NeurDMK.dmk_motorch_from_point(motorch_point=motorch_point, logger=logger)
        foldmk = FolDMK(**foldmk_point, logger=logger, loglevel=30)
        ParaSave.save_point(foldmk)

# Human Driven DMK - enables a human to make a decision with TK GUI
class HuDMK(StaMaDMK):

    def __init__(
            self,
            tk_gui: GUI_HDMK, # pass to get ques
            **kwargs):
        StaMaDMK.__init__(self, **kwargs)
        self.tk_IQ = tk_gui.tk_que
        self.tk_OQ = tk_gui.out_que

    # additionally sends incoming states to TK
    def _encode_states(
            self,
            player_id,
            player_stateL: List[STATE]) -> List[GameState]:
        for state in player_stateL:
            message = QMessage(type='state', data=state)
            self.tk_IQ.put(message)
        return super()._encode_states(player_id, player_stateL)

    # prepares probs <- TK <- human decision
    def _calc_probs(self) -> None:
        probs = np.zeros(self.n_moves)
        for pid in self._states_new:
            if self._states_new[pid]:
                last_state = self._states_new[pid][-1]
                if last_state.possible_moves:

                    # send data to TK
                    message = QMessage(
                        type=   'possible_moves',
                        data=   {
                            'possible_moves':   last_state.possible_moves,
                            'moves_cash':       last_state.moves_cash})
                    self.tk_IQ.put(message)

                    # get decision from TK
                    tk_message = self.tk_OQ.get()
                    probs[tk_message.data] = 1

                    last_state.probs = probs

    def save(self): pass