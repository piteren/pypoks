import math
from abc import abstractmethod, ABC
from multiprocessing import Process
import numpy as np
from pypaq.pytypes import NPL
from pypaq.lipytools.stats import mam
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.time_reporter import TimeRep
from pypaq.mpython.mptools import Que, QMessage
from pypaq.mpython.mpdecor import proc_wait
from pypaq.pms.parasave import ParaSave
from pypaq.pms.base import POINT
import random
import statistics
import time
from typing import List, Tuple, Optional, Dict, Any

from envy import DMK_MODELS_FD, PLAYER_STATS_USED, PyPoksException
from pologic.podeck import PDeck
from pologic.hand_history import STATE
from podecide.game_state import GameState
from podecide.dmk_motorch import DMK_MOTorch
from podecide.stats.won_manager import WonMan
from podecide.stats.player_stats import PStatsEx
from podecide.tools.tbwr_dmk import TBwr_DMK

# DMK won interval
# number of hands for which WON$ and some other stats are accumulated and then processed
# it is an important constant for DMK (-> WonMan & GamesManager)
# since produces data used to monitor separation, stddev, progress -> training process
WON_IV = 1000


# ***************************************************************************************************** abstract classes

class DMK(ABC):
    """ Decision MaKer
    basic interface to decide for poker players (PPlayer on PTable) """

    SAVE_TOPDIR = DMK_MODELS_FD

    def __init__(
            self,
            name: str,                          # name should be unique (@table)
            table_size: int,                    # number of table players / opponents
            table_moves: List,                  # moves supported by DMK
            n_players: int=             1,      # number of table players managed by one DMK
            family: str=                'a',    # family (type) used to group DMKs and manage together
            save_topdir: Optional[str]= None,
            logger=                     None,
            loglevel=                   20,
    ):

        self.name = name
        self.save_topdir = save_topdir or self.SAVE_TOPDIR

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                add_stamp=  False,
                folder=     f'{self.save_topdir}/{self.name}',
                level=      loglevel)
        self._logger = logger

        self._player_ids = [f'{self.name}_{ix}' for ix in range(n_players)] if n_players>1 else [self.name] # DMK keeps unique ids (str) of table players
        self.table_moves = table_moves
        self.table_size = table_size
        self.family = family

        self._logger.info(f'*** {self.__class__.__name__} (DMK) : {self.name} *** initialized')
        self._logger.debug(f'> {len(self._player_ids)} table player(s)')
        moves_nfo = f'> {len(self.table_moves)} supported moves:'
        for ix,m in enumerate(self.table_moves):
            moves_nfo += f'\n> {ix}: {m}'
        self._logger.debug(moves_nfo)

    @abstractmethod
    def collect_states(
            self,
            player_id: str,
            player_states: List[Any]):
        """ takes & stores player states """
        pass

    @abstractmethod
    def make_decisions(self) -> List[Tuple[str,int,NPL]]:
        """ makes decisions [(player_id,move,probs)..] for one/some/all players with allowed moves """
        pass


class MethDMK(DMK, ABC):
    """ Methods DMK
    extends DMK baseline with sub-methods and data structures """

    def __init__(self, seed=123, **kwargs):

        DMK.__init__(self, **kwargs)

        self._rng = np.random.default_rng(seed) # random numbers generator

        # variables below store new states data (states sent by table and not decided by DMK yet)
        self._states_new: Dict[str,List[GameState]] = {pid: [] for pid in self._player_ids} # dict of new states lists
        self._n_states_new = 0  # cache, number of all _states_new == sum([len(l) for l in self._states_new.values()])

    def collect_states(
            self,
            player_id: str,
            player_states: List[Tuple]):
        """ takes player states, encodes and saves """

        encoded_states = self._encode_states(player_id, player_states)

        # save
        if encoded_states:
            self._states_new[player_id] += encoded_states
            self._n_states_new += len(encoded_states)

    @abstractmethod
    def _encode_states(
            self,
            player_id: str,
            player_stateL: List[STATE],
    ) -> List[GameState]:
        """ encodes player states into type appropriate for DMK to make decisions """
        return [GameState(state_orig_data=value) for value in player_stateL] # wraps into list of GameState

    def _collect_allowed_moves(
            self,
            player_id: str,
            allowed_moves :List[bool],
            moves_cash :List[int]):
        """ takes allowed_moves (and their cash) from poker player """
        last_state = self._states_new[player_id][-1]
        last_state.allowed_moves = allowed_moves
        last_state.moves_cash = moves_cash

    def make_decisions(self) -> List[Tuple[str,int,NPL]]:
        """ makes decisions then flushes states """

        decL = self._decisions_from_new_states()  # get decisions list

        # flush new states using decisions list
        for dec in decL:
            pid, _, _ = dec
            self._states_new[pid] = []  # reset

        return decL

    def _decisions_from_new_states(self) -> List[Tuple[str,int,NPL]]:
        """ makes decisions & returns list of decisions using data from _states_new """
        self._compute_probs()
        decL = self._sample_moves_for_ready_players()
        return decL

    @abstractmethod
    def _compute_probs(self) -> None:
        """ computes probabilities for some states in _states_new """
        pass

    def _sample_moves_for_ready_players(self) -> List[Tuple[str,int,NPL]]:
        """ samples moves for players with ready data (allowed_moves and probs) """
        decL = []
        for pid in self._states_new:
            if self._states_new[pid]: # not empty
                last_state = self._states_new[pid][-1]
                if last_state.allowed_moves is not None and last_state.probs is not None:
                    move = self._sample_move(
                        probs=              last_state.probs,
                        allowed_moves =     last_state.allowed_moves,
                        pid=                pid)
                    last_state.move = move # add move to the last state
                    decL.append((pid, move, last_state.probs))
        return decL

    def _sample_move(
            self,
            probs: np.ndarray,
            allowed_moves: List[bool],
            pid: str        # pid to be used by inherited implementations
    ) -> int:
        """ samples move form allowed_moves using given probabilities """

        probs = probs * allowed_moves                      # mask probs

        # take mask if no intersection
        if sum(probs) == 0:
            probs = np.asarray(allowed_moves).astype(float)

        return self._rng.choice(len(self.table_moves), p=probs/sum(probs))


class MeTrainDMK(MethDMK, ABC):
    """ Methods Trainable DMK
    adds training methods and data structures
    training is triggered while making decisions """

    def __init__(
            self,
            trainable=          False,
            upd_trigger: int=   30000,  # half of this number is an estimated target batch size (number of decisions) for update (check policy of states selection @UPD - half_rectangle in trapeze)
            upd_step=           0,      # updates counter (clock)
            **kwargs):

        MethDMK.__init__(self, **kwargs)

        self.trainable =    trainable
        self.upd_trigger =  upd_trigger
        self.upd_step =     upd_step

        # variables below store moves/decisions data
        self._states_dec: Dict[str, List[GameState]] = {pid: [] for pid in self._player_ids}  # dict with decided states lists (for update)
        self._n_states_dec = 0  # cache, number of all _states_dec == sum([len(l) for l in self._states_dec.values()])

        # ques of Update Synchronizer - if set - will be used to synchronize update
        self._upd_sync_que_out: Optional[Que] = None
        self._upd_sync_que_in: Optional[Que] = None
        self._waiting_for_permission = False

    def make_decisions(self) -> List[Tuple[str,int,NPL]]:
        """ makes decisions > moves states > trains policy - overrides MethDMK method, which does not train """
        decL = self._decisions_from_new_states()    # get decisions list
        self.__move_states(decL)                    # move states from _states_new to _states_dec

        # learn/train policy, it is additionally triggered inside
        if self.trainable:
            self.__train_policy()

        decr = f'\n{self.name} made decisions:\n'
        for dec in decL:
            decr += f'> {dec}\n'
        self._logger.debug(decr)

        return decL

    def __move_states(self, decL :List[Tuple[str,int,NPL]]):
        """ moves states (from _new to _dec) having decisions list """

        for dec in decL:

            pid, move, _ = dec
            states = self._states_new[pid]      # take all new states
            self._states_new[pid] = []          # reset
            self._n_states_new -= len(states)

            # put states to dec
            if self.trainable:
                self._states_dec[pid] += states
                self._n_states_dec += len(states)

    def set_upd_sync(
            self,
            que_out: Optional[Que]= None,
            que_in: Optional[Que]=  None,
    ):
        """ allows to set Update Synchronizer ques """
        self._upd_sync_que_out = que_out
        self._upd_sync_que_in = que_in

    def __train_policy(self) -> None:
        """ trains DMK (policy) """

        if self._n_states_dec > self.upd_trigger: # only when trigger fires

            update_allowed = True

            # additionally triggers update with ticket request
            if self._upd_sync_que_out is not None:

                if not self._waiting_for_permission:
                    msg = QMessage(type='update_request', data=self.name)
                    self._upd_sync_que_out.put(msg)
                    self._waiting_for_permission = True

                update_allowed = False
                msg = self._upd_sync_que_in.get(block=False)
                if msg:  # since UpdSync sends ONLY tickets to DMK, msg type is not checked here
                    update_allowed = True
                    self._waiting_for_permission = False

            if update_allowed:
                ust_details = self._training_core()
                self._flush_states_dec(ust_details)
                self.upd_step += 1

                # return ticket
                if self._upd_sync_que_out is not None:
                    msg = QMessage(type='ticket', data=self.name)
                    self._upd_sync_que_out.put(msg)

    def _training_core(self):
        """ trains (updates self policy) with _states_dec (cache of taken moves & received rewards) """
        return None

    def _flush_states_dec(self, ust_details) -> None:
        """ flushes _states_dec using information from ust_details """
        self._states_dec = {pid: [] for pid in self._player_ids} # flushes all _states_dec (here baseline, not uses ust_details)
        self._n_states_dec = 0


class QueDMK(MeTrainDMK, ABC):
    """ Qued DMK
    extends DMK with Ques and Process (QueDMK has a _process property)
    (to act on QPTable with QPPlayer and to be managed by GamesManager) """

    def __init__(
            self,
            name: str,
            publishFWD=         True, # allows to publish process.FWD stats
            publishUPD=         True, # allows to publish process.UPD stats
            collect_loop_stats= False,
            **kwargs):

        MeTrainDMK.__init__(self, name=name, **kwargs)

        self._process = Process(name=f'QueDMK_process:{name}', target=self.__dmk_proc)

        self._que_to_gm = None # here QuedDMK sends data to GamesManager, data is in form (name, command, data)
        self._que_from_gm = Que() # here QuedDMK receives data from GamesManager, data is in form (command, data)

        self._running_process = False  # flag for running process loop
        self._running_game = False  # flag for running game loop

        # stats of the qued process, check __reset for details
        self._processFWD_stats_data = {}
        self._reset_processFWD_stats_data()

        # every DMK creates part of DMK-Player Ques network
        self._que_from_player = Que() # here player puts data for DMK
        self._queD_to_player = {pid: Que() for pid in self._player_ids} # dict with ques where DMK puts data for each player

        self.publishFWD = publishFWD
        self.publishUPD = publishUPD
        self._collect_loop_stats = collect_loop_stats
        self._tbwr: Optional[TBwr_DMK] = None

    def start(self):
        self._process.start()

    def __dmk_proc(self):
        """ target of Process """

        # first call _pre_process()
        self._pre_process()
        message = QMessage(
            type=   'dmk_status',
            data=   f'{self.name} (DMK) _pre_process() done')
        self._que_to_gm.put(message)

        # here wait for start loop message (or other after)
        self._running_process = True
        while self._running_process:
            gm_data = self.que_from_gm.get()
            self._do_what_GM_says(gm_data)

        # being here means QueDMK finishes its process..
        message = QMessage(
            type=   'dmk_status',
            data=   f'{self.name} (DMK) process finished')
        self._que_to_gm.put(message)

    def _pre_process(self) -> None:
        """ method called BEFORE process loop, builds objects that HAVE to be built in process memory scope """

        # build TBwr here, inside a process
        fd = f'{self.save_topdir}/{self.name}'
        prep_folder(fd)
        self._tbwr = TBwr_DMK(collect_loop_stats=self._collect_loop_stats, logdir=fd)

    def _do_what_GM_says(self, message: QMessage):
        """ processes GM messages """

        if message.type == 'start_dmk_loop':
            self._running_game = True
            self.__decisions_loop()

        if message.type == 'publish_loop_stats':
            d = message.data
            self._tbwr.publish_loop_stats(step=d['step'])
            self._tbwr.add_force(value=d['position'],       tag='loop/position',       step=d['step'])
            self._tbwr.add_force(value=d['wonH'],           tag='loop/wonH',           step=d['step'])
            self._tbwr.add_force(value=d['wonH_IV_stddev'], tag='loop/wonH_IV_stddev', step=d['step'])
            self._tbwr.flush()

        if message.type == 'stop_dmk_loop':
            self._running_game = False

        if message.type == 'stop_dmk_process':
            self._running_process = False

        if message.type == 'save_dmk':
            self.save()
            dmk_message = QMessage(type='dmk_saved', data=self.name)
            self._que_to_gm.put(dmk_message)

    def __decisions_loop(self):
        """ processes incoming data (from players or GM) and sends back decisions (to players) or other data (to GM)
        it is a core method that call other important - previously implemented, like:
       - collect_states
       - collect_allowed_moves
       - make_decisions, which triggers training_core """

        message = QMessage(
            type=   'dmk_status',
            data=   f'{self.name} (DMK) decisions_loop started')
        self._que_to_gm.put(message)

        n_waiting = 0 # num players (-> tables) waiting for decision
        while self._running_game:

            n_players = len(self._player_ids)

            # 'flush' the que of data from players
            pmL = []
            while True:
                player_message = self.que_from_player.get(block=False)
                if player_message: pmL.append(player_message)
                else: break

            n_msg_decision = 0
            for player_message in pmL:

                data = player_message.data

                if player_message.type == 'state_changes':
                    self.collect_states(
                        player_id=      data['id'],
                        player_states=  data['state_changes'])

                # decision request, ..it blocks the source table
                if player_message.type == 'make_decision':
                    self._collect_allowed_moves(
                        player_id=      data['id'],
                        allowed_moves=  data['allowed_moves'],
                        moves_cash=     data['moves_cash'])
                    n_msg_decision += 1

            n_waiting += n_msg_decision
            self._processFWD_stats_data['0.requestedF'].append(n_msg_decision / n_players)

            # if got any waiting >> make decisions and put them to players
            if n_waiting:
                self._processFWD_stats_data['1.waitingF'].append(n_waiting / n_players)

                s_time = time.time()
                decL = self.make_decisions()
                n_waiting -= len(decL)
                self._processFWD_stats_data['2.unlockedF'].append(len(decL) / n_players)
                self._processFWD_stats_data['3.decisions_time'].append((time.time() - s_time) / len(decL))

                # send decisions
                for d in decL:
                    pid, move, probs = d
                    message = QMessage(
                        type=   'move',
                        data=   {'selected_move':move, 'probs':probs})
                    self.queD_to_player[pid].put(message)
                    self._processFWD_stats_data['probs'].append(probs)

            # eventually get data from GM, ..way to exit game_loop
            gm_message = self.que_from_gm.get(block=False)
            if gm_message:
                self._do_what_GM_says(gm_message)

        message = QMessage(
            type=   'dmk_status',
            data=   f'{self.name} (DMK) decisions_loop stopped')
        self._que_to_gm.put(message)

    def _decisions_from_new_states(self) -> List[Tuple[str,int,NPL]]:
        """ adds histogram """

        ### add histogram data to _process_stats

        nd = {}
        for pid in self._states_new:
            l = len(self._states_new[pid])
            if l not in nd: nd[l] = 0
            nd[l] += 1

        hist_nfo = ''
        for k in sorted(list(nd.keys())):
            hist_nfo += f'{k:d}:{nd[k]:d} '
        self._processFWD_stats_data['new_states_hist'].append(hist_nfo[:-1])

        return super()._decisions_from_new_states()

    def _sample_move(
            self,
            probs: np.ndarray,
            allowed_moves :List[bool],
            pid: str,
    ) -> int:
        """ adds not allowed moves probs stats monitoring """
        not_allowed_moves = ~np.asarray(allowed_moves)
        self._processFWD_stats_data['probs_nam'].append(sum(probs * not_allowed_moves))
        return super()._sample_move(probs, allowed_moves, pid)

    def _publish_FWD_stats(self, step) -> None:
        """ publishes processFWD stats, resets """

        # manage histogram of new states
        new_states_hist_data = []
        for str_upd in self._processFWD_stats_data.pop('new_states_hist'):
            # str_upd looks like: ['0:207 2:93', '0:232 3:63 4:5',..], it says that 207 players had 0 new states, 93 had 2, in next iteration 232 had 0..
            sn = str_upd.split(' ')
            for sns in sn:
                snss = sns.split(':')
                val = int(snss[0])
                if val: # remove 0
                    new_states_hist_data += [val] * int(snss[1])
        self._tbwr.add_histogram(
            values= np.asarray(new_states_hist_data),
            tag=    'process.FWD.new_states',
            step=   step)

        ### manage probs stats
        probs = self._processFWD_stats_data.pop('probs')
        probs = np.stack(probs, axis=0)

        # entropy
        entropy = np.mean((-probs * np.log2(probs)).sum(axis=-1))
        self._tbwr.add(value=entropy, tag=f'policy/entropy', step=step)

        # 123mean
        for r in [1, 2, 3]:
            inds = np.argmax(probs, axis=-1)
            vals = np.max(probs, axis=-1)
            probs[range(len(inds)), inds] = 0
            self._tbwr.add(value=np.mean(vals), tag=f'policy/probs_{r}mean', step=step)

        na_probs = self._processFWD_stats_data.pop('probs_nam')
        self._tbwr.add(
            value=  sum(na_probs) / len(na_probs) if len(na_probs) else 0,
            tag=    f'policy/probs_nam',
            step=   step)

        # process rest of stats (average)
        for k in self._processFWD_stats_data:
            st = self._processFWD_stats_data[k]
            val = sum(st) / len(st) if len(st) else 0
            self._tbwr.add(value=val, tag=f'process.FWD/{k}', step=step)

        self._reset_processFWD_stats_data()

    def _reset_processFWD_stats_data(self) -> None:
        self._processFWD_stats_data = {
            '0.requestedF':        [], # List[float] - factor of players that send request for a decision in this loop
            '1.waitingF':          [], # List[float] - factor of waiting players
            '2.unlockedF':         [], # List[float] - factor of players unlocked - received decision in this loop
            '3.decisions_time':    [], # List[float] - decision time - FWD call (s)
            '4.n_rows':            [], # List[int]   - number of rows needed to be computed to get to allowed moves
            '5.row_widthF':        [], # List[float] - factor of players for which probs in a row were computed
            'probs_nam':           [], # List[float] - factor of probs given for not allowed moves
            'probs':               [], # List[np.array] - list of FWD probs
            'new_states_hist':     []} # List[str]   - histogram of num new states (while calculating probs & making decisions)

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


class StaMaDMK(QueDMK, ABC):
    """ Stats Manager DMK
    adds WonMan & PStatsEx objects
    communicates with GamesManager (GM)
    """

    def __init__(
            self,
            won_iv=                 WON_IV,
            fwd_stats_step=         0,      # FWD stats step, increased every WON_IV
            build_villain_stats=    False,  # TODO: use it in the future
            publish_player_stats=   True,   # player (poker + FWD) stats
            n_player_stats=         10,     # publish stats every n step
            **kwargs):

        QueDMK.__init__(self, **kwargs)

        self.won_iv = won_iv
        self.fwd_stats_step = fwd_stats_step

        self.build_villain_stats = build_villain_stats
        self.publish_player_stats = publish_player_stats
        self.n_player_stats = n_player_stats

    def _accumulate_global_stats(self) -> Dict[str,float]:
        """ prepares macro-averaged stats of players -> DMK stats """
        my_statsL = [self._pstats_ex[pid][0].player_stats for pid in self._player_ids]
        my_stats = {k: 0.0 for k in my_statsL[0]}
        nk = len(self._player_ids)
        for e in my_statsL:
            for k in my_stats:
                my_stats[k] += e[k]
        for k in my_stats:
            my_stats[k] /= nk
        return my_stats

    def _encode_states(
            self,
            player_id,
            player_stateL: List[STATE],
    ) -> List[GameState]:
        """ adds stats management """

        for i in (range(self.table_size) if self.build_villain_stats else [0]):
            self._pstats_ex[player_id][i].process_states(player_stateL)

        wonD = self._wm.process_states(player_stateL) # send states to WonMan
        if wonD:

            self._wonH_IV.append(wonD['wonH'])
            self._wonH_afterIV.append(sum(self._wonH_IV) / len(self._wonH_IV))

            if self.fwd_stats_step % self.n_player_stats == 0:

                if self.publish_player_stats:

                    my_stats = self._accumulate_global_stats()
                    for l,k in zip('abcdefghijklmnoprs'[:len(my_stats)], my_stats):
                        self._tbwr.add(
                            value=  my_stats[k],
                            tag=    f'player_stats/{l}.{k}',
                            step=   self.fwd_stats_step)

                    self._tbwr.add(
                        value=  self._wonH_IV[-1],
                        tag=    f'player_won/wonH_IV',
                        step=   self.fwd_stats_step)
                    self._tbwr.add(
                        value=  self._wonH_afterIV[-1],
                        tag=    f'player_won/wonH_afterIV',
                        step=   self.fwd_stats_step)

                    if len(self._wonH_IV) > 1:
                        wonH_IVstd = statistics.stdev(self._wonH_IV)
                        self._tbwr.add(
                            value=  wonH_IVstd, # wonH_IV stddev
                            tag=    f'player_won/wonH_IV_std',
                            step=   self.fwd_stats_step)
                        self._tbwr.add(
                            value=  wonH_IVstd / math.sqrt(len(self._wonH_IV)), # wonH_IV mean stddev, 12.37: https://pl.wikibooks.org/wiki/Statystyka_matematyczna/Twierdzenie_o_rozk%C5%82adzie_normalnym_jednowymiarowym
                            tag=    f'player_won/wonH_IV_mean_std',
                            step=   self.fwd_stats_step)

                if self.publishFWD:
                    self._publish_FWD_stats(step=self.fwd_stats_step)

            if self.publish_player_stats or self.publishFWD:
                self.fwd_stats_step += 1

        return super()._encode_states(player_id, player_stateL)

    def _pre_process(self) -> None:
        """ SM & PStatsEx have to be build here, inside process method """

        self._wm = WonMan(won_iv=self.won_iv)
        self._wonH_IV = []      # my wonH of interval (DMK_STATS_IV), computed by WonMan
        self._wonH_afterIV = [] # my wonH AFTER interval, sum(wonH_IV)/len(wonH_IV)

        # PStatsEx for each table player, 0-my, 1-1st villain, ..
        ps_logger = get_child(logger=self._logger, name='pstatsex')
        self._pstats_ex = {
            pid: {ix: PStatsEx(
                player=         ix,
                table_size=     self.table_size,
                table_moves=    self.table_moves,
                use_initial=    False,
                upd_freq=       10,
                logger=         ps_logger,
            ) for ix in (range(self.table_size) if self.build_villain_stats else [0])} for pid in self._player_ids}

        super()._pre_process()

    def _do_what_GM_says(self, message: QMessage):

        super()._do_what_GM_says(message)

        if message.type == 'send_dmk_report':
            self._que_to_gm.put(QMessage(
                type=   'dmk_report',
                data=   {
                    'dmk_name':     self.name,
                    'n_hands':      self._wm.get_global_nhands(),           # current number of hands (since init)
                    'wonH_IV':      self._wonH_IV[message.data:],           # wonH of intervals GM is asking for
                    'wonH_afterIV': self._wonH_afterIV[message.data:]}))    # wonH AFTER intervals GM is asking for

        if message.type == 'send_global_stats':
            self._que_to_gm.put(QMessage(
                type=   'global_stats',
                data=   {
                    'dmk_name':     self.name,
                    'global_stats': self._accumulate_global_stats()}))


class ExaDMK(StaMaDMK, ABC):
    """ Exploring Advanced DMK
    implements Policy of EXploring (PEX) while making a decision (active while training only).
    DMK is a probabilistic model, which by nature explores,
    this is an additional policy that may be turned od and configured.

    INFO: ExaDMK functionality probably should be turned off for PPO
    """

    def __init__(
            self,
            enable_pex: bool=           False,  # enables/disables PEX
            pex_max: float=             0.05,   # maximal pex value
            prob_zero: float=           0.2,    # prob of setting: pex = 0
            prob_max: float=            0.2,    # prob of setting: pex = pex_max
            step_min: int=              1000,   # minimal step count to choose new pex
            step_max: int=              100000, # maximal step count to choose new pex
            pid_pex_fraction: float=    1.0,    # performs pex only on fraction <0.0-1.0> of players
            publish_pex=                False,  # publish pex to TB
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

    def __pex_probs(
            self,
            probs: np.ndarray,
            pid: str,
    ) -> np.ndarray:
        """ random probs forced by pex-advanced - keeps pex for n steps, then samples new value """

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
            n_moves = len(self.table_moves)
            move_ix = self._rng.choice(n_moves)
            probs = np.zeros(shape=n_moves)
            probs[move_ix] = 1
        return probs

    def _sample_move(
            self,
            probs: np.ndarray,
            allowed_moves :List[bool],
            pid: str,
    ) -> int:
        """ adds sampling with PEX """
        if self.enable_pex and self.trainable:
            probs = self.__pex_probs(probs, pid)
        return super()._sample_move(probs, allowed_moves, pid)

    def _publish_FWD_stats(self, step):
        """ adds pex to TB """
        super()._publish_FWD_stats(step)
        if self.enable_pex and self.publish_pex:
            self._tbwr.add(value=self._pex, tag='process.FWD/pex', step=step)

# ***************************************************************************************** NOT abstract implementations

class RanDMK(StaMaDMK):
    """ Random DMK
    implements baseline/equal (random decision) probs """

    def _compute_probs(self) -> None:
        """ calculates probabilities - baseline: sets equal for ALL new states of ALL players with allowed moves """
        n_moves = len(self.table_moves)
        for pid in self._states_new:
            if self._states_new[pid]:
                if self._states_new[pid][-1].allowed_moves:
                    self._states_new[pid][-1].probs = np.asarray([1 / n_moves] * n_moves) # equal probs

    def save(self): pass


class NeurDMK(ExaDMK):
    """ Neural DMK
    with NN (MOTorch) as a deciding model

    DMK_MOTorch is a sub-object - it is initialized inside a sub-Process
    - in this project torch NN is always build inside a subprocess
    """

    def __init__(
            self,
            table_size: int,
            table_cash_start: int,
            table_moves: List,
            motorch_type: type(DMK_MOTorch)=    DMK_MOTorch, # abstract passed here..
            motorch_point: Optional[POINT]=     None,
            reward_share: Optional[int]=        5, # reward sharing (between states) policy, for None every state gets reward/len(moves), for int gets reward/N
            **kwargs):

        ExaDMK.__init__(
            self,
            table_size=     table_size,
            table_moves=    table_moves,
            **kwargs)

        self.table_cash_start = table_cash_start

        self.motorch_type = motorch_type

        #manage motorch_point -> set some motorch_point parameters with DMK values
        self.motorch_point = motorch_point or {}
        update_with = {
            'name':         self.name,
            'save_topdir':  self.save_topdir,
            'table_size':   self.table_size,
            'table_moves':  self.table_moves}
        for k in update_with:
            if k in self.motorch_point:
                self._logger.warning(f'key: {k} already present in motorch_point with value: {self.motorch_point[k]} <- overriding with: {update_with[k]}')
            self.motorch_point[k] = update_with[k]

        self._mdl: Optional[DMK_MOTorch] = None

        self.reward_share = reward_share
        self._time_upd_fin = None # update time save, for reports

    def _encode_states(
            self,
            player_id,
            player_stateL: List[STATE],
    ) -> List[GameState]:
        """ encodes selection of HH states data into a form accepted by NN input
        returns only selected states (used by NN) """

        es = super()._encode_states(player_id, player_stateL)
        es_sel = [] # selected states
        ser = f'\nstates encoding report for {player_id}\n'
        for s in es:

            val = s.state_orig_data
            ser += f'> {val}\n'

            # update table cash
            if val[0] == 'T$$':
                self._table_cash[player_id] = list(val[1])

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
                    self._pos[player_id][val[1][0]] = val[1][1]     # save POS for next MOVes
                else:
                    event_id = 1 + val[1][1]                        # 1 + mov_id (index in table_moves)
                    mv_cash = val[1][2]
                    pl_cash = list(val[1][4])

                # merge and normalize
                cash = [mv_cash, *pl_cash, *self._table_cash[player_id]]
                cash = [v / self.table_cash_start for v in cash]
                pl_stats = list(self._pstats_ex[player_id][val[1][0]].player_stats.values()) if self.build_villain_stats else [0.0 for _ in PLAYER_STATS_USED]

                nval = {
                    'cards':    [] + self._my_cards[player_id],     # List[7 x int] copy of my cards: 0-52
                    'event_id': event_id,                           # int: O-(1+len(table_moves)) <- POS + all MOVes
                    'cash':     cash,                               # List[8 x float] move cash, pl.cash, pl.cash_ch, pl.cash_cs, table.pot, table.cash_cs, table.cash_tc
                    'pl_id':    val[1][0],                          # int player ID: 0-table_size
                    'pl_pos':   self._pos[player_id][val[1][0]],    # int player pos: 0-table_size
                    'pl_stats': pl_stats,                           # List[float,..] 0.0-1.0
                }

                es_sel.append(GameState(nval))
                ser += f'---> {nval}\n'

            if val[0] == 'PRS' and val[1][0] == 0: # my result
                if self._states_dec[player_id]:
                    self._states_dec[player_id][-1].reward = val[1][1] # we can append reward to last state here

                # reset
                self._my_cards[player_id] = []
                self._table_cash[player_id] = [0,0,0,0]

        self._logger.debug(ser)

        return es_sel

    def _compute_probs(self) -> None:
        """ compute probabilities for at least some states (compute until get to allowed_moves) """

        n_rows = 0
        got_probs_for_allowed = False
        while not got_probs_for_allowed:

            player_ids: List[str] = []
            game_statesL: List[List[GameState]] = []
            for pid in self._states_new:
                if self._states_new[pid]:
                    for s in self._states_new[pid]:
                        if s.probs is None:
                            player_ids.append(pid)
                            game_statesL.append([s])
                            break

            # it is possible, that all probs are done (for example allowed moves appeared after probs calculated)
            if not player_ids:
                break

            else:
                batch = self._mdl.build_batch(
                    player_ids=     player_ids,
                    game_statesL=   game_statesL,
                    for_training=   False)
                probs_array = self._mdl.run_policy(player_ids=player_ids, batch=batch)
                probs_array = np.squeeze(probs_array, axis=-2)
                n_rows += 1

                # distribute probs
                for pid, probs in zip(player_ids, probs_array):
                    for s in self._states_new[pid]:
                        if s.probs is None:
                            s.probs = probs
                            if s.allowed_moves:
                                got_probs_for_allowed = True
                            break

                self._processFWD_stats_data['5.row_widthF'].append(len(player_ids) / len(self._player_ids))

        self._processFWD_stats_data['4.n_rows'].append(n_rows)

    def _training_core(self):
        """ data preparation and NN update """

        tr = TimeRep()
        time_upd_start = time.time()

        ### prepare reward shared (reward_sh) for self._states_dec

        player_ids = [] + self._player_ids

        # for every player:
        # 1. move rewards down to (last) move (state with move)
        # 2. build rewards - a dict {pid: [[99,95,92][85,81,77,74]..]}
        #    with reversed indexes of moves (among all player states)
        #    first move in the sublist is always rewarded
        rewards = {}
        for pid in player_ids:
            rewards[pid] = []
            reward = None
            passed_first_reward = False # some last (first from the reversed) moves need to be skipped - those do not having rewards yet
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
        pids_not_rewarded = []
        for pid in player_ids:
            if not rewards[pid]:
                pids_not_rewarded.append(pid)
        if pids_not_rewarded:
            self._logger.debug(f'got not rewarded players: {pids_not_rewarded}')
            for pid in pids_not_rewarded:
                player_ids.remove(pid)
                rewards.pop(pid)

        # share (down) + normalize rewards
        for pid in player_ids:
            for move_ixL in rewards[pid]:

                # only when already not shared (..from previous update)
                rIX = move_ixL[0] # index of reward
                if self._states_dec[pid][rIX].reward_sh is None:

                    if self.reward_share is None:
                        rew_sh = self._states_dec[pid][rIX].reward / len(move_ixL)
                    else:
                        rew_sh = self._states_dec[pid][rIX].reward / self.reward_share

                    for mIX in move_ixL:
                        self._states_dec[pid][mIX].reward_sh = rew_sh / self.table_cash_start
                else:
                    break

        ### select players for update

        last_rewarded_move = [(pid, rewards[pid][0][0]) for pid in player_ids] # [(pid, index of state)]
        last_rewarded_move = sorted(last_rewarded_move, key=lambda x: x[1], reverse=True) # sort decreasing

        half_players = len(self._player_ids) // 2
        if len(last_rewarded_move) < half_players:
            half_players = len(last_rewarded_move)

        last_rewarded_move = last_rewarded_move[:half_players]          # trim
        n_states_upd = last_rewarded_move[-1][1] + 1                    # n states to use for update (+1 since moves are indexed from 0 - height of the batch)
        player_ids_upd = [e[0] for e in last_rewarded_move]                    # extract pid to update (width of the batch)

        # publish UPD process stats
        if self.publishUPD:

            # num of states
            n_sts = mam([len(self._states_dec[pid]) for pid in self._player_ids])
            self._tbwr.add(value=n_sts[0],     tag='process.UPD/a.n_sts_min', step=self.upd_step) # the shortest states list
            self._tbwr.add(value=n_states_upd, tag='process.UPD/b.n_sts_upd', step=self.upd_step) # == height of the batch
            self._tbwr.add(value=n_sts[2],     tag='process.UPD/c.n_sts_max', step=self.upd_step) # the longest states list
            val = (n_states_upd * len(player_ids_upd)) / self._n_states_dec
            self._tbwr.add(value=val,          tag='process.UPD/d.sts_updF',  step=self.upd_step) # factor of states taken for update

            # num of states with moves
            n_mov = mam([sum([len(ml) for ml in rewards[pid]]) for pid in player_ids_upd])
            self._tbwr.add(value=n_mov[0], tag='process.UPD/e.n_mov_min', step=self.upd_step)
            self._tbwr.add(value=n_mov[2], tag='process.UPD/f.n_mov_max', step=self.upd_step)

            # num of states with rewards
            n_rew = mam([len(rewards[pid]) for pid in player_ids_upd])
            self._tbwr.add(value=n_rew[0], tag='process.UPD/g.n_rew_min', step=self.upd_step)
            self._tbwr.add(value=n_rew[2], tag='process.UPD/h.n_rew_max', step=self.upd_step)

            self._tbwr.add(value=n_sts[1]/n_mov[1], tag='process.UPD/i.n_sts/mov', step=self.upd_step)
            self._tbwr.add(value=n_sts[1]/n_rew[1], tag='process.UPD/j.n_sts/rew', step=self.upd_step)
            self._tbwr.add(value=n_mov[1]/n_rew[1], tag='process.UPD/k.n_mov/rew', step=self.upd_step)

        tr.log('prepare_data')

        batch = self._mdl.build_batch(
            player_ids=     player_ids_upd,
            game_statesL=   [self._states_dec[pid][:n_states_upd] for pid in player_ids_upd],
            for_training=   True)
        tr.log('build_batch')

        self._mdl.update_policy(player_ids=player_ids_upd, batch=batch)
        tr.log('update_policy')

        if self.publishUPD:
            time_rep = {}
            tr_report = tr.get_report()
            for k in tr_report:
                time_rep[f'{k}_time'] = tr_report[k]
            ct = time.time()
            if self._time_upd_fin is not None:
                upd_time = ct-time_upd_start
                time_rep['upd_time_total'] = upd_time
                all_time = ct-self._time_upd_fin
                time_rep['upd_time_factor'] = upd_time/all_time
            self._time_upd_fin = ct
            for l,k in zip('abcdefghijklmnopqrstuvwxyz'[:len(time_rep)], time_rep):
                self._tbwr.add(value=time_rep[k], tag=f'backprop.time/{l}.{k}', step=self.upd_step)

        return n_states_upd, player_ids_upd

    def _flush_states_dec(self, ust_details) -> None:
        """ flush properly """

        # to remove all while not learning
        if ust_details is None:
            super()._flush_states_dec(ust_details)

        # leave only not used
        else:
            n_moves_upd, upd_pid = ust_details
            for pid in upd_pid:
                self._states_dec[pid] = self._states_dec[pid][n_moves_upd:]
            self._n_states_dec -= n_moves_upd * len(upd_pid)

    def _pre_process(self):
        """ adds DMK_MOTorch """

        super()._pre_process()

        self._mdl = self.motorch_type(
            player_ids= self._player_ids,
            logger=     self._logger, # since DMK_MOTorch is a core component of DMK it takes same logger
            tbwr=       self._tbwr,
            do_TB=      self.publishUPD,
            **self.motorch_point)

        # properties below are updated while encoding states
        self._my_cards =   {pa: []        for pa in self._player_ids}  # current cards of player
        self._table_cash = {pa: [0,0,0,0] for pa in self._player_ids}  # current (before player move) table cash (from T$$ state)
        self._pos =        {pa: {}        for pa in self._player_ids}  # current players positions {pl_id:pos}

    def _do_what_GM_says(self, message: QMessage):

        super()._do_what_GM_says(message)

        if message.type == 'reload_model':
            self._mdl.load_ckpt()
            dmk_message = QMessage(type='dmk_model_ckpt_reloaded', data=self.name)
            self._que_to_gm.put(dmk_message)

        if message.type == 'reset_fwd_state':
            self._mdl.reset_fwd_state()
            dmk_message = QMessage(type='dmk_model_fwd_state_reset', data=self.name)
            self._que_to_gm.put(dmk_message)

    @classmethod
    def save_policy_backup(cls, dmk_name:str, save_topdir:Optional[str]=None):
        if not save_topdir: save_topdir=cls.SAVE_TOPDIR
        DMK_MOTorch.save_checkpoint_backup(model_name=dmk_name, save_topdir=save_topdir)

    @classmethod
    def restore_policy_backup(cls, dmk_name:str, save_topdir:Optional[str]=None):
        if not save_topdir: save_topdir = cls.SAVE_TOPDIR
        DMK_MOTorch.restore_checkpoint_backup(model_name=dmk_name, save_topdir=save_topdir)

    def save(self):

        err_msg = ''
        if not self.trainable:
            err_msg = 'You asked not trainable NeurDMK to .save(), cannot continue!'
        if not self._mdl:
            err_msg = 'NeurDMK self._mdl is None, you are probably calling .save() from out of the process!'
        if err_msg:
            self._logger.error(err_msg)
            raise PyPoksException(err_msg)

        self._mdl.save()

    @property
    def device(self):
        """ device is the property taken form the model or MOTorch POINT """
        if self._mdl: device = self._mdl.device
        else:         device = self.motorch_point.get('device', False)
        return device


class FolDMK(ParaSave, NeurDMK):
    """ Foldered DMK = ParaSave (POINT) + NeurDMK (MOTorch) -->
    + saving
    + serialization
    + genetic crossing (GX)
    + version management """

    # TODO: do we need those here?
    FolDMK_DEFAULTS = {
        'family':   'a',
        'age':       0,
    }

    # SAVE_TOPDIR set here again since FolDMK extends two classes and it makes some mess in class attrs
    SAVE_TOPDIR = DMK_MODELS_FD
    SAVE_FN_PFX = 'dmk_dna'

    def __init__(
            self,
            name: str,
            motorch_point: Optional[POINT]= None,
            save_topdir: Optional[str]=     None,
            logger=                         None,
            loglevel=                       20,
            **kwargs):

        DMK_MOTorch.SAVE_TOPDIR = self.SAVE_TOPDIR
        if not save_topdir: save_topdir = self.SAVE_TOPDIR

        if not logger:
            logger = get_pylogger(
                name=       name,
                add_stamp=  False,
                folder=     f'{save_topdir}/{name}',
                level=      loglevel)
        self._logger = logger

        ParaSave.__init__(
            self,
            name=           name,
            save_topdir=    save_topdir,
            logger=         get_child(self._logger),
            **kwargs)
        point = self.get_point()
        self._logger.debug(f'point of ParaSave: {point}')

        for k in self.FolDMK_DEFAULTS:
            if k not in point:
                point[k] = self.FolDMK_DEFAULTS[k]

        # force family to override ParaSave None
        if point['family'] is None:
            point['family'] = self.FolDMK_DEFAULTS['family']

        point.update(kwargs)

        # any motorch_point SAVED by FolDMK should be overridden with given here (FolDMK.__init__)
        # MOTorch manages its POINT by itself, we want to give him POINT only if something new was given
        if motorch_point is None:
            motorch_point = {}
        if 'family' not in motorch_point:
            motorch_point['family'] = point['family']
        point['motorch_point'] = motorch_point

        self.update(point)
        point = self.get_point()
        self._logger.debug(f'full point of FolDMK: {point}')

        ### prepare NeurDMK point <- remove ParaSave-specific parameters

        point_neurdmk = {}
        point_neurdmk.update(point)
        for k in ['save_fn_pfx','psdd','gxable','age','parents','assert_saved','lock_managed_params']:
            if k in point_neurdmk:
                point_neurdmk.pop(k)
        point_neurdmk['logger'] = self._logger

        self._logger.debug(f'point of NeurDMK: {point_neurdmk}')
        NeurDMK.__init__(self, **point_neurdmk)

    def save(self):
        """ this method may be called only from the inside of the subprocess """
        ParaSave.save_point(self)
        NeurDMK.save(self)

    def _pre_process(self):
        super()._pre_process()
        if self.publishFWD and self.fwd_stats_step == 0:
            kv_str = '\n'.join([f'|{k}|{v}|' for k, v in self.get_point().items()])
            self._tbwr.add_text(text=f"|param|value|\n|-|-|\n{kv_str}", tag="POINT")

    @staticmethod
    @proc_wait
    def copy_saved(
            name_src: str,
            name_trg: str,
            save_topdir_src: Optional[str]= None,
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: Optional[str]=     None,
            logger=                         None,
            loglevel=                       30,
    ):
        """ copies saved FolDMK """

        if not save_topdir_src: save_topdir_src = FolDMK.SAVE_TOPDIR
        if not save_topdir_trg: save_topdir_trg = save_topdir_src

        FolDMK.copy_saved_point(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            save_fn_pfx=        save_fn_pfx,
            logger=             logger,
            loglevel=           loglevel)

        DMK_MOTorch.copy_saved(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            logger=             logger,
            loglevel=           loglevel)

    @classmethod
    @proc_wait
    def gx_saved(
            cls,
            name_parentA: str,
            name_parentB: Optional[str],                # if not given makes GX only with main parent
            name_child: str,
            save_topdir_parentA: Optional[str]= None,
            save_topdir_parentB: Optional[str]= None,
            save_topdir_child: Optional[str]=   None,
            save_fn_pfx: Optional[str]=         None,
            do_gx_ckpt=                         True,
            ratio=                              0.5,
            noise=                              0.03,
            logger=                             None,
            loglevel=                           20,
    ) -> None:
        """ GX on saved FolDMK """

        if not save_topdir_parentA: save_topdir_parentA = cls.SAVE_TOPDIR
        if not save_topdir_parentB: save_topdir_parentB = save_topdir_parentA
        if not save_topdir_child: save_topdir_child = save_topdir_parentA
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        cls.gx_saved_point(
            name_parentA=           name_parentA,
            name_parentB=           name_parentB,
            name_child=             name_child,
            save_topdir_parentA=    save_topdir_parentA,
            save_topdir_parentB=    save_topdir_parentB,
            save_topdir_child=      save_topdir_child,
            save_fn_pfx=            save_fn_pfx,
            logger=                 logger,
            loglevel=               loglevel)

        # oversave to have age 0
        cls.oversave_point(
            name=           name_child,
            save_topdir=    save_topdir_child,
            save_fn_pfx=    save_fn_pfx,
            age=            0)
        
        DMK_MOTorch.gx_saved(
            name_parentA=           name_parentA,
            name_parentB=           name_parentB,
            name_child=             name_child,
            save_topdir_parentA=    save_topdir_parentA,
            save_topdir_parentB=    save_topdir_parentB,
            save_topdir_child=      save_topdir_child,
            save_fn_pfx=            DMK_MOTorch.SAVE_FN_PFX,
            do_gx_ckpt=             do_gx_ckpt,
            ratio=                  ratio,
            noise=                  noise,
            logger=                 logger,
            loglevel=               loglevel)

    @classmethod
    @proc_wait
    def build_from_point(
            cls,
            dmk_point:POINT,
            logger=     None,
            loglevel=   20,
    ) -> None:
        """ builds DMK and its MOTorch, saves
        (DMK_MOTorch is built within a process, after QueDMK.start() in NeurDMK._pre_process()) """

        if not logger:
            logger = get_pylogger(name='build_DMK_from_point', level=loglevel)

        dmk = cls(**dmk_point, logger=logger)

        que_to_gm = Que()
        dmk.que_to_gm = que_to_gm

        dmk.start()
        msg = que_to_gm.get()
        logger.debug(f'GM receives message from DMK, type: {msg.type}, data: {msg.data}')

        for msg_type in ['save_dmk', 'stop_dmk_process']:
            msg = QMessage(type=msg_type, data=None)
            dmk.que_from_gm.put(msg)
            msg = que_to_gm.get()
            logger.debug(f'GM receives message from DMK, type: {msg.type}, data: {msg.data}')

    @property
    def logger(self):
        return self._logger


class HumanDMK(StaMaDMK):
    """ Human Driven DMK
    allows a human to make a decision with TK GUI """

    def __init__(self, **kwargs):

        StaMaDMK.__init__(self, trainable=False, n_players=1, **kwargs)

        # needs to be updated by HGM
        self.gui_queI = None
        self.gui_queO = None

    def _encode_states(
            self,
            player_id,
            player_stateL: List[STATE],
    ) -> List[GameState]:
        """ additionally sends incoming states to TK """
        for state in player_stateL:
            message = QMessage(type='state', data=state)
            self.gui_queI.put(message)
        return super()._encode_states(player_id, player_stateL)

    def _compute_probs(self) -> None:
        """ sends allowed_moves to TK -> human makes decision -> decision is cast to OH probs """

        probs = np.zeros(len(self.table_moves))
        for pid in self._states_new:
            if self._states_new[pid]:
                last_state = self._states_new[pid][-1]
                if last_state.allowed_moves:

                    # send data to TK
                    message = QMessage(
                        type=   'allowed_moves',
                        data=   {
                            'allowed_moves':    last_state.allowed_moves,
                            'moves_cash':       last_state.moves_cash})
                    self.gui_queI.put(message)

                    # get decision from TK
                    tk_message = self.gui_queO.get()
                    probs[tk_message.data] = 1

                    last_state.probs = probs