"""

 2019 (c) piteren

 DMK (decision maker) makes decisions for poker players
    uses MSOD (many states one decision) concept

     DMK assumes that:
        - DMK receives data from table/player
            - player sends list of states (by calling: DMK.take_states, ...does so before a move or after a hand)
            - from time to time player sends list of possible_moves (by calling: DMK.take_possible_moves)
                > possible_moves are added (by DMK) to the last state of player
                    > now player/table has to wait for a while (for its decision)
                    > no new states from the table (for that player) will come until decision will be made
                    > table with player (that waits for decision) is locked (no other player will send anything then...)

        - DMK makes decisions - moves for players:
            > move should be selected:
                    - regarding received states (saved in _new_states - policy input) and ANY history saved by DMK
                    - using DMK learnable? policy
                    - from possible_moves
            > DMK decides WHEN to make decisions (DMK.make_decisions, but DMK does not define when to do it...)
            > makes decisions for (one-some-all) players with possible_moves @_new_states
            > states used to make decisions are moved(appended) to _done_states

        - DMK runs update to learn policy of making decisions
            > update is based on _done_states (taken moves & received rewards)

     DMK leaves for implementation:
        _enc_states                         - states encoding/processing
        _decisions_from_new_states_subtask  - how to use _new_states to make decisions
        _update_subtask                     - how to learn using _done_states
        _flush_done_states                  - which states remove from _done_states after update

 QDMK is a DMK implemented as a process with ques
    - communicates with table and players using ques
    - assumes that (DMK) decisions are made by probabilistic model (soft probabilities of decisions)
        + implements _dec_from_new_states as a 2 step process:
            > calculate move probabilities (_calc_probs) - assumes that decisions will be made by probabilistic model
            > selects move (argmax) using probs (__sample_move)
    - optionally communicates with StatsManager(SM) (SM is not obligatory for QDMK)

    QDMK leaves for implementation:
        _enc_states
        _calc_probs                         - how to calculate probabilities for states
        _do_what_gm_says                    - how to run GM commands
        _update_subtask
        _flush_done_states



 ExDMK implements exploration while sampling move (forces random model instead of probabilistic one)
    + implements basic exploration reduction (policy)

 NeurDMK is a ProDMK implemented with Neural Network to make decisions
     + encodes states for NN
     + makes decisions with NN
     + updates NN (learns)
        - from time to time updates itself(learns NN) using information from _done_states

"""

from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
import numpy as np
import random
import tensorflow as tf
from typing import List
from queue import Empty

from ptools.neuralmess.nemodel import NEModel
from ptools.neuralmess.base_elements import ZeroesProcessor

from podecide.stats_manager import StatsMNG
from pologic.poenvy import N_TABLE_PLAYERS
from pologic.podeck import PDeck
from pologic.potable import POS_NMS, TBL_MOV

from gui.tkproc import TkProc



# State type, here DMK saves table events
class State:

    def __init__(
            self,
            value):

        self.value = value                                  # value of state (any object)
        self.possible_moves :List[bool] or None=    None    # list of player possible moves
        self.moves_cash: List[int] or None =        None    # list of player cash amount for possible moves
        self.probs :List[float] or None=            None    # probabilities of moves
        self.move :int or None=                     None    # move (performed/selected by DMK)
        self.reward :float or None=                 None    # reward (after hand finished)
        self.move_rew :float or None=               None    # move shared reward

# (abstract) DMK defines basic interface of DMK to act on PTable with PPlayer (DMK is called by table/player)
class DMK(ABC):

    def __init__(
            self,
            name :str,                          # name should be unique (@table)
            n_players :int,                     # number of managed players
            n_moves :int=       len(TBL_MOV),   # number of (all) moves supported by DM, has to match table
            upd_trigger :int=   1000,           # how much _done_states to accumulate to launch update procedure
            verb=               0):

        self.name = name
        self.verb = verb
        self.n_players = n_players
        self.p_addrL = [f'{self.name}_{ix}' for ix in range(self.n_players)] # DMK builds addresses for players
        self.n_moves = n_moves

        self.upd_trigger = upd_trigger
        self.upd_step = 0               # updates counter (clock)

        if self.verb>0: print(f'\n *** DMK *** initialized, name {self.name}, players {self.n_players}')

        # variables below store moves/decisions data
        self._new_states =      {} # dict of lists of new states {state: }, should not have empty keys (player indexes)
        self._n_new = 0         # cache number of states @_new_states
        self._done_states =     {pa: [] for pa in self.p_addrL}  # dict of processed (with decision) states
        self._n_done = 0        # cache number of done @_done_states

    # encodes table states into DMK form (appropriate to make decisions)
    @abstractmethod
    def _enc_states(
            self,
            p_addr,
            player_stateL: List[list]) -> List[State]:
        return [State(value) for value in player_stateL]  # wraps into list of States

    # takes player states, encodes and saves in self._new_states, updates cache
    def take_states(
            self,
            p_addr,
            player_stateL: List[list]) -> None:
        encoded_states = self._enc_states(p_addr, player_stateL)
        if encoded_states:
            if p_addr not in self._new_states: self._new_states[p_addr] = []
            self._new_states[p_addr] += encoded_states
            self._n_new += len(encoded_states)

    # takes player possible_moves, saves, updates cache
    def take_possible_moves(
            self,
            p_addr,
            possible_moves :List[bool],
            moves_cash :List[int]):
        assert p_addr in self._new_states # TODO (dev safety check): player should have new states while getting possible moves
        # add to last new state
        last_state = self._new_states[p_addr][-1]
        last_state.possible_moves = possible_moves
        last_state.moves_cash = moves_cash

    # using data from _new_states prepares list of decisions in form: [(p_addr,move)...]
    @abstractmethod
    def _decisions_from_new_states_subtask(self) -> List[tuple]: pass

    # move states (from _new to _done) having list of decisions, updates caches
    def __move_states(self, decL :List[tuple]) -> None:
        for dec in decL:
            p_addr, move = dec
            states = self._new_states.pop(p_addr)
            states[-1].move = move # write move into state
            self._done_states[p_addr] += states
            self._n_new -= len(states)
            self._n_done += len(states)

    # makes decisions (for one-some-all) players with possible moves
    def make_decisions_task(self) -> List[tuple]:
        decL = self._decisions_from_new_states_subtask()
        assert decL

        self.__move_states(decL) # move states

        if self._n_done > self.upd_trigger:
            self._run_update_task() # self update (learn)

        return decL

    # learn policy from _done_states (core of update task)
    @abstractmethod
    def _learning_subtask(self): return 'no_update_done'

    # flush (all) & reset,
    @abstractmethod
    def _update_done_states(self, ust_details) -> None:
        self._done_states = {pa: [] for pa in self._done_states}
        self._n_done = 0

    # runs update of DMK based on saved _done_states, called in make_decisions (when trigger fires)
    def _run_update_task(self) -> None:
        ust_details = self._learning_subtask()
        self._update_done_states(ust_details)
        self.upd_step += 1

# (abstract) Qued DMK defines basic interface to act on QPTable with QPPlayer (QDMK is called by qued table/player)
class QDMK(Process, DMK, ABC):

    def __init__(
            self,
            gm_que :Queue,              # GamesManager que, here DMK puts data for GM, data is always put in form of tuple (name, type, data) # TODO: implement tuple everywhere...
            stats_iv=           1000,
            acc_won_iv=         (100000,200000),
            **kwargs):

        Process.__init__(self, target=self._dmk_proc)
        DMK.__init__(self, **kwargs)

        self.gm_que = gm_que
        self.in_que = Queue() # here DMK receives data only! from GM

        # every QDMK creates part of ques network
        self.dmk_in_que = Queue()
        self.pl_in_queD = {pa: Queue() for pa in self.p_addrL}

        self.sm = None
        self.start_hand = 0 # ProDMK has no possibility to set other than 0 (since no save), but neural will have...
        self.stats_iv = stats_iv
        self.acc_won_iv = acc_won_iv
        self.process_stats = {  # any stats of the process, published & flushed during update
            '1.len_pdL':    [], # length of player data list (number of updates from tables taken in one loop)
            '2.n_waiting':  [], # number of waiting players while making decisions
            '3.n_dec':      []} # number decisions made

    # adds stats management with SM while encoding taken states
    @abstractmethod
    def _enc_states(
            self,
            pID,
            player_stateL: List[list]) -> List[State]:
        if self.sm: self.sm.take_states(pID, player_stateL)
        return super()._enc_states(pID,player_stateL)

    # calculates probabilities for at least some _new_states with possible_moves (to be done with probabilistic model...)
    @abstractmethod
    def _calc_probs(self) -> None: pass

    # selects single move form possible_moves using given probabilities (probs argmax)
    def _sample_move(
            self,
            probs :List[float],
            possible_moves :List[bool]) -> int:
        prob_mask = np.asarray(possible_moves).astype(int)  # cast bool to int
        probs = probs * prob_mask                           # mask probs
        if np.sum(probs) == 0: probs = prob_mask            # take mask if no intersection
        moves_arr = np.arange(self.n_moves)                 # array with moves indexes
        return moves_arr[np.argmax(probs)]                  # take max from probs

    # returns list of decisions
    def _decisions_from_new_states_subtask(self) -> List[tuple]:

        if self.verb>1:
            nd = {}
            for p_addr in self._new_states:
                l = len(self._new_states[p_addr])
                if l not in nd: nd[l] = 0
                nd[l] += 1
                if l > 10:
                    for s in self._new_states[p_addr]: print(s)
            print(' >> (@_madec) _new_states histogram:')
            for k in sorted(list(nd.keys())): print(' >> %d:%d'%(k,nd[k]))

        self._calc_probs()

        # make decisions for playes with ready data (possible_moves and probs)
        decL = []
        for p_addr in self._new_states:
            if self._new_states[p_addr][-1].possible_moves is not None and self._new_states[p_addr][-1].probs is not None:
                move = self._sample_move(
                    probs=          self._new_states[p_addr][-1].probs,
                    possible_moves =self._new_states[p_addr][-1].possible_moves)
                decL.append((p_addr, move))

        return decL

    # prepare process stats, publish and reset
    def __publish_proces_stats(self):
        if self.sm:
            for k in self.process_stats:
                val = sum(self.process_stats[k])/len(self.process_stats[k]) if len(self.process_stats[k]) else 0
                val_summ = tf.Summary(value=[tf.Summary.Value(tag=f'reports.PCS/{k}', simple_value=val)])
                self.sm.summ_writer.add_summary(val_summ, self.upd_step)
        for k in self.process_stats: self.process_stats[k] = [] # reset

    # update with process stats
    def _run_update_task(self) -> None:
        self.__publish_proces_stats()
        super()._run_update_task() # to flush _done_states and increase upd_step

    # method called BEFORE process loop, builds objects that HAVE to be build in process memory scope
    def _pre_process(self) -> None:

        # TB stats saver (has to be build here, inside process method, since summ_writer...)
        self.sm = StatsMNG(
            name=       self.name,
            p_addrL=    self.p_addrL,
            start_hand= self.start_hand,
            stats_iv=   self.stats_iv,
            acc_won_iv= self.acc_won_iv,
            verb=       self.verb) if self.stats_iv else None

    # runs GamesManager commands
    @abstractmethod
    def _do_what_GM_says(self, gm_data) -> None: pass

    # process method (loop, target of process)
    def _dmk_proc(self):

        self._pre_process()
        self.gm_que.put(f'{self.name} (DMK process) started')
        self.in_que.get() # waits for GO!

        n_waiting = 0 # num players ( ~> tables) waiting for decision
        while True:

            # 'flush' the que of data from players
            pdL = []
            while True:
                try:            player_data = self.dmk_in_que.get_nowait()
                except Empty:   break
                if player_data: pdL.append(player_data)
            self.process_stats['1.len_pdL'].append(len(pdL))

            for player_data in pdL:
                p_addr = player_data['id']

                if 'state_changes' in player_data:
                    self.take_states(p_addr, player_data['state_changes'])

                if 'possible_moves' in player_data:
                    self.take_possible_moves(p_addr, player_data['possible_moves'], player_data['moves_cash'])
                    n_waiting += 1

            # now, if got any waiting >> make decisions
            if n_waiting:
                self.process_stats['2.n_waiting'].append(n_waiting)
                decL = self.make_decisions_task()
                self.process_stats['3.n_dec'].append(len(decL))
                n_waiting -= len(decL)
                for d in decL:
                    p_addr, move = d
                    self.pl_in_queD[p_addr].put(move)

            try: # eventually get data from GM
                gm_data = self.in_que.get_nowait()
                if gm_data:
                    # stop DMK process
                    if gm_data == 'stop':
                        self.gm_que.put((self.name, 'finished', None))
                        break
                    else: self._do_what_GM_says(gm_data)
            except Empty: pass

# Random Qued DMK (implements baseline/equal probs >> choice is fully random then)
class RnDMK(QDMK):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # nothing new, just copy
    def _enc_states(
            self,
            pID,
            player_stateL: List[list]) -> List[State]:
        return super()._enc_states(pID, player_stateL)

    # calculates probabilities - baseline: sets equal for all new states of players with possible moves
    def _calc_probs(self) -> None:
        baseline_probs = [1/self.n_moves] * self.n_moves # equal probs
        for p_addr in self._new_states:
            if self._new_states[p_addr][-1].possible_moves:
                self._new_states[p_addr][-1].probs = baseline_probs

    # nothing new, just implementation
    def _learning_subtask(self): return super()._run_update_task()

    # nothing new, just implementation
    def _update_done_states(self, ust_details) -> None: super()._update_done_states(ust_details)

    # nothing new, just implementation
    def _do_what_GM_says(self, gm_data) -> None: super()._do_what_GM_says(gm_data)

# Human driven QDMK
class HDMK(QDMK):

    def __init__(
            self,
            tk_proc :TkProc,
            **kwargs):
        super().__init__(n_players=1, **kwargs)
        self.family = None # TODO <<

        self.tk_IQ = tk_proc.tk_que
        self.tk_OQ = tk_proc.out_que

    # send incoming states to tk
    def _enc_states(
            self,
            pID,
            player_stateL: List[list]) -> List[State]:
        for state in player_stateL: self.tk_IQ.put(state)
        return super()._enc_states(pID, player_stateL)

    # waits for a human decision
    def _calc_probs(self) -> None:
        probs = [0] * self.n_moves
        for p_addr in self._new_states:
            if self._new_states[p_addr][-1].possible_moves:
                pmd = {
                    'possible_moves':   self._new_states[p_addr][-1].possible_moves,
                    'moves_cash':       self._new_states[p_addr][-1].moves_cash}
                self.tk_IQ.put(pmd)
                val = self.tk_OQ.get()
                probs[val] = 1
                self._new_states[p_addr][-1].probs = probs

    # nothing new, just implementation
    def _learning_subtask(self): return super()._run_update_task()

    # nothing new, just implementation
    def _update_done_states(self, ust_details) -> None: super()._update_done_states(ust_details)

    # nothing new, just implementation
    def _do_what_GM_says(self, gm_data) -> None: super()._do_what_GM_says(gm_data)

# (abstract) Exploring Qued DMK
class ExDMK(QDMK, ABC):

    def __init__(
            self,
            pmex_init :float=   0.2,    # <0;1> possible_moves exploration (probability of random sampling from possible_moves space)
            pmex_trg=           0.02,   # pmex target value
            ex_reduce=          0.95,   # exploration reduction factor (with each update)
            **kwargs):

        QDMK.__init__(self, **kwargs)

        self.pmex = pmex_init
        self.pmex_trg = pmex_trg
        self.ex_reduce = ex_reduce
        # self.suex - by now not used, <0;1> self uncertainty exploration (probability of sampling from probability (rather than taking max))

    # random probs forced by self.pmex
    def __pmex_probs(
            self,
            probs :List[float]):
        if self.pmex > 0 and random.random() < self.pmex: probs = np.random.rand(self.n_moves)
        return probs

    # using pmex and suex samples single move for given probabilities and possibilities
    def _sample_move(
            self,
            probs :List[float],
            possible_moves :List[bool]) -> int:
        probs = self.__pmex_probs(probs)
        return super()._sample_move(probs, possible_moves)

    # publish pmex to TB
    def __publish_pmex(self):
        self.sm.summ_writer.add_summary(
            tf.Summary(
                value=[tf.Summary.Value(
                    tag=            'nn/4.pmex',
                    simple_value=   self.pmex)]),
            self.upd_step)

    # applies policy for pmex value
    def _apply_pmex_policy(self):
        # reduce pmex/suex after update
        if self.pmex > self.pmex_trg:
            self.pmex *= self.ex_reduce
            if self.pmex < self.pmex_trg: self.pmex = self.pmex_trg

    # add exploration reduction
    def _run_update_task(self) -> None:
        self.__publish_pmex()
        super()._run_update_task()
        self._apply_pmex_policy()

# Neural DMK
class NeurDMK(ExDMK):

    def __init__(
            self,
            fwd_func,               # neural graph FWD func
            mdict :dict=    None,   # model dict
            family=         'A',    # family (type) saved for GAX purposes etc.
            device=         None,   # cpu/gpu (check dev_manager @ptools)
            trainable=      True,
            upd_BS=         50000,  # estimated target batch size of update
            **kwargs):

        upd_trigger = 2*upd_BS # twice size since updating half_rectangle in trapeze (check selection state policy @UPD)
        super().__init__(upd_trigger=upd_trigger, **kwargs)
        assert self.stats_iv > 0 # Stats Manager is obligatory for NeurDMK # TODO

        self.fwd_func = fwd_func
        self.mdict = mdict
        if self.mdict is None: self.mdict = {}
        self.mdict['name'] = self.name

        self.family = family
        self.device = device

        self.trainable = trainable

    # prepares state into form of nn input
    #  - encodes only selection of states: [POS,PLH,TCD,MOV,PRS] ...does not use: HST,TST,PSB,PBB,T$$
    #  - each event has values (ids):
    #       0 : pad (...for cards factually)
    #       1,2,3 : my positions SB,BB,BTN
    #       4,5,6,7, 8,9,10,11 : moves of two opponents(1,2) * 4 moves(C/F,CLL,BR5,BR8)
    # TODO: make compatible with pologic.poenvy constants
    def _enc_states(
            self,
            p_addr,
            player_stateL: list):

        es = super()._enc_states(p_addr, player_stateL)
        news = [] # newly encoded states
        for s in es:
            val = s.value
            nval = None

            if val[0] == 'POS' and val[1][0] == 0: # my position
                nval = {
                    'cards':    None,
                    'event':    1 + self.pos_nms_r[val[1][1]]}

            if val[0] == 'PLH' and val[1][0] == 0: # my hand
                self.my_cards[p_addr] = [PDeck.cti(c) for c in val[1][1:]]
                nval = {
                    'cards':    [] + self.my_cards[p_addr], # copy cards
                    'event':    0}

            if val[0] == 'TCD': # my hand update
                self.my_cards[p_addr] += [PDeck.cti(c) for c in val[1]]
                nval = {
                    'cards':    [] + self.my_cards[p_addr], # copy cards
                    'event':    0}

            if val[0] == 'MOV' and val[1][0] != 0: # moves, all but mine
                nval = {
                    'cards':    None,
                    'event':    4 + self.tbl_mov_r[val[1][1]] + 4*(val[1][0]-1)}  # hardcoded for 2 opponents

            if val[0] == 'PRS' and val[1][0] == 0: # my result
                reward = val[1][1]
                if self._done_states[p_addr]: self._done_states[p_addr][-1].reward = reward # we can append reward to last state in _done_states (reward is for moves, moves are only there)
                self.my_cards[p_addr] = [] # reset my cards

            if nval: news.append(State(nval))

        if self.verb>1:
            print(' >> (@_enc_states):')
            print(' >> states to encode:')
            for s in es: print(' > %s'%s.value)
            print(' >> encoded states:')
            for s in news: print(' > %s'%s.value)

        return news

    # calculate probs for a row
    def __calc_probs_vr(
            self,
            vals_row :List[tuple]):

        probs_row = []

        # build batches
        cards_batch = []
        event_batch = []
        switch_batch = []
        state_batch = []
        p_addrL = []
        for vr in vals_row:
            p_addr, val = vr
            p_addrL.append(p_addr) # save list of p_addr
            switch = 1

            cards = val['cards']
            if not cards:
                cards = []
                switch = 0
            cards += [52]*(7-len(cards)) # pads cards

            event = val['event']

            # append samples wrapped in (seq axis)
            cards_batch.append([cards])
            event_batch.append([event])
            switch_batch.append([[switch]])

            state_batch.append(self.last_fwd_state[p_addr])

        feed = {
            self.mdl['cards_PH']:   np.asarray(cards_batch),
            self.mdl['train_PH']:   False,
            self.mdl['event_PH']:   np.asarray(event_batch),
            self.mdl['switch_PH']:  np.asarray(switch_batch),
            self.mdl['state_PH']:   np.asarray(state_batch)}

        fetches = [self.mdl['probs'], self.mdl['fin_state']]
        probs, fwd_states = self.mdl.session.run(fetches, feed_dict=feed)
        probs = np.squeeze(probs, axis=1) # remove sequence axis (1)

        for ix in range(fwd_states.shape[0]):
            p_addr = p_addrL[ix]
            probs_row.append((p_addr, probs[ix]))
            self.last_fwd_state[p_addr] = fwd_states[ix] # save fwd states

        return probs_row

    # add probabilities for at least some states with possible_moves (called with make_decisions)
    def _calc_probs(self) -> None:

        got_probs_for_possible = False
        while not got_probs_for_possible:

            vals_row = []
            for p_addr in self._new_states:
                for s in self._new_states[p_addr]:
                    if s.probs is None:
                        vals_row.append((p_addr,s.value))
                        break
            if self.verb>1: print(' > (@_calc_probs) got row of %d'%len(vals_row))

            if not vals_row: break # it is possible, that all probs are done (e.g. possible moves appeared after probs calculated)
            else:
                probs_row = self.__calc_probs_vr(vals_row)
                for pr in probs_row:
                    p_addr, probs = pr
                    for s in self._new_states[p_addr]:
                        if s.probs is None:
                            s.probs = probs
                            if s.possible_moves: got_probs_for_possible = True
                            break
                if self.verb>1 and not got_probs_for_possible: print(' > (@_calc_probs) another loop...')

    # min, avg, max ..of num list
    @staticmethod
    def __mam(valL: list):
        return [min(valL), sum(valL) / len(valL), max(valL)]

    # adds summary with upd_step
    def __add_upd_summ(self, summ: tf.compat.v1.Summary):
        self.sm.summ_writer.add_summary(summ, self.upd_step)

        # learn/update from _done_states
    # NN update
    def _learning_subtask(self):

        if self.trainable:

            p_addrL = sorted(list(self._done_states.keys()))

            # move rewards down to moves (and build rewards dict)
            rewards = {} # {p_addr: [[99,95,92][85,81,77,74]...]} # indexes of moves, first always rewarded
            for p_addr in p_addrL:
                rewards[p_addr] = []
                reward = None
                passed_first_reward = False # we need to skip last(top) moves that do not have rewards yet
                mL = [] # list of moveIX
                for ix in reversed(range(len(self._done_states[p_addr]))):

                    st = self._done_states[p_addr][ix]

                    if st.reward is not None:
                        passed_first_reward = True
                        if reward is not None:  reward += st.reward # caught earlier reward without a move, add it here
                        else:                   reward = st.reward
                        st.reward = None

                    if st.move is not None and passed_first_reward: # got move here and it will share some reward
                        if reward is not None: # put that reward here
                            st.reward = reward
                            reward = None
                            # got previous list of mL
                            if mL:
                                rewards[p_addr].append(mL)
                                mL = []
                        mL.append(ix) # always add cause passed first reward

                if mL: rewards[p_addr].append(mL) # finally add last

            # remove not rewarded players (rare, but possible)
            nrp_addrL = []
            for p_addr in p_addrL:
                if not rewards[p_addr]:
                    rewards.pop(p_addr)
                    nrp_addrL.append(p_addr)
            if nrp_addrL:
                print('@@@ WARNING: got not rewarded players!!!')
                for p_addr in nrp_addrL: p_addrL.remove(p_addr)

            # share rewards:
            for p_addr in p_addrL:
                for mL in rewards[p_addr]:
                    rIX = mL[0] # index of reward
                    # only when already not shared (...from previous update)
                    if self._done_states[p_addr][rIX].move_rew is None:
                        sh_rew = self._done_states[p_addr][rIX].reward / len(mL)
                        for mIX in mL:
                            self._done_states[p_addr][mIX].move_rew = sh_rew
                    else: break

            lrm = [rewards[p_addr][0][0] for p_addr in p_addrL]     # last rewarded move of player (index of state)
            lrmT = zip(p_addrL, lrm)
            lrmT = sorted(lrmT, key=lambda x: x[1], reverse=True)   # sort decreasing
            half_players = len(self._done_states) // 2
            if len(lrmT) < half_players: half_players = len(lrmT)
            upd_p = lrmT[:half_players]                             # longer half
            n_upd = upd_p[-1][1] + 1                                # n states to use for update
            upd_p = [e[0] for e in upd_p]                           # list of p_addr to update

            # num of states (@_done_states per player)
            n_sts = self.__mam([len(self._done_states[p_addr]) for p_addr in p_addrL]) #
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/0.n_states_min', simple_value=n_sts[0])]))
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/1.n_states_max', simple_value=n_sts[2])]))

            # num of states with moves
            n_mov = self.__mam([sum([len(ml) for ml in rewards[p_addr]]) for p_addr in p_addrL])
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/2.n_moves_min', simple_value=n_mov[0])]))
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/3.n_moves_max', simple_value=n_mov[2])]))

            # num of states with rewards
            n_rew = self.__mam([len(rewards[p_addr]) for p_addr in p_addrL])
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/4.n_rewards_min', simple_value=n_rew[0])]))
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/5.n_rewards_max', simple_value=n_rew[2])]))

            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/6.n_sts/mov', simple_value=n_sts[1] / n_mov[1])]))
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/7.n_sts/rew', simple_value=n_sts[1] / n_rew[1])]))
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/8.n_mov/rew', simple_value=n_mov[1] / n_rew[1])]))

            cards_batch = []
            switch_batch = []
            event_batch = []
            state_batch = []
            move_batch = []
            reward_batch = []
            for p_addr in upd_p:
                cards_seq = []
                switch_seq = []
                event_seq = []
                move_seq = []
                reward_seq = []
                for state in self._done_states[p_addr][:n_upd]:

                    val = state.value

                    switch = 1
                    cards = val['cards']
                    if not cards:
                        cards = []
                        switch = 0
                    cards += [52]*(7-len(cards))  # pads cards

                    cards_seq.append(cards)
                    switch_seq.append([switch])
                    event_seq.append(val['event'])
                    move_seq.append(state.move if state.move is not None else 0)
                    reward_seq.append(state.move_rew/500 if state.move_rew is not None else 0) #TODO: hardcoded 500

                cards_batch.append(cards_seq)
                switch_batch.append(switch_seq)
                event_batch.append(event_seq)
                move_batch.append(move_seq)
                reward_batch.append(reward_seq)
                state_batch.append(self.last_upd_state[p_addr])

            feed = {
                self.mdl['train_PH']:   True,
                self.mdl['cards_PH']:   np.asarray(cards_batch),
                self.mdl['switch_PH']:  np.asarray(switch_batch),
                self.mdl['event_PH']:   np.asarray(event_batch),
                self.mdl['state_PH']:   np.asarray(state_batch),
                self.mdl['move_PH']:    np.asarray(move_batch),
                self.mdl['rew_PH']:     np.asarray(reward_batch)}

            fetches = [
                self.mdl['optimizer'],
                self.mdl['fin_state'],
                self.mdl['loss'],
                self.mdl['scaled_LR'],
                self.mdl['enc_zeroes'],
                self.mdl['cnn_zeroes'],
                self.mdl['gg_norm'],
                self.mdl['avt_gg_norm']]
            _, fstat, loss, sLR, enc_zs, cnn_zs, gn, agn = self.mdl.session.run(fetches, feed_dict=feed)

            # save upd states
            for ix in range(fstat.shape[0]):
                self.last_upd_state[upd_p[ix]] = fstat[ix]

            self.ze_pro_enc.process(enc_zs, self.upd_step)
            self.ze_pro_cnn.process(cnn_zs, self.upd_step)

            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/0.loss', simple_value=loss)]))
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/1.sLR', simple_value=sLR)]))
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/2.gn', simple_value=gn)]))
            self.__add_upd_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/3.agn', simple_value=agn)]))

            return n_upd, upd_p

        return None

    # flush properly
    def _update_done_states(self, ust_details) -> None:
        if ust_details is None: super()._update_done_states(ust_details) # to remove all while not learning
        # leave only not used
        else:
            n_upd, upd_p = ust_details
            for p_addr in upd_p:
                self._done_states[p_addr] = self._done_states[p_addr][n_upd:]
            self._n_done -= n_upd * len(upd_p)

    # run in the process_target_method (before the loop)
    def _pre_process(self):

        self.mdl = NEModel(
            fwd_func=   self.fwd_func,
            mdict=      self.mdict,
            devices=    self.device,
            verb=       self.verb)

        # get counters
        self.start_hand = self.mdl.session.run(self.mdl['n_hands'])
        self.upd_step = self.mdl.session.run(self.mdl['g_step'])
        super()._pre_process() # activates SM

        self.zero_state = self.mdl.session.run(self.mdl['single_zero_state'])
        self.last_fwd_state =   {pa: self.zero_state    for pa in self.p_addrL}  # net state after last fwd
        self.last_upd_state =   {pa: self.zero_state    for pa in self.p_addrL}  # net state after last upd
        self.my_cards =         {pa: []                 for pa in self.p_addrL}  # current cards of player, updated while encoding states

        # reversed dicts from ptable, helpful while encoding states
        self.pos_nms_r = {k:POS_NMS[3].index(k) for k in POS_NMS[3]}  # hardcoded 3 here
        self.tbl_mov_r = {TBL_MOV[k]: k for k in TBL_MOV}

        self.ze_pro_cnn = ZeroesProcessor(
            intervals=      (5,20),
            tag_pfx=        'nane_cnn',
            summ_writer=    self.sm.summ_writer)
        self.ze_pro_enc = ZeroesProcessor(
            intervals=      (5,20),
            tag_pfx=        'nane_enc',
            summ_writer=    self.sm.summ_writer)

    # runs GamesManager decisions (called within _dmk_proc method)
    def _do_what_GM_says(self, gm_data):

        supported_commands = [
            'send_report',
            'save_model',
            'reload_model']
        assert gm_data in supported_commands

        if gm_data == 'send_report':
            report = {
                'n_hands':  self.sm.stats['nH'][0],
                'acc_won':  self.sm.acc_won}
            self.gm_que.put((self.name, 'report', report))

        if gm_data == 'save_model':
            self._save_model()
            self.gm_que.put((self.name, 'model_saved', None))

        if gm_data == 'reload_model':
            self._reload_model()
            self.gm_que.put((self.name, 'model_reloaded', None))

    # saves model checkpoint
    def _save_model(self):
        self.mdl.session.run(self.mdl['n_hands'].assign(self.sm.stats['nH'][0]))
        self.mdl.saver.save()

    # reloads model checkpoint (after GX)
    def _reload_model(self): self.mdl.saver.load()
