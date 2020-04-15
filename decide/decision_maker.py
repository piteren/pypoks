"""

 2019 (c) piteren

 DMK (decision maker) makes decisions for poker players based on MSOD (many states one decision) concept

     DMK assumes that:
        - player sends list of states (DMK.take_states)
        - from time to time player sends list of possible_moves (DMK.take_possible_moves)
            > possible_moves are added (by DMK) to the last state of player
            > move should be chosen by DMK regarding received states (saved in _new_states) and possible_moves
                > player/table can wait for a while
                > no new states from the table (for that player) will come until move will be made
                > table with player (that waits for move) is locked (no other player will send anything then...)

        - DMK makes decisions - moves for players (DMK.make_decisions):
            > makes decisions for (one-some-all) players with possible_moves @_new_states
            > states with move are moved(appended) to _done_states of processed players

        - DMK runs update to learn to make good decisions
            > update is based on _done_states (~history)

     DMK leaves for implementation:
        - _enc_states               - states encoding/processing
        - _dec_from_new_states      - how to use _new_states to make decisions
        - _run_update               - how to learn using _done_states

 ProDMK is a DMK implemented as a process
    - communicates with table and players with ques
    - introduces move_probabilities (but lefts implementation of calculating them - _calc_probs)

    ProDMK leaves for implementation:
        - _pre_process              - called before process loop
        - _enc_states               - states encoding/processing
        - _calc_probs               - how to calculate probabilities for states
        - _run_update               - how to learn using _done_states

 NeurDMK is a ProDMK implemented with Neural Network to make decisions
    - from time to time updates itself(learns NN) using information from _done_states

"""

from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
import numpy as np
import random
import tensorflow as tf
from typing import List
import time
from queue import Empty

from pologic.podeck import PDeck
from pologic.potable import POS_NMS, TBL_MOV

from putils.neuralmess.nemodel import NEModel
from putils.neuralmess.base_elements import ZeroesProcessor

# state type, here DMK saves table events
class State:

    def __init__(
            self,
            value):

        self.value = value                                  # value of state (any object)
        self.possible_moves :List[bool] or None=    None    # list of (current) possible moves
        self.probs :List[float] or None=            None    # probabilities of moves
        self.move :int or None=                     None    # move (performed/selected by DMK)
        self.reward :float or None=                 None    # reward (after hand finished)
        self.move_rew :float or None=               None    # move shared reward

# Stats Manager for DMK (one DMK has N table players)
# - uses n_hands as a counter (number of hands performed by ALL table players of DMK)
# - is an essential component of aware DMK / GamesManager since prepares data useful to make higher level decisions
class StatsMNG:

    def __init__(
            self,
            name :str,
            p_addrL :list,              # list of (unique) player ids, used as keys in dicts
            stats_iv=   1000,           # interval (n_hands) for putting stats on TB
            acc_won_iv= (30000,100000), # should be multiplication of stats_iv
            verb=       0):

        self.verb = verb
        self.stats_iv = stats_iv
        for v in acc_won_iv: assert v % stats_iv == 0
        self.speed =        None # speed of running in H/s
        self.won_save =     {0: 0} # {n_hand: $won} saved while putting to TB (for stats_iv), it will grow but won't be big...
        self.acc_won =      {k: 0 for k in acc_won_iv} # $won/hand in ranges of acc_won_iv
        self.stats =        {} # stats of DMK (for all players)
        self.chsd =         {pID: None for pID in p_addrL} # current hand stats data (per player)
        self.is_BB =        {pID: False for pID in p_addrL} # BB position of player at table {pID: True/False}
        self.is_preflop =   {pID: True for pID in p_addrL} # preflop indicator of player at table {pID: True/False}

        self.reset_stats()
        for pID in self.chsd: self.__reset_chsd(pID)

        self.summ_writer = tf.summary.FileWriter(logdir='_models/' + name, flush_secs=10)
        self.stime = time.time()

    # resets stats (DMK)
    def reset_stats(self):
        """
        by now implemented stats:
          VPIP  - Voluntarily Put $ in Pot %H: how many hands (%) player put money in pot (SB and BB do not count)
          PFR   - Preflop Raise: how many hands (%) player raised preflop
          HF    - Hands Folded: %H where player folds
          AGG   - Postflop Aggression Frequency %: (totBet + totRaise) / anyMove *100
        """
        self.stats = {  # [total,interval]
            'nH':       [0,0],  # n hands played
            '$':        [0,0],  # $ won
            'nVPIP':    [0,0],  # n hands VPIP
            'nPFR':     [0,0],  # n hands PFR
            'nHF':      [0,0],  # n hands folded
            'nPM':      [0,0],  # n moves postflop
            'nAGG':     [0,0]}  # n aggressive moves postflop

    # resets self.chsd for player (per player stats)
    def __reset_chsd(
            self,
            pID):

        self.chsd[pID] = {
            'VPIP':     False,
            'PFR':      False,
            'HF':       False,
            'nPM':      0,      # num of postflop moves
            'nAGG':     0}

    # updates self.chsd with given player move
    def __upd_chsd(
            self,
            pID,
            move :str):

        if move == 'C/F': self.chsd[pID]['HF'] = True
        if self.is_preflop[pID]:
            if move == 'CLL' and not self.is_BB[pID] or 'BR' in move: self.chsd[pID]['VPIP'] = True
            if 'BR' in move: self.chsd[pID]['PFR'] = True
        else:
            self.chsd[pID]['nPM'] += 1
            if 'BR' in move: self.chsd[pID]['nAGG'] += 1

    # puts DMK stats to TB
    def __push_TB(self):

        speed_summ = tf.Summary(value=[tf.Summary.Value(tag=f'reports/speed(H/s)', simple_value=self.speed)])
        self.summ_writer.add_summary(speed_summ, self.stats['nH'][0])

        for k in self.acc_won:
            if self.stats['nH'][0] >= k:
                acw_summ = tf.Summary(value=[tf.Summary.Value(tag=f'sts_acc_won/{k}', simple_value=self.acc_won[k])])
                self.summ_writer.add_summary(acw_summ, self.stats['nH'][0])

        won_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/0.$wonT', simple_value=self.stats['$'][0])])
        vpip = self.stats['nVPIP'][1] / self.stats['nH'][1] * 100
        vpip_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/1.VPIP', simple_value=vpip)])
        pfr = self.stats['nPFR'][1] / self.stats['nH'][1] * 100
        pfr_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/2.PFR', simple_value=pfr)])
        agg = self.stats['nAGG'][1] / self.stats['nPM'][1] * 100 if self.stats['nPM'][1] else 0
        agg_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/3.AGG', simple_value=agg)])
        ph = self.stats['nHF'][1] / self.stats['nH'][1] * 100
        ph_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/4.HF', simple_value=ph)])
        self.summ_writer.add_summary(won_summ, self.stats['nH'][0])
        self.summ_writer.add_summary(vpip_summ, self.stats['nH'][0])
        self.summ_writer.add_summary(pfr_summ, self.stats['nH'][0])
        self.summ_writer.add_summary(agg_summ, self.stats['nH'][0])
        self.summ_writer.add_summary(ph_summ, self.stats['nH'][0])


    # extracts stats from player states
    def take_states(
            self,
            pID,
            states :List[list]):

        for s in states:
            if s[0] == 'TST' and s[1] == 'preflop': self.is_preflop[pID] =  True
            if s[0] == 'TST' and s[1] == 'flop':    self.is_preflop[pID] =  False
            if s[0] == 'POS' and s[1][0] == 0:      self.is_BB[pID] =       s[1][1]=='BB'
            if s[0] == 'MOV' and s[1][0] == 0:      self.__upd_chsd(pID, s[1][1])
            if s[0] == 'PRS' and s[1][0] == 0:
                my_reward = s[1][1]
                for ti in [0,1]:
                    self.stats['nH'][ti] += 1
                    self.stats['$'][ti] += my_reward

                    # update self.stats with self.chsd
                    if self.chsd[pID]['VPIP']:  self.stats['nVPIP'][ti] += 1
                    if self.chsd[pID]['PFR']:   self.stats['nPFR'][ti] += 1
                    if self.chsd[pID]['HF']:    self.stats['nHF'][ti] += 1
                    self.stats['nPM'][ti] +=    self.chsd[pID]['nPM']
                    self.stats['nAGG'][ti] +=   self.chsd[pID]['nAGG']

                self.__reset_chsd(pID)

                # put stats on TB
                if self.stats['nH'][1] == self.stats_iv:

                    self.speed = self.stats_iv/(time.time()-self.stime)
                    self.stime = time.time()

                    hand_num = self.stats['nH'][0]
                    self.won_save[hand_num] = self.stats['$'][0]
                    for k in self.acc_won:
                        if hand_num-k >= 0:
                            self.acc_won[k] = (self.won_save[hand_num]-self.won_save[hand_num-k])/k

                    self.__push_TB()
                    for key in self.stats.keys(): self.stats[key][1] = 0 # reset interval values

# abstract class of DMK, defines basic interface of DMK to act on PTable with PPlayer
class DMK(ABC):

    def __init__(
            self,
            name :str,                  # name should be unique (@table)
            n_players :int,             # number of managed players
            n_moves :int=       4,      # number of (all) moves supported by DM, has to match table/player
            upd_trigger :int=   1000,   # how much _done_states to accumulate to launch update
            verb=               0):

        self.name = name
        self.verb = verb
        self.n_players = n_players
        self.p_addrL = [f'{self.name}_{ix}' for ix in range(self.n_players)] # DMK builds addresses for players
        self.n_moves = n_moves
        self.upd_trigger = upd_trigger
        self.upd_step = 0
        if self.verb>0: print(f'\n *** DMK *** initialized, name {self.name}, players {self.n_players}')

        # variables below store moves/decisions data
        self._new_states =      {} # dict of lists of new states {state: }, should not have empty keys (player indexes)
        self._n_new = 0         # cache number of states @_new_states
        self._n_pmoves = 0      # cache number of possible moves @_new_states
        self._done_states =     {pa: [] for pa in self.p_addrL}  # dict of processed (with decision) states
        self._n_done = 0        # cache number of done @_done_states

    # encodes table states into DMK form (appropriate to make decisions)
    @abstractmethod
    def _enc_states(
            self,
            p_addr,
            player_stateL: List[list]) -> List[State]:
        return [State(value) for value in player_stateL]  # wraps into dictL

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
            possible_moves :List[bool]):

        assert p_addr in self._new_states # TODO (dev safety check): player should have new states while getting possible moves
        self._new_states[p_addr][-1].possible_moves = possible_moves # add to last new state
        self._n_pmoves += 1

    # using data from _new_states prepares list of decisions in form: [(p_addr,move)...]
    @abstractmethod
    def _dec_from_new_states(self) -> List[tuple]: pass

    # move states (from _new to _done) having list of decisions, updates caches
    def __move_states(self, decL :List[tuple]) -> None:
        self._n_pmoves -= len(decL)
        for dec in decL:
            p_addr, move = dec
            states = self._new_states.pop(p_addr)
            states[-1].move = move # write move into state
            self._done_states[p_addr] += states
            self._n_new -= len(states)
            self._n_done += len(states)

    # makes decisions (for one-some-all) players with possible moves
    def make_decisions(self) -> List[tuple]:

        assert self._n_pmoves > 0 # TODO (dev safety check): DMK should have possible_moves when asked for decisions
        decL = self._dec_from_new_states()
        assert decL

        self.__move_states(decL)                                # move states
        if self._n_done > self.upd_trigger: self._run_update()  # self update (learn)

        return decL

    # runs update of DMK based on saved _done_states
    @abstractmethod
    def _run_update(self) -> None:
        self.upd_step += 1
        # baseline (no learn): flush all & reset
        self._done_states = {pa: [] for pa in self._done_states}
        self._n_done = 0

# process(ed) DMK
# + communicates with ques (with players/tables and with GamesManager)
# + implements _dec_from_new_states as a 2 step process:
#       > calculate move probabilities (_calc_probs)
#       > sample move using probs (__sample_move)
# + implements exploration while sampling move
class ProDMK(Process, DMK, ABC):

    def __init__(
            self,
            gm_que :Queue,          # GamesManager que, here DMK puts data only! for GM, data is always put in form of tuple (name, type, data)
            pmex :float=    0.0,    # <0;1> possible_moves exploration (probability of random sampling from possible_moves space)
            suex :float=    0.2,    # <0;1> self uncertainty exploration (probability of sampling from probability (rather than taking max))
            **kwargs):

        Process.__init__(self, target=self._dmk_proc)
        DMK.__init__(self, **kwargs)

        self.suex = suex
        self.pmex = pmex

        self.gm_que = gm_que
        self.in_que = Queue() # here DMK receives data only! from GM

        # every ProDMK creates part of ques network
        self.dmk_in_que = Queue()
        self.pl_in_queD = {pa: Queue() for pa in self.p_addrL}

    # method called BEFORE process loop, builds objects that HAVE to be build in process memory scope
    @abstractmethod
    def _pre_process(self) -> None: pass

    # calculates probabilities for at least some states with possible_moves
    @abstractmethod
    def _calc_probs(self) -> None: pass

    # using pmex and suex samples single move for given probabilities and possibilities
    def __sample_move(
            self,
            probs :List[float],
            possible_moves :List[bool]) -> int:

        if self.pmex > 0 and random.random() < self.pmex: probs = np.random.rand(self.n_moves) # forced random probability

        prob_mask = np.asarray(possible_moves).astype(int) # to np
        probs = probs * prob_mask  # mask probs
        if np.sum(probs)==0: probs = prob_mask  # take mask if no intersection

        moves_arr = np.arange(self.n_moves)
        if self.suex > 0 and random.random() < self.suex:
            probs /= np.sum(probs)  # normalize
            move = np.random.choice(moves_arr, p=probs) # sample from probs
        else: move = moves_arr[np.argmax(probs)] # take max from probs

        return move

     # returns list of decisions
    def _dec_from_new_states(self) -> List[tuple]:

        super()._dec_from_new_states()

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

        decL = []
        for p_addr in self._new_states:
            if self._new_states[p_addr][-1].possible_moves is not None and self._new_states[p_addr][-1].probs is not None:
                move = self.__sample_move(
                    probs=          self._new_states[p_addr][-1].probs,
                    possible_moves =self._new_states[p_addr][-1].possible_moves)
                decL.append((p_addr, move))

        return decL

    # runs GamesManager decisions
    def _do_what_gm_says(self, gm_data) -> None: pass

    # process method (loop, target of process)
    def _dmk_proc(self):

        self._pre_process()

        n_waiting = 0 # num players ( ~> tables) waiting for decision
        while True:

            # 'flush' the que of data from players
            pdL = []
            while True:
                try:            player_data = self.dmk_in_que.get_nowait()
                except Empty:   break
                if player_data: pdL.append(player_data)

            #print(len(pdL))
            for player_data in pdL:
                p_addr = player_data['id']

                if 'state_changes' in player_data:
                    self.take_states(p_addr, player_data['state_changes'])

                if 'possible_moves' in player_data:
                    self.take_possible_moves(p_addr, player_data['possible_moves'])
                    n_waiting += 1

            # now, if got any waiting >> make decisions
            if n_waiting:
                decL = self.make_decisions()
                #print(' %d >> %d'%(n_waiting,len(decL)))
                n_waiting -= len(decL)
                for d in decL:
                    p_addr, move = d
                    self.pl_in_queD[p_addr].put(move)

            try: # eventually get data from GM
                gm_data = self.in_que.get_nowait()
                if gm_data:
                    if gm_data == 'stop':
                        self.gm_que.put((self.name, 'finished', None))
                        break
                    else: self._do_what_gm_says(gm_data)
            except Empty: pass

# ProDMK with baseline equal probs
# + implements stats
class RProDMK(ProDMK):

    def __init__(
            self,
            stats_iv=       1000, # collects players/DMK stats, runs TB (for None/0 does not)
            acc_won_iv=     (30000,100000),
            **kwargs):

        super().__init__(**kwargs)

        self.sm = None
        self.stats_iv = stats_iv
        self.acc_won_iv = acc_won_iv

    # builds stats_manager before process loop (since summ_writer...)
    def _pre_process(self):

        super()._pre_process()

        # TB stats saver (has to be build here, inside process method)
        self.sm = StatsMNG(
            name=       self.name,
            p_addrL=    self.p_addrL,
            stats_iv=   self.stats_iv,
            acc_won_iv= self.acc_won_iv,
            verb=       self.verb) if self.stats_iv else None

    # adds stats management
    def _enc_states(
            self,
            pID,
            player_stateL: List[list]) -> List[State]:
        if self.sm: self.sm.take_states(pID, player_stateL)
        return super()._enc_states(pID,player_stateL)

    # calculates probabilities - baseline: sets equal for all new states of players with possible moves
    def _calc_probs(self) -> None:
        baseline_probs = [1/self.n_moves] * self.n_moves # equal probs
        for p_addr in self._new_states:
            if self._new_states[p_addr][-1].possible_moves:
                self._new_states[p_addr][-1].probs = baseline_probs

    # nothing new added
    def _run_update(self) -> None: super()._run_update()

# Neural DMK
# + encodes states for NN
# + makes decisions with NN
# + updates NN (learns)
# + exploration reduction (policy)
class NeurDMK(RProDMK):

    def __init__(
            self,
            fwd_func,
            device=         1,
            upd_BS=         50000,  # estimated target batch size of update
            ex_reduce=      0.95,   # exploration reduction factor (with each update)
            **kwargs):

        upd_trigger = 2*upd_BS # twice size since updating half_rectangle in trapeze
        super().__init__(upd_trigger=upd_trigger, **kwargs)
        assert self.stats_iv > 0 # stats are obligatory for NeurDMK

        self.fwd_func = fwd_func
        self.device = device

        self.upd_BS = upd_BS
        self.ex_reduce = ex_reduce

    # run in the process_target_method (before the loop)
    def _pre_process(self):

        super()._pre_process()

        self.mdl = NEModel(
            fwd_func=   self.fwd_func,
            mdict=      {'name':self.name, 'verb':0}, # TODO: by now base concept
            devices=    self.device,
            verb=       self.verb)

        self.zero_state = self.mdl.session.run(self.mdl['single_zero_state'])
        self.last_fwd_state =   {pa: self.zero_state    for pa in self.p_addrL}  # net state after last fwd
        self.my_cards =         {pa: []                 for pa in self.p_addrL}  # current cards of player, updated while encoding states
        # TODO: do I need upd_state?
        # self.last_upd_state =   {ix: zero_state for ix in range(self.n_players)}  # net state after last upd

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

    # prepares state into form of nn input
    #  - encodes only selection of states
    #   event has values (ids):
    #       0 : pad
    #       1,2,3 : my positions SB,BB,BTN
    #       4,5,6,7, 8,9,10,11 : moves of two opponents 1,2 * C/F,CLL,BR5,BR8
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

        # TODO: try np. for speed
        feed = {
            self.mdl['cards_PH']:   cards_batch,
            self.mdl['train_PH']:   False,
            self.mdl['event_PH']:   event_batch,
            self.mdl['switch_PH']:  switch_batch,
            self.mdl['state_PH']:   state_batch}

        fetches = [self.mdl['probs'], self.mdl['fin_state']]
        probs, fwd_states = self.mdl.session.run(fetches, feed_dict=feed)
        probs = np.squeeze(probs, axis=1) # remove sequence axis (1)

        for ix in range(fwd_states.shape[0]):
            p_addr = p_addrL[ix]
            probs_row.append((p_addr, probs[ix]))
            self.last_fwd_state[p_addr] = fwd_states[ix]

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

    # runs update of DMK based on saved _done_states
    def _run_update(self) ->None:

        def mam(valL: list) -> list: return [min(valL), sum(valL)/len(valL), max(valL)]

        def add_summ(summ :tf.Summary) -> None: self.sm.summ_writer.add_summary(summ, self.upd_step)

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

        lrm = [rewards[p_addr][0][0] for p_addr in p_addrL] # last rewarded move of player (index of state)
        # print(f' last rm : {mam(lrm)}')
        lrmT = zip(p_addrL, lrm)
        lrmT = sorted(lrmT, key=lambda x: x[1], reverse=True)
        half_players = len(self._done_states) // 2
        if len(lrmT) < half_players: half_players = len(lrmT)
        upd_p = lrmT[:half_players]
        n_upd = upd_p[-1][1] + 1  # n states to use for update
        upd_p = [e[0] for e in upd_p]  # list of p_addr to update
        # print(f' >>> BS   : {len(upd_p)} x {n_upd} ({len(upd_p)*n_upd})')

        # num of states (@_done_states per player)
        n_sts = mam([len(self._done_states[p_addr]) for p_addr in p_addrL]) #
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/0.n_states_min', simple_value=n_sts[0] )]))
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/1.n_states_max', simple_value=n_sts[2] )]))

        # num of states with moves
        n_mov = mam([sum([len(ml) for ml in rewards[p_addr]]) for p_addr in p_addrL])
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/2.n_moves_min', simple_value=n_mov[0] )]))
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/3.n_moves_max', simple_value=n_mov[2] )]))

        # num of states with rewards
        n_rew = mam([len(rewards[p_addr]) for p_addr in p_addrL])
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/4.n_rewards_min', simple_value=n_rew[0] )]))
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/5.n_rewards_max', simple_value=n_rew[2] )]))

        add_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/6.n_sts/mov', simple_value=n_sts[1]/ n_mov[1] )]))
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/7.n_sts/rew', simple_value=n_sts[1]/ n_rew[1] )]))
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='reports.UPD/8.n_mov/rew', simple_value=n_mov[1] / n_rew[1] )]))

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
            state_batch.append(self.zero_state)

        # TODO: try np. for speed
        feed = {
            self.mdl['train_PH']:   True,
            self.mdl['cards_PH']:   cards_batch,
            self.mdl['switch_PH']:  switch_batch,
            self.mdl['event_PH']:   event_batch,
            self.mdl['state_PH']:   state_batch,
            self.mdl['move_PH']:    move_batch,
            self.mdl['rew_PH']:     reward_batch}

        fetches = [
            self.mdl['optimizer'],
            self.mdl['loss'],
            self.mdl['scaled_LR'],
            self.mdl['enc_zeroes'],
            self.mdl['cnn_zeroes'],
            self.mdl['gg_norm'],
            self.mdl['avt_gg_norm']]
        _, loss, sLR, enc_zs, cnn_zs, gn, agn = self.mdl.session.run(fetches, feed_dict=feed)

        self.ze_pro_enc.process(enc_zs, self.upd_step)
        self.ze_pro_cnn.process(cnn_zs, self.upd_step)

        add_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/0.loss',    simple_value=loss)]))
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/1.sLR',     simple_value=sLR)]))
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/2.gn',      simple_value=gn)]))
        add_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/3.agn',     simple_value=agn)]))

        # reduce after each update
        if self.suex > 0:
            self.suex *= self.ex_reduce
            if self.suex < 0.001: self.suex = 0
            add_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/4.suex', simple_value=self.suex)]))
        if self.pmex > 0:
            self.pmex *= self.ex_reduce
            if self.pmex < 0.001: self.pmex = 0
            add_summ(tf.Summary(value=[tf.Summary.Value(tag='nn/4.pmex', simple_value=self.pmex)]))

        # leave only not used
        for p_addr in upd_p:
            self._done_states[p_addr] = self._done_states[p_addr][n_upd:]
        self._n_done -= n_upd*len(upd_p)

        self.upd_step += 1

    # runs GamesManager decisions
    def _do_what_gm_says(self, gm_data):

        supported_commands = [
            'send_report',
            'save_model',
            'reload_model']
        assert gm_data in supported_commands

        if gm_data == 'send_report':
            report = {
                'n_hand':   self.sm.stats['nH'][0],
                'acc_won':  self.sm.acc_won}
            self.gm_que.put((self.name, 'report', report))

        if gm_data == 'save_model':
            self._save_model()
            self.gm_que.put((self.name, 'model_saved', None))

        if gm_data == 'reload_model':
            self._reload_model()
            self.gm_que.put((self.name, 'model_reloaded', None))

    # saves checkpoints
    def _save_model(self): self.mdl.saver.save()

    # reloads model checkpoint (after GX)
    def _reload_model(self): self.mdl.saver.load()
