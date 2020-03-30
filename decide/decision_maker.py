"""

 2019 (c) piteren

 DMK (decision maker) is an object that makes decisions for poker players
 DMK assumes that:
    - player sends list of states
    - from time to time player sends list of possible moves,
        > it means that move should be chosen by DMK using received states
        > possible moves are added to the last state of player
        > no new states will come until move will be made
        > table with player (that waits for move) is locked (no other player will send anything then...)

        > when DMK is asked to make_decisions (moves for playes) it:
            > has to make at least one move ( >> for at least one player)
            > states with move are appended to _done_states
"""

from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
import numpy as np
import random
import tensorflow as tf
from typing import List
import time

from pologic.podeck import PDeck
from pologic.potable import POS_NMS, TBL_MOV

from putils.neuralmess.nemodel import NNModel

class State:

    def __init__(
            self,
            value):

        self.value = value                                  # value of state (any object)
        self.possible_moves :List[bool] or None=    None
        self.probs :List[float] or None=            None
        self.move :int or None=                     None
        self.reward :float or None=                 None
        self.move_rew :float or None=               None    # move shared reward

# stats manager for DMK case (one DMK and N players)
class StatsMNG:

    def __init__(
            self,
            name :str,
            pl_IDL :list,       # list of (unique) player ids
            stats_iv=   1000):

        self.stats_iv = stats_iv
        self.stats =        {} # stats of DMK (for all players)
        self.chsd =         {pID: None  for pID in pl_IDL} # current hand stats data (per player)
        self.is_BB =        {pID: False for pID in pl_IDL} # BB position of player at table {pID: True/False}
        self.is_preflop =   {pID: True  for pID in pl_IDL} # preflop indicator of player at table {pID: True/False}

        self.reset_stats()
        for pID in self.chsd: self.__reset_chsd(pID)

        self.summ_writer = tf.summary.FileWriter(logdir='_models/' + name, flush_secs=10)

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
        won = tf.Summary(value=[tf.Summary.Value(tag='sts/0.$wonT', simple_value=self.stats['$'][0])])
        vpip = self.stats['nVPIP'][1] / self.stats['nH'][1] * 100
        vpip = tf.Summary(value=[tf.Summary.Value(tag='sts/1.VPIP', simple_value=vpip)])
        pfr = self.stats['nPFR'][1] / self.stats['nH'][1] * 100
        pfr = tf.Summary(value=[tf.Summary.Value(tag='sts/2.PFR', simple_value=pfr)])
        agg = self.stats['nAGG'][1] / self.stats['nPM'][1] * 100 if self.stats['nPM'][1] else 0
        agg = tf.Summary(value=[tf.Summary.Value(tag='sts/3.AGG', simple_value=agg)])
        ph = self.stats['nHF'][1] / self.stats['nH'][1] * 100
        ph = tf.Summary(value=[tf.Summary.Value(tag='sts/4.HF', simple_value=ph)])
        self.summ_writer.add_summary(won, self.stats['nH'][0])
        self.summ_writer.add_summary(vpip, self.stats['nH'][0])
        self.summ_writer.add_summary(pfr, self.stats['nH'][0])
        self.summ_writer.add_summary(agg, self.stats['nH'][0])
        self.summ_writer.add_summary(ph, self.stats['nH'][0])

    # extracts stats from player states
    def get_states(
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
                    self.__push_TB()
                    for key in self.stats.keys(): self.stats[key][1] = 0 # reset interval values

# abstract class of DMK, defines basic interface of DMK to act on PTable with PPlayer
class DMK(ABC):

    def __init__(
            self,
            name :str,              # name should be unique (@table)
            n_players :int,         # number of managed players
            n_moves :int=   4,      # number of (all) moves supported by DM, has to match table/player
            stats_iv=       1000,   # collects players/DMK stats, runs TB (for None does not)
            verb=           0):

        self.name = name
        self.verb = verb
        self.n_players = n_players
        self.n_moves = n_moves
        if self.verb>0: print('\n *** DMK *** initialized, name %s, players %d' % (self.name, self.n_players))

        # variables below store moves/decisions data
        self._new_states =      {} # dict of lists of new states {state: }, should not have empty keys (player indexes)
        self._n_waiting =       0 # number of players (@this DMK) actually waiting for move (cached number)
        self._done_states =     {ix: [] for ix in range(self.n_players)}  # dict of processed (with decision) states

        # create stats manager
        self.stats_mng = StatsMNG(
            name=       self.name,
            n_players=  self.n_players,
            stats_iv=   stats_iv) if stats_iv else None

    # encodes table states into DMK form (appropriate to make decisions), TO BE OVERRIDDEN
    def _enc_states(
            self,
            pIX: int,
            player_stateL: List[list]) -> List[State]:
        if self.stats_mng: self.stats_mng.get_states(pIX, player_stateL)
        return [State(value) for value in player_stateL]  # wraps into dictL

    # takes player states, encodes and saves in self._new_states
    def take_states(
            self,
            pIX,
            player_stateL: List[list]) -> None:

        if pIX not in self._new_states: self._new_states[pIX] = []
        self._new_states[pIX] += self._enc_states(pIX, player_stateL)
        if not self._new_states[pIX]: self._new_states.pop(pIX) # case of encoded states actually empty

    # takes player possible_moves and saves
    def take_possible_moves(
            self,
            pIX,
            possible_moves :List[bool]):

        self._n_waiting += 1
        assert pIX in self._new_states # TODO: to remove, develop safety check, player should have new states while getting possible moves
        self._new_states[pIX][-1].possible_moves = possible_moves # add to last new state

    # returns number of players waiting
    def num_waiting(self) -> int: return self._n_waiting

    # should return list of decisions in form: [(pIX,move)...], cannot return empty list
    # decision should be made based on stored data (...in _new_states)
    @abstractmethod
    def _madec(self) -> List[tuple]: pass

    # returns decisions
    def make_decisions(self) -> List[tuple]:

        decL = self._madec()
        assert decL
        for dec in decL:
            pIX, move = dec
            states = self._new_states.pop(pIX)
            states[-1].move = move
            self._done_states[pIX] += states # move new into done (where decisions made)
        self._n_waiting -= len(decL)
        return decL

# process(ed) DMK
class PDMK(Process):

    def __init__(
            self,
            name :str,
            n_players=  30):

        Process.__init__(self, target=self.__dmk_proc)
        self.name = name
        self.n_players = n_players
        p_addrL = ['%s_%d'%(self.name,ix) for ix in range(self.n_players)] # list of player addresses

        self.pl_out_que = Queue()
        self.pl_in_queD = {pa: Queue() for pa in p_addrL}


    def __dmk_proc(self):

        p_addrL = list(self.pl_in_queD.keys())
        sm = StatsMNG(name=self.name, pl_IDL=p_addrL)

        decision = [0,1,2,3]  # hardcoded to speed-up
        while True:
            player_data = self.pl_out_que.get()
            p_addr = player_data['id']

            if 'state_changes' in player_data:
                sm.get_states(p_addr,player_data['state_changes'])

            if 'possible_moves' in player_data:
                possible_moves = player_data['possible_moves']

                pm_probs = [int(pm) for pm in possible_moves]
                dec = random.choices(decision, weights=pm_probs)[0]

                self.pl_in_queD[p_addr].put(dec)

# random DMK implementation (first baseline)
# + introduces move probabilities
class RDMK(DMK):

    def __init__(
            self,
            **kwargs):

        DMK.__init__(self, **kwargs)

    # samples single move for given probabilities and possibilities
    def __sample_move(
            self,
            probs :List[float],
            possible_moves :List[bool]):

        prob_mask = [int(pM) for pM in possible_moves]  # cast bool to int
        prob_mask = np.asarray(prob_mask) # to np
        probs = probs * prob_mask  # mask probs
        if np.sum(probs) == 0: probs = prob_mask  # take mask if no intersection
        probs /= np.sum(probs)  # normalize
        move = np.random.choice(np.arange(self.n_moves), p=probs)  # sample from probs # TODO: for tournament should take max
        return move

    # returns list of decisions
    def _madec(self) -> list:

        if self.verb>1:
            nd = {}
            for pIX in self._new_states:
                l = len(self._new_states[pIX])
                if l not in nd: nd[l] = 0
                nd[l] += 1
                if l > 10:
                    for s in self._new_states[pIX]: print(s)
            print(' >> (@_madec) _new_states histogram:')
            for k in sorted(list(nd.keys())): print(' >> %d:%d'%(k,nd[k]))

        decL = []
        for pIX in self._new_states:
            if self._new_states[pIX][-1].possible_moves is not None and self._new_states[pIX][-1].probs is not None:
                possible_moves =    self._new_states[pIX][-1].possible_moves
                probs =             self._new_states[pIX][-1].probs
                move =              self.__sample_move(probs, possible_moves)
                decL.append([pIX, move])
        return decL

    # add probabilities for at least some states with possible_moves, baseline sets for all (last states) with possible moves
    # TO BE OVERRIDEN
    def _calc_probs(self) -> None:
        baseline_probs = [1/self.n_moves] * self.n_moves # equal probs
        for pIX in self._new_states:
            if self._new_states[pIX][-1].possible_moves:
                self._new_states[pIX][-1].probs = baseline_probs

    # returns decisions
    def make_decisions(self) -> list:
        self._calc_probs()
        return DMK.make_decisions(self)

# Neural DMK
# + encodes states
# + makes decisions with NN
# + updates NN (learns)
class NDMK(RDMK):

    def __init__(
            self,
            fwd_func,
            rand_moveF=     0.0,    # how often move will be sampled from random
            n_stat_upd=     50000, # ...for all players
            **kwargs):

        RDMK.__init__( self, **kwargs)

        self.mdl = NNModel(
            fwd_func=   fwd_func,
            mdict=      {'name':self.name, 'verb':2}, # TODO: by now base concept
            devices=    1,
            verb=       0)

        self.rand_move = rand_moveF
        self.n_stat_upd = n_stat_upd

        self.zero_state = self.mdl.session.run(self.mdl['single_zero_state'])
        self.last_fwd_state =   {ix: self.zero_state for ix in range(self.n_players)}  # net state after last fwd
        self.my_cards =         {ix: []         for ix in range(self.n_players)}  # current cards of player, updated while encoding states
        # I do not need upd_state, since I do a hard-reset of states after each update
        # self.last_upd_state =   {ix: zero_state for ix in range(self.n_players)}  # net state after last upd

        # reversed dicts from ptable, helpful while encoding states
        self.pos_nms_r = {k:POS_NMS[3].index(k) for k in POS_NMS[3]}  # hardcoded 3 here
        self.tbl_mov_r = {TBL_MOV[k]: k for k in TBL_MOV}

    # prepares state into form of nn input
    #  - encodes only selection of states
    #   event has values (ids):
    #       0 : pad
    #       1,2,3 : my positions SB,BB,BTN
    #       4,5,6,7, 8,9,10,11 : moves of two opponents 1,2 * C/F,CLL,BR5,BR8
    def _enc_states(
            self,
            pIX: int,
            player_stateL: list):

        es = RDMK._enc_states(self, pIX, player_stateL)
        news = [] # newly encoded states
        for s in es:
            val = s.value
            nval = None

            if val[0] == 'POS' and val[1][0] == 0: # my position
                nval = {
                    'cards':    None,
                    'event':    1 + self.pos_nms_r[val[1][1]]}

            if val[0] == 'PLH' and val[1][0] == 0: # my hand
                self.my_cards[pIX] = [PDeck.cti(c) for c in val[1][1:]]
                nval = {
                    'cards':    [] + self.my_cards[pIX], # copy cards
                    'event':    0}

            if val[0] == 'TCD': # my hand update
                self.my_cards[pIX] += [PDeck.cti(c) for c in val[1]]
                nval = {
                    'cards':    [] + self.my_cards[pIX], # copy cards
                    'event':    0}

            if val[0] == 'MOV' and val[1][0] != 0: # moves, all but mine
                nval = {
                    'cards':    None,
                    'event':    4 + self.tbl_mov_r[val[1][1]] + 4*(val[1][0]-1)}  # hardcoded for 2 opponents

            if val[0] == 'PRS' and val[1][0] == 0: # my result
                self.my_cards[pIX] = [] # reset my cards

                reward = val[1][1]
                # it is a bit tricky but we have to append reward to last state, which we have to find
                list_to_append = None
                if news: list_to_append = news
                else:
                    if self._new_states[pIX]: list_to_append = self._new_states[pIX]
                    else:
                        if self._done_states[pIX]: list_to_append = self._done_states[pIX]
                if list_to_append: list_to_append[-1].reward = reward # there may be a case, when list_to_append does not exists (after update etc.)

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
        pIXL = []
        for vr in vals_row:
            pIX, val = vr
            pIXL.append(pIX) # save list of pIX
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

            state_batch.append(self.last_fwd_state[pIX])

        feed = {
            self.mdl['cards_PH']:   cards_batch,
            self.mdl['train_PH']:   False,
            self.mdl['event_PH']:   event_batch,
            self.mdl['switch_PH']:  switch_batch,
            self.mdl['state_PH']:   state_batch}

        fetches = [self.mdl['probs'], self.mdl['fin_state']]
        probs, fwd_states = self.mdl.session.run(fetches, feed_dict=feed)

        probs = np.reshape(probs, newshape=(probs.shape[0], probs.shape[-1]))  # TODO: maybe any smarter way

        for ix in range(fwd_states.shape[0]):
            pIX = pIXL[ix]
            probs_row.append((pIX, probs[ix]))
            self.last_fwd_state[pIX] = fwd_states[ix]

        return probs_row

    # add probabilities for at least some states with possible_moves
    def _calc_probs(self) -> None:

        got_probs_for_possible = False
        while not got_probs_for_possible:

            vals_row = []
            for pIX in self._new_states:
                for s in self._new_states[pIX]:
                    if s.probs is None:
                        vals_row.append((pIX,s.value))
                        break
            assert vals_row # TODO: I can imagine empty vals_row but with current alg should not appear
            if self.verb>1: print(' > (@_calc_probs) got row of %d'%len(vals_row))

            probs_row = self.__calc_probs_vr(vals_row)
            for pr in probs_row:
                pIX, probs = pr
                for s in self._new_states[pIX]:
                    if s.probs is None:
                        s.probs = probs
                        if s.possible_moves: got_probs_for_possible = True
                        break
            if self.verb>0 and not got_probs_for_possible: print(' > (@_calc_probs) another loop...')

    # runs update of DMK based on saved _done_states
    def _run_update(self):

        n_stat = [len(self._done_states[pIX]) for pIX in self._done_states] # number of saved states per player (not all rewarded)
        n_statAV = int(sum(n_stat)/len(n_stat)) # avg
        if n_statAV > self.n_stat_upd/self.n_players:
            if self.verb>0: print('(@ _run_update) n_stat: min %d max %d avg %d'%(min(n_stat), max(n_stat), n_statAV))
            # TODO: get here some stats about number of rewards (hands), moves etc...
            # TODO: get here some time stats

            ix_rew = {pIX: [] for pIX in self._done_states} # indexes of states, where rewards are
            for pIX in self._done_states:
                for ix in range(len(self._done_states[pIX])):
                    st = self._done_states[pIX][ix]
                    if st.reward is not None: ix_rew[pIX].append(ix)

            n_rew = [len(ix_rew[pIX]) for pIX in self._done_states]
            if self.verb > 0: print('(@ _run_update) n_rew: min %d max %d avg %d' % (min(n_rew), max(n_rew), sum(n_rew)/len(n_rew)))

            ix_mov = {pIX: [] for pIX in self._done_states}  # indexes of states, where moves are
            for pIX in self._done_states:
                for ix in range(len(self._done_states[pIX])):
                    st = self._done_states[pIX][ix]
                    if st.move is not None: ix_mov[pIX].append(ix)

            n_stat = [len(ix_mov[pIX]) for pIX in self._done_states]
            if self.verb > 0: print('(@ _run_update) n_mov: min %d max %d avg %d' % (min(n_stat), max(n_stat), sum(n_stat) / len(n_stat)))

            # ********************** take for update from ix of the lowest last rewarded move

            # build rewards dict (for every player, under indexes of rewards: list of indexes of moves)
            rewD = {} # all players dict
            for pIX in self._done_states:
                rewD[pIX] = {} # player dict
                crewIX = None
                for ix in reversed(range(len(self._done_states[pIX]))): # go down over _done_states of player
                    st = self._done_states[pIX][ix]
                    if st.reward is not None:
                        crewIX = ix # current reward index
                        rewD[pIX][crewIX] = [] # current reward list of indexes with moves
                    if st.move is not None and crewIX is not None: # got move of this reward
                        rewD[pIX][crewIX].append(ix)

            """
                There is a CASE(just in game), when player gets a reward without ANY move:
                 everyone folds against BB (preflop),
                 by now we will remove such rewards,
                 but it is an interesting case: such behaviour(folding) may be caused by my previous moves, so that reward should be moved(back) there.
                In my implementation there may be some more cases with reward without a move, but we are not interested by now in better implementation.
                For long states it is not a problem, because after BB player has to make move >> gets reward,
                 so it is not possible to have a lot of states without ANY rewards
                
            """
            # remove such rewards
            for pIX in rewD:
                rIXL = list(rewD[pIX].keys())
                for rIX in rIXL:
                    if not rewD[pIX][rIX]:
                        rewD[pIX].pop(rIX)
                        self._done_states[pIX][rIX].reward = None

            for pIX in rewD: assert rewD[pIX] # safety check, possible only when updating to often (small range of states)

            # we do not need code below (we do a hard-reset of states after each update)
            """
            # remove already 'done' rewards (during previous update)
            for pIX in rewD:
                rIXL = sorted(list(rewD[pIX].keys()))
                for rIX in rIXL:
                    mIX = rewD[pIX][rIX][0] # any (0) move from this reward
                    if self._done_states[pIX][mIX].move_rew is None: break
                    else: rewD[pIX].pop(rIX)
            """

            # put shared reward to states
            for pIX in rewD:
                for rIX in rewD[pIX]:
                    sh_rew = self._done_states[pIX][rIX].reward / len(rewD[pIX][rIX]) # amount of reward shared among all moves
                    for mIX in rewD[pIX][rIX]:
                        self._done_states[pIX][mIX].move_rew = sh_rew

            # find ix of last rewarded move
            last_move_rew = []
            for pIX in self._done_states:
                for ix in reversed(range(len(self._done_states[pIX]))):
                    if self._done_states[pIX][ix].move_rew is not None:
                        last_move_rew.append(ix)
                        break
            if self.verb > 0: print('(@ _run_update) last_move_rew: min %d max %d avg %d' % (min(last_move_rew), max(last_move_rew), sum(last_move_rew)/len(last_move_rew)))
            last_move_rew = min(last_move_rew)

            # TODO: NN here
            cards_batch = []
            event_batch = []
            switch_batch = []
            state_batch = []
            correct_batch = []
            reward_batch = []
            for pIX in self._done_states:
                cards_seq = []
                event_seq = []
                switch_seq = []
                correct_seq = []
                reward_seq = []
                for state in self._done_states[pIX][:last_move_rew]:
                    val = state.value

                    switch = 1

                    cards = val['cards']
                    if not cards:
                        cards = []
                        switch = 0
                    cards += [52]*(7-len(cards))  # pads cards

                    event = val['event']

                    correct = state.move if state.move is not None else 0
                    reward = state.move_rew/500 if state.move_rew is not None else 0 #TODO: hardcoded 500

                    cards_seq.append(cards)
                    event_seq.append(event)
                    switch_seq.append([switch])
                    correct_seq.append(correct)
                    reward_seq.append(reward)

                cards_batch.append(cards_seq)
                event_batch.append(event_seq)
                switch_batch.append(switch_seq)
                correct_batch.append(correct_seq)
                reward_batch.append(reward_seq)
                state_batch.append(self.zero_state)

            feed = {
                self.mdl['cards_PH']:   cards_batch,
                self.mdl['train_PH']:   True,
                self.mdl['event_PH']:   event_batch,
                self.mdl['switch_PH']:  switch_batch,
                self.mdl['state_PH']:   state_batch,
                self.mdl['correct_PH']: correct_batch,
                self.mdl['rew_PH']:     reward_batch}

            fetches = [
                self.mdl['optimizer'],
                self.mdl['loss'],
                self.mdl['gg_norm']]
            _, loss, gn = self.mdl.session.run(fetches, feed_dict=feed)
            print(loss,gn)

            for pIX in self._done_states:
                # leave only from the next to last_move_rew
                # self._done_states[pIX] = self._done_states[pIX][last_move_rew+1:]
                self._done_states[pIX] = [] # hard-reset: clear all unused >> keeps the upd alg stable, but wastes some hands


    # overrides with update
    def make_decisions(self) -> List[tuple]:
        self._run_update()
        decL = RDMK.make_decisions(self)
        return decL

    # saves checkpoints
    def save(self): self.mdl.saver.save()

    # closes model session
    def close(self): self.mdl.session.close()