"""

 2019 (c) piteren

 DMK (decision maker) is an object that makes decisions for poker players
 DMK assumes that:
    - player sends list of states
    - from time to time player sends list of possible moves,
        > it means that move should be chosen by DMK using received states
        > no new states will come until move will be made
        > table with player that waits for move is locked (no other player will send anything then...)

"""

from abc import ABC, abstractmethod
import numpy as np
import random
import tensorflow as tf
import time

from pologic.podeck import PDeck
from pologic.potable import POS_NMS

from putils.neuralmess.nemodel import NNModel

# abstract class of DMK, defines basic interface of DMK to act on PTable with PPlayer
class DMK(ABC):

    def __init__(
            self,
            name :str,          # name should be unique (@table)
            n_players :int,     # number of managed players
            n_moves :int=   4): # number of (all) moves supported by DM, has to match table/player

        self.name = name
        self.n_players = n_players
        self.n_moves = n_moves

        # variables below store moves/decisions data
        self._new_states =      {}  # dict of new states                    {state: } # should not have empty keys (player indexes)
        self._n_waiting =       0 # number of players (@this DMK) actually waiting for move
        self._done_states =     {ix: [] for ix in range(self.n_players)}  # dict of processed (with decision) states

    # encodes table states into DMK form (appropriate to make decisions)
    # TO BE OVERRIDDEN
    def _enc_states(
            self,
            pIX: int,
            table_state_changes: list) -> list: return table_state_changes # baseline does nothing

    # takes player states, encodes and saves in self._new_states
    def take_states(
            self,
            pIX,
            state_changes: list):

        if pIX not in self._new_states: self._new_states[pIX] = []
        self._new_states[pIX] += [{'state': state} for state in self._enc_states(pIX, state_changes)]

    # takes player possible_moves and saves
    def take_possible_moves(
            self,
            pIX,
            possible_moves :list):

        self._n_waiting += 1
        assert pIX in self._new_states # # TODO: to remove, develop safety check, player should have new states while getting possible moves
        self._new_states[pIX][-1]['possible_moves'] = possible_moves # append to last new state

    # returns number of players waiting
    def num_waiting(self) -> int:
        return self._n_waiting

    # should return list of decisions in form: [(pIX,move)...], cannot return empty list
    # decision should be made based on stored data
    @abstractmethod
    def _mdec(self) -> list: pass

    # returns decisions
    def make_decisions(self) -> list:

        decL = self._mdec()
        assert decL
        for dec in decL:
            pIX = dec[0]
            self._done_states[pIX].append(self._new_states.pop(pIX)) # move new into done (where decisions made)
        self._n_waiting -= len(decL)
        return decL

# random DMK implementation (first baseline), introduces move probabilities
class RDMK(DMK):

    def __init__(
            self,
            verb=   0,
            **kwargs):

        super(RDMK, self).__init__(**kwargs)
        self.verb = verb

    # samples move for given probabilities and possibilities
    def __sample_move(
            self,
            probs :list,
            possible_moves :list):

        prob_mask = [int(pM) for pM in possible_moves]  # cast bool to int
        prob_mask = np.asarray(prob_mask)
        probs = probs * prob_mask  # mask probs
        if np.sum(probs) == 0: probs = prob_mask  # take mask if no intersection
        probs /= np.sum(probs)  # normalize
        move = np.random.choice(np.arange(self.n_moves), p=probs)  # sample from probs # TODO: for tournament should take max
        return move

    # returns list of decisions
    def _mdec(self) -> list:

        decL = []
        for pIX in self._new_states:
            if 'possible_moves' in self._new_states[pIX][-1] and 'probs' in self._new_states[pIX][-1]:
                possible_moves =    self._new_states[pIX][-1]['possible_moves']
                probs =             self._new_states[pIX][-1]['probs']
                move =              self.__sample_move(probs, possible_moves)
                decL.append([pIX, move])
        return decL

    # add probabilities for at least some states with possible_moves, baseline sets for all
    # TO BE OVERRIDEN
    def _calc_probs(self):
        baseline_probs = [1/self.n_moves] * self.n_moves # equal probs
        for pIX in self._new_states:
            if 'possible_moves' in self._new_states[pIX][-1]:
                self._new_states[pIX][-1]['probs'] = baseline_probs

    # returns decisions
    def make_decisions(self) -> list:
        self._calc_probs()
        decL = super().make_decisions()
        if self.verb > 0: print(' > made %d decisions'%len(decL))
        return decL

# implementation of DMK
# + states encoding
# + stats with TB
class DMKT(DMK):

    def __init__(
            self,
            rand_moveF=     0.2,    # how often move will be sampled from random
            verb=           0,
            **kwargs):

        super(DMKT, self).__init__(**kwargs)
        self.verb = verb
        self.rand_move = rand_moveF
        if self.verb>0: print(' *** DMKT *** initialized, name %s, players %d'%(self.name, self.n_players))

        # variables below store moves/decisions data
        # TODO: store processed states
        self._prob_states = {}  # intermediate dict of states with calculated probs   [{'state': 'probs':}]
        self.SMR =              {ix: [] for ix in range(self.n_players)} # dict of lists of dicts {'state': 'move': 'reward':}
        self.my_cards =         {ix: [] for ix in range(self.n_players)} # current cards of player

    # TO BE CUSTOM IMPLEMENTED
    # it is a baseline, encodes only selection of states(5) into list of {type: POS,PLC,MOV,REW, value: }
    # updates self.my_cards
    def _enc_states(
            self,
            pIX :int,
            table_state_changes :list):

        if self.verb > 1:
            print('\n > states:')
            for s in table_state_changes: print(s)

        es = []
        for s in table_state_changes:

            ns = None
            if s[0] == 'POS':
                if s[1][0] == 0: # my position
                    ns = {
                        'type':     'POS',
                        'value':    s[1][1]} # SB, BB ...

            if s[0] == 'PLH':
                if s[1][0] == 0:
                    self.my_cards[pIX] = s[1][1:] # set cards
                    ns = {
                        'type':     'PLC',
                        'value':    [] + self.my_cards[pIX]} # list of cards in str form

            if s[0] == 'MOV':
                if s[1][0] != 0: # do not take my moves
                    ns = {
                        'type':     'MOV',
                        'value':    (s[1][0],s[1][1])} # player id 1,2..., just move type (no value - baseline)

            if s[0] == 'TCD':
                self.my_cards[pIX] += s[1]
                ns = {
                    'type':         'PLC',
                    'value':        [] + self.my_cards[pIX]} # list of cards in str form

            if s[0] == 'PRS':
                if s[1][0] == 0:
                    self.my_cards[pIX] = []  # reset my cards
                    ns = {
                        'type':     'REW',
                        'value':    s[1][1]} # reward value

            if ns: es.append(ns)

        if self.verb > 1:
            print('\n > encoded states:')
            for s in es: print(s)

        return es # this list is not empty, but may be without an error

    # calculates probabilities for selected row of states, TO BE CUSTOM IMPLEMENTED
    def _calc_probs_row(self, states_row):
        probs = [1 / self.n_moves] * self.n_moves # equal probs for all (base)
        sp_row = [{
            'pIX':      s['pIX'],
            'state':    s['state'],
            'probs':    probs} for s in states_row]
        return sp_row

    # calculates probabilities for not processed states (not all...)
    def _calc_probs(self):

        # build row of states
        states_row = [{
            'pIX':      pIX,
            'state':    self._new_states[pIX].pop(-1)} for pIX in self._new_states]
        if self.verb>1: print(' > states_row:',len(states_row))
        pIXL = list(self._new_states.keys())
        # remove from self._new_states empty lists now
        for pIX in pIXL:
            if not self._new_states[pIX]: # empty list
                self._new_states.pop(pIX) # delete key
        if self.verb>1: print(' > cleared states:',len(states_row)-len(self._new_states))

        sp_row = self._calc_probs_row(states_row)
        for sp in sp_row:
            pIX =   sp['pIX']
            if pIX not in self._prob_states: self._prob_states[pIX] = []
            self._prob_states[pIX].append({
                'state': sp['state'],
                'probs': sp['probs']})

    # samples move for given probabilities and possibilities
    def __select_move(
            self,
            probs :list,
            possible_moves :list):

        if random.random() < self.rand_move: probs = [1/self.n_moves] * self.n_moves  # force random move
        prob_mask = [int(pM) for pM in possible_moves]  # cast bool to int
        prob_mask = np.asarray(prob_mask)
        probs = probs * prob_mask  # mask probs
        if np.sum(probs) == 0: probs = prob_mask  # take mask if no intersection
        probs /= np.sum(probs)  # normalize
        move = np.random.choice(np.arange(self.n_moves), p=probs)  # sample from probs # TODO: for tournament should take max
        return move

    # makes decisions
    def _mdec(self) -> list:

        decL = []
        if self.verb > 1: print('\n%s gets decisions' % self.name)
        self._calc_probs()
        pIXL = list(self._possible_moves.keys())  # players with possible_moves
        if self.verb > 1: print(' > possible_moves:', len(pIXL))
        for pIX in pIXL:
            if pIX in self._prob_states and pIX not in self._new_states:  # has processed states and hasn't not-processed states
                sp = self._prob_states.pop(pIX)
                move = self.__select_move(
                    probs=          sp[-1]['probs'],  # take last probs
                    possible_moves= self._possible_moves[pIX])
                decL.append((pIX, move))

        # TODO: store states and decisions
        if self.verb > 1: print(' > decisions:', len(decL))
        return decL

    # runs update of DMK based on saved SPMR (state,move,reward)
    def _run_update(self):
        # reset only (base)
        for pIX in self.SMR:
            if self.SMR[pIX]:
                self.SMR[pIX] = [self.SMR[pIX][-1]]

# DMKT with stats, works for 4 move types
class STATS_DMKT(DMKT):

    def __init__(
            self,
            stats_iv=   1000,  # stats interval
            **kwargs):

        super(STATS_DMKT, self).__init__(**kwargs)

        self.stats_iv = stats_iv
        self.stats = {} # stats of DMK (for all players)
        self.chsd = {ix: None for ix in range(self.n_players)} # current hand stats data (per player)
        self.is_BB = {} # BB position of player at table {pIX: True/False}

        self.reset_stats()
        for pIX in range(self.n_players): self._reset_chsd(pIX)

        self.summ_writer = tf.summary.FileWriter(logdir='_models/' + self.name, flush_secs=10)

    # resets stats (one stats for whole DMK)
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
    def _reset_chsd(
            self,
            pIX):

        self.chsd[pIX] = {
            'VPIP':     False,
            'PFR':      False,
            'HF':       False,
            'nPM':      0,      # num of postflop moves
            'nAGG':     0}

    # updates self.chsd with given player move (after making decisions)
    def _upd_chsd(
            self,
            pIX,
            move):

        if move == 0: self.chsd[pIX]['HF'] = True
        if len(self.my_cards[pIX]) < 3: # preflop
            if move == 1 and not self.is_BB[pIX] or move > 1: self.chsd[pIX]['VPIP'] = True
            if move > 1: self.chsd[pIX]['PFR'] = True
        else:
            self.chsd[pIX]['nPM'] += 1
            if move > 1: self.chsd[pIX]['nAGG'] += 1

    # puts DMK stats to TB
    def _push_TB(self):
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
    def _extract_stats(
            self,
            pIX,
            encoded_states):

        for s in encoded_states:
            if s['type'] == 'POS':
                self.is_BB[pIX] = s['value'] = 'BB'
            if s['type'] == 'REW':
                my_reward = s['value']
                for ti in [0,1]:
                    self.stats['nH'][ti] += 1
                    self.stats['$'][ti] += my_reward

                    # update self.sts with self.cHSdata
                    if self.chsd[pIX]['VPIP']:  self.stats['nVPIP'][ti] += 1
                    if self.chsd[pIX]['PFR']:   self.stats['nPFR'][ti] += 1
                    if self.chsd[pIX]['HF']:    self.stats['nHF'][ti] += 1
                    self.stats['nPM'][ti] +=    self.chsd[pIX]['nPM']
                    self.stats['nAGG'][ti] +=   self.chsd[pIX]['nAGG']

                self._reset_chsd(pIX)

                # put stats on TB
                if self.stats['nH'][1] == self.stats_iv:
                    self._push_TB()
                    for key in self.stats.keys(): self.stats[key][1] = 0 # reset interval values

    # overrides with stats (adds nothing to states)
    def _enc_states(
            self,
            pIX :int,
            table_state_changes :list):

        es = super()._enc_states(pIX, table_state_changes)
        self._extract_stats(pIX,es)
        return es

    # overrides with updating self.chsd
    def make_decisions(self):
        decL = super().make_decisions()
        for dec in decL: self._upd_chsd(*dec)
            #TODO: delete if above works
            #pIX, move = dec
            #self._upd_chsd(pIX,move)
        return decL

# Neural DMKT (with stats)
class NSDMKT(STATS_DMKT):

    def __init__(
            self,
            fwd_func,
            **kwargs):

        super(NSDMKT, self).__init__(**kwargs)

        self.mdl = NNModel(
            fwd_func=   fwd_func,
            mdict=      {'name':self.name},
            devices=    1,
            verb=       0)

        zero_state = self.mdl.session.run(self.mdl['single_zero_state'])
        self.last_fwd_state = {n: zero_state for n in range(self.n_players)}  # net state after last fwd

    # prepares state in form of nn input
    def _enc_states(
            self,
            pIX: int,
            table_state_changes: list):

        es = super()._enc_states(pIX, table_state_changes)
        news = [] # newly encoded states
        for s in es:
            ns = None
            if s['type'] == 'REW':
                # TODO: put it back somewhere
                pass
            if s['type'] == 'POS':
                ns = {
                    'cards':    None,
                    'switch':   0,
                    'move':     1+POS_NMS[3].index(s['value'])} # TODO: hardcoded 3 here
            if s['type'] == 'PLC':
                ns = {
                    'cards':    s['value'],
                    'switch':   1,
                    'move':     0}
            if s['type'] == 'MOV':
                ns = {}
            if ns: news.append(ns)

        if not news: print('@@@ got empty new states') # TODO: empty news may appear here
        return news


# Base-Neural-DMK (for neural model)
class BaNeDMK(DMK):

    def __init__(
            self,
            fwd_func,
            mdict :dict,
            n_players=  100,
            rand_moveF= 0.01,
            n_mov_upd=  1000):  # number of moves between updates (backprop)

        super().__init__(
            name=       mdict['name'],
            n_players=  n_players,
            rand_moveF= rand_moveF,
            run_TB=     True)

        self.mdl = NNModel(
            fwd_func=   fwd_func,
            mdict=      mdict,
            devices=    1,
            verb=       0)

        zero_state = self.mdl.session.run(self.mdl['single_zero_state'])
        self.last_fwd_state =   {n: zero_state for n in range(self.n_players)} # net state after last fwd
        self.last_upd_state =   {n: zero_state for n in range(self.n_players)} # net state after last update
        self.my_cards =         {n: None       for n in range(self.n_players)} # player+table cards

        self.n_mov_upd = n_mov_upd

    # prepares state in form of nn input
    def _enc_state(
            self,
            pIX: int,
            table_state_changes: list):

        super()._enc_state(pIX, table_state_changes)

        mt = []  # list of moves (4)
        mv = []  # list of vectors (4)
        for state in table_state_changes:
            key = list(state.keys())[0]

            if key == 'playersPC':
                myCards = state[key][self.hand_pos[pIX][0]][1]
                self.my_cards[pIX] = [PDeck.cti(card) for card in myCards]

            if key == 'newTableCards':
                tCards = state[key]
                self.my_cards[pIX] += [PDeck.cti(card) for card in tCards]

            if key == 'moveData':
                who = state[key]['pIX'] # who moved
                if who: # my index is 0, so do not include my moves

                    mt.append(int(state[key]['plMove'][0]))  # move type

                    vec = np.zeros(shape=self.mdl['mvW'])

                    vec[0] = who
                    vec[1] = self.hand_pos[pIX][who] # what position

                    vec[2] = state[key]['tBCash'] / 1500
                    vec[3] = state[key]['pBCash'] / 500
                    vec[4] = state[key]['pBCHandCash'] / 500
                    vec[5] = state[key]['pBCRiverCash'] / 500
                    vec[6] = state[key]['bCashToCall'] / 500
                    vec[7] = state[key]['tACash'] / 1500
                    vec[8] = state[key]['pACash'] / 500
                    vec[9] = state[key]['pACHandCash'] / 500
                    vec[10] = state[key]['pACRiverCash'] / 500

                    mv.append(vec)

        inC = [] + self.my_cards[pIX]
        while len(inC) < 7: inC.append(52) # pad cards
        # TODO: what "the padding" here:
        while len(mt) < 2*(len(self.hand_pos[pIX])-1): mt.append(4) # pad moves
        while len(mv) < 2*(len(self.hand_pos[pIX])-1): mv.append(np.zeros(shape=self.mdl['mvW'])) # pad vectors

        nn_input = [inC], [mt], [mv] # seq of values (seq because of shape)
        return nn_input

    # calculates probs with NN for batches
    def _calc_probs(self):

        probsL = []
        pIXL = sorted(list(self.enc_states.keys())) # sorted list of pIX that will be processed
        if pIXL:

            c_batch = []
            mt_batch = []
            mv_batch = []
            state_batch = []
            for pIX in pIXL:
                c_seq, mt_seq, mv_seq = self.enc_states[pIX]
                c_batch.append(c_seq)
                mt_batch.append(mt_seq)
                mv_batch.append(mv_seq)
                state_batch.append(self.last_fwd_state[pIX])

            feed = {
                self.mdl['c_PH']:       c_batch,
                self.mdl['mt_PH']:      mt_batch,
                self.mdl['mv_PH']:      mv_batch,
                self.mdl['state_PH']:   state_batch}

            fetches = [self.mdl['probs'], self.mdl['fin_state']]
            probs, fwd_states = self.mdl.session.run(fetches, feed_dict=feed)

            probs = np.reshape(probs, newshape=(probs.shape[0],probs.shape[-1])) # TODO: maybe any smarter way

            for ix in range(fwd_states.shape[0]):
                pIX = pIXL[ix]
                probsL.append((pIX,probs[ix]))
                self.last_fwd_state[pIX] = fwd_states[ix]

        return probsL if probsL else None

    # saves checkpoints
    def save(self): self.mdl.saver.save()

    # runs update of net (only when condition met)
    def run_update(self):

        n_mov = [len(self.DMR[ix]) for ix in range(self.n_players)] # number of saved moves per player (not all rewarded)
        n_movAV = int(sum(n_mov)/len(n_mov)) # avg

        # does update
        updF = self.n_mov_upd / (self.n_players / 2)
        if n_movAV > updF:
            #print('min med max nM', min(nM), avgNM, max(nM))
            n_rew = [0]* self.n_players # factual num of rewarded moves
            for pix in range(self.n_players):
                for mix in reversed(range(len(self.DMR[pix]))):
                    if self.DMR[pix][mix]['reward'] is not None:
                        n_rew[pix] = mix
                        break
            #print('@@@ n_mov',n_mov)
            #print('@@@ n_rew',n_rew)
            avgn_rew = int(sum(n_rew)/len(n_rew))
            if avgn_rew: # exclude 0 case
                print('@@@ %s updating'%self.name)
                #print('min med max nR', min(nR), avgNR, max(nR))
                uPIX = [ix for ix in range(self.n_players) if n_rew[ix] >= avgn_rew]
                #print('len(uPIX)', len(uPIX))
                #print('upd size:',len(uPIX)*avgNR)

                # build batches of data
                inCbatch = []
                inMTbatch = []
                inVbatch = []
                moveBatch = []
                rewBatch = []
                for pix in uPIX:
                    inCseq = []
                    inMTseq = []
                    inVseq = []
                    moveSeq = []
                    rewSeq = []
                    for n_mov in range(avgn_rew):
                        mDict = self.DMR[pix][n_mov]
                        decState = mDict['decState']
                        inCseq += decState[0]
                        inMTseq += decState[1]
                        inVseq += decState[2]
                        moveSeq.append(mDict['move'])
                        #rew = 1 if mDict['reward'] > 0 else -1
                        #if mDict['reward'] == 0: rew = 0
                        rew = mDict['reward']
                        rewSeq.append(rew)
                    inCbatch.append(inCseq)
                    inMTbatch.append(inMTseq)
                    inVbatch.append(inVseq)
                    moveBatch.append(moveSeq)
                    rewBatch.append(rewSeq)
                statesBatch = [self.last_upd_state[ix] for ix in uPIX]  # build directly from dict of Upd states

                feed = {
                    self.mdl['c_PH']:       inCbatch,
                    self.mdl['mt_PH']:      inMTbatch,
                    self.mdl['mv_PH']:      inVbatch,
                    self.mdl['cmv_PH']:     moveBatch,
                    self.mdl['rew_PH']:     rewBatch,
                    self.mdl['state_PH']:   statesBatch}

                fetches = [
                    self.mdl['optimizer'],
                    self.mdl['loss'],
                    self.mdl['gg_norm'],
                    self.mdl['fin_state']]
                _, loss, gn, upd_states = self.mdl.session.run(fetches, feed_dict=feed)

                for ix in range(len(uPIX)):
                    pIX = uPIX[ix]
                    self.last_upd_state[pIX] = upd_states[ix] # save states
                    self.DMR[pIX] = self.DMR[pIX][avgn_rew:] # trim

                #print('%s :%4d: loss %.3f gN %.3f' % (self.name, len(uPIX)*avgNR, loss, gN))
                if self.summ_writer:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag='gph/0.loss', simple_value=loss)])
                    gn_sum = tf.Summary(value=[tf.Summary.Value(tag='gph/1.gn', simple_value=gn)])
                    self.summ_writer.add_summary(loss_sum, self.stats['nH'][0])
                    self.summ_writer.add_summary(gn_sum, self.stats['nH'][0])

    # closes BaNeDMK (as much as possible)
    def close(self): self.mdl.session.close()
