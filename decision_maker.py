"""

 2019 (c) piteren

 DMK (decision maker) is an object that makes decisions for poker players

    receives States in form of constant size vector
    for every state outputs Decision, every Decision is rewarded with cash

    first implementation: training time: 10.7sec/1KH = 93.4H/s

"""

import numpy as np
import random
import tensorflow as tf
import time

from pologic.podeck import PDeck

from putils.neuralmess.nemodel import NNModel


# basic implementation of DMK (random sampling)
class DMK:

    def __init__(
            self,
            name :str,              # name should be unique (@table)
            n_players=      100,    # number of managed players
            n_moves=        4,      # number of (all) moves supported by DM, has to match table/player
            rand_moveF=     0.2,    # how often move will be sampled from random
            stats_iv=       1000,   # stats interval
            run_TB=         False):

        self.name = name
        self.n_players = n_players
        self.n_moves = n_moves
        self.rand_move = rand_moveF

        # variables below store data for FWD and BWD batches and state evaluation, updated (by encState, runUpdate...)
        #TODO: do we need to init like that ???
        self.DMR =              {ix: [] for ix in range(self.n_players)} # dict of dicts {'decState': 'move': 'reward':}
        self.preflop =          {ix: True for ix in range(self.n_players)} # preflop indicator
        self.hand_pos =         {ix: [] for ix in range(self.n_players)} # positions @table of players from self.pls
        self.enc_states =       {} # encoded states save
        self.possible_moves =   {} # possible moves save
        self.hand_ix = 0 # number of hands

        self.stats = {} # stats of DMK (for all players)
        self.stats_iv = stats_iv
        self.chsd = {ix: None for ix in range(self.n_players)} # current hand stats data (per player)
        self._reset_stats()
        for pIX in range(self.n_players): self._reset_chsd(pIX)

        self.stored_hand = None
        self.store_next_hand = False
        self.store_hand_pix = -1 # player index to store hand

        self.rep_time = time.time()
        self.summ_writer = tf.summary.FileWriter(logdir='_models/' + self.name, flush_secs=10) if run_TB else None

    # resets self.cHSdata for player (per player stats)
    def _reset_chsd(
            self,
            pIX):

        self.chsd[pIX] = {
            'VPIP':     False,
            'PFR':      False,
            'HF':       False,
            'nPM':      0,
            'nAGG':     0}

    # resets stats (one stats for whole DMK)
    def _reset_stats(self):
        """
        by now implemented stats:
          VPIP  - Voluntarily Put $ in Pot %H; how many hands (%) player put money in pot (SB and BB do not count)
          PFR   - Preflop Raise %H; how many hands (%) player raised preflop
          HF    - Hands Folded; %H where player folds
          AGG   - Postflop Aggression Frequency %; (totBet + totRaise) / anyMove *100, only postflop
        """
        self.stats = {  # [total,interval]
            'nH':       [0,0],  # n hands played
            '$':        [0,0],  # $ won
            'nVPIP':    [0,0],  # n hands with VPIP
            'nPFR':     [0,0],  # n hands with PFR
            'nHF':      [0,0],  # n hands folded
            'nPM':      [0,0],  # n moves postflop
            'nAGG':     [0,0]}  # n aggressive moves postflop

    # table state encoder (into decState form - readable by DMC.getProbs method)
    # + updates some info (pos, pre/postflop table_state) in self variables
    def _enc_state(
            self,
            pIX :int,
            table_state_changes :list):

        # update storage
        if self.store_next_hand and self.store_hand_pix == pIX:
            self.stored_hand += table_state_changes

        for state in table_state_changes:
            key = list(state.keys())[0]

            # update positions of players @table for new hand, enter preflop
            if key == 'playersPC':

                # start storage
                if self.store_next_hand:
                    if self.stored_hand is None:
                        self.stored_hand = table_state_changes
                        self.store_hand_pix = pIX

                new_pos = [0]*len(state[key])
                for ix in range(len(state[key])):
                    new_pos[state[key][ix][0]] = ix
                self.hand_pos[pIX] = new_pos
                self.preflop[pIX] = True

            # enter postflop
            if key == 'newTableCards' and len(state[key]) == 3: self.preflop[pIX] = False

            # get reward and update
            if key == 'winnersData':

                if self.store_next_hand and self.store_hand_pix == pIX:
                    print('\nHand of %s from %s' % (self.name, time.strftime('%H.%M')))
                    for el in self.stored_hand: print(el)
                    self.stored_hand = None
                    self.store_next_hand = False
                    self.store_hand_pix = -1

                self.hand_ix += 1

                my_reward = 0
                for el in state[key]:
                    if el['pIX'] == 0: my_reward = el['won']

                # update reward backward
                n_div = 1
                for ix in reversed(range(len(self.DMR[pIX]))):
                    if self.DMR[pIX][ix]['reward'] is None: n_div += 1
                    else: break
                scaled_reward = my_reward / n_div
                for ix in reversed(range(len(self.DMR[pIX]))):
                    if self.DMR[pIX][ix]['reward'] is None: self.DMR[pIX][ix]['reward'] = scaled_reward
                    else: break

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

                # stats
                if self.stats['nH'][1] == self.stats_iv:

                    # reporting
                    if self.summ_writer:
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

                    for key in self.stats.keys(): self.stats[key][1] = 0 # reset interval values

                self.run_update()

        # custom implementation should add further dec_state preparation
        dec_state = None
        return dec_state

    # returns probabilities of for move in form of [(pIX,probs)...] or None
    def _calc_probs(self):

        """
        here goes custom implementation with:
         - more than random
         - multi player calculation >> probs will be calculated in batches
        """
        probsL = [(pIX, [1 / self.n_moves] * self.n_moves) for pIX in self.enc_states] # equal probs (base)
        # self.dec_states and self.possible_moves will be reset while returning decisions (@get_decisions)
        return probsL if probsL else None

    # updates current hand data for stats based on move performing
    def _upd_move_stats(
            self,
            pIX,    # player index
            move):  # player move

        if move == 0: self.chsd[pIX]['HF'] = True
        if self.preflop[pIX]:
            if move == 1 and self.hand_pos[pIX][0] != 1 or move > 1: self.chsd[pIX]['VPIP'] = True
            if move > 1: self.chsd[pIX]['PFR'] = True
        else:
            self.chsd[pIX]['nPM'] += 1
            if move > 1: self.chsd[pIX]['nAGG'] += 1

    # makes decisions based on state_changes - selects move from possible_moves using calculated probabilities
    # returns decisions in form of [(pIX,move)...] or None
    def get_decisions(self):

        p_probsL = self._calc_probs()

        # players probabilities list will be returned from time to time (return decisions then)
        decs = None
        if p_probsL is not None:
            decs = []
            for p_probs in p_probsL:
                pIX, probs = p_probs
                if random.random() < self.rand_move: probs = [1 / self.n_moves] * self.n_moves

                if self.possible_moves[pIX]:
                    prob_mask = [int(pM) for pM in self.possible_moves[pIX]]    # cast bool to int
                    probs = probs * np.asarray(prob_mask)                       # mask
                    if np.sum(probs) > 0: probs /= np.sum(probs)                # normalize
                else: probs = [1 / self.n_moves] * self.n_moves

                move = np.random.choice(np.arange(self.n_moves), p=probs) # sample from probs # TODO: for tournament should take max
                decs.append((pIX, move))

                # save state and move (for updates etc.)
                self.DMR[pIX].append({
                    'decState': self.enc_states[pIX],
                    'move':     move,
                    'reward':   None})

                self._upd_move_stats(pIX, move)  # stats

        # reset here
        self.enc_states = {}
        self.possible_moves = {}

        return decs

    # returns number of waiting decisions @DMK
    def get_waiting_num(self): return len(self.enc_states)

    # takes state from player, encodes and saves
    def take_player_state(
            self,
            pIX,
            state_changes,
            possible_moves):

        self.enc_states[pIX] = self._enc_state(pIX, state_changes)  # save encode table state
        self.possible_moves[pIX] = possible_moves                   # save possible moves

    # runs update of DMK based on saved: dec_states, moves and rewards
    def run_update(self): pass

# Base-Neural-DMK (for neural model)
class BaNeDMK(DMK):

    def __init__(
            self,
            fwd_func,
            mdict :dict,
            n_players=  100,
            rand_moveF= 0.01,
            n_mov_upd=  1000):  # number of moves between updates (bakprop)

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
                print('@@@ mt_seq',mt_seq,len(mv_seq[0]))
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
            print('@@@ n_mov',n_mov)
            print('@@@ n_rew',n_rew)
            avgn_rew = int(sum(n_rew)/len(n_rew))
            if avgn_rew: # exclude 0 case
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