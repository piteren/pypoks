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
            batch_factor=   0.3,    # sets batch size (n_players*batch_factor), for safety it should be lower than 1/(n_players_per_table)
            stats_iv=       1000,   # stats interval
            run_TB=         False):

        self.name = name
        self.n_players = n_players
        self.n_moves = n_moves
        self.rand_move = rand_moveF
        self.batch_size = int(n_players*batch_factor)

        # variables below store data for FWD and BWD batches and state evaluation, updated (by encState, runUpdate...)
        self.lDMR =         {ix: [] for ix in range(self.n_players)} # dict of dicts {'decState': 'move': 'reward':}
        self.preflop =      {ix: True for ix in range(self.n_players)} # preflop indicator
        self.plsHpos =      {ix: [] for ix in range(self.n_players)} # positions @table of players from self.pls
        self.psbl_moves =   {ix: [] for ix in range(self.n_players)} # possible moves save
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

                newPos = [0]*len(state[key])
                for ix in range(len(state[key])):
                    newPos[state[key][ix][0]] = ix
                self.plsHpos[pIX] = newPos
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
                for ix in reversed(range(len(self.lDMR[pIX]))):
                    if self.lDMR[pIX][ix]['reward'] is None: self.lDMR[pIX][ix]['reward'] = my_reward  # update reward
                    else: break

                for ti in [0,1]:
                    self.stats['nH'][ti] += 1
                    self.stats['$'][ti] += my_reward

                    # update self.sts with self.cHSdata
                    if self.chsd[pIX]['VPIP']:    self.stats['nVPIP'][ti] += 1
                    if self.chsd[pIX]['PFR']:     self.stats['nPFR'][ti] += 1
                    if self.chsd[pIX]['HF']:      self.stats['nHF'][ti] += 1
                    self.stats['nPM'][ti] += self.chsd[pIX]['nPM']
                    self.stats['nAGG'][ti] += self.chsd[pIX]['nAGG']
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

                    # reset interval values
                    for key in self.stats.keys():
                        self.stats[key][1] = 0

                self.run_update()

        # custom implementation should add further dec_state preparation
        dec_state = None
        return dec_state

    # returns probabilities of for move in form of [(pIX,probs)...] or None
    def _calc_probs(
            self,
            pIX,
            decState):

        """
        here goes custom implementation with:
         - more than random
         - multi player calculation >> probs will be calculated in batches
        """
        # equal probs
        return [(pIX, [1 / self.n_moves] * self.n_moves)]

    # updates current hand data for stats based on move performing
    def _upd_move_stats(
            self,
            pIX,    # player index
            move):  # player move

        if move == 0: self.chsd[pIX]['HF'] = True
        if self.preflop[pIX]:
            if move == 1 and self.plsHpos[pIX][0] != 1 or move > 1: self.chsd[pIX]['VPIP'] = True
            if move > 1: self.chsd[pIX]['PFR'] = True
        else:
            self.chsd[pIX]['nPM'] += 1
            if move > 1: self.chsd[pIX]['nAGG'] += 1

    # makes decisions based on stateChanges - selects move from possibleMoves using calculated probabilities
    # returns decisions in form of [(pIX,move)...] or None
    # TODO: prep mDec in tournament mode (no sampling from distribution, but max)
    def _make_dec(
            self,
            pIX :int,
            dec_state,
            possible_moves :list):

        pProbsL = self._calc_probs(pIX, dec_state)
        self.psbl_moves[pIX] = possible_moves # save possible moves

        # players probabilities list will be returned from time to time, same for decisions
        decs = None
        if pProbsL is not None:
            decs = []
            for pProbs in pProbsL:
                pIX, probs = pProbs
                if random.random() < self.rand_move: probs = [1 / self.n_moves] * self.n_moves

                probMask = [int(pM) for pM in self.psbl_moves[pIX]]
                probs = probs * np.asarray(probMask)
                if np.sum(probs) > 0: probs = probs / np.sum(probs)
                else: probs = [1 / self.n_moves] * self.n_moves

                move = np.random.choice(np.arange(self.n_moves), p=probs) # sample from probs
                decs.append((pIX, move))

                # save state and move (for updates etc.)
                self.lDMR[pIX].append({
                    'decState': dec_state,
                    'move':     move,
                    'reward':   None})

                self._upd_move_stats(pIX, move)  # stats

        return decs

    # takes player data and wraps stateEncoding+makingDecision
    def proc_player_data(
            self,
            p_addr,
            state_changes,
            possible_moves):

        dIX, pIX = p_addr
        decState = self._enc_state(pIX, state_changes)  # encode table state with DMK encoder
        decs = self._make_dec(pIX, decState, possible_moves) if possible_moves is not None else None
        return decs

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
        self.my_cards =         {n: None        for n in range(self.n_players)} # player+table cards
        self.dec_states =       {}

        self.n_mov_upd = n_mov_upd

    # prepares state in form of nn input
    def _enc_state(
            self,
            pIX: int,
            table_state_changes: list):

        super()._enc_state(pIX, table_state_changes)

        inMT = []  # list of moves (2)
        inV = []  # list of vectors (2)
        for state in table_state_changes:
            key = list(state.keys())[0]

            if key == 'playersPC':
                myCards = state[key][self.plsHpos[pIX][0]][1]
                self.my_cards[pIX] = [PDeck.cti(card) for card in myCards]

            if key == 'newTableCards':
                tCards = state[key]
                self.my_cards[pIX] += [PDeck.cti(card) for card in tCards]

            if key == 'moveData':
                who = state[key]['pIX'] # who moved
                if who: # my index is 0, so do not include my moves

                    inMT.append(int(state[key]['plMove'][0]))  # move type

                    vec = np.zeros(shape=self.mdl['mvW'])

                    vec[0] = who
                    vec[1] = self.plsHpos[pIX][who] # what position

                    vec[2] = state[key]['tBCash'] / 1500
                    vec[3] = state[key]['pBCash'] / 500
                    vec[4] = state[key]['pBCHandCash'] / 500
                    vec[5] = state[key]['pBCRiverCash'] / 500
                    vec[6] = state[key]['bCashToCall'] / 500
                    vec[7] = state[key]['tACash'] / 1500
                    vec[8] = state[key]['pACash'] / 500
                    vec[9] = state[key]['pACHandCash'] / 500
                    vec[10] = state[key]['pACRiverCash'] / 500

                    inV.append(vec)

        inC = [] + self.my_cards[pIX]
        while len(inC) < 7: inC.append(52) # pad cards
        while len(inMT) < 2*(len(self.plsHpos[pIX])-1): inMT.append(4) # pad moves
        while len(inV) < 2*(len(self.plsHpos[pIX])-1): inV.append(np.zeros(shape=self.mdl['mvW'])) # pad vectors

        nn_input = [inC], [inMT], [inV] # seq of values (seq because of shape)
        return nn_input

    # calculates probs with NN for batches
    def _calc_probs(
            self,
            pIX,        # player index
            dec_state):

        self.dec_states[pIX] = dec_state

        probsL = None
        if len(self.dec_states) == self.batch_size:

            pIXL = sorted(list(self.dec_states.keys())) # sorted list of pIX that will be processed

            inCbatch = []
            inMTbatch = []
            inVbatch = []
            statesBatch = []
            for pIX in pIXL:
                inCseq, inMTseq, inVseq = self.dec_states[pIX]
                inCbatch.append(inCseq)
                inMTbatch.append(inMTseq)
                inVbatch.append(inVseq)
                statesBatch.append(self.last_fwd_state[pIX])

            feed = {
                self.mdl['c_PH']:       inCbatch,
                self.mdl['mt_PH']:      inMTbatch,
                self.mdl['mv_PH']:      inVbatch,
                self.mdl['state_PH']:   statesBatch}

            fetches = [self.mdl['probs'], self.mdl['fin_state']]
            probs, fwd_states = self.mdl.session.run(fetches, feed_dict=feed)

            probs = np.reshape(probs, newshape=(probs.shape[0],probs.shape[-1])) # TODO: maybe any smarter way

            probsL = []
            for ix in range(fwd_states.shape[0]):
                pIX = pIXL[ix]
                probsL.append((pIX,probs[ix]))
                self.last_fwd_state[pIX] = fwd_states[ix]

            self.dec_states = {}

        return probsL

    # saves checkpoints
    def save(self): self.mdl.saver.save()

    # runs update of net (only when condition met)
    def run_update(self):

        n_mov = [len(self.lDMR[ix]) for ix in range(self.n_players)] # number of saved moves per player (not all rewarded)
        n_movAV = int(sum(n_mov)/len(n_mov)) # avg

        # does update
        updF = self.n_mov_upd / (self.n_players / 2)
        if n_movAV > updF:
            #print('min med max nM', min(nM), avgNM, max(nM))
            n_rew = [0]* self.n_players # factual num of rewarded moves
            for pix in range(self.n_players):
                for mix in reversed(range(len(self.lDMR[pix]))):
                    if self.lDMR[pix][mix]['reward'] is not None:
                        n_rew[pix] = mix
                        break
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
                        mDict = self.lDMR[pix][n_mov]
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
                    self.lDMR[pIX] = self.lDMR[pIX][avgn_rew:] # trim

                #print('%s :%4d: loss %.3f gN %.3f' % (self.name, len(uPIX)*avgNR, loss, gN))
                if self.summ_writer:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag='gph/0.loss', simple_value=loss)])
                    gn_sum = tf.Summary(value=[tf.Summary.Value(tag='gph/1.gn', simple_value=gn)])
                    self.summ_writer.add_summary(loss_sum, self.stats['nH'][0])
                    self.summ_writer.add_summary(gn_sum, self.stats['nH'][0])