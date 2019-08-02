"""

 2019 (c) piteren

"""

from multiprocessing import Process, Queue
import numpy as np
import random
import tensorflow as tf
import time

from pUtils.littleTools.littleMethods import shortSCIN
from pUtils.nnTools.nnBaseElements import defInitializer, layDENSE, numVFloats

from pLogic.pDeck import PDeck

# TODO:
#   - implement simple algorithmic DMK


# basic implementation of DMK (random sampling)
class DMK:

    def __init__(
            self,
            name :str,          # name should be unique (@table)
            nPl=        100,    # number of managed players
            nMoves=     4,      # number of (all) moves supported by DM, has to match table/player
            randMove=   0.2,    # how often move will be sampled from random
            runTB=      False):

        self.name = name
        self.nMoves = nMoves
        self.randMove = randMove

        self.nPl = nPl
        # variables below store data for FWD and BWD batches and state evaluation, are permanently updated (by encState, runUpdate...)
        self.lDMR = {ix: [] for ix in range(self.nPl)}  # list of dicts {'decState': 'move': 'reward':}
        self.preflop = {ix: True for ix in range(self.nPl)} # preflop indicator
        self.plsHpos = {ix: [] for ix in range(self.nPl)} # positions @table of players from self.pls
        self.psblMoves = {ix: [] for ix in range(self.nPl)} # possible moves save
        self.nH = 0 # number of hands

        self.runTB = runTB
        self.summWriter = tf.summary.FileWriter(logdir='_nnTB/' + self.name, flush_secs=10) if runTB else None

        self.repTime = time.time()
        self.stsV = 1000 # stats interval
        self.sts = {} # stats
        self.cHSdata = {} # current hand data for stats
        self._resetSTS()
        self._resetCSHD()

    # resets self.cHSdata
    def _resetCSHD(self):
        self.cHSdata = {
            'VPIP':     False,
            'PFR':      False,
            'SH':       False,
            'nPM':      0,
            'nAGG':     0}

    # resets stats
    def _resetSTS(self):
        """
        by now implemented stats:
          VPIP - Voluntarily Put $ in Pot %H; how many hands (%) player put money in pot (SB and BB do not count)
          PFR - Preflop Raise %H; how many hands (%) player raised preflop
          SH - Stacked Hands; %H where player stacked
          AGG - Postflop Aggression Frequency %; (totBet + totRaise) / anyMove *100, only postflop
        """
        self.sts = {
                      # [total,interval]
            'nH':       [0,0],  # n hands played
            '$':        [0,0],  # $ won
            'nVPIP':    [0,0],  # n hands with VPIP
            'nPFR':     [0,0],  # n hands with PFR
            'nSH':      [0,0],  # n hands stacked
            'nPM':      [0,0],  # n moves postflop
            'nAGG':     [0,0],  # n aggressive moves postflop
        }

    """
    # resets knowledge, stats, name of DMK
    def resetME(self, newName=None):

        if newName: self.name = newName
        self._resetSTS()
        if self.runTB: self.summWriter = tf.summary.FileWriter(logdir='_nnTB/' + self.name, flush_secs=10)
    """

    # table state encoder (into decState form - readable by DMC.getProbs method)
    # + updates some info (pos, pre/postflop tableState) in self variables
    def _encState(
            self,
            pIX :int,
            tableStateChanges :list):

        for state in tableStateChanges:
            key = list(state.keys())[0]

            # update positions of players @table for new hand, enter preflop
            if key == 'playersPC':
                newPos = [0]*len(state[key])
                for ix in range(len(state[key])):
                    newPos[state[key][ix][0]] = ix
                self.plsHpos[pIX] = newPos
                self.preflop[pIX] = True
            # enter postflop
            if key == 'newTableCards' and len(state[key]) == 3: self.preflop[pIX] = False

            # get reward and update
            if key == 'winnersData':

                self.nH += 1
                if self.nH % 1000 == 0:
                    print(' >>> running time: %.1fsec/%d' % (time.time() - self.repTime, 1000))
                    self.repTime = time.time()

                myReward = 0
                for el in state[key]:
                    if el['pIX'] == 0: myReward = el['won']

                # update reward backward
                for ix in reversed(range(len(self.lDMR[pIX]))):
                    if self.lDMR[pIX][ix]['reward'] is None: self.lDMR[pIX][ix]['reward'] = myReward  # update reward
                    else: break

                self.sts['nH'][0] += 1
                self.sts['nH'][1] += 1
                self.sts['$'][0] += myReward
                self.sts['$'][1] += myReward

                # update self.sts with self.cHSdata
                if self.cHSdata['VPIP']:
                    self.sts['nVPIP'][0] += 1
                    self.sts['nVPIP'][1] += 1
                if self.cHSdata['PFR']:
                    self.sts['nPFR'][0] += 1
                    self.sts['nPFR'][1] += 1
                if self.cHSdata['SH']:
                    self.sts['nSH'][0] += 1
                    self.sts['nSH'][1] += 1
                self.sts['nPM'][0] += self.cHSdata['nPM']
                self.sts['nPM'][1] += self.cHSdata['nPM']
                self.sts['nAGG'][0] += self.cHSdata['nAGG']
                self.sts['nAGG'][1] += self.cHSdata['nAGG']
                self._resetCSHD()

                # sts reporting procedure
                def repSTS(V=False):

                    ix, cx = (0, 'T') if not V else (1, 'V')  # interval or total

                    won = tf.Summary(value=[tf.Summary.Value(tag='sts%s/0_won$' % cx, simple_value=self.sts['$'][ix])])
                    vpip = self.sts['nVPIP'][ix] / self.sts['nH'][ix] * 100
                    vpip = tf.Summary(value=[tf.Summary.Value(tag='sts%s/1_VPIP' % cx, simple_value=vpip)])
                    pfr = self.sts['nPFR'][ix] / self.sts['nH'][ix] * 100
                    pfr = tf.Summary(value=[tf.Summary.Value(tag='sts%s/2_PFR' % cx, simple_value=pfr)])
                    sh = self.sts['nSH'][ix] / self.sts['nH'][ix] * 100
                    sh = tf.Summary(value=[tf.Summary.Value(tag='sts%s/4_sh' % cx, simple_value=sh)])
                    agg = self.sts['nAGG'][ix] / self.sts['nPM'][ix] * 100 if self.sts['nPM'][ix] else 0
                    agg = tf.Summary(value=[tf.Summary.Value(tag='sts%s/3_AGG' % cx, simple_value=agg)])
                    self.summWriter.add_summary(won, self.sts['nH'][0])
                    self.summWriter.add_summary(vpip, self.sts['nH'][0])
                    self.summWriter.add_summary(pfr, self.sts['nH'][0])
                    self.summWriter.add_summary(sh, self.sts['nH'][0])
                    self.summWriter.add_summary(agg, self.sts['nH'][0])

                    # reset interval values
                    if V:
                        for key in self.sts.keys():
                            self.sts[key][1] = 0
                if self.summWriter and self.sts['nH'][1] % self.stsV == 0:  repSTS(True)
                if self.summWriter and self.sts['nH'][0] % 1000 == 0:       repSTS()

                self.runUpdate()

        # custom implementation should add further decState preparation
        decState = None
        return decState

    # returns probabilities of for move in form of [(pIX,probs)...] or None
    def _calcProbs(
            self,
            pIX,
            decState):

        """
        here goes custom implementation with:
         - more than random
         - multi player calculation >> probs will be calculated in batches
        """
        # equal probs
        return [(pIX, [1/self.nMoves]*self.nMoves)]

    # updates current hand data for stats based on move performing
    def _updMoveStats(
            self,
            pIX,    # player index
            move):  # player move

        if move == 3: self.cHSdata['SH'] = True
        if self.preflop[pIX]:
            if move == 1 and self.plsHpos[pIX][0] != 1 or move > 1: self.cHSdata['VPIP'] = True
            if move > 1: self.cHSdata['PFR'] = True
        else:
            self.cHSdata['nPM'] += 1
            if move > 1: self.cHSdata['nAGG'] += 1

    # makes decisions based on stateChanges - selects move from possibleMoves using calculated probabilities
    # returns decisions in form of [(pIX,move)...] or None
    # TODO: prep mDec in tournament mode (no sampling from distribution, but max)
    def _makeDec(
            self,
            pIX :int,
            decState,
            possibleMoves :list):

        pProbsL = self._calcProbs(pIX, decState)
        self.psblMoves[pIX] = possibleMoves # save possible moves

        # players probabilities list will be returned from time to time, same for decisions
        decs = None
        if pProbsL is not None:
            decs = []
            for pProbs in pProbsL:
                pIX, probs = pProbs
                if random.random() < self.randMove: probs = [1/self.nMoves] * self.nMoves

                probMask = [int(pM) for pM in self.psblMoves[pIX]]
                probs = probs * np.asarray(probMask)
                if np.sum(probs) > 0: probs = probs / np.sum(probs)
                else: probs = [1/self.nMoves] * self.nMoves

                move = np.random.choice(np.arange(self.nMoves), p=probs) # sample from probs
                decs.append((pIX, move))

                # save state and move (for updates etc.)
                self.lDMR[pIX].append({
                    'decState': decState,
                    'move':     move,
                    'reward':   None})

                self._updMoveStats(pIX, move)  # stats

        return decs

    # takes player data and wraps stateEncoding+makingDecision
    def procPLData(
            self,
            pAddr,
            stateChanges,
            possibleMoves):

        dix, pix = pAddr
        decState = self._encState(pix, stateChanges)  # encode table state with DMK encoder
        decs = self._makeDec(pix, decState, possibleMoves) if possibleMoves is not None else None
        return decs

    # runs update of DMK based saved decStates, moves and rewards
    def runUpdate(self): pass

# Base-Neural-DMK (LSTM with single-state to decision)
class BNDMK(DMK):

    def __init__(
            self,
            session :tf.Session,
            name :str,
            nPl=        100,
            randMove=   0.2):

        super().__init__(
            name=       name,
            nPl=        nPl,
            randMove=   randMove,
            runTB=      True)

        self.session = session

        self.wC = 16        # card (single) emb width
        self.wMT = 4        # move type emb width
        self.wV = 11        # values vector width, holds position @table and cash values
        self.cellW = 188    # cell width

        self._buildGraph()

        zeroState = self.session.run(self.singleZeroState)#.tolist()
        self.lastFwdState = {ix: zeroState  for ix in range(self.nPl)} # netState after last fwd
        self.lastUpdState = {ix: zeroState  for ix in range(self.nPl)} # netState after last update
        self.myCards =      {ix: None       for ix in range(self.nPl)} # player+table cards
        self.decStates =    {}

        self.session.run(tf.initializers.variables(var_list=self.vars+self.optVars))
        self.nHF = 0 # num of forward predictions

    # builds NN graph
    def _buildGraph(self):

        with tf.variable_scope('snnDMK_%s'%self.name):

            self.inC = tf.placeholder( # 3 cards
                name=           'inC',
                dtype=          tf.int32,
                shape=          [None,None,7]) # [bsz,seq,7cards]

            cEMB = tf.get_variable( # cards embeddings
                name=           'cEMB',
                shape=          [53,self.wC], # one card for 'no_card'
                dtype=          tf.float32,
                initializer=    defInitializer())

            self.inMT =         tf.placeholder( # event type
                name=           'inMT',
                dtype=          tf.int32,
                shape=          [None,None,4]) # [bsz,seq,2*2oppon]

            mtEMB = tf.get_variable( # event type embeddings
                name=           'mtEMB',
                shape=          [5,self.wMT], # 4 moves + no_move
                dtype=          tf.float32,
                initializer=    defInitializer())

            self.inV = tf.placeholder( # event float values
                name=           'inV',
                dtype=          tf.float32,
                shape=          [None,None,4,self.wV]) # [bsz,seq,2*2,vec]

            self.move = tf.placeholder( # "correct" move (class)
                name=           'move',
                dtype=          tf.int32,
                shape=          [None,None]) # [bsz,seq]

            self.reward = tf.placeholder( # reward for "correct" move
                name=           'reward',
                dtype=          tf.float32,
                shape=          [None,None]) # [bsz,seq]

            inCemb = tf.nn.embedding_lookup(params=cEMB, ids=self.inC)
            print(' > inCemb:', inCemb)
            inCemb = tf.unstack(inCemb, axis=-2)
            inCemb = tf.concat(inCemb, axis=-1)
            print(' > inCemb (flattened):', inCemb)

            inMTemb = tf.nn.embedding_lookup(params=mtEMB, ids=self.inMT)
            print(' > inMTemb:', inMTemb)
            inMTemb = tf.unstack(inMTemb, axis=-2)
            inMTemb = tf.concat(inMTemb, axis=-1)
            print(' > inMTemb (flattened):', inMTemb)

            inV = tf.unstack(self.inV, axis=-2)
            inV = tf.concat(inV, axis=-1)
            print(' > inV (flattened):', inV)

            input = tf.concat([inCemb, inMTemb, inV], axis=-1)
            print(' > input (concatenated):', input) # width = self.wC*3 + (self.wMT + self.wV)*2
            denseOut = layDENSE(
                input=      input,
                units=      self.cellW,
                activation= tf.nn.relu)
            input = denseOut['output']
            input = tf.contrib.layers.layer_norm(
                    inputs=             input,
                    begin_norm_axis=    -1,
                    begin_params_axis=  -1)
            print(' > input (dense+LN):', input)

            bsz = tf.shape(input)[0]
            self.inState = tf.placeholder(
                name=       'state',
                dtype=      tf.float32,
                shape=      [None,2,self.cellW])

            self.singleZeroState = tf.zeros(shape=[2,self.cellW])

            # state is a tensor of shape [batch_size, cell_state_size]
            c, h = tf.unstack(self.inState, axis=1)
            cellZS = tf.nn.rnn_cell.LSTMStateTuple(c,h)
            print(' > cell zero state:', cellZS)

            cell = tf.contrib.rnn.NASCell(self.cellW)
            out, state = tf.nn.dynamic_rnn(
                cell=           cell,
                inputs=         input,
                initial_state=  cellZS,
                dtype=          tf.float32)

            print(' > out:', out)
            print(' > state:', state)
            state = tf.concat(state, axis=-1)
            self.finState = tf.reshape(state, shape=[-1,2,self.cellW])
            print(' > finState:', self.finState)

            denseOut = layDENSE(
                input=      out,
                units=      4,
                activation= tf.nn.relu)
            logits = denseOut['output']
            print(' > logits:', logits)

            self.probs = tf.nn.softmax(logits)

            self.vars = tf.trainable_variables(scope=tf.get_variable_scope().name)
            print(' ### num of vars %s'%shortSCIN(numVFloats(self.vars)))

            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.loss = loss(self.move, logits, sample_weight=self.reward)
            self.gradients = tf.gradients(self.loss, self.vars)
            self.gN = tf.global_norm(self.gradients)

            self.gradients, _ = tf.clip_by_global_norm(t_list=self.gradients, clip_norm=1, use_norm=self.gN)

            #optimizer = tf.train.GradientDescentOptimizer(1e-5)
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-6)

            self.optimizer = optimizer.apply_gradients(zip(self.gradients,self.vars))

            # select optimizer vars
            self.optVars = []
            for var in tf.global_variables(scope=tf.get_variable_scope().name):
                if var not in self.vars: self.optVars.append(var)

    """
    def resetME(self, newName=None):
        super().resetME(newName)
        self.session.run(tf.initializers.global_variables()
    """

    # prepares state in form of nn input
    def _encState(
            self,
            pIX: int,
            tableStateChanges: list):

        super()._encState(pIX, tableStateChanges)

        inMT = []  # list of moves (2)
        inV = []  # list of vectors (2)
        for state in tableStateChanges:
            key = list(state.keys())[0]

            if key == 'playersPC':
                myCards = state[key][self.plsHpos[pIX][0]][1]
                self.myCards[pIX] = [PDeck.cti(card) for card in myCards]

            if key == 'newTableCards':
                tCards = state[key]
                self.myCards[pIX] += [PDeck.cti(card) for card in tCards]

            if key == 'moveData':
                who = state[key]['pIX'] # who moved
                if who: # my index is 0, so do not include my moves

                    inMT.append(state[key]['plMove'][0])  # move type

                    vec = np.zeros(shape=self.wV)

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

        inC = [] + self.myCards[pIX]
        while len(inC) < 7: inC.append(52) # pad cards
        while len(inMT) < 2*(len(self.plsHpos[pIX])-1): inMT.append(4) # pad moves
        while len(inV) < 2*(len(self.plsHpos[pIX])-1): inV.append(np.zeros(shape=self.wV)) # pad vectors
        nnInput = [inC], [inMT], [inV] # seq of values (seq because of shape)
        return nnInput

    # calculates probs with NN for batches
    def _calcProbs(
            self,
            pIX,
            decState):

        self.decStates[pIX] = decState

        pProbsL = None
        if len(self.decStates) == self.nPl // 3: # TODO: hardcoded amount!

            pIXsl = sorted(list(self.decStates.keys())) # sorted list of pIX that will be processed

            inCbatch = []
            inMTbatch = []
            inVbatch = []
            statesBatch = []
            for pIX in pIXsl:
                inCseq, inMTseq, inVseq = self.decStates[pIX]
                inCbatch.append(inCseq)
                inMTbatch.append(inMTseq)
                inVbatch.append(inVseq)
                statesBatch.append(self.lastFwdState[pIX])

            feed = {
                self.inC:       inCbatch,
                self.inMT:      inMTbatch,
                self.inV:       inVbatch,
                self.inState:   statesBatch}

            fetches = [self.probs, self.finState]
            probs, fwdStates = self.session.run(fetches, feed_dict=feed)

            probs = np.reshape(probs, newshape=(probs.shape[0],probs.shape[-1])) # TODO: maybe smarter way

            pProbsL = []
            for ix in range(fwdStates.shape[0]):
                pIX = pIXsl[ix]
                pProbsL.append((pIX,probs[ix]))
                self.lastFwdState[pIX] = fwdStates[ix]

            self.decStates = {}

        if pProbsL:
            self.nHF += len(pProbsL)
            #print(self, self.nHF)
        return pProbsL

    # runs update of net
    def runUpdate(self):

        nM = [len(self.lDMR[ix]) for ix in range(self.nPl)] # number of saved moves per player (not all rewarded)
        avgNM = sum(nM)/len(nM) # avg

        # do update
        if avgNM > 100: # TODO: hardcoded value!
            print('min med max nM', min(nM), avgNM, max(nM))
            nR = [0]* self.nPl # factual num of rewarded moves
            for pix in range(self.nPl):
                for mix in reversed(range(len(self.lDMR[pix]))):
                    if self.lDMR[pix][mix]['reward'] is not None:
                        nR[pix] = mix
                        break
            minNR = min(nR)
            print('min med max nR', minNR, sum(nR)/len(nR), max(nR))

            # build batches of data
            inCbatch = []
            inMTbatch = []
            inVbatch = []
            moveBatch = []
            rewBatch = []
            for pix in range(self.nPl):
                inCseq = []
                inMTseq = []
                inVseq = []
                moveSeq = []
                rewSeq = []
                for nM in range(minNR):
                    mDict = self.lDMR[pix][nM]
                    decState = mDict['decState']
                    inCseq += decState[0]
                    inMTseq += decState[1]
                    inVseq += decState[2]
                    moveSeq.append(mDict['move'])
                    rew = 1 if mDict['reward'] > 0 else -1
                    if mDict['reward'] == 0: rew = 0
                    rewSeq.append(rew)
                inCbatch.append(inCseq)
                inMTbatch.append(inMTseq)
                inVbatch.append(inVseq)
                moveBatch.append(moveSeq)
                rewBatch.append(rewSeq)
            statesBatch = [self.lastUpdState[ix] for ix in range(self.nPl)]  # build directly from dict of Upd states

            """
            inCbatch = np.asarray(inCbatch)
            inMTbatch = np.asarray(inMTbatch)
            inVbatch = np.asarray(inVbatch)
            moveBatch = np.asarray(moveBatch)
            rewBatch = np.asarray(rewBatch)
            statesBatch = np.asarray(statesBatch)
            print(inCbatch.shape)
            print(inMTbatch.shape)
            print(inVbatch.shape)
            print(moveBatch.shape)
            print(rewBatch.shape)
            print(statesBatch.shape)
            """

            feed = {
                self.inC:       inCbatch,
                self.inMT:      inMTbatch,
                self.inV:       inVbatch,
                self.move:      moveBatch,
                self.reward:    rewBatch,
                self.inState:   statesBatch}

            fetches = [self.optimizer, self.loss, self.gN, self.gradients, self.finState]
            _, loss, gN, grads, updStates = self.session.run(fetches, feed_dict=feed)
            print('loss %.3f gN %.3f' %(loss, gN))

            for pIX in range(self.nPl):
                self.lastUpdState[pIX] = updStates[pIX] # save states
                self.lDMR[pIX] = self.lDMR[pIX][minNR:] # trim
