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
class DecisionMaker(Process):

    def __init__(
            self,
            name :str,          # name should be unique (@table)
            iQue :Queue,        # input que
            oQue :Queue,        # output que
            nMoves=     4,      # number of (all) moves supported by DM, has to match table/player
            randMove=   0.2,    # how often move will be sampled from random
            runTB=      False):

        super().__init__()

        self.name = name
        self.nMoves = nMoves
        self.randMove = randMove

        self.lDSTMVwR = []  # list of dicts {'decState': 'move': 'reward':}

        self.runTB = runTB
        self.summWriter = None

        self.preflop = True # preflop indicator
        self.plsHpos = [] # positions @table of players from self.pls

        self.repTime = 0
        self.stsV = 1000 # stats interval
        self.sts = {} # stats
        self.cHSdata = {} # current hand data for stats
        self._resetSTS()
        self._resetCSHD()

        self.iQue = iQue
        self.oQue = oQue

    def run(self):

        while True:
            pIX, stateChanges, possibleMoves = self.iQue.get()
            move = self.mDec(stateChanges, possibleMoves)
            self.oQue.put([pIX, move])

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

    # starts DMK for new game (table, player)
    def start(self): # list of player names

        if self.runTB: self.summWriter = tf.summary.FileWriter(logdir='_nnTB/' + self.name, flush_secs=10)
        self.repTime = time.time()
        self.start()

    # resets knowledge, stats, name of DMK
    def resetME(self, newName=None):

        if newName: self.name = newName
        self._resetSTS()
        if self.runTB: self.summWriter = tf.summary.FileWriter(logdir='_nnTB/' + self.name, flush_secs=10)

    # encodes table stateChanges to decState object (state readable by getProbs)
    def encState(
            self,
            tableStateChanges: list):

        for state in tableStateChanges:
            key = list(state.keys())[0]

            # update positions of players @table for new hand
            if key == 'playersPC':
                self.plsHpos = [0]*len(state[key])
                for ix in range(len(state[key])):
                    self.plsHpos[state[key][ix][0]] = ix
            # check for preflop >> postflop
            if key == 'newTableCards':
                if len(state[key]) == 3: self.preflop = False

        decState = None
        return decState

    # returns probabilities of moves
    def getProbs(
            self,
            decState):

        return [1/self.nMoves] * self.nMoves

    # updates current hand data for stats based on move performing
    def _updMoveStats(
            self,
            move):

        if move == 3: self.cHSdata['SH'] = True
        if self.preflop:
            if move == 1 and self.plsHpos[0] != 1 or move > 1: self.cHSdata['VPIP'] = True
            if move > 1: self.cHSdata['PFR'] = True
        else:
            self.cHSdata['nPM'] += 1
            if move > 1: self.cHSdata['nAGG'] += 1


    # makes decision based on stateChanges - selects move from possibleMoves
    # TODO: prep mDec in tournament mode (no sampling from distribution, but max)
    def mDec(
            self,
            tableStateChanges: list,
            possibleMoves: list): # how often sample move from random

        decState = self.encState(tableStateChanges)
        probs = self.getProbs(decState)
        if random.random() < self.randMove: probs = [1/self.nMoves] * self.nMoves

        probMask = [int(pM) for pM in possibleMoves]
        probs = probs * np.asarray(probMask)
        if np.sum(probs) > 0: probs = probs / np.sum(probs)
        else: probs = [1/self.nMoves] * self.nMoves

        move = np.random.choice(np.arange(self.nMoves), p=probs) # sample from probs
        self._updMoveStats(move)

        # save state and move (for updates etc.)
        self.lDSTMVwR.append({
            'decState': decState,
            'move':     move,
            'reward':   None})

        return move

    # takes reward for last decision
    # performs after-hand opps (stats etc.)
    def getReward(
            self,
            reward: int):

        if self.lDSTMVwR: # only when there was any decision
            if self.lDSTMVwR[-1]['reward'] is None: # only when last decision was not rewarded yet
                self.lDSTMVwR[-1]['reward'] = reward # update reward

        self.sts['nH'][0] += 1
        self.sts['nH'][1] += 1
        self.sts['$'][0] += reward
        self.sts['$'][1] += reward

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

            ix, cx = (0,'T') if not V else (1,'V') # interval or total

            won = tf.Summary(value=[tf.Summary.Value(tag='sts%s/0_won$'%cx, simple_value=self.sts['$'][ix])])
            vpip = self.sts['nVPIP'][ix] / self.sts['nH'][ix] * 100
            vpip = tf.Summary(value=[tf.Summary.Value(tag='sts%s/1_VPIP'%cx, simple_value=vpip)])
            pfr = self.sts['nPFR'][ix] / self.sts['nH'][ix] * 100
            pfr = tf.Summary(value=[tf.Summary.Value(tag='sts%s/2_PFR'%cx, simple_value=pfr)])
            sh = self.sts['nSH'][ix] / self.sts['nH'][ix] * 100
            sh = tf.Summary(value=[tf.Summary.Value(tag='sts%s/4_sh'%cx, simple_value=sh)])
            agg = self.sts['nAGG'][ix] / self.sts['nPM'][ix] * 100 if self.sts['nPM'][ix] else 0
            agg = tf.Summary(value=[tf.Summary.Value(tag='sts%s/3_AGG'%cx, simple_value=agg)])
            self.summWriter.add_summary(won, self.sts['nH'][0])
            self.summWriter.add_summary(vpip, self.sts['nH'][0])
            self.summWriter.add_summary(pfr, self.sts['nH'][0])
            self.summWriter.add_summary(sh, self.sts['nH'][0])
            self.summWriter.add_summary(agg, self.sts['nH'][0])

            # reset interval values
            if V:
                for key in self.sts.keys():
                    self.sts[key][1] = 0

        if self.summWriter and self.sts['nH'][1] % self.stsV == 0:
            repSTS(True)
        if self.summWriter and self.sts['nH'][0] % 1000 == 0:
            repSTS()
            print(' >>> training time: %.1fsec/%d'%(time.time() - self.repTime, 1000))
            self.repTime = time.time()


        self.preflop = True

        # here custom implementation may update decision alg

# first, base, neural-DMK, LSTM with multi-state to decision
class BNdmk(DecisionMaker):

    def __init__(
            self,
            iQue: Queue,
            oQue: Queue,
            name=       None,
            randMove=   0.2):

        super().__init__(
            name=       name,
            iQue=       iQue,
            oQue=       oQue,
            randMove=   randMove,
            runTB=      True)

        self.session = tf.compat.v1.Session()

        self.wET = 8 # event type emb width
        self.wC = 20 # card (single) emb width
        self.wV = 120 # width of values vector

        self.lastFwdState = None # netState after last fwd
        self.lastUpdState = None # netState after last update

        self._buildGraph()
        self.session.run(tf.initializers.variables(var_list=self.vars+self.optVars))

    # builds NN graph
    def _buildGraph(self):

        with tf.variable_scope('bnnDMK_%s'%self.name):

            width = self.wET + self.wC*3 + self.wV
            cell = tf.contrib.rnn.NASCell(width)

            self.inET =         tf.placeholder( # event type
                name=           'inET',
                dtype=          tf.int32,
                shape=          [None,None]) # [bsz,seq]

            etEMB = tf.get_variable( # event type embeddings
                name=           'etEMB',
                shape=          [10,self.wET],
                dtype=          tf.float32,
                initializer=    defInitializer())

            self.inC = tf.placeholder( # 3 cards
                name=           'inC',
                dtype=          tf.int32,
                shape=          [None,None,3]) # [bsz,seq,3cards]

            cEMB = tf.get_variable( # cards embeddings
                name=           'cEMB',
                shape=          [53,self.wC], # one card for 'no_card'
                dtype=          tf.float32,
                initializer=    defInitializer())

            self.inV = tf.placeholder( # event float values
                name=           'inV',
                dtype=          tf.float32,
                shape=          [None,None,self.wV]) # [bsz,seq,vec]

            self.move = tf.placeholder( # "correct" move (class)
                name=           'move',
                dtype=          tf.int32,
                shape=          [None]) # [bsz]

            self.reward = tf.placeholder( # reward for "correct" move
                name=           'reward',
                dtype=          tf.float32,
                shape=          [None]) # [bsz]

            inETemb = tf.nn.embedding_lookup(params=etEMB, ids=self.inET)
            print(' > inETemb:', inETemb)

            inCemb = tf.nn.embedding_lookup(params=cEMB, ids=self.inC)
            print(' > inCemb:', inCemb)
            inCemb = tf.unstack(inCemb, axis=-2)
            inCemb = tf.concat(inCemb, axis=-1)
            print(' > inCemb:', inCemb)

            input = tf.concat([inETemb, inCemb, self.inV], axis=-1)
            denseOut = layDENSE(
                input=      input,
                units=      width,
                activation= tf.nn.relu)
            input = denseOut['output']
            input = tf.contrib.layers.layer_norm(
                    inputs=             input,
                    begin_norm_axis=    -1,
                    begin_params_axis=  -1)
            print(' > input:', input)

            bsz = tf.shape(input)[0]
            self.inState = tf.placeholder_with_default(
                input=      tf.zeros(shape=[2,bsz,width]),
                shape=      [2,None,width],
                name=       'state')

            # state is a tensor of shape [batch_size, cell_state_size]
            c, h = tf.unstack(self.inState, axis=0)
            cellZS = tf.nn.rnn_cell.LSTMStateTuple(c,h)
            print(' > cell zero state:', cellZS)

            out, state = tf.nn.dynamic_rnn(
                cell=           cell,
                inputs=         input,
                initial_state=  cellZS,
                dtype=          tf.float32)

            self.finState = tf.stack(state, axis=0)
            print(' > out:', out)
            print(' > state:', self.finState)

            denseOut = layDENSE(
                input=      out[:,-1,:],
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


    def resetME(self, newName=None):
        super().resetME(newName)
        self.session.run(tf.initializers.global_variables())

    # prepares state in form of nn input
    def encState(
            self,
            tableStateChanges: list):

        super().encState(tableStateChanges)

        inET = []  # list of ints
        inC = []  # list of (int,int,int)
        inV = []  # list of vectors
        for state in tableStateChanges:
            key = list(state.keys())[0]

            if key == 'playersPC':
                cards = state[key][self.plsHpos[0]][1]
                vec = np.zeros(shape=self.wV)
                vec[0] = self.plsHpos[0]

                inET.append(0)
                inC.append((PDeck.cti(cards[0]), PDeck.cti(cards[1]), 52))
                inV.append(vec)

            if key == 'newTableCards':
                cards = state[key]
                cards = [PDeck.cti(card) for card in cards]
                while len(cards) < 3: cards.append(52)

                inET.append(1)
                inC.append(cards)
                inV.append(np.zeros(shape=self.wV))

            if key == 'moveData':
                vec = np.zeros(shape=self.wV)
                vec[0] = state[key]['pIX']
                vec[1] = state[key]['tBCash'] / 1500
                vec[2] = state[key]['pBCash'] / 500
                vec[3] = state[key]['pBCHandCash'] / 500
                vec[4] = state[key]['pBCRiverCash'] / 500
                vec[5] = state[key]['bCashToCall'] / 500
                vec[6] = state[key]['plMove'][0] / 3
                vec[7] = state[key]['tACash'] / 1500
                vec[8] = state[key]['pACash'] / 500
                vec[9] = state[key]['pACHandCash'] / 500
                vec[10] = state[key]['pACRiverCash'] / 500

                inET.append(2)
                inC.append((52, 52, 52))
                inV.append(vec)

        nnInput = [inET], [inC], [inV]
        return nnInput

    # runs fwd to get probs (np.array)
    def getProbs(
            self,
            decState):

        inET, inC, inV = decState
        feed = {
            self.inET:  inET,
            self.inC:   inC,
            self.inV:   inV}
        if self.lastFwdState is not None: feed[self.inState] = self.lastFwdState
        fetches = [self.probs, self.finState]
        probs, self.lastFwdState = self.session.run(fetches, feed_dict=feed)
        return probs[0]

    # runs update of net
    def runUpdate(self):

        if self.lDSTMVwR: # only when there was any decision
            # reverse update of rewards
            rew = 0
            for dec in reversed(self.lDSTMVwR):
                if dec['reward'] is not None:
                    # rew = upInput[0]/1500
                    rew = 1 if dec['reward'] > 0 else -1
                    if dec['reward'] == 0: rew = 0
                else: dec['reward'] = rew
            for dec in self.lDSTMVwR:

                inET =  dec['decState'][0]
                inC =   dec['decState'][1]
                inV =   dec['decState'][2]
                move =  dec['move']
                rew =   dec['reward']

                feed = {
                    self.inET:      inET,
                    self.inC:       inC,
                    self.inV:       inV,
                    self.move:      [move],
                    self.reward:    [rew]}
                if self.lastUpdState is not None: feed[self.inState] = self.lastUpdState

                fetches = [self.optimizer, self.loss, self.gN, self.gradients, self.finState]
                _, loss, gN, grads, self.lastUpdState = self.session.run(fetches, feed_dict=feed)
                """
                for gr in grads:
                    print(type(gr), end=' ')
                    if type(gr) is np.ndarray: print(gr.shape)
                    else: print(gr.dense_shape)
                """
                #print('loss %.3f gN %.3f' %(loss, gN))

        self.lastFwdState = self.lastUpdState
        self.lDSTMVwR = []

    # takes reward (updates net)
    def getReward(
            self,
            reward: int):

        super().getReward(reward)
        self.runUpdate()

# second, neural-DMK, LSTM with single-state to decision
class SNdmk(DecisionMaker):

    def __init__(
            self,
            iQue: Queue,
            oQue: Queue,
            name=       None,
            randMove=   0.2):

        super().__init__(
            name=       name,
            iQue=       iQue,
            oQue=       oQue,
            randMove=   randMove,
            runTB=      True)

        self.session = tf.compat.v1.Session()

        self.wC = 16        # card (single) emb width
        self.wMT = 4        # move type emb width
        self.wV = 11        # values vector width, holds position @table and cash values
        self.cellW = 188    # cell width

        self.lastFwdState = None # netState after last fwd
        self.lastUpdState = None # netState after last update
        self.myCards = []

        self._buildGraph()
        self.session.run(tf.initializers.variables(var_list=self.vars+self.optVars))

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
                shape=          [None]) # [bsz]

            self.reward = tf.placeholder( # reward for "correct" move
                name=           'reward',
                dtype=          tf.float32,
                shape=          [None]) # [bsz]

            inCemb = tf.nn.embedding_lookup(params=cEMB, ids=self.inC)
            print(' > inCemb:', inCemb)
            inCemb = tf.unstack(inCemb, axis=-2)
            inCemb = tf.concat(inCemb, axis=-1)
            print(' > inCemb:', inCemb)

            inMTemb = tf.nn.embedding_lookup(params=mtEMB, ids=self.inMT)
            print(' > inMTemb:', inMTemb)
            inMTemb = tf.unstack(inMTemb, axis=-2)
            inMTemb = tf.concat(inMTemb, axis=-1)
            print(' > inMTemb:', inMTemb)

            inV = tf.unstack(self.inV, axis=-2)
            inV = tf.concat(inV, axis=-1)
            print(' > inV:', inV)

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
            self.inState = tf.placeholder_with_default(
                input=      tf.zeros(shape=[2,bsz,self.cellW]),
                shape=      [2,None,self.cellW],
                name=       'state')

            # state is a tensor of shape [batch_size, cell_state_size]
            c, h = tf.unstack(self.inState, axis=0)
            cellZS = tf.nn.rnn_cell.LSTMStateTuple(c,h)
            print(' > cell zero state:', cellZS)

            cell = tf.contrib.rnn.NASCell(self.cellW)
            out, state = tf.nn.dynamic_rnn(
                cell=           cell,
                inputs=         input,
                initial_state=  cellZS,
                dtype=          tf.float32)

            self.finState = tf.stack(state, axis=0)
            print(' > out:', out)
            print(' > state:', self.finState)

            denseOut = layDENSE(
                input=      out[:,-1,:],
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


    def resetME(self, newName=None):
        super().resetME(newName)
        self.session.run(tf.initializers.global_variables())

    # prepares state in form of nn input
    def encState(
            self,
            tableStateChanges: list):

        super().encState(tableStateChanges)

        inMT = []  # list of moves (2)
        inV = []  # list of vectors (2)
        for state in tableStateChanges:
            key = list(state.keys())[0]

            if key == 'playersPC':
                myCards = state[key][self.plsHpos[0]][1]
                self.myCards = [PDeck.cti(card) for card in myCards]

            if key == 'newTableCards':
                tCards = state[key]
                self.myCards += [PDeck.cti(card) for card in tCards]

            if key == 'moveData':

                who = state[key]['pIX'] # who moved
                if who: # my index is 0, so do not include my moves

                    inMT.append(state[key]['plMove'][0])  # move type

                    vec = np.zeros(shape=self.wV)

                    vec[0] = who
                    vec[1] = self.plsHpos[who] # what position

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

        inC = [] + self.myCards
        while len(inC) < 7: inC.append(52) # pad cards
        while len(inMT) < 2*(len(self.plsHpos)-1): inMT.append(4) # pad moves
        while len(inV) < 2*(len(self.plsHpos)-1): inV.append(np.zeros(shape=self.wV)) # pad vectors
        nnInput = [[inC]], [[inMT]], [[inV]]
        return nnInput

    # runs fwd to get probs (np.array)
    def getProbs(
            self,
            decState):

        inC, inMT, inV = decState
        feed = {
            self.inC:   inC,
            self.inMT:  inMT,
            self.inV:   inV}
        if self.lastFwdState is not None: feed[self.inState] = self.lastFwdState
        fetches = [self.probs, self.finState]
        probs, self.lastFwdState = self.session.run(fetches, feed_dict=feed)
        return probs[0]

    # runs update of net
    def runUpdate(self):

        if self.lDSTMVwR: # only when there was any decision
            # reverse update of rewards
            rew = 0
            for dec in reversed(self.lDSTMVwR):
                if dec['reward'] is not None:
                    # rew = upInput[0]/1500
                    rew = 1 if dec['reward'] > 0 else -1
                    if dec['reward'] == 0: rew = 0
                else: dec['reward'] = rew
            for dec in self.lDSTMVwR:

                inC =   dec['decState'][0]
                inMT =  dec['decState'][1]
                inV =   dec['decState'][2]
                move =  dec['move']
                rew =   dec['reward']

                feed = {
                    self.inC:       inC,
                    self.inMT:      inMT,
                    self.inV:       inV,
                    self.move:      [move],
                    self.reward:    [rew]}
                if self.lastUpdState is not None: feed[self.inState] = self.lastUpdState

                fetches = [self.optimizer, self.loss, self.gN, self.gradients, self.finState]
                _, loss, gN, grads, self.lastUpdState = self.session.run(fetches, feed_dict=feed)
                """
                for gr in grads:
                    print(type(gr), end=' ')
                    if type(gr) is np.ndarray: print(gr.shape)
                    else: print(gr.dense_shape)
                """
                #print('loss %.3f gN %.3f' %(loss, gN))

        self.lastFwdState = self.lastUpdState
        self.lDSTMVwR = []

    # takes reward (updates net)
    def getReward(
            self,
            reward: int):

        super().getReward(reward)
        self.myCards = []
        #self.runUpdate()
        if self.sts['nH'][0] % 100 == 0: self.runUpdate()