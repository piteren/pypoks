"""

 2019 (c) piteren

 https://stackoverflow.com/questions/46772685/how-to-accumulate-gradients-in-tensorflow
 https://github.com/tensorflow/benchmarks/issues/210

 TODO:
  - implement simple algorithmic DMK
  - separate tf.graph from DMK, DMK should be universal neural interface
  - implement simple stats for DMK

"""

import numpy as np
import tensorflow as tf
import time

from pUtils.littleTools.littleMethods import shortSCIN
from pUtils.nnTools.nnBaseElements import defInitializer, layDENSE, numVFloats
from pLogic.pDeck import PDeck


# basic implementation of DMK (random sampling)
class DecisionMaker:

    def __init__(
            self,
            name :str=  None,
            nMoves=     4,      # number of (all) moves
            runTB=      True#False
    ):

        self.name = name
        if self.name is None: self.name = self.getName()
        self.nMoves = nMoves

        self.myTableID = None
        self.myOponents = [] # table ids of oponents

        self.lDSTMVwR = []  # list of dicts {'decState': 'move': 'reward':}

        self.runTB = runTB
        self.summWriter = None
        self.counter = 0
        self.accumRew = 0

    # starts DMK for new game (table, player)
    def start(
            self,
            table,
            player):

        self.myTableID = player.tID
        self.myOponents = [ix for ix in range(table.nPlayers) if ix != player.tID]
        if self.runTB: self.summWriter = tf.summary.FileWriter(logdir='_nnTB/' + self.name, flush_secs=10)

    # generates name regarding class policy
    def getName(self): return 'rnDMK_%s'%time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3]

    # resets knowledge, stats, name of DMK
    def resetME(self, newName=None):

        if newName is True: self.name = self.getName()
        elif newName: self.name = newName

        if self.runTB: self.summWriter = tf.summary.FileWriter(logdir='_nnTB/' + self.name, flush_secs=10)
        self.counter = 0
        self.accumRew = 0

    # translates table stateChanges to decState object (table state readable by getProbs)
    def decState(
            self,
            tableStateChanges: list):

        decState = None
        return decState

    # returns probabilities of moves
    def getProbs(
            self,
            decState):

        return [1/self.nMoves] * self.nMoves

    # makes decision based on stateChanges - selects move from possibleMoves
    def mDec(
            self,
            tableStateChanges: list,
            possibleMoves: list):

        decState = self.decState(tableStateChanges)
        probs = self.getProbs(decState)

        probMask = [int(pM) for pM in possibleMoves]
        probs = probs * np.asarray(probMask)
        if np.sum(probs) > 0: probs = probs / np.sum(probs)
        else: probs = [1/self.nMoves] * self.nMoves

        move = np.random.choice(np.arange(self.nMoves), p=probs) # sample from probs

        # save state and move (for updates etc.)
        self.lDSTMVwR.append({
            'decState': decState,
            'move':     move,
            'reward':   None})

        return move

    # takes reward for last decision
    def getReward(
            self,
            reward: int):

        if self.lDSTMVwR: # only when there was any decision
            if self.lDSTMVwR[-1]['reward'] is None: # only when last decision was not rewarded yet
                self.lDSTMVwR[-1]['reward'] = reward # update reward
                self.accumRew += reward

        if self.summWriter and self.counter % 1000 == 0:
            accSumm = tf.Summary(value=[tf.Summary.Value(tag='wonTot', simple_value=self.accumRew)])
            self.summWriter.add_summary(accSumm, self.counter)
        self.counter += 1
        # here custom implementation may update decision alg

# base neural decision maker implementation
# LSTM with updated state
class BNdmk(DecisionMaker):

    def __init__(
            self,
            session :tf.compat.v1.Session,
            name=   None):

        super().__init__(name, runTB=True)
        self.session = session

        self.wET = 8 # event type emb width
        self.wC = 20 # card (single) emb width
        self.wV = 120 #

        self.lastFwdState = None # netState after last fwd
        self.lastUpdState = None # netState after last update

        self._buildGraph()
        self.session.run(tf.initializers.variables(var_list=self.vars+self.optVars))


    def getName(self): return 'bnnDMK_%s'%time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3]

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
    def decState(
            self,
            tableStateChanges: list):

        inET = []  # list of ints
        inC = []  # list of (int,int,int)
        inV = []  # list of vectors
        for state in tableStateChanges:
            key = list(state.keys())[0]
            # print(' *** state:', state)

            if key == 'playersPC':
                myPos = 0
                for ix in range(len(state[key])):
                    if state[key][ix][0] == self.myTableID: myPos = ix
                cards = state[key][myPos][1]
                vec = np.zeros(shape=self.wV)
                vec[0] = myPos - 1

                inET.append(0)
                inC.append((PDeck.cardToInt(cards[0]), PDeck.cardToInt(cards[1]), 52))
                inV.append(vec)

            if key == 'newTableCards':
                cards = state[key]
                cards = [PDeck.cardToInt(card) for card in cards]
                while len(cards) < 3: cards.append(52)

                inET.append(1)
                inC.append(cards)
                inV.append(np.zeros(shape=self.wV))

            if key == 'moveData':
                vec = np.zeros(shape=self.wV)
                vec[0] = state[key]['pltID']
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
                # print(vec)
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
