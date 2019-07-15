"""

 2019 (c) piteren

"""

import numpy as np
import os
import tensorflow as tf

from pUtils.nnTools.nnBaseElements import defInitializer, layDENSE
from pLogic.pDeck import PDeck


# proto of neural decision maker
class DecisionMaker:

    def __init__(self):

        # tf verbosity
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.players = ['pl0','pl1','pl2'] # myself always 1st

        self.wET = 8 # event type emb width
        self.wC = 20 # card (single) emb width
        self.wV = 120 #

        self.lastFwdState = None # netState after last fwd
        self.lastUpdState = None # netState after last update
        self.lInputs = [] # list of inputs dicts (save till reward)
        self.upInputs = [] # list of tuples (reward,inputs)

        self._buildGraph()

        self.session = tf.Session()
        self.session.run(tf.initializers.global_variables())

    # builds NN graph
    def _buildGraph(self):

        with tf.variable_scope('decMaker'):

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
                name=           'imVar',
                shape=          [53, self.wC],
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
            inCemb = tf.unstack(inCemb, axis=-1)
            inCemb = tf.concat(inCemb, axis=-1)
            print(' > inCemb:', inCemb)

            input = tf.concat([inETemb, inCemb, self.inV], axis=-1)
            print(' > input:', input)

            bsz = tf.shape(input)[0]
            self.inState = tf.placeholder_with_default(
                input=  tf.zeros(shape=[2,bsz,width]),
                shape=  [2,None,width],
                name=   'state')

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

            vars = tf.trainable_variables()
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            rewNorm = 3*tf.nn.tanh(self.reward/100)
            loss = loss(self.move, logits, sample_weight=rewNorm)
            gradients = tf.gradients(loss, vars)
            optimizer = tf.train.GradientDescentOptimizer(1e-5)
            self.optimizer = optimizer.apply_gradients(zip(gradients,vars))

    # runs fwd to get probs
    def _getProbs(
            self,
            inET,   # [bsz,seq]
            inC,    # [bsz,seq,3]
            inV):   # [bsz,seq,inVw]

        feed = {
            self.inET:  inET,
            self.inC:   inC,
            self.inV:   inV}
        if self.lastFwdState is not None: feed[self.inState] = self.lastFwdState
        fetches = [self.probs, self.finState]
        probs, self.lastFwdState = self.session.run(fetches, feed_dict=feed)
        return probs

    # runs update of net with self.upInputs
    def _runUpdate(self):

        for upInput in self.upInputs:
            rew = upInput[0]
            inputs = upInput[1]
            if len(inputs):
                rewPI = rew / len(inputs)
                for inp in inputs:
                    inET =  inp['inET']
                    inC =   inp['inC']
                    inV =   inp['inV']
                    move =  inp['move']

                    feed = {
                        self.inET:      inET,
                        self.inC:       inC,
                        self.inV:       inV,
                        self.move:      move,
                        self.reward:    [rewPI]}
                    if self.lastUpdState is not None: feed[self.inState] = self.lastUpdState
                    fetches = [self.optimizer, self.finState]
                    _, self.lastUpdState = self.session.run(fetches, feed_dict=feed)

    # returns int
    def mDec(
            self,
            stateChanges: list,
            possibleMoves: list):

        inET = [] # list of ints
        inC = [] # list of (int,int,int)
        inV = [] # list of vectors
        for state in stateChanges:
            key = list(state.keys())[0]
            # print(' *** state:', state)

            if key == 'playersPC':
                myPos = 0
                for ix in range(len(state[key])):
                    if state[key][ix][0] == self.players[0]: myPos = ix
                cards = state[key][myPos][1]
                vec = np.zeros(shape=self.wV)
                vec[0] = myPos - 1

                inET.append(0)
                inC.append((PDeck.cardToInt(cards[0]),PDeck.cardToInt(cards[1]),52))
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
                vec[0] = self.players.index(state[key]['pName'])-1
                vec[1] = state[key]['tBCash']       /750-1
                vec[2] = state[key]['pBCash']       /250-1
                vec[3] = state[key]['pBCHandCash']  /250-1
                vec[4] = state[key]['pBCRiverCash'] /250-1
                vec[5] = state[key]['bCashToCall']  /250-1
                vec[6] = state[key]['plMove'][0]    /1.5-1
                vec[7] = state[key]['tACash']       /750-1
                vec[8] = state[key]['pACash']       /250-1
                vec[9] = state[key]['pACHandCash']  /250-1
                vec[10]= state[key]['pACRiverCash'] /250-1

                inET.append(2)
                inC.append((52,52,52))
                inV.append(np.zeros(shape=self.wV))

        probs = self._getProbs([inET],[inC],[inV])
        probs = probs[0]
        print(' *** nn probs:', probs)

        probMask = [0,0,0,0]
        for pos in possibleMoves: probMask[pos] = 1
        probMask = np.asarray(probMask)
        print(' *** probMask:', probMask)
        probs = probs*probMask
        probs = probs/np.sum(probs)
        print(' *** probs masked:', probs)

        move = np.random.choice(np.arange(4), p=probs)
        print(' *** move:', move)

        inDict = {
            'inET': [inET],
            'inC':  [inC],
            'inV':  [inV],
            'move': [move]}
        self.lInputs.append(inDict)

        return move

    # takes reward and updates net if "condition"
    def getReward(
            self,
            reward: int):

        self.upInputs.append((reward, self.lInputs))
        self.lInputs = []
        self._runUpdate()
        self.upInputs = []