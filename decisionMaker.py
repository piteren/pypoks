"""

 2019 (c) piteren

"""

import numpy as np
import os
import tensorflow as tf

from pUtils.nnTools.nnBaseElements import defInitializer, layDENSE


# proto of neural decision maker
class DecisionMaker:

    def __init__(self):

        # tf verbosity
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.wET = 8 # event type emb width
        self.wC = 20 # card (single) emb width
        self.wV = 120 #

        self.lastFwdState = None # netState after last fwd
        self.lastUpdState = None # netState after last update
        self.inputs = None # inputs save for update

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
                shape=          [None,None])

            etEMB = tf.get_variable( # event type embeddings
                name=           'etEMB',
                shape=          [10,self.wET],
                dtype=          tf.float32,
                initializer=    defInitializer())

            self.inC = tf.placeholder( # 3 cards
                name=           'inC',
                dtype=          tf.int32,
                shape=          [None,None,3])

            cEMB = tf.get_variable( # cards embeddings
                name=           'imVar',
                shape=          [53, self.wC],
                dtype=          tf.float32,
                initializer=    defInitializer())

            self.inV = tf.placeholder( # event float values
                name=           'inV',
                dtype=          tf.float32,
                shape=          [None,None,self.wV])

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

    def _getProbs(
            self,
            inET,   # [bsz, seq]
            inC,    # [bsz, seq, 3]
            inV):   # [bsz, seq, inVw]

        feed = {
            self.inET:  inET,
            self.inC:   inC,
            self.inV:   inV}
        if self.lastFwdState is not None: feed[self.inState] = self.lastFwdState
        fetches = [self.probs, self.finState]
        probs, self.lastFwdState = self.session.run(fetches, feed_dict=feed)
        return probs

    # returns int
    def mDec(
            self,
            stateChanges: list,
            possibleMoves: list):

        for state in stateChanges:
            print(' *** state:', state)

        probs = self._getProbs([[2]],[[[0,0,0]]],np.random.random(size=[1,1,self.wV]))
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

        return move

    # takes reward and updates net if "condition"
    def getReward(
            self,
            reward: int):

        pass