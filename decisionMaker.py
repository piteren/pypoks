"""

 2019 (c) piteren

"""

import numpy as np
import tensorflow as tf

from pUtils.nnTools.nnBaseElements import defInitializer, layDENSE


# proto of neural decision maker
class DecisionMaker:

    def __init__(self):

        self.inIw = 8
        self.inVw = 120
        self.lastFwdState = None # netState after last fwd
        self.lastUpdState = None # netState after last update
        self.inputs = None # inputs save for update
        self._buildGraph()
        self.session = tf.Session()
        self.session.run(tf.initializers.global_variables())

    # builds NN graph
    def _buildGraph(self):

        with tf.variable_scope('decMaker'):

            width = self.inIw + self.inVw
            cell = tf.contrib.rnn.NASCell(width)

            self.inI = tf.placeholder(
                name=   'inI',
                dtype=  tf.int32,
                shape=  [None,None])

            self.inV = tf.placeholder(
                name=   'inV',
                dtype=  tf.float32,
                shape=  [None,None,self.inVw])

            imVar = tf.get_variable(
                name=           'imVar',
                shape=          [10,self.inIw],
                dtype=          tf.float32,
                initializer=    defInitializer())

            embInI = tf.nn.embedding_lookup(params=imVar, ids=self.inI)
            print(' > embInI:', embInI)

            input = tf.concat([embInI, self.inV], axis=-1)
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
            inI,    # [bsz, seq]
            inV):   # [bsz, seq, inVw]

        feed = {self.inI: inI, self.inV: inV}
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

        inV = np.random.random(size=[1,1,self.inVw])

        probs = self._getProbs([[2]],inV)
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