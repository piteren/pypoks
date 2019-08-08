"""

 2019 (c) piteren

"""

import tensorflow as tf

from pUtils.littleTools.littleMethods import shortSCIN
from pUtils.nnTools.nnBaseElements import defInitializer, layDENSE, numVFloats
from pUtils.nnTools.nnEncoders import encDR

# base LSTM neural graph
def lstmGraphFN(
        scope :str,
        wC=         16,     # card (single) emb width
        wMT=        1,      # move type emb width
        wV=         11,     # values vector width, holds player move data(type, pos, cash)
        nDR=        3,      # num of encDR lay
        cellW=      1024,   # cell width
        optAda=     True,
        lR=         1e-4):

    with tf.variable_scope(scope):

        inC = tf.placeholder(  # 3 cards
            name=           'inC',
            dtype=          tf.int32,
            shape=          [None, None, 7])  # [bsz,seq,7cards]

        cEMB = tf.get_variable(  # cards embeddings
            name=           'cEMB',
            shape=          [53, wC],  # one card for 'no_card'
            dtype=          tf.float32,
            initializer=    defInitializer())

        inCemb = tf.nn.embedding_lookup(params=cEMB, ids=inC)
        print(' > inCemb:', inCemb)
        inCemb = tf.unstack(inCemb, axis=-2)
        inCemb = tf.concat(inCemb, axis=-1)
        print(' > inCemb (flattened):', inCemb)

        inMT = tf.placeholder(  # event type
            name=           'inMT',
            dtype=          tf.int32,
            shape=          [None, None, 4])  # [bsz,seq,2*2oppon]

        mtEMB = tf.get_variable(  # event type embeddings
            name=           'mtEMB',
            shape=          [5, wMT],  # 4 moves + no_move
            dtype=          tf.float32,
            initializer=    defInitializer())

        inMTemb = tf.nn.embedding_lookup(params=mtEMB, ids=inMT)
        print(' > inMTemb:', inMTemb)
        inMTemb = tf.unstack(inMTemb, axis=-2)
        inMTemb = tf.concat(inMTemb, axis=-1)
        print(' > inMTemb (flattened):', inMTemb)

        inV = tf.placeholder(  # event float values
            name=           'inV',
            dtype=          tf.float32,
            shape=          [None, None, 4, wV])  # [bsz,seq,2*2,vec]

        inVec = tf.unstack(inV, axis=-2)
        inVec = tf.concat(inVec, axis=-1)
        print(' > inV (flattened):', inVec)

        input = tf.concat([inCemb, inMTemb, inVec], axis=-1)
        print(' > input (concatenated):', input)  # width = self.wC*3 + (self.wMT + self.wV)*2

        encDRout = encDR(
            input=      input,
            nLayers=    nDR,
            layWidth=   cellW,
            nHL=        0,
            verbLev=    1)
        input = encDRout['output']

        inState = tf.placeholder(
            name=           'state',
            dtype=          tf.float32,
            shape=          [None, 2, cellW])

        singleZeroState = tf.zeros(shape=[2, cellW])

        # state is a tensor of shape [batch_size, cell_state_size]
        c, h = tf.unstack(inState, axis=1)
        cellZS = tf.nn.rnn_cell.LSTMStateTuple(c, h)
        print(' > cell zero state:', cellZS)

        cell = tf.contrib.rnn.NASCell(cellW)
        out, state = tf.nn.dynamic_rnn(
            cell=           cell,
            inputs=         input,
            initial_state=  cellZS,
            dtype=          tf.float32)

        print(' > out:', out)
        print(' > state:', state)
        state = tf.concat(state, axis=-1)
        finState = tf.reshape(state, shape=[-1, 2, cellW])
        print(' > finState:', finState)

        denseOut = layDENSE(
            input=      out,
            units=      4,
            #activation= tf.nn.relu,
            useBias=    False,)
        logits = denseOut['output']
        print(' > logits:', logits)

        probs = tf.nn.softmax(logits)

        vars = tf.trainable_variables(scope=tf.get_variable_scope().name)
        print(' ### num of vars %s' % shortSCIN(numVFloats(vars)))

        move = tf.placeholder(  # "correct" move (class)
            name=           'move',
            dtype=          tf.int32,
            shape=          [None, None])  # [bsz,seq]

        reward = tf.placeholder(  # reward for "correct" move
            name=           'reward',
            dtype=          tf.float32,
            shape=          [None, None])  # [bsz,seq]

        rew = reward/500 # lineary scale rewards

        # this loss is auto averaged with reduction parameter
        #loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        #loss = loss(y_true=move, y_pred=logits, sample_weight=rew)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=     move,
            logits=     logits,
            weights=    rew)

        gradients = tf.gradients(loss, vars)
        gN = tf.global_norm(gradients)

        gradients, _ = tf.clip_by_global_norm(t_list=gradients, clip_norm=1, use_norm=gN)

        optimizer = tf.train.AdamOptimizer(lR) if optAda else tf.train.GradientDescentOptimizer(lR)
        optimizer = optimizer.apply_gradients(zip(gradients, vars))

        # select optimizer vars
        optVars = []
        for var in tf.global_variables(scope=tf.get_variable_scope().name):
            if var not in vars: optVars.append(var)

        return{
            'scope':                scope,
            'inC':                  inC,
            'inMT':                 inMT,
            'inV':                  inV,
            'wV':                   wV,
            'move':                 move,
            'reward':               reward,
            'inState':              inState,
            'singleZeroState':      singleZeroState,
            'probs':                probs,
            'finState':             finState,
            'optimizer':            optimizer,
            'loss':                 loss,
            'gN':                   gN,
            'vars':                 vars,
            'optVars':              optVars}

# base CNN+RES neural graph
def cnnRGraphFN(
        scope :str,
        wC=         16,     # card (single) emb width
        wMT=        1,      # move type emb width
        wV=         11,     # values vector width, holds player move data(type, pos, cash)
        nLay=       24,     # number of CNNR layers
        reW=        512,    # representation width (number of filters)
        optAda=     True,
        lR=         1e-4):

    with tf.variable_scope(scope):

        inC = tf.placeholder(  # 3 cards
            name=           'inC',
            dtype=          tf.int32,
            shape=          [None, None, 7])  # [bsz,seq,7cards]

        cEMB = tf.get_variable(  # cards embeddings
            name=           'cEMB',
            shape=          [53, wC],  # one card for 'no_card'
            dtype=          tf.float32,
            initializer=    defInitializer())

        inCemb = tf.nn.embedding_lookup(params=cEMB, ids=inC)
        print(' > inCemb:', inCemb)
        inCemb = tf.unstack(inCemb, axis=-2)
        inCemb = tf.concat(inCemb, axis=-1)
        print(' > inCemb (flattened):', inCemb)

        inMT = tf.placeholder(  # event type
            name=           'inMT',
            dtype=          tf.int32,
            shape=          [None, None, 4])  # [bsz,seq,2*2oppon]

        mtEMB = tf.get_variable(  # event type embeddings
            name=           'mtEMB',
            shape=          [5, wMT],  # 4 moves + no_move
            dtype=          tf.float32,
            initializer=    defInitializer())

        inMTemb = tf.nn.embedding_lookup(params=mtEMB, ids=inMT)
        print(' > inMTemb:', inMTemb)
        inMTemb = tf.unstack(inMTemb, axis=-2)
        inMTemb = tf.concat(inMTemb, axis=-1)
        print(' > inMTemb (flattened):', inMTemb)

        inV = tf.placeholder(  # event float values
            name=           'inV',
            dtype=          tf.float32,
            shape=          [None, None, 4, wV])  # [bsz,seq,2*2,vec]

        inVec = tf.unstack(inV, axis=-2)
        inVec = tf.concat(inVec, axis=-1)
        print(' > inV (flattened):', inVec)

        input = tf.concat([inCemb, inMTemb, inVec], axis=-1)
        print(' > input (concatenated):', input)  # width = self.wC*3 + (self.wMT + self.wV)*2

        # projection without activation and bias
        denseOut = layDENSE(
            input=          input,
            units=          reW,
            useBias=        False,
            initializer=    defInitializer())
        projInput = denseOut['output']
        print(' > projInput (projected):', projInput)

        inState = tf.placeholder(
            name=           'state',
            dtype=          tf.float32,
            shape=          [None,nLay,2,reW]) # [bsz,nLay,2,reW]

        singleZeroState = tf.zeros(shape=[nLay,2,reW]) # [nLay,2,reW]

        # unstack layers of inState
        inStateLays = tf.unstack(inState, axis=-3)
        print(' > inStateLays len %d of:' %len(inStateLays), inStateLays[0])

        subOutput = tf.contrib.layers.layer_norm(
            inputs=             projInput,
            begin_norm_axis=    -1,
            begin_params_axis=  -1)
        layInputLays = []
        for depth in range(nLay):

            layInputLays.append(tf.concat([inStateLays[depth],subOutput], axis=-2))
            print(' > layInput of %d lay'%depth, layInputLays[-1])

            layName = 'cnnREncLay_%d' % depth
            with tf.variable_scope(layName):

                convLay = tf.layers.Conv1D(
                    filters=            reW,
                    kernel_size=        3,
                    dilation_rate=      1,
                    activation=         None,
                    use_bias=           True,
                    kernel_initializer= defInitializer(),
                    padding=            'valid',
                    data_format=        'channels_last')

            cnnOutput = convLay(layInputLays[-1])
            cnnOutput = tf.nn.relu(cnnOutput) # activation
            print(' > cnnOutput of %d lay' % depth, cnnOutput)
            subOutput += cnnOutput
            print(' > subOutput (RES) of %d lay' % depth, cnnOutput)
            subOutput = tf.contrib.layers.layer_norm(
                inputs=             subOutput,
                begin_norm_axis=    -1,
                begin_params_axis=  -1)

        out = subOutput
        print(' > out:', out)

        state = tf.stack(layInputLays, axis=-3)
        print(' > state (stacked):', state)
        finState = tf.split(state, num_or_size_splits=[-1,2], axis=-2)[1]
        print(' > finState (split):', finState)


        # projection to logits
        denseOut = layDENSE(
            input=          out,
            units=          4,
            useBias=        False,
            initializer=    defInitializer())
        logits = denseOut['output']
        print(' > logits:', logits)

        probs = tf.nn.softmax(logits)

        vars = tf.trainable_variables(scope=tf.get_variable_scope().name)
        print(' ### num of vars %s' % shortSCIN(numVFloats(vars)))

        move = tf.placeholder(  # "correct" move (class)
            name=           'move',
            dtype=          tf.int32,
            shape=          [None, None])  # [bsz,seq]

        reward = tf.placeholder(  # reward for "correct" move
            name=           'reward',
            dtype=          tf.float32,
            shape=          [None, None])  # [bsz,seq]

        rew = reward/500 # lineary scale rewards

        # this loss is auto averaged with reduction parameter
        #loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        #loss = loss(y_true=move, y_pred=logits, sample_weight=rew)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=     move,
            logits=     logits,
            weights=    rew)

        gradients = tf.gradients(loss, vars)
        gN = tf.global_norm(gradients)

        gradients, _ = tf.clip_by_global_norm(t_list=gradients, clip_norm=1, use_norm=gN)

        optimizer = tf.train.AdamOptimizer(lR) if optAda else tf.train.GradientDescentOptimizer(lR)
        optimizer = optimizer.apply_gradients(zip(gradients, vars))

        # select optimizer vars
        optVars = []
        for var in tf.global_variables(scope=tf.get_variable_scope().name):
            if var not in vars: optVars.append(var)

        return{
            'scope':                scope,
            'inC':                  inC,
            'inMT':                 inMT,
            'inV':                  inV,
            'wV':                   wV,
            'move':                 move,
            'reward':               reward,
            'inState':              inState,
            'singleZeroState':      singleZeroState,
            'probs':                probs,
            'finState':             finState,
            'optimizer':            optimizer,
            'loss':                 loss,
            'gN':                   gN,
            'vars':                 vars,
            'optVars':              optVars}