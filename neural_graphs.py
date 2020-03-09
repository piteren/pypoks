"""

 2019 (c) piteren

"""

import tensorflow as tf

from putils.lipytools.little_methods import short_scin
from putils.neuralmess.base_elements import my_initializer, num_var_floats
from putils.neuralmess.layers import lay_dense
from putils.neuralmess.encoders import encDRT

from cardNet.card_network import card_enc


# base LSTM neural graph
def lstm_GFN(
        scope :str,
        wC=         16,     # card emb width
        wMT=        1,      # move type emb width
        wV=         11,     # values vector width, holds player move data(type, pos, cash)
        nDR=        3,      # num of encDR lay
        cellW=      1024,   # cell width
        optAda=     True,
        lR=         7e-6):

    with tf.variable_scope(scope):

        print()
        inC = tf.placeholder(  # 7 cards
            name=           'inC',
            dtype=          tf.int32,
            shape=          [None, None, 7])  # [bsz,seq,7cards]

        cEMB = tf.get_variable(  # cards embeddings
            name=           'cEMB',
            shape=          [53, wC],  # one card for 'no_card'
            dtype=          tf.float32,
            initializer=    my_initializer())

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
            initializer=    my_initializer())

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

        encDRout = encDRT(
            input=      input,
            n_layers=   nDR,
            lay_width=  cellW,
            nHL=        0,
            verb=       1)
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

        denseOut = lay_dense(
            input=      out,
            units=      4,
            #activation= tf.nn.relu,
            useBias=    False,)
        logits = denseOut['output']
        print(' > logits:', logits)

        probs = tf.nn.softmax(logits)

        vars = tf.trainable_variables(scope=tf.get_variable_scope().name)
        print(' ### num of vars %s' % short_scin(num_var_floats(vars)))

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

# base CNN+RES neural graph (nemodel compatible)
def cnn_GFN(
        name :str,
        c_embW :int=    16,     # card emb width
        mt_embW :int=   1,      # move type emb width
        mvW :int=       11,     # values vector width, holds player move data(type, pos, cash)
        n_lay=          24,     # number of CNNR layers
        width=          512,    # representation width (number of filters)
        iLR=            7e-7,
        **kwargs):

    with tf.variable_scope(name):

        print()
        c_PH = tf.placeholder(  # 7 cards
            name=           'c_PH',
            dtype=          tf.int32,
            shape=          [None, None, 7])  # [bsz,seq,7cards]

        c_emb = tf.get_variable(  # cards embeddings
            name=           'c_emb',
            shape=          [53, c_embW],  # one card for 'no_card'
            dtype=          tf.float32,
            initializer=    my_initializer())

        in_cemb = tf.nn.embedding_lookup(params=c_emb, ids=c_PH)
        print(' > in_cemb:', in_cemb)
        in_cemb = tf.unstack(in_cemb, axis=-2)
        in_cemb = tf.concat(in_cemb, axis=-1)
        print(' > in_cemb (flattened):', in_cemb)

        mt_PH = tf.placeholder(  # event type
            name=           'mt_PH',
            dtype=          tf.int32,
            shape=          [None, None, 4])  # [bsz,seq,2*2oppon]

        mt_emb = tf.get_variable(  # event type embeddings
            name=           'mt_emb',
            shape=          [5, mt_embW],  # 4 moves + no_move
            dtype=          tf.float32,
            initializer=    my_initializer())

        in_mt = tf.nn.embedding_lookup(params=mt_emb, ids=mt_PH)
        print(' > in_memb:', in_mt)
        in_mt = tf.unstack(in_mt, axis=-2)
        in_mt = tf.concat(in_mt, axis=-1)
        print(' > mt_emb (flattened):', in_mt)

        mv_PH = tf.placeholder(  # event float values
            name=           'mv_PH',
            dtype=          tf.float32,
            shape=          [None, None, 4, mvW])  # [bsz,seq,2*2,vec]

        in_mv = tf.unstack(mv_PH, axis=-2)
        in_mv = tf.concat(in_mv, axis=-1)
        print(' > inV (flattened):', in_mv)

        input = tf.concat([in_cemb, in_mt, in_mv], axis=-1)
        print(' > input (concatenated):', input)  # width = self.wC*3 + (self.wMT + self.wV)*2

        # projection without activation and bias
        dense_out = lay_dense(
            input=          input,
            units=          width,
            useBias=        False)
        input = dense_out['output']
        print(' > projected input (projected):', input)

        state_shape = [n_lay, 2, width]
        state_PH = tf.placeholder(
            name=           'state_PH',
            dtype=          tf.float32,
            shape=          [None] + state_shape) # [bsz,nLay,2,reW]

        single_zero_state = tf.zeros(shape=state_shape) # [nLay,2,reW]

        # unstack layers of state_PH
        state_lays = tf.unstack(state_PH, axis=-3)
        print(' > state_lays len %d of:' %len(state_lays), state_lays[0])

        # layer_norm
        sub_output = tf.contrib.layers.layer_norm(
            inputs=             input,
            begin_norm_axis=    -1,
            begin_params_axis=  -1)

        input_lays = []
        for depth in range(n_lay):

            input_lays.append(tf.concat([state_lays[depth],sub_output], axis=-2))
            print(' > lay input of %d lay'%depth, input_lays[-1])

            with tf.variable_scope('cnnREncLay_%d' % depth):
                cnn_lay = tf.layers.Conv1D(
                    filters=            width,
                    kernel_size=        3,
                    dilation_rate=      1,
                    activation=         None,
                    use_bias=           True,
                    kernel_initializer= my_initializer(),
                    padding=            'valid',
                    data_format=        'channels_last')

            cnn_out = cnn_lay(input_lays[-1]) # cnn
            cnn_out = tf.nn.relu(cnn_out) # activation
            print(' > cnn_out (%d lay)' % depth, cnn_out)
            sub_output += cnn_out
            print(' > sub_output (RES %d lay)' % depth, cnn_out)
            sub_output = tf.contrib.layers.layer_norm(
                inputs=             sub_output,
                begin_norm_axis=    -1,
                begin_params_axis=  -1)

        out = sub_output
        print(' > out:', out)

        state = tf.stack(input_lays, axis=-3)
        print(' > state (stacked):', state)
        fin_state = tf.split(state, num_or_size_splits=[-1,2], axis=-2)[1] # TODO: may fin state be wider than 2 ...?
        print(' > finState (split):', fin_state)

        # projection to logits
        dense_out = lay_dense(
            input=          out,
            units=          4, #TODO: hardcoded
            useBias=        False)
        logits = dense_out['output']
        print(' > logits:', logits)

        probs = tf.nn.softmax(logits)

        train_vars = tf.trainable_variables(scope=tf.get_variable_scope().name)
        print(' ### num of train_vars %s' % short_scin(num_var_floats(train_vars)))

        cmv_PH = tf.placeholder(  # "correct" move (label)
            name=           'cmv_PH',
            dtype=          tf.int32,
            shape=          [None, None])  # [bsz,seq]

        rew_PH = tf.placeholder(  # reward for "correct" move
            name=           'rew_PH',
            dtype=          tf.float32,
            shape=          [None, None])  # [bsz,seq]

        rew = rew_PH/500 # lineary scale rewards #TODO: hardcoded

        # this loss is auto averaged with reduction parameter
        # loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss = loss(y_true=move, y_pred=logits, sample_weight=rew)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=     cmv_PH,
            logits=     logits,
            weights=    rew)

        return{
            'name':                 name,
            'c_PH':                 c_PH,
            'mt_PH':                mt_PH,
            'mv_PH':                mv_PH,
            'cmv_PH':               cmv_PH,
            'rew_PH':               rew_PH,
            'state_PH':             state_PH,
            'single_zero_state':    single_zero_state,
            'probs':                probs,
            'fin_state':            fin_state,
            'loss':                 loss,
            'train_vars':           train_vars}

# base CNN+RES+CE neural graph (nemodel compatible)
def cnnCE_GFN(
        name :str,
        train_ce :bool= False,  # train cards encoder
        c_embW :int=    24,     # card emb width
        mt_embW :int=   12,     # move type emb width
        mvW :int=       11,     # values vector width, holds player move data(type, pos, cash)
        n_lay=          6,#24,     # number of CNNR layers
        width=          128,#512,    # representation width (number of filters)
        iLR=            1e-5,
        **kwargs):

    print('\nBuilding CNN+RES+CE graph...')

    with tf.variable_scope('card_net'):

        c_PH = tf.placeholder(  # 7 cards
            name=   'c_PH',
            dtype=  tf.int32,
            shape=  [None, None, 7])  # [bsz,seq,7cards]

        train_PH = tf.placeholder_with_default(  # train placeholder
            input=  False,
            name=   'train_PH',
            shape=  [])

        cenc_out = card_enc(train_flag= train_PH, c_ids=c_PH, c_embW=c_embW)
        in_cenc =   cenc_out['output']
        enc_vars =  cenc_out['enc_vars']
        print(' ### num of enc_vars %s' % short_scin(num_var_floats(enc_vars)))
        print(' > input cards encoded:', in_cenc)

    with tf.variable_scope(name):

        mt_PH = tf.placeholder(  # event type
            name=           'mt_PH',
            dtype=          tf.int32,
            shape=          [None, None, 4])  # [bsz,seq,2*2oppon]

        mt_emb = tf.get_variable(  # event type embeddings
            name=           'mt_emb',
            shape=          [5, mt_embW],  # 4 moves + no_move
            dtype=          tf.float32,
            initializer=    my_initializer())

        in_mt = tf.nn.embedding_lookup(params=mt_emb, ids=mt_PH)
        print(' > in_memb:', in_mt)
        in_mt = tf.unstack(in_mt, axis=-2)
        in_mt = tf.concat(in_mt, axis=-1)
        print(' > mt_emb (flattened):', in_mt)

        mv_PH = tf.placeholder(  # event float values
            name=           'mv_PH',
            dtype=          tf.float32,
            shape=          [None, None, 4, mvW])  # [bsz,seq,2*2,vec]

        in_mv = tf.unstack(mv_PH, axis=-2)
        in_mv = tf.concat(in_mv, axis=-1)
        print(' > inV (flattened):', in_mv)

        input = tf.concat([in_cenc, in_mt, in_mv], axis=-1)
        print(' > input (concatenated):', input)  # width = self.wC*3 + (self.wMT + self.wV)*2

        # projection without activation and bias
        dense_out = lay_dense(
            input=          input,
            units=          width,
            useBias=        False)
        input = dense_out['output']
        print(' > projected input (projected):', input)

        state_shape = [n_lay, 2, width]
        state_PH = tf.placeholder(
            name=           'state_PH',
            dtype=          tf.float32,
            shape=          [None] + state_shape) # [bsz,nLay,2,reW]

        single_zero_state = tf.zeros(shape=state_shape) # [nLay,2,reW]

        # unstack layers of state_PH
        state_lays = tf.unstack(state_PH, axis=-3)
        print(' > state_lays len %d of:' %len(state_lays), state_lays[0])

        # layer_norm
        sub_output = tf.contrib.layers.layer_norm(
            inputs=             input,
            begin_norm_axis=    -1,
            begin_params_axis=  -1)

        input_lays = []
        for depth in range(n_lay):

            input_lays.append(tf.concat([state_lays[depth],sub_output], axis=-2))
            print(' > lay input of %d lay'%depth, input_lays[-1])

            with tf.variable_scope('cnnREncLay_%d' % depth):
                cnn_lay = tf.layers.Conv1D(
                    filters=            width,
                    kernel_size=        3,
                    dilation_rate=      1,
                    activation=         None,
                    use_bias=           True,
                    kernel_initializer= my_initializer(),
                    padding=            'valid',
                    data_format=        'channels_last')

            cnn_out = cnn_lay(input_lays[-1]) # cnn
            cnn_out = tf.nn.relu(cnn_out) # activation
            print(' > cnn_out (%d lay)' % depth, cnn_out)
            sub_output += cnn_out
            print(' > sub_output (RES %d lay)' % depth, cnn_out)
            sub_output = tf.contrib.layers.layer_norm(
                inputs=             sub_output,
                begin_norm_axis=    -1,
                begin_params_axis=  -1)

        out = sub_output
        print(' > out:', out)

        state = tf.stack(input_lays, axis=-3)
        print(' > state (stacked):', state)
        fin_state = tf.split(state, num_or_size_splits=[-1,2], axis=-2)[1] # TODO: may fin state be wider than 2 ...?
        print(' > finState (split):', fin_state)

        # projection to logits
        dense_out = lay_dense(
            input=          out,
            units=          4, #TODO: hardcoded
            useBias=        False)
        logits = dense_out['output']
        print(' > logits:', logits)

        probs = tf.nn.softmax(logits)

        cnn_vars = tf.trainable_variables(scope=tf.get_variable_scope().name)
        print(' ### num of cnn_vars %s' % short_scin(num_var_floats(cnn_vars)))

        cmv_PH = tf.placeholder(  # "correct" move (label)
            name=           'cmv_PH',
            dtype=          tf.int32,
            shape=          [None, None])  # [bsz,seq]

        rew_PH = tf.placeholder(  # reward for "correct" move
            name=           'rew_PH',
            dtype=          tf.float32,
            shape=          [None, None])  # [bsz,seq]

        rew = rew_PH/500 # lineary scale rewards #TODO: hardcoded

        # this loss is auto averaged with reduction parameter
        # loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss = loss(y_true=move, y_pred=logits, sample_weight=rew)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=     cmv_PH,
            logits=     logits,
            weights=    rew)

    train_vars = cnn_vars
    if train_ce: train_vars += enc_vars

    return{
        'name':                 name,
        'c_PH':                 c_PH,
        'mt_PH':                mt_PH,
        'mv_PH':                mv_PH,
        'cmv_PH':               cmv_PH,
        'rew_PH':               rew_PH,
        'state_PH':             state_PH,
        'single_zero_state':    single_zero_state,
        'probs':                probs,
        'fin_state':            fin_state,
        'loss':                 loss,
        'enc_vars':             enc_vars,
        'cnn_vars':             cnn_vars,
        'train_vars':           train_vars}