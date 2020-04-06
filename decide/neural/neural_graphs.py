"""

 2019 (c) piteren

"""

import tensorflow as tf

from putils.lipytools.little_methods import short_scin
from putils.neuralmess.base_elements import my_initializer, num_var_floats
from putils.neuralmess.layers import lay_dense, zeroes
from putils.neuralmess.encoders import enc_CNN

from cardNet.card_network import card_enc


# base CNN+RES+CE+move neural graph (nemodel compatible)
def cnnCEM_GFN(
        name :str,
        train_ce :bool= True,   # train cards encoder
        c_embW :int=    12,     # card emb width
        n_lay=          12,     # number of CNNR layers
        width=          None,   # representation width (number of filters), for none uses input width
        activation=     tf.nn.relu, # TODO: maybe gelu?
        opt_class=      tf.train.GradientDescentOptimizer,
        iLR=            5e-3,#3e-4,
        warm_up=        100,    # since we do updates rarely, num of steps has to be small
        avt_SVal=       0.04,
        do_clip=        True,
        verb=           1,
        **kwargs):

    if verb>0: print('\nBuilding %s (CNN+RES+CE+M) graph...'%name)

    with tf.variable_scope(name):

        cards_PH = tf.placeholder(  # 7 cards placeholder
            name=   'cards_PH',
            dtype=  tf.int32,
            shape=  [None, None, 7])  # [bsz,seq,7cards]

        train_PH = tf.placeholder(  # train placeholder
            name=   'train_PH',
            dtype=  tf.bool,
            shape=  [])

        ce_out = card_enc(train_flag=train_PH, c_ids=cards_PH, c_embW=c_embW)
        cards_encoded = ce_out['output']
        enc_vars =      ce_out['enc_vars']
        enc_zsL =       ce_out['zeroes']
        if verb>1: print(' ### num of enc_vars (%d) %s'%(len(enc_vars),short_scin(num_var_floats(enc_vars))))
        if verb>1: print(' > cards encoded:', cards_encoded)

        switch_PH = tf.placeholder( # switch placeholder
            name=           'switch_PH',
            dtype=          tf.int32, # 0 for move, 1 for cards
            shape=          [None, None, 1])  # [bsz,seq,val]

        event_PH = tf.placeholder(  # event id placeholder
            name=           'event_PH',
            dtype=          tf.int32,
            shape=          [None, None])  # [bsz,seq]

        event_emb = tf.get_variable(  # event type embeddings
            name=           'event_emb',
            shape=          [12, cards_encoded.shape[-1]],
            dtype=          tf.float32,
            initializer=    my_initializer())

        event_in = tf.nn.embedding_lookup(params=event_emb, ids=event_PH)
        if verb>1: print(' > event_in:', event_in)

        switch = tf.cast(switch_PH, dtype=tf.float32)
        input = switch*cards_encoded + (1-switch)*event_in
        if verb>1: print(' > input (merged):', input)

        # projection without activation and bias
        if width:
            input = lay_dense(
                input=          input,
                units=          width,
                use_bias=        False)
            if verb>1: print(' > projected input (projected):', input)
        else: width = cards_encoded.shape[-1]

        # layer_norm
        sub_output = tf.contrib.layers.layer_norm(
            inputs=             input,
            begin_norm_axis=    -1,
            begin_params_axis=  -1)

        state_shape = [n_lay, 2, width]
        single_zero_state = tf.zeros(shape=state_shape)  # [n_lay,2,width]

        state_PH = tf.placeholder(
            name=           'state_PH',
            dtype=          tf.float32,
            shape=          [None] + state_shape) # [bsz,n_lay,2,width]

        cnn_enc_out = enc_CNN(
            input=          sub_output,
            history=        state_PH,
            n_layers=       n_lay,
            n_filters=      width,
            activation=     activation,
            n_hist=         0)
        out =       cnn_enc_out['output']
        fin_state = cnn_enc_out['state']
        cnn_zsL =   cnn_enc_out['zeroes']

        if verb > 1:
            print(' > out:', out)
            print(' > finState (split):', fin_state)

        # projection to logits
        logits = lay_dense(
            input=          out,
            units=          4, #TODO: hardcoded
            use_bias=        False)
        if verb>1: print(' > logits:', logits)

        probs = tf.nn.softmax(logits)

        cnn_vars = tf.trainable_variables(scope=tf.get_variable_scope().name)
        cnn_vars = [var for var in cnn_vars if var not in enc_vars]
        if verb>1: print(' ### num of cnn_vars (%d) %s'%(len(cnn_vars),short_scin(num_var_floats(cnn_vars))))

        correct_PH = tf.placeholder(  # "correct" move (label)
            name=           'correct_PH',
            dtype=          tf.int32,
            shape=          [None, None])  # [bsz,seq]

        rew_PH = tf.placeholder(  # reward for "correct" move
            name=           'rew_PH',
            dtype=          tf.float32,
            shape=          [None, None])  # [bsz,seq]

        # this loss is auto averaged with reduction parameter
        # loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss = loss(y_true=move, y_pred=logits, sample_weight=rew)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=     correct_PH,
            logits=     logits,
            weights=    rew_PH)

    train_vars = cnn_vars
    if train_ce: train_vars += enc_vars

    return{
        'name':                 name,
        'cards_PH':             cards_PH,
        'train_PH':             train_PH,
        'switch_PH':            switch_PH,
        'event_PH':             event_PH,
        'correct_PH':           correct_PH,
        'rew_PH':               rew_PH,
        'state_PH':             state_PH,
        'single_zero_state':    single_zero_state,
        'probs':                probs,
        'fin_state':            fin_state,
        'enc_zeroes':           tf.concat(enc_zsL, axis=-1),
        'cnn_zeroes':           tf.concat(cnn_zsL, axis=-1),
        'loss':                 loss,
        'enc_vars':             enc_vars,
        'cnn_vars':             cnn_vars,
        'train_vars':           train_vars}


if __name__ == "__main__":
    n = cnnCEM_GFN('pio',verb=2)