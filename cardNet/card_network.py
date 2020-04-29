"""

 2019 (c) piteren

"""

from functools import partial
import tensorflow as tf

from ptools.neuralmess.base_elements import my_initializer
from ptools.neuralmess.layers import lay_dense
from ptools.neuralmess.encoders import enc_DRT, enc_TNS


# cards encoder graph (Transformer for 7 cards representations)
def card_enc(
        train_flag,                 # train flag (bool tensor)
        c_ids,                      # seven cards (ids tensor)
        tat_case :bool=     False,  # task attention transformer architecture
        c_embW :int=        24,     # cards embedding width
        intime_drop :float= 0.0,
        infeat_drop :float= 0.0,
        in_proj :int=       None,
        n_layers :int=      8,
        dense_mul :int=     4,      # transformer dense multiplication
        dropout :float=     0.0,    # transformer dropout
        verb=               0):

    if verb > 0: print('\nBuilding card encoder...')

    with tf.variable_scope('card_enc'):

        c_emb = tf.get_variable(  # cards embeddings
            name=           'c_emb',
            shape=          [53, c_embW],  # one card for 'no_card'
            dtype=          tf.float32,
            initializer=    my_initializer())
        in_cemb = tf.nn.embedding_lookup(params=c_emb, ids=c_ids)
        if verb > 1: print(' > in_cemb:', in_cemb)
        hist_summ = [tf.summary.histogram('1.c_emb', c_emb, family='enc_input')]

        my_cemb = tf.get_variable(  # my cards embeddings
            name=           'my_cemb',
            shape=          [2, c_emb.shape[-1]],
            dtype=          tf.float32,
            initializer=    my_initializer())
        my_celook = tf.nn.embedding_lookup(params=my_cemb, ids=[0,0,1,1,1,1,1])
        if verb > 1: print(' > my_celook:', my_celook)
        in_cemb += my_celook

        seq_len = 7  # sequence length (time)
        seq_width = c_embW  # sequence width (features)
        batch_size = tf.shape(in_cemb)[-3]

        # time (per vector) dropout
        if intime_drop:
            time_drop = tf.ones(shape=[batch_size,seq_len])
            time_drop = tf.layers.dropout(
                inputs=     time_drop,
                rate=       intime_drop,
                training=   train_flag,
                seed=       121)
            in_cemb = in_cemb * tf.expand_dims(time_drop, axis=-1)

        # feature (constant in time) dropout
        if infeat_drop:
            feats_drop = tf.ones(shape=[batch_size, seq_width])
            feats_drop = tf.layers.dropout(
                inputs=     feats_drop,
                rate=       infeat_drop,
                training=   train_flag,
                seed=       121)
            in_cemb = in_cemb * tf.expand_dims(feats_drop, axis=-2)

        # TODO: what is that : ?
        """
        # sequence layer norm (on (dropped)input, always)
        in_cemb = tf.contrib.layers.layer_norm(
            inputs=             in_cemb,
            begin_norm_axis=    -2,
            begin_params_axis=  -2)
        if verb > 1: print(' > normalized in_cemb:', in_cemb)
        hist_summ.append(tf.summary.histogram('2.in_cemb.LN', in_cemb, family='enc_input'))
        """

        # input projection (without activation)
        if in_proj:
            in_cemb = lay_dense(
                input=          in_cemb,
                units=          in_proj,
                name=           'c_proj',
                reuse=          tf.AUTO_REUSE,
                use_bias=        False)
            if verb > 1: print(' > in_cemb projected:', in_cemb)
        elif verb > 1: print(' > in_cemb:', in_cemb)

        enc_out = enc_TNS(
            in_seq=         in_cemb,
            name=           'TAT' if tat_case else 'TNS',
            seq_out=        not tat_case,
            add_PE=         False,
            n_blocks=       n_layers,
            n_heads=        1,
            dense_mul=      dense_mul,
            max_seq_len=    7,
            dropout=        dropout,
            dropout_att=    0,
            drop_flag=      train_flag,
            n_hist=        3,
            verb=           verb)
        output = enc_out['output']
        if not tat_case:
            output = tf.unstack(output, axis=-2)
            output = tf.concat(output, axis=-1)
            if verb > 1:print(' > encT reshaped output:', output)
        elif verb > 1: print(' > encT output:', output)

        enc_vars = tf.global_variables(scope=tf.get_variable_scope().name)

    return {
        'output':       output,
        'enc_vars':     enc_vars,
        'hist_summ':    enc_out['hist_summ'] + hist_summ,
        'zeroes':       enc_out['zeroes']}

# cards network graph (FWD)
def card_net(
        tat_case :bool=     False,
        c_embW :int=        24,
        intime_drop: float= 0.0,
        infeat_drop: float= 0.0,
        in_proj :int=       None,   # None, 0 or int
        # TRNS
        n_layers :int=      8,
        dense_mul=          4,
        dropout=            0.0,    # dropout of encoder transformer
        # DRT
        dense_proj=         None,   # None, 0 or int
        dr_layers=          2,      # None, 0 or int
        dropout_DR=         0.0,    # DR dropout
        # train parameters
        opt_class=          partial(tf.compat.v1.train.AdamOptimizer, beta1=0.7, beta2=0.7),
        iLR=                1e-3,
        warm_up=            10000,
        ann_base=           0.999,
        ann_step=           0.04,
        avt_SVal=           0.1,
        avt_window=         500,
        avt_max_upd=        1.5,
        do_clip=            False,#True,
        verb=               0,
        **kwargs):

    with tf.variable_scope('card_net'):

        train_PH = tf.placeholder_with_default(  # train placeholder
            input=          False,
            name=           'train_PH',
            shape=          [])

        inA_PH = tf.placeholder( # 7 cards of A
            name=           'inA_PH',
            dtype=          tf.int32,
            shape=          [None, 7])  # [bsz,7cards]

        inB_PH = tf.placeholder( # 7 cards of B
            name=           'inB_PH',
            dtype=          tf.int32,
            shape=          [None, 7])  # [bsz,7cards]

        won_PH = tf.placeholder( # wonPH class (labels of winner 0,1-A,B wins,2-draw)
            name=           'won_PH',
            dtype=          tf.int32,
            shape=          [None])  # [bsz]

        rnkA_PH = tf.placeholder( # rank A class (labels <0,8>)
            name=           'rnkA_PH',
            dtype=          tf.int32,
            shape=          [None])  # [bsz]

        rnkB_PH = tf.placeholder( # rank B class (labels <0,8>)
            name=           'rnkB_PH',
            dtype=          tf.int32,
            shape=          [None])  # [bsz]

        mcA_PH = tf.placeholder( # chances of winning for A (montecarlo)
            name=           'mcA_PH',
            dtype=          tf.float32,
            shape=          [None])  # [bsz]

        # encoders for A and B
        enc_outL = []
        for cPH in [inA_PH, inB_PH]:
            enc_outL.append(card_enc(
                c_ids=          cPH,
                c_embW=         c_embW,
                train_flag=     train_PH,
                tat_case=       tat_case,
                intime_drop=    intime_drop,
                infeat_drop=    infeat_drop,
                in_proj=        in_proj,
                dense_mul=      dense_mul,
                dropout=        dropout,
                n_layers=       n_layers,
                verb=           verb))

        enc_vars =      enc_outL[0]['enc_vars']     # encoder variables (with cards embeddings)
        zsL =           enc_outL[0]['zeroes']       # get nn_zeros
        hist_summ =     enc_outL[0]['hist_summ']    # get histograms from A

        # where all cards of A are known
        where_all_ca = tf.reduce_max(inA_PH, axis=-1)
        where_all_ca = tf.where(
            condition=  where_all_ca < 52,
            x=          tf.ones_like(where_all_ca),
            y=          tf.zeros_like(where_all_ca))
        if verb > 1: print('\n > where_all_ca', where_all_ca)
        where_all_caF = tf.cast(where_all_ca, dtype=tf.float32) # cast to float

        # projection to 9 ranks A
        logits_RA = lay_dense(
            input=      enc_outL[0]['output'],
            units=      9,
            name=       'dense_RC',
            reuse=      tf.AUTO_REUSE,
            use_bias=    False)
        loss_RA = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss rank A
            labels=     rnkA_PH,
            logits=     logits_RA)
        loss_RA = tf.reduce_mean(loss_RA * where_all_caF) # lossRA masked (where all cards @A)

        # projection to 9 ranks B
        logits_RB = lay_dense(
            input=      enc_outL[1]['output'],
            units=      9,
            name=       'dense_RC',
            reuse=      tf.AUTO_REUSE,
            use_bias=    False)
        loss_RB = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss rank B
            labels=     rnkB_PH,
            logits=     logits_RB)
        loss_RB = tf.reduce_mean(loss_RB)

        loss_R = loss_RA + loss_RB
        if verb > 1: print(' > loss_R:', loss_R)

        # winner classifier (on concatenated representations)
        out_conc = tf.concat([enc_outL[0]['output'],enc_outL[1]['output']], axis=-1)
        if verb > 1: print(' > out_conc:', out_conc)
        if dr_layers:
            enc_out = enc_DRT(
                input=          out_conc,
                name=           'drC',
                lay_width=      dense_proj,
                n_layers=       dr_layers,
                dropout=        dropout_DR,
                training_flag=  train_PH,
                n_hist=         0,
                verb=           verb)
            out_conc =  enc_out['output']
            zsL +=      enc_out['zeroes']

        # projection to 3 winner logits
        logits_W = lay_dense(
            input=          out_conc,
            units=          3,
            name=           'dense_W',
            reuse=          tf.AUTO_REUSE,
            use_bias=        False)
        if verb > 1: print(' > logits_W:', logits_W)
        loss_W = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss wonPH
            labels=     won_PH,
            logits=     logits_W)
        loss_W = tf.reduce_mean(loss_W * where_all_caF) # loss winner classifier, masked
        if verb > 1: print(' > loss_W:', loss_W)

        # projection to probability of winning of A cards (regression value)
        a_WP = lay_dense(
            input=          enc_outL[0]['output'],
            units=          1,
            name=           'dense_WP',
            reuse=          tf.AUTO_REUSE,
            activation=     tf.nn.relu,
            use_bias=        False)
        a_WP = tf.reshape(a_WP, shape=[-1])
        if verb > 1: print(' > player a win probability:', a_WP)
        loss_AWP = tf.losses.mean_squared_error(
            labels=         mcA_PH,
            predictions=    a_WP)
        if verb > 1: print(' > loss_AWP:', loss_AWP)

        diff_AWP = tf.sqrt(tf.square(mcA_PH-a_WP))
        diff_AWP_mn = tf.reduce_mean(diff_AWP)
        diff_AWP_mx = tf.reduce_max(diff_AWP)

        loss = loss_W + loss_R + loss_AWP # this is how total loss is constructed

        # accuracy of winner classifier scaled by where all cards
        predictions_W = tf.argmax(logits_W, axis=-1, output_type=tf.int32)
        if verb > 1: print(' > predictionsW:', predictions_W)
        correct_W = tf.equal(predictions_W, won_PH)
        if verb > 1: print(' > correct_W:', correct_W)
        correct_WF = tf.cast(correct_W, dtype=tf.float32)
        correct_WF_where = correct_WF * where_all_caF
        acc_W = tf.reduce_sum(correct_WF_where) / tf.reduce_sum(where_all_caF)
        if verb > 1: print(' > acc_W:', acc_W)

        # accuracy of winner classifier per class scaled by where all cards
        oh_won = tf.one_hot(indices=won_PH, depth=3) # OH [batch,3], 1 where wins, dtype tf.float32
        oh_won_where = oh_won * tf.stack([where_all_caF]*3, axis=1) # masked where all cards
        won_density = tf.reduce_mean(oh_won_where, axis=0) # [3] measures density of 1 @batch per class
        oh_correct = tf.where(condition=correct_W, x=oh_won_where, y=tf.zeros_like(oh_won)) # [batch,3]
        won_corr_density = tf.reduce_mean(oh_correct, axis=0)
        acc_WC = won_corr_density / won_density

        oh_notcorrect_W = tf.where(condition=tf.logical_not(correct_W), x=oh_won, y=tf.zeros_like(oh_won)) # OH wins where not correct
        oh_notcorrect_W *= tf.stack([where_all_caF]*3, axis=1) # masked with all cards

        # acc of rank(B)
        predictions_R = tf.argmax(logits_RB, axis=-1, output_type=tf.int32)
        correct_R = tf.equal(predictions_R, rnkB_PH)
        acc_R = tf.reduce_mean(tf.cast(correct_R, dtype=tf.float32))
        if verb > 1: print(' > acc_R:', acc_R)

        # acc of rank(B) per class
        ohRnkB = tf.one_hot(indices=rnkB_PH, depth=9)
        rnkBdensity = tf.reduce_mean(ohRnkB, axis=0)
        ohCorrectR = tf.where(condition=correct_R, x=ohRnkB, y=tf.zeros_like(ohRnkB))
        rnkBcorrDensity = tf.reduce_mean(ohCorrectR, axis=0)
        acc_RC = rnkBcorrDensity/rnkBdensity

        oh_notcorrect_R = tf.where(condition=tf.logical_not(correct_R), x=ohRnkB, y=tf.zeros_like(ohRnkB)) # OH ranks where not correct

        cls_vars = tf.global_variables(scope=tf.get_variable_scope().name)
        cls_vars = [var for var in cls_vars if var not in enc_vars]

    return{
        'train_PH':             train_PH,
        'inA_PH':               inA_PH,
        'inB_PH':               inB_PH,
        'won_PH':               won_PH,
        'rnkA_PH':              rnkA_PH,
        'rnkB_PH':              rnkB_PH,
        'mcA_PH':               mcA_PH,
        'loss':                 loss, # total loss for training (OPT)
        'loss_W':               loss_W, # loss of winner classifier
        'loss_R':               loss_R, # loss of rank classifier
        'loss_AWP':             loss_AWP, # loss of prob win (value) of A
        'diff_AWP_mn':          diff_AWP_mn,
        'diff_AWP_mx':          diff_AWP_mx,
        'acc_W':                acc_W,
        'acc_WC':               acc_WC,
        'predictions_W':        predictions_W,
        'oh_notcorrect_W':      oh_notcorrect_W,
        'acc_R':                acc_R,
        'acc_RC':               acc_RC,
        'predictions_R':        predictions_R,
        'oh_notcorrect_R':      oh_notcorrect_R,
        'hist_summ':            tf.summary.merge(hist_summ),
        'zeroes':               tf.concat(zsL, axis=-1),
        'enc_vars':             enc_vars,
        'cls_vars':             cls_vars}