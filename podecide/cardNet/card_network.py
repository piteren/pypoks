"""

 2019 (c) piteren

    cards encoding network

"""

from functools import partial
import tensorflow as tf

from ptools.neuralmess.base_elements import my_initializer
from ptools.neuralmess.layers import lay_dense, tf_drop
from ptools.neuralmess.encoders import enc_DRT, enc_TNS


# cards encoder graph (Transformer for 7 cards representations)
def cards_enc(
        train_flag,                     # train flag (bool tensor)
        c_ids,                          # seven cards (ids tensor)
        tat_case :bool=     False,      # task attention transformer architecture
        emb_width :int=     24,         # cards embedding width
        t_drop :float=      0,
        f_drop :float=      0,
        in_proj :int=       None,
        n_layers :int=      8,
        dense_mul :int=     4,          # transformer dense multiplication
        activation=         tf.nn.relu,
        dropout :float=     0,          # transformer dropout
        # DRT (for concat enc)
        seed=               12321,
        verb=               0):

    if verb > 0: print('\nBuilding card encoder...')

    with tf.variable_scope('cards_enc'):

        zsL = []
        hist_summ = []

        c_emb = tf.get_variable(  # cards embeddings
            name=           'c_emb',
            shape=          [53, emb_width],  # one card for 'no_card'
            dtype=          tf.float32,
            initializer=    my_initializer(seed=seed))
        hist_summ += [tf.summary.histogram('1.c_emb', c_emb, family='c_emb')]

        c_emb_look = tf.nn.embedding_lookup(params=c_emb, ids=c_ids)
        if verb > 1: print(' > 1.c_emb_look:', c_emb_look)

        myc_emb = tf.get_variable(  # my cards embeddings
            name=           'myc_emb',
            shape=          [2, c_emb.shape[-1]],
            dtype=          tf.float32,
            initializer=    my_initializer(seed=seed))

        myc_emb_look = tf.nn.embedding_lookup(params=myc_emb, ids=[0,0,1,1,1,1,1])
        if verb > 1: print(' > myc_emb_look:', myc_emb_look)

        input = c_emb_look + myc_emb_look

        if t_drop or f_drop:
            input = tf_drop(
                input=      input,
                time_drop=  t_drop,
                feat_drop=  f_drop,
                train_flag= train_flag,
                seed=       seed)

        # input projection (without activation)
        if in_proj:
            input = lay_dense(
                input=          input,
                units=          in_proj,
                name=           'c_proj',
                reuse=          tf.AUTO_REUSE,
                use_bias=       False,
                seed=           seed)
            if verb > 1: print(' > input projected:', input)
        elif verb > 1: print(' > input:', input)

        enc_out = enc_TNS(
            in_seq=         input,
            name=           'TAT' if tat_case else 'TNS',
            seq_out=        not tat_case,
            add_PE=         False,
            n_blocks=       n_layers,
            n_heads=        1,
            dense_mul=      dense_mul,
            activation=     activation,
            max_seq_len=    7,
            dropout=        dropout,
            dropout_att=    0,
            drop_flag=      train_flag,
            seed=           seed,
            n_hist=         3,
            verb=           verb)
        output =     enc_out['output']
        zsL +=       enc_out['zeroes']
        hist_summ += enc_out['hist_summ']
        if not tat_case:
            output = tf.unstack(output, axis=-2)
            output = tf.concat(output, axis=-1)
            if verb > 1:print(' > encT reshaped output:', output)
        elif verb > 1: print(' > encT output:', output)

        enc_vars = tf.global_variables(scope=tf.get_variable_scope().name)

    return {
        'output':       output,
        'enc_vars':     enc_vars,
        'hist_summ':    hist_summ,
        'zeroes':       zsL}

# cards network graph (FWD)
def card_net(
        name=               'card_net',
        tat_case :bool=     False,
        emb_width :int=     24,
        t_drop: float=      0,
        f_drop: float=      0,
        in_proj :int=       None,   # None, 0 or int
        activation=         tf.nn.relu,
        # TRNS
        n_layers :int=      8,
        dense_mul=          4,
        dropout=            0,      # dropout of encoder transformer
        # DRT & classif
        dense_proj=         None,   # None, 0 or int
        dr_layers=          2,      # None, 0 or int
        dr_scale=           6,
        dropout_DR=         0,      # DR dropout
        # train parameters
        opt_class=          partial(tf.compat.v1.train.AdamOptimizer, beta1=0.7, beta2=0.7),
        iLR=                1e-3,
        warm_up=            10000,
        ann_base=           0.999,
        ann_step=           0.04,
        n_wup_off=          1,
        avt_SVal=           0.1,
        avt_window=         500,
        avt_max_upd=        1.5,
        do_clip=            False,#True,
        seed=               12321,
        verb=               0):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        zsL = []
        hist_summ = []

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

        # cards encoders for A and B
        enc_outL = []
        for cPH in [inA_PH, inB_PH]:
            enc_outL.append(cards_enc(
                c_ids=          cPH,
                emb_width=      emb_width,
                train_flag=     train_PH,
                tat_case=       tat_case,
                t_drop=         t_drop,
                f_drop=         f_drop,
                in_proj=        in_proj,
                dense_mul=      dense_mul,
                activation=     activation,
                dropout=        dropout,
                n_layers=       n_layers,
                seed=           seed,
                verb=           verb))

        enc_vars =      enc_outL[0]['enc_vars']     # encoder variables (with cards embeddings)
        zsL +=          enc_outL[0]['zeroes']       # get nn_zeros from A
        hist_summ +=    enc_outL[0]['hist_summ']    # get histograms from A

        # where all cards of A are known
        where_all_ca = tf.reduce_max(inA_PH, axis=-1)
        where_all_ca = tf.where(
            condition=  where_all_ca < 52,
            x=          tf.ones_like(where_all_ca),
            y=          tf.zeros_like(where_all_ca))
        if verb > 1: print('\n > where_all_ca', where_all_ca)
        where_all_caF = tf.cast(where_all_ca, dtype=tf.float32) # cast to float

        # rank A classifier
        logits_RA = lay_dense(
            input=      enc_outL[0]['output'],
            units=      9,
            name=       'dense_RC',
            reuse=      tf.AUTO_REUSE,
            use_bias=   False,
            seed=       seed)
        loss_RA = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss rank A
            labels=     rnkA_PH,
            logits=     logits_RA)
        loss_RA = tf.reduce_mean(loss_RA * where_all_caF) # lossRA masked (where all cards @A)

        # rank B classifier
        logits_RB = lay_dense(
            input=      enc_outL[1]['output'],
            units=      9,
            name=       'dense_RC',
            reuse=      tf.AUTO_REUSE,
            use_bias=   False,
            seed=       seed)
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
                name=           'drt_W',
                lay_width=      dense_proj,
                n_layers=       dr_layers,
                dns_scale=      dr_scale,
                activation=     activation,
                dropout=        dropout_DR,
                training_flag=  train_PH,
                n_hist=         0,
                seed=           seed,
                verb=           verb)
            out_conc =   enc_out['output']
            zsL +=       enc_out['zeroes']
            hist_summ += enc_out['hist_summ']
        logits_W = lay_dense( # projection to 3 winner logits
            input=          out_conc,
            units=          3,
            name=           'dense_W',
            reuse=          tf.AUTO_REUSE,
            use_bias=       False,
            seed=           seed)
        if verb > 1: print(' > logits_W:', logits_W)
        loss_W = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss wonPH
            labels=     won_PH,
            logits=     logits_W)
        loss_W = tf.reduce_mean(loss_W * where_all_caF) # loss winner classifier, masked
        if verb > 1: print(' > loss_W:', loss_W)

        # probability of A winning regressor
        a_WP = lay_dense(
            input=          enc_outL[0]['output'],
            units=          1,
            name=           'dense_WP',
            reuse=          tf.AUTO_REUSE,
            activation=     activation,
            use_bias=       False,
            seed=           seed)
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

        # accuracy of winner classifier (where all cards)
        predictions_W = tf.argmax(logits_W, axis=-1, output_type=tf.int32)
        if verb > 1: print(' > predictionsW:', predictions_W)
        correct_W = tf.equal(predictions_W, won_PH)
        if verb > 1: print(' > correct_W:', correct_W)
        correct_WF = tf.cast(correct_W, dtype=tf.float32)
        correct_WF_where = correct_WF * where_all_caF
        acc_W = tf.reduce_sum(correct_WF_where) / tf.reduce_sum(where_all_caF)
        if verb > 1: print(' > acc_W:', acc_W)

        # accuracy of winner classifier per class (where all cards)
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
        oh_rnkB = tf.one_hot(indices=rnkB_PH, depth=9)
        rnkB_density = tf.reduce_mean(oh_rnkB, axis=0)
        oh_correct_R = tf.where(condition=correct_R, x=oh_rnkB, y=tf.zeros_like(oh_rnkB))
        rnkB_corr_density = tf.reduce_mean(oh_correct_R, axis=0)
        acc_RC = rnkB_corr_density/rnkB_density

        oh_notcorrect_R = tf.where(condition=tf.logical_not(correct_R), x=oh_rnkB, y=tf.zeros_like(oh_rnkB)) # OH ranks where not correct

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
        'loss':                 loss,           # total training loss (sum)
        'loss_W':               loss_W,         # loss of winner classifier
        'loss_R':               loss_R,         # loss of rank classifier
        'loss_AWP':             loss_AWP,       # loss of A prob win
        'diff_AWP_mn':          diff_AWP_mn,    # min diff of A prob win
        'diff_AWP_mx':          diff_AWP_mx,    # max diff of A prob win
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