"""

 2019 (c) piteren

"""

from functools import partial
import tensorflow as tf

from putils.neuralmess.base_elements import my_initializer
from putils.neuralmess.layers import lay_dense
from putils.neuralmess.encoders import encDR, encTRNS


# cards encoder graph (Transformer for 7 cards representations)
def card_enc(
        c_ids,              # seven cards (tensor of ids)
        c_emb,              # cards embedding tensor
        train_flag,         # train flag tensor
        tat_case :bool,     # task attention transformer architecture
        in_proj,
        dense_mul,          # transformer dense multiplication
        dropout=    0.0,    # transformer dropout
        n_layers=   6,
        verb=       0):

    if verb > 0: print('\nBuilding card encoder...')

    in_cemb = tf.nn.embedding_lookup(params=c_emb, ids=c_ids)
    if verb > 1: print(' > in_cemb:', in_cemb)

    my_cemb = tf.get_variable(  # my cards embeddings
        name=           'my_cemb',
        shape=          [2, c_emb.shape[-1]],
        dtype=          tf.float32,
        initializer=    my_initializer())
    my_celook = tf.nn.embedding_lookup(params=my_cemb, ids=[0,0,1,1,1,1,1])
    if verb > 1: print(' > my_celook:', my_celook)
    in_cemb += my_celook

    # input projection
    if in_proj:
        proj_out = lay_dense(
            input=          in_cemb,
            units=          in_proj,
            name=           'c_proj',
            reuse=          tf.AUTO_REUSE,
            useBias=        False)
        in_cemb = proj_out['output']
        if verb > 1: print(' > in_cemb projected:', in_cemb)
    elif verb > 1: print(' > in_cemb:', in_cemb)

    encOUT = encTRNS(
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
        n_histL=        3,
        verb=           verb)
    output = encOUT['tns_out']
    if not tat_case:
        output = tf.unstack(output, axis=-2)
        output = tf.concat(output, axis=-1)
        if verb > 1:print(' > encT reshaped output:', output)
    elif verb > 1: print(' > encT output:', output)

    return {
        'output':       output,
        'hist_summ':    encOUT['hist_summ'],
        'nn_zeros':     encOUT['nn_zeros']}

# cards netetwork graph (FWD)
def card_net(
        tat_case :bool= False,
        c_embW :int=    24,
        n_layers :int=  8,
        in_proj :int=   None,   # None, 0 or int
        dense_mul=      4,
        dense_proj=     None,   # None, 0 or int
        dr_layers=      2,      # None, 0 or int
        dropout=        0.0,    # dropout of encoder transformer
        dropout_DR=     0.0,    # DR dropout
        # train parameters
        opt_class=      partial(tf.train.AdamOptimizer, beta1=0.7, beta2=0.7),
        iLR=            1e-3,
        warm_up=        10000,
        ann_base=       0.999,
        ann_step=       0.04,
        avt_SVal=       0.1,
        avt_window=     500,
        avt_max_upd=    1.5,
        do_clip=        True,
        verb=           0,
        **kwargs):

    train_PH = tf.placeholder_with_default(  # train placeholder
        input=          False,
        name=           'train_PH',
        shape=          [])

    c_emb = tf.get_variable(  # cards embeddings
        name=           'c_emb',
        shape=          [53, c_embW],  # one card for 'no_card'
        dtype=          tf.float32,
        initializer=    my_initializer())

    hist_summ = [tf.summary.histogram('cEMB', c_emb, family='cEMB')]

    inA_PH = tf.placeholder( # 7 cards of A
        name=           'inA_PH',
        dtype=          tf.int32,
        shape=          [None, 7])  # [bsz,7cards]

    inB_PH = tf.placeholder( # 7 cards of B
        name=           'inB_PH',
        dtype=          tf.int32,
        shape=          [None, 7])  # [bsz,7cards]

    won_PH = tf.placeholder( # wonPH class (lables of winner 0,1-ABwins,2-draw)
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
    ceA_out = card_enc(
        c_ids=          inA_PH,
        c_emb=          c_emb,
        train_flag=     train_PH,
        tat_case=       tat_case,
        in_proj=        in_proj,
        dense_mul=      dense_mul,
        dropout=        dropout,
        n_layers=       n_layers,
        verb=           verb)
    ceB_out = card_enc(
        c_ids=          inB_PH,
        c_emb=          c_emb,
        train_flag=     train_PH,
        tat_case=       tat_case,
        in_proj=        in_proj,
        dense_mul=      dense_mul,
        dropout=        dropout,
        n_layers=       n_layers,
        verb=           verb)

    # get nn_zeros
    nn_zerosA = ceA_out['nn_zeros']
    nn_zerosA = tf.reshape(tf.stack(nn_zerosA), shape=[-1])
    nn_zerosB = ceB_out['nn_zeros']
    nn_zerosB = tf.reshape(tf.stack(nn_zerosB), shape=[-1])
    hist_summ.append(ceA_out['hist_summ']) # get histograms from A

    # where all cards of A are known
    where_all_ca = tf.reduce_max(inA_PH, axis=-1)
    where_all_ca = tf.where(
        condition=  where_all_ca < 52,
        x=          tf.ones_like(where_all_ca),
        y=          tf.zeros_like(where_all_ca))
    if verb > 1: print('\n > where_all_ca', where_all_ca)
    where_all_caF = tf.cast(where_all_ca, dtype=tf.float32) # cast to float

    # projection to 9 ranks A
    dout_RA = lay_dense(
        input=      ceA_out['output'],
        units=      9,
        name=       'dense_RC',
        reuse=      tf.AUTO_REUSE,
        useBias=    False)
    logits_RA = dout_RA['output']
    loss_RA = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss rank A
        labels=     rnkA_PH,
        logits=     logits_RA)
    loss_RA = tf.reduce_mean(loss_RA * where_all_caF) # lossRA masked (where all cards @A)

    # projection to 9 ranks B
    dout_RB = lay_dense(
        input=      ceB_out['output'],
        units=      9,
        name=       'dense_RC',
        reuse=      tf.AUTO_REUSE,
        useBias=    False)
    logits_RB = dout_RB['output']
    loss_RB = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss rank B
        labels=     rnkB_PH,
        logits=     logits_RB)
    loss_RB = tf.reduce_mean(loss_RB)

    loss_R = loss_RA + loss_RB
    if verb > 1: print(' > loss_R:', loss_R)

    # winner classifier (on concatenated representations)
    out_conc = tf.concat([ceA_out['output'],ceB_out['output']], axis=-1)
    if verb > 1: print(' > out_conc:', out_conc)
    if dr_layers:
        enc_out = encDR(
            input=      out_conc,
            name=       'drC',
            layWidth=   dense_proj,
            nLayers=    dr_layers,
            dropout=    dropout_DR,
            dropFlagT=  train_PH,
            nHL=        0,
            verbLev=    verb)
        out_conc = enc_out['output']

    # projection to 3 winner logits
    dout_W = lay_dense(
        input=          out_conc,
        units=          3,
        name=           'dense_W',
        reuse=          tf.AUTO_REUSE,
        useBias=        False)
    logits_W = dout_W['output']
    if verb > 1: print(' > logits_W:', logits_W)
    loss_W = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss wonPH
        labels=     won_PH,
        logits=     logits_W)
    loss_W = tf.reduce_mean(loss_W * where_all_caF) # loss winner classifier, masked
    if verb > 1: print(' > loss_W:', loss_W)

    # projection to probability of winning of A cards (regression value)
    dout_WP = lay_dense(
        input=          ceA_out['output'],
        units=          1,
        name=           'dense_WP',
        reuse=          tf.AUTO_REUSE,
        activation=     tf.nn.relu,
        useBias=        False)
    a_WP = dout_WP['output']
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
    avg_acc_W = tf.reduce_sum(correct_WF_where) / tf.reduce_sum(where_all_caF)
    if verb > 1: print(' > avg_acc_W:', avg_acc_W)

    # accuracy of winner classifier per class scaled by where all cards
    ohWon = tf.one_hot(indices=won_PH, depth=3) # OH [batch,3], 1 where wins, dtype tf.float32
    ohWonWhere = ohWon * tf.stack([where_all_caF]*3, axis=1) # masked where all cards
    wonDensity = tf.reduce_mean(ohWonWhere, axis=0) # [3] measures density of 1 @batch per class
    ohCorrect = tf.where(condition=correct_W, x=ohWonWhere, y=tf.zeros_like(ohWon)) # [batch,3]
    wonCorrDensity = tf.reduce_mean(ohCorrect, axis=0)
    avgAccWC = wonCorrDensity / wonDensity

    ohNotCorrectW = tf.where(condition=tf.logical_not(correct_W), x=ohWon, y=tf.zeros_like(ohWon)) # OH wins where not correct
    ohNotCorrectW *= tf.stack([where_all_caF]*3, axis=1) # masked with all cards

    # acc of rank(B)
    predictionsR = tf.argmax(logits_RB, axis=-1, output_type=tf.int32)
    correctR = tf.equal(predictionsR, rnkB_PH)
    avgAccR = tf.reduce_mean(tf.cast(correctR, dtype=tf.float32))
    if verb > 1: print(' > avgAccR:', avgAccR)

    # acc of rank(B) per class
    ohRnkB = tf.one_hot(indices=rnkB_PH, depth=9)
    rnkBdensity = tf.reduce_mean(ohRnkB, axis=0)
    ohCorrectR = tf.where(condition=correctR, x=ohRnkB, y=tf.zeros_like(ohRnkB))
    rnkBcorrDensity = tf.reduce_mean(ohCorrectR, axis=0)
    avgAccRC = rnkBcorrDensity/rnkBdensity

    ohNotCorrectR = tf.where(condition=tf.logical_not(correctR), x=ohRnkB, y=tf.zeros_like(ohRnkB)) # OH ranks where not correct

    return{
        'trPH':                 train_PH,
        'inACPH':               inA_PH,
        'inBCPH':               inB_PH,
        'wonPH':                won_PH,
        'rnkAPH':               rnkA_PH,
        'rnkBPH':               rnkB_PH,
        'mcACPH':               mcA_PH,
        'loss':                 loss, # total loss for training (OPT)
        'lossW':                loss_W, # loss of winner classifier
        'lossR':                loss_R, # loss of rank classifier
        'lossPAWR':             loss_AWP, # loss of prob win (value) of A
        'diffPAWRmn':           diff_AWP_mn,
        'diffPAWRmx':           diff_AWP_mx,
        'avgAccW':              avg_acc_W,
        'avgAccWC':             avgAccWC,
        'predictionsW':         predictions_W,
        'ohNotCorrectW':        ohNotCorrectW,
        'accR':                 avgAccR,
        'accRC':                avgAccRC,
        'predictionsR':         predictionsR,
        'ohNotCorrectR':        ohNotCorrectR,
        'hist_summ':             tf.summary.merge(hist_summ),
        'nn_zerosA':             nn_zerosA,
        'nn_zerosB':             nn_zerosB}