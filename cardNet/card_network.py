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
        sevc_PH,            # seven cards placeholder
        cEMB,               # cards embedding tensor
        trPH,               # train placeholder
        tat_case :bool,     # task attention transformer architecture
        in_proj,
        dense_mul,          # transformer dense multiplication
        dropout=    0.0,    # transformer dropout
        n_layers=   6,
        verb=       0):

    if verb > 0: print('\nBuilding cEncT (T encoder)...')

    inCemb = tf.nn.embedding_lookup(params=cEMB, ids=sevc_PH)
    if verb > 1: print(' > inCemb:', inCemb)

    myCEMB = tf.get_variable(  # my cards embeddings
        name=           'myCEMB',
        shape=          [2, cEMB.shape[-1]],
        dtype=          tf.float32,
        initializer=    my_initializer())
    myCElook = tf.nn.embedding_lookup(params=myCEMB, ids=[0,0,1,1,1,1,1])
    if verb > 1: print(' > myCElook:', myCElook)
    inCemb += myCElook

    # input projection
    if in_proj:
        cProjOUT = lay_dense(
            input=          inCemb,
            units=          in_proj,
            name=           'cProj',
            reuse=          tf.AUTO_REUSE,
            useBias=        False)
        inCemb = cProjOUT['output']
        if verb > 1: print(' > inCemb projected:', inCemb)
    elif verb > 1: print(' > inCemb:', inCemb)

    TATcase = tat_case
    encOUT = encTRNS(
        in_seq=         inCemb,
        name=           'TAT' if TATcase else 'TNS',
        seq_out=        not TATcase,
        add_PE=         False,
        n_blocks=       n_layers,
        n_heads=        1,
        dense_mul=      dense_mul,
        max_seq_len=    7,
        dropout=        dropout,
        dropout_att=    0,
        drop_flag=      trPH,
        n_histL=        3,
        verb=           verb)

    output = encOUT['tns_out']
    if not TATcase:
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

    trPH = tf.placeholder_with_default(  # train placeholder
        input=          False,
        name=           'trPH',
        shape=          [])

    cEMB = tf.get_variable(  # cards embeddings
        name=           'cEMB',
        shape=          [53, c_embW],  # one card for 'no_card'
        dtype=          tf.float32,
        initializer=    my_initializer())

    with tf.device('/device:CPU:0'):
        hist_summ = [tf.summary.histogram('cEMB', cEMB, family='cEMB')]

    inACPH = tf.placeholder( # 7 cards of A
        name=           'inACPH',
        dtype=          tf.int32,
        shape=          [None, 7])  # [bsz,7cards]

    inBCPH = tf.placeholder( # 7 cards of B
        name=           'inBCPH',
        dtype=          tf.int32,
        shape=          [None, 7])  # [bsz,7cards]

    wonPH = tf.placeholder( # wonPH class (lables of winner 0,1-ABwins,2-draw)
        name=           'wonPH',
        dtype=          tf.int32,
        shape=          [None])  # [bsz]

    rnkAPH = tf.placeholder( # rank A class (labels <0,8>)
        name=           'rnkAPH',
        dtype=          tf.int32,
        shape=          [None])  # [bsz]

    rnkBPH = tf.placeholder( # rank B class (labels <0,8>)
        name=           'rnkBPH',
        dtype=          tf.int32,
        shape=          [None])  # [bsz]

    mcACPH = tf.placeholder( # chances of winning for A (?montecarlo)
        name=           'mcACPH',
        dtype=          tf.float32,
        shape=          [None])  # [bsz]

    # encoders for A and B
    cRGAout = card_enc(
        sevc_PH=     inACPH,
        cEMB=       cEMB,
        trPH=       trPH,
        tat_case=        tat_case,
        in_proj=     in_proj,
        dense_mul=   dense_mul,
        dropout=    dropout,
        n_layers=    n_layers,
        verb=    verb)
    cRGBout = card_enc(
        sevc_PH=     inBCPH,
        cEMB=       cEMB,
        trPH=       trPH,
        tat_case=        tat_case,
        in_proj=     in_proj,
        dense_mul=   dense_mul,
        dropout=    dropout,
        n_layers=    n_layers,
        verb=    verb)

    # get nn_zeros
    nn_zerosA = cRGAout['nn_zeros']
    nn_zerosA = tf.reshape(tf.stack(nn_zerosA), shape=[-1])
    nn_zerosB = cRGBout['nn_zeros']
    nn_zerosB = tf.reshape(tf.stack(nn_zerosB), shape=[-1])
    hist_summ.append(cRGAout['hist_summ']) # get histograms from A

    # where all cards of A are known
    whereAllCardsA = tf.reduce_max(inACPH, axis=-1)
    whereAllCardsA = tf.where(
        condition=  whereAllCardsA < 52,
        x=          tf.ones_like(whereAllCardsA),
        y=          tf.zeros_like(whereAllCardsA))
    if verb > 1: print('\n > whereAllCardsA', whereAllCardsA)
    whereAllCardsF = tf.cast(whereAllCardsA, dtype=tf.float32) # cast to float

    # projection to 9 ranks A
    denseOutA = lay_dense(
        input=      cRGAout['output'],
        units=      9,
        name=       'denseRC',
        reuse=      tf.AUTO_REUSE,
        useBias=    False)
    rankAlogits = denseOutA['output']
    lossRA = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss rank A
        labels=     rnkAPH,
        logits=     rankAlogits)
    lossRA = tf.reduce_mean(lossRA * whereAllCardsF) # lossRA masked (where all cards @A)

    # projection to 9 ranks B
    denseOutB = lay_dense(
        input=      cRGBout['output'],
        units=      9,
        name=       'denseRC',
        reuse=      tf.AUTO_REUSE,
        useBias=    False)
    rankBlogits = denseOutB['output']
    lossRB = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss rank B
        labels=     rnkBPH,
        logits=     rankBlogits)
    lossRB = tf.reduce_mean(lossRB)

    lossR = lossRA + lossRB
    if verb > 1: print(' > lossR:', lossR)

    # winner classifier (on concatenated representations)
    output = tf.concat([cRGAout['output'],cRGBout['output']], axis=-1)
    if verb > 1: print(' > concRepr:', output)
    if dr_layers:
        encOUT = encDR(
            input=      output,
            name=       'drC',
            layWidth=   dense_proj,
            nLayers=    dr_layers,
            dropout=    dropout_DR,
            dropFlagT=  trPH,
            nHL=        0,
            verbLev=    verb)
        output = encOUT['output']

    # projection to 3 winner logits
    denseOut = lay_dense(
        input=          output,
        units=          3,
        name=           'denseW',
        reuse=          tf.AUTO_REUSE,
        useBias=        False)
    wonLogits = denseOut['output']
    if verb > 1: print(' > wonLogits:', wonLogits)
    lossW = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss wonPH
        labels=     wonPH,
        logits=     wonLogits)
    lossW = tf.reduce_mean(lossW * whereAllCardsF) # loss winner classifier, masked
    if verb > 1: print(' > lossW:', lossW)

    # projection to probability of winning of A cards (regression value)
    denseOut = lay_dense(
        input=          cRGAout['output'],
        units=          1,
        name=           'denseREG',
        reuse=          tf.AUTO_REUSE,
        activation=     tf.nn.relu,
        useBias=        False)
    probAWvReg = denseOut['output'] # probAWvReg
    probAWvReg = tf.reshape(probAWvReg, shape=[-1])
    if verb > 1: print(' > probAWvReg:', probAWvReg)
    lossPAWR = tf.losses.mean_squared_error(
        labels=         mcACPH,
        predictions=    probAWvReg)
    if verb > 1: print(' > lossPAWR:', lossPAWR)

    diffPAWR = tf.sqrt(tf.square(mcACPH-probAWvReg))
    diffPAWRmn = tf.reduce_mean(diffPAWR) # avg of diff PAWR
    diffPAWRmx = tf.reduce_max(diffPAWR) # max of diff PAWR

    loss = lossW + lossR + lossPAWR # this is how loss is constructed

    # accuracy of winner classifier scaled by where all cards
    predictionsW = tf.argmax(wonLogits, axis=-1, output_type=tf.int32)
    if verb > 1: print(' > predictionsW:', predictionsW)
    correctW = tf.equal(predictionsW, wonPH)
    if verb > 1: print(' > correctW:', correctW)
    correctWF = tf.cast(correctW, dtype=tf.float32)
    correctWFwhere = correctWF * whereAllCardsF
    avgAccW = tf.reduce_sum(correctWFwhere) / tf.reduce_sum(whereAllCardsF)
    if verb > 1: print(' > avgAccW:', avgAccW)

    # accuracy of winner classifier per class scaled by where all cards
    ohWon = tf.one_hot(indices=wonPH, depth=3) # OH [batch,3], 1 where wins, dtype tf.float32
    ohWonWhere = ohWon * tf.stack([whereAllCardsF]*3, axis=1) # masked where all cards
    wonDensity = tf.reduce_mean(ohWonWhere, axis=0) # [3] measures density of 1 @batch per class
    ohCorrect = tf.where(condition=correctW, x=ohWonWhere, y=tf.zeros_like(ohWon)) # [batch,3]
    wonCorrDensity = tf.reduce_mean(ohCorrect, axis=0)
    avgAccWC = wonCorrDensity / wonDensity

    ohNotCorrectW = tf.where(condition=tf.logical_not(correctW), x=ohWon, y=tf.zeros_like(ohWon)) # OH wins where not correct
    ohNotCorrectW *= tf.stack([whereAllCardsF]*3, axis=1) # masked with all cards

    # acc of rank(B)
    predictionsR = tf.argmax(rankBlogits, axis=-1, output_type=tf.int32)
    correctR = tf.equal(predictionsR, rnkBPH)
    avgAccR = tf.reduce_mean(tf.cast(correctR, dtype=tf.float32))
    if verb > 1: print(' > avgAccR:', avgAccR)

    # acc of rank(B) per class
    ohRnkB = tf.one_hot(indices=rnkBPH, depth=9)
    rnkBdensity = tf.reduce_mean(ohRnkB, axis=0)
    ohCorrectR = tf.where(condition=correctR, x=ohRnkB, y=tf.zeros_like(ohRnkB))
    rnkBcorrDensity = tf.reduce_mean(ohCorrectR, axis=0)
    avgAccRC = rnkBcorrDensity/rnkBdensity

    ohNotCorrectR = tf.where(condition=tf.logical_not(correctR), x=ohRnkB, y=tf.zeros_like(ohRnkB)) # OH ranks where not correct

    return{
        'trPH':                 trPH,
        'inACPH':               inACPH,
        'inBCPH':               inBCPH,
        'wonPH':                wonPH,
        'rnkAPH':               rnkAPH,
        'rnkBPH':               rnkBPH,
        'mcACPH':               mcACPH,
        'loss':                 loss, # total loss for training (OPT)
        'lossW':                lossW, # loss of winner classifier
        'lossR':                lossR, # loss of rank classifier
        'lossPAWR':             lossPAWR, # loss of prob win (value) of A
        'diffPAWRmn':           diffPAWRmn,
        'diffPAWRmx':           diffPAWRmx,
        'avgAccW':              avgAccW,
        'avgAccWC':             avgAccWC,
        'predictionsW':         predictionsW,
        'ohNotCorrectW':        ohNotCorrectW,
        'accR':                 avgAccR,
        'accRC':                avgAccRC,
        'predictionsR':         predictionsR,
        'ohNotCorrectR':        ohNotCorrectR,
        'hist_summ':             tf.summary.merge(hist_summ),
        'nn_zerosA':             nn_zerosA,
        'nn_zerosB':             nn_zerosB}