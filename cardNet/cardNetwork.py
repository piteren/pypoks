"""

 2019 (c) piteren

"""

from functools import partial
import tensorflow as tf

from pUtils.nnTools.nnBaseElements import defInitializer, layDENSE
from pUtils.nnTools.nnEncoders import encDR, encTRNS


# cards Transformer encoder graph (7 cards representations)
def cEncT(
        sevenC,                     # seven cards placeholder
        cEMB,                       # cards embedding tensor
        trPH,                       # train placeholder
        tat :bool,                  # task attention transformer architecture
        inProj,
        denseMul,
        dropout=    0.0,
        nLayers=    6,
        verbLev=    0):

    if verbLev > 0: print('\nBuilding cEncT (T encoder)...')

    inCemb = tf.nn.embedding_lookup(params=cEMB, ids=sevenC)
    if verbLev > 1: print(' > inCemb:', inCemb)

    myCEMB = tf.get_variable(  # my cards embeddings
        name=           'myCEMB',
        shape=          [2, cEMB.shape[-1]],
        dtype=          tf.float32,
        initializer=    defInitializer())
    myCElook = tf.nn.embedding_lookup(params=myCEMB, ids=[0,0,1,1,1,1,1])
    if verbLev > 1: print(' > myCElook:', myCElook)
    inCemb += myCElook

    # input projection
    if inProj:
        cProjOUT = layDENSE(
            input=          inCemb,
            units=          inProj,
            name=           'cProj',
            reuse=          tf.AUTO_REUSE,
            useBias=        False)
        inCemb = cProjOUT['output']
        if verbLev > 1: print(' > inCemb projected:', inCemb)
    elif verbLev > 1: print(' > inCemb:', inCemb)

    TATcase = tat
    encOUT = encTRNS(
        input=      inCemb,
        seqOut=     not TATcase,
        addPE=      False,
        name=       'TAT' if TATcase else 'TNS',
        nBlocks=    nLayers,
        nHeads=     1,
        denseMul=   denseMul,
        maxSeqLen=  7,
        dropoutAtt= 0,
        dropout=    dropout,
        dropFlagT=  trPH,
        nHistL=     3,
        verbLev=    verbLev)

    output = encOUT['eTOut']
    if not TATcase:
        output = tf.unstack(output, axis=-2)
        output = tf.concat(output, axis=-1)
        if verbLev > 1:print(' > encT reshaped output:', output)
    elif verbLev > 1: print(' > encT output:', output)

    return {
        'output':   output,
        'histSumm': encOUT['histSumm'],
        'nnZeros':  encOUT['nnZeros']}

# cards net FWD graph
def cardFWDng(
        tat=        False,
        cEmbW=      24,
        nLayers=    8,
        inProj=     None,   # None, 0 or int
        denseMul=   4,
        denseProj=  None,   # None, 0 or int
        drLayers=   2,      # None, 0 or int
        dropout=    0.0,
        dropoutDRE= 0.0,
        # train parameters
        optClass=   partial(tf.train.AdamOptimizer, beta1=0.7, beta2=0.7),
        iLR=        1e-3,
        warmUp=     10000,
        annbLr=     0.999,
        stepLr=     0.04,
        avtStartV=  0.1,
        avtWindow=  500,
        avtMaxUpd=  1.5,
        doClip=     False,
        verbLev=    0,
        **kwargs):

    trPH = tf.placeholder_with_default(  # train placeholder
        input=          False,
        name=           'trPH',
        shape=          [])

    cEMB = tf.get_variable(  # cards embeddings
        name=           'cEMB',
        shape=          [53, cEmbW],  # one card for 'no_card'
        dtype=          tf.float32,
        initializer=    defInitializer())

    with tf.device('/device:CPU:0'):
        histSumm = [tf.summary.histogram('cEMB', cEMB, family='cEMB')]

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
    cRGAout = cEncT(
        sevenC=     inACPH,
        cEMB=       cEMB,
        trPH=       trPH,
        tat=        tat,
        inProj=     inProj,
        denseMul=   denseMul,
        dropout=    dropout,
        nLayers=    nLayers,
        verbLev=    verbLev)
    cRGBout = cEncT(
        sevenC=     inBCPH,
        cEMB=       cEMB,
        trPH=       trPH,
        tat=        tat,
        inProj=     inProj,
        denseMul=   denseMul,
        dropout=    dropout,
        nLayers=    nLayers,
        verbLev=    verbLev)

    # get nnZeros
    nnZerosA = cRGAout['nnZeros']
    nnZerosA = tf.reshape(tf.stack(nnZerosA), shape=[-1])
    nnZerosB = cRGBout['nnZeros']
    nnZerosB = tf.reshape(tf.stack(nnZerosB), shape=[-1])
    histSumm.append(cRGAout['histSumm']) # get histograms from A

    # where all cards of A are known
    whereAllCardsA = tf.reduce_max(inACPH, axis=-1)
    whereAllCardsA = tf.where(
        condition=  whereAllCardsA < 52,
        x=          tf.ones_like(whereAllCardsA),
        y=          tf.zeros_like(whereAllCardsA))
    if verbLev > 1: print('\n > whereAllCardsA', whereAllCardsA)
    whereAllCardsF = tf.cast(whereAllCardsA, dtype=tf.float32) # cast to float

    # projection to 9 ranks A
    denseOutA = layDENSE(
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
    denseOutB = layDENSE(
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
    if verbLev > 1: print(' > lossR:', lossR)

    # winner classifier (on concatenated representations)
    output = tf.concat([cRGAout['output'],cRGBout['output']], axis=-1)
    if verbLev > 1: print(' > concRepr:', output)
    if drLayers:
        encOUT = encDR(
            input=      output,
            name=       'drC',
            layWidth=   denseProj,
            nLayers=    drLayers,
            dropout=    dropoutDRE,
            dropFlagT=  trPH,
            nHL=        0,
            verbLev=    verbLev)
        output = encOUT['output']

    # projection to 3 winner logits
    denseOut = layDENSE(
        input=          output,
        units=          3,
        name=           'denseW',
        reuse=          tf.AUTO_REUSE,
        useBias=        False)
    wonLogits = denseOut['output']
    if verbLev > 1: print(' > wonLogits:', wonLogits)
    lossW = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss wonPH
        labels=     wonPH,
        logits=     wonLogits)
    lossW = tf.reduce_mean(lossW * whereAllCardsF) # loss winner classifier, masked
    if verbLev > 1: print(' > lossW:', lossW)

    # projection to probability of winning of A cards (regression value)
    denseOut = layDENSE(
        input=          cRGAout['output'],
        units=          1,
        name=           'denseREG',
        reuse=          tf.AUTO_REUSE,
        activation=     tf.nn.relu,
        useBias=        False)
    probAWvReg = denseOut['output'] # probAWvReg
    probAWvReg = tf.reshape(probAWvReg, shape=[-1])
    if verbLev > 1: print(' > probAWvReg:', probAWvReg)
    lossPAWR = tf.losses.mean_squared_error(
        labels=         mcACPH,
        predictions=    probAWvReg)
    if verbLev > 1: print(' > lossPAWR:', lossPAWR)

    loss = lossW + lossR + lossPAWR # this is how loss is constructed

    # accuracy of winner classifier scaled by where all cards
    predictionsW = tf.argmax(wonLogits, axis=-1, output_type=tf.int32)
    if verbLev > 1: print(' > predictionsW:', predictionsW)
    correctW = tf.equal(predictionsW, wonPH)
    if verbLev > 1: print(' > correctW:', correctW)
    correctWF = tf.cast(correctW, dtype=tf.float32)
    correctWFwhere = correctWF * whereAllCardsF
    avgAccW = tf.reduce_sum(correctWFwhere) / tf.reduce_sum(whereAllCardsF)
    if verbLev > 1: print(' > avgAccW:', avgAccW)

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
    if verbLev > 1: print(' > avgAccR:', avgAccR)

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
        #'whereAllCardsF':       whereAllCardsF,
        #'rankAlogits':          rankAlogits,
        #'rankBlogits':          rankBlogits,
        #'wonLogits':            wonLogits,
        'loss':                 loss, # total loss for training (OPT)
        'lossW':                lossW, # loss of winner classifier
        'lossR':                lossR, # loss of rank classifier
        'lossPAWR':             lossPAWR, # loss of prob win (value) of A
        'avgAccW':              avgAccW,
        'avgAccWC':             avgAccWC,
        'predictionsW':         predictionsW,
        'ohNotCorrectW':        ohNotCorrectW,
        'accR':                 avgAccR,
        'accRC':                avgAccRC,
        'predictionsR':         predictionsR,
        'ohNotCorrectR':        ohNotCorrectR,
        'histSumm':             tf.summary.merge(histSumm),
        'nnZerosA':             nnZerosA,
        'nnZerosB':             nnZerosB}