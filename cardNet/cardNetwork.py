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

    wonPH = tf.placeholder( # wonPH class
        name=           'wonPH',
        dtype=          tf.int32,
        shape=          [None])  # [bsz]

    rnkAPH = tf.placeholder( # rank A class
        name=           'rnkAPH',
        dtype=          tf.int32,
        shape=          [None])  # [bsz]

    rnkBPH = tf.placeholder( # rank B class
        name=           'rnkBPH',
        dtype=          tf.int32,
        shape=          [None])  # [bsz]

    mcACPH = tf.placeholder( # MontCarlo chances of winning for A
        name=           'mcACPH',
        dtype=          tf.float32,
        shape=          [None])  # [bsz]

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

    # get nnZeros from A
    nnZeros = cRGAout['nnZeros']
    nnZeros = tf.reshape(tf.stack(nnZeros), shape=[-1])
    histSumm.append(cRGAout['histSumm']) # get histograms from A

    # where all cards of A are known
    whereAllCards = tf.reduce_max(inACPH, axis=-1)
    whereAllCards = tf.where(
        condition=  whereAllCards < 52,
        x=          tf.ones_like(whereAllCards),
        y=          tf.zeros_like(whereAllCards))
    whereAllCardsF = tf.cast(whereAllCards, dtype=tf.float32)
    if verbLev > 1: print(' > whereAllCards', whereAllCards)

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
    lossRA = tf.reduce_mean(lossRA * whereAllCardsF)

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

    # dense classifier of output
    output = tf.concat([cRGAout['output'],cRGBout['output']], axis=-1)
    if verbLev > 1: print('\n > concRepr:', output)
    if drLayers:
        encOUT = encDR(
            input=      output,
            name=       'drC',
            layWidth=   denseProj,
            nLayers=    drLayers,
            dropout=    dropoutDRE,
            dropFlagT=  trPH,
            nHL=        0,
            verbLev=    2)
        output = encOUT['output']

    # projection to 3 winner logits
    denseOut = layDENSE(
        input=          output,
        units=          3,
        name=           'denseW',
        reuse=          tf.AUTO_REUSE,
        useBias=        False)
    wonLogits = denseOut['output']
    if verbLev > 1: print(' > logits:', wonLogits)

    lossW = tf.nn.sparse_softmax_cross_entropy_with_logits( # loss wonPH
        labels=     wonPH,
        logits=     wonLogits)
    lossW = tf.reduce_mean(lossW * whereAllCardsF)
    if verbLev > 1: print(' > lossW:', lossW)

    # projection to regression value
    denseOut = layDENSE(
        input=          cRGAout['output'],
        units=          1,
        name=           'denseREG',
        reuse=          tf.AUTO_REUSE,
        activation=     tf.nn.relu,
        useBias=        False)
    mcACregVal = denseOut['output']
    mcACregVal = tf.reshape(mcACregVal, shape=[-1])
    if verbLev > 1: print(' > mcACregVal:', mcACregVal)

    lossMCAC = tf.losses.mean_squared_error(
        labels=         mcACPH,
        predictions=    mcACregVal)
    if verbLev > 1: print(' > lossMCAC:', lossMCAC)

    loss = lossW + lossR + lossMCAC # this is how loss is constructed

    predictionsRA = tf.argmax(rankAlogits, axis=-1, output_type=tf.int32)
    predictionsRB = tf.argmax(rankBlogits, axis=-1, output_type=tf.int32)
    correctRA = tf.equal(predictionsRA, rnkAPH)
    correctRB = tf.equal(predictionsRB, rnkBPH)
    correctRAFwhere = tf.cast(correctRA, dtype=tf.float32) * whereAllCardsF
    avgAccR = tf.reduce_sum(correctRAFwhere) / tf.reduce_sum(whereAllCardsF) + tf.reduce_mean(tf.cast(correctRB, dtype=tf.float32))
    avgAccR /= 2
    if verbLev > 1: print(' > avgAccR:', avgAccR)

    ohRnkA = tf.one_hot(indices=rnkAPH, depth=9)
    ohRnkB = tf.one_hot(indices=rnkBPH, depth=9)
    rnkAdensity = tf.reduce_mean(ohRnkA, axis=-2)
    rnkBdensity = tf.reduce_mean(ohRnkB, axis=-2)
    ohCorrectRA = tf.where(condition=correctRA, x=ohRnkA, y=tf.zeros_like(ohRnkA))
    ohCorrectRB = tf.where(condition=correctRB, x=ohRnkB, y=tf.zeros_like(ohRnkB))
    rnkAcorrDensity = tf.reduce_mean(ohCorrectRA, axis=-2)
    rnkBcorrDensity = tf.reduce_mean(ohCorrectRB, axis=-2)
    avgAccRC = (rnkAcorrDensity/rnkAdensity + rnkBcorrDensity/rnkBdensity)/2

    ohNotCorrectRA = tf.where(condition=tf.logical_not(correctRA), x=ohRnkA, y=tf.zeros_like(ohRnkA))

    predictionsW = tf.argmax(wonLogits, axis=-1, output_type=tf.int32)
    if verbLev > 1: print(' > predictionsW:', predictionsW)
    correctW = tf.equal(predictionsW, wonPH)
    if verbLev > 1: print(' > correctW:', correctW)
    correctWFwhere = tf.cast(correctW, dtype=tf.float32) * whereAllCardsF
    avgAccW = tf.reduce_sum(correctWFwhere) / tf.reduce_sum(whereAllCardsF)
    if verbLev > 1: print(' > avgAccW:', avgAccW)

    ohWon = tf.one_hot(indices=wonPH, depth=3)
    wonDensity = tf.reduce_mean(ohWon, axis=-2)
    ohCorrect = tf.where(condition=correctW, x=ohWon, y=tf.zeros_like(ohWon))
    wonCorrDensity = tf.reduce_mean(ohCorrect, axis=-2)
    avgAccC = wonCorrDensity / wonDensity

    ohNotCorrect = tf.where(condition=tf.logical_not(correctW), x=ohWon, y=tf.zeros_like(ohWon))

    return{
        'trPH':                 trPH,
        'inACPH':               inACPH,
        'inBCPH':               inBCPH,
        'wonPH':                wonPH,
        'rnkAPH':               rnkAPH,
        'rnkBPH':               rnkBPH,
        'mcACPH':               mcACPH,
        'whereAllCardsF':       whereAllCardsF,
        'rankAlogits':          rankAlogits,
        'rankBlogits':          rankBlogits,
        'wonLogits':            wonLogits,
        'loss':                 loss,
        'lossW':                lossW,
        'lossR':                lossR,
        'lossMCAC':             lossMCAC,
        'avgAccW':              avgAccW,
        'avgAccC':              avgAccC,
        'predictionsW':         predictionsW,
        'ohNotCorrect':         ohNotCorrect,
        'accR':                 avgAccR,
        'accRC':                avgAccRC,
        'predictionsRA':        predictionsRA,
        'ohNotCorrectRA':       ohNotCorrectRA,
        'histSumm':             tf.summary.merge(histSumm),
        'nnZeros':              nnZeros}