"""

 2019 (c) piteren

"""

import tensorflow as tf

from pUtils.littleTools.littleMethods import shortSCIN
from pUtils.nnTools.nnBaseElements import defInitializer, layDENSE, numVFloats, gradClipper, lRscaler
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
        nLayers=    6):

    print('\nBuilding cEncT (T encoder)...')

    inCemb = tf.nn.embedding_lookup(params=cEMB, ids=sevenC)
    print(' > inCemb:', inCemb)

    myCEMB = tf.get_variable(  # my cards embeddings
        name=           'myCEMB',
        shape=          [2, cEMB.shape[-1]],
        dtype=          tf.float32,
        initializer=    defInitializer())
    myCElook = tf.nn.embedding_lookup(params=myCEMB, ids=[0,0,1,1,1,1,1])
    print(' > myCElook:', myCElook)
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
        print(' > inCemb projected:', inCemb)
    else: print(' > inCemb:', inCemb)

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
        verbLev=    2)

    output = encOUT['eTOut']
    if not TATcase:
        output = tf.unstack(output, axis=-2)
        output = tf.concat(output, axis=-1)
        print(' > encT reshaped output:', output)
    else: print(' > encT output:', output)

    return {
        'output':   output,
        'histSumm': encOUT['histSumm'],
        'nnZeros':  encOUT['nnZeros']}

# cards net graph
def cardGFN(
        tat=        False,
        cEmbW=      24,
        nLayers=    12,
        inProj=     None,   # None, 0 or int
        denseMul=   4,
        denseProj=  None,   # None, 0 or int
        drLayers=   4,      # None, 0 or int
        lR=         1e-3,
        warmUp=     10000,
        annbLr=     0.999,
        stepLr=     0.04,
        dropout=    0.0,
        dropoutDRE= 0.0,
        avtStartV=  0.1,
        avtWindow=  500,
        avtMaxUpd=  1.5,
        doClip=     False):

    with tf.variable_scope('CNG', reuse=tf.AUTO_REUSE):

        trPH = tf.placeholder_with_default(  # train placeholder
            input=          False,
            name=           'trPH',
            shape=          [])

        cEMB = tf.get_variable(  # cards embeddings
            name=           'cEMB',
            shape=          [53, cEmbW],  # one card for 'no_card'
            dtype=          tf.float32,
            initializer=    defInitializer())

        histSumm = [tf.summary.histogram('cEMB', cEMB, family='cEMB')]

        inAC = tf.placeholder(  # 7 cards of A
            name=           'inAC',
            dtype=          tf.int32,
            shape=          [None, 7])  # [bsz,7cards]

        inBC = tf.placeholder(  # 7 cards of B
            name=           'inBC',
            dtype=          tf.int32,
            shape=          [None, 7])  # [bsz,7cards]

        won = tf.placeholder(  # won class
            name=           'won',
            dtype=          tf.int32,
            shape=          [None])  # [bsz,seq]

        rnkA = tf.placeholder( # rank A class
            name=           'rnkA',
            dtype=          tf.int32,
            shape=          [None])  # [bsz,seq]

        rnkB = tf.placeholder( # rank B class
            name=           'rnkB',
            dtype=          tf.int32,
            shape=          [None])  # [bsz,seq]

        cRGAout = cEncT(
            sevenC=     inAC,
            cEMB=       cEMB,
            trPH=       trPH,
            tat=        tat,
            inProj=     inProj,
            denseMul=   denseMul,
            dropout=    dropout,
            nLayers=    nLayers)
        cRGBout = cEncT(
            sevenC=     inBC,
            cEMB=       cEMB,
            trPH=       trPH,
            tat=        tat,
            inProj=     inProj,
            denseMul=   denseMul,
            dropout=    dropout,
            nLayers=    nLayers)
        # get nnZeros from A
        nnZeros = cRGAout['nnZeros']
        nnZeros = tf.reshape(tf.stack(nnZeros), shape=[-1])
        histSumm.append(cRGAout['histSumm']) # get histograms from A

        # projection to 9 ranks A
        denseOutA = layDENSE(
            input=          cRGAout['output'],
            units=          9,
            name=           'denseRC',
            useBias=        False)
        rankAlogits = denseOutA['output']

        # projection to 9 ranks B
        denseOutB = layDENSE(
            input=          cRGBout['output'],
            units=          9,
            name=           'denseRC',
            reuse=          True,
            useBias=        False)
        rankBlogits = denseOutB['output']

        output = tf.concat([cRGAout['output'],cRGBout['output']], axis=-1)
        print('\n > concRepr:', output)

        # dense classifier of output
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
            useBias=        False,
            initializer=    defInitializer())
        wonLogits = denseOut['output']
        print(' > logits:', wonLogits)

        vars = tf.trainable_variables()
        print(' ### num of (%d) vars %s'%(len(vars), shortSCIN(numVFloats(vars))))
        #for var in vars: print(var)

        lossRA = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels= rnkA,
            logits= rankAlogits)
        lossRA = tf.reduce_mean(lossRA)

        lossRB = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels= rnkB,
            logits= rankBlogits)
        lossRB = tf.reduce_mean(lossRB)
        lossR = lossRA+lossRB
        print(' > lossR:', lossR)

        lossW = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=     won,
            logits=     wonLogits)
        lossW = tf.reduce_mean(lossW)
        print(' > lossW:', lossW)
        loss = lossW + lossR
        #loss = lossW

        predictionsRA = tf.argmax(rankAlogits, axis=-1, output_type=tf.int32)
        predictionsRB = tf.argmax(rankBlogits, axis=-1, output_type=tf.int32)
        correctRA = tf.equal(predictionsRA, rnkA)
        correctRB = tf.equal(predictionsRB, rnkB)
        avgAccR = tf.reduce_mean(tf.cast(correctRA, dtype=tf.float32)) + tf.reduce_mean(tf.cast(correctRB, dtype=tf.float32))
        avgAccR /= 2
        print(' > avgAccR:', avgAccR)

        ohRnkA = tf.one_hot(indices=rnkA, depth=9)
        ohRnkB = tf.one_hot(indices=rnkB, depth=9)
        rnkAdensity = tf.reduce_mean(ohRnkA, axis=-2)
        rnkBdensity = tf.reduce_mean(ohRnkB, axis=-2)
        ohCorrectRA = tf.where(condition=correctRA, x=ohRnkA, y=tf.zeros_like(ohRnkA))
        ohCorrectRB = tf.where(condition=correctRB, x=ohRnkB, y=tf.zeros_like(ohRnkB))
        rnkAcorrDensity = tf.reduce_mean(ohCorrectRA, axis=-2)
        rnkBcorrDensity = tf.reduce_mean(ohCorrectRB, axis=-2)
        avgAccRC = (rnkAcorrDensity/rnkAdensity + rnkBcorrDensity/rnkBdensity)/2

        ohNotCorrectRA = tf.where(condition=tf.logical_not(correctRA), x=ohRnkA, y=tf.zeros_like(ohRnkA))

        predictions = tf.argmax(wonLogits, axis=-1, output_type=tf.int32)
        print(' > predictions:', predictions)
        correct = tf.equal(predictions, won)
        print(' > correct:', correct)
        avgAcc = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
        print(' > avgAcc:', avgAcc)

        ohWon = tf.one_hot(indices=won, depth=3)
        wonDensity = tf.reduce_mean(ohWon, axis=-2)
        ohCorrect = tf.where(condition=correct, x=ohWon, y=tf.zeros_like(ohWon))
        wonCorrDensity = tf.reduce_mean(ohCorrect, axis=-2)
        avgAccC = wonCorrDensity / wonDensity

        ohNotCorrect = tf.where(condition=tf.logical_not(correct), x=ohWon, y=tf.zeros_like(ohWon))

        globalStep = tf.get_variable(  # global step
            name=           'gStep',
            shape=          [],
            trainable=      False,
            initializer=    tf.constant_initializer(0),
            dtype=          tf.int32)

        lRs = lRscaler(
            iLR=            lR,
            gStep=          globalStep,
            warmUpSteps=    warmUp,
            annbLr=         annbLr,
            stepLr=         stepLr,
            verbLev=        1)

        optimizer = tf.train.AdamOptimizer(lRs, beta1=0.7, beta2=0.7)
        #optimizer = tf.train.GradientDescentOptimizer(lRs)
        #optimizer = tf.train.MomentumOptimizer(lRs, momentum=0.9)
        #optimizer = tf.train.AdagradOptimizer(lRs)

        clipOUT = gradClipper(
            gradients=  tf.gradients(loss, vars),
            avtStartV=  avtStartV,
            avtWindow=  avtWindow,
            avtMaxUpd=  avtMaxUpd,#1.2,
            doClip=     doClip)
        gradients = clipOUT['gradients']
        gN = clipOUT['gGNorm']
        agN = clipOUT['avtGGNorm']
        optimizer = optimizer.apply_gradients(zip(gradients, vars), global_step=globalStep)

        # select optimizer vars
        optVars = []
        for var in tf.global_variables(scope=tf.get_variable_scope().name):
            if var not in vars: optVars.append(var)

        return{
            'trPH':                 trPH,
            'inAC':                 inAC,
            'inBC':                 inBC,
            'won':                  won,
            'rnkA':                 rnkA,
            'rnkB':                 rnkB,
            'loss':                 loss,
            'acc':                  avgAcc,
            'accC':                 avgAccC,
            'predictions':          predictions,
            'ohNotCorrect':         ohNotCorrect,
            'accR':                 avgAccR,
            'accRC':                avgAccRC,
            'predictionsRA':        predictionsRA,
            'ohNotCorrectRA':       ohNotCorrectRA,
            'lRs':                  lRs,
            'gN':                   gN,
            'agN':                  agN,
            'vars':                 vars,
            'optVars':              optVars,
            'optimizer':            optimizer,
            'histSumm':             tf.summary.merge(histSumm),
            'nnZeros':              nnZeros}