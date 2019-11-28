"""

 2019 (c) piteren

 card net running

 there are 2,598,960 hands not considering different suits, it is (52/5) - combination

 stats of poks hands, for 5 (not 7) randomly taken:
 0 (highCard)       - 50.1177%
 1 (pair)           - 42.2569%
 2 (2pairs)         -  4.7539%
 3 (threeOf)        -  2.1128%
 4 (straight)       -  0.3925%
 5 (flush)          -  0.1965%
 6 (FH)             -  0.1441%
 7 (fourOf)         -  0.0240%
 8 (straightFlush)  -  0.001544%

 for seven cards (from 0 to 8):
 0.17740 0.44400 0.23040 0.04785 0.04150 0.03060 0.02660 0.00145 0.00020 (fraction)
 0.17740 0.62140 0.85180 0.89965 0.94115 0.97175 0.99835 0.99980 1.00000 (cumulative fraction)

 for seven cards, when try_to_balance it (first attempt) (from 0 to 8):
 0.11485 0.22095 0.16230 0.10895 0.08660 0.09515 0.08695 0.06490 0.05935
 0.11485 0.33580 0.49810 0.60705 0.69365 0.78880 0.87575 0.94065 1.00000

"""

from functools import partial
import numpy as np
import tensorflow as tf
import time

from pUtils.nnTools.nnBaseElements import loggingSet
from pUtils.queMultiProcessor import QueMultiProcessor
from pUtils.nnTools.multiSaver import MultiSaver
from pUtils.nnTools.nnModel import NNModel

from pLogic.pDeck import getASC
from cardNet.cardBatcher import prep2X7Batch
from cardNet.cardNetwork import cardFWDng

#TODO:
# - accuracy_masked continue
# - loss components influence

# training function
def trainCardNet(
        cardNetDict :dict,
        nBatches=   50000,
        trainSM=    (1000,10),
        doTest=     True,
        testSM=     (2000,100),   # test size and MonteC samples
        rQueTSize=  200,
        verbLev=    0):

    #asc = getASC()
    asc = None

    # prepare tests data
    cTuples = None
    testBatch = None
    if doTest:
        if verbLev > 0: print('\nPreparing test batch...')
        testBatch = prep2X7Batch(
            bs=         testSM[0],
            nMonte=     testSM[1],
            asc=        asc,
            verbLev=    verbLev)
        cTuples = []
        for ix in range(testSM[0]):
            cTuples.append(tuple(sorted(testBatch['crd7AB'][ix])))
            cTuples.append(tuple(sorted(testBatch['crd7BB'][ix])))
        if verbLev > 1: print('Got %d of hands in testBatch'%len(cTuples))
        cTuples = dict.fromkeys(cTuples, 1)
        if verbLev > 1: print('of which %d is unique'%len(cTuples))

    iPF = partial(prep2X7Batch, bs=trainSM[0], nMonte=trainSM[1], asc=asc)
    qmp = QueMultiProcessor( # QMP
        iProcFunction=  iPF,
        taskObject=     cTuples,
        nProc=          10,
        rQueTSize=      rQueTSize,
        verbLev=        verbLev)

    cNet = NNModel( # model
        mDict=          cardNetDict,
        fwdF=           cardFWDng,
        #useAllCUDA=     True,
        verbLev=        verbLev)

    summWriter = tf.summary.FileWriter(logdir='_nnTB/%s_%s'% (cNet['name'], time.strftime('%m.%d_%H.%M')), flush_secs=10)

    izT = [50,500,5000]
    indZeros = [[] for _ in izT]

    repFreq = 100
    hisFreq = 500
    nHighAcc = 0
    sTime = time.time()
    for b in range(nBatches):

        # feed loop for towers
        batches = [qmp.getResult() for _ in cNet.gFWD]
        feed = {}
        for ix in range(len(cNet.gFWD)):
            batch = batches[ix]
            tNet = cNet.gFWD[ix]
            feed.update({
                tNet['trPH']:       True,
                tNet['inACPH']:     batch['crd7AB'],
                tNet['inBCPH']:     batch['crd7BB'],
                tNet['wonPH']:      batch['winsB'],
                tNet['rnkAPH']:     batch['rankAB'],
                tNet['rnkBPH']:     batch['rankBB'],
                tNet['mcACPH']:     batch['mcAChanceB']})
        batch = batches[0]

        fetches = [
            cNet['optimizer'],
            cNet['loss'],
            cNet['lossW'],
            cNet['lossR'],
            cNet['lossMCAC'],
            cNet['avgAccW'],
            cNet['avgAccC'],
            cNet['predictionsW'],
            cNet['ohNotCorrect'],
            cNet['accR'],
            cNet['accRC'],
            cNet['predictionsRA'],
            cNet['ohNotCorrectRA'],
            cNet['gGNorm'],
            cNet['avtGGNorm'],
            cNet['scaledLR'],
            cNet['nnZeros']]
        lenNH = len(fetches)
        if b % hisFreq == 0: fetches.append(cNet['histSumm'])

        out = cNet['session'].run(fetches, feed_dict=feed)
        if len(out)==lenNH: out.append(None)
        _, loss, lossW, lossR, lossMCAC, acc, accC, pr, nc, accR, accRC, prRA, ncRA, gN, agN, lRs, zeros, histSumm = out

        if not zeros.size: zeros = np.asarray([0])
        for ls in indZeros: ls.append(zeros)

        if b % repFreq == 0:

            """
            # prints stats of rank @batch
            if verbLev > 2:
                rStats = batch[5] TODO: replace with dict (batch is not a list)
                nHands = 2*len(batch[0])
                for ix in range(len(rStats)):
                    rStats[ix] /= nHands
                    print('%.5f '%rStats[ix], end='')
                print()
                cum = 0
                for ix in range(len(rStats)):
                    cum += rStats[ix]
                    print('%.5f ' %cum, end='')
                print()

                wStats = batch[6]
                nWins = len(batch[0])
                for ix in range(len(wStats)):
                    wStats[ix] /= nWins
                    print('%.3f ' % wStats[ix], end='')
                print()
            """

            print('%6d, loss: %.6f, acc: %.6f, gN: %.6f, (%d/s)' % (b, loss, acc, gN, repFreq*trainSM[0]/(time.time()-sTime)))
            sTime = time.time()

            accsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/0_acc', simple_value=1-acc)])
            accRsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/1_accR', simple_value=1-accR)])
            losssum = tf.Summary(value=[tf.Summary.Value(tag='crdN/2_loss', simple_value=loss)])
            lossWsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/3_lossW', simple_value=lossW)])
            lossRsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/4_lossR', simple_value=lossR)])
            lossMCACsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/5_lossMCAC', simple_value=lossMCAC)])
            gNsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/6_gN', simple_value=gN)])
            agNsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/7_agN', simple_value=agN)])
            lRssum = tf.Summary(value=[tf.Summary.Value(tag='crdN/8_lRs', simple_value=lRs)])
            summWriter.add_summary(accsum, b)
            summWriter.add_summary(accRsum, b)
            summWriter.add_summary(losssum, b)
            summWriter.add_summary(lossWsum, b)
            summWriter.add_summary(lossRsum, b)
            summWriter.add_summary(lossMCACsum, b)
            summWriter.add_summary(gNsum, b)
            summWriter.add_summary(agNsum, b)
            summWriter.add_summary(lRssum, b)

            accRC = accRC.tolist()
            for cx in range(len(accRC)):
                csum = tf.Summary(value=[tf.Summary.Value(tag='Rca/%dca'%cx, simple_value=1-accRC[cx])])
                summWriter.add_summary(csum, b)
            accC = accC.tolist()
            accC01 = (accC[0]+accC[1])/2
            accC2 = accC[2]
            c01sum = tf.Summary(value=[tf.Summary.Value(tag='Wca/01ca', simple_value=1-accC01)])
            c2sum = tf.Summary(value=[tf.Summary.Value(tag='Wca/2ca', simple_value=1-accC2)])
            summWriter.add_summary(c01sum, b)
            summWriter.add_summary(c2sum, b)

            zerosT0 = np.mean(zeros)
            naneT0Summ = tf.Summary(value=[tf.Summary.Value(tag='nane/naneT0', simple_value=zerosT0)])
            summWriter.add_summary(naneT0Summ, b)
            for ix in range(len(izT)):
                cizT = izT[ix]
                if len(indZeros[ix]) > cizT - 1:
                    indZeros[ix] = indZeros[ix][-cizT:]
                    zerosT = np.mean(np.where(np.mean(np.stack(indZeros[ix], axis=0), axis=0)==1, 1, 0))
                    indZeros[ix] = []
                    naneTSumm = tf.Summary(value=[tf.Summary.Value(tag='nane/naneT%d' % cizT, simple_value=zerosT)])
                    summWriter.add_summary(naneTSumm, b)

            """
            if acc > 0.99: nHighAcc += 1
            if nHighAcc > 10:
                nS = prRA.shape[0]
                nBS = 0
                for sx in range(nS):
                    ncRASL = ncRA[sx].tolist()
                    if max(ncRASL):
                        nBS += 1
                        if nBS < 5:
                            cards = sorted(batch['crd7AB'][sx])
                            cards = [PDeck.itc(c) for c in cards]
                            cS7 = ''
                            for c in cards:
                                cS7 += ' %s' % PDeck.cts(c)
                            cr = PDeck.cardsRank(cards)
                            print(prRA[sx],ncRASL.index(1),cS7,cr[-1])
                if nBS: print(nBS)
                nBS = 0
                for sx in range(nS):
                    ncSL = nc[sx].tolist()
                    if max(ncSL):
                        nBS += 1
                        if nBS < 5:
                            cardsA = batch['crd7AB'][sx]
                            cardsA = [PDeck.itc(c) for c in cardsA]
                            cardsB = batch['crd7BB'][sx]
                            cardsB = [PDeck.itc(c) for c in cardsB]
                            cS7A = ''
                            for c in cardsA:
                                cS7A += ' %s' % PDeck.cts(c)
                            cS7A = cS7A[1:]
                            cS7B = ''
                            for c in cardsB:
                                cS7B += ' %s' % PDeck.cts(c)
                            cS7B = cS7B[1:]
                            crA = PDeck.cardsRank(cardsA)
                            crB = PDeck.cardsRank(cardsB)
                            print(pr[sx], ncSL.index(1), crA[-1][:2], crB[-1][:2], '(%s - %s = %s - %s)'%(cS7A,cS7B,crA[-1][3:],crB[-1][3:]))
                if nBS: print(nBS)
            """
        if histSumm: summWriter.add_summary(histSumm, b)

        # test
        if b%1000 == 0 and testBatch is not None:

            batch = testBatch
            feed = {
                cNet['inACPH']:     batch['crd7AB'],
                cNet['inBCPH']:     batch['crd7BB'],
                cNet['wonPH']:      batch['winsB'],
                cNet['rnkAPH']:     batch['rankAB'],
                cNet['rnkBPH']:     batch['rankBB'],
                cNet['mcACPH']:     batch['mcAChanceB']}

            fetches = [
                cNet['loss'],
                cNet['lossW'],
                cNet['lossR'],
                cNet['lossMCAC'],
                cNet['avgAccW'],
                cNet['avgAccC'],
                cNet['predictionsW'],
                cNet['ohNotCorrect'],
                cNet['accR'],
                cNet['accRC'],
                cNet['predictionsRA'],
                cNet['ohNotCorrectRA']]

            out = cNet['session'].run(fetches, feed_dict=feed)
            if len(out)==lenNH: out.append(None)
            loss, lossW, lossR, lossMCAC, acc, accC, pr, nc, accR, accRC, prRA, ncRA = out

            print('%6dT loss: %.7f acc: %.7f' % (b, loss, acc))

            accsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/0_acc', simple_value=1-acc)])
            accRsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/1_accR', simple_value=1-accR)])
            losssum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/2_loss', simple_value=loss)])
            lossWsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/3_lossW', simple_value=lossW)])
            lossRsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/4_lossR', simple_value=lossR)])
            lossMCACsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/5_lossMCAC', simple_value=lossMCAC)])
            summWriter.add_summary(accsum, b)
            summWriter.add_summary(accRsum, b)
            summWriter.add_summary(losssum, b)
            summWriter.add_summary(lossWsum, b)
            summWriter.add_summary(lossRsum, b)
            summWriter.add_summary(lossMCACsum, b)

            accC = accC.tolist()
            accC01 = (accC[0]+accC[1])/2
            accC2 = accC[2]
            c01sum = tf.Summary(value=[tf.Summary.Value(tag='WcaT/01ca', simple_value=1-accC01)])
            c2sum = tf.Summary(value=[tf.Summary.Value(tag='WcaT/2ca', simple_value=1-accC2)])
            summWriter.add_summary(c01sum, b)
            summWriter.add_summary(c2sum, b)

    cNet['saver'].save(step=cNet['globalStep'])
    qmp.close()
    if verbLev > 0: print('%s done' % cNet['name'])

# inference function
def inferW(
        cNet,
        session,
        batch):

    feed = {
        cNet['inACPH']: batch['crd7AB'],
        cNet['inBCPH']: batch['crd7BB']}

    fetches = [cNet['predictionsW']]
    return session.run(fetches, feed_dict=feed)

# inference wrap
def infer():

    verbLev = 1

    loggingSet(None, manageGPUs=False)

    cNet = NNModel(  # model
        mDict=      {'name': 'cNet'},
        fwdF=       cardFWDng,
        verbLev=    verbLev)

    session = tf.Session( # session
        graph=      cNet['graph'],
        config=     tf.ConfigProto(allow_soft_placement=True))

    nSaver = MultiSaver( # saver
        modelName=  cNet['name'],
        variables=  {'FWD': cNet['tVars']},
        savePath=   '_models',
        session=    session,
        verbLev=    verbLev)
    nSaver.load()

    bs = 1000000
    rs = 20
    inferBatch = prep2X7Batch(
        bs=         bs,
        rBalance=   False,
        dBalance=   False,
        nMonte=     0,
        verbLev=    verbLev)
    sTime = time.time()
    for ix in range(rs):
        res = inferW(cNet,session,inferBatch)
        print(ix)
    print('Finished, speed: %d/sec'%(int(bs*rs/(time.time()-sTime))))


if __name__ == "__main__":

    trainCardNet(cardNetDict={'name':'cNet'},nBatches=1000,trainSM=(1000,10),testSM=(2000,100),rQueTSize=200,verbLev=1)
    #infer()