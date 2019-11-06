"""

 2019 (c) piteren

 card net training

 there are 2,598,960 hands not considering different suits it is (52/5) - combination

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

import numpy as np
import random
import tensorflow as tf
import time

from pUtils.nnTools.nnBaseElements import loggingSet
from pUtils.queMultiProcessor import QueMultiProcessor

from pLogic.pDeck import PDeck

from neuralGraphs import cardGFN

# prepares batch of 7cards, MP ready
def prepBatch(
        task=       None,   # needed by QMP, here passed avoidCTuples - list of sorted_card tuples to avoid in batch
        bs=         1000,
        rBalance=   True,   # balance rank
        dBalance=   0.1,    # False or fraction of draws
):

    deck = PDeck() # since it is hard to give any object to function of process...
    avoidCTuples = task

    crd7AB, crd7BB, winsB, rankAB, rankBB = [],[],[],[],[] # batches
    numRanks = [0]*9
    numWins = [0]*3
    hS = ['']*9
    for s in range(bs):
        deck.resetDeck()

        # look 4 the smallest number rank
        nMinRank = min(numRanks)
        desiredRank = numRanks.index(nMinRank)
        desiredDraw = False if dBalance is False else numWins[2] < dBalance * sum(numWins)

        crd7A = None
        crd7B = None
        aRank = None
        bRank = None
        gotDesiredCards = False
        while not gotDesiredCards:

            crd7A = deck.get7ofRank(desiredRank) if rBalance else [deck.getCard() for _ in range(7)] # 7 cards for A
            crd7B = [deck.getCard() for _ in range(2)] + crd7A[2:] # 2+5 cards for B

            # randomly swap hands of A with B to avoid wins bias
            if random.random() > 0.5:
                temp = crd7A
                crd7A = crd7B
                crd7B = temp

            # get cards ranks
            aRank = deck.cardsRank(crd7A)
            bRank = deck.cardsRank(crd7B)

            if not desiredDraw or (desiredDraw and aRank[1]==bRank[1]): gotDesiredCards = True
            if gotDesiredCards and type(avoidCTuples) is list and (tuple(sorted(crd7A)) in avoidCTuples or tuple(sorted(crd7B)) in avoidCTuples): gotDesiredCards = False

        paCRV = aRank[1]
        pbCRV = bRank[1]
        ha = aRank[0]
        hb = bRank[0]
        numRanks[aRank[0]]+=1
        numRanks[bRank[0]]+=1
        hS[aRank[0]] = aRank[-1]
        hS[bRank[0]] = bRank[-1]
        diff = paCRV-pbCRV
        wins = 0 if diff>0 else 1
        if diff==0: wins = 2 # remis
        numWins[wins] += 1

        # convert cards tuples to ints
        crd7A = [PDeck.cti(c) for c in crd7A]
        crd7B = [PDeck.cti(c) for c in crd7B]

        """
        # mask some table cards
        nMask = random.randrange(5)
        for ix in range(2+5-nMask,7):
            pa7[ix] = 52
            pb7[ix] = 52
        """

        crd7AB.append(crd7A)    # 7 cards of A
        crd7BB.append(crd7B)    # 7 cards of B
        winsB.append(wins)      # who wins {0,1,2}
        rankAB.append(ha)       # rank of A
        rankBB.append(hb)       # rank ok B

    #for s in hS: print(s)
    #print()
    return crd7AB, crd7BB, winsB, rankAB, rankBB, numRanks, numWins


if __name__ == "__main__":

    loggingSet('_log', customName='pypCN', forceLast=True)

    doTest = True
    #doTest = False
    if doTest:
        testSize = 10000
        testBatch = prepBatch(bs=testSize)
        cTuples = []
        for ix in range(testSize):
            cTuples.append(tuple(sorted(testBatch[0][ix])))
            cTuples.append(tuple(sorted(testBatch[1][ix])))
        print('\nGot %d of hands in testBatch'%len(cTuples))
        cTuples = dict.fromkeys(cTuples, 1)
        print('of which %d is unique'%len(cTuples))
    else:
        cTuples = None
        testBatch = None

    qmp = QueMultiProcessor(
        iProcFunction=  prepBatch,
        taskObject=     cTuples,
        rQueTSize=      100,
        verbLev=        1)

    cardNG = cardGFN(
        cEmbW=      24,
        nLayers=    8,
        denseMul=   4,
        drLayers=   2)

    session = tf.Session()

    session.run(tf.initializers.variables(var_list=cardNG['vars']+cardNG['optVars']))
    summWriter = tf.summary.FileWriter(logdir='_nnTB/crdN_%s'% time.strftime('%m.%d_%H.%M'), flush_secs=10)

    izT = [50,500,5000]
    indZeros = [[] for _ in izT]

    nHighAcc = 0

    repFreq = 100
    hisFreq = 500
    for b in range(200000):

        batch = qmp.getResult()

        feed = {
            cardNG['trPH']:     True,
            cardNG['inAC']:     batch[0],
            cardNG['inBC']:     batch[1],
            cardNG['won']:      batch[2],
            cardNG['rnkA']:     batch[3],
            cardNG['rnkB']:     batch[4]}

        fetches = [
            cardNG['optimizer'],
            cardNG['loss'],
            cardNG['acc'],
            cardNG['accC'],
            cardNG['predictions'],
            cardNG['ohNotCorrect'],
            cardNG['accR'],
            cardNG['accRC'],
            cardNG['predictionsRA'],
            cardNG['ohNotCorrectRA'],
            cardNG['gN'],
            cardNG['agN'],
            cardNG['lRs'],
            cardNG['nnZeros']]
        lenNH = len(fetches)
        if b % hisFreq == 0: fetches.append(cardNG['histSumm'])

        out = session.run(fetches, feed_dict=feed)
        if len(out)==lenNH: out.append(None)
        _, loss, acc, accC, pr, nc, accR, accRC, prRA, ncRA, gN, agN, lRs, zeros, histSumm = out

        if not zeros.size: zeros = np.asarray([0])
        for ls in indZeros: ls.append(zeros)

        if b % repFreq == 0:

            # prints stats of rank @batch
            """
            rStats = batch[5]
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

            #"""

            print('%6d, loss: %.6f, acc: %.6f, gN: %.6f' % (b, loss, acc, gN))

            accsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/1_acc', simple_value=1-acc)])
            accRsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/1_accR', simple_value=1-accR)])
            losssum = tf.Summary(value=[tf.Summary.Value(tag='crdN/2_loss', simple_value=loss)])
            gNsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/3_gN', simple_value=gN)])
            agNsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/4_agN', simple_value=agN)])
            lRssum = tf.Summary(value=[tf.Summary.Value(tag='crdN/5_lRs', simple_value=lRs)])
            summWriter.add_summary(accsum, b)
            summWriter.add_summary(accRsum, b)
            summWriter.add_summary(losssum, b)
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

            if acc > 0.99: nHighAcc += 1
            if nHighAcc > 10:
                nS = prRA.shape[0]
                nBS = 0
                for sx in range(nS):
                    ncRASL = ncRA[sx].tolist()
                    if max(ncRASL):
                        nBS += 1
                        if nBS < 5:
                            cards = sorted(batch[0][sx])
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
                            cardsA = batch[0][sx]
                            cardsA = [PDeck.itc(c) for c in cardsA]
                            cardsB = batch[1][sx]
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

        if histSumm: summWriter.add_summary(histSumm, b)

        # test
        if b%1000 == 0 and testBatch is not None:
            batch = testBatch

            feed = {
                cardNG['inAC']:     batch[0],
                cardNG['inBC']:     batch[1],
                cardNG['won']:      batch[2],
                cardNG['rnkA']:     batch[3],
                cardNG['rnkB']:     batch[4],
            }

            fetches = [
                cardNG['loss'],
                cardNG['acc'],
                cardNG['accC'],
                cardNG['predictions'],
                cardNG['ohNotCorrect'],
                cardNG['accR'],
                cardNG['accRC'],
                cardNG['predictionsRA'],
                cardNG['ohNotCorrectRA']]

            out = session.run(fetches, feed_dict=feed)
            if len(out)==lenNH: out.append(None)
            loss, acc, accC, pr, nc, accR, accRC, prRA, ncRA = out

            print('%6dT loss: %.7f acc: %.7f' % (b, loss, acc))

            accsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/1_acc', simple_value=1-acc)])
            accRsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/1_accR', simple_value=1-accR)])
            losssum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/2_loss', simple_value=loss)])
            summWriter.add_summary(accsum, b)
            summWriter.add_summary(accRsum, b)
            summWriter.add_summary(losssum, b)

            accC = accC.tolist()
            accC01 = (accC[0]+accC[1])/2
            accC2 = accC[2]
            c01sum = tf.Summary(value=[tf.Summary.Value(tag='WcaT/01ca', simple_value=1-accC01)])
            c2sum = tf.Summary(value=[tf.Summary.Value(tag='WcaT/2ca', simple_value=1-accC2)])
            summWriter.add_summary(c01sum, b)
            summWriter.add_summary(c2sum, b)

    qmp.close()
    print('done')
