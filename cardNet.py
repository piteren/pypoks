"""

 2019 (c) piteren

 card net training

 there are 2,598,960 hands not counting different suits it is (52/5) - combination

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

 for seven cards it looks like (from 0 to 8):
 0.17740 0.44400 0.23040 0.04785 0.04150 0.03060 0.02660 0.00145 0.00020 (fraction)
 0.17740 0.62140 0.85180 0.89965 0.94115 0.97175 0.99835 0.99980 1.00000 (cumulative fraction)

"""

import random
import tensorflow as tf
import time

from pUtils.nnTools.nnBaseElements import loggingSet
from pUtils.queMultiProcessor import QueMultiProcessor

from pLogic.pDeck import PDeck

from neuralGraphs import cardGFN


def prepBatch(
        task=   None,
        bs=     10000):

    deck = PDeck()
    pa7B, pb7B, wB, haB, hbB = [],[],[],[],[]
    nH = [0]*9
    #hS = ['']*9
    for _ in range(bs):
        deck.resetDeck()
        paC = [deck.getCard() for _ in range(2)] # two cards for A
        pbC = [deck.getCard() for _ in range(2)] # two cards for B
        cmC = [deck.getCard() for _ in range(5)] # five cards from table
        pa7 = paC + cmC
        pb7 = pbC + cmC

        # get cards ranks and calc labels
        paCR = deck.cardsRank(pa7)
        pbCR = deck.cardsRank(pb7)
        paCRV = paCR[1]
        pbCRV = pbCR[1]
        ha = paCR[0]
        hb = pbCR[0]
        nH[paCR[0]]+=1
        nH[pbCR[0]]+=1
        #hS[paCR[0]] = paCR[-1]
        #hS[pbCR[0]] = pbCR[-1]
        diff = paCRV-pbCRV
        wins = 0 if diff>0 else 1
        if diff==0: wins = 2

        # convert cards tuples to ints
        pa7 = [PDeck.cti(c) for c in pa7]
        pb7 = [PDeck.cti(c) for c in pb7]

        """
        # mask some table cards
        nMask = random.randrange(5)
        for ix in range(2+5-nMask,7):
            pa7[ix] = 52
            pb7[ix] = 52
        """

        pa7B.append(pa7)
        pb7B.append(pb7)
        wB.append(wins)
        haB.append(ha)
        hbB.append(hb)

    #for s in hS: print(s)
    #print()
    return pa7B, pb7B, wB, haB, hbB, nH


if __name__ == "__main__":

    loggingSet('_log', customName='pypCN', forceLast=True)

    qmp = QueMultiProcessor(
        iProcFunction=  prepBatch,
        #nProc=          10,
    )

    cardNG = cardGFN(
        wC=         2,
        nLayers=    24,
        rWidth=     256,
        lR=         1e-3, # for nLays==48 lR=1e-4
        doClip=     False
    )
    session = tf.Session()

    session.run(tf.initializers.variables(var_list=cardNG['vars']+cardNG['optVars']))
    summWriter = tf.summary.FileWriter(logdir='_nnTB/crdN_%s'% time.strftime('%m.%d_%H.%M'), flush_secs=10)

    for b in range(1000000):

        batch = qmp.getResult()
        """
        hStats = batch[-1]
        for ix in range(len(hStats)):
            hStats[ix] /= 20000
            print('%.5f '%hStats[ix], end='')
        print()
        cum = 0
        for ix in range(len(hStats)):
            cum += hStats[ix]
            print('%.5f ' %cum, end='')
        print()
        """

        feed = {
            cardNG['inAC']:     batch[0],
            cardNG['inBC']:     batch[1],
            cardNG['won']:      batch[2]}

        fetches = [cardNG['optimizer'], cardNG['loss'], cardNG['acc'], cardNG['gN'], cardNG['agN']]
        _, loss, acc, gN, agN = session.run(fetches, feed_dict=feed)
        print('%6d, loss: %.3f, acc: %.3f, gN: %.3f'%(b, loss, acc, gN))
        if b%100 == 0:
            accsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/1_acc', simple_value=acc)])
            losssum = tf.Summary(value=[tf.Summary.Value(tag='crdN/2_loss', simple_value=loss)])
            gNsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/3_gN', simple_value=gN)])
            agNsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/4_agN', simple_value=agN)])
            summWriter.add_summary(accsum, b)
            summWriter.add_summary(losssum, b)
            summWriter.add_summary(gNsum, b)
            summWriter.add_summary(agNsum, b)

    qmp.close()
    print('done')
