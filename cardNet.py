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

 for seven cards when try_to_balance it looks like (from 0 to 8):
 0.11485 0.22095 0.16230 0.10895 0.08660 0.09515 0.08695 0.06490 0.05935
 0.11485 0.33580 0.49810 0.60705 0.69365 0.78880 0.87575 0.94065 1.00000

"""

import random
import tensorflow as tf
import time

from pUtils.nnTools.nnBaseElements import loggingSet
from pUtils.queMultiProcessor import QueMultiProcessor

from pLogic.pDeck import PDeck

from neuralGraphs import cardGFN


def prepBatch(
        task=       None, # needed by QMP
        bs=         1000,
        tBalance=   True):

    deck = PDeck() # since it is hard to give any object to function of process...

    crd7AB, crd7BB, winsB, rankAB, rankBB = [],[],[],[],[] # batches
    numRanks = [0]*9
    #hS = ['']*9
    for s in range(bs):
        deck.resetDeck()

        # look 4 the smallest number rank
        nMinRank = min(numRanks)
        desiredRank = numRanks.index(nMinRank)
        crd7A = deck.get7ofRank(desiredRank) if tBalance else [deck.getCard() for _ in range(7)] # 7 cards for A
        crd7B = [deck.getCard() for _ in range(2)] + crd7A[2:] # 7 cards for B

        # randomly swap A with B to avoid wins bias
        if tBalance:
            if random.random() > 0.5:
                temp = crd7A
                crd7A = crd7B
                crd7B = temp

        # get cards ranks and calc labels
        aRank = deck.cardsRank(crd7A)
        bRank = deck.cardsRank(crd7B)
        paCRV = aRank[1]
        pbCRV = bRank[1]
        ha = aRank[0]
        hb = bRank[0]
        numRanks[aRank[0]]+=1
        numRanks[bRank[0]]+=1
        #hS[paCR[0]] = paCR[-1]
        #hS[pbCR[0]] = pbCR[-1]
        diff = paCRV-pbCRV
        wins = 0 if diff>0 else 1
        if diff==0: wins = 2 # remis

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
    return crd7AB, crd7BB, winsB, rankAB, rankBB, numRanks


if __name__ == "__main__":

    loggingSet('_log', customName='pypCN', forceLast=True)

    qmp = QueMultiProcessor(
        iProcFunction=  prepBatch,
        #nProc=          10,
        verbLev=        1)

    cardNG = cardGFN(
        wC=         12,#6,#2,
        nLayers=    12,#36,#24,
        rWidth=     128,#,#256
        drLayers=   None,
        lR=         3e-4,#1e-5 # for nLays==48 lR=1e-4
        #doClip=     False
    )

    session = tf.Session()

    session.run(tf.initializers.variables(var_list=cardNG['vars']+cardNG['optVars']))
    summWriter = tf.summary.FileWriter(logdir='_nnTB/crdN_%s'% time.strftime('%m.%d_%H.%M'), flush_secs=10)

    for b in range(100000):

        batch = qmp.getResult()

        # prints stats of rank @batch
        """
        hStats = batch[-1]
        nHands = 2*len(batch[0])
        for ix in range(len(hStats)):
            hStats[ix] /= nHands
            print('%.5f '%hStats[ix], end='')
        print()
        cum = 0
        for ix in range(len(hStats)):
            cum += hStats[ix]
            print('%.5f ' %cum, end='')
        print()
        #"""

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
