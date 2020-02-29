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

from putils.neuralmess.dev_manager import nestarter
from putils.neuralmess.multi_saver import MultiSaver
from putils.neuralmess.nemodel import NNModel
from putils.que_MProcessor import QueMultiProcessor

from pLogic.pDeck import PDeck
from cardNet.card_batcher import prep2X7Batch, getTestBatch
from cardNet.card_network import card_net

#TODO: - loss components influence


# training function
def train_cn(
        cn_dict :dict,
        n_batches=  50000,
        tr_SM=      (1000,10), # train (size,montecarlo)
        ts_SM=      (2000,100),   # test (size,montecarlo)
        do_test=    True,
        rQueTSize=  200,
        verb=       0):

    test_batch, c_tuples = None, None
    if do_test: test_batch, c_tuples = getTestBatch(ts_SM[0], ts_SM[1])

    iPF = partial(prep2X7Batch, bs=tr_SM[0], nMonte=tr_SM[1])
    qmp = QueMultiProcessor( # QMP
        iProcFunction=  iPF,
        #taskObject=     c_tuples,
        #nProc=          10,
        rQueTSize=      rQueTSize,
        verb=           verb)

    cnet = NNModel( # model
        fwd_func=       card_net,
        mdict=          cn_dict,
        #devices=
        verb=           verb)

    izT = [50,500,5000]
    indZerosA = [[] for _ in izT]
    indZerosB = [[] for _ in izT]

    repFreq = 100
    hisFreq = 500
    nHighAcc = 0
    sTime = time.time()
    for b in range(1, n_batches):

        # feed loop for towers
        batches = [qmp.getResult() for _ in cnet.gFWD]
        feed = {}
        for ix in range(len(cnet.gFWD)):
            batch = batches[ix]
            tNet = cnet.gFWD[ix]
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
            cnet['optimizer'],
            cnet['loss'],
            cnet['lossW'],
            cnet['lossR'],
            cnet['lossPAWR'],
            cnet['avgAccW'],
            cnet['avgAccWC'],
            cnet['predictionsW'],
            cnet['ohNotCorrectW'],
            cnet['accR'],
            cnet['accRC'],
            cnet['predictionsR'],
            cnet['ohNotCorrectR'],
            cnet['gg_norm'],
            cnet['avt_gg_norm'],
            cnet['scaled_LR'],
            cnet['nn_zerosA'],
            cnet['nn_zerosB']]
        lenNH = len(fetches)
        if b % hisFreq == 0: fetches.append(cnet['hist_summ'])

        out = cnet.session.run(fetches, feed_dict=feed)
        if len(out)==lenNH: out.append(None)
        _, loss, lossW, lossR, lossPAWR, accW, accWC, prW, ncW, accR, accRC, prR, ncR, gN, agN, lRs, zerosA, zerosB, histSumm = out

        if histSumm: cnet.summ_writer.add_summary(histSumm, b)

        if not zerosA.size: zerosA = np.asarray([0])
        for ls in indZerosA: ls.append(zerosA)
        if not zerosB.size: zerosB = np.asarray([0])
        for ls in indZerosB: ls.append(zerosB)

        if b % repFreq == 0:

            """ prints stats of rank @batch
            if verbLev > 2:
                rStats = batch['numRanks']
                nHands = 2*len(batch['crd7AB'])
                for ix in range(len(rStats)):
                    rStats[ix] /= nHands
                    print('%.5f '%rStats[ix], end='')
                print()
                cum = 0
                for ix in range(len(rStats)):
                    cum += rStats[ix]
                    print('%.5f ' %cum, end='')
                print()

                wStats = batch['numWins']
                nWins = nHands / 2
                for ix in range(len(wStats)):
                    wStats[ix] /= nWins
                    print('%.3f ' % wStats[ix], end='')
                print()
            #"""

            print('%6d, loss: %.6f, accW: %.6f, gN: %.6f, (%d/s)' % (b, loss, accW, gN, repFreq * tr_SM[0] / (time.time() - sTime)))
            sTime = time.time()

            accsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/0_accW', simple_value=1-accW)])
            accRsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/1_accR', simple_value=1-accR)])
            losssum = tf.Summary(value=[tf.Summary.Value(tag='crdN/2_loss', simple_value=loss)])
            lossWsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/3_lossW', simple_value=lossW)])
            lossRsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/4_lossR', simple_value=lossR)])
            lossMCACsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/5_lossPAWR', simple_value=lossPAWR)])
            gNsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/6_gN', simple_value=gN)])
            agNsum = tf.Summary(value=[tf.Summary.Value(tag='crdN/7_agN', simple_value=agN)])
            lRssum = tf.Summary(value=[tf.Summary.Value(tag='crdN/8_lRs', simple_value=lRs)])
            cnet.summ_writer.add_summary(accsum, b)
            cnet.summ_writer.add_summary(accRsum, b)
            cnet.summ_writer.add_summary(losssum, b)
            cnet.summ_writer.add_summary(lossWsum, b)
            cnet.summ_writer.add_summary(lossRsum, b)
            cnet.summ_writer.add_summary(lossMCACsum, b)
            cnet.summ_writer.add_summary(gNsum, b)
            cnet.summ_writer.add_summary(agNsum, b)
            cnet.summ_writer.add_summary(lRssum, b)

            accRC = accRC.tolist()
            for cx in range(len(accRC)):
                csum = tf.Summary(value=[tf.Summary.Value(tag='Rca/%dca'%cx, simple_value=1-accRC[cx])])
                cnet.summ_writer.add_summary(csum, b)

            accWC = accWC.tolist()
            accC01 = (accWC[0]+accWC[1])/2
            accC2 = accWC[2]
            c01sum = tf.Summary(value=[tf.Summary.Value(tag='Wca/01ca', simple_value=1-accC01)])
            c2sum = tf.Summary(value=[tf.Summary.Value(tag='Wca/2ca', simple_value=1-accC2)])
            cnet.summ_writer.add_summary(c01sum, b)
            cnet.summ_writer.add_summary(c2sum, b)

            zerosAT0 = np.mean(zerosA)
            naneT0Summ = tf.Summary(value=[tf.Summary.Value(tag='nane/naneAT0', simple_value=zerosAT0)])
            cnet.summ_writer.add_summary(naneT0Summ, b)
            for ix in range(len(izT)):
                cizT = izT[ix]
                if len(indZerosA[ix]) > cizT - 1:
                    indZerosA[ix] = indZerosA[ix][-cizT:]
                    zerosAT = np.mean(np.where(np.mean(np.stack(indZerosA[ix], axis=0), axis=0)==1, 1, 0))
                    indZerosA[ix] = []
                    naneTSumm = tf.Summary(value=[tf.Summary.Value(tag='nane/naneAT%d' % cizT, simple_value=zerosAT)])
                    cnet.summ_writer.add_summary(naneTSumm, b)
            zerosBT0 = np.mean(zerosB)
            naneT0Summ = tf.Summary(value=[tf.Summary.Value(tag='nane/naneBT0', simple_value=zerosBT0)])
            cnet.summ_writer.add_summary(naneT0Summ, b)
            for ix in range(len(izT)):
                cizT = izT[ix]
                if len(indZerosB[ix]) > cizT - 1:
                    indZerosB[ix] = indZerosB[ix][-cizT:]
                    zerosBT = np.mean(np.where(np.mean(np.stack(indZerosB[ix], axis=0), axis=0) == 1, 1, 0))
                    indZerosB[ix] = []
                    naneTSumm = tf.Summary(value=[tf.Summary.Value(tag='nane/naneBT%d' % cizT, simple_value=zerosBT)])
                    cnet.summ_writer.add_summary(naneTSumm, b)

            #""" reporting of almost correct cases in late training
            if accW > 0.99: nHighAcc += 1
            if nHighAcc > 10 and accW < 1:
                nS = prR.shape[0] # batch size (num samples)
                nBS = 0
                for sx in range(nS):
                    ncRsl = ncR[sx].tolist() # OH sample
                    if max(ncRsl): # there is 1 >> not correct
                        nBS += 1
                        if nBS < 3: # print max 2
                            cards = sorted(batch['crd7BB'][sx])
                            cS7 = ''
                            for c in cards:
                                cS7 += ' %s' % PDeck.cts(c)
                            cr = PDeck.cardsRank(cards)
                            print(prR[sx],ncRsl.index(1),cS7,cr[-1])
                if nBS: print(nBS)
                nBS = 0
                for sx in range(nS):
                    ncWsl = ncW[sx].tolist()
                    if max(ncWsl):
                        nBS += 1
                        if nBS < 3:
                            cardsA = batch['crd7AB'][sx]
                            cardsB = batch['crd7BB'][sx]
                            cS7A = ''
                            for c in cardsA: cS7A += ' %s' % PDeck.cts(c)
                            cS7A = cS7A[1:]
                            cS7B = ''
                            for c in cardsB: cS7B += ' %s' % PDeck.cts(c)
                            cS7B = cS7B[1:]
                            crA = PDeck.cardsRank(cardsA)
                            crB = PDeck.cardsRank(cardsB)
                            print(prW[sx], ncWsl.index(1), crA[-1][:2], crB[-1][:2], '(%s - %s = %s - %s)'%(cS7A,cS7B,crA[-1][3:],crB[-1][3:]))
                if nBS: print(nBS)
            #"""

        # test
        if b%1000 == 0 and test_batch is not None:

            batch = test_batch
            feed = {
                cnet['inACPH']:     batch['crd7AB'],
                cnet['inBCPH']:     batch['crd7BB'],
                cnet['wonPH']:      batch['winsB'],
                cnet['rnkAPH']:     batch['rankAB'],
                cnet['rnkBPH']:     batch['rankBB'],
                cnet['mcACPH']:     batch['mcAChanceB']}

            fetches = [
                cnet['loss'],
                cnet['lossW'],
                cnet['lossR'],
                cnet['lossPAWR'],
                cnet['diffPAWRmn'],
                cnet['diffPAWRmx'],
                cnet['avgAccW'],
                cnet['avgAccWC'],
                cnet['predictionsW'],
                cnet['ohNotCorrectW'],
                cnet['accR'],
                cnet['accRC'],
                cnet['predictionsR'],
                cnet['ohNotCorrectR']]

            out = cnet.session.run(fetches, feed_dict=feed)
            if len(out)==lenNH: out.append(None)
            loss, lossW, lossR, lossPAWR, dPAWRmn, dPAWRmx, accW, accWC, prW, ncW, accR, accRC, prR, ncR = out

            print('%6dT loss: %.7f accW: %.7f' % (b, loss, accW))

            accsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/0_accW', simple_value=1-accW)])
            accRsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/1_accR', simple_value=1-accR)])
            losssum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/2_loss', simple_value=loss)])
            lossWsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/3_lossW', simple_value=lossW)])
            lossRsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/4_lossR', simple_value=lossR)])
            lossMCACsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/5_lossPAWR', simple_value=lossPAWR)])
            dPAWRmnsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/6_dPAWRmn', simple_value=dPAWRmn)])
            dPAWRmxsum = tf.Summary(value=[tf.Summary.Value(tag='crdNT/7_dPAWRmx', simple_value=dPAWRmx)])
            cnet.summ_writer.add_summary(accsum, b)
            cnet.summ_writer.add_summary(accRsum, b)
            cnet.summ_writer.add_summary(losssum, b)
            cnet.summ_writer.add_summary(lossWsum, b)
            cnet.summ_writer.add_summary(lossRsum, b)
            cnet.summ_writer.add_summary(lossMCACsum, b)
            cnet.summ_writer.add_summary(dPAWRmnsum, b)
            cnet.summ_writer.add_summary(dPAWRmxsum, b)

            accWC = accWC.tolist()
            accC01 = (accWC[0]+accWC[1])/2
            accC2 = accWC[2]
            c01sum = tf.Summary(value=[tf.Summary.Value(tag='WcaT/01ca', simple_value=1-accC01)])
            c2sum = tf.Summary(value=[tf.Summary.Value(tag='WcaT/2ca', simple_value=1-accC2)])
            cnet.summ_writer.add_summary(c01sum, b)
            cnet.summ_writer.add_summary(c2sum, b)

    cnet.saver.save(step=cnet['globalStep'])
    qmp.close()
    if verb > 0: print('%s done' % cnet['name'])

# inference function
def inferW(
        cNet,
        batch):

    feed = {
        cNet['inACPH']: batch['crd7AB'],
        cNet['inBCPH']: batch['crd7BB']}

    fetches = [cNet['predictionsW']]
    return cNet.session.run(fetches, feed_dict=feed)

# inference wrap
# TODO: rewrite for NNmodel interface
def infer():

    verbLev = 1

    cNet = NNModel(  # model
        fwd_func=   card_net,
        mdict=      {'name': 'cNet'},
        verb=       verbLev)

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
        res = inferW(cNet,inferBatch)
        print(ix)
    print('Finished, speed: %d/sec'%(int(bs*rs/(time.time()-sTime))))


if __name__ == "__main__":

    nestarter(devices=None, verb=1)

    cndGD = {
        'name':         'cnet_test',
        'opt_class':    tf.train.GradientDescentOptimizer,
        'iLR':          3e-2,
        'warm_up':      None,
        'ann_base':     1}

    train_cn(
        cn_dict=        cndGD,
        n_batches=      200000,
        tr_SM=          (1000,10),
        ts_SM=          (2000,10000000),
        rQueTSize=      200,
        verb=           1)
    #infer()