"""

 2019 (c) piteren

    cardNet running script (training, infer)

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
import tensorflow as tf
import time

from ptools.lipytools.little_methods import prep_folder
from ptools.neuralmess.dev_manager import nestarter
from ptools.neuralmess.nemodel import NEModel
from ptools.neuralmess.base_elements import ZeroesProcessor
from ptools.mpython.qmp import QueMultiProcessor

from pypoks_envy import MODELS_FD, CN_MODELS_FD, get_cardNet_name
from pologic.podeck import PDeck
from podecide.cardNet.cardNet_batcher import prep2X7Batch, get_test_batch
from podecide.cardNet.cardNet_graph import card_net


# training function
def train_cn(
        cn_dict :dict,
        device=     -1,
        n_batches=  50000,
        tr_SM=      (1000,10),      # train (size,montecarlo samples)
        ts_SM=      (2000,100000),  # test  (size,montecarlo samples)
        do_test=    True,
        rq_trgsize= 200,
        rep_freq=   100,
        his_freq=   500,
        verb=       0):

    prep_folder(MODELS_FD)
    prep_folder(CN_MODELS_FD)

    test_batch, c_tuples = None, None
    if do_test: test_batch, c_tuples = get_test_batch(ts_SM[0], ts_SM[1])

    iPF = partial(prep2X7Batch, bs=tr_SM[0], n_monte=tr_SM[1])
    qmp = QueMultiProcessor( # QMP
        proc_func=      iPF,
        rq_trgsize=     rq_trgsize,
        verb=           verb)

    cnet = NEModel( # model
        fwd_func=       card_net,
        mdict=          cn_dict,
        devices=        device,
        save_TFD=       CN_MODELS_FD,
        verb=           verb)

    ze_pro = ZeroesProcessor(
        intervals=      (50,500,5000),
        tag_pfx=        '7_zeroes',
        summ_writer=    cnet.summ_writer)

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
                tNet['train_PH']:   True,
                tNet['inA_PH']:     batch['cA'],
                tNet['inB_PH']:     batch['cB'],
                tNet['won_PH']:     batch['wins'],
                tNet['rnkA_PH']:    batch['rA'],
                tNet['rnkB_PH']:    batch['rB'],
                tNet['mcA_PH']:     batch['mAWP']})
        batch = batches[0]

        fetches = [
            cnet['optimizer'],
            cnet['loss'],
            cnet['loss_W'],
            cnet['loss_R'],
            cnet['loss_AWP'],
            cnet['acc_W'],
            cnet['acc_WC'],
            cnet['predictions_W'],
            cnet['oh_notcorrect_W'],
            cnet['acc_R'],
            cnet['acc_RC'],
            cnet['predictions_R'],
            cnet['oh_notcorrect_R'],
            cnet['gg_norm'],
            cnet['avt_gg_norm'],
            cnet['scaled_LR'],
            cnet['zeroes']]
        lenNH = len(fetches)
        if his_freq and b % his_freq == 0: fetches.append(cnet['hist_summ'])

        out = cnet.session.run(fetches, feed_dict=feed)
        if len(out)==lenNH: out.append(None)
        _, loss, loss_W, loss_R, loss_AWP, acc_W, acc_WC, pred_W, ohnc_W, acc_R, acc_RC, pred_R, ohnc_R, gn, agn, lRs, zs, hist_summ = out

        if hist_summ: cnet.summ_writer.add_summary(hist_summ, b)

        ze_pro.process(zs, b)

        if b % rep_freq == 0:

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

            print('%6d, loss: %.6f, accW: %.6f, gn: %.6f, (%d/s)' % (b, loss, acc_W, gn, rep_freq * tr_SM[0] / (time.time() - sTime)))
            sTime = time.time()

            accsum = tf.Summary(value=[tf.Summary.Value(tag='1_crdN/0_iacW', simple_value=1-acc_W)])
            accRsum = tf.Summary(value=[tf.Summary.Value(tag='1_crdN/1_iacR', simple_value=1-acc_R)])
            losssum = tf.Summary(value=[tf.Summary.Value(tag='1_crdN/2_loss', simple_value=loss)])
            lossWsum = tf.Summary(value=[tf.Summary.Value(tag='1_crdN/3_lossW', simple_value=loss_W)])
            lossRsum = tf.Summary(value=[tf.Summary.Value(tag='1_crdN/4_lossR', simple_value=loss_R)])
            lossAWPsum = tf.Summary(value=[tf.Summary.Value(tag='1_crdN/5_lossAWP', simple_value=loss_AWP)])
            gNsum = tf.Summary(value=[tf.Summary.Value(tag='1_crdN/6_gn', simple_value=gn)])
            agNsum = tf.Summary(value=[tf.Summary.Value(tag='1_crdN/7_agn', simple_value=agn)])
            lRssum = tf.Summary(value=[tf.Summary.Value(tag='1_crdN/8_lRs', simple_value=lRs)])
            cnet.summ_writer.add_summary(accsum, b)
            cnet.summ_writer.add_summary(accRsum, b)
            cnet.summ_writer.add_summary(losssum, b)
            cnet.summ_writer.add_summary(lossWsum, b)
            cnet.summ_writer.add_summary(lossRsum, b)
            cnet.summ_writer.add_summary(lossAWPsum, b)
            cnet.summ_writer.add_summary(gNsum, b)
            cnet.summ_writer.add_summary(agNsum, b)
            cnet.summ_writer.add_summary(lRssum, b)

            acc_RC = acc_RC.tolist()
            for cx in range(len(acc_RC)):
                csum = tf.Summary(value=[tf.Summary.Value(tag=f'3_Rcia/{cx}ica', simple_value=1-acc_RC[cx])])
                cnet.summ_writer.add_summary(csum, b)

            acc_WC = acc_WC.tolist()
            accC01 = (acc_WC[0]+acc_WC[1])/2
            accC2 = acc_WC[2]
            c01sum = tf.Summary(value=[tf.Summary.Value(tag='5_Wcia/01cia', simple_value=1-accC01)])
            c2sum = tf.Summary(value=[tf.Summary.Value(tag='5_Wcia/2cia', simple_value=1-accC2)])
            cnet.summ_writer.add_summary(c01sum, b)
            cnet.summ_writer.add_summary(c2sum, b)

            #""" reporting of almost correct cases in late training
            if acc_W > 0.99: nHighAcc += 1
            if nHighAcc > 10 and acc_W < 1:
                nS = pred_R.shape[0] # batch size (num samples)
                nBS = 0
                for sx in range(nS):
                    ncRsl = ohnc_R[sx].tolist() # OH sample
                    if max(ncRsl): # there is 1 >> not correct
                        nBS += 1
                        if nBS < 3: # print max 2
                            cards = sorted(batch['cB'][sx])
                            cS7 = ''
                            for c in cards:
                                cS7 += ' %s' % PDeck.cts(c)
                            cr = PDeck.cards_rank(cards)
                            print(pred_R[sx],ncRsl.index(1),cS7,cr[-1])
                if nBS: print(nBS)
                nBS = 0
                for sx in range(nS):
                    ncWsl = ohnc_W[sx].tolist()
                    if max(ncWsl):
                        nBS += 1
                        if nBS < 3:
                            cardsA = batch['cA'][sx]
                            cardsB = batch['cB'][sx]
                            cS7A = ''
                            for c in cardsA: cS7A += ' %s' % PDeck.cts(c)
                            cS7A = cS7A[1:]
                            cS7B = ''
                            for c in cardsB: cS7B += ' %s' % PDeck.cts(c)
                            cS7B = cS7B[1:]
                            crA = PDeck.cards_rank(cardsA)
                            crB = PDeck.cards_rank(cardsB)
                            print(pred_W[sx], ncWsl.index(1), crA[-1][:2], crB[-1][:2], '(%s - %s = %s - %s)'%(cS7A,cS7B,crA[-1][3:],crB[-1][3:]))
                if nBS: print(nBS)
            #"""

        # test
        if b%1000 == 0 and test_batch is not None:

            batch = test_batch
            feed = {
                cnet['inA_PH']:     batch['cA'],
                cnet['inB_PH']:     batch['cB'],
                cnet['won_PH']:     batch['wins'],
                cnet['rnkA_PH']:    batch['rA'],
                cnet['rnkB_PH']:    batch['rB'],
                cnet['mcA_PH']:     batch['mAWP']}

            fetches = [
                cnet['loss'],
                cnet['loss_W'],
                cnet['loss_R'],
                cnet['loss_AWP'],
                cnet['diff_AWP_mn'],
                cnet['diff_AWP_mx'],
                cnet['acc_W'],
                cnet['acc_WC'],
                cnet['predictions_W'],
                cnet['oh_notcorrect_W'],
                cnet['acc_R'],
                cnet['acc_RC'],
                cnet['predictions_R'],
                cnet['oh_notcorrect_R']]

            out = cnet.session.run(fetches, feed_dict=feed)
            if len(out)==lenNH: out.append(None)
            loss, loss_W, loss_R, loss_AWP, dAWPmn, dAWPmx, acc_W, acc_WC, pred_W, ohnc_W, acc_R, acc_RC, pred_R, ohnc_R = out

            print('%6dT loss: %.7f accW: %.7f' % (b, loss, acc_W))

            accsum = tf.Summary(value=[tf.Summary.Value(tag='2_crdNT/0_iacW', simple_value=1-acc_W)])
            accRsum = tf.Summary(value=[tf.Summary.Value(tag='2_crdNT/1_iacR', simple_value=1-acc_R)])
            losssum = tf.Summary(value=[tf.Summary.Value(tag='2_crdNT/2_loss', simple_value=loss)])
            lossWsum = tf.Summary(value=[tf.Summary.Value(tag='2_crdNT/3_lossW', simple_value=loss_W)])
            lossRsum = tf.Summary(value=[tf.Summary.Value(tag='2_crdNT/4_lossR', simple_value=loss_R)])
            lossAWPsum = tf.Summary(value=[tf.Summary.Value(tag='2_crdNT/5_lossAWP', simple_value=loss_AWP)])
            dAWPmnsum = tf.Summary(value=[tf.Summary.Value(tag='2_crdNT/6_dAWPmn', simple_value=dAWPmn)])
            dAWPmxsum = tf.Summary(value=[tf.Summary.Value(tag='2_crdNT/7_dAWPmx', simple_value=dAWPmx)])
            cnet.summ_writer.add_summary(accsum, b)
            cnet.summ_writer.add_summary(accRsum, b)
            cnet.summ_writer.add_summary(losssum, b)
            cnet.summ_writer.add_summary(lossWsum, b)
            cnet.summ_writer.add_summary(lossRsum, b)
            cnet.summ_writer.add_summary(lossAWPsum, b)
            cnet.summ_writer.add_summary(dAWPmnsum, b)
            cnet.summ_writer.add_summary(dAWPmxsum, b)

            acc_RC = acc_RC.tolist()
            for cx in range(len(acc_RC)):
                csum = tf.Summary(value=[tf.Summary.Value(tag=f'4_RciaT/{cx}ca', simple_value=1-acc_RC[cx])]) # cia stands for "classification inverted accuracy"
                cnet.summ_writer.add_summary(csum, b)

            acc_WC = acc_WC.tolist()
            accC01 = (acc_WC[0]+acc_WC[1])/2
            accC2 = acc_WC[2]
            c01sum = tf.Summary(value=[tf.Summary.Value(tag='6_WciaT/01cia', simple_value=1-accC01)])
            c2sum = tf.Summary(value=[tf.Summary.Value(tag='6_WciaT/2cia', simple_value=1-accC2)])
            cnet.summ_writer.add_summary(c01sum, b)
            cnet.summ_writer.add_summary(c2sum, b)

    cnet.saver.save(step=cnet['g_step'])
    qmp.close()
    if verb > 0: print('%s done' % cnet['name'])

# inference on given batch
def infer(cn, batch):

    feed = {
        cn['inA_PH']: batch['cA'],
        cn['inB_PH']: batch['cB']}

    fetches = [cn['predictions_W']]
    return cn.session.run(fetches, feed_dict=feed)


if __name__ == "__main__":

    device = -1
    c_embW = 12

    name = get_cardNet_name(c_embW)

    nestarter(custom_name=name, devices=False)

    cn_dict = {
        'name':         name,
        'emb_width':    c_embW,
        'verb':         1}

    train_cn(
        cn_dict=        cn_dict,
        device=         device,
        #his_freq=       0,
        verb=           1)