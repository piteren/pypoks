from ompr.runner import RunningWorker, OMPRunner
from pypaq.mpython.mpdecor import proc
import torch
from torchness.comoneural.zeroes_processor import ZeroesProcessor
import time

from podecide.cardNet.cardNet_module import CardNet_MOTorch #, get_cardNet

from pologic.podeck import PDeck
from podecide.cardNet.cardNet_batcher import prep2X7batch, get_test_batch


class Batch2X7_RW(RunningWorker):

    def __init__(
            self,
            batch_size: int,
            n_monte: int):
        self.deck = PDeck()
        self.batch_size = batch_size
        self.n_monte = n_monte

    def process(self, **kwargs):
        batch = prep2X7batch(
            deck=       self.deck,
            batch_size= self.batch_size,
            n_monte=    self.n_monte)
        batch.pop('rank_counter')
        batch.pop('won_counter')
        return {k: torch.tensor(batch[k]) for k in batch}


# training function
def train_cardNet(
        cards_emb_width: int,
        device=             -1,
        n_batches=          50000,
        tr_SM=              (1000,10),      # train (batch_size, montecarlo samples)
        ts_SM=              (2000,100000),  # test  (batch_size, montecarlo samples)
        do_test=            True,
        target_ready_tasks= 200,
        rep_freq=           100,
        loglevel=           20):

    cnet = CardNet_MOTorch(
        cards_emb_width=    cards_emb_width,
        device=             device,
        use_huber=          True,
        read_only=          False,
        loglevel=           loglevel)
    #print(cnet)
    logger = cnet.logger

    test_batch = None
    if do_test:
        test_batch, _ = get_test_batch(*ts_SM)
        test_batch.pop('rank_counter')
        test_batch.pop('won_counter')
        test_batch = {k: cnet.convert(data=test_batch[k]) for k in test_batch}

    ompr = OMPRunner(
        rw_class=           Batch2X7_RW,
        rw_init_kwargs=     {'batch_size':tr_SM[0], 'n_monte':tr_SM[1]},
        devices=            0.9,
        ordered_results=    False,
        logger=             logger)
    task_pack = [{}] * (target_ready_tasks // 10) # tasks we add to OMP at once

    # assert len(cnet.gFWD) <= rq_trg, 'ERR: needs to increase rq_trg!'

    ze_pro = ZeroesProcessor(
        intervals=  (50,500,5000),
        tag_pfx=    '7_zeroes',
        tbwr=       cnet.tbwr)

    nHighAcc = 0
    sTime = time.time()
    for b in range(1, n_batches):

        task_st = ompr.get_tasks_stats()
        if task_st['n_tasks_received'] - task_st['n_results_returned'] < target_ready_tasks:
            ompr.process(tasks=task_pack)

        batch = ompr.get_result()
        out = cnet.backward(**batch)

        ze_pro.process(out['zsL'])

        if b % rep_freq == 0:

            loss =          out['loss']
            loss_W =        out['loss_winner']
            loss_R =        out['loss_rank']
            loss_AWP =      out['loss_won_A']
            acc_W =         out['accuracy_winner']
            acc_R =         out['accuracy_rank']
            gn =            out['gg_norm']
            gnc =           out['gg_norm_clip']
            lRs =           out['currentLR']
            speed = int(rep_freq * tr_SM[0] / (time.time() - sTime))
            sTime = time.time()

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

            logger.info(f'{b:6}, loss: {loss:.6f}, accW: {acc_W:.6f}, gn: {gn:.6f}, {speed} H/s')

            cnet.log_TB(value=1-acc_W,  tag='1_crdN/0_iacW',    step=b)
            cnet.log_TB(value=1-acc_R,  tag='1_crdN/1_iacR',    step=b)
            cnet.log_TB(value=loss,     tag='1_crdN/2_loss',    step=b)
            cnet.log_TB(value=loss_W,   tag='1_crdN/3_lossW',   step=b)
            cnet.log_TB(value=loss_R,   tag='1_crdN/3_lossR',   step=b)
            cnet.log_TB(value=loss_AWP, tag='1_crdN/5_lossAWP', step=b)
            cnet.log_TB(value=gn,       tag='1_crdN/6_gn',      step=b)
            cnet.log_TB(value=gnc,      tag='1_crdN/7_gnc',     step=b)
            cnet.log_TB(value=lRs,      tag='1_crdN/8_lRs',     step=b)
            cnet.log_TB(value=speed,    tag='1_crdN/9_speed',   step=b)

            """
            acc_RC = acc_RC.tolist()
            for cx in range(len(acc_RC)):
                cnet.log_TB(value=1-acc_RC[cx], tag=f'3_Rcia/{cx}ica', step=b)

            acc_WC = acc_WC.tolist()
            accC01 = (acc_WC[0]+acc_WC[1])/2
            accC2 = acc_WC[2]
            cnet.log_TB(value=1-accC01, tag='5_Wcia/01cia', step=b)
            cnet.log_TB(value=1-accC2,  tag='5_Wcia/2cia',  step=b)
            """

            """ reporting of almost correct cases in late training
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

            out = cnet.backward(**test_batch)

            """
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
            """
            loss =          out['loss']
            loss_W =        out['loss_winner']
            loss_R =        out['loss_rank']
            loss_AWP =      out['loss_won_A']
            acc_W =         out['accuracy_winner']
            acc_R =         out['accuracy_rank']
            dAWPmn =        out['diff_won_prob_mean']
            dAWPmx =        out['diff_won_prob_max']

            logger.info('%6dT loss: %.7f accW: %.7f' % (b, loss, acc_W))

            cnet.log_TB(value=1-acc_W,  tag='2_crdNT/0_iacW',    step=b)
            cnet.log_TB(value=1-acc_R,  tag='2_crdNT/1_iacR',    step=b)
            cnet.log_TB(value=loss,     tag='2_crdNT/2_loss',    step=b)
            cnet.log_TB(value=loss_W,   tag='2_crdNT/3_lossW',   step=b)
            cnet.log_TB(value=loss_R,   tag='2_crdNT/3_lossR',   step=b)
            cnet.log_TB(value=loss_AWP, tag='2_crdNT/5_lossAWP', step=b)
            cnet.log_TB(value=dAWPmn,   tag='2_crdNT/6_dAWPmn',  step=b)
            cnet.log_TB(value=dAWPmx,   tag='2_crdNT/7_dAWPmx',  step=b)

            """
            acc_RC = acc_RC.tolist()
            for cx in range(len(acc_RC)):
                cnet.log_TB(value=1-acc_RC[cx], tag=f'4_RciaT/{cx}ca', step=b) # cia stands for "classification inverted accuracy"

            acc_WC = acc_WC.tolist()
            accC01 = (acc_WC[0]+acc_WC[1])/2
            accC2 = acc_WC[2]
            cnet.log_TB(value=1-accC01, tag='6_WciaT/01cia', step=b)
            cnet.log_TB(value=1-accC2,  tag='6_WciaT/2cia',  step=b)
            """

    cnet.save()
    ompr.exit()
    logger.info(f'{cnet["name"]} training done!')

@proc
def train_wrap(cards_emb_width, device, **kwargs):
    train_cardNet(
        cards_emb_width=    cards_emb_width,
        device=             device,
        **kwargs)


if __name__ == "__main__":
    #train_wrap(cards_emb_width=12, device=0)
    train_wrap(cards_emb_width=24, device=1)