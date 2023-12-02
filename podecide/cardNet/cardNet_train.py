from pypaq.mpython.mpdecor import proc
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.pylogger import get_child
from torchness.comoneural.zeroes_processor import ZeroesProcessor
import time

from podecide.cardNet.cardNet_module import CardNet_MOTorch
from podecide.cardNet.cardNet_batcher import get_train_batches, get_test_batch, OMPRunner_Batch2X7



def train_cardNet(
        cards_emb_width: int,
        device=             -1,
        n_batches=          10000,
        tr_batch_size=      1000,
        tr_n_monte=         10,
        tr_get_cache=       True,
        ts_batch_size=      2000,
        ts_n_monte=         1000,
        do_test=            True,
        target_ready_tasks= 500,
        tasks_pack_size=    10,
        rep_freq=           100,
        loglevel=           20,
):
    """ CN training function """

    card_net = CardNet_MOTorch(
        #name=               f'cn12{stamp(month=False, letters=None)}',
        #n_layers=           12,
        #ann_step=           0.02,
        cards_emb_width=    cards_emb_width,
        device=             device,
        gc_do_clip=         True,
        read_only=          False,
        loglevel=           loglevel)
    logger = card_net.logger
    logger.info(card_net)

    ompr = None
    train_batches = None
    n_train_batches = 0
    if tr_get_cache:
        train_batches = get_train_batches(
            batch_size= tr_batch_size,
            n_monte=    tr_n_monte,
            logger=     get_child(logger))
        n_train_batches = len(train_batches)
    else:
        ompr = OMPRunner_Batch2X7(
            batch_size= tr_batch_size,
            n_monte=    tr_n_monte,
            devices=    0.9,
            logger=     get_child(logger))

    test_batch = None
    if do_test:
        test_batch, _ = get_test_batch(batch_size=ts_batch_size, n_monte=ts_n_monte)
        test_batch = {k: card_net.convert(data=test_batch[k]) for k in test_batch}

    ze_pro = ZeroesProcessor(
        intervals=  (50,500,5000),
        tag_pfx=    '7_zeroes',
        tbwr=       card_net.tbwr)

    sTime = time.time()
    for bix in range(1, n_batches):

        if ompr:
            task_st = ompr.get_tasks_stats()
            n_tasks = target_ready_tasks - (task_st['n_tasks_received'] - task_st['n_results_returned'])
            if n_tasks > 0:
                ompr.process(tasks=[{}] * tasks_pack_size)

            batch = ompr.get_result()

        else:
            batch = train_batches[bix % n_train_batches]

        out = card_net.backward(**batch)

        ze_pro.process(out['zeroes'])

        if bix % rep_freq == 0:

            loss =    out['loss']
            acc_WIN = out['accuracy_winner']
            gn =      out['gg_norm']
            speed = int(rep_freq * tr_batch_size / (time.time() - sTime))
            sTime = time.time()

            logger.info(f'{bix:6}, loss:{loss:.6f}, accW:{acc_WIN:.6f}, gn:{gn:.6f}, {speed}H/s')

            card_net.log_TB(value=1-acc_WIN,              tag='1_CN_TR/0_inv_accWIN', step=bix)
            card_net.log_TB(value=1-out['accuracy_rank'], tag='1_CN_TR/1_inv_accRNK', step=bix)
            card_net.log_TB(value=loss,                   tag='1_CN_TR/2_loss',       step=bix)
            card_net.log_TB(value=out['loss_winner'],     tag='1_CN_TR/3_lossWIN',    step=bix)
            card_net.log_TB(value=out['loss_rank'],       tag='1_CN_TR/3_lossRNK',    step=bix)
            card_net.log_TB(value=out['loss_won_A'],      tag='1_CN_TR/5_lossAWP',    step=bix)
            card_net.log_TB(value=gn,                     tag='1_CN_TR/6_gn',         step=bix)
            card_net.log_TB(value=out['gg_norm_clip'],    tag='1_CN_TR/7_gnc',        step=bix)
            card_net.log_TB(value=out['currentLR'],       tag='1_CN_TR/8_LRs',        step=bix)
            card_net.log_TB(value=speed,                  tag='1_CN_TR/9_speed',      step=bix)

        # test
        if bix%1000 == 0 and test_batch is not None:

            out = card_net.loss(**test_batch, bypass_data_conv=True)
            loss =          out['loss']
            acc_WIN =       out['accuracy_winner']

            logger.info(f'{bix:6}, loss:{loss:.6f}, accW:{acc_WIN:.6f}')

            card_net.log_TB(value=1-acc_WIN,                 tag='2_CN_TS/0_inv_accWIN',  step=bix) # inverted accuracy 1-..
            card_net.log_TB(value=1-out['accuracy_rank'],    tag='2_CN_TS/1_inv_accRNK',  step=bix)
            card_net.log_TB(value=loss,                      tag='2_CN_TS/2_loss',        step=bix)
            card_net.log_TB(value=out['loss_winner'],        tag='2_CN_TS/3_lossWIN',     step=bix)
            card_net.log_TB(value=out['loss_rank'],          tag='2_CN_TS/3_lossRNK',     step=bix)
            card_net.log_TB(value=out['loss_won_A'],         tag='2_CN_TS/5_lossAWP',     step=bix)
            card_net.log_TB(value=out['diff_won_prob_mean'], tag='2_CN_TS/6_diffAWPmean', step=bix)
            card_net.log_TB(value=out['diff_won_prob_max'],  tag='2_CN_TS/7_diffAWPmax',  step=bix)

            card_net.tbwr.add_histogram(values=out['diff_won_prob'], tag='TS/won_prob', step=bix)

    card_net.save()
    if ompr: ompr.exit()
    logger.info(f'{card_net["name"]} training done!')

@proc
def train_wrap(**kwargs):
    train_cardNet(**kwargs)


if __name__ == "__main__":

    base_conf = {
        'n_batches':        50000,
        'tr_batch_size':    1000,
        'tr_n_monte':       10,
        'ts_batch_size':    2000,
        'ts_n_monte':       100000,
    }
    more_conf = {
        'n_batches':        200000,
        'tr_batch_size':    1000,
        'tr_n_monte':       100,
        'ts_batch_size':    2000,
        'ts_n_monte':       10000000,
        'loglevel':         10,
    }

    #config = {}
    #config = base_conf
    config = more_conf

    #train_wrap(cards_emb_width=12, device=0, **config)
    train_wrap(cards_emb_width=24, device=1, **config)