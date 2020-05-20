"""

 2019 (c) piteren

    cardNet inference script


"""


import time

from ptools.neuralmess.dev_manager import nestarter
from ptools.neuralmess.nemodel import NEModel

from podecide.cardNet.cardNet_batcher import prep2X7Batch
from podecide.cardNet.cardNet_graph import card_net
from pypoks_envy import CN_MODELS_FD, get_cardNet_name


# inference on given batch
def infer(cn, batch):

    feed = {
        cn['inA_PH']: batch['cA'],
        cn['inB_PH']: batch['cB']}

    fetches = [cn['predictions_W']]
    return cn.session.run(fetches, feed_dict=feed)

# inference wrap
def example_inference(
        cn_dict,
        device,
        bs=     100000,
        rs=     20,
        verb=   1):

    cnet = NEModel(
        fwd_func=   card_net,
        mdict=      cn_dict,
        devices=    device,
        save_TFD=   CN_MODELS_FD,
        verb=       verb)

    infer_batch = prep2X7Batch(
        bs=         bs,
        r_balance=  False,
        d_balance=  False,
        n_monte=    0,
        verb=       verb)

    s_time = time.time()
    for ix in range(rs):
        res = infer(cnet, infer_batch)
        print(ix)
    print('Finished, speed: %d/sec'%(int(bs*rs/(time.time()-s_time))))


if __name__ == "__main__":

    device = -1
    c_embW = 12

    name = get_cardNet_name(c_embW)

    nestarter(custom_name=name, devices=False)

    cn_dict = {
        'name':         name,
        'emb_width':    c_embW,
        'verb':         1}

    example_inference(cn_dict, device, verb=1)