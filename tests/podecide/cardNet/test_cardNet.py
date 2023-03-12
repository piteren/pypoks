import numpy as np
import time
import unittest

from podecide.cardNet.cardNet_module import CardNet_MOTorch
from podecide.cardNet.cardNet_batcher import prep2X7batch


class Test_cardNet(unittest.TestCase):

    def test_base(self):

        cnet = CardNet_MOTorch(cards_emb_width=12)
        #print(cnet)

        out = cnet(
            cards_A=    [1,2,3,4,5,6,7],
            cards_B=    [8,9,10,11,12,13,14])
        for k in ['logits_rank_A','logits_rank_B','reg_won_A','logits_winner']:
            print(k, out[k].shape)

        out = cnet(
            cards_A=    [[1,2,3,4,5,6,7]],
            cards_B=    [[8,9,10,11,12,13,14]])
        for k in ['logits_rank_A','logits_rank_B','reg_won_A','logits_winner']:
            print(k, out[k].shape)


    def test_card_enc(self):

        cnet = CardNet_MOTorch(cards_emb_width=12)
        card_enc = cnet.module.card_enc
        #print(card_enc)

        for cards in [
            [1,2,3,4,5,6,7],                                    # sequence of cards (gives 2-dim input tensor to Transformer)
            np.random.randint(low=0, high=52, size=(5,7)),      # 5 batches of sequences of cards
            np.random.randint(low=0, high=52, size=(4,1,7)),    # 4 players of seq(1) of sequences of cards
            np.random.randint(low=0, high=52, size=(4,3,7)),    # 4 players of seq(3) of sequences of cards
        ]:
            cards = cnet.convert(cards)
            out = card_enc(cards=cards)
            ce = out['out']
            print(ce.shape)
            self.assertTrue(ce.shape[-1] == card_enc.enc_width)
            self.assertTrue(ce.shape[:-1] == cards.shape[:-1])


    def test_batch_inference(
            self,
            batch_size=     10000,
            runs=           20):
        cnet = CardNet_MOTorch(cards_emb_width=12, device=-1)

        infer_batch = prep2X7batch(
            batch_size= batch_size,
            r_balance=  False,
            d_balance=  False,
            n_monte=    0)
        infer_batch = {k: infer_batch[k] for k in ['cards_A','cards_B']}
        s_time = time.time()
        for ix in range(runs):
            out = cnet(**infer_batch)
            print(ix)
        print('Finished, speed: %d/sec'%(int(batch_size*runs/(time.time()-s_time))))
