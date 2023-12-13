import numpy as np
from pypaq.lipytools.files import prep_folder
import time
import unittest

from podecide.cardNet.cardNet_module import CardNet_MOTorch
from podecide.cardNet.cardNet_batcher import prep2X7batch

TMP_MODELS_DIR = f'tests/podecide/_tmp/_models/cn'
CardNet_MOTorch.SAVE_TOPDIR = TMP_MODELS_DIR


class Test_cardNet(unittest.TestCase):

    def setUp(self) -> None:
        prep_folder(TMP_MODELS_DIR, flush_non_empty=True)

    def test_base_init(self):

        cnet = CardNet_MOTorch(
            name=               'cn12',
            cards_emb_width=    12,
            read_only=          False,
            device=             -1,
            loglevel=           10,
            flat_child=         True,
        )
        cnet.save()

        cnet = CardNet_MOTorch(
            name=               'cn24',
            cards_emb_width=    24,
            read_only=          False,
            device=             -1,
            loglevel=           10,
            flat_child=         True,
        )
        cnet.save()

    def test_base_fwd(self):

        cnet = CardNet_MOTorch(
            cards_emb_width=    12,
            loglevel=           10,
            flat_child=         True,
        )

        # cards
        out = cnet(
            cards_A=    [1,2,3,4,5,6,7],
            cards_B=    [8,9,10,11,12,13,14])
        for k in ['logits_rank_A','logits_rank_B','reg_won_A','logits_winner']:
            print(k, out[k].shape)
        self.assertTrue(list(out['logits_rank_A'].shape) == list(out['logits_rank_B'].shape) == [9])
        self.assertTrue(list(out['reg_won_A'].shape) == [1])
        self.assertTrue(list(out['logits_winner'].shape) == [3])

        # bach of cards
        out = cnet(
            cards_A=    [[1,2,3,4,5,6,7]],
            cards_B=    [[8,9,10,11,12,13,14]])
        for k in ['logits_rank_A','logits_rank_B','reg_won_A','logits_winner']:
            print(k, out[k].shape)
        self.assertTrue(list(out['logits_rank_A'].shape) == list(out['logits_rank_B'].shape) == [1,9])
        self.assertTrue(list(out['reg_won_A'].shape) == [1,1])
        self.assertTrue(list(out['logits_winner'].shape) == [1,3])


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
        print(f'Finished, speed: {int(batch_size*runs/(time.time()-s_time))}d/sec')
