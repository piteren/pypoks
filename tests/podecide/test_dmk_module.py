import torch
from torchness.motorch import MOTorch
import unittest

from podecide.dmk_module_pg import ProCNN_DMK_PG


class TestProCNN_DMK(unittest.TestCase):

    def test_init_module(self):
        dmk_net = ProCNN_DMK_PG()
        dmk_net.card_net.load_ckpt() # load pretrained cardNet
        dmk_net = dmk_net.cpu()
        #print(dmk_net)
        out = dmk_net(
            cards=  torch.tensor([[1,2,3,4,5,6,7]]),
            event=  torch.tensor([0]),
            switch= torch.tensor([1]),
        )
        print(out)

    def test_init_motorch(self):
        dmk_motorch = MOTorch(
            module_type=    ProCNN_DMK_PG,
            device=         None)
        out = dmk_motorch(
            cards=  [[1,2,3,4,5,6,7]],
            event=  [0],
            switch= [1])
        print(out)