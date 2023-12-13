import torch
import unittest

from envy import load_game_config
from podecide.dmk_module import ProCNN_DMK_PG
from podecide.dmk_motorch import DMK_MOTorch_PG

GAME_CONFIG = load_game_config(name='2players_2bets')
TBL_CFG = {k: GAME_CONFIG[k] for k in ['table_size', 'table_moves']}


class TestProCNN_DMK(unittest.TestCase):

    def test_module_call(self):
        dmk_net = ProCNN_DMK_PG(**TBL_CFG)
        out = dmk_net(
            cards=      torch.tensor([1,2,3,4,5,6,7]),
            event_id=   torch.tensor(0),
            cash=       torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
            pl_id=      torch.tensor(0),
            pl_pos=     torch.tensor(0),
            pl_stats=   torch.tensor(0.1),
        )
        print(out)

    def test_module_call_wrapped(self):
        dmk_net = ProCNN_DMK_PG(**TBL_CFG)
        out = dmk_net(
            cards=      torch.tensor([[1,2,3,4,5,6,7]]),
            event_id=   torch.tensor([0]),
            cash=       torch.tensor([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]),
            pl_id=      torch.tensor([0]),
            pl_pos=     torch.tensor([0]),
            pl_stats=   torch.tensor([0.1]),
        )
        print(out)

    def test_motorch_call(self):
        dmk_motorch = DMK_MOTorch_PG(device=None, **TBL_CFG)
        out = dmk_motorch(
            cards=      torch.tensor([1,2,3,4,5,6,7]),
            event_id=   torch.tensor(0),
            cash=       torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),
            pl_id=      torch.tensor(0),
            pl_pos=     torch.tensor(0),
            pl_stats=   torch.tensor(0.1),
        )
        print(out)
