import unittest

from envy import load_game_config, PyPoksException
from podecide.dmk import RanDMK, NeurDMK
from podecide.dmk_motorch import DMK_MOTorch_PG
from podecide.game_manager import GameManager, GameManager_PTR

GAME_CONFIG = load_game_config(name='2players_2bets')


class TestGameManager(unittest.TestCase):

    def test_short_game(self):
        points = [{
            'name':                 f'dmk{n:02}',
            'table_size':           GAME_CONFIG['table_size'],
            'table_moves':          GAME_CONFIG['table_moves'],
            'trainable':            False,
            'publish_player_stats': False,
            'publishFWD':           False,
            'publishUPD':           False,
        } for n in range(2)]
        gm = GameManager(
            game_config=    GAME_CONFIG,
            dmks_recipe=    [(RanDMK,point) for point in points],
            loglevel=       20,
        )
        res = gm.run_game(game_size=10000, sleep=1)['dmk_results']
        print(res)

    def test_random_PTR(self):
        """ tests random ref_pattern in GM PTR """

        rpL = []
        for _ in range(3):
            gm = GameManager_PTR(
                game_config=    GAME_CONFIG,
                dmk_point_refL= [{'name': f'dmk_r{n:02}'} for n in range(3)],
                dmk_point_PLL=  [{'name': f'dmk{n:02}'} for n in range(3)],
                loglevel=       20,
            )
            rp = gm.ref_pattern
            rpL.append(rp)

        prev = None
        for rp in rpL:
            print(rp)
            if prev is None:
                prev = rp
            else:
                self.assertTrue(rp == prev)
                prev = rp

    def test_upd_sync_on_cpu(self):
        points = [{
            'name':             f'dmk{n:02}',
            'n_players':        100,
            'table_size':       GAME_CONFIG['table_size'],
            'table_moves':      GAME_CONFIG['table_moves'],
            'table_cash_start': GAME_CONFIG['table_cash_start'],
            'motorch_type':     DMK_MOTorch_PG,
            'motorch_point':    {
                'load_cardnet_pretrained':  True,
                'device':                   None},
            'publish_player_stats': False,
            'publishFWD':       False,
            'publishUPD':       False,
        } for n in range(2)]
        gm = GameManager(
            game_config=    GAME_CONFIG,
            dmks_recipe=    [(NeurDMK,point) for point in points],
            loglevel=       10,
        )
        res = gm.run_game(game_size=30000, sleep=1)['dmk_results']
        print(res)


class TestGamesManager_PTR(unittest.TestCase):

    def test_init_GamesManager_PTR(self):

        kwargs = {
            'game_config':    GAME_CONFIG,
            'dmk_point_PLL':  [{'name': f'dmkP{n:02}'} for n in range(8)],
            'dmk_point_TRL':  [{'name': f'dmkT{n:02}'} for n in range(10)],
            'dmk_n_players':  60}
        self.assertRaises(PyPoksException, GameManager_PTR, **kwargs)

        GameManager_PTR(
            game_config=    GAME_CONFIG,
            dmk_point_PLL=  None,
            dmk_point_TRL=  [{'name': f'dmkT{n:02}'} for n in range(10)],
            dmk_n_players=  60)

        GameManager_PTR(
            game_config=    GAME_CONFIG,
            dmk_point_PLL=  [{'name': f'dmkP{n:02}'} for n in range(10)],
            dmk_point_TRL=  None,
            dmk_n_players=  60)


    def test_run_PTR(self):

        gm = GameManager_PTR(
            game_config=    GAME_CONFIG,
            dmk_point_TRL=  [{'name':f'dmk{n:02}'} for n in range(10)],
            dmk_n_players=  100,
            loglevel=       20,
        )
        res = gm.run_game(game_size=20000)['dmk_results']
        print(res)