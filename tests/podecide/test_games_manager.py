import unittest

from podecide.dmk import RanDMK, NeurDMK, FolDMK
from podecide.games_manager import GamesManager, GamesManager_PTR


class TestGamesManager(unittest.TestCase):

    def test_short_RanDMK(self):
        gm = GamesManager(
            dmk_pointL= [{
                'dmk_type': RanDMK,
                'name':     f'dmk{n:02}',
                'publish_player_stats': False,
                'publish_pex':          False,
                'publish_more':         False,
            } for n in range(3)],
            loglevel=   20,
        )
        res_list = gm.run_game(game_size=100000, sleep=1)['dmk_results']
        for r in res_list: print(r)

    def test_some_NeurDMK(self):
        gm = GamesManager(
            dmk_pointL= [{
                'dmk_type':     NeurDMK,
                'name':         f'dmk{n:02}',
                'n_players':    300,
                'module_point':   {
                    'load_cardnet_pretrained':  True,
                    'device':                   n%2},
            } for n in range(10)],
            loglevel=   20,
        )
        res_list = gm.run_game(game_size=3000000, sleep=1)['dmk_results']
        for r in res_list: print(r)


class TestGamesManager_PTR(unittest.TestCase):

    def test_init_GamesManager_PTR(self):

        gm = GamesManager_PTR(
            dmk_point_PLL=  [{'name': f'dmkP{n:02}'} for n in range(8)],
            dmk_point_TRL=  [{'name': f'dmkT{n:02}'} for n in range(10)],
            dmk_n_players=  60,
            verb=           1)

        gm = GamesManager_PTR(
            dmk_point_PLL=  None,
            dmk_point_TRL=  [{'name': f'dmkT{n:02}'} for n in range(10)],
            dmk_n_players=  60,
            verb=           1)

        gm = GamesManager_PTR(
            dmk_point_PLL=  [{'name': f'dmkP{n:02}'} for n in range(10)],
            dmk_point_TRL=  None,
            dmk_n_players=  60,
            verb=           1)


    def test_big_run(self):

        gm = GamesManager_PTR(
            game_size=      20000,
            dmk_point_PLL=    [{
                'dmk_type': FolDMK,
                'name':     f'dmkP{n:02}'
            } for n in range(10)],
            dmk_point_TRL=    [{
                'dmk_type': FolDMK,
                'name':     f'dmkT{n:02}'
            } for n in range(10)],
            dmk_n_players=  15,
            loglevel=       20
        )
        res_list = gm.run_game()['dmk_results']
        for r in res_list: print(r)