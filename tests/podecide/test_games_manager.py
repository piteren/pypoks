import unittest

from envy import load_game_config, PyPoksException
from podecide.games_manager import GamesManager, GamesManager_PTR

GAME_CONFIG = load_game_config(name='2players_2bets')


class TestGamesManager(unittest.TestCase):

    def test_short_game(self):
        gm = GamesManager(
            game_config=    GAME_CONFIG,
            dmk_pointL=     [{
                'name':                 f'dmk{n:02}',
                'publish_player_stats': False,
                'publishFWD':           False,
                'publishUPD':           False,
            } for n in range(3)],
            loglevel=       20,
        )
        res = gm.run_game(game_size=10000, sleep=1)['dmk_results']
        print(res)

    def test_random_PTR(self):
        """ tests random ref_pattern in GM PTR """

        rpL = []
        for _ in range(3):
            gm = GamesManager_PTR(
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
        gm = GamesManager(
            game_config=    GAME_CONFIG,
            dmk_pointL=     [{
                'name':         f'dmk{n:02}',
                'n_players':    100,
                'motorch_point': {
                    'load_cardnet_pretrained':  True,
                    'device':                   None},
            } for n in range(2)],
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
        self.assertRaises(PyPoksException, GamesManager_PTR, **kwargs)

        GamesManager_PTR(
            game_config=    GAME_CONFIG,
            dmk_point_PLL=  None,
            dmk_point_TRL=  [{'name': f'dmkT{n:02}'} for n in range(10)],
            dmk_n_players=  60)

        GamesManager_PTR(
            game_config=    GAME_CONFIG,
            dmk_point_PLL=  [{'name': f'dmkP{n:02}'} for n in range(10)],
            dmk_point_TRL=  None,
            dmk_n_players=  60)


    def test_run_PTR(self):

        gm = GamesManager_PTR(
            game_config=    GAME_CONFIG,
            dmk_point_TRL=  [{'name':f'dmk{n:02}'} for n in range(10)],
            dmk_n_players=  100,
            loglevel=       20,
        )
        res = gm.run_game(game_size=20000)['dmk_results']
        print(res)