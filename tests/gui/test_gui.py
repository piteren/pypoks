import unittest

from envy import load_game_config
from gui.human_game_gui import HumanGameGUI


class TestPDeck(unittest.TestCase):

    def test_just_show(self):

        game_config = load_game_config('2players_2bets')

        tk_gui = HumanGameGUI(
            players=    [f'pl{ix}' for ix in range(game_config['table_size'])],
            imgs_FD=    'gui/imgs',
            **game_config)
        tk_gui.run_loop()