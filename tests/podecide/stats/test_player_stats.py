import unittest

from envy import load_game_config
from pologic.potable import PTable
from podecide.stats.player_stats import PStatsEx


class TestPStatsEx(unittest.TestCase):

    def test_base(self):

        game_config = load_game_config('2players_2bets')

        pse = PStatsEx(
            player=         'pl0',
            table_size=     game_config['table_size'],
            table_moves=    game_config['table_moves'],
            use_initial=    False,
            upd_freq=       1)

        psei = PStatsEx(
            player=         'pl0',
            table_size=     game_config['table_size'],
            table_moves=    game_config['table_moves'],
            use_initial=    True,
            initial_size=   100,
            upd_freq=       10)

        table = PTable(
            name=       'table',
            moves=      game_config['table_moves'],
            cash_start= game_config['table_cash_start'],
            cash_sb=    game_config['table_cash_sb'],
            cash_bb=    game_config['table_cash_bb'],
            pl_ids= [f'pl{ix}' for ix in range(game_config['table_size'])])

        for r in range(500):
            hh = table.run_hand()
            pse.process_states(hh.events)
            psei.process_states(hh.events)
            stats = pse.player_stats
            ss = ''
            for k in stats:
                ss += f'{k}:{stats[k]:.3f} '
            print(f'{ss[:-1]}')

        print(f'\n{pse}')
        print(f'\n{psei}')