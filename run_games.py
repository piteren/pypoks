"""

 2020 (c) piteren

"""


from putils.neuralmess.dev_manager import nestarter

from decide.games_manager import GamesManager
from decide.gx import xross


if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    # reload loop
    while True:
        gm = GamesManager(
            n_dmk=      4,
            dmk_players=15,
            stats_iv=   1000,
            acc_won_iv= (5000,10000),
            verb=       1)
        gx_last_list = gm.run_games(gx_limit=2)
