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
            stats_iv=   5000,
            acc_won_iv= (50000,100000),
            verb=       1)
        gx_last_list = gm.run_games(gx_limit=20)
        xres = xross(gx_last_list, n_par=6, n_mix=7, verb=2)
