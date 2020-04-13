"""

 2020 (c) piteren

"""


from putils.neuralmess.dev_manager import nestarter

from decide.games_manager import GamesManager


if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    gm = GamesManager(
        stats_iv=   5000,
        acc_won_iv= (50000,100000),
        verb=       1)
    gm.run_games()
