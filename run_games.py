"""

 2020 (c) piteren

"""


from putils.neuralmess.dev_manager import nestarter
from putils.mtasking.decor import subprocess_wait

from decide.games_manager import GamesManager


if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    # reload loop (runs in waiting subprocess)
    @subprocess_wait
    def reload_loop():
        gm = GamesManager(
            #n_dmk=          4,
            #dmk_players=    15,
            #stats_iv=       1000,
            #acc_won_iv=     (5000,10000),
            verb=           1)
        gm.run_games(gx_limit=2)
        print('...loop finished!')

    while True: reload_loop()
