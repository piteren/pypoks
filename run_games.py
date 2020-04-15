"""

 2020 (c) piteren

"""


from putils.neuralmess.dev_manager import nestarter
from putils.mtasking.mdecor import subprocess_wait
from putils.lipytools.decorators import timing

from decide.games_manager import GamesManager


if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    @timing # reports time
    @subprocess_wait # runs in waiting subprocess
    def run(gx_limit=5):
        gm = GamesManager(
            gname=          'bmk',
            #n_dmk=          4,
            #dmk_players=    15,
            stats_iv=       1000,
            acc_won_iv=     (2000,300000),
            #acc_won_iv=     (5000,10000),
            verb=           1)
        gm.run_games(gx_limit=gx_limit)
        print('...loop finished!')

    run(gx_limit=1)
    # while True: run()
