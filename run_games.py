"""

 2020 (c) piteren

"""


from putils.neuralmess.dev_manager import nestarter
from putils.mpython.mdecor import proc_wait
from putils.lipytools.decorators import timing

from decide.games_manager import GamesManager


if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    @timing # reports time
    @proc_wait # runs in waiting subprocess
    def run(gx_limit=10):
        gm = GamesManager(
            gname=          'cmk',
            #n_dmk=          4,
            #dmk_players=    15,
            #stats_iv=       1000,
            #acc_won_iv=     (5000,10000),
            verb=           1)
        gm.run_games(gx_limit=gx_limit)

    #run(gx_limit=2)
    while True: run()
