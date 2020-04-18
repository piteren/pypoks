"""

 2020 (c) piteren

"""


from putils.neuralmess.dev_manager import nestarter
from putils.mpython.mpdecor import proc_wait
from putils.lipytools.decorators import timing

from decide.games_manager import GamesManager


if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    pmex_init = 0.2
    pmex_trg = 0.05

    @timing # reports time
    @proc_wait # runs in waiting subprocess
    def run(
            gx_limit=   10,
            pmex_init=  0.2,
            pmex_trg=   0.02):
        gm = GamesManager(
            dmk_names=      [f'fm{ix}' for ix in range(14)],#+[f'bm{ix}' for ix in range(14)],
            #n_players=      15,
            pmex_init=      pmex_init,
            pmex_trg=       pmex_trg,
            #stats_iv=       1000,
            #acc_won_iv=     (5000,10000),
            verb=           1)
        gm.run_games(gx_limit=gx_limit)

    #run(gx_limit=2)
    loopIX = 0
    while True:
        run(
            pmex_init=  pmex_init if not loopIX else pmex_trg,
            pmex_trg=   pmex_trg)
        loopIX += 1
