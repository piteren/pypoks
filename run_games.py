"""

 2020 (c) piteren

"""

from putils.neuralmess.dev_manager import nestarter
from putils.mpython.mpdecor import proc_wait
from putils.lipytools.decorators import timing

from decide.games_manager import GamesManager
from decide.neural.neural_graphs import cnnCEM_GFN


@timing # reports time
@proc_wait # runs in waiting subprocess
def run(ddna, gx_limit=10):
    gm = GamesManager(
        dmk_dna=        ddna,
        acc_won_iv=     (100000,200000),
        verb=           1)
    gm.run_games(
        #gx_loop_sh=     False,
        #gx_exit_sh=     False,
        gx_limit=       gx_limit)


if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    dmk_dna = {
        f'fm{ix}':
            {
                'family':       None,
                'fwd_func':     cnnCEM_GFN,
                'mdict':        {'verb':0},
                'n_players':    150,
                'pmex_init':    0.0,#0.2,
                'pmex_trg':     0.0,#0.05,
                'stats_iv':     10000,
                'trainable':    False,
                'verb':         0,
            } for ix in range(7)}
    #"""
    for k in ['fm0','fm4','fm3','fm6','fm10','fm11','fm9']: dmk_dna.pop(k)
    dmk_dna.update({
        f'gm{ix}':
            {
                'family':       'G',
                'fwd_func':     cnnCEM_GFN,
                'mdict':        {'verb':0},
                'n_players':    150,
                'pmex_init':    0.2,
                'pmex_trg':     0.05,
                'stats_iv':     10000,
                'trainable':    True,
                'verb':         0,
            } for ix in range(7)})
    #"""
    loopIX = 0
    while True:
        if loopIX:
            #break  # to break after first loop
            for dn in dmk_dna: dmk_dna[dn]['pmex_init'] = dmk_dna[dn]['pmex_trg']
        run(dmk_dna)
        loopIX += 1
