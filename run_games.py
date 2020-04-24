"""

 2020 (c) piteren

"""
from functools import partial
import tensorflow as tf

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
        f'am{ix}': {
                'family':       'A',
                'fwd_func':     cnnCEM_GFN,
                #'mdict':        {},
                'n_players':    150,
                'pmex_init':    0.2,
                'pmex_trg':     0.05,
                'stats_iv':     10000,
                #'trainable':    False,
            } for ix in range(7)}
    #"""
    #for k in ['fm0','fm4','fm3','fm6','fm10','fm11','fm9']: dmk_dna.pop(k)
    dmk_dna.update({
        f'bm{ix}': {
                'family':       'B',
                'fwd_func':     cnnCEM_GFN,
                'mdict':        {'c_embW':18, 'n_lay':18},
                'n_players':    150,
                'pmex_init':    0.2,
                'pmex_trg':     0.05,
                'stats_iv':     10000,
                #'trainable':    False,
            } for ix in range(7)})
    #"""
    loopIX = 0
    while True:
        if loopIX:
            #break  # to break after first loop
            for dn in dmk_dna: dmk_dna[dn]['pmex_init'] = dmk_dna[dn]['pmex_trg']
        run(dmk_dna, 3)
        loopIX += 1
