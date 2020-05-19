"""

 2020 (c) piteren

"""

from ptools.neuralmess.dev_manager import nestarter
from ptools.mpython.mpdecor import proc_wait
from ptools.lipytools.decorators import timing

from podecide.games_manager import GamesManager
from podecide.decision_maker import NeurDMK
from podecide.decision_neural_graph import cnnCEM_GFN



@timing
@proc_wait
def run_GM_training(ddna, gx_limit=10):
    gm = GamesManager(
        dmk_dna=        ddna,
        #acc_won_iv=     (10000,20000),
        verb=           1)
    gm.run_games(
        #gx_loop_sh=     False,
        #gx_exit_sh=     False,
        gx_limit=       gx_limit)


def start_big_games():
    dmk_dna = {
        f'am{ix}': (NeurDMK, {
                'family':       'A',
                'fwd_func':     cnnCEM_GFN,
                'mdict':        {'n_lay':12},
                'n_players':    150,
                'pmex_init':    0.2,
                'pmex_trg':     0.05,
                #'stats_iv':     1000,
                #'trainable':    False,
            }) for ix in range(5)}
    #for k in ['fm0','fm4','fm3','fm6','fm10','fm11','fm9']: dmk_dna.pop(k)
    dmk_dna.update({
        f'bm{ix}': (NeurDMK, {
                'family':       'B',
                'fwd_func':     cnnCEM_GFN,
                'mdict':        {'n_lay':22},
                'n_players':    150,
                'pmex_init':    0.2,
                'pmex_trg':     0.05,
                #'stats_iv':     1000,
                #'trainable':    False,
            }) for ix in range(5)})
    dmk_dna.update({
        f'cm{ix}': (NeurDMK, {
                'family':       'C',
                'fwd_func':     cnnCEM_GFN,
                'mdict':        {'n_lay':36},
                'n_players':    150,
                'pmex_init':    0.2,
                'pmex_trg':     0.05,
                #'stats_iv':     1000,
                #'trainable':    False,
            }) for ix in range(5)})
    loopIX = 0
    while True:
        if loopIX:
            #break  # to break after first loop
            for dn in dmk_dna: dmk_dna[dn][1]['pmex_init'] = dmk_dna[dn][1]['pmex_trg']
        run_GM_training(dmk_dna)
        loopIX += 1


if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_training', silent_error=True)

    start_big_games()
