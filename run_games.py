"""

 2020 (c) piteren

"""

from ptools.neuralmess.dev_manager import nestarter
from ptools.mpython.mpdecor import proc_wait, proc
from ptools.lipytools.decorators import timing

from pologic.poenvy import N_TABLE_PLAYERS

from podecide.games_manager import GamesManager
from podecide.decision_maker import NeurDMK, HDMK
from podecide.neural.neural_graphs import cnnCEM_GFN

from gui.tkproc import TkProc


def run_human_eval(ddna):
    gm = GamesManager(
        dmk_dna=        ddna,
        acc_won_iv=     (100000,200000),
        verb=           1)
    gm.run_games(
        gx_loop_sh=     False,
        gx_exit_sh=     False,
        gx_limit=       None)

@timing
@proc_wait
def run_GM_training(ddna, gx_limit=10):
    gm = GamesManager(
        dmk_dna=        ddna,
        acc_won_iv=     (100000,200000),
        verb=           1)
    gm.run_games(
        #gx_loop_sh=     False,
        #gx_exit_sh=     False,
        gx_limit=       gx_limit)


def start_big_games():
    dmk_dna = {
        f'cm{ix}': (NeurDMK, {
                'family':       'A',
                'fwd_func':     cnnCEM_GFN,
                #'mdict':        {},
                'n_players':    150,
                'pmex_init':    0.2,
                'pmex_trg':     0.05,
                'stats_iv':     10000,
                #'trainable':    False,
            }) for ix in range(7)}
    #"""
    #for k in ['fm0','fm4','fm3','fm6','fm10','fm11','fm9']: dmk_dna.pop(k)
    dmk_dna.update({
        f'dm{ix}': (NeurDMK, {
                'family':       'B',
                'fwd_func':     cnnCEM_GFN,
                'mdict':        {'c_embW':18, 'n_lay':18},
                'n_players':    150,
                'pmex_init':    0.2,
                'pmex_trg':     0.05,
                'stats_iv':     10000,
                #'trainable':    False,
            }) for ix in range(7)})
    #"""
    loopIX = 0
    while True:
        if loopIX:
            #break  # to break after first loop
            for dn in dmk_dna: dmk_dna[dn][1]['pmex_init'] = dmk_dna[dn][1]['pmex_trg']
        run_GM_training(dmk_dna, 3)
        loopIX += 1


def start_human_game():

    @proc
    def runh(tk_proc :TkProc):
        dmk_dna = {
        'am0': (NeurDMK, {
                'family':       'A',
                'fwd_func':     cnnCEM_GFN,
                #'mdict':        {},
                'n_players':    2,
                'pmex_init':    0,
                'pmex_trg':     0,
                'stats_iv':     10,
                'trainable':    False}),
        'hm0': (HDMK, {
                'tk_proc':      tk_proc,
                'stats_iv':     10})}
        run_human_eval(dmk_dna)

    tk_proc = TkProc(N_TABLE_PLAYERS)
    runh(tk_proc)
    tk_proc.run_tk()


if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    #start_big_games()
    start_human_game()
