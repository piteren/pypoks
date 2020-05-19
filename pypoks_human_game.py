"""

 2020 (c) piteren

"""

from ptools.neuralmess.dev_manager import nestarter
from ptools.mpython.mpdecor import proc

from pologic.poenvy import N_TABLE_PLAYERS
from podecide.games_manager import GamesManager
from podecide.decision_maker import NeurDMK, HDMK
from podecide.decision_neural_graph import cnnCEM_GFN

from gui.gui_hdmk import GUI_HDMK


# function running human game in separate process that communicates with GUI via HDMK ques
@proc
def run_human_eval(tk_gui :GUI_HDMK, model_name):

    dmk_dna = {
        model_name: (NeurDMK, {
                'fwd_func':     cnnCEM_GFN,
                'n_players':    2,
                'pmex_init':    0,
                'pmex_trg':     0,
                'stats_iv':     10,
                'trainable':    False}),
        'hm0': (HDMK, {
                'tk_gui':       tk_gui,
                'stats_iv':     10})}

    gm = GamesManager(
        dmk_dna=        dmk_dna,
        verb=           1)
    gm.run_games(
        gx_loop_sh=     False,
        gx_exit_sh=     False,
        gx_limit=       None)


if __name__ == "__main__":

    nestarter('_log', custom_name='human_game', silent_error=True)

    tk_gui = GUI_HDMK(N_TABLE_PLAYERS)
    run_human_eval(tk_gui, 'bm3')
    tk_gui.run_tk()

