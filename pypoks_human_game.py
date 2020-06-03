"""

 2020 (c) piteren

"""

import os
import sys
working_folder = os.getcwd()
sys.path.insert(0, working_folder) # working folder

from ptools.neuralmess.dev_manager import nestarter

from pologic.poenvy import N_TABLE_PLAYERS
from podecide.games_manager import GamesManager
from podecide.dmk import NeurDMK, HDMK
from podecide.dmk_graph import cnn_DMG

from gui.gui_hdmk import GUI_HDMK


if __name__ == "__main__":

    model_name = 'am0'

    nestarter('_log', custom_name='human_game', silent_error=True)

    tk_gui = GUI_HDMK(N_TABLE_PLAYERS)

    dmk_dna = {
        model_name: {
                'dmk_type':     NeurDMK,
                'fwd_func':     cnn_DMG,
                'n_players':    2,
                'pmex_init':    0,
                'pmex_trg':     0,
                'stats_iv':     10,
                'trainable':    False},
        'hm0': {
                'dmk_type':     HDMK,
                'tk_gui':       tk_gui,
                'stats_iv':     10}}

    gm = GamesManager(
        dmk_dna=            dmk_dna,
        use_pretrained_cn=  False,
        verb=               1)

    gm.start_games()
    tk_gui.run_tk()
    gm.kill_games()
