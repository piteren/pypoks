"""

 2020 (c) piteren

"""

import os
import sys
working_folder = os.getcwd()
sys.path.insert(0, working_folder) # working folder

from copy import deepcopy

from ptools.neuralmess.dev_manager import nestarter
from ptools.mpython.mpdecor import proc_wait
from ptools.lipytools.decorators import timing

from podecide.games_manager import GamesManager
from podecide.dmk import NeurDMK
from podecide.dmk_graph import cnn_DMG


@timing
@proc_wait
def run_GM_training(
        ddna,
        use_pretrained_cn=  False):

    gm = GamesManager(
        dmk_dna=            ddna,
        use_pretrained_cn=  use_pretrained_cn)
    gm.run_gx_games()


def start_big_games(n_reloads=5):

    print('Starting Big Games...')
    dmks_spec = {
        'a': {
            'mdict': {'n_lay': 12},
            'dmk_type':     NeurDMK,
            'fwd_func':     cnn_DMG,
            'n_players':    150,
            'pmex_init':    0.2,
            'pmex_trg':     0.05}}

    dmk_dna = {}
    for fm in dmks_spec:
        for ix in range(10):
            dmk_name = f'{fm}m{ix}'
            dmk_dict = {'family': fm}
            dmk_dict.update(deepcopy(dmks_spec[fm]))
            dmk_dna[dmk_name] = dmk_dict

    loopIX = 0
    while loopIX < n_reloads:

        if loopIX==1:
            for dn in dmk_dna:
                dmk_dna[dn]['pmex_init'] = dmk_dna[dn]['pmex_trg']

        run_GM_training(
            dmk_dna,
            use_pretrained_cn=  loopIX==0)
        loopIX += 1
    print('Big Games finished!')


if __name__ == "__main__":

    nestarter('_log', custom_name='big_games_training', silent_error=True)
    start_big_games()