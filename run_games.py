"""

 2019 (c) piteren

 TODO:
  - player model:
    - predicts next move
    - predicts player cards
    - models player stats
  - player stats puts as state (input), if player model do not works ok

"""

import tensorflow as tf
import time

from decision_maker import BaNeDMK
from dmk_manager import DMKManager
from neural_graphs import lstm_GFN, cnn_GFN

from putils.neuralmess.dev_manager import nestarter


if __name__ == "__main__":

    nestarter('_log', custom_name='pyp')

    mdictA = {
        'name':     'cA_%s' % time.strftime('%m.%d_%H.%M'),
        'n_lay':    24,
        'width':    512}

    dmkL = [BaNeDMK(fwd_func=cnn_GFN, mdict=mdictA, n_players=15)]
    dmkm = DMKManager(
        dmkL=    dmkL,
        pmsg=    True,#False,
        verb=    0)
    dmkm.run_games()
