"""

 2019 (c) piteren

 receives States in form of constant size vector
 for every state outputs Decision, every Decision is rewarded with cash

 first implementation: training time: 10.7sec/1KH = 93.4H/s


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

    gfdl = [
        cnn_GFN(scope='cA_%s' % time.strftime('%m.%d_%H.%M'), nLay=24, reW=512,  optAda=False, lR=3e-3),
        cnn_GFN(scope='cB_%s' % time.strftime('%m.%d_%H.%M'), nLay=24, reW=512,  optAda=False, lR=3e-3),
        #cnn_GFN(scope='cC_%s' % time.strftime('%m.%d_%H.%M'), nLay=8,  reW=768,  optAda=False, lR=1e-3),
        #cnn_GFN(scope='cD_%s' % time.strftime('%m.%d_%H.%M'), nLay=8,  reW=768,  optAda=False, lR=1e-3),
        #cnn_GFN(scope='cE_%s' % time.strftime('%m.%d_%H.%M'), nLay=4,  reW=1024, optAda=False, lR=7e-4),
        #cnn_GFN(scope='cF_%s' % time.strftime('%m.%d_%H.%M'), nLay=4,  reW=1024, optAda=False, lR=7e-4),
    ]

    dmkL = [BaNeDMK(session=tf.Session(), gfd=gfd, n_players=3) for gfd in gfdl]
    dmkm = DMKManager(
        dmkL=    dmkL,
        pmsg=    True,#False,
        verb=    0)
    dmkm.run_games()
