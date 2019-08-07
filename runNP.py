"""

 2019 (c) piteren

 receives States in form of constant size vector
 for every state outputs Decision, every Decision is rewarded with cash

 first implementation: training time: 10.7sec/1KH = 93.4H/s


 TODO:
  - better net arch and updates
  - cards network with embeddings
  - player model:
    - predicts next move
    - predicts player cards
    - models player stats
  - player stats puts as state (input), if player model do not works ok

"""

import tensorflow as tf
import time
import os

from decisionMaker import DMK, BNDMK, nGraphFN
from dmkManager import DMKManager


if __name__ == "__main__":

    # tf verbosity
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    session = tf.Session()

    gfdl = [
        nGraphFN(scope='dmA_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], lR=1e-4),
        nGraphFN(scope='dmB_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], lR=5e-5),
        nGraphFN(scope='dmC_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], lR=1e-5),
        nGraphFN(scope='dmD_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], optAda=False, lR=1e-1),
        nGraphFN(scope='dmE_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], optAda=False, lR=5e-2),
        #nGraphFN(scope='dmF_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], optAda=False, lR=1e-2),
    ]

    dMKs = [BNDMK(session=session, gFND=gfd, nPl=150) for gfd in gfdl]
    dmkMan = DMKManager(
        dMKs=       dMKs,
        pMsg=       False,
        verbLev=    0)
    dmkMan.runGames()
