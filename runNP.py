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
  ** tf process runs from another thread and communicates through batch-queue

"""

import tensorflow as tf
import time
import os

from decisionMaker import DMK, BNDMK, nGraphFN
from dmkManager import DMKManager


if __name__ == "__main__":

    # tf verbosity
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    session = tf.Session()

    gfdl = []
    gfdl.append(nGraphFN(scope='dmA_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3]))
    gfdl.append(nGraphFN(scope='dmB_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3]))
    gfdl.append(nGraphFN(scope='dmC_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3]))
    gfdl.append(nGraphFN(scope='dmD_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3]))
    gfdl.append(nGraphFN(scope='dmE_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], optAda=False))
    gfdl.append(nGraphFN(scope='dmF_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], optAda=False))
    gfdl.append(nGraphFN(scope='dmG_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], optAda=False))
    gfdl.append(nGraphFN(scope='dmH_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], optAda=False))

    dMKs = [BNDMK(session=session, gFND=gfd, nPl=150) for gfd in gfdl]
    #dMKs.append(DMK(name='dmk', nPl=300))
    dmkMan = DMKManager(
        dMKs=       dMKs,
        pMsg=       False,
        verbLev=    0)
    dmkMan.runGames()
