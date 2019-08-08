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
import os

from decisionMaker import DMK, BNDMK
from dmkManager import DMKManager
from neuralGraphs import lstmGraphFN, cnnRGraphFN


if __name__ == "__main__":

    # tf verbosity
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    session = tf.Session()

    gfdl = [
        lstmGraphFN(scope='dmA_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3],lR=3e-5),
        cnnRGraphFN(scope='dmB_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3],nLay=24,reW=512,lR=1e-5),
        cnnRGraphFN(scope='dmC_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3],nLay=48,reW=256,lR=1e-5),
        cnnRGraphFN(scope='dmD_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3],nLay=12,reW=768,lR=1e-5),
        cnnRGraphFN(scope='dmE_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3],nLay=6,reW=1024,lR=1e-5),
        cnnRGraphFN(scope='dmF_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3],nLay=6,reW=256,lR=1e-5),
    ]

    dMKs = [BNDMK(session=session, gFND=gfd, nPl=150) for gfd in gfdl]
    dmkMan = DMKManager(
        dMKs=       dMKs,
        pMsg=       False,
        verbLev=    0)
    dmkMan.runGames()
