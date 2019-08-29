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

from decisionMaker import BNDMK
from dmkManager import DMKManager
from neuralGraphs import lstmGraphFN, cnnRGraphFN

from pUtils.nnTools.nnBaseElements import loggingSet


if __name__ == "__main__":

    loggingSet('_log', customName='pyp', forceLast=True)

    gfdl = [
       #lstmGraphFN(scope='lA_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3],                  optAda=False, lR=7e-4),
        cnnRGraphFN(scope='cB_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], nLay=24,reW=512, optAda=False, lR=3e-4),
       #cnnRGraphFN(scope='cC_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], nLay=12,reW=256,               lR=7e-7),
        cnnRGraphFN(scope='cD_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], nLay=8, reW=768, optAda=False, lR=1e-4),
       #cnnRGraphFN(scope='cE_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], nLay=8, reW=512,               lR=7e-7),
        cnnRGraphFN(scope='cF_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], nLay=4, reW=1024,optAda=False, lR=7e-5),
       #cnnRGraphFN(scope='cG_%s' % time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], nLay=4, reW=512,               lR=7e-7),
    ]

    dMKs = [BNDMK(session=tf.Session(), gFND=gfd, nPl=120) for gfd in gfdl]
    dmkMan = DMKManager(
        dMKs=       dMKs,
        pMsg=       False,
        verbLev=    0)
    dmkMan.runGames()
