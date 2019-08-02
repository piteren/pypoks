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

from pLogic.pTable import PTable
from decisionMaker import DecisionMaker, BNdmk, SNdmk
from dmkManager import DMKManager


if __name__ == "__main__":

    # tf verbosity
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    session = tf.Session()

    dMKs = [SNdmk(session=session, name='dmk%d'%ix, nPl=900) for ix in range(1)]
    dmkMan = DMKManager(
        dMKs=       dMKs,
        pMsg=       False,
        verbLev=    1)
    dmkMan.runGames()
