"""

 2019 (c) piteren

 receives States in form of constant size vector
 for every state outputs Decision, every Decision is rewarded with cash

 first implementation: training time: 10.7sec/1KH


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
from decisionMaker import BNdmk


if __name__ == "__main__":

    # tf verbosity
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    session = tf.Session()

    pTable = PTable(
        pMsg=       False,
        verbLev=    0
    )
    dMKa = BNdmk(session, 'dMKa_%s'% time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3])
    #dMKb = BNdmk(session, 'dMKb_%s'% time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3])
    pTable.addDMK(dMKa)
    #pTable.addDMK(dMKb)

    for _ in range(1):
        print()
        n = 0
        while n < 500000:
            n += 1
            pTable.runHand()
            if n % 1000 == 0:
                print(dMKa.sts['$'][0], n)
                #print(dMKa.sts['$'][0], dMKb.sts['$'][0], n)
        #dMKa.resetME(newName='dMKa_%s'% time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3])
