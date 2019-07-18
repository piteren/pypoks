"""

 2019 (c) piteren


 the simplest neural model:

 receives States in form of constant size vector
 for every state outputs Decision, every Decision is rewarded with cash


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
import os

from pLogic.pTable import PTable
from decisionMaker import BNdmk


if __name__ == "__main__":

    # tf verbosity
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    session = tf.Session()

    pTable = PTable(pMsg=False, verbLev=0)
    dMKa = BNdmk(session, 'dMKa')
    dMKb = BNdmk(session, 'dMKb')
    pTable.addDMK(dMKa)
    pTable.addDMK(dMKb)

    for _ in range(1):
        print()
        n = 0
        while n < 500000:
            n += 1
            pTable.runHand()
            if n % 1000 == 0: print(dMKa.accumRew, dMKb.accumRew, n)
        #dMK.resetME(newName=True)
