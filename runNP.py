"""

 2019 (c) piteren


 the simplest neural model:

 receives States in form of constant size vector
 for every state outputs Decision, every Decision is rewarded with cash

 Decision types:
  0 - C/F
  1 - CLL
  2 - B/R # bet size is defined by algorithm (by now so simple)
  3 - ALL

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

from pLogic.pPlayer import PPlayer
from pLogic.pTable import PTable
from decisionMaker import BNdmk


if __name__ == "__main__":

    # tf verbosity
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    for _ in range(100):
        print()
        pTable = PTable(pMsg=False, verbLev=0)
        dMK = BNdmk(name='nnBase')
        aiPlayer = PPlayer(
            name=   'pl0',
            dMK=    dMK)
        pTable.addPlayer(aiPlayer)
        for ix in range(1, pTable.maxPlayers): pTable.addPlayer(PPlayer('pl%d'%ix))

        n = 0
        while n < 100000:
            n += 1
            pTable.runHand()
            #print(aiPlayer.wonTotal)
            if n % 1000 == 0: print(aiPlayer.wonTotal, n)
