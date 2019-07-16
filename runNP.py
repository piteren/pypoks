"""

 2019 (c) piteren

 http://karpathy.github.io/2016/05/31/rl/
 http://spinningup.openai.com/en/latest/
 http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
 https://stackoverflow.com/questions/51944199/calculating-loss-from-action-and-reward-in-tensorflow


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
  - player stats

"""

from pLogic.pPlayer import PPlayer
from pLogic.pTable import PTable
from decisionMaker import DecisionMaker


if __name__ == "__main__":

    for _ in range(100):
        print()
        pTable = PTable(pMsg=False, verbLev=0)
        dMK = DecisionMaker()
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
