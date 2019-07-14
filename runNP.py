"""

 2019 (c) piteren


 http://spinningup.openai.com/en/latest/
 http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
 https://stackoverflow.com/questions/51944199/calculating-loss-from-action-and-reward-in-tensorflow


 the simplest neural model:

 receives States in form of constant size vector
 for every state outputs Decision, every Decision is rewarded with cash

 Decision types:
 - idle
 - C/F
 - CLL
 - B/R # bet size is defined by algorithm (by now so simple)
 - ALL

 States:

 - (hand starts)
        myPosition(iR)              # 0,1,2
        myLastWon                   # -500:1500
        player1lastWon              # -500:1500
        player2lastWon              # -500:1500
 - (player decision)
        tableCash(R)                # 0:1500
        tableCards(LE)              # [E,E,E,E,E] # encodes table state
        playerID(iR)                # 0,1
        playerDecisionType(iR)      # 0,1,2,3
        playerDecisionCash(R)       # 0:500

"""

from pLogic.pPlayer import PPlayer
from pLogic.pTable import PTable
from decisionMaker import DecisionMaker


if __name__ == "__main__":

    print()
    pTable = PTable()
    dMK = DecisionMaker()
    aiPlayer = PPlayer(
        name=   'pl0',
        dMK=    dMK)
    pTable.addPlayer(aiPlayer)
    for ix in range(1, pTable.maxPlayers): pTable.addPlayer(PPlayer('pl%d'%ix))
    for _ in range(50):
        pTable.runHand()
        #for event in pTable.hands[-1]: print(event)