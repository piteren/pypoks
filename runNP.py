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

from pokerLogic.pokerPlayer import PokerPlayer
from pokerLogic.pokerTable import PokerTable


if __name__ == "__main__":

    print()
    pTable = PokerTable()
    for ix in range(pTable.maxPlayers): pTable.addPlayer(PokerPlayer('pl%d'%ix))
    for _ in range(5): pTable.runHand()