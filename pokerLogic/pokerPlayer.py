"""

 2019 (c) piteren

"""

# TODO:
#  - finalize makeMove

import random
from pokerLogic.pokerDeck import PokerDeck

# player possible moves
PLR_MVS = {
    0:      'check/fold',
    1:      'call',
    2:      'bet/raise',
    3:      'allin'}

class PokerPlayer:

    def __init__(self, name='pPlayer'):

        self.verbLev = 1
        self.pMsg = True
        self.name = name
        self.cash = 500
        self.upFilledCash = 0
        self.table = None
        self.hand = None
        self.myLastMoveCash = 0
        if self.verbLev: print('(player)%s created' % self.name)

    # player takes hand
    def takeHand(self, ca, cb):
        self.hand = ca, cb
        if self.pMsg: print(' ### (player)%s taken hand %s %s' %(self.name, PokerDeck.cardToStr(ca), PokerDeck.cardToStr(cb)))

    # player returns hand
    def rtrnHand(self): self.hand = None

    # makes player decision (having table status ...and any other info)
    def makeMove(
            self):

        possibleMoves = {x: True for x in range(4)}
        if self.table.cashToCall - self.myLastMoveCash == 0: possibleMoves[1] = False # cannot call (already called)
        if self.cash < 2*self.table.cashToCall: possibleMoves[2] = False # cannot bet/raise
        if self.cash == self.table.cashToCall - self.myLastMoveCash: possibleMoves[1] = False # cannot call (just allin)
        possibleMovesCash = {
            0:  0,
            1:  self.table.cashToCall - self.myLastMoveCash,
            2:  2*self.table.cashToCall - self.myLastMoveCash if self.table.cashToCall else self.table.cash // 2, # by now simple
            3:  self.cash}

        # random move
        selectedMove = [key for key in possibleMoves.keys() if possibleMoves[key]]
        random.shuffle(selectedMove)
        selectedMove = selectedMove[0]
        self.cash -= possibleMovesCash[selectedMove]
        self.myLastMoveCash += possibleMovesCash[selectedMove]

        return selectedMove, possibleMovesCash[selectedMove]