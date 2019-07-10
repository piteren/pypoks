"""

 2019 (c) piteren

"""

import random
from pokerLogic.pokerDeck import PokerDeck

# player possible moves
PLR_MVS = {
    0:      'C/F',
    1:      'CLL',
    2:      'B/R',
    3:      'ALL'}


class PokerPlayer:

    def __init__(self, name='pPlayer'):

        self.verbLev = 1
        self.pMsg = True
        self.name = name
        self.cash = 500
        self.wonLast = 0
        self.wonTotal = 0
        self.table = None
        self.hand = None
        self.cRiverCash = 0 # current river cash (amount put by player on current river yet)
        if self.verbLev: print('(player)%s created' % self.name)

    # player takes hand
    def takeHand(self, ca, cb):
        self.hand = ca, cb
        if self.pMsg: print(' ### (player)%s taken hand %s %s' %(self.name, PokerDeck.cardToStr(ca), PokerDeck.cardToStr(cb)))

    # player returns hand
    def rtrnHand(self): self.hand = None

    # makes player decision (having table status ...and any other info)
    def makeMove(self):

        possibleMoves = {x: True for x in range(4)}
        if self.table.cashToCall - self.cRiverCash == 0: possibleMoves[1] = False # cannot call (already called)
        if self.cash < 2*self.table.cashToCall: possibleMoves[2] = False # cannot bet/raise
        if self.cash == self.table.cashToCall - self.cRiverCash: possibleMoves[1] = False # cannot call (just allin)
        possibleMovesCash = {
            0:  0,
            1:  self.table.cashToCall - self.cRiverCash,
            2:  2*self.table.cashToCall - self.cRiverCash if self.table.cashToCall else self.table.cash // 2, # by now simple
            3:  self.cash}

        # random move
        selectedMove = [key for key in possibleMoves.keys() if possibleMoves[key]]
        random.shuffle(selectedMove)
        selectedMove = selectedMove[0]
        self.cash -= possibleMovesCash[selectedMove]
        self.cRiverCash += possibleMovesCash[selectedMove]

        return selectedMove, possibleMovesCash[selectedMove]
