"""

 2019 (c) piteren

"""

# TODO:
#  - finalize makeMove

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
        self.table = None
        self.hand = None
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
        # TODO:
        #  - check for possible moves - player should know its moves
        #  - make move
        pass