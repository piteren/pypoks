"""

 2019 (c) piteren

"""

# TODO:
#  - finalize runHand

from pokerLogic.pokerPlayer import PokerPlayer
from pokerLogic.pokerDeck import PokerDeck

# table states
TBL_STT = {
    0:  'idle',
    1:  'preflop',
    2:  'flop',
    3:  'turn',
    4:  'river',
    5:  'handFin'}

class PokerTable:

    def __init__(self, name='pTable'):

        self.verbLev = 1
        self.pMsg = True

        self.name = name
        self.maxPlayers = 3
        self.SB = 2
        self.BB = 5
        self.startCash = 500

        self.players = [] # list of table players, order: SB, BB ...
        self.deck = PokerDeck()
        self.state = 0
        self.cash = 0 # cash on table

        if self.verbLev: print('(table)%s created' % self.name)

    # puts player on self (table)
    def addPlayer(
            self,
            pPlayer: PokerPlayer):

        pPlayer.table = self
        pPlayer.cash = self.startCash
        self.players.append(pPlayer)
        if self.verbLev: print('(player)%s joined (table)%s' % (pPlayer.name, self.name))

    # runs single hand
    def runHand(self):

        if self.verbLev: print('(table)%s starts new hand' % self.name)
        self.deck.resetDeck()
        handPlayers = [] + self.players # original order of players for current hand (SB, BB, ..)
        movePlayers = [] + handPlayers

        # put blinds on table
        movePlayers[0].cash -= self.SB
        self.cash += self.SB
        if self.pMsg: print(' ### (player)%s put SB %d$'%(movePlayers[0].name, self.SB))
        movePlayers.append(movePlayers.pop(0))
        movePlayers[0].cash -= self.BB
        self.cash += self.BB
        if self.pMsg: print(' ### (player)%s put BB %d$' % (movePlayers[0].name, self.BB))
        movePlayers.append(movePlayers.pop(0))

        for player in handPlayers: player.takeHand(self.deck.getCard(), self.deck.getCard()) # give cards for players

        while self.state < 5 and len(handPlayers) > 1:
            self.state += 1
            if self.verbLev: print('(table)%s currently @state %s' % (self.name, TBL_STT[self.state]))
            # TODO: ask players for moves

        self.state = 0
        for player in handPlayers: player.hand = None # players return cards
        self.cash = 0
        self.players.append(self.players.pop(0)) # circle table players
        if self.verbLev: print('(table)%s hand finished, table state %s' % (self.name, TBL_STT[self.state]))


if __name__ == "__main__":

    print()
    pTable = PokerTable()
    for ix in range(pTable.maxPlayers): pTable.addPlayer(PokerPlayer('pl%d'%ix))
    pTable.runHand()
