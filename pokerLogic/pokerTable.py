"""

 2019 (c) piteren

"""


from pokerLogic.pokerPlayer import PokerPlayer, PLR_MVS
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
        self.lastMoveID = None
        self.cashToCall = 0

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

        # put blinds on table
        handPlayers[0].cash -= self.SB
        handPlayers[0].myLastMoveCash = self.SB
        self.cash += self.SB
        if self.pMsg: print(' ### (player)%s put SB %d$'%(handPlayers[0].name, self.SB))
        handPlayers[1].cash -= self.BB
        handPlayers[1].myLastMoveCash = self.BB
        self.cash += self.BB
        if self.pMsg: print(' ### (player)%s put BB %d$' % (handPlayers[1].name, self.BB))
        self.cashToCall = self.BB
        moveLoopIX = 1 # index of player that closes the loop of current river
        cmpIX = 2 # index of currently moving player

        for player in handPlayers: player.takeHand(self.deck.getCard(), self.deck.getCard()) # give cards for players

        while self.state < 5 and len(handPlayers) > 1:
            self.state += 1
            if self.verbLev: print('(table)%s currently @state %s' % (self.name, TBL_STT[self.state]))

            # ask players for moves
            while len(handPlayers) > 1:

                if cmpIX == len(handPlayers): cmpIX = 0 # next loop

                print('  >> %d'%cmpIX, end=' ')
                for pl in handPlayers: print(pl.name, end=' ')
                print(moveLoopIX)

                playerFolded = False
                playerRaised = False
                # player has cash (not allined yet)
                if handPlayers[cmpIX].cash:
                    playerMove = handPlayers[cmpIX].makeMove()
                    if playerMove[0] > 1: self.cashToCall = handPlayers[cmpIX].myLastMoveCash
                    self.cash += playerMove[1]
                    if self.pMsg:
                        print(' ### (player)%s moved %s with %d$' %(handPlayers[cmpIX].name, PLR_MVS[playerMove[0]], playerMove[1]))
                        print(' ### (table)%s cash %d toCall %d' %(self.name, self.cash, self.cashToCall))
                    # raised, so he closes now
                    if playerMove[0] > 1:
                        moveLoopIX = cmpIX - 1
                        if moveLoopIX < 0: moveLoopIX = len(handPlayers) -1
                        playerRaised = True
                    if playerMove[0] == 0 and self.cashToCall - handPlayers[cmpIX].myLastMoveCash > 0:
                        del(handPlayers[cmpIX])
                        playerFolded = True

                if moveLoopIX == cmpIX and not playerRaised: break

                if not playerFolded: cmpIX += 1
                elif moveLoopIX > cmpIX: moveLoopIX -= 1

            # reset for next river
            moveLoopIX = len(handPlayers)-1
            cmpIX = 0
            self.cashToCall = 0
            for pl in self.players: pl.myLastMoveCash = 0

        # winners
        if len(handPlayers) == 1:
            if self.pMsg:
                handPlayers[0].cash += self.cash
                print(' ### (player)%s won %d$' %(handPlayers[0].name, self.cash))
        else:
            # choose top hand
            cardsRanks = [PokerDeck.cardsRank(cards)[0] for cards in [pl.hand for pl in handPlayers]]
            topRank = max(cardsRanks)
            idWinners = [ix for ix in range(len(handPlayers)) if cardsRanks[ix] == topRank]
            prize = self.cash // len(idWinners)
            for id in idWinners:
                print('### (player)%s won %d$ with rank %d' %(handPlayers[id].name, prize, topRank))

        self.state = 0
        for player in handPlayers: player.hand = None # players return cards
        self.cash = 0
        self.players.append(self.players.pop(0)) # circle table players

        # fill players cash up to table startCash
        for player in self.players:
            if player.cash < self.startCash:
                player.upFilledCash += self.startCash - player.cash
                player.cash = self.startCash

        if self.verbLev: print('(table)%s hand finished, table state %s' % (self.name, TBL_STT[self.state]))


if __name__ == "__main__":

    print()
    pTable = PokerTable()
    for ix in range(pTable.maxPlayers): pTable.addPlayer(PokerPlayer('pl%d'%ix))
    pTable.runHand()
