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
        self.cards = [] # table cards (max 5)
        self.cashToCall = 0 # how much player has to put to call (on current river)

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
        self.state = 1
        self.deck.resetDeck()
        handPlayers = [] + self.players # original order of players for current hand (SB, BB, ..)

        # put blinds on table
        handPlayers[0].cash -= self.SB
        handPlayers[0].cRiverCash = self.SB
        self.cash += self.SB
        if self.pMsg: print(' ### (player)%s put SB %d$'%(handPlayers[0].name, self.SB))
        handPlayers[1].cash -= self.BB
        handPlayers[1].cRiverCash = self.BB
        self.cash += self.BB
        if self.pMsg: print(' ### (player)%s put BB %d$' % (handPlayers[1].name, self.BB))

        clcIX = 1 # current loop closing player index
        cmpIX = 2 # currently moving player index (for 2 players not valid but validated at first river loop)
        self.cashToCall = self.BB

        for player in handPlayers: player.takeHand(self.deck.getCard(), self.deck.getCard()) # give cards for players

        while self.state < 5 and len(handPlayers) > 1:

            if self.state == 2: self.cards = [self.deck.getCard(), self.deck.getCard(), self.deck.getCard()]
            if self.state in [3,4]: self.cards += [self.deck.getCard()]
            if self.verbLev:
                print('(table)%s currently @state %s, table cards: ' % (self.name, TBL_STT[self.state]), end='')
                for card in self.cards: print(PokerDeck.cardToStr(card), end=' ')
                print()

            # ask players for moves
            while len(handPlayers) > 1: # more important condition breaks below

                if cmpIX == len(handPlayers): cmpIX = 0 # next loop

                print('  >> %d'%cmpIX, end=' ')
                for pl in handPlayers: print(pl.name, end=' ')
                print(clcIX)

                playerFolded = False
                playerRaised = False
                if handPlayers[cmpIX].cash: # player has cash (not allined yet)
                    hadCash = handPlayers[cmpIX].cash
                    playerMove = handPlayers[cmpIX].makeMove() # player makes move
                    self.cash += playerMove[1]

                    if playerMove[0] > 1:
                        playerRaised = True
                        self.cashToCall = handPlayers[cmpIX].cRiverCash
                        clcIX = cmpIX - 1 if cmpIX > 0 else len(handPlayers) - 1 # player before in loop

                    if self.pMsg: print(' ### (player)%s had %3d$, moved %s with %3d$, tableCash %4d toCall %3d' %(handPlayers[cmpIX].name, hadCash, PLR_MVS[playerMove[0]], playerMove[1], self.cash, self.cashToCall))

                    if playerMove[0] == 0 and self.cashToCall > handPlayers[cmpIX].cRiverCash:
                        playerFolded = True
                        del(handPlayers[cmpIX])

                if clcIX == cmpIX and not playerRaised: break # player closing loop made decision (without raise)

                if not playerFolded: cmpIX += 1
                elif clcIX > cmpIX: clcIX -= 1 # move index left because of del

            # reset for next river
            clcIX = len(handPlayers)-1
            cmpIX = 0
            self.cashToCall = 0
            for pl in self.players: pl.cRiverCash = 0
            self.state += 1  # move table to next state

        if self.verbLev: print('(table)%s rivers finished, time to select winners' % self.name)
        # winners
        if len(handPlayers) == 1:
            if self.pMsg:
                handPlayers[0].cash += self.cash
                print(' ### (player)%s won %d$' %(handPlayers[0].name, self.cash))
        else:
            # choose top hand
            playerCards = [list(pl.hand) for pl in handPlayers]
            playerCards = [cards+self.cards for cards in playerCards]
            cardsRanks = [PokerDeck.cardsRank(cards) for cards in playerCards]
            topRank = max(cardsRanks)
            idWinners = [ix for ix in range(len(handPlayers)) if cardsRanks[ix] == topRank]
            prize = self.cash // len(idWinners)
            for id in idWinners:
                print('### (player)%s won %d$ with rank %s' %(handPlayers[id].name, prize, cardsRanks[id]))
        self.cash = 0
        self.cards = []

        # fill players cash up to table startCash
        for player in self.players:
            if player.cash < self.startCash:
                player.upFilledCash += self.startCash - player.cash
                player.cash = self.startCash

        for player in handPlayers: player.hand = None # players return cards
        self.players.append(self.players.pop(0)) # circle table players
        self.state = 0

        if self.verbLev: print('(table)%s hand finished, table state %s' % (self.name, TBL_STT[self.state]))


if __name__ == "__main__":

    print()
    pTable = PokerTable()
    for ix in range(pTable.maxPlayers): pTable.addPlayer(PokerPlayer('pl%d'%ix))
    pTable.runHand()
