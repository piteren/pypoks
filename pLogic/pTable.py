"""

 2019 (c) piteren

 simplified poker hand algorithm because:
 - no ante
 - constant sb, bb
 - 3 players
 - constant / simplified betting sizes
 - every player starts hand with startCash (no need for advanced win cash distribution)

 every played hand generates its history
 hand history is a list of events

"""

from pLogic.pPlayer import PPlayer, PLR_MVS
from pLogic.pDeck import PDeck

# table states
TBL_STT = {
    0:  'idle',
    1:  'preflop',
    2:  'flop',
    3:  'turn',
    4:  'river',
    5:  'handFin'}

# returns list of table position names in dependency of num of table players
def posNames(nP=3):

    pNames = ['SB','BB','BTN']
    if nP == 2: pNames = pNames[:-1]
    if nP == 6: pNames = ['SB','BB','UTG','MP','CT','BTN']
    if nP == 9: pNames = ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']
    return pNames


class PTable:

    def __init__(self, name='pTable'):

        self.verbLev = 1
        self.pMsg = True

        self.name = name
        self.maxPlayers = 3
        self.SB = 2
        self.BB = 5
        self.startCash = 500

        self.players = [] # list of table players, order: SB, BB ...
        self.deck = PDeck()
        self.state = 0
        self.cash = 0 # cash on table
        self.cards = [] # table cards (max 5)
        self.cashToCall = 0 # how much player has to put to call (on current river)

        self.handID = -1 # int incremented every hand
        self.hands = [] # list of histories of played hands

        if self.verbLev: print('(table)%s created' % self.name)

    # puts player on table (self)
    def addPlayer(
            self,
            pPlayer: PPlayer):

        pPlayer.table = self
        pPlayer.cash = self.startCash
        self.players.append(pPlayer)
        if self.verbLev: print('(player)%s joined (table)%s' % (pPlayer.name, self.name))

    # runs single hand
    def runHand(self):

        self.handID += 1
        self.state = 1
        if self.verbLev: print('\n(table)%s starts new hand, handID %d' % (self.name, self.handID))

        handPlayers = [] + self.players # original order of players for current hand (SB, BB, ..)
        if self.pMsg:
            print(' ### (table)%s hand players:' % self.name)
            for player in handPlayers: print(' ### (player)%s'%player.name)
        hHis = [{'handPlayers': [player.name for player in handPlayers]}]
        self.hands.append(hHis)

        # put blinds on table
        handPlayers[0].cash -= self.SB
        handPlayers[0].cHandCash = self.SB
        handPlayers[0].cRiverCash = self.SB
        self.cash += self.SB
        if self.pMsg: print(' ### (player)%s put SB %d$'%(handPlayers[0].name, self.SB))
        handPlayers[1].cash -= self.BB
        handPlayers[1].cHandCash = self.BB
        handPlayers[1].cRiverCash = self.BB
        self.cash += self.BB
        if self.pMsg: print(' ### (player)%s put BB %d$' % (handPlayers[1].name, self.BB))
        self.cashToCall = self.BB
        # by now blinds info is not needed for hand history

        clcIX = 1 # current loop closing player index
        cmpIX = 2 # currently moving player index (for 2 players not valid but validated at first river loop)

        # hand cards
        for player in handPlayers:
            ca, cb = self.deck.getCard(), self.deck.getCard()
            player.hand = ca, cb
            if self.pMsg: print(' ### (player)%s taken hand %s %s' % (player.name, PDeck.cardToStr(ca), PDeck.cardToStr(cb)))
        hHis.append({'playersCards': [player.hand for player in handPlayers]})

        while self.state < 5 and len(handPlayers) > 1:

            if self.pMsg: print(' ### (table)%s state %s' % (self.name, TBL_STT[self.state]))

            # manage table cards
            newTableCards = []
            if self.state == 2: newTableCards = [self.deck.getCard(), self.deck.getCard(), self.deck.getCard()]
            if self.state in [3,4]: newTableCards = [self.deck.getCard()]
            if newTableCards:
                self.cards += newTableCards
                hHis.append({'newTableCards': newTableCards})
                if self.pMsg:
                    print(' ### (table)%s cards: '%self.name, end='')
                    for card in self.cards: print(PDeck.cardToStr(card), end=' ')
                    print()

            # ask players for moves
            while len(handPlayers) > 1: # more important condition breaks below

                if cmpIX == len(handPlayers): cmpIX = 0 # next loop

                playerFolded = False
                playerRaised = False
                player = handPlayers[cmpIX]
                if player.cash: # player has cash (not allined yet)

                    playerMove = handPlayers[cmpIX].makeMove() # player makes move

                    moveData = {
                        'tState':       self.state,
                        'tBCash':       self.cash,
                        'pName':        player.name,
                        'pBCash':       player.cash,
                        'pBCHandCash':  player.cHandCash,
                        'pBCRiverCash': player.cRiverCash,
                        'bCashToCall':  self.cashToCall,
                        'plMove':       playerMove}

                    player.cash -= playerMove[1]
                    player.cHandCash += playerMove[1]
                    player.cRiverCash += playerMove[1]
                    self.cash += playerMove[1]

                    moveData['tACash'] =        self.cash
                    moveData['pACash'] =        player.cash
                    moveData['pACHandCash'] =   player.cHandCash
                    moveData['pACRiverCash'] =  player.cRiverCash

                    if playerMove[0] > 1:
                        playerRaised = True
                        self.cashToCall = handPlayers[cmpIX].cRiverCash
                        clcIX = cmpIX - 1 if cmpIX > 0 else len(handPlayers) - 1 # player before in loop
                    moveData['aCashToCall'] = self.cashToCall

                    if self.pMsg: print(' ### (player)%s had %4d$, moved %s with %4d$, now: tableCash %4d$ toCall %4d$' %(player.name, moveData['pBCash'], PLR_MVS[playerMove[0]], playerMove[1], self.cash, self.cashToCall))

                    if playerMove[0] == 0 and self.cashToCall > player.cRiverCash:
                        playerFolded = True
                        del(handPlayers[cmpIX])

                    hHis.append({'moveData': moveData})

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

        # winners template
        winnersData = []
        for player in self.players:
            winnersData.append({
                'plName':       player.name,
                'winner':       False,
                'fullRank':     None,
                'simpleRank':   0,
                'won':          0})

        if len(handPlayers) == 1:
            winIX = self.players.index(handPlayers[0])
            winnersData[winIX]['winner'] = True
            winnersData[winIX]['fullRank'] = 'muck'
            nWinners = 1
        else:
            topRank = 0
            for player in handPlayers:
                cards = list(player.hand)+self.cards
                rank = PDeck.cardsRank(cards)
                plIX = self.players.index(player)
                if topRank < rank[1]:
                    topRank = rank[1]
                    winnersData[plIX]['fullRank'] = rank
                else:
                    winnersData[plIX]['fullRank'] = 'muck'
                winnersData[plIX]['simpleRank'] = rank[1]
            nWinners = 0
            for data in winnersData:
                if data['simpleRank'] == topRank:
                    data['winner'] = True
                    nWinners += 1

        # manage cash and information about
        prize = self.cash // nWinners
        for ix in range(len(self.players)):
            player = self.players[ix]
            myWon = -player.cHandCash  # netto lost
            if winnersData[ix]['winner']: myWon += prize # add netto winning
            player.wonLast = myWon
            player.wonTotal += myWon
            winnersData[ix]['won'] = myWon
            player.getReward(myWon)
            if self.pMsg:
                print(' ### (player)%s : %d$' %(player.name, myWon), end='')
                if winnersData[ix]['fullRank']:
                    if type(winnersData[ix]['fullRank']) is str: print(', mucked hand', end ='')
                    else: print(' with %s'%winnersData[ix]['fullRank'][-1], end='')
                print()

            # reset player data (cash, cards)
            player.hand = None  # players return cards
            player.cash = self.startCash
            player.cHandCash = 0
            player.cRiverCash = 0

        hHis.append({'winnersData': winnersData})

        # table reset
        self.cash = 0
        self.cards = []
        self.deck.resetDeck()
        self.state = 0
        self.players.append(self.players.pop(0)) # rotate table players for next hand

        if self.verbLev: print('(table)%s hand finished, table state %s' % (self.name, TBL_STT[self.state]))
