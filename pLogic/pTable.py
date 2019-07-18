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

import random

from pLogic.pDeck import PDeck
from decisionMaker import DecisionMaker

# table states
TBL_STT = {
    0:  'idle',
    1:  'preflop',
    2:  'flop',
    3:  'turn',
    4:  'river',
    5:  'handFin'}

# table moves
TBL_MOV = {
    0:  'C/F',
    1:  'CLL',
    2:  'B/R',  # bet size is defined by algorithm (by now so simple)
    3:  'ALL'}

# returns list of table position names in dependency of num of table players
def posNames(nP=3):

    pNames = ['SB','BB','BTN'] # default for 3
    if nP == 2: pNames = pNames[:-1]
    if nP == 6: pNames = ['SB','BB','UTG','MP','CT','BTN']
    if nP == 9: pNames = ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']
    return pNames


class PTable:

    # player is a bridge between table and DMK
    class PPlayer:

        def __init__(
                self,
                tID: int,  # table ID
                dMK=        None):  # decisionMaker

            self.tID = tID
            self.dMK = dMK if dMK else DecisionMaker()  # given or default DMK

            self.nextStateUpdate = 0  # number of hand state to_update_from next

            # managed ONLY by table
            self.table = None
            self.hand = None
            self.cash = 0
            self.wonLast = 0
            self.wonTotal = 0
            self.cHandCash = 0  # current hand cash (amount put by player on current hand)
            self.cRiverCash = 0  # current river cash (amount put by player on current river yet)

        # asks DMK for move decision (having table status ...and any other info)
        def makeMove(self):

            # calculate possible moves and cash based on table state and hand history
            possibleMoves = [True for x in range(4)]  # by now all
            if self.table.cashToCall - self.cRiverCash == 0: possibleMoves[1] = False  # cannot call (already called)
            if self.cash < 2 * self.table.cashToCall: possibleMoves[2] = False  # cannot bet/raise
            if self.cash == self.table.cashToCall - self.cRiverCash: possibleMoves[
                1] = False  # cannot call (just allin)
            # by now simple hardcoded amounts of cash
            possibleMovesCash = {
                0: 0,
                1: self.table.cashToCall - self.cRiverCash,
                2: 2 * self.table.cashToCall - self.cRiverCash if self.table.cashToCall else self.table.cash // 2,
                3: self.cash}

            currentHandH = self.table.hands[-1]
            stateChanges = currentHandH[self.nextStateUpdate:]
            self.nextStateUpdate = len(currentHandH)
            selectedMove = self.dMK.mDec(stateChanges, possibleMoves)

            return selectedMove, possibleMovesCash[selectedMove]

        # called by table to inform player about reward for last hand
        # reward is forwarded to DMK
        def getReward(
                self,
                reward: int):

            self.nextStateUpdate = 0
            self.dMK.getReward(reward)


    def __init__(
            self,
            name=       'pTable',
            nPlayers=   3,
            pMsg=       True,
            verbLev=    1):

        self.verbLev = verbLev
        self.pMsg = pMsg

        self.name = name

        self.nPlayers = nPlayers
        self.players = [self.PPlayer(ix, dMK=DecisionMaker('rnDMK_%d'%ix)) for ix in range(self.nPlayers)] # generic table players

        self.SB = 2
        self.BB = 5
        self.startCash = 500

        self.deck = PDeck()
        self.state = 0
        self.cash = 0 # cash on table
        self.cards = [] # table cards (max 5)
        self.cashToCall = 0 # how much player has to put to call (on current river)

        self.handID = -1 # int incremented every hand
        self.hands = [] # list of histories of played hands

        if self.verbLev: print('(table)%s created' % self.name)

    # inits players/DMK for new game
    def _initPlayers(self):

        for pl in self.players:
            pl.table = self
            pl.cash = self.startCash
            pl.dMK.start(self,pl)

    # puts DMK on table (self)
    def addDMK(
            self,
            dMK: DecisionMaker):

        added = False
        sPlayers = [] + self.players
        random.shuffle(sPlayers) # shuffled players for random placement
        for pl in sPlayers:
            if type(pl.dMK) is DecisionMaker:
                pl.dMK = dMK
                added = True
                break

        assert added, 'cannot add DMK!'
        if self.verbLev: print('(dmk)%s joined (table)%s' % (dMK.name, self.name))

    # runs single hand
    def runHand(self):

        if self.handID < 0: self._initPlayers()

        self.handID += 1
        self.state = 1
        if self.verbLev: print('\n(table)%s starts new hand, handID %d' % (self.name, self.handID))

        handPlayers = [] + self.players # original order of players for current hand (SB, BB, ..)
        if self.pMsg:
            print(' ### (table)%s hand players:' % self.name)
            pNames = posNames(self.nPlayers)
            for ix in range(self.nPlayers):
                pl = handPlayers[ix]
                print(' ### (pl)%d @ %s'%(pl.tID, pNames[ix]))

        # put blinds on table
        handPlayers[0].cash -= self.SB
        handPlayers[0].cHandCash = self.SB
        handPlayers[0].cRiverCash = self.SB
        self.cash += self.SB
        if self.pMsg: print(' ### (pl)%d put SB %d$'%(handPlayers[0].tID, self.SB))
        handPlayers[1].cash -= self.BB
        handPlayers[1].cHandCash = self.BB
        handPlayers[1].cRiverCash = self.BB
        self.cash += self.BB
        if self.pMsg: print(' ### (pl)%d put BB %d$' % (handPlayers[1].tID, self.BB))
        self.cashToCall = self.BB
        # by now blinds info is not needed for hand history

        clcIX = 1 # current loop closing player index
        cmpIX = 2 # currently moving player index (for 2 players not valid but validated at first river loop)

        # hand cards
        for pl in handPlayers:
            ca, cb = self.deck.getCard(), self.deck.getCard()
            pl.hand = ca, cb
            if self.pMsg: print(' ### (pl)%d taken hand %s %s' % (pl.tID, PDeck.cardToStr(ca), PDeck.cardToStr(cb)))
        hHis = [{'playersPC': [(pl.tID, pl.hand) for pl in handPlayers]}]
        self.hands.append(hHis)

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
                pl = handPlayers[cmpIX]
                if pl.cash: # player has cash (not allined yet)

                    plMV = handPlayers[cmpIX].makeMove() # player makes move

                    mvD = {
                        'tState':       self.state,
                        'tBCash':       self.cash,
                        'pltID':        pl.tID,
                        'pBCash':       pl.cash,
                        'pBCHandCash':  pl.cHandCash,
                        'pBCRiverCash': pl.cRiverCash,
                        'bCashToCall':  self.cashToCall,
                        'plMove':       plMV}

                    pl.cash -= plMV[1]
                    pl.cHandCash += plMV[1]
                    pl.cRiverCash += plMV[1]
                    self.cash += plMV[1]

                    mvD['tACash'] =        self.cash
                    mvD['pACash'] =        pl.cash
                    mvD['pACHandCash'] =   pl.cHandCash
                    mvD['pACRiverCash'] =  pl.cRiverCash

                    if plMV[0] > 1:
                        playerRaised = True
                        self.cashToCall = handPlayers[cmpIX].cRiverCash
                        clcIX = cmpIX - 1 if cmpIX > 0 else len(handPlayers) - 1 # player before in loop
                    mvD['aCashToCall'] = self.cashToCall

                    if self.pMsg: print(' ### (pl)%d had %4d$, moved %s with %4d$, now: tableCash %4d$ toCall %4d$' %(pl.tID, mvD['pBCash'], TBL_MOV[plMV[0]], plMV[1], self.cash, self.cashToCall))

                    if plMV[0] == 0 and self.cashToCall > pl.cRiverCash:
                        playerFolded = True
                        del(handPlayers[cmpIX])

                    hHis.append({'moveData': mvD})

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
        for pl in self.players:
            winnersData.append({
                'pltID':        pl.tID,
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
            for pl in handPlayers:
                cards = list(pl.hand)+self.cards
                rank = PDeck.cardsRank(cards)
                plIX = self.players.index(pl)
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
            pl = self.players[ix]
            myWon = -pl.cHandCash  # netto lost
            if winnersData[ix]['winner']: myWon += prize # add netto winning
            pl.wonLast = myWon
            pl.wonTotal += myWon
            winnersData[ix]['won'] = myWon
            if self.pMsg:
                print(' ### (pl)%d : %d$' %(pl.tID, myWon), end='')
                if winnersData[ix]['fullRank']:
                    if type(winnersData[ix]['fullRank']) is str: print(', mucked hand', end ='')
                    else: print(' with %s'%winnersData[ix]['fullRank'][-1], end='')
                print()
            pl.getReward(myWon)

            # reset player data (cash, cards)
            pl.hand = None  # players return cards
            pl.cash = self.startCash
            pl.cHandCash = 0
            pl.cRiverCash = 0

        hHis.append({'winnersData': winnersData})

        # table reset
        self.cash = 0
        self.cards = []
        self.deck.resetDeck()
        self.state = 0
        self.players.append(self.players.pop(0)) # rotate table players for next hand

        if self.verbLev: print('(table)%s hand finished, table state %s' % (self.name, TBL_STT[self.state]))
