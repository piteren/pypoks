"""

 2019 (c) piteren

"""

import copy
from multiprocessing import Process, Queue
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
    2:  'B/R',
    3:  'ALL'}

# returns list of table position names in dependency of num of table players
def posNames(nP=3):

    pNames = ['SB','BB','BTN'] # default for 3
    if nP == 2: pNames = pNames[:-1]
    if nP == 6: pNames = ['SB','BB','UTG','MP','CT','BTN']
    if nP == 9: pNames = ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']
    return pNames

# Poker Table is a single poker game environment
class PTable(Process):

    # Table Player is a bridge interface between poker table and DMK
    class PPlayer:

        def __init__(
                self,
                dMK :DecisionMaker):

            self.dMK = dMK
            self.dmkIX = dMK.getPIX()
            self.name = '%s_%d' % (self.dMK.name, self.dmkIX)  # player takes name after DMK

            self.dmkIQue = self.dMK.dmkIQue
            self.dmkMQue = self.dMK.dmkMQues[self.dmkIX]

            # TODO: init ...names
            self.pls = [] # names of all players @table, initialised with start(), self.name always first
            self.cash = 0 # player cash
            self.hand = None
            self.nhsIX = 0 # next hand state index to update from
            self.cHandCash = 0 # current hand cash (amount put by player on current hand)
            self.cRiverCash = 0 # current river cash (amount put by player on current river)

        # asks DMK for move decision (having table status ...and any other info)
        def makeMove(
                self,
                handH,          # hand history
                tblCash,        # table cash
                tblCashTC):     # table cash to call

            # calculate possible moves and cash based on table state and hand history
            possibleMoves = [True for x in range(4)]  # by now all
            if tblCashTC - self.cRiverCash == 0: possibleMoves[1] = False  # cannot call (already called)
            if self.cash < 2 * tblCashTC: possibleMoves[2] = False  # cannot bet/raise
            if self.cash == tblCashTC - self.cRiverCash: possibleMoves[1] = False  # cannot call (just allin)

            # by now simple hardcoded amounts of cash
            possibleMovesCash = {
                0: 0,
                1: tblCashTC - self.cRiverCash,
                2: 2 * tblCashTC - self.cRiverCash if tblCashTC else tblCash // 2,
                3: self.cash}

            stateChanges = copy.deepcopy(handH[self.nhsIX:]) # copy part of history
            self.nhsIX = len(handH) # update index for next

            # update table history with player history
            for state in stateChanges:
                key = list(state.keys())[0]

                if key == 'playersPC':
                    for el in state[key]:
                        el[0] = self.pls.index(el[0]) # replace names with indexes
                        if el[0]: el[1] = None # hide cards of not mine (just in case)

                if key == 'moveData' or key == 'winnersData':
                    state[key]['pIX'] = self.pls.index(state[key]['pName']) # insert player index
                    del(state[key]['pName']) # remove player name

            decState = self.dMK.encState(self.dmkIX, stateChanges)
            self.dmkIQue.put([self.dmkIX, decState, possibleMoves])
            selectedMove = self.dmkMQue.get()

            return selectedMove, possibleMovesCash[selectedMove]


    def __init__(
            self,
            dMKs: list,             # list of DMKs, their number defines table size
            name=       'pTable',
            pMsg=       True,
            verbLev=    1):

        super().__init__(target=self.rHProc)

        self.verbLev = verbLev
        self.pMsg = pMsg

        self.name = name
        self.nPlayers = len(dMKs)

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

        self.players = [self.PPlayer(dMK) for dMK in dMKs]

        for pl in self.players:

            # update player players names with self on 1st pos
            pls = [pl.name for pl in self.players]
            pls.remove(pl.name)
            pl.pls = [pl.name] + pls

        if self.verbLev: print('(table)%s created' % self.name)

    # runs hands in loop (for sep. process)
    def rHProc(self):
        while True: self.runHand()

    # runs single hand
    def runHand(self):

        # reset player data (cash, cards)
        for pl in self.players:
            pl.hand = None  # players return cards
            pl.cash = self.startCash
            pl.nhsIX = 0
            pl.cHandCash = 0
            pl.cRiverCash = 0

        # reset table data
        self.cash = 0
        self.cards = []
        self.deck.resetDeck()
        self.state = 0

        self.handID += 1
        self.state = 1
        if self.verbLev: print('\n(table)%s starts new hand, handID %d' % (self.name, self.handID))

        handPlayers = [] + self.players # original order of players for current hand (SB, BB, ..)
        if self.pMsg:
            print(' ### (table)%s hand players:' % self.name)
            pNames = posNames(self.nPlayers)
            for ix in range(self.nPlayers):
                pl = handPlayers[ix]
                print(' ### (pl)%s @ %s' % (pl.name, pNames[ix]))

        # put blinds on table
        handPlayers[0].cash -= self.SB
        handPlayers[0].cHandCash = self.SB
        handPlayers[0].cRiverCash = self.SB
        self.cash += self.SB
        if self.pMsg: print(' ### (pl)%s put SB %d$' % (handPlayers[0].name, self.SB))
        handPlayers[1].cash -= self.BB
        handPlayers[1].cHandCash = self.BB
        handPlayers[1].cRiverCash = self.BB
        self.cash += self.BB
        if self.pMsg: print(' ### (pl)%s put BB %d$' % (handPlayers[1].name, self.BB))
        self.cashToCall = self.BB
        # by now blinds info is not needed for hand history

        clcIX = 1 # current loop closing player index
        cmpIX = 2 # currently moving player index (for 2 players not valid but validated at first river loop)

        # hand cards
        for pl in handPlayers:
            ca, cb = self.deck.getCard(), self.deck.getCard()
            pl.hand = ca, cb
            if self.pMsg: print(' ### (pl)%s taken hand %s %s' % (pl.name, PDeck.cts(ca), PDeck.cts(cb)))
        hHis = [{'playersPC': [[pl.name, pl.hand] for pl in handPlayers]}]
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
                    for card in self.cards: print(PDeck.cts(card), end=' ')
                    print()

            # ask players for moves
            while len(handPlayers) > 1: # more important condition breaks below

                if cmpIX == len(handPlayers): cmpIX = 0 # next loop

                playerFolded = False
                playerRaised = False
                pl = handPlayers[cmpIX]
                if pl.cash: # player has cash (not allined yet)

                    # before move values
                    mvD = {
                        'tState':       self.state,
                        'tBCash':       self.cash,
                        'pName':        pl.name,
                        'pBCash':       pl.cash,
                        'pBCHandCash':  pl.cHandCash,
                        'pBCRiverCash': pl.cRiverCash,
                        'bCashToCall':  self.cashToCall}

                    # player makes move
                    plMV = handPlayers[cmpIX].makeMove(
                        handH=      self.hands[-1],
                        tblCash=    self.cash,
                        tblCashTC=  self.cashToCall)
                    mvD['plMove'] = plMV

                    pl.cash -= plMV[1]
                    pl.cHandCash += plMV[1]
                    pl.cRiverCash += plMV[1]
                    self.cash += plMV[1]

                    # cash after move
                    mvD['tACash'] =        self.cash
                    mvD['pACash'] =        pl.cash
                    mvD['pACHandCash'] =   pl.cHandCash
                    mvD['pACRiverCash'] =  pl.cRiverCash

                    if plMV[0] > 1:
                        playerRaised = True
                        self.cashToCall = handPlayers[cmpIX].cRiverCash
                        clcIX = cmpIX - 1 if cmpIX > 0 else len(handPlayers) - 1 # player before in loop
                    mvD['aCashToCall'] = self.cashToCall

                    if self.pMsg: print(' ### (pl)%s had %4d$, moved %s with %4d$, now: tableCash %4d$ toCall %4d$' % (pl.name, mvD['pBCash'], TBL_MOV[plMV[0]], plMV[1], self.cash, self.cashToCall))

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
                'pName':        pl.name,
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
            winnersData[ix]['won'] = myWon
            if self.pMsg:
                print(' ### (pl)%s : %d$' % (pl.name, myWon), end='')
                if winnersData[ix]['fullRank']:
                    if type(winnersData[ix]['fullRank']) is str: print(', mucked hand', end ='')
                    else: print(' with %s'%winnersData[ix]['fullRank'][-1], end='')
                print()

        hHis.append({'winnersData': winnersData})

        self.players.append(self.players.pop(0)) # rotate table players for next hand

        if self.verbLev: print('(table)%s hand finished, table state %s' % (self.name, TBL_STT[self.state]))