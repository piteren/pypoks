"""

 2019 (c) piteren

 PTable is a process that holds single poker game environment (with table players)

 DMK (decision maker) communicates with player (@table) through ques:
 - each player uses same (common) que to send his game state (state changes from last move)
 - each player has its que (unique) where receives decisions from DMK

 keys of hahi (hand history):
    playersPC:  list of [pl_name,pl_hand]

"""

import copy
from multiprocessing import Process, Queue

from pologic.podeck import PDeck

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
    2:  'BR5',
    3:  'BR8'}

"""
# table moves 4 with ALL
TBL_MOV = {
    0:  'C/F',
    1:  'CLL',
    2:  'B/R',
    3:  'ALL'}
"""

# returns list of table position names in dependency of players number @table
def pos_names(nP=3):

    pNames = ['SB','BB','BTN'] # default for 3
    if nP == 2: pNames = pNames[:-1]
    if nP == 6: pNames = ['SB','BB','UTG','MP','CT','BTN']
    if nP == 9: pNames = ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']
    return pNames

# Poker Table is a single poker game environment
class PTable(Process):

    # Table Player is an interface of player used by DMK
    class PPlayer:

        def __init__(
                self,
                addr :tuple,    # player address in form (dix,pix)
                i_que :Queue,   # player input que
                o_que :Queue):  # player output que

            self.addr = addr
            self.name = 'p%s_%s' % (addr[0],addr[1])  # player takes name after DMK
            self.i_que = i_que
            self.o_que = o_que

            # fields below are managed(updated) by table
            self.pls = [] # names of all players @table, initialised with table constructor
            self.cash = 0 # player cash
            self.hand = None
            self.nhs_IX = 0 # next hand state index to update from (while sending game states)
            self.ch_cash = 0 # current hand cash (amount put by player on current hand)
            self.cr_cash = 0 # current river cash (amount put by player on current river)

        # translates table history into player history
        def _translate_TH(
                self,
                state_changes):

            for state in state_changes:
                key = list(state.keys())[0]

                if key == 'playersPC':
                    for el in state[key]:
                        el[0] = self.pls.index(el[0]) # replace names with indexes
                        if el[0]: el[1] = None # hide cards of not mine (just in case)

                if key == 'moveData':
                    state[key]['pIX'] = self.pls.index(state[key]['pName']) # insert player index
                    del(state[key]['pName']) # remove player name

                if key == 'winnersData':
                    for el in state[key]:
                        el['pIX'] = self.pls.index(el['pName']) # insert player index
                        del(el['pName']) # remove player name

        # asks DMK for move (decision) having table status ...and any other info
        def make_move(
                self,
                hahi,           # hand history
                tbl_cash,       # table cash
                tbl_cash_tc):   # table cash to call

            # by now simple hardcoded amounts of cash
            moves_cash = {
                0: 0,
                1: tbl_cash_tc - self.cr_cash,
                2: int(tbl_cash_tc * 1.5) if tbl_cash_tc else int(0.5 * tbl_cash),
                3: int(tbl_cash_tc * 2) if tbl_cash_tc else int(0.8 * tbl_cash)}

            # calculate possible moves and cash based on table state and hand history
            possible_moves = [True]*4 # by now all
            if tbl_cash_tc == self.cr_cash: possible_moves[1] = False  # cannot call (already called)
            if self.cash <= moves_cash[2]: possible_moves[3] = False # cannot make higher B/R

            # eventually reduce cash amount for call and smaller bet
            if self.cash < moves_cash[2]: moves_cash[2] = self.cash
            if possible_moves[1] and self.cash < moves_cash[1]: moves_cash[1] = self.cash
            if possible_moves[3] and self.cash < moves_cash[3]: moves_cash[3] = self.cash

            state_changes = copy.deepcopy(hahi[self.nhs_IX:]) # copy part of history
            self.nhs_IX = len(hahi) # update index for next

            self._translate_TH(state_changes) # update table history with player history

            # below move (decision) is made
            self.o_que.put([self.addr, state_changes, possible_moves]) # put current state
            selected_move = self.i_que.get() # get move from DMK
            if selected_move == 'game_end': return None # breaks game

            return selected_move, moves_cash[selected_move]

        # sends DMK table update (without asking for move - possibleMoves==None)
        def upd_state(
                self,
                hahi):

            state_changes = copy.deepcopy(hahi[self.nhs_IX:])  # copy part of history
            self.nhs_IX = len(hahi)  # update index for next

            self._translate_TH(state_changes) # update table history with player history

            self.o_que.put([self.addr, state_changes, None])


    def __init__(
            self,
            pi_ques :dict,           # dict of player input ques, their number defines table size (keys - player addresses)
            po_que :Queue,           # players output que
            name=       'potable',
            SB=         2,
            BB=         5,
            start_cash= 500,
            pmsg=       True,
            verb=       0):

        super().__init__(target=self.rh_proc)

        self.verb = verb
        self.pmsg = pmsg

        self.name = name
        self.n_players = len(pi_ques)

        self.SB = SB
        self.BB = BB
        self.start_cash = start_cash

        self.deck = PDeck()
        self.state = 0
        self.cards = []     # table cards (max 5)
        self.cash = 0       # on table
        self.cash_cr = 0    # current river
        self.cash_tc = 0    # how much player has to put to call ON CURRENT RIVER

        self.hand_ID = -1 # int incremented every hand

        self.players = [
            self.PPlayer(
                addr=   key,
                i_que=  pi_ques[key],
                o_que=  po_que) for key in pi_ques]
        # update players names with self on 1st pos
        for pl in self.players:
            pls = [pl.name for pl in self.players]
            pls.remove(pl.name)
            pl.pls = [pl.name] + pls

        if self.verb: print('T(%s) created' % self.name)

    # runs hands in loop (for sep. process)
    def rh_proc(self):
        while True:
            if not self.run_hand(): break

    # runs single hand
    def run_hand(self):

        # reset player data (cash, cards)
        for pl in self.players:
            pl.hand = None  # players return cards
            pl.cash = self.start_cash
            pl.nhs_IX = 0
            pl.ch_cash = 0
            pl.cr_cash = 0

        # reset table data
        self.cards = []
        self.deck.reset_deck()
        self.state = 0

        self.hand_ID += 1
        self.state = 1
        if self.verb:
            if self.hand_ID % 1000 == 0 or self.verb > 1:
                print('\nT(%s) starts new hand, handID %d' % (self.name, self.hand_ID))

        h_pls = [] + self.players # original order of players for current hand (SB, BB, ..)
        if self.pmsg:
            print(' ### T(%s) hand players:' % self.name)
            p_names = pos_names(self.n_players)
            for ix in range(self.n_players):
                pl = h_pls[ix]
                print(' ### P(%s) @ %s' % (pl.name, p_names[ix]))

        # put blinds on table
        h_pls[0].cash -= self.SB
        h_pls[0].ch_cash = self.SB
        h_pls[0].cr_cash = self.SB
        if self.pmsg: print(' ### P(%s) put SB %d$' % (h_pls[0].name, self.SB))
        h_pls[1].cash -= self.BB
        h_pls[1].ch_cash = self.BB
        h_pls[1].cr_cash = self.BB
        self.cash = self.SB + self.BB
        self.cash_cr = self.cash
        if self.pmsg: print(' ### P(%s) put BB %d$' % (h_pls[1].name, self.BB))
        self.cash_tc = self.BB
        # by now blinds info is not needed for hand history

        clc_pIX = 1 # current loop closing player index
        cm_pIX = 2 # currently moving player index (for 2 players not valid but validated at first river loop)

        # hand cards
        for pl in h_pls:
            ca, cb = self.deck.getCard(), self.deck.getCard()
            pl.hand = ca, cb
            if self.pmsg: print(' ### P(%s) taken hand %s %s' % (pl.name, PDeck.cts(ca), PDeck.cts(cb)))
        hahi = [{'playersPC': [[pl.name, pl.hand] for pl in h_pls]}] # hand history list, one for all players
        for pl in self.players: pl.upd_state(hahi=hahi)

        while self.state < 5 and len(h_pls) > 1:

            if self.pmsg: print(' ### T(%s) state %s' % (self.name, TBL_STT[self.state]))

            # manage table cards
            new_table_cards = []
            if self.state == 2: new_table_cards = [self.deck.getCard(), self.deck.getCard(), self.deck.getCard()]
            if self.state in [3,4]: new_table_cards = [self.deck.getCard()]
            if new_table_cards:
                self.cards += new_table_cards
                hahi.append({'newTableCards': new_table_cards})
                if self.pmsg:
                    print(' ### T(%s) cards: '%self.name, end='')
                    for card in self.cards: print(PDeck.cts(card), end=' ')
                    print()

            # ask players for moves
            while len(h_pls) > 1: # another important condition breaks in the loop

                if cm_pIX == len(h_pls): cm_pIX = 0 # next loop

                player_folded = False
                player_raised = False
                pl = h_pls[cm_pIX]
                if pl.cash: # player has cash (not allined yet)

                    # before move values
                    mvD = {
                        'tState':       self.state,
                        'tBCash':       self.cash,
                        'pName':        pl.name,
                        'pBCash':       pl.cash,
                        'pBCHandCash':  pl.ch_cash,
                        'pBCRiverCash': pl.cr_cash,
                        'bCashToCall':  self.cash_tc}

                    # player makes move
                    plMV = h_pls[cm_pIX].make_move(
                        hahi=           hahi,
                        tbl_cash=       self.cash,
                        tbl_cash_tc=    self.cash_tc)
                    if plMV is None: return False # breaks hand and game
                    mvD['plMove'] = plMV

                    pl.cash -= plMV[1]
                    pl.ch_cash += plMV[1]
                    pl.cr_cash += plMV[1]
                    self.cash += plMV[1]
                    self.cash_cr += plMV[1]

                    # cash after move
                    mvD['tACash'] =        self.cash
                    mvD['pACash'] =        pl.cash
                    mvD['pACHandCash'] =   pl.ch_cash
                    mvD['pACRiverCash'] =  pl.cr_cash

                    if plMV[0] > 1:
                        player_raised = True
                        self.cash_tc = h_pls[cm_pIX].cr_cash
                        clc_pIX = cm_pIX - 1 if cm_pIX > 0 else len(h_pls) - 1 # player before in loop
                    mvD['aCashToCall'] = self.cash_tc

                    if self.pmsg: print(' ### P(%s) had %4d$, moved %s with %4d$ (pCR:%4d$), now: tableCash %4d$ (tCR:%4d$) toCall %4d$' % (pl.name, mvD['pBCash'], TBL_MOV[plMV[0]], plMV[1], pl.cr_cash, self.cash, self.cash_cr, self.cash_tc))

                    if plMV[0] == 0 and self.cash_tc > pl.cr_cash:
                        player_folded = True
                        del(h_pls[cm_pIX])

                    hahi.append({'moveData': mvD})

                if clc_pIX == cm_pIX and not player_raised: break # player closing loop made decision (without raise)

                if not player_folded: cm_pIX += 1
                elif clc_pIX > cm_pIX: clc_pIX -= 1 # move index left because of del

            # reset for next river
            clc_pIX = len(h_pls)-1
            cm_pIX = 0
            self.cash_cr = 0
            self.cash_tc = 0
            for pl in self.players: pl.cr_cash = 0

            self.state += 1  # move table to next state

        if self.verb > 1: print('T(%s) rivers finished, time to select winners' % self.name)

        # winners template
        winnersData = []
        for pl in self.players:
            winnersData.append({
                'pName':        pl.name,
                'winner':       False,
                'fullRank':     None,
                'simpleRank':   0,
                'won':          0})

        if len(h_pls) == 1:
            winIX = self.players.index(h_pls[0])
            winnersData[winIX]['winner'] = True
            winnersData[winIX]['fullRank'] = 'muck'
            nWinners = 1
        else:
            top_rank = 0
            for pl in h_pls:
                cards = list(pl.hand)+self.cards
                rank = PDeck.cards_rank(cards)
                plIX = self.players.index(pl)
                if top_rank < rank[1]:
                    top_rank = rank[1]
                    winnersData[plIX]['fullRank'] = rank
                else:
                    winnersData[plIX]['fullRank'] = 'muck'
                winnersData[plIX]['simpleRank'] = rank[1]
            nWinners = 0
            for data in winnersData:
                if data['simpleRank'] == top_rank:
                    data['winner'] = True
                    nWinners += 1

        # manage cash and information about
        prize = self.cash / nWinners
        for ix in range(len(self.players)):
            pl = self.players[ix]
            myWon = -pl.ch_cash # netto lost
            if winnersData[ix]['winner']: myWon += prize # add netto winning
            winnersData[ix]['won'] = myWon
            if self.pmsg:
                print(' ### P(%s) : %d$' % (pl.name, myWon), end='')
                if winnersData[ix]['fullRank']:
                    if type(winnersData[ix]['fullRank']) is str: print(', mucked hand', end ='')
                    else: print(' with %s'%winnersData[ix]['fullRank'][-1], end='')
                print()

        hahi.append({'winnersData': winnersData})
        for pl in self.players: pl.upd_state(hahi=hahi)

        self.players.append(self.players.pop(0)) # rotate table players for next hand

        if self.verb > 1: print('T(%s) hand finished' % self.name)

        if self.verb > 2:
            print('HandHistory:')
            for el in hahi: print(el)

        return True
