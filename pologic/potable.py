"""

 2019 (c) piteren

 PTable is a process that holds single poker game environment (with table players)

 DMK (decision maker) communicates with player (@table) through ques:
 - each player uses same (common) que to send his (from his perspective) game state (state changes from last move)
 - each player has its que (unique) where receives decisions from DMK

 During single hand table builds hand history, each player translates this history to its perspective
    (e.g. does not see other player cards)

 Keys of hahi (hand history):
    playersPC:  list of [pl_name,pl_hand]

"""

import copy
from multiprocessing import Process, Queue
import random

from pologic.podeck import PDeck

# table states
TBL_STT = {
    0:  'idle',
    1:  'preflop',
    2:  'flop',
    3:  'turn',
    4:  'river',
    5:  'fin'}

# supported table moves (moves of player supported by table)
TBL_MOV = {
    -1: 'END',  # game end
    0:  'C/F',  # check/fold
    1:  'CLL',  # call
    2:  'BR5',  # bet/raise 0.5
    3:  'BR8'}  # bet/raise 0.8
    # 4:'ALL'   # all-in

# position names for table sizes
POS_NMS = {
    2:  ['SB','BB'],
    3:  ['SB','BB','BTN'],
    6:  ['SB','BB','UTG','MP','CT','BTN'],
    9:  ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']}

# poker hand history
class history:

    """
        TIN:    table state (table_name:str)
        HST:    hand starts ((table_name:str,hand_id:int))
        TST:    table state (state:str)                         # state comes from potable.TBL_STT
        POS:    player position (pl.name:str, pos:str)          # pos in potable.POS_NMS
        PSB:    player small blind (pl.name:str, SB:int)
        PBB:    player big blind (pl.name:str, BB:int)
        PLH:    player hand (pl.name:str, ca:str, cb:str)       # c.str comes from PDeck.cts
        TCD:    table cards dealt (c0,c1,c2...)                 # only new cards are shown
        MOV:    player move (pl.name:str, move:str, cash:int)   # move from TBL_MOV.values()
        PRS:    player result (pl.name, won:int)
    """

    def __init__(self):
        self.events = []

    # adds action-value to history
    def add(self, act:str, val):
        self.events.append((act,val))

    # history to str
    def __str__(self):
        hstr = ''
        for el in self.events: hstr += '%s %s\n'%el
        return hstr[:-1]

# PPlayer is an interface of player #table (used by DMK)
class PPlayer:

    def __init__(
            self,
            id):    # player id/address, unique for all tables

        self.id = id
        self.name = 'pl_%s' % self.id

        # fields below are managed(updated) by table
        self.pls = [] # names of all players @table, initialised with table constructor, self name always first
        self.cash = 0 # player cash
        self.hand = None
        self.nhs_IX = 0 # next hand_state index to update from (while sending game states)
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

    # makes decision based on state, base implementation below selects random from possible moves
    def make_decision(self, state :dict):
        decision = sorted(list(TBL_MOV.keys()))
        decision.remove(-1)
        pm_probs = [0]*len(decision) # possible move probabilities
        possible_moves = state['possible_moves']
        for ix in range(len(possible_moves)):
            if possible_moves[ix]: pm_probs[ix] = 1
        return random.choices(decision, weights=pm_probs)[0] # decision returned as int from TBL_MOV

    # asks DMK for move (decision) having table status ...and any other info
    def make_move(
            self,
            hahi,           # hand history
            tbl_cash,       # table cash
            tbl_cash_tc):   # table cash to call

        # by now simple hardcoded amounts of cash
        moves_cash = {
            0:  0,
            1:  tbl_cash_tc - self.cr_cash,
            2:  int(tbl_cash_tc * 1.5) if tbl_cash_tc else int(0.5*tbl_cash),
            3:  int(tbl_cash_tc * 2) if tbl_cash_tc else int(0.8*tbl_cash)}

        # calculate possible moves and cash based on table state and hand history
        possible_moves = [True]*4 # by now all
        if tbl_cash_tc == self.cr_cash: possible_moves[1] = False  # cannot call (already called)
        if self.cash <= moves_cash[2]: possible_moves[3] = False # cannot make higher B/R

        # eventually reduce cash amount for call and smaller bet
        if self.cash < moves_cash[2]: moves_cash[2] = self.cash
        if possible_moves[1] and self.cash < moves_cash[1]: moves_cash[1] = self.cash
        if possible_moves[3] and self.cash < moves_cash[3]: moves_cash[3] = self.cash

        state_changes = copy.deepcopy(hahi[self.nhs_IX:]) # TODO: here was deepcopy, needed?
        self.nhs_IX = len(hahi) # update index for next

        self._translate_TH(state_changes) # update table history with player history

        # below move (decision) is made
        state_dict = {
            'pl_id':            self.id,
            'state_changes':    state_changes,
            'possible_moves':   possible_moves}
        selected_move = self.make_decision(state_dict)
        return selected_move, moves_cash[selected_move]

    # gets final hh (for reward etc.)
    def upd_state(
            self,
            hahi):
        pass

# PPlayer with (communication) ques
class QuedPPlayer(PPlayer):

    def __init__(self,id):

        super(QuedPPlayer,self).__init__(id)
        self.i_que = None # player input que
        self.o_que = None # player output que

    # makes decision based on state, base implementation below selects random from possible moves
    def make_decision(self, state :dict):
        self.o_que.put(state)  # put current state
        selected_move = self.i_que.get()  # get move from DMK
        return selected_move

    # sends DMK table update (without asking for move <- possible_moves is None)
    def upd_state(
            self,
            hahi):

        state_changes = hahi[self.nhs_IX:]  # copy part of history
        self.nhs_IX = len(hahi)  # update index for next

        self._translate_TH(state_changes) # update table history with player history

        self.o_que.put([self.id, state_changes, None])

# Poker Table is a single poker game environment
class PTable:

    def __init__(
            self,
            pl_ids :list,
            pl_class :type(PPlayer)=    PPlayer,
            name=                       'potable',
            SB=                         2,
            BB=                         5,
            start_cash=                 500,
            pmsg=                       True,
            verb=                       0):

        self.verb = verb
        self.pmsg = pmsg

        self.name = name

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

        self.players = self._make_players(pl_class, pl_ids)

        if self.verb: print('T(%s) created' % self.name)

    # builds players list
    def _make_players(self, pl_class, pl_ids):
        players = [pl_class(id) for id in pl_ids]
        # update players names with self on 1st pos, then next to me, then next...
        pls = [pl.name for pl in players] # list of names
        for pl in players:
            pl.pls = [] + pls # copy
            # rotate
            while pl.pls[0] != pl.name:
                nm = pl.pls.pop(0)
                pl.pls.append(nm)

        return players

    # runs single hand
    def run_hand(self):

        self.hand_ID += 1

        hh = history()
        hh.add('HST', (self.name,self.hand_ID))

        self.state = 0
        hh.add('TST', TBL_STT[self.state])

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

        self.state = 1

        h_pls = [] + self.players # original order of players for current hand (SB, BB, ..) ...every hand players are rotated
        p_names = POS_NMS[len(h_pls)]
        for ix in range(len(h_pls)): hh.add('POS',(h_pls[ix].name,p_names[ix]))

        # put blinds on table
        h_pls[0].cash -= self.SB
        h_pls[0].ch_cash = self.SB
        h_pls[0].cr_cash = self.SB
        hh.add('PSB',(h_pls[0].name,self.SB))

        h_pls[1].cash -= self.BB
        h_pls[1].ch_cash = self.BB
        h_pls[1].cr_cash = self.BB
        hh.add('PBB', (h_pls[1].name, self.BB))

        self.cash = self.SB + self.BB
        self.cash_cr = self.cash
        self.cash_tc = self.BB

        clc_pIX = 1 # current loop closing player index
        cmv_pIX = 2 # currently moving player index (for 2 players not valid but validated at first river loop)

        # hand cards
        for pl in h_pls:
            ca, cb = self.deck.get_card(), self.deck.get_card()
            pl.hand = PDeck.cts(ca), PDeck.cts(cb)
            hh.add(pl.name,(pl.hand[0],pl.hand[1]))

        hahi = [{'playersPC': [[pl.name, pl.hand] for pl in h_pls]}] # hand history list, one for all players
        # TODO: for pl in self.players: pl.upd_state(hahi=hahi)  ?? why it was send here at the beginning...

        # rivers loop
        while self.state < 5 and len(h_pls) > 1:

            hh.add('TST', TBL_STT[self.state])

            # manage table cards
            new_table_cards = []
            if self.state == 2: new_table_cards = [self.deck.get_card(), self.deck.get_card(), self.deck.get_card()]
            if self.state in [3,4]: new_table_cards = [self.deck.get_card()]
            if new_table_cards:
                new_table_cards = [PDeck.cts(c) for c in new_table_cards]
                self.cards += new_table_cards
                hh.add('TCD',new_table_cards)
                hahi.append({'newTableCards': new_table_cards})

            # ask players for moves
            while len(h_pls)>1: # game end breaks in the loop

                if cmv_pIX == len(h_pls): cmv_pIX = 0 # next loop

                player_folded = False
                player_raised = False
                pl = h_pls[cmv_pIX]
                if pl.cash: # player has cash (not all-in-ed yet)

                    # before move
                    mv_d = {
                        'tState':       self.state,
                        'tBCash':       self.cash,
                        'pName':        pl.name,
                        'pBCash':       pl.cash,        # player (before) cash
                        'pBCHandCash':  pl.ch_cash,
                        'pBCRiverCash': pl.cr_cash,
                        'bCashToCall':  self.cash_tc}   # (before) cash to call

                    # player makes move
                    pl_mv = h_pls[cmv_pIX].make_move(
                        hahi=           hahi,   # TODO: hh
                        tbl_cash=       self.cash,
                        tbl_cash_tc=    self.cash_tc)
                    if pl_mv[0]<0: return self._stop_hand(h_pls[cmv_pIX]) # game end
                    hh.add('MOV',(h_pls[cmv_pIX].name,TBL_MOV[pl_mv[0]],pl_mv[1]))
                    # TODO: add to hh other cash info (player,table)

                    pl.cash -= pl_mv[1]
                    pl.ch_cash += pl_mv[1]
                    pl.cr_cash += pl_mv[1]
                    self.cash += pl_mv[1]
                    self.cash_cr += pl_mv[1]

                    if pl_mv[0] > 1:
                        player_raised = True
                        self.cash_tc = h_pls[cmv_pIX].cr_cash
                        clc_pIX = cmv_pIX-1 if cmv_pIX>0 else len(h_pls) - 1 # player before in loop

                    if pl_mv[0] == 0 and self.cash_tc > pl.cr_cash:
                        player_folded = True
                        del(h_pls[cmv_pIX])

                    # after move
                    mv_d.update({
                        'plMove':       pl_mv,
                        'tACash':       self.cash,
                        'pACash':       pl.cash,
                        'pACHandCash':  pl.ch_cash,
                        'pACRiverCash': pl.cr_cash,
                        'aCashToCall':  self.cash_tc}) # (after) cash to call
                    hahi.append({'moveData': mv_d})

                if clc_pIX == cmv_pIX and not player_raised: break # player closing loop made decision (without raise)

                if not player_folded: cmv_pIX += 1
                elif clc_pIX > cmv_pIX: clc_pIX -= 1 # move index left because of del

            # reset for next river
            clc_pIX = len(h_pls)-1
            cmv_pIX = 0
            self.cash_cr = 0
            self.cash_tc = 0
            for pl in self.players: pl.cr_cash = 0

            self.state += 1  # move table to next state

        # winners template
        winners_data = []
        for pl in self.players:
            winners_data.append({
                'pName':        pl.name,
                'winner':       False,
                'fullRank':     None,
                'simpleRank':   0,
                'won':          0})

        if len(h_pls) == 1:
            winIX = self.players.index(h_pls[0])
            winners_data[winIX]['winner'] = True
            winners_data[winIX]['fullRank'] = 'muck'
            nWinners = 1
        else:
            top_rank = 0
            for pl in h_pls:
                cards = list(pl.hand)+self.cards
                rank = PDeck.cards_rank(cards)
                plIX = self.players.index(pl)
                if top_rank < rank[1]:
                    top_rank = rank[1]
                    winners_data[plIX]['fullRank'] = rank
                else:
                    winners_data[plIX]['fullRank'] = 'muck'
                winners_data[plIX]['simpleRank'] = rank[1]
            nWinners = 0
            for data in winners_data:
                if data['simpleRank'] == top_rank:
                    data['winner'] = True
                    nWinners += 1

        # manage cash and information about
        prize = self.cash / nWinners
        for ix in range(len(self.players)):
            pl = self.players[ix]
            myWon = -pl.ch_cash # netto lost
            if winners_data[ix]['winner']: myWon += prize # add netto winning
            winners_data[ix]['won'] = myWon

        for data in winners_data:
            hh.add('PRS',(data['pName'],data['won']))

        hahi.append({'winnersData': winners_data})
        for pl in self.players: pl.upd_state(hahi=hahi) # TODO: hh

        self.players.append(self.players.pop(0)) # rotate table players for next hand

        if self.hand_ID % 1000 == 0 or self.verb>1: print(hh)

        return True

    # stop hand initialised by player pl
    def _stop_hand(self,pl): return False

# Poker Table with (communication) ques
class QuedPTable(PTable, Process):

    def __init__(
            self,
            pi_ques :dict,  # dict of player input ques, their number defines table size (keys - player addresses)
            po_que :Queue,  # players output que
            *kwargs):

        PTable.__init__(
            self,
            pl_ids=     list(pi_ques.keys()),
            pl_class=   QuedPPlayer,
            *kwargs)
        Process.__init__(self,target=self.rh_proc)

        # add ques for players
        for pl in self.players:
            pl.i_que = pi_ques[pl.id]
            pl.o_que = po_que

    # runs hands in loop (for sep. process)
    def rh_proc(self):
        while True:
            if not self.run_hand(): break

    # stop hand initialised by player pl, sends triggers via ques
    def _stop_hand(self, pl):
        pl.i_que.put('anything') # put back for finishing player
        for pl in self.players:
            pl.i_que.get()
        self.players[0].o_que.put('finished %s' % self.name)
        return False


if __name__ == "__main__":

    table = PTable([0,1,2],verb=1)
    table.run_hand()