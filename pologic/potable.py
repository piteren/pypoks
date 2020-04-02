"""

 2019 (c) piteren

 PTable is a process that holds single poker game environment (with table players)

 During single hand table builds hand history (hh)
 Each PPlayer uses this history (translated to its perspective - e.g. does not see other player cards)
 and is asked to make decisions (based on hh)

 QPPlayer (qued PPlayer) sends with que hh (o_que) and asks for decisions from que (i_que)
 o_que is common for all game players, i_que is one for player
 QPPlayer sends with o_que dicts of two types:
    { 'id': 'state_changes': }  - informs about new states @table (list of states)
    { 'id': 'possible_moves': } - asks for move (from list of possible moves)

 QPTable (qued PTable) is a PTable process to run hands with QPPlayers

"""

import copy
from multiprocessing import Process, Queue
import random
import time
from typing import List

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
   -1:  'END',  # game end
    0:  'C/F',  # check/fold
    1:  'CLL',  # call
    2:  'BR5',  # bet/raise 0.5
    3:  'BR8'}  # bet/raise 0.8
    # 4:'BRA'   # all-in

# position names for table sizes
POS_NMS = {
    2:  ['SB','BB'],
    3:  ['SB','BB','BTN'],
    6:  ['SB','BB','UTG','MP','CT','BTN'],
    9:  ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']}

# poker hand history
class HHistory:

    """
        TIN:    table state (table_name:str)
        HST:    hand starts ((table_name:str,hand_id:int))
        TST:    table state (state:str)                         # state comes from potable.TBL_STT
        POS:    player position (pl.name:str, pos:str)          # pos in potable.POS_NMS
        PSB:    player small blind (pl.name:str, SB:int)
        PBB:    player big blind (pl.name:str, BB:int)
        T$$:    table cash(cash:int, cash_cr:int, cash_tc)      # on table, current river, to call(river)
        PLH:    player hand (pl.name:str, ca:str, cb:str)       # c.str comes from PDeck.cts
        TCD:    table cards dealt (c0,c1,c2...)                 # only new cards are shown
        MOV:    player move (pl.name:str, move:str, cash:int)   # move from TBL_MOV.values()
        PRS:    player result (pl.name, won:int)
    """

    def __init__(self):
        self.events = []

    # adds action-value to history
    def add(self, act:str, val):
        self.events.append([act,val])

    # returns translated into player history part of events[fr:to]
    def translated(
            self,
            pls :list,          # players (list of names)
            fr :int=    0,      # starting index
            to :int=    None):  # ending index

        if to is None: to = len(self.events)
        trns = copy.deepcopy(self.events[fr:to])
        for state in trns:

            # replace pl.names with indexes
            if state[0] in ['POS', 'PSB', 'PBB', 'PLH', 'MOV', 'PRS']:
                state[1][0] = pls.index(state[1][0])

            # remove not 0 (I'am 0) cards
            if state[0] == 'PLH':
                if state[1][0]:
                    state[1][1], state[1][2] = None, None

        return trns

    # history to str
    def __str__(self):
        hstr = ''
        for el in self.events: hstr += '%s %s\n'%(el[0],el[1])
        return hstr[:-1]

# PPlayer is an interface of player #table (used by DMK)
class PPlayer:

    def __init__(
            self,
            id,     # player id/address, unique for all tables
            table): # player table

        self.id = id
        self.name = 'pl_%s' % str(self.id)
        self.table = table

        # fields below are managed(updated) by table
        self.pls = [] # names of all players @table, initialised with table constructor, self name always first
        self.cash = 0 # player cash
        self.cash_cr = 0  # current river cash (amount put by player on current river)
        self.hand = None
        self.nhs_IX = 0 # next hand_state index to update from (while sending game states)

    # makes decision for possible moves (base implementation with random), to be implemented using taken hh
    def _make_decision(self, possible_moves :List[bool]):
        #decision = sorted(list(TBL_MOV.keys()))
        #decision.remove(-1)
        decision = [0,1,2,3] # hardcoded to speed-up
        pm_probs = [int(pm) for pm in possible_moves]
        dec = random.choices(decision, weights=pm_probs)[0] # decision returned as int from TBL_MOV
        return dec

    # returns possible moves and move cash (based on table cash)
    def _pmc(self):

        # by now simple hardcoded amounts of cash
        move_cash = {
           -1:  None,
            0:  0,
            1:  self.table.cash_tc - self.cash_cr,
            2:  int(self.table.cash_tc *1.5) if self.table.cash_tc else int(0.5*self.table.cash),
            3:  int(self.table.cash_tc *  2) if self.table.cash_tc else int(0.8*self.table.cash)}

        # calculate possible moves and cash based on table state and hand history
        possible_moves = [True]*4 # by now all
        if self.table.cash_tc == self.cash_cr: possible_moves[1] = False  # cannot call (already called)
        if self.cash <= move_cash[2]: possible_moves[3] = False # cannot make higher B/R

        # eventually reduce cash amount for call and smaller bet
        if self.cash < move_cash[2]: move_cash[2] = self.cash
        if possible_moves[1] and self.cash < move_cash[1]: move_cash[1] = self.cash
        if possible_moves[3] and self.cash < move_cash[3]: move_cash[3] = self.cash

        return possible_moves, move_cash

    # prepares list of new & translated events from hh, will be used by send_states
    def _prepare_nt_states(self, hh):
        state_changes = hh.translated(pls=self.pls, fr=self.nhs_IX)
        self.nhs_IX = len(hh.events)  # update index for next
        return state_changes

    # takes actual hh from table, to be implemented how to use that information
    def take_hh(self, hh): pass

    # makes move (based on hand history)
    def make_move(self):
        possible_moves, move_cash = self._pmc()
        selected_move = self._make_decision(possible_moves)
        return selected_move, move_cash[selected_move]

# PPlayer with (communication) ques
class QPPlayer(PPlayer):

    def __init__(self,id,table):

        PPlayer.__init__(self,id,table)
        self.i_que = None # player input que
        self.o_que = None # player output que

    # makes decision (communicates with ques)
    def _make_decision(self, possible_moves :list):
        qd = {
            'id':               self.id,
            'possible_moves':   possible_moves}
        self.o_que.put(qd)
        selected_move = self.i_que.get()  # get move from DMK
        return selected_move

    # takes actual hh from table, sends new & translated states using que
    def take_hh(self, hh):
        qd = {
            'id':               self.id,
            'state_changes':    self._prepare_nt_states(hh)}
        self.o_que.put(qd)

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
            verb=                       0):

        self.verb = verb
        self.name = name

        self.SB = SB
        self.BB = BB
        self.start_cash = start_cash

        self.deck = PDeck()
        self.state = 0
        self.cards = []     # table cards (max 5)
        self.cash = 0       # on table
        self.cash_cr = 0    # current river cash
        self.cash_tc = 0    # how much player has to put to call ON CURRENT RIVER

        self.hand_ID = -1   # int incremented every hand

        self.players = self._make_players(pl_class, pl_ids)

        if self.verb: print(' *** PTable *** %s created' % self.name)

    # builds players list
    def _make_players(self, pl_class, pl_ids):
        players = [pl_class(id,self) for id in pl_ids]
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

        hh = HHistory()
        hh.add('HST', [self.name, self.hand_ID])

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
        for ix in range(len(h_pls)): hh.add('POS', [h_pls[ix].name, p_names[ix]])

        # put blinds on table
        h_pls[0].cash -= self.SB
        h_pls[0].ch_cash = self.SB
        h_pls[0].cr_cash = self.SB
        hh.add('PSB', [h_pls[0].name, self.SB])

        h_pls[1].cash -= self.BB
        h_pls[1].ch_cash = self.BB
        h_pls[1].cr_cash = self.BB
        hh.add('PBB', [h_pls[1].name, self.BB])

        self.cash = self.SB + self.BB
        self.cash_cr = self.cash
        self.cash_tc = self.BB
        hh.add('T$$', [self.cash,self.cash_cr,self.cash_tc])

        clc_pIX = 1 # current loop closing player index
        cmv_pIX = 2 # currently moving player index (for 2 players not valid but validated at first river loop)

        # hand cards
        for pl in h_pls:
            ca, cb = self.deck.get_card(), self.deck.get_card()
            pl.hand = PDeck.cts(ca), PDeck.cts(cb)
            hh.add('PLH', [pl.name, pl.hand[0], pl.hand[1]])

        #hahi = [{'playersPC': [[pl.name, pl.hand] for pl in h_pls]}] # hand history list, one for all players
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
                hh.add('TCD', new_table_cards)
                #hahi.append({'newTableCards': new_table_cards})

            # ask players for moves
            while len(h_pls)>1: # game end breaks in the loop

                if cmv_pIX == len(h_pls): cmv_pIX = 0 # next loop

                player_folded = False
                player_raised = False
                pl = h_pls[cmv_pIX]
                if pl.cash: # player has cash (not all-in-ed yet)

                    """
                    # before move
                    mv_d = {
                        'tState':       self.state,
                        'tBCash':       self.cash,
                        'pName':        pl.name,
                        'pBCash':       pl.cash,        # player (before) cash
                        'pBCHandCash':  pl.ch_cash,
                        'pBCRiverCash': pl.cr_cash,
                        'bCashToCall':  self.cash_tc}   # (before) cash to call
                    """

                    # player makes move
                    pl.take_hh(hh) # takes actual hh from table
                    mv_id, mv_cash = pl.make_move()  # makes move
                    if mv_id<0: return self._stop_hand(pl) # game end
                    hh.add('MOV', [pl.name, TBL_MOV[mv_id], mv_cash])
                    # TODO: add to hh other cash info: player, before, after...

                    pl.cash -= mv_cash
                    pl.ch_cash += mv_cash
                    pl.cr_cash += mv_cash
                    self.cash += mv_cash
                    self.cash_cr += mv_cash

                    if mv_id == 0 and self.cash_tc > pl.cr_cash:
                        player_folded = True
                        h_pls.pop(cmv_pIX)

                    if mv_id > 1:
                        player_raised = True
                        self.cash_tc = pl.cr_cash
                        clc_pIX = cmv_pIX-1 if cmv_pIX>0 else len(h_pls) - 1 # player before in loop

                    """
                    # after move
                    mv_d.update({
                        'plMove':       pl_mv,
                        'tACash':       self.cash,
                        'pACash':       pl.cash,
                        'pACHandCash':  pl.ch_cash,
                        'pACRiverCash': pl.cr_cash,
                        'aCashToCall':  self.cash_tc}) # (after) cash to call
                    """
                    hh.add('T$$', [self.cash, self.cash_cr, self.cash_tc])

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
            hh.add('PRS', [data['pName'], data['won']])
        # TODO: add hand/mucked to hh

        for pl in self.players: pl.take_hh(hh) # occasion to take reward

        self.players.append(self.players.pop(0)) # rotate table players for next hand

        #print('\n@@@ hh\n%s'%hh)
        return hh

    # stop hand initialised by player pl
    def _stop_hand(self,pl): return False

# Poker Table with (communication) ques, implemented as process
class QPTable(PTable, Process):

    def __init__(
            self,
            pl_ques :dict,  # dict of player ques, their number defines table size (keys - player addresses)
            **kwargs):

        pl_ids = list(pl_ques.keys())
        PTable.__init__(
            self,
            pl_ids=     pl_ids,
            pl_class=   QPPlayer,
            **kwargs)
        Process.__init__(self, target=self.__rh_proc)

        # add ques for players
        for pl in self.players:
            pl.i_que, pl.o_que = pl_ques[pl.id]

    # runs hands in loop (for sep. process)
    def __rh_proc(self):
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
    n_hands = 100000
    stime = time.time()
    for _ in range(n_hands):
        table.run_hand()
        #hh = table.run_hand()
        #print('%s\n'%hh)
    n_sec = time.time()-stime
    print('time taken: %.1fsec (%d h/s)'%(n_sec, n_hands/n_sec))