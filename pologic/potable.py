"""

 2019 (c) piteren

    PTable is a process that holds single poker game environment (with table players)

        During single hand table builds hand history (hh)
        Each PPlayer uses this history translated to its perspective:
            > does not see other player cards
            > players are replaced with ints: 0-this player, 1-next ....
        Each PPlayer is asked to make decisions (based on hh)

    QPTable (qued PTable) is a PTable process to run hands with QPPlayers

        QPPlayer (qued PPlayer) sends with que hh (o_que) and asks for decisions from que (i_que)
            > o_que is common for all game players, i_que is one for player
        QPPlayer sends with o_que dicts of two types:
            { 'id': 'state_changes': }                  - sends new states @table (list of states)
            { 'id': 'possible_moves': 'moves_cash': }   - asks for move (from list of possible moves)
"""

import copy
from multiprocessing import Process, Queue
import random
import time
from typing import List
from tqdm import tqdm
from queue import Empty

from pologic.podeck import PDeck

from pologic.poenvy import N_TABLE_PLAYERS, \
    TBL_MOV, POS_NMS, \
    TABLE_CASH_START, TABLE_SB, TABLE_BB, DEBUG_MODE

# table states
TBL_STT = {
    0:  'idle',
    1:  'preflop',
    2:  'flop',
    3:  'turn',
    4:  'river',
    5:  'fin'}


# poker hand history
class HHistory:

    """
    hand history is build by the table, while playing a hand, below are implemented states:
    HST:    [table_name:str, hand_id:int]                           hand starts - maybe later add game info (table size, SB,BB ... )
    TST:    state:str                                               table state (potable.TBL_STT)
    POS:    [pln:str, pos:str]                                      player position (potable.POS_NMS)
    PSB:    [pln:str, SB:int]                                       player puts small blind
    PBB:    [pln:str, BB:int]                                       player puts big blind
    T$$:    [cash:int, cash_cr:int, cash_tc]                        table cash (on table, current river, to call(river))
    PLH:    [pln:str, ca:str, cb:str]                               player hand (PDeck.cts)
    TCD:    [c0,c1,c2...]                                           table cards dealt, only new cards are shown
    MOV:    [pln:str, move:str, mv_$:int, (pl.$, pl.$_ch, pl.$_cr)] player move (TBL_MOV.values()[0]), pl.cashes BEFORE move!
    PRS:    [pln, won:int, full_rank]                               player result, full_rank is a tuple returned by PDeck.cards_rank
    HFN:    [table_name:str, hand_id:int]                           hand finished
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
            if state[0] == 'PLH' and not DEBUG_MODE:
                if state[1][0]:
                    state[1][1], state[1][2] = None, None

        return trns

    # history to str
    def __str__(self):
        hstr = ''
        for el in self.events: hstr += '%s %s\n'%(el[0],el[1])
        return hstr[:-1]

# PPlayer is an interface of player @table (used deciding object (...DMK))
class PPlayer:

    def __init__(self, id): # player id/address, unique for all tables

        self.id = id
        self.name = 'pl_%s' % str(self.id)

        # fields below are managed(updated) by table
        self.table = None
        self.pls = [] # names of all players @table, initialised with table constructor, self name always first

        self.hand = None
        self.cash = 0       # cash (player left)
        self.cash_ch = 0    # cash current hand (how much player put yet, now its not needed cause can be calculated, but when player starts with diff cash amount it will be needed)
        self.cash_cr = 0    # cash current river (how much player put on current river)

        self.nhs_IX = 0 # next hand_state index to update from (while sending game states)

    # makes decision for possible moves (base implementation with random), to be implemented using taken hh
    def _make_decision(
            self,
            possible_moves :List[bool],
            moves_cash: List[int]) -> int:
        decision = self.table.moves
        pm_probs = [int(pm) for pm in possible_moves]
        dec = random.choices(decision, weights=pm_probs)[0] # decision returned as int from TBL_MOV
        return dec

    # returns possible moves and move cash (based on table cash)
    def _pmc(self) -> tuple:

        possible_moves = [True] * len(TBL_MOV)  # by now all

        # calc moves cash
        moves_cash = {}
        for mIX in TBL_MOV:
            val = 0
            if mIX == 1: val = self.table.cash_tc - self.cash_cr
            if mIX > 1:
                if self.table.state == 1:
                    val = round(TBL_MOV[mIX][1] * self.table.cash_tc)
                    val -= self.cash_cr
                else:
                    val = round(TBL_MOV[mIX][2] * self.table.cash)
                    if val < 2 * self.table.cash_tc: val = 2 * self.table.cash_tc
                    val -= self.cash_cr

            moves_cash[mIX] = val

        if moves_cash[1] == 0: possible_moves[1] = False # cannot call (...nobody bet on the river yet ...check or bet)
        for mIX in range(2,len(TBL_MOV)):
            if moves_cash[mIX] <= moves_cash[1]: possible_moves[mIX] = False # cannot BET less than CALL

        # eventually reduce moves_cash and disable next possibilities
        for mIX in range(1,len(TBL_MOV)):
            if possible_moves[mIX] and self.cash <= moves_cash[mIX]:
                moves_cash[mIX] = self.cash
                for mnIX in range(mIX+1,len(TBL_MOV)): possible_moves[mnIX] = False

        return possible_moves, moves_cash

    # prepares list of new & translated events from table hh
    def _prepare_nt_states(self, hh):
        state_changes = hh.translated(pls=self.pls, fr=self.nhs_IX)
        self.nhs_IX = len(hh.events)  # update index for next
        return state_changes

    # takes actual hh from table, to be implemented how to use that information
    # called twice @table loop: before making a move and after a hand finished(...last states and rewards)
    def take_hh(self, hh): pass

    # makes move (based on hand history)
    def make_move(self) -> tuple:
        possible_moves, moves_cash = self._pmc()
        selected_move = self._make_decision(possible_moves, moves_cash)
        return selected_move, moves_cash[selected_move]

# Poker Table is a single poker game environment
class PTable:

    def __init__(
            self,
            name,
            pl_ids :list,                           # len(pl_ids) == num players @table
            pl_class :type(PPlayer)=    PPlayer,
            verb=                       0):

        self.name = name
        self.verb = verb

        self.moves = sorted(list(TBL_MOV.keys()))
        self.SB = TABLE_SB
        self.BB = TABLE_BB
        self.start_cash = TABLE_CASH_START

        self.deck = PDeck()
        self.state = 0
        self.cards = []     # table cards (max 5)
        self.cash = 0       # cash (on table total)
        self.cash_cr = 0    # cash current river
        self.cash_tc = 0    # cash to call by player (on current river) = highest bet on river

        self.hand_ID = -1   # int incremented every hand

        self.players = self._make_players(pl_class, pl_ids)

        if self.verb: print(' *** PTable *** %s created' % self.name)

    # builds players list
    def _make_players(self, pl_class :type(PPlayer), pl_ids :list):

        players = [pl_class(id) for id in pl_ids]

        for pl in players: pl.table = self # update table

        # update players names with self on 1st pos, then next to me, then next...
        pls = [pl.name for pl in players] # list of names
        for pl in players:
            pl.pls = [] + pls # copy
            # rotate for every player to put his on first pos
            while pl.pls[0] != pl.name:
                nm = pl.pls.pop(0)
                pl.pls.append(nm)

        return players

    # runs single hand
    def run_hand(self):

        self.hand_ID += 1
        hh = HHistory()
        hh.add('HST', [self.name, self.hand_ID])

        self.cash, self.cash_cr, self.cash_tc = 0, 0, 0
        hh.add('T$$', [self.cash, self.cash_cr, self.cash_tc]) # needed for GUI

        self.state = 0
        hh.add('TST', TBL_STT[self.state])

        # reset player data (cash, cards)
        for pl in self.players:
            pl.hand = None  # players return cards
            pl.cash = self.start_cash
            pl.cash_ch = 0
            pl.cash_cr = 0
            pl.nhs_IX = 0

        # reset table data
        self.cards = []
        self.deck.reset_deck()

        self.state = 1

        h_pls = [] + self.players # original order of players for current hand (SB, BB, ..) ...every hand players are rotated
        p_names = POS_NMS[len(h_pls)]
        for ix in range(len(h_pls)): hh.add('POS', [h_pls[ix].name, p_names[ix]])

        # put blinds on table
        h_pls[0].cash -= self.SB
        h_pls[0].cash_ch = self.SB
        h_pls[0].cash_cr = self.SB
        hh.add('PSB', [h_pls[0].name, self.SB])

        h_pls[1].cash -= self.BB
        h_pls[1].cash_ch = self.BB
        h_pls[1].cash_cr = self.BB
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

        # rivers loop
        while self.state < 5 and len(h_pls) > 1:

            hh.add('TST', TBL_STT[self.state])
            hh.add('T$$', [self.cash, self.cash_cr, self.cash_tc])

            # manage table cards
            new_table_cards = []
            if self.state == 2: new_table_cards = [self.deck.get_card(), self.deck.get_card(), self.deck.get_card()]
            if self.state in [3,4]: new_table_cards = [self.deck.get_card()]
            if new_table_cards:
                new_table_cards = [PDeck.cts(c) for c in new_table_cards]
                self.cards += new_table_cards
                hh.add('TCD', new_table_cards)

            # ask players for moves
            while len(h_pls)>1: # game end breaks in the loop

                if cmv_pIX == len(h_pls): cmv_pIX = 0 # next loop

                player_folded = False
                player_raised = False
                pl = h_pls[cmv_pIX]
                if pl.cash: # player has cash (not all-in-ed yet)

                    # player makes move
                    pl.take_hh(hh) # takes actual hh from table
                    mv_id, mv_cash = pl.make_move()  # makes move
                    hh.add('MOV', [pl.name, TBL_MOV[mv_id][0], mv_cash, (pl.cash, pl.cash_ch, pl.cash_cr)])

                    pl.cash -= mv_cash
                    pl.cash_ch += mv_cash
                    pl.cash_cr += mv_cash
                    self.cash += mv_cash
                    self.cash_cr += mv_cash

                    if mv_id == 0 and self.cash_tc > pl.cash_cr:
                        player_folded = True
                        h_pls.pop(cmv_pIX)

                    if mv_id > 1:
                        player_raised = True
                        self.cash_tc = pl.cash_cr
                        clc_pIX = cmv_pIX-1 if cmv_pIX>0 else len(h_pls) - 1 # player before in loop

                    hh.add('T$$', [self.cash, self.cash_cr, self.cash_tc])

                if clc_pIX == cmv_pIX and not player_raised: break # player closing loop made decision (without raise)

                if not player_folded: cmv_pIX += 1
                elif clc_pIX > cmv_pIX: clc_pIX -= 1 # move index left because of del

            # reset for next river
            clc_pIX = len(h_pls)-1
            cmv_pIX = 0
            self.cash_cr = 0
            self.cash_tc = 0
            for pl in self.players: pl.cash_cr = 0

            self.state += 1  # move table to next state

        winnersD = {
            pl.name: {
                'winner':       False,
                'full_rank':    'muck',
                'won':          0} for pl in self.players}

        # one player left finally (other passed)
        if len(h_pls) == 1:
            w_name = h_pls[0].name
            winnersD[w_name]['winner'] = True
            winnersD[w_name]['full_rank'] = 'not_shown'
            n_winners = 1
        # got more than one player @showdown
        else:
            # get their ranks and top rank
            top_rank = 0
            for pl in h_pls:
                cards = list(pl.hand)+self.cards
                rank = PDeck.cards_rank(cards)
                winnersD[pl.name]['full_rank'] = rank
                if top_rank < rank[1]: top_rank = rank[1]

            # who's got top rank
            n_winners = 0
            for pln in winnersD:
                if winnersD[pln]['full_rank'][1] == top_rank:
                    winnersD[pln]['winner'] = True
                    n_winners += 1

            # not shown for rest
            for pln in winnersD:
                if not winnersD[pln]['winner']:
                    if winnersD[pln]['full_rank'] != 'muck':
                        winnersD[pln]['full_rank'] = 'not_shown'

        # manage cash and information about
        prize = self.cash / n_winners
        for pl in self.players:
            my_won = -pl.cash_ch  # netto lost
            if winnersD[pl.name]['winner']: my_won += prize  # add netto winning
            winnersD[pl.name]['won'] = my_won

        for pln in winnersD: hh.add('PRS', [pln, winnersD[pln]['won'], winnersD[pln]['full_rank']])

        hh.add('HFN', [self.name, self.hand_ID])

        for pl in self.players: pl.take_hh(hh) # occasion to take reward
        self.players.append(self.players.pop(0)) # rotate table players for next hand
        return hh

# PPlayer with (communication) ques
class QPPlayer(PPlayer):

    def __init__(self, id):

        super().__init__(id)
        self.i_que = None # player input que
        self.o_que = None # player output que

    # makes decision (communicates with ques)
    def _make_decision(
            self,
            possible_moves :List[bool],
            moves_cash :List[int]):
        qd = {
            'id':               self.id,
            'possible_moves':   possible_moves,
            'moves_cash':       moves_cash}
        self.o_que.put(qd)
        selected_move = self.i_que.get()  # get move from DMK
        return selected_move

    # takes actual hh from table, sends new & translated states using que
    def take_hh(self, hh):
        qd = {
            'id':               self.id,
            'state_changes':    self._prepare_nt_states(hh)}
        self.o_que.put(qd)

# Poker Table with (communication) ques(in fact managed by QPPlayer), implemented as a process
class QPTable(PTable, Process):

    def __init__(
            self,
            gm_que :Queue,  # GamesManager que, here Table puts data for GM
            pl_ques :dict,  # dict of player ques, their number defines table size (keys - player addresses)
            **kwargs):

        Process.__init__(self, target=self.__rh_proc)

        pl_ids = list(pl_ques.keys())
        random.shuffle(pl_ids)
        PTable.__init__(
            self,
            pl_ids=     pl_ids,
            pl_class=   QPPlayer,
            **kwargs)


        self.gm_que = gm_que
        self.in_que = Queue()  # here Table receives data from GM (...only poison object)

        # add ques for players
        for pl in self.players:
            pl.i_que, pl.o_que = pl_ques[pl.id]

    # runs hands in loop (for sep. process)
    def __rh_proc(self):
        self.gm_que.put(f'{self.name} (table process) started')
        while True:
            self.run_hand()

            try: # eventually stop table process
                if self.in_que.get_nowait():
                    self.gm_que.put('table_finished')
                    break
            except Empty: pass


    def kill(self): Process.terminate(self)


def example_table_speed(n_hands=100000):
    table = PTable(
        name=       'table_speed',
        pl_ids=     [0,1,2],
        verb=       1)
    stime = time.time()
    for _ in tqdm(range(n_hands)):
        table.run_hand()
        #hh = table.run_hand()
        #print('%s\n'%hh)
    n_sec = time.time()-stime
    print('time taken: %.1fsec (%d h/s)'%(n_sec, n_hands/n_sec))


def example_table_history(n=3):
    table = PTable(
        name=   'table_exh',
        pl_ids= [0,1,2],
        verb=   1)
    for _ in range(n):
        hh = table.run_hand()
        print(f'{hh}\n')


if __name__ == "__main__":

    #example_table_speed(100000)
    example_table_history(3)