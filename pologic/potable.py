from multiprocessing import Process
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from pypaq.pytypes import NPL
from pypaq.mpython.mptools import Que, QMessage

from pologic.hand_history import HHistory
from pologic.podeck import PDeck
from envy import TABLE_CASH_START, TABLE_CASH_SB, TABLE_CASH_BB, TBL_MOV


# PPlayer is an interface of player @table (used by deciding class like DMK)
class PPlayer:

    def __init__(self, id:str):

        self.id = id        # player id/address, unique for all tables

        # fields below are managed(updated) by table._early_update_players()
        self.table = None
        self.pls = []       # names of all players @table, self name always first, then players to the right

        self.hand = None
        self.cash = 0       # current player cash
        self.cash_ch = 0    # cash current hand (how much player put in total on current hand up to now)
        self.cash_cr = 0    # cash current river (how much player put on current river up to now)

        self.nhs_IX = 0     # next hand_state index to update from (while sending game states)

    # returns possible moves and move cash (based on table cash)
    def _pmc(self) -> Tuple[List[bool], List[int]]:

        n_tbl_mov = len(TBL_MOV)
        possible_moves = [True] * n_tbl_mov     # by now all are possible

        # calculate moves cash
        moves_cash = [0] * n_tbl_mov            # by now all have 0
        for mIX in range(n_tbl_mov):
            val = 0
            if mIX == 1:
                val = self.table.cash_tc - self.cash_cr
            if mIX > 1:
                if self.table.state == 1:
                    val = round(TBL_MOV[mIX][1] * self.table.cash_tc)
                else:
                    val = round(TBL_MOV[mIX][2] * self.table.pot)
                    if val < 2 * self.table.cash_tc: val = 2 * self.table.cash_tc
                val -= self.cash_cr # TODO: why???

            moves_cash[mIX] = val

        if moves_cash[1] == 0:
            possible_moves[1] = False # cannot call (..nobody bet on the river yet ..check or bet)

        for mIX in range(2,n_tbl_mov):
            if moves_cash[mIX] <= moves_cash[1]: possible_moves[mIX] = False # cannot BET less than CALL

        # eventually reduce moves_cash and disable next (higher) moves
        for mIX in range(1,n_tbl_mov):
            if possible_moves[mIX] and self.cash <= moves_cash[mIX]:
                moves_cash[mIX] = self.cash
                for mnIX in range(mIX+1,n_tbl_mov): possible_moves[mnIX] = False

        return possible_moves, moves_cash

    # makes decision for possible moves (baseline implementation with random), returns move IX + list of probs
    def _make_decision(
            self,
            possible_moves :List[bool],
            moves_cash: List[int],
    ) -> Tuple[int, NPL]:
        n_moves = len(self.table.moves)
        probs = np.random.rand(n_moves)
        probs *= possible_moves
        probs /= sum(probs)
        dec = np.random.choice(n_moves, p=probs)
        return dec, probs

    # prepares list of new & translated events from table hh
    def _prepare_nt_states(self, hh: HHistory):
        state_changes = hh.translated(pls=self.pls, fr=self.nhs_IX)
        self.nhs_IX = len(hh.events)  # update index for next
        return state_changes

    # takes actual hand history (hh) from table, to be implemented how to use that information
    # called twice by table in a hand loop: before making a move and after a hand finished (last states and rewards)
    def take_hh(self, hh: HHistory): pass

    # makes move (based on hand history), called by table in a hand loop
    def select_move(self) -> Tuple[int,int,NPL]:
        possible_moves, moves_cash = self._pmc()
        selected_move, probs = self._make_decision(possible_moves, moves_cash)
        return selected_move, moves_cash[selected_move], probs


class PTable:
    """
    PTable is an object that runs single poker game (with players)

        During single hand table builds hand history (hh) (past events)
        Each PPlayer makes decisions/moves and may use history data:
            > history is translated to player perspective:
                > does not see other player cards
                > players are replaced with ints: 0-this player, 1-next..
    """

    def __init__(
            self,
            name: str,
            pl_ids: Optional[List[str]]=    None,
            logger=                         None):

        self.logger = logger

        self.name = name

        self.moves = sorted(list(TBL_MOV.keys()))
        self.SB = TABLE_CASH_SB
        self.BB = TABLE_CASH_BB
        self.start_cash = TABLE_CASH_START

        self.deck =     PDeck()

        self.state =    0   # table state while running hand (int)
        self.cards =    []  # table cards (max 5)
        self.pot =      0   # table pot (main)
        self.cash_cr =  0   # cash of current river
        self.cash_tc =  0   # cash to call by player (on current river) = highest bet on river
        self.cash_rs =  0   # recent raise size, TODO: should be updated after bet / raise / all-in
            # minbet = BB
            # minraise = cash_tc + cash_rs

        self.hand_ID: int=  0   # hand counter

        self.players = None
        # create players and put on the table
        if pl_ids:
            self.players = [PPlayer(id) for id in pl_ids]
            self._early_update_players()

        if self.logger: self.logger.info(f'*** PTable *** {self.name} initialized')

    # update some players info since their position on table is known
    def _early_update_players(self):

        # update table in player
        for pl in self.players:
            pl.table = self

        # update players names with self on 1st pos, then next to me, then next..
        pls = [pl.id for pl in self.players] # list of ids
        for pl in self.players:
            pl.pls = [] + pls # copy
            # rotate for every player to put his on first position
            while pl.pls[0] != pl.id:
                nm = pl.pls.pop(0)
                pl.pls.append(nm)

    # rotates table players (moves BTN right)
    def rotate_players(self):
        self.players.append(self.players.pop(0))

    # runs single hand
    def run_hand(
            self,
            hh_given: Optional[Union["HHistory",List[str]]]=    None,   # if given, runs hand with given moves and cards
    ):

        if self.logger: self.logger.debug(f'{self.name} starts hand {self.hand_ID}')

        hh_mvh = HHistory.extract_mvh(hh_given) if hh_given else None
        #print(hh_mvh)

        hh = HHistory()
        hh.add('HST', (self.name, self.hand_ID))

        self.pot, self.cash_cr, self.cash_tc = 0, 0, 0
        hh.add('T$$', (self.pot, self.cash_cr, self.cash_tc))

        self.state = 0
        hh.add('TST', (self.state,))

        # reset player data (cash, cards)
        for pl in self.players:
            pl.hand = None  # players return cards
            pl.cash = self.start_cash
            pl.cash_ch = 0
            pl.cash_cr = 0
            pl.nhs_IX = 0

        # reset table data
        self.cards = []
        self.deck.reset()

        self.state = 1

        # rotate table to match mvh, remove SB move
        if hh_mvh:
            sb_pl_id = hh_mvh.pop(0)[0]
            while sb_pl_id != self.players[0].id:
                self.rotate_players()
        # remove also BB move
        if hh_mvh:
            hh_mvh.pop(0)

        h_pls = [] + self.players  # copy order of players for current hand (SB, BB, ..)

        for ix in range(len(h_pls)):
            hh.add('POS', (h_pls[ix].id, ix))

        # put blinds on table
        h_pls[0].cash -= self.SB
        h_pls[0].cash_ch = self.SB
        h_pls[0].cash_cr = self.SB
        hh.add('PSB', (h_pls[0].id, self.SB))

        h_pls[1].cash -= self.BB
        h_pls[1].cash_ch = self.BB
        h_pls[1].cash_cr = self.BB
        hh.add('PBB', (h_pls[1].id, self.BB))

        self.pot = self.SB + self.BB
        self.cash_cr = self.pot
        self.cash_tc = self.BB
        hh.add('T$$', (self.pot, self.cash_cr, self.cash_tc))

        clc_pIX = 1 # current loop closing player index
        cmv_pIX = 2 # currently moving player index (for 2 players not valid but validated at first river loop)

        # hand cards
        for pl in h_pls:
            if hh_mvh:
                cas, cbs = hh_mvh.pop(0)[1:]
                self.deck.getex_card(cas)
                self.deck.getex_card(cbs)
                pl.hand = cas, cbs
            else:
                ca, cb = self.deck.get_card(), self.deck.get_card()
                pl.hand = PDeck.cts(ca), PDeck.cts(cb)
            hh.add('PLH', (pl.id, pl.hand[0], pl.hand[1]))

        # rivers loop
        while self.state < 5 and len(h_pls) > 1:

            hh.add('TST', (self.state,))
            hh.add('T$$', (self.pot, self.cash_cr, self.cash_tc))

            # manage table cards
            new_table_cards = []
            if self.state == 2:
                # try to get table cards from mvh
                for _ in range(3):
                    if hh_mvh:
                        c = hh_mvh.pop(0)[1]
                        self.deck.getex_card(c)
                        new_table_cards.append(c)
                # eventually fill with random
                while len(new_table_cards) < 3:
                    new_table_cards.append(PDeck.cts(self.deck.get_card()))
            if self.state in [3,4]:
                if hh_mvh:
                    c = hh_mvh.pop(0)[1]
                    self.deck.getex_card(c)
                    new_table_cards = [c]
                else:
                    new_table_cards = [PDeck.cts(self.deck.get_card())]
            if new_table_cards:
                self.cards += new_table_cards
                hh.add('TCD', tuple(new_table_cards))

            # ask players for moves
            while len(h_pls)>1: # game end breaks in the loop

                # next loop
                if cmv_pIX == len(h_pls):
                    cmv_pIX = 0

                pl = h_pls[cmv_pIX]
                player_folded = False
                player_raised = False
                if pl.cash: # player has cash (not all-in-ed yet)

                    # move is taken from mvh
                    if hh_mvh:
                        mv = hh_mvh.pop(0)
                        mv_id = mv[1]
                        mv_cash = mv[2]
                    # player makes move
                    else:
                        pl.take_hh(hh)  # takes actual hh from table
                        mv_id, mv_cash, probs = pl.select_move()
                        prs = f'[{" ".join([f"{p:.4f}" for p in probs])}]'
                        if self.logger: self.logger.debug(f'player: {pl.id} selected move #{mv_id} from probs: {prs}')
                    hh.add('MOV', (pl.id, mv_id, mv_cash, (pl.cash, pl.cash_ch, pl.cash_cr)))

                    pl.cash -= mv_cash
                    pl.cash_ch += mv_cash
                    pl.cash_cr += mv_cash
                    self.pot += mv_cash
                    self.cash_cr += mv_cash

                    if mv_id == 0 and self.cash_tc > pl.cash_cr:
                        player_folded = True
                        h_pls.pop(cmv_pIX)

                    if mv_id > 1:
                        player_raised = True
                        self.cash_tc = pl.cash_cr
                        clc_pIX = cmv_pIX-1 if cmv_pIX>0 else len(h_pls) - 1 # player before in loop

                    hh.add('T$$', (self.pot, self.cash_cr, self.cash_tc))

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
            pl.id: {
                'winner':       False,
                'full_rank':    'muck',
                'won':          0} for pl in self.players}

        # one player left finally (other passed)
        if len(h_pls) == 1:
            w_name = h_pls[0].id
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
                winnersD[pl.id]['full_rank'] = rank
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
        prize = self.pot / n_winners
        for pl in self.players:
            my_won = -pl.cash_ch  # netto lost
            if winnersD[pl.id]['winner']: my_won += prize  # add netto winning
            winnersD[pl.id]['won'] = my_won

        for pl_id in winnersD:
            hh.add('PRS', (pl_id, winnersD[pl_id]['won'], winnersD[pl_id]['full_rank']))

        hh.add('HFN', (self.name, self.hand_ID))

        # occasion to take reward
        for pl in self.players:
            pl.take_hh(hh)

        self.rotate_players() # rotate table players for next hand

        self.hand_ID += 1

        return hh

# PPlayer that uses Ques
class QPPlayer(PPlayer):
    """
        QPPlayer (qued PPlayer) sends hh with que o_que and asks i_que for decisions
            > o_que is common for all game players, i_que is one for player
        QPPlayer sends with que_from_player QMessages of two types:
        - 'make_decision' with data needed to make a decision (move)
        - 'state_changes' with new state changes
    """

    def __init__(
            self,
            id: str,
            que_to_player: Que,     # from DMK to player
            que_from_player: Que,   # from player to DMK
    ):
        super(QPPlayer,self).__init__(id)
        self.que_to_player = que_to_player
        self.que_from_player = que_from_player

    # makes decision (communicates with ques), sends via que data with state before move and gets decision & probs from incoming que
    def _make_decision(
            self,
            possible_moves :List[bool],
            moves_cash :List[int],
    ) -> Tuple[int, NPL]:

        message = QMessage(
            type = 'make_decision',
            data = {
                'id':               self.id,
                'possible_moves':   possible_moves,
                'moves_cash':       moves_cash})
        self.que_from_player.put(message)

        message = self.que_to_player.get()  # get move from DMK

        return message.data['selected_move'], message.data['probs']

    # takes actual hh from table, puts new & translated states to DMK
    def take_hh(self, hh: HHistory):
        message = QMessage(
            type = 'state_changes',
            data = {
                'id':               self.id,
                'state_changes':    self._prepare_nt_states(hh)})
        self.que_from_player.put(message)

# Poker Table as a Process using Ques (Ques are managed by QPPlayer)
class QPTable(PTable, Process):

    def __init__(
            self,
            que_to_gm :Que,                       # Que to GamesManager, here Table puts data for GamesManager
            pl_ques: Dict[str, Tuple[Que,Que]],
            **kwargs):

        Process.__init__(
            self,
            name=   kwargs['name'],
            target= self.run_hand_loop)

        PTable.__init__(
            self,
            **kwargs)

        self.que_to_gm = que_to_gm
        self.que_from_gm = Que()  # here Table receives data from GM

        pl_ids = list(pl_ques.keys())
        self.players = [QPPlayer(
            id=                 id,
            que_to_player=      pl_ques[id][0],
            que_from_player=    pl_ques[id][1]) for id in pl_ids]

        self._early_update_players()

    # runs hands in a loop
    def run_hand_loop(self):
        # INFO: after starting the table target loop GM is waiting for a message from table
        message = QMessage(
            type = 'table_status',
            data = f'{self.name} (QPTable) process started')
        self.que_to_gm.put(message)
        while True:

            self.run_hand()

            # eventually process Game Manager command
            message = self.que_from_gm.get(block=False)
            if message:
                if message.type == 'stop_table':
                    table_message = QMessage(
                        type = 'table_status',
                        data = f'{self.name} (QPTable) process started')
                    self.que_to_gm.put(table_message)
                    break

    # kills self (process)
    def kill(self): self.terminate()

# Qued Poker Table wrapped with a Process
class ProcPTable(Process):

    def __init__(
            self,
            que_from_gm: Que,
            que_to_gm: Que,
            pl_ques: Dict[str, Tuple[Que, Que]]):
        Process.__init__(self, target=self.__run)
        self.que_from_gm = que_from_gm
        self.que_to_gm = que_to_gm
        self.pl_ques = pl_ques

    def __run(self):
        message: QMessage = self.que_from_gm.get()
        assert message.type == 'init'
        table = QPTable(
            que_from_gm =   self.que_from_gm,
            que_to_gm=      self.que_to_gm,
            pl_ques=        self.pl_ques,
            **message.data)
        table.run_hand_loop()