from multiprocessing import Process
import numpy as np
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.pytypes import NPL
from pypaq.mpython.mptools import Que, QMessage
import time
from typing import List, Dict, Tuple, Optional, Union

from envy import DEBUG_MODE
from pologic.hand_history import HHistory, STATE
from pologic.podeck import PDeck


class PPlayer:
    """ PPlayer is an interface of player @table
    PPlayer is "a part of" poker table (PTable)
    it is only kind of mechanical interface with $
    the second part of player - policy - its "brain" - is managed by DMK """

    def __init__(
            self,
            id:str,
            table_moves: List,
            logger=     None,
            loglevel=   20,
    ):
        if not logger:
            logger = get_pylogger(name='PPlayer', level=loglevel)
        self.logger = logger

        self.id = id        # player id/address, unique for all tables
        self.table_moves = table_moves

        # fields below are managed(updated) by table._early_update_players()
        self.table = None
        self.pls = []       # names of all players @table, self name always first, then players to the right

        self.hand = None

        # player cash values, updated after each MOV by table
        self.cash = 0       # current player cash
        self.cash_ch = 0    # cash current hand (how much player put in total on current hand up to now)
        self.cash_cs = 0    # cash current street (how much player put on current street up to now)

        # helpers prepared once and used in _amc()
        self.n_moves = len(self.table_moves)
        self.enabledBRM = self.table_moves[3][0] == 'BRM'
        self.enabledBRA = self.table_moves[-1][0] == 'BRA'
        self.indexesBRX = [ix for ix in range(len(self.table_moves)) if 'BR' in self.table_moves[ix][0]]
        if self.enabledBRM:
            self.indexesBRX = self.indexesBRX[1:]
        if self.enabledBRA:
            self.indexesBRX = self.indexesBRX[:-1]

        self.nhs_IX = 0     # next hand_state index to update from (while sending game states)

        self.rng = np.random.default_rng()

    def _amc(self) -> Tuple[List[bool], List[int]]:
        """ computes allowed_moves and moves_cash
        returned moves_cash values = cash to be added to pot with current move
        it is a diff between BET-TO and player.cash_cs """

        allowed_moves = [True] * self.n_moves # by now all are allowed
        moves_cash =    [0]    * self.n_moves # by now all have 0

        moves_cash[2] = self.table.cash_tc - self.cash_cs # CLL

        min_bet_size = self.table.cash_tc + self.table.cash_rs

        if self.enabledBRM:
            moves_cash[3] = min_bet_size - self.cash_cs

        # BR-X
        for mIX in self.indexesBRX:

            mov_def = self.table_moves[mIX]
            if self.table.state == 1: val = round(mov_def[1] * self.table.cash_tc)
            else:                     val = round(mov_def[2] * self.table.pot)

            # check if bet meets min-bet size condition
            if val < min_bet_size:
                allowed_moves[mIX] = False
            else:
                moves_cash[mIX] = val - self.cash_cs # reduce by cash already put by the player on the current street

        # BRA (all-in)
        if self.enabledBRA:
            moves_cash[-1] = self.cash

        ### up to now "baseline" is set, it is time to update with more conditions

        # if there is cash to CLL then cannot CCK
        if moves_cash[2] > 0:
            allowed_moves[0] = False
        # if CLL cash is 0 then cannot CLL (nobody bet on the street yet -> CCK or BR)
        else:
            allowed_moves[2] = False

        # if can CCK then cannot FLD
        if allowed_moves[0]:
            allowed_moves[1] = False

        # not enough to make full CLL -> reduce
        if moves_cash[2] > self.cash:
            moves_cash[2] = self.cash

        # disable BRA if BRA cash == CLL cash
        if self.enabledBRA and moves_cash[2] == moves_cash[-1]:
            allowed_moves[-1] = False
            moves_cash[-1] = 0

        # eventually reduce moves_cash of BRM + BR-X and disable all next (higher)
        # INFO: if BRM will be reduced then this is not-valid-raise case
        already_reduced = False
        indexes_to_reduce = self.indexesBRX if not self.enabledBRM else [3] + self.indexesBRX
        for mIX in indexes_to_reduce:

            if already_reduced:
                allowed_moves[mIX] = False
                moves_cash[mIX] = 0

            else:
                if allowed_moves[mIX] and moves_cash[mIX] >= self.cash:
                    if not self.enabledBRA:
                        moves_cash[mIX] = self.cash
                    else:
                        allowed_moves[mIX] = False
                        moves_cash[mIX] = 0
                    already_reduced = True

        return allowed_moves, moves_cash


    def _make_decision(
            self,
            allowed_moves :List[bool],
            moves_cash: List[int],
    ) -> Tuple[int, NPL]:
        """ makes a decision given allowed moves and cash
        baseline implementation with random
        in "brained" implementations all the data (HH) may be used to make a decision
        returns move IX + list of probs """
        n_moves = len(self.table.moves)
        probs = self.rng.random(n_moves)
        probs *= allowed_moves
        probs /= sum(probs)
        dec = self.rng.choice(n_moves, p=probs)
        return dec, probs


    def take_hh(self, hh:HHistory):
        """ takes actual hand history (HH) from the table
        it is up to DMK, how to use this information
        called twice by table in a hand loop:
        - before making a move
        - after a hand finished (last states and rewards) """
        pass


    def select_move(self) -> Tuple[int,int,NPL]:
        """ with this method player is asked to make a move
        called by table in a hand loop
        returns:
        - ix of selected move from table_moves
        - cash of move (it is "raise by", NOT "raise to")
        - probabilities for each move from table_moves (policy probs) """

        self.logger.debug(f"$$$ cash values when player {self.id} was asked for the move:\n{self._cash_str()}")

        allowed_moves, moves_cash = self._amc()
        selected_move, probs = self._make_decision(allowed_moves, moves_cash)

        s = (f'player {self.id} selected move #{selected_move} {self.table_moves[selected_move][0]} ({moves_cash[selected_move]}$):\n'
             f'{self._after_decision_str(moves_cash, allowed_moves, probs, selected_move)}')
        self.logger.debug(s)

        return selected_move, moves_cash[selected_move], probs

    def _cash_str(self) -> str:
        """ prepares string from player & table cash values, for debug purposes """
        in_val = {
            '> self.cash':            self.cash,
            '> self.cash_ch':         self.cash_ch,
            '> self.cash_cs':         self.cash_cs,
            '> self.table.pot':       self.table.pot,
            '> self.table.cash_cs':   self.table.cash_cs,
            '> self.table.cash_tc':   self.table.cash_tc,
            '> self.table.cash_rs':   self.table.cash_rs}
        return '\n'.join([f'{k:20}: {in_val[k]}' for k in in_val])

    def _after_decision_str(
            self,
            moves_cash: List[int],
            allowed_moves: List[bool],
            probs,
            selected_move,
    ) -> str:
        """ prepares nice string after move has been selected, for debug purposes """
        pmc_strL = []
        for ix, (mv, mc, am, pr) in enumerate(zip(self.table_moves, moves_cash, allowed_moves, probs)):
            this = '<--- selected' if ix == selected_move else ''
            pmc_strL.append(f'> {ix:2} {mv[0]} {mc:3} -> {str(am):5} {pr:.3f} {this}')
        return '\n'.join(pmc_strL)


class PTable:
    """ PTable runs poker game hands """

    def __init__(
            self,
            name: str,
            moves: List[Tuple],
            cash_start: int,
            cash_sb: int,
            cash_bb: int,
            pl_ids: Optional[List[str]]=    None,
            logger=                         None,
            loglevel=                       20,
    ):
        if not logger:
            logger = get_pylogger(name='PTable', level=loglevel)
        self.logger = logger

        self.name = name

        self.moves = moves
        self.cash_sb = cash_sb
        self.cash_bb = cash_bb
        self.cash_start = cash_start

        self.deck =     PDeck()

        self.state =   0            # table state while running hand (int)
        self.cards =   []           # table cards (max 5)

        # cash
        self.pot =     0            # table pot (main)
        self.cash_cs = 0            # cash of current street
        self.cash_tc = 0            # cash to call by player (on current street) = highest bet on street
        self.cash_rs = self.cash_bb # legal raise size (recent raise)

        self.hand_ID: int=  0   # hand counter

        self.players = None
        # create players and put on the table, order of players reflects their current positions at table
        if pl_ids:
            self.players = self._build_players(pl_ids)
            self._early_update_players()

        size_nfo = f'(size:{len(pl_ids)}) ' if pl_ids else ''
        self.logger.info(f'*** PTable : {self.name} {size_nfo}*** initialized')

    def _build_players(self, pl_ids:List[str]) -> List[PPlayer]:
        players = [
            PPlayer(
                id=             id,
                table_moves=    self.moves,
                logger=         self.logger,
            ) for id in pl_ids]
        return players

    @property
    def is_headsup(self) -> bool:
        if self.players and len(self.players) == 2:
            return True
        return False

    def _early_update_players(self):
        """ updates players info since their position on table is known """

        # update table in player
        for pl in self.players:
            pl.table = self

        # update players names with self on 1st pos, then next to me, then next..
        pls = [pl.id for pl in self.players] # list of ids
        for pl in self.players:
            pl.pls = [] + pls # copy
            # rotate for every player to put his on first position
            while pl.pls[0] != pl.id:
                pl.pls.append(pl.pls.pop(0))

    def rotate_players(self):
        """ rotates table players (moves BTN right) """
        self.players.append(self.players.pop(0))

    def run_hand(self, hh_given:Optional[Union["HHistory",List[str]]]=None) -> HHistory:
        """ runs single hand
        if hh_given is given -> runs hand with given moves and cards """

        time_reset = time.time()
        tt, mt = 0, 0 # table & move time

        hh = HHistory(table_size=len(self.players), table_moves=self.moves)
        state = ('HST', (self.name, self.hand_ID))
        hh.events.append(state)
        self.logger.debug(hh.readable_event(state))

        hh_mvh = HHistory.extract_mvh(hh_given) if hh_given else None
        if hh_mvh:
            self.logger.debug(f'table was given hh_mvh: {hh_mvh}')

        # idle
        self.state = 0
        state = ('TST', (self.state,))
        hh.events.append(state)

        self.pot, self.cash_cs, self.cash_tc, self.cash_rs = 0, 0, 0, self.cash_bb
        state = ('T$$', (self.pot, self.cash_cs, self.cash_tc, self.cash_rs))
        hh.events.append(state)

        # reset player data (cash, cards)
        for pl in self.players:
            pl.hand = None  # players return cards
            pl.cash = self.cash_start
            pl.cash_ch = 0
            pl.cash_cs = 0
            pl.nhs_IX = 0

        # reset table data
        self.cards = []
        self.deck.reset()

        # rotate table to match mvh, remove SB move
        if hh_mvh:
            sb_pl_id = hh_mvh.pop(0)[0]
            while sb_pl_id != self.players[0].id:
                self.rotate_players()
        # remove also BB move
        if hh_mvh:
            hh_mvh.pop(0)

        hand_pls = [] + self.players  # copy order of players for current hand (SB, BB, ..)

        for ix in range(len(hand_pls)):
            state = ('POS', (hand_pls[ix].id, ix, hand_pls[ix].cash))
            hh.events.append(state)
            self.logger.debug(hh.readable_event(state))

        ### put blinds on table

        hand_pls[0].cash -= self.cash_sb
        hand_pls[0].cash_ch = self.cash_sb
        hand_pls[0].cash_cs = self.cash_sb
        state = ('PSB', (hand_pls[0].id, self.cash_sb))
        hh.events.append(state)
        self.logger.debug(hh.readable_event(state))

        hand_pls[1].cash -= self.cash_bb
        hand_pls[1].cash_ch = self.cash_bb
        hand_pls[1].cash_cs = self.cash_bb
        state = ('PBB', (hand_pls[1].id, self.cash_bb))
        hh.events.append(state)
        self.logger.debug(hh.readable_event(state))

        self.pot = self.cash_sb + self.cash_bb
        self.cash_cs = self.pot
        self.cash_tc = self.cash_bb
        self.cash_rs = self.cash_bb
        state = ('T$$', (self.pot, self.cash_cs, self.cash_tc, self.cash_rs))
        hh.events.append(state)

        ### hand cards

        c_pls = [] + hand_pls

        # rotate players for heads-up cards dealing (at heads-up BB is dealt first)
        if self.is_headsup:
            c_pls.append(c_pls.pop(0))

        for pl in c_pls:
            if hh_mvh:
                cas, cbs = hh_mvh.pop(0)[1:]
                self.deck.get_ex_card(cas)
                self.deck.get_ex_card(cbs)
                pl.hand = cas, cbs
            else:
                ca, cb = self.deck.get_card(), self.deck.get_card()
                pl.hand = PDeck.cts(ca), PDeck.cts(cb)
            state = ('PLH', (pl.id, pl.hand[0], pl.hand[1]))
            hh.events.append(state)

        # set preflop values
        lc_pIX = 1                           # loop closing player index
        mv_pIX = 0 if self.is_headsup else 2 # moving       player index

        # streets loop
        while True:

            # next street starts
            self.state += 1
            state = ('TST', (self.state,))
            hh.events.append(state)
            self.logger.debug(hh.readable_event(state))

            # reset some values for postflop
            if self.state > 1:

                lc_pIX = 0 if self.is_headsup else len(hand_pls)-1
                mv_pIX = 1 if self.is_headsup else 0

                self.cash_cs = 0
                self.cash_tc = 0
                self.cash_rs = self.cash_bb
                for pl in self.players:
                    pl.cash_cs = 0

                state = ('T$$', (self.pot, self.cash_cs, self.cash_tc, self.cash_rs))
                hh.events.append(state)

            # manage table cards
            new_table_cards = []
            if self.state == 2:
                # try to get table cards from mvh
                for _ in range(3):
                    if hh_mvh:
                        c = hh_mvh.pop(0)[1]
                        self.deck.get_ex_card(c)
                        new_table_cards.append(c)
                # eventually fill with random
                while len(new_table_cards) < 3:
                    new_table_cards.append(PDeck.cts(self.deck.get_card()))
            if self.state in [3,4]:
                if hh_mvh:
                    c = hh_mvh.pop(0)[1]
                    self.deck.get_ex_card(c)
                    new_table_cards = [c]
                else:
                    new_table_cards = [PDeck.cts(self.deck.get_card())]
            if new_table_cards:
                self.cards += new_table_cards
                state = ('TCD', tuple(new_table_cards))
                hh.events.append(state)
                self.logger.debug(hh.readable_event(state))

            # ask players for moves
            while len(hand_pls)>1:

                # next loop
                if mv_pIX == len(hand_pls):
                    mv_pIX = 0

                pl = hand_pls[mv_pIX]
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

                        ct = time.time()
                        tt += ct-time_reset
                        time_reset = ct

                        mv_id, mv_cash, probs = pl.select_move()

                        ct = time.time()
                        self.logger.debug(f'decision taken: {ct-time_reset:.7f}s')
                        mt += ct-time_reset
                        time_reset = ct

                    state = ('MOV', (pl.id, mv_id, mv_cash, (pl.cash, pl.cash_ch, pl.cash_cs)))
                    hh.events.append(state)

                    pl.cash -= mv_cash
                    pl.cash_ch += mv_cash
                    pl.cash_cs += mv_cash
                    self.pot += mv_cash
                    self.cash_cs += mv_cash

                    # FLD case
                    if mv_id == 1:
                        player_folded = True
                        hand_pls.pop(mv_pIX)

                    # BR case
                    if mv_id > 2:
                        player_raised = True
                        self.cash_rs = pl.cash_cs - self.cash_tc
                        self.cash_tc = pl.cash_cs
                        lc_pIX = mv_pIX-1 if mv_pIX>0 else len(hand_pls) - 1 # player before in loop

                    state = ('T$$', (self.pot, self.cash_cs, self.cash_tc, self.cash_rs))
                    hh.events.append(state)

                # player closing loop made decision (without raise)
                if lc_pIX == mv_pIX and not player_raised:
                    break

                if not player_folded: mv_pIX += 1
                elif lc_pIX > mv_pIX: lc_pIX -= 1 # move index left because of del

            # EXIT if river finished or everybody folded (to one player)
            if self.state == 4 or len(hand_pls) == 1:
                break

        winnersD = {
            pl.id: {
                'winner':       False,
                'full_rank':    'muck',
                'won':          0} for pl in self.players}

        # one player left finally (other passed)
        if len(hand_pls) == 1:
            w_name = hand_pls[0].id
            winnersD[w_name]['winner'] = True
            winnersD[w_name]['full_rank'] = 'not_shown'
            n_winners = 1

        # got more than one player -> showdown
        else:
            state = ('TST', (5,))
            hh.events.append(state)
            self.logger.debug(hh.readable_event(state))

            # get their ranks and top rank
            top_rank = 0
            for pl in hand_pls:
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
            state = ('PRS', (pl_id, winnersD[pl_id]['won'], winnersD[pl_id]['full_rank']))
            hh.events.append(state)
            self.logger.debug(hh.readable_event(state))

        state = ('HFN', (self.name, self.hand_ID))
        hh.events.append(state)

        # occasion to take a reward
        for pl in self.players:
            pl.take_hh(hh)

        if DEBUG_MODE:
            for pl in self.players:
                self.logger.debug(f'CARDS DEBUG: {pl.id}: {pl.hand[0]} {pl.hand[1]}')

        tt += time.time() - time_reset
        self.logger.debug(f' ### time - TBL:{tt:.7f}s MOV:{mt:.7f}s - {self.name} hand:{self.hand_ID}') # hand TBL / MOV time report

        self.rotate_players()  # rotate table players for next hand
        self.hand_ID += 1

        return hh


class QPPlayer(PPlayer):
    """ QPPlayer (PPlayer that uses Ques)
    sends hh with que o_que and asks i_que for decisions
        - o_que is common for all game players, i_que is one for player
    QPPlayer sends with que_from_player QMessages of two types:
        - 'make_decision' with data needed to make a decision (move)
        - 'state_changes' with new state changes """

    def __init__(
            self,
            que_to_player: Que,     # from DMK to player
            que_from_player: Que,   # from player to DMK
            **kwargs):
        PPlayer.__init__(self, **kwargs)
        self.que_to_player = que_to_player
        self.que_from_player = que_from_player

    def _make_decision(
            self,
            allowed_moves :List[bool],
            moves_cash :List[int],
    ) -> Tuple[int, NPL]:
        """ makes decision (communicates with ques)
        sends via que data with state before move and gets decision & probs from incoming que """

        message = QMessage(
            type = 'make_decision',
            data = {
                'id':               self.id,
                'allowed_moves':    allowed_moves,
                'moves_cash':       moves_cash})
        self.que_from_player.put(message)

        message = self.que_to_player.get()  # get move from DMK

        return message.data['selected_move'], message.data['probs']

    def _prepare_nt_states(self, hh:HHistory) -> List[STATE]:
        """ prepares list of new & translated events from table hh """
        state_changes = hh.translated(pls=self.pls, fr=self.nhs_IX)
        self.nhs_IX = len(hh.events)  # update index for next
        return state_changes

    def take_hh(self, hh: HHistory):
        """ takes actual hh from table, puts new & translated states to DMK """
        message = QMessage(
            type = 'state_changes',
            data = {
                'id':               self.id,
                'state_changes':    self._prepare_nt_states(hh)})
        self.que_from_player.put(message)


class QPTable(PTable, Process):
    """ QPTable is a Poker Table as a Process using Ques
    (Ques are managed by QPPlayer) """

    def __init__(
            self,
            que_to_gm :Que,                       # Que to GamesManager, here Table puts data for GamesManager
            pl_ques: Dict[str, Tuple[Que,Que]],
            **kwargs):

        Process.__init__(
            self,
            name=   kwargs['name'],
            target= self.run_hand_loop)

        self.que_to_gm = que_to_gm
        self.que_from_gm = Que()  # here Table receives data from GM

        PTable.__init__(self, **kwargs)

        pl_ids = list(pl_ques.keys())
        self.pl_ques = pl_ques
        self.players = self._build_players(pl_ids)
        self._early_update_players()


    def _build_players(self, pl_ids:List[str]) -> List[QPPlayer]:
        players = [
            QPPlayer(
                id=                 id,
                table_moves=        self.moves,
                que_to_player=      self.pl_ques[id][0],
                que_from_player=    self.pl_ques[id][1],
                logger=             self.logger,
            ) for id in pl_ids]
        return players

    def run_hand_loop(self):
        """ runs hands in a loop
        after starting the table target loop GM is waiting for a message from table """

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

    def kill(self):
        """ kills self (process) """
        self.terminate()