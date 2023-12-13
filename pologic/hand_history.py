from pypaq.lipytools.files import w_json, r_json
from typing import List, Tuple, Optional, Union

from envy import TBL_STT, DEBUG_MODE, get_pos_names

STATE = Tuple[str,Tuple] # state type


# poker hand history
class HHistory:
    """ Hand History

    HH is build by the table, while playing a hand, below are implemented states:

    HST: (table_name:str, hand_id:int)                                          hand starts
    TST: (state:int,)                                                           table state
    POS: (pl_id:str, pos:int, pl.cash)                                          player position and starting cash
    PSB: (pl_id:str, SB$:int)                                                   player puts SB
    PBB: (pl_id:str, BB$:int)                                                   player puts BB
    T$$: (pot:int, cash_cs:int, cash_tc:int, cash_rs:int)                       table cash, added after each player MOV (PSB+PBB <- together)
    PLH: (pl_id:str, ca:str, cb:str)                                            player hand
    TCD: (c0,c1,c2..:str)                                                       table cards dealt, only new cards are shown
    MOV: (pl_id:str, mv:int, mv_cash:int, (pl.cash, pl.cash_ch, pl.cash_cs))    player move (pl.cashes BEFORE move)
    PRS: (pl_id:str, won:int, full_rank)                                        player result (full_rank is a tuple returned by PDeck.cards_rank)
    HFN: (table_name:str, hand_id:int)                                          hand finished

    below generic flow of states:

        HST                             <- hand starts
        TST(0)                          <- idle
        T$$
        POS,POS,..
        PSB,PBB
        T$$
        PLH,PLH,..

        TST(1)                          <- preflop
         --loop of MOV > T$$

        TST(2)                          <- flop
        TCD
         --loop of MOV > T$$

        .. (next streets like flop)

        TST(5)                          <- showdown [optional]
        PRS,PRS,..
        HFN                             <- hand finished """

    def __init__(self, table_size:int, table_moves:List):
        self.pos_names = get_pos_names(table_size)
        self.table_moves = table_moves
        self.events: List[STATE] = []

    def translated(
            self,
            pls: List[str],             # players (list of ids)
            fr: Optional[int]=  None,   # starting index
            to: Optional[int]=  None,   # ending index
    ) -> List[STATE]:
        """ returns events translated into "player perspective" - I am 0
        - player names (pl_id:str) is replaced with int, where 0 means "me" and other players are marked with 1,2..
        - PLH of other players are removed (if not DEBUG_MODE) """

        if fr is None: fr = 0
        if to is None: to = len(self.events)

        trns = []
        for st in self.events[fr:to]:

            state = list(st)

            # replace pl.id with indexes
            if state[0] in ('POS','PSB','PBB','PLH','MOV','PRS'):
                sd = list(state[1])
                sd[0] = pls.index(sd[0])

                # remove not my (0) cards
                if state[0] == 'PLH' and not DEBUG_MODE:
                    if sd[0] != 0:
                        sd[1], sd[2] = None, None

                state[1] = tuple(sd)

            trns.append(tuple(state))

        return trns

    @staticmethod
    def extract_mvh(hh:Union["HHistory",List[str]]) -> List[Tuple]:
        """ extracts moves and hands from HH or list[readable_events]
        those events are needed to run the table same hand again """
        if type(hh) is HHistory:
            mvh = []
            for e in hh.events:
                if e[0] in ['PSB','PBB']:
                    mvh.append((e[1][0], e[0][1:], e[1][1]))
                if e[0] in ['MOV','PLH']:
                    mvh.append((e[1][0], e[1][1], e[1][2]))
                if e[0] == 'TCD':
                    for c in e[1]:
                        mvh.append(('tc',c))
            return mvh
        else:
            mvh = [e.split() for e in hh if e.startswith('pl')]
            print(mvh)
            return [
                (e[0], e[1], int(e[2])) if e[1] != 'cards:' else (e[0], e[2], e[3])
                for e in mvh
            ]

    def readable_event(
            self,
            st:STATE,
    ) -> Optional[str]:
        """ extracts simple readable events """

        if st[0] == 'HST':
            return f'***** hand #{st[1][1]} starts'

        if st[0] == 'POS':
            return f'{st[1][0]} at {self.pos_names[st[1][1]]} with ${st[1][2]}'

        if st[0] in ['PSB','PBB']:
            return f'{st[1][0]} {st[0][1:]} {st[1][1]}'

        if st[0] == 'PLH':
            return f'{st[1][0]} cards: {st[1][1]} {st[1][2]}'

        if st[0] == 'TST':
            if st[1][0] != 0: # not idle
                return f'** {TBL_STT[st[1][0]]}'

        if st[0] == 'TCD':
            return f'table cards: {" ".join(st[1])}'

        if st[0] == 'MOV':
            return f'{st[1][0]} {self.table_moves[st[1][1]][0]} {st[1][2]}'

        if st[0] == 'PRS':
            r = st[1][2] if type(st[1][2]) is str else st[1][2][-1]
            return f'$$$: {st[1][0]} {st[1][1]} {r}'

    def save(self, file:str):
        w_json(self.events, file)
        re = [self.readable_event(e) for e in self.events]
        re = [e for e in re if e]
        with open(f'{file}_txt', 'w') as file:
            for e in re:
                file.write(f'{e}\n')

    def load(self, file:str):
        self.events = r_json(file)

    def __str__(self):
        hstr = ''
        for e in self.events:
            re = self.readable_event(e)
            if re:
                hstr += f'{re}\n'
        if hstr: hstr = hstr[:-1]
        return hstr