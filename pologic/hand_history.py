from typing import List, Tuple, Optional, Union

from envy import TBL_STT, TBL_MOV, DEBUG_MODE

STATE = Tuple[str,Tuple] # type


# poker hand history
class HHistory:

    """
    Hand History is build by the table, while playing a hand, below are implemented states:

    HST: (table_name:str, hand_id:int)                                      hand starts
    TST: (state:int,)                                                       table state
    POS: (pln:str, pos:int)                                                 player position
    PSB: (pln:str, SB$:int)                                                 player puts SB
    PBB: (pln:str, BB$:int)                                                 player puts BB
    T$$: (pot:int, cash_cr:int, cash_tc:int)                                table cash
    PLH: (pln:str, ca:str, cb:str)                                          player hand
    TCD: (c0,c1,c2..:str)                                                   table cards dealt, only new cards are shown
    MOV: (pln:str, mv:int, mv_cash:int, (pl.cash, pl.cash_ch, pl.cash_cr))  player move (pl.cashes BEFORE move)
    PRS: (pln:str, won:int, full_rank)                                      player result (full_rank is a tuple returned by PDeck.cards_rank)
    HFN: (table_name:str, hand_id:int)                                      hand finished
    """

    def __init__(self):
        self.events: List[STATE] = []

    # adds action-value to history
    def add(self, act:str, val:Tuple):
        self.events.append((act,val))

    # returns translated into player history part of events[fr:to]
    def translated(
            self,
            pls: List[str],             # players (list of names)
            fr: Optional[int]=  None,   # starting index
            to: Optional[int]=  None,   # ending index
    ) -> List[Tuple]:

        if fr is None: fr = 0
        if to is None: to = len(self.events)

        trns = []
        for st in self.events[fr:to]:

            state = list(st)

            # replace pl.names with indexes
            if state[0] in ('POS','PSB','PBB','PLH','MOV','PRS'):
                sd = list(state[1])
                sd[0] = pls.index(sd[0])

                # remove not 0 (I am 0) cards
                if state[0] == 'PLH' and not DEBUG_MODE:
                    if sd[0] != 0:
                        sd[1], sd[2] = None, None

                state[1] = tuple(sd)

            trns.append(tuple(state))

        return trns

    # extracts moves and hands from HH or list[readable_events] <- those events are needed to run the table same hand again
    @staticmethod
    def extract_mvh(hh:Union["HHistory",List[str]]) -> List[Tuple]:
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

    # extracts simple readable events
    @staticmethod
    def readable_event(st:STATE) -> Optional[str]:

        if st[0] == 'HST':
            return '***** hand starts'

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
            return f'{st[1][0]} {TBL_MOV[st[1][1]]} {st[1][2]}'

        if st[0] == 'PRS':
            r = st[1][2] if type(st[1][2]) is str else st[1][2][-1]
            return f'$$$: {st[1][0]} {st[1][1]} {r}'

    # history to str
    def __str__(self):
        hstr = ''
        for el in self.events: hstr += f'{el[0]} {el[1]}\n'
        return hstr[:-1]