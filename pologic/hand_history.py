"""

    2020 (c) piteren

    hand history is build by the table, while playing a hand, below are implemented states:

        HST:    (table_name:str, hand_id:int)                           hand starts - maybe later add game info (table size, SB,BB.. )
        TST:    (state:str,)                                            table state (potable.TBL_STT)
        POS:    (pln:str, pos:str)                                      player position (potable.POS_NMS)
        PSB:    (pln:str, SB:int)                                       player puts small blind
        PBB:    (pln:str, BB:int)                                       player puts big blind
        T$$:    (cash:int, cash_cr:int, cash_tc:int)                    table cash (on table, current river, to call(river))
        PLH:    (pln:str, ca:str, cb:str)                               player hand (PDeck.cts)
        TCD:    (c0,c1,c2..:int)                                        table cards dealt, only new cards are shown
        MOV:    (pln:str, move:str, mv_$:int, (pl.$, pl.$_ch, pl.$_cr)) player move (TBL_MOV.values()[0]), pl.cashes BEFORE move!
        PRS:    (pln:str, won:int, full_rank)                           player result, full_rank is a tuple returned by PDeck.cards_rank
        HFN:    (table_name:str, hand_id:int)                           hand finished

"""
from typing import List, Tuple, Optional

from pypoks_envy import DEBUG_MODE

STATE = Tuple[str,Tuple] # type
NAMED_EVENTS = ('POS','PSB','PBB','PLH','MOV','PRS') # events where player name


# poker hand history
class HHistory:

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
            if state[0] in NAMED_EVENTS:
                sd = list(state[1])
                sd[0] = pls.index(sd[0])

                # remove not 0 (I'am 0) cards
                if state[0] == 'PLH' and not DEBUG_MODE:
                    if sd[0] != 0:
                        sd[1], sd[2] = None, None

                state[1] = tuple(sd)

            trns.append(tuple(state))

        return trns

    # history to str
    def __str__(self):
        hstr = ''
        for el in self.events: hstr += f'{el[0]} {el[1]}\n'
        return hstr[:-1]