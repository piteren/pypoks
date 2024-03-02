from pypaq.lipytools.files import w_jsonl, r_jsonl
from typing import List, Tuple, Optional, Dict

from envy import TBL_STT, DEBUG_MODE, get_pos_names
from pologic.game_config import GameConfig

STATE = Tuple[str,Tuple] # state type


# poker hand history
class HHistory:
    """ Hand History
    HH is build by the table while playing a hand.
    For more details and syntax check pologic.md file """

    def __init__(self, game_config:GameConfig):
        self.gc = game_config
        self.pos_names = get_pos_names(self.gc.table_size)
        self.events: List[STATE] = []
        self.events.append(('GCF',(self.gc.name,)))

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
    def extract_mvh(hh:List[str]) -> List[List[str]]:
        """ extracts some essential states from given List[str] (HHtexts) """
        mvh = [e.split() for e in hh]
        return [e for e in mvh if e[0] in ['POS:','PLH:','MOV:','TCD:']]

    @staticmethod
    def gc_from_events(events: List[STATE]) -> GameConfig:
        game_config_name = events[0][1][0]
        return GameConfig.from_name(game_config_name)

    def save(self, file:str):
        w_jsonl(self.events, file)

    @classmethod
    def from_file(cls, file:str) -> "HHistory":
        events = r_jsonl(file)
        hh = cls(cls.gc_from_events(events))
        hh.events = events
        return hh

    def __str__(self):
        return '\n'.join(states2HHtexts(self.events, game_config=self.gc))


def states2HHtexts(
        states: List[STATE],
        game_config: Optional[GameConfig]=  None,
        add_probs: bool=                    False,
        rename: Optional[Dict]=             None,
) -> List[str]:

    if not game_config:
        game_config = HHistory.gc_from_events(states)

    texts = []

    for st in states:
        text = None

        if st[0] == 'GCF':
            text = f'GCF: {st[1][0]}'

        if st[0] == 'POS':
            pos_names = get_pos_names(game_config.table_size)
            text = f'POS: {st[1][0]} {pos_names[st[1][1]]} {st[1][2]}'

        if st[0] in ['PSB','PBB']:
            text = f'{st[1][0]} {st[0][1:]} {st[1][1]}'

        if st[0] == 'T$$':
            text = f'table POT: {st[1][0]}'

        if st[0] == 'PLH':
            text = f'PLH: {st[1][0]} {st[1][1]} {st[1][2]}'

        if st[0] == 'TST':
            if st[1][0] != 0: # not idle
                text = f'** {TBL_STT[st[1][0]]}'

        if st[0] == 'TCD':
            text = f'TCD: {" ".join(st[1])}'

        if st[0] == 'MOV':
            text = f'MOV: {st[1][0]} {game_config.table_moves[st[1][1]][0]} {st[1][2]}'
            if add_probs and st[1][3]:
                text += f' {st[1][3]}'

        if st[0] == 'PRS':
            r = st[1][2] if type(st[1][2]) is str else st[1][2][-1]
            text = f'result: {st[1][0]} {st[1][1]} {r}'

        if text:
            texts.append(text)

    if rename:
        for k in rename:
            texts = [t.replace(k,rename[k]) for t in texts]

    return texts