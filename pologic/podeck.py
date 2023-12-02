import itertools
import numpy as np
from ompr.runner import RunningWorker, OMPRunner
from pypaq.lipytools.files import r_pickle, w_pickle
from pypaq.lipytools.pylogger import get_pylogger
import random
import time
from torchness.types import NUM, NPL
from typing import Any, Union, Tuple, Optional, List, Iterable
from tqdm import tqdm

from envy import ASC_FP


ALL_CARDS = np.arange(52) # used by monte_carlo_prob_won()

# card figures
CRD_FIG = {
    0:      '2',
    1:      '3',
    2:      '4',
    3:      '5',
    4:      '6',
    5:      '7',
    6:      '8',
    7:      '9',
    8:      'T',
    9:      'J',
    10:     'D',
    11:     'K',
    12:     'A',
    13:     'X'}

# inverted card figures
CF_I = {
    '2':     0,
    '3':     1,
    '4':     2,
    '5':     3,
    '6':     4,
    '7':     5,
    '8':     6,
    '9':     7,
    'T':     8,
    'J':     9,
    'D':     10,
    'K':     11,
    'A':     12,
    'X':     13}

# card colors
CRD_COL = {
    0:      'S',    # spades, black (wino)
    1:      'H',    # hearts, red (serce)
    2:      'D',    # diamonds, blue (diament)
    3:      'C'}    # clubs, green (żołądź)

# inverted card colors
CC_I = {
    'S':    0,
    'H':    1,
    'D':    2,
    'C':    3}

# hand (5 cards) ranks (codes)
HND_RNK = {
    0:      'hc',   # high card
    1:      '2_',   # pair
    2:      '22',   # two pairs
    3:      '3_',   # three of
    4:      'ST',   # straight
    5:      'FL',   # flush
    6:      '32',   # full house
    7:      '4_',   # four of
    8:      'SF'}   # straight flush


class PDeck:
    """ Poker Cards Deck """

    def __init__(self):
        self.__full_init_deck = [PDeck._ctt_NUM(ci) for ci in range(52)]
        self.cards = None
        self.reset()


    def reset(self):
        """ resets deck to initial state """
        self.cards = [] + self.__full_init_deck
        random.shuffle(self.cards)


    def get_card(self) -> Tuple[int,int]:
        """ returns one card from deck """
        return self.cards.pop()


    def get_ex_card(self, card:Union[int,tuple,str]) -> Optional[int]:
        """ returns exact card from deck
        if id not present returns None """
        if type(card) is int or type(card) is str:
            card = PDeck.ctt(card)
        if card in self.cards:
            self.cards.remove(card)
            return card
        return None


    def get_7of_rank(self, rank:int) -> List[int]:
        """ returns seven card of given rank """

        seven = []
        if rank == 0:
            while True:
                self.reset()
                seven = [self.get_card() for _ in range(7)]
                if self.cards_rank(seven)[0] == 0:
                    break
        if rank == 1:
            while True:
                self.reset()
                seven = [self.get_card() for _ in range(7)]
                if self.cards_rank(seven)[0] == 1:
                    break
        if rank == 2:
            while True:
                self.reset()
                seven = [self.get_card() for _ in range(7)]
                if self.cards_rank(seven)[0] == 2:
                    break
        if rank == 3:
            while True:
                self.reset()
                fig = random.randrange(12)
                col = [c for c in range(4)]
                random.shuffle(col)
                col = col[:-1]
                seven = [(fig,c) for c in col]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(4)]
                if self.cards_rank(seven)[0] == 3:
                    break
        if rank == 4:
            while True:
                self.reset()
                fig = random.randrange(8)
                seven = [(fig+ix,random.randrange(4)) for ix in range(5)]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(2)]
                if self.cards_rank(seven)[0] == 4:
                    break
        if rank == 5:
            while True:
                self.reset()
                col = random.randrange(4)
                fig = [f for f in range(12)]
                random.shuffle(fig)
                fig = fig[:5]
                seven = [(f,col) for f in fig]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(2)]
                if self.cards_rank(seven)[0] == 5:
                    break
        if rank == 6:
            while True:
                self.reset()
                fig = [f for f in range(12)]
                random.shuffle(fig)
                fig = fig[:2]
                col = [c for c in range(4)]
                random.shuffle(col)
                col = col[:3]
                seven = [(fig[0],c) for c in col]
                col = [c for c in range(4)]
                random.shuffle(col)
                col = col[:2]
                seven += [(fig[1],c) for c in col]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(2)]
                if self.cards_rank(seven)[0] == 6:
                    break
        if rank == 7:
            while True:
                self.reset()
                fig = random.randrange(12)
                seven = [(fig,c) for c in range(4)]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(3)]
                if self.cards_rank(seven)[0] == 7:
                    break
        if rank == 8:
            while True:
                self.reset()
                fig = random.randrange(8)
                col = random.randrange(4)
                seven = [(fig + ix, col) for ix in range(5)]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(2)]
                if self.cards_rank(seven)[0] == 8:
                    break
        random.shuffle(seven)
        return seven

    @staticmethod
    def _stt(card:str) -> Tuple[int,int]:
        """ card str to tuple """
        return CF_I[card[0]],CC_I[card[1]]

    @staticmethod
    def cti(card:Union[int,tuple,str]) -> int:
        """ any card representation to int """
        if type(card) is str:
            card = PDeck._stt(card) # to tuple
        if type(card) is tuple:
            return card[0]*4+card[1]
        return card

    @staticmethod
    def _ctt_NUM(card:NUM) -> Tuple[int,int]:
        """ NUM fast implementation """
        return int(card / 4), card % 4

    @staticmethod
    def ctt(card:Union[int,tuple,str]) -> Tuple[int,int]:
        """ any card representation to Tuple[int,int] """

        if type(card) is tuple:
            return card
        if type(card) is str:
            return PDeck._stt(card)

        return PDeck._ctt_NUM(card)

    @staticmethod
    def cts(card:Union[int,tuple,str]) -> str:
        """ any card representation to str """
        # to tuple
        if type(card) is int:
            card = PDeck.ctt(card)
        if type(card) is tuple:
            return CRD_FIG[card[0]] + CRD_COL[card[1]]
        return card

    @staticmethod
    def cards_rank_tuples(
            cards: List[Tuple[int,int]]
    ) -> Tuple[int, int, List[Tuple[int,int]], str]:
        """ returns rank given cards
        core implementation for Iterable[Tuple[int,int]]
        selects 5 best from given 7 cards
        evaluates about 100K 7cards /sec  """

        cards = sorted(cards)

        # calc possible multiFig and colours
        c_fig = [[] for _ in range(13)]
        c_col = [[] for _ in range(4)]
        for c in cards:
            c_fig[c[0]].append(c)
            c_col[c[1]].append(c)

        n_fig = [len(f) for f in c_fig] # multiple figures

        # search for flush
        col_cards = None
        colour = None
        for cL in c_col:
            if len(cL) > 4:
                col_cards = cL
                colour = cL[0][1]
                break

        sc_fig = c_fig[-1:] + c_fig # add Ace at the beginning for A to 5 ST

        # straight case check; split sc_fig into continuous sequences
        sequences = []
        seq = []
        prev_ix = -1
        for ix,c in enumerate(sc_fig):
            if c:
                if prev_ix + 1 == ix:
                    seq.append(c)
                else:
                    if seq:
                        sequences.append(seq)
                    seq = [c]
                prev_ix = ix
        if seq: sequences.append(seq)

        in_row = max(sequences, key=lambda x:len(x)) if sequences else []
        possible_straight = len(in_row) > 4

        # straightFlush case check
        possible_straight_flush = False
        sequences = []
        seq = []
        if possible_straight and col_cards:
            for cs in in_row:
                c = [c for c in cs if c[1]==colour] # leave only cards in colour
                if c: seq.append(c)
                else:
                    if seq: sequences.append(seq)
                    seq = []
            if seq: sequences.append(seq)
        col_in_row = max(sequences, key=lambda x:len(x)) if sequences else []
        if len(col_in_row) > 4:
            possible_straight_flush = True
            col_in_row = [c[0] for c in col_in_row[-5:]]
        elif possible_straight:
            in_row = [c[0] for c in in_row[-5:]]

        if possible_straight_flush:                             top_rank = 8 # straight flush
        elif 4 in n_fig:                                        top_rank = 7 # four of
        elif (3 in n_fig and 2 in n_fig) or n_fig.count(3) > 1: top_rank = 6 # full house
        elif col_cards:                                         top_rank = 5 # flush
        elif possible_straight:                                 top_rank = 4 # straight
        elif 3 in n_fig:                                        top_rank = 3 # three of
        elif n_fig.count(2) > 1:                                top_rank = 2 # two pairs
        elif 2 in n_fig:                                        top_rank = 1 # pair
        else:                                                   top_rank = 0 # high card

        # find five cards
        five_cards = []

        if top_rank == 8:
            five_cards = col_in_row

        if top_rank == 7:
            four = []
            for cL in c_fig:
                if len(cL) == 4:
                    four = cL
                    break
            for c in four:
                cards.remove(c)
            five_cards = [cards[-1]] + four

        if top_rank == 6:
            five_cards = []
            for cL in c_fig:
                if len(cL) == 2: five_cards += cL
            for cL in c_fig:
                if len(cL) == 3: five_cards += cL
            five_cards = five_cards[-5:]

        if top_rank == 5:
            if len(col_cards) > 5:
                col_cards = col_cards[len(col_cards)-5:]
            five_cards = col_cards

        if top_rank == 4:
            if len(in_row) > 5:
                in_row = in_row[len(in_row)-5:]
            five_cards = in_row

        if top_rank == 3:
            three = []
            for cL in c_fig:
                if len(cL) == 3:
                    three = cL
            for c in three:
                cards.remove(c)
            five_cards = cards[-2:] + three

        if top_rank == 2:
            two2 = []
            for cL in c_fig:
                if len(cL) == 2:
                    two2 += cL
            if len(two2) > 4:
                two2 = two2[len(two2)-4:]
            for c in two2:
                cards.remove(c)
            five_cards = cards[-1:] + two2

        if top_rank == 1:
            two = []
            for cL in c_fig:
                if len(cL) == 2:
                    two = cL
            for c in two:
                cards.remove(c)
            five_cards = cards[-3:] + two

        if top_rank == 0:
            five_cards = cards[-5:]

        rank_value = 1000000*top_rank
        for ix in range(5):
            rank_value += five_cards[ix][0]*13**ix

        string = f'{HND_RNK[top_rank]} {rank_value:7} {" ".join([PDeck.cts(c) for c in five_cards])}'

        return top_rank, rank_value, five_cards, string

    @staticmethod
    def cards_rank_NPL(cards:NPL):
        """ fast implementation for NPL """
        return PDeck.cards_rank_tuples(cards=[PDeck._ctt_NUM(c) for c in cards])

    @staticmethod
    def cards_rank(cards: Iterable[Union[int, Tuple[int,int], str]]):
        """ basic implementation for any type """
        return PDeck.cards_rank_tuples(cards=[PDeck.ctt(c) for c in cards])


class ASC(dict):
    """ a dictionary with rank value of every sorted 7 cards ints
    example: {(0,1,9,20,30,34,43): 1001801}
    it speeds up rank "computation" massively:
    100K/sec with PDeck.cards_rank() to 500K/sec with ASC
    but ASC because of its size cannot be used with MP """

    def __init__(
            self,
            file_FP: str=   ASC_FP,
            use_QMP=        True,
            logger=         None,
            loglevel=       20,
    ):

        super().__init__()

        if not logger:
            logger = get_pylogger(name='ASC', level=loglevel)

        logger.info(f'Loading ASC dict cache from: {file_FP} ..')
        s_time = time.time()
        asc_ranks = r_pickle(file_FP)

        if asc_ranks:
            taken = time.time() - s_time
            logger.info(f' > using cached ASC dict (read taken {taken:.1f}s)')
        else:
            logger.info(' > cache not found, building All-Seven-Cards rank dict ..')
            comb_list = list(itertools.combinations([x for x in range(52)], 7))
            logger.info(f' > got {len(comb_list)} combinations')

            if use_QMP:

                class CRW(RunningWorker):
                    def process(self, **kwargs) -> Any:
                        return [(t, PDeck.cards_rank(t)[1]) for t in kwargs['tasks']]

                logger.info(f' > preparing tasks..')
                tasks = []
                task_bunch = []
                for cmb in tqdm(comb_list):
                    task_bunch.append(cmb)
                    if len(task_bunch) > 10000:
                        tasks.append({'tasks':task_bunch})
                        task_bunch = []
                if task_bunch:
                    tasks.append({'tasks':task_bunch})

                ompr = OMPRunner(rw_class=CRW)
                ompr.process(tasks=tasks)

                asc_ranks = {}
                for r in ompr.get_all_results():
                    for c, cr in r:
                        asc_ranks[c] = cr
                ompr.exit()

            else:
                asc_ranks = {cmb: PDeck.cards_rank(cmb)[1] for cmb in tqdm(comb_list)}

            logger.info(f'writing ASC to {file_FP} ..')
            w_pickle(asc_ranks, file_FP)

        self.update(asc_ranks)


    def cards_rank(self, c:Tuple[int]) -> int:
        """ returns rank for 7 cards (cards have to be sorted!) """
        return self[c]


def monte_carlo_prob_won(
        cards: Iterable[int],  # cards as an iterable of ints
        n_samples: int,
        asc: Optional[ASC] = None,
) -> float:
    """ winning probability estimation (Monte Carlo) for given cards """

    rng = np.random.default_rng()

    got_cards = np.asarray(cards)
    got_cards = np.delete(got_cards, np.where(got_cards == 52))

    all_cards_left = np.setdiff1d(ALL_CARDS, got_cards)
    n_missing = 9-len(got_cards)

    # TODO: it may be numpyized with batches of cards

    n_wins = 0
    for it in range(n_samples):

        sampled_cards = rng.choice(all_cards_left, size=n_missing, replace=False)
        nine_cards = np.concatenate([got_cards, sampled_cards])

        my_cards = nine_cards[:7]
        op_cards = nine_cards[2:]
        if asc:
            my_rank_value = asc[tuple(sorted(my_cards))]
            op_rank_value = asc[tuple(sorted(op_cards))]
        else:
            my_rank_value = PDeck.cards_rank_NPL(my_cards)[1]
            op_rank_value = PDeck.cards_rank_NPL(op_cards)[1]

        if my_rank_value > op_rank_value:  n_wins += 1
        if my_rank_value == op_rank_value: n_wins += 0.5

    return n_wins / n_samples


if __name__ == "__main__":
    asc = ASC()
