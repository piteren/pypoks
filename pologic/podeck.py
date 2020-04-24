"""

 2019 (c) piteren

    cards may be represented in 3 forms:
        int 0-52            # 52 is pad (no card) - this is default internal deck representation
        tuple (0-13,0-3)
        str AC
"""

import itertools
import random
import time
from tqdm import tqdm

import ptools.lipytools.little_methods as lM

from ptools.mpython.qmp import QueMultiProcessor

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

# inverted card colors
CC_I = {
    'S':    0,
    'H':    1,
    'D':    2,
    'C':    3}

# card colors
CRD_COL = {
    0:      'S',    # spades, black (wino)
    1:      'H',    # hearts, red (serce)
    2:      'D',    # diamonds, blue (diament)
    3:      'C'}    # clubs, green (żołądź)

# hand (5 cards) ranks (codes)
HND_RNK = {
    0:  'hc',
    1:  '2_',
    2:  '22',
    3:  '3_',
    4:  'ST',
    5:  'FL',
    6:  '32',
    7:  '4_',
    8:  'SF'}


class PDeck:

    def __init__(self):

        self.__fullInitDeck = [PDeck.ctt(ci) for ci in range(52)]
        self.cards = None
        self.reset_deck()

    # resets deck to initial state
    def reset_deck(self):
        self.cards = [] + self.__fullInitDeck
        random.shuffle(self.cards)

    # returns one card from deck, returns int
    def get_card(self): return self.cards.pop()

    # returns exact card from deck, id not present return None
    def getex_card(self, card: tuple or int):
        if type(card) is int: card = PDeck.ctt(card)
        if card in self.cards:
            self.cards.remove(card)
            return card
        return None

    # returns seven card of given rank
    def get7of_rank(self, rank :int):

        seven = []
        if rank == 0:
            while True:
                self.reset_deck()
                seven = [self.get_card() for _ in range(7)]
                if self.cards_rank(seven)[0] == 0: break
        if rank == 1:
            while True:
                self.reset_deck()
                seven = [self.get_card() for _ in range(7)]
                if self.cards_rank(seven)[0] == 1: break
        if rank == 2:
            while True:
                self.reset_deck()
                seven = [self.get_card() for _ in range(7)]
                if self.cards_rank(seven)[0] == 2: break
        if rank == 3:
            while True:
                self.reset_deck()
                fig = random.randrange(12)
                col = [c for c in range(4)]
                random.shuffle(col)
                col = col[:-1]
                seven = [(fig,c) for c in col]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(4)]
                if self.cards_rank(seven)[0] == 3: break
        if rank == 4:
            while True:
                self.reset_deck()
                fig = random.randrange(8)
                seven = [(fig+ix,random.randrange(4)) for ix in range(5)]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(2)]
                if self.cards_rank(seven)[0] == 4: break
        if rank == 5:
            while True:
                self.reset_deck()
                col = random.randrange(4)
                fig = [f for f in range(12)]
                random.shuffle(fig)
                fig = fig[:5]
                seven = [(f,col) for f in fig]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(2)]
                if self.cards_rank(seven)[0] == 5: break
        if rank == 6:
            while True:
                self.reset_deck()
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
                if self.cards_rank(seven)[0] == 6: break
        if rank == 7:
            while True:
                self.reset_deck()
                fig = random.randrange(12)
                seven = [(fig,c) for c in range(4)]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(3)]
                if self.cards_rank(seven)[0] == 7: break
        if rank == 8:
            while True:
                self.reset_deck()
                fig = random.randrange(8)
                col = random.randrange(4)
                seven = [(fig + ix, col) for ix in range(5)]
                for card in seven: self.cards.remove(card)
                seven += [self.get_card() for _ in range(2)]
                if self.cards_rank(seven)[0] == 8: break
        random.shuffle(seven)
        return seven

    # card(str) >> tuple
    @staticmethod
    def _stt(card :str):
        return CF_I[card[0]],CC_I[card[1]]

    # card(any)  >> int
    @staticmethod
    def cti(card :int or tuple or str):
        if type(card) is str: card = PDeck._stt(card) # to tuple
        if type(card) is tuple: return card[0]*4+card[1]
        return card # int case

    # card(any)  >> tuple
    @staticmethod
    def ctt(card :int or tuple or str):
        if type(card) is str: return PDeck._stt(card)
        if type(card) is int: return int(card/4), card%4
        return card # tuple case

    # card(any)  >> str
    @staticmethod
    def cts(card :int or tuple or str):
        if type(card) is int: card = PDeck.ctt(card) # to tuple
        if type(card) is tuple: return CRD_FIG[card[0]] + CRD_COL[card[1]]
        return card # string case

    # returns rank of 5 from 7 given cards (evaluates about 68K * 7cards/sec)
    @staticmethod
    def cards_rank(cards: list):

        # to tuplesL
        if type(cards[0]) is int: cards = [PDeck.ctt(c) for c in cards]
        if type(cards[0]) is str: cards = [PDeck._stt(c) for c in cards]
        cards = sorted(cards)

        # calc possible multiFig and colours
        cFig = [[] for _ in range(13)]
        cCol = [[] for _ in range(4)]
        for c in cards:
            cFig[c[0]].append(c)
            cCol[c[1]].append(c)

        nFig = [len(f) for f in cFig] # multiple figures

        # search for flush
        colCards = None
        colour = None
        for cL in cCol:
            if len(cL) > 4:
                colCards = cL
                colour = cL[0][1]
                break

        scFig = cFig[-1:] + cFig # with aces at the beginning
        inRow = []
        pix = -2
        for ix in range(14):
            if len(scFig[ix]):
                # select card
                crd = scFig[ix][0]
                if len(scFig[ix]) > 1 and colour:
                    for c in scFig[ix]:
                        if c[1] == colour:
                            crd = c
                            break
                if pix + 1 == ix:
                    inRow.append(crd)
                else:
                    if len(inRow) in [3,4]: break # no chance anymore
                    if len(inRow) in [0,1,2]: inRow = [crd] # still a chance
                    else: break # got 5
                pix = ix
        possibleStraight = len(inRow) > 4

        # straightFlush case check
        possibleStraightFlush = False
        if possibleStraight and colCards:

            # remove from row cards out of colour
            colInRow = [] + inRow
            toDelete = []
            for c in colInRow:
                if c[1] != colour: toDelete.append(c)
            for c in toDelete: colInRow.remove(c) # after this col may be not in row (may be split)

            if len(colInRow) > 4:
                possibleStraightFlush = True # assume true

                splitIX = [] # indexes of split from
                for ix in range(1,len(colInRow)):
                    if colInRow[ix-1][0]+1 != colInRow[ix][0]: splitIX.append(ix)

                if splitIX:
                    if len(colInRow)<6 or len(splitIX)>1: possibleStraightFlush = False # any split gives possibility for SF only for 6 cards (7 with one removed from inside/notEdge/ gives 6 with split) with one split
                    else:
                        if splitIX[0] not in [1,5]: possibleStraightFlush = False
                        else:
                            ixF = 0
                            ixT = 5
                            if splitIX[0]==1:
                                ixF = 1
                                ixT = 6
                            colInRow = colInRow[ixF:ixT]

                if len(colInRow) > 5: colInRow = colInRow[len(colInRow)-5:] # trim

        if possibleStraightFlush:                               top_rank = 8 # straightFlush
        elif 4 in nFig:                                         top_rank = 7 # fourOf
        elif (3 in nFig and 2 in nFig) or nFig.count(3) > 1:    top_rank = 6 # fullHouse
        elif colCards:                                          top_rank = 5 # flush
        elif possibleStraight:                                  top_rank = 4 # straight
        elif 3 in nFig:                                         top_rank = 3 # threeOf
        elif nFig.count(2) > 1:                                 top_rank = 2 # twoPairs
        elif 2 in nFig:                                         top_rank = 1 # pair
        else:                                                   top_rank = 0 # highCard

        # find five cards
        five_cards = []
        if top_rank == 8: five_cards = colInRow
        if top_rank == 7:
            four = []
            for cL in cFig:
                if len(cL) == 4:
                    four = cL
                    break
            for c in four: cards.remove(c)
            five_cards = [cards[-1]] + four
        if top_rank == 6:
            five_cards = []
            for cL in cFig:
                if len(cL) == 2: five_cards += cL
            for cL in cFig:
                if len(cL) == 3: five_cards += cL
            five_cards = five_cards[-5:]
        if top_rank == 5:
            if len(colCards) > 5: colCards = colCards[len(colCards)-5:]
            five_cards = colCards
        if top_rank == 4:
            if len(inRow) > 5: inRow = inRow[len(inRow)-5:]
            five_cards = inRow
        if top_rank == 3:
            three = []
            for cL in cFig:
                if len(cL) == 3: three = cL
            for c in three: cards.remove(c)
            five_cards = cards[-2:] + three
        if top_rank == 2:
            two2 = []
            for cL in cFig:
                if len(cL) == 2: two2 += cL
            if len(two2) > 4: two2 = two2[len(two2)-4:]
            for c in two2: cards.remove(c)
            five_cards = cards[-1:] + two2
        if top_rank == 1:
            two = []
            for cL in cFig:
                if len(cL) == 2: two = cL
            for c in two: cards.remove(c)
            five_cards = cards[-3:] + two
        if top_rank == 0:
            five_cards = cards[-5:]

        # calc rank_value
        rank_value = 0
        for ix in range(5): rank_value += five_cards[ix][0]*13**ix
        rank_value += 1000000*top_rank

        # prep string
        string = HND_RNK[top_rank] + ' %7s' % rank_value
        for c in five_cards: string += ' %s' % PDeck.cts(c)

        return top_rank, rank_value, five_cards, string

# a dictionary with rank values of every (tuple of sorted ints (7 cards))
class ASC(dict):

    def __init__(
            self,
            file_FP :str,
            use_QMP=    True):

        super().__init__()

        print('\nReading ASC dict cache...')
        as_cards = lM.r_pickle(file_FP)
        if as_cards: print(' > using cached ASC dict')
        else:
            print(' > cache not found, building all seven cards rank dict...')
            as_cards = {}
            comb_list = list(itertools.combinations([x for x in range(52)], 7))

            if use_QMP:
                def iPF(task):
                    tv = []
                    for t in task: tv.append((t,PDeck.cards_rank(t)[1]))
                    return tv

                qmp = QueMultiProcessor( # QMP
                    proc_func=  iPF,
                    reload=    1000,
                    user_tasks=      True,
                    verb=           1)

                np = 0
                tcmb = []
                for cmb in comb_list:
                    tcmb.append(cmb)
                    if len(tcmb) > 10000:
                        qmp.putTask(tcmb)
                        tcmb = []
                        np += 1
                if tcmb:
                    qmp.putTask(tcmb)
                    np += 1
                for _ in tqdm(range(np)):
                    res = qmp.getResult()
                    for r in res:
                        as_cards[r[0]] = r[1]
                qmp.close()

            else: as_cards = {cmb: PDeck.cards_rank(cmb)[1] for cmb in tqdm(comb_list)}

            lM.w_pickle(as_cards, file_FP)

        self.update(as_cards)

    # returns rank for 7 cards (sorted!)
    def cards_rank(self, c :tuple): return self[c]

# tests speed of ranking
def test_rank_speed(num_ask=123000):

    tdeck = PDeck()
    scL = []
    print('\nPreparing cards...')
    for _ in tqdm(range(num_ask)):
        scL.append([tdeck.get_card() for _ in range(7)])
        tdeck.reset_deck()

    x = int(num_ask/2)
    for c in scL[x:x+10]: print(c)

    s_time = time.time()
    for sc in scL:
        #print(' ', end='')
        #for card in sorted(sc): print(PDeck.cts(card), end=' ')
        #print()

        cR = PDeck.cards_rank(sc)
        #if cR[0]==8: print(cR[-1])
        #print(cR[-1])
    e_time = time.time()

    print('time taken %.2fsec'%(e_time-s_time))
    print('speed %d/sec' % (num_ask/(e_time-s_time)))

# some tests on deck
def test_deck():

    tdeck = PDeck()
    for ix in range(9):
        cards = tdeck.get7of_rank(ix)
        print(ix, cards, PDeck.cards_rank(cards)[-1])

    cards = [(12,0),(0,1),(1,2),(2,0),(3,0),(5,1),(9,2)]
    print(PDeck.cards_rank(cards)[-1])
    cards = [(12,0),(0,1),(1,2),(2,0),(3,0),(4,1),(9,2)]
    print(PDeck.cards_rank(cards)[-1])

    cards = [1,5,8,22,36]
    print(PDeck.cards_rank(cards)[-1])
    cards = [0,1,6,3,45]
    print(PDeck.cards_rank(cards)[-1])

# compares speed of ASC and PDeck
def compare_ranks(num_ask=1000000):

    print('\nPreparing combinations of 7 from 52...',end='')
    comb_list = list(itertools.combinations([x for x in range(52)], 7))
    print(' done!, got %d'%len(comb_list))
    x = 1235143
    for c in comb_list[x:x+10]: print(c)

    ask_cards = [comb_list[random.randint(0, len(comb_list))] for _ in range(num_ask)]

    asc = ASC('_cache/asc.dict')
    s_time = time.time()
    for c in tqdm(ask_cards): res = asc.cards_rank(c)
    print('speed %d/sec' % (num_ask / (time.time() - s_time)))

    s_time = time.time()
    for c in tqdm(ask_cards): res = PDeck.cards_rank(c)
    print('speed %d/sec' % (num_ask / (time.time() - s_time)))


if __name__ == "__main__":

    #test_rank_speed(1234321)

    compare_ranks(1000000)


