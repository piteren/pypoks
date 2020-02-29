"""

 2019 (c) piteren

"""

import itertools
import random
import time
from tqdm import tqdm

import putils.lipytools.little_methods as lM
from putils.que_MProcessor import QueMultiProcessor

# card figures
CRD_FIG = {
    0:  '2',
    1:  '3',
    2:  '4',
    3:  '5',
    4:  '6',
    5:  '7',
    6:  '8',
    7:  '9',
    8:  'T',
    9:  'J',
    10: 'D',
    11: 'K',
    12: 'A',
    13: 'X'}

# card colors
CRD_COL = {
    0:  'S',    # spades, black (wino)
    1:  'H',    # hearts, red (serce)
    2:  'D',    # diamonds, blue (diament)
    3:  'C'}    # clubs, green (żołądź)

# card ranks
CRD_RNK = {
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

        self.__fullInitDeck = [PDeck.itc(ci) for ci in range(52)]
        self.cards = None
        self.resetDeck()

    # resets deck to initial state
    def resetDeck(self):
        self.cards = [] + self.__fullInitDeck
        random.shuffle(self.cards)

    # returns one card from deck
    def getCard(self): return self.cards.pop()

    # returns exact card from deck, id not present return None
    def getECard(self, card: tuple or int):
        if type(card) is int: card = PDeck.itc(card)
        if card in self.cards:
            self.cards.remove(card)
            return card
        return None

    # returns seven card of given rank
    def get7ofRank(self, rank :int):

        seven = []
        if rank == 0:
            while True:
                self.resetDeck()
                seven = [self.getCard() for _ in range(7)]
                if self.cardsRank(seven)[0] == 0: break
        if rank == 1:
            while True:
                self.resetDeck()
                seven = [self.getCard() for _ in range(7)]
                if self.cardsRank(seven)[0] == 1: break
        if rank == 2:
            while True:
                self.resetDeck()
                seven = [self.getCard() for _ in range(7)]
                if self.cardsRank(seven)[0] == 2: break
        if rank == 3:
            while True:
                self.resetDeck()
                fig = random.randrange(12)
                col = [c for c in range(4)]
                random.shuffle(col)
                col = col[:-1]
                seven = [(fig,c) for c in col]
                for card in seven: self.cards.remove(card)
                seven += [self.getCard() for _ in range(4)]
                if self.cardsRank(seven)[0] == 3: break
        if rank == 4:
            while True:
                self.resetDeck()
                fig = random.randrange(8)
                seven = [(fig+ix,random.randrange(4)) for ix in range(5)]
                for card in seven: self.cards.remove(card)
                seven += [self.getCard() for _ in range(2)]
                if self.cardsRank(seven)[0] == 4: break
        if rank == 5:
            while True:
                self.resetDeck()
                col = random.randrange(4)
                fig = [f for f in range(12)]
                random.shuffle(fig)
                fig = fig[:5]
                seven = [(f,col) for f in fig]
                for card in seven: self.cards.remove(card)
                seven += [self.getCard() for _ in range(2)]
                if self.cardsRank(seven)[0] == 5: break
        if rank == 6:
            while True:
                self.resetDeck()
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
                seven += [self.getCard() for _ in range(2)]
                if self.cardsRank(seven)[0] == 6: break
        if rank == 7:
            while True:
                self.resetDeck()
                fig = random.randrange(12)
                seven = [(fig,c) for c in range(4)]
                for card in seven: self.cards.remove(card)
                seven += [self.getCard() for _ in range(3)]
                if self.cardsRank(seven)[0] == 7: break
        if rank == 8:
            while True:
                self.resetDeck()
                fig = random.randrange(8)
                col = random.randrange(4)
                seven = [(fig + ix, col) for ix in range(5)]
                for card in seven: self.cards.remove(card)
                seven += [self.getCard() for _ in range(2)]
                if self.cardsRank(seven)[0] == 8: break
        random.shuffle(seven)
        return seven

    # returns card id (int)
    @staticmethod
    def cti(card: tuple): return card[0]*4+card[1]

    # returns card tuple
    @staticmethod
    def itc(card: int):
        cf = int(card/4)
        cc = card%4
        return cf,cc

    # returns card str
    @staticmethod
    def cts(card: tuple or int):
        if type(card) is int: card = PDeck.itc(card)
        return CRD_FIG[card[0]] + CRD_COL[card[1]]

    # returns rank of 5 from 7 given cards
    # simple implementation evaluates about:  12,3K*7cards/sec   or  66K*5cards/sec
    @staticmethod
    def cardsRank(cards: list):

        if type(cards[0]) is int: cards = [PDeck.itc(c) for c in cards]
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

        if possibleStraightFlush:                               topRank = 8 # straightFlush
        elif 4 in nFig:                                         topRank = 7 # fourOf
        elif (3 in nFig and 2 in nFig) or nFig.count(3) > 1:    topRank = 6 # fullHouse
        elif colCards:                                          topRank = 5 # flush
        elif possibleStraight:                                  topRank = 4 # straight
        elif 3 in nFig:                                         topRank = 3 # threeOf
        elif nFig.count(2) > 1:                                 topRank = 2 # twoPairs
        elif 2 in nFig:                                         topRank = 1 # pair
        else:                                                   topRank = 0 # highCard

        # find five cards
        fiveCards = []
        if topRank == 8: fiveCards = colInRow
        if topRank == 7:
            four = []
            for cL in cFig:
                if len(cL) == 4:
                    four = cL
                    break
            for c in four: cards.remove(c)
            fiveCards = [cards[-1]] + four
        if topRank == 6:
            fiveCards = []
            for cL in cFig:
                if len(cL) == 2: fiveCards += cL
            for cL in cFig:
                if len(cL) == 3: fiveCards += cL
            fiveCards = fiveCards[-5:]
        if topRank == 5:
            if len(colCards) > 5: colCards = colCards[len(colCards)-5:]
            fiveCards = colCards
        if topRank == 4:
            if len(inRow) > 5: inRow = inRow[len(inRow)-5:]
            fiveCards = inRow
        if topRank == 3:
            three = []
            for cL in cFig:
                if len(cL) == 3: three = cL
            for c in three: cards.remove(c)
            fiveCards = cards[-2:] + three
        if topRank == 2:
            two2 = []
            for cL in cFig:
                if len(cL) == 2: two2 += cL
            if len(two2) > 4: two2 = two2[len(two2)-4:]
            for c in two2: cards.remove(c)
            fiveCards = cards[-1:] + two2
        if topRank == 1:
            two = []
            for cL in cFig:
                if len(cL) == 2: two = cL
            for c in two: cards.remove(c)
            fiveCards = cards[-3:] + two
        if topRank == 0:
            fiveCards = cards[-5:]

        # calc rankValue
        rankValue = 0
        for ix in range(5): rankValue += fiveCards[ix][0]*13**ix
        rankValue += 1000000*topRank

        # prep string
        string = CRD_RNK[topRank] + ' %7s'%rankValue
        for c in fiveCards: string += ' %s' % PDeck.cts(c)

        return topRank, rankValue, fiveCards, string

# returns a dictionary with rank values of every (tuple of sorted ints) 5 cards
def getASC(useQMP=True):

    print('\nReading asc dict cache...')
    pickleDictFN = '_cache/asc.dict'
    pickle = lM.r_pickle(pickleDictFN)
    if pickle:
        asCards = pickle
        print(' > using cached asc dict')
    else:
        print(' > cache not found, building all seven cards rank dict...')
        asCards = {}
        combList = list(itertools.combinations([x for x in range(52)], 7))

        if useQMP:
            def iPF(task):
                tv = []
                for t in task: tv.append((t,PDeck.cardsRank(t)[1]))
                return tv

            qmp = QueMultiProcessor( # QMP
                iProcFunction=  iPF,
                reloadEvery=    1000,
                userTasks=      True,
                verb=        1)

            np = 0
            tcmb = []
            for cmb in combList:
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
                    asCards[r[0]] = r[1]
            qmp.close()

        else: asCards = {cmb: PDeck.cardsRank(cmb)[1] for cmb in tqdm(combList)}

        lM.w_pickle(asCards, pickleDictFN)

    return asCards


if __name__ == "__main__":

    """
    testDeck = PDeck()
    sTime = time.time()
    for _ in range(123000):
        sevenCards = [testDeck.getCard() for _ in range(7)]
        
        print(' ', end='')
        for card in sorted(sevenCards):
            print(PDeck.cts(card), end=' ')
        print()
        
        cR = PDeck.cardsRank(sevenCards)
        #if cR[0]==8: print(cR[-1])
        #print(cR[-1])
        testDeck.resetDeck()
    print('time taken %.2fsec'%(time.time()-sTime))
    """
    """
    for ix in range(9):
        cards = testDeck.get7ofRank(ix)
        print(ix, cards, PDeck.cardsRank(cards)[-1])
    """
    """
    cards = [(12,0),(0,1),(1,2),(2,0),(3,0),(5,1),(9,2)]
    print(PDeck.cardsRank(cards)[-1])
    cards = [(12,0),(0,1),(1,2),(2,0),(3,0),(4,1),(9,2)]
    print(PDeck.cardsRank(cards)[-1])
    """
    """
    cards = [1,5,8,22,36]
    print(PDeck.cardsRank(cards)[-1])
    cards = [0,1,6,3,45]
    print(PDeck.cardsRank(cards)[-1])
    """

    num = 10000000
    combList = list(itertools.combinations([x for x in range(52)], 7))
    asc = getASC()
    askCards = [combList[random.randint(0,len(combList))] for _ in range(num)]

    sTime = time.time()
    for c in tqdm(askCards): res = asc[c]
    print('speed %d/sec' % (num / (time.time() - sTime)))

    sTime = time.time()
    for c in tqdm(askCards): res = PDeck.cardsRank(c)
    print('speed %d/sec' % (num / (time.time() - sTime)))


