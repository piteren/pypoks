"""

 2019 (c) piteren

"""

import random
import time

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
    8:  '10',
    9:  'J',
    10: 'D',
    11: 'K',
    12: 'A'}

# card colors
CRD_COL = {
    0:  'S',    # spades, black (wino)
    1:  'H',    # hearts, red (serce)
    2:  'D',    # diamonds, blue (diament)
    3:  'C'}    # clubs, green (żołądź)

# card rankings
CRD_RNK = {
    0:  'highCard',
    1:  'pair',
    2:  'twoPairs',
    3:  'threeOf',
    4:  'straight',
    5:  'flush',
    6:  'fullHouse',
    7:  'fourOf',
    8:  'straightFlush'}


class PDeck:

    def __init__(self):

        self.cards = None
        self.resetDeck()

    # resets deck to initial state
    def resetDeck(self):
        self.cards = []
        for f in range(13):
            for c in range(4):
                self.cards.append((f, c))
        random.shuffle(self.cards)
        random.shuffle(self.cards)

    # returns one card from deck
    def getCard(self): return self.cards.pop()

    # returns card id (int)
    @staticmethod
    def cti(card: tuple): return card[0]*4+card[1]

    # retund card str
    @staticmethod
    def cts(card: tuple): return CRD_FIG[card[0]] + CRD_COL[card[1]]

    # returns rank of 5 from 7 given cards
    # simple implementation evaluates about 12,3K*7cards/sec
    @staticmethod
    def cardsRank(cards: list):
        cards = sorted(cards)

        # calc possible multiFig and colours
        cFig = [[] for _ in range(13)]
        cCol = [[] for _ in range(4)]
        for c in cards:
            cFig[c[0]].append(c)
            cCol[c[1]].append(c)

        nFig = [len(f) for f in cFig]  # multiple figures

        # search for flush
        colCards = None
        colour = None
        for cL in cCol:
            if len(cL) > 4:
                colCards = cL
                colour = cL[0][1]
                break

        inRow = []
        pix = -2
        for ix in range(13):
            if len(cFig[ix]):
                # select card
                c = cFig[ix][0]
                if len(cFig[ix]) > 1 and colour:
                    for c in cFig[ix]:
                        if c[1] == colour:
                            c = c
                            break
                if pix + 1 == ix:
                    inRow.append(c)
                else:
                    if len(inRow) in [3,4]: break # no chance anymore
                    if len(inRow) in [0,1,2]: inRow = [c] # still a chance
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

        if possibleStraightFlush:       topRank = 8 # straightFlush
        elif 4 in nFig:                 topRank = 7 # fourOf
        elif 3 in nFig and 2 in nFig:   topRank = 6 # fullHouse
        elif colCards:                  topRank = 5 # flush
        elif possibleStraight:          topRank = 4 # straight
        elif 3 in nFig:                 topRank = 3 # threeOf
        elif nFig.count(2) > 1:         topRank = 2 # twoPairs
        elif 2 in nFig:                 topRank = 1 # pair
        else:                           topRank = 0 # highCard

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
            three = []
            for cL in cFig:
                if len(cL) == 3: three = cL
            two = []
            for cL in cFig:
                if len(cL) == 2: two = cL
            fiveCards = two + three
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
        string = CRD_RNK[topRank] + ' %s'%rankValue
        for c in fiveCards:
            string += ' %s' % PDeck.cts(c)

        return topRank, rankValue, fiveCards, string


if __name__ == "__main__":

    testDeck = PDeck()

    sTime = time.time()
    for _ in range(123000):
        sevenCards = [testDeck.getCard() for _ in range(7)]
        """
        print(' ', end='')
        for card in sorted(sevenCards):
            print(PDeck.cts(card), end=' ')
        print()
        #"""
        cR = PDeck.cardsRank(sevenCards)
        #if cR[0]==8: print(cR[-1])
        #print(cR[-1])
        testDeck.resetDeck()
    print('time taken %.2fsec'%(time.time()-sTime))

