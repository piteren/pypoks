"""

 2019 (c) piteren

"""

import random

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
    def cardToInt(card: tuple): return card[0]*4+card[1]

    # retund card str
    @staticmethod
    def cardToStr(card: tuple): return CRD_FIG[card[0]] + CRD_COL[card[1]]

    # returns rank of 5 from 7 given cards
    @staticmethod
    def cardsRank(cards: list):

        cards = sorted(cards)

        # calc possible multiFig and colours
        cFig = [[] for _ in range(13)]
        cCol = [[] for _ in range(4)]
        for card in cards:
            cFig[card[0]].append(card)
            cCol[card[1]].append(card)

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
                card = cFig[ix][0]
                if len(cFig[ix]) > 1 and colour:
                    for c in cFig[ix]:
                        if c[1] == colour:
                            card = c
                            break
                if pix + 1 == ix:
                    inRow.append(card)
                else:
                    if len(inRow) in [3,4]: break # no chance anymore
                    if len(inRow) in [0,1,2]: inRow = [card] # still a chance
                    else: break # got 5
                pix = ix
        possibleStraight = len(inRow) > 4

        # straightFlush case check
        possibleStraightFlush = False
        colInRow = [] + inRow
        if possibleStraight and colCards:

            toDelete = []
            for c in colInRow:
                if c[1] != colour: toDelete.append(c)
            for c in toDelete: colInRow.remove(c)

            if len(colInRow) > 4:
                possibleStraightFlush = True
                splitIX = [] # indexes of split from
                for ix in range(1,len(colInRow)):
                    if colInRow[ix-1][0]+1 != colInRow[ix][0]: splitIX.append(ix)
                if splitIX and len(colInRow)-len(splitIX) > 4:
                    splitIX = [0]+splitIX+[len(colInRow)]
                    max = 0
                    ixFrom = 0
                    for ix in range(len(splitIX)-1):
                        diff = splitIX[ix+1]-splitIX[ix]
                        if diff > max:
                            max = diff
                            ixFrom = ix
                    if max < 5: possibleStraightFlush = False
                    if possibleStraightFlush: colInRow = colInRow[splitIX[ixFrom]:splitIX[ixFrom+1]]
                if len(colInRow) > 5: colInRow = colInRow[len(colInRow)-5:]

        cRank = [False for _ in range(9)]
        if True:                    cRank[0] = True # highCard
        if 2 in nFig:               cRank[1] = True # pair
        if nFig.count(2) > 1:       cRank[2] = True # twoPairs
        if 3 in nFig:               cRank[3] = True # threeOf
        if possibleStraight:        cRank[4] = True # straight
        if colCards:                cRank[5] = True # flush
        if 3 in nFig and 2 in nFig: cRank[6] = True # fullHouse
        if 4 in nFig:               cRank[7] = True # fourOf
        if possibleStraightFlush:   cRank[8] = True # straightFlush

        # find topRank
        topRank = 0
        for ix in reversed(range(9)):
            if cRank[ix]:
                topRank = ix
                break

        # find five cards
        fiveCards = []
        if topRank == 8: fiveCards = colInRow
        if topRank == 7:
            four = []
            for cL in cFig:
                if len(cL) == 4: four = cL
            for c in four: cards.remove(c)
            fiveCards = cards[-1] + four
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

        # prep string
        string = CRD_RNK[topRank] + ' %s'%rankValue
        for card in fiveCards:
            string += ' %s' % PDeck.cardToStr(card)

        return topRank, rankValue, fiveCards, string


if __name__ == "__main__":

    testDeck = PDeck()

    for _ in range(10):
        sevenCards = [testDeck.getCard() for _ in range(7)]
        print(' ', end='')
        for card in sorted(sevenCards):
            print(PDeck.cardToStr(card), end=' ')
        print()
        cR = PDeck.cardsRank(sevenCards)
        print(cR[-1])
        testDeck.resetDeck()
