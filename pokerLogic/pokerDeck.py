"""

 2019 (c) piteren

"""

# TODO:
#  - finalize cardsRank

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


class PokerDeck:

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

    # returns card in form of int
    @staticmethod
    def cardToInt(card: tuple): return card[0]*4+card[1]

    # retund card in form of str
    @staticmethod
    def cardToStr(card: tuple): return CRD_FIG[card[0]] + CRD_COL[card[1]]

    # returns rank of 5 from 7 given cards
    @staticmethod
    def cardsRank(cards: list):

        cards = sorted(cards)

        cRank = [False for _ in range(9)]
        cRank[0] = True # for sure highCard

        # calc possible multiFig and colours
        cFig = [[] for _ in range(13)]
        cCol = [[] for _ in range(4)]
        for card in cards:
            cFig[card[0]].append(card)
            cCol[card[1]].append(card)

        colCards = None
        for cards in cCol:
            if len(cards) > 4: colCards = cards

        nFig = [len(f) for f in cFig] # multiple figures

        nInRow = 0
        pix = -2
        for ix in range(13):
            if len(cFig[ix]):
                if pix + 1 == ix:
                    nInRow += 1
                else:
                    nInRow = 1
                pix = ix
        possibleStraight = nInRow > 4

        if 2 in nFig:               cRank[1] = True # pair
        if nFig.count(2) > 1:       cRank[2] = True # twoPairs
        if 3 in nFig:               cRank[3] = True # threeOf
        if possibleStraight:        cRank[4] = True # straight
        if colCards:                cRank[5] = True # flush
        if 3 in nFig and 2 in nFig: cRank[6] = True # fullHouse
        if 4 in nFig:               cRank[7] = True # fourOf

        # straightFlush case check
        if cRank[4] and cRank[5]:

            ccFig = [[] for _ in range(13)]
            for card in colCards: ccFig[card[0]].append(card)

            nInRow = 0
            pix = -2
            for ix in range(13):
                if len(ccFig[ix]):
                    if pix + 1 == ix:
                        nInRow += 1
                    else:
                        nInRow = 1
                    pix = ix
            possibleStraightFlush = nInRow > 4
            if possibleStraightFlush: cRank[8] = True # straightFlush

        # calc topRank
        topRank = 0
        for ix in reversed(range(9)):
            if cRank[ix]:
                topRank = ix
                break

        # calc topRank value
        if topRank == 8:
            pass

        return topRank


if __name__ == "__main__":

    testDeck = PokerDeck()

    for _ in range(10):
        sevenCards = [testDeck.getCard() for _ in range(7)]
        for card in sevenCards:
            print(PokerDeck.cardToStr(card), end=' ')
        print()
        print(PokerDeck.cardsRank(sevenCards))
        testDeck.resetDeck()


