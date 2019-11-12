"""

 2019 (c) piteren

 there are 2,598,960 hands not considering different suits it is (52/5) - combination

 stats of poks hands, for 5 (not 7) randomly taken:
 0 (highCard)       - 50.1177%
 1 (pair)           - 42.2569%
 2 (2pairs)         -  4.7539%
 3 (threeOf)        -  2.1128%
 4 (straight)       -  0.3925%
 5 (flush)          -  0.1965%
 6 (FH)             -  0.1441%
 7 (fourOf)         -  0.0240%
 8 (straightFlush)  -  0.001544%

 for seven cards (from 0 to 8):
 0.17740 0.44400 0.23040 0.04785 0.04150 0.03060 0.02660 0.00145 0.00020 (fraction)
 0.17740 0.62140 0.85180 0.89965 0.94115 0.97175 0.99835 0.99980 1.00000 (cumulative fraction)

 for seven cards, when try_to_balance it (first attempt) (from 0 to 8):
 0.11485 0.22095 0.16230 0.10895 0.08660 0.09515 0.08695 0.06490 0.05935
 0.11485 0.33580 0.49810 0.60705 0.69365 0.78880 0.87575 0.94065 1.00000

"""

import random
from tqdm import tqdm

from pLogic.pDeck import PDeck


# prepares batch of 2x 7cards with regression, MP ready
def prep2X7Batch(
        task=       None,   # needed by QMP, here passed avoidCTuples - list of sorted_card tuples to avoid in batch
        bs=         1000,   # batch size
        rBalance=   True,   # balance rank
        dBalance=   0.1,    # False or fraction of draws
        #probNoMask= 1.0,    # probability of not masking (all cards are known), for None uses full random
        probNoMask= None,
        #nMonte=     0):     # num of MonteCarlo runs for A estimation
        nMonte=     5,
        verbLev=    0):

    deck = PDeck() # since it is hard to give any object to function of process...
    avoidCTuples = task

    crd7AB, crd7BB, winsB, rankAB, rankBB, mcAChanceB = [],[],[],[],[],[] # batches
    numRanks = [0]*9
    numWins = [0]*3
    hS = ['']*9

    iter = range(bs)
    if verbLev: iter = tqdm(iter)
    for s in iter:
        deck.resetDeck()

        # look 4 the smallest number rank
        nMinRank = min(numRanks)
        desiredRank = numRanks.index(nMinRank)
        desiredDraw = False if dBalance is False else numWins[2] < dBalance * sum(numWins)

        crd7A = None
        crd7B = None
        aRank = None
        bRank = None
        gotDesiredCards = False
        while not gotDesiredCards:

            crd7A = deck.get7ofRank(desiredRank) if rBalance else [deck.getCard() for _ in range(7)] # 7 cards for A
            crd7B = [deck.getCard() for _ in range(2)] + crd7A[2:] # 2+5 cards for B

            # randomly swap hands of A and B (to avoid win bias)
            if random.random() > 0.5:
                temp = crd7A
                crd7A = crd7B
                crd7B = temp

            # get cards ranks
            aRank = deck.cardsRank(crd7A)
            bRank = deck.cardsRank(crd7B)

            if not desiredDraw or (desiredDraw and aRank[1]==bRank[1]): gotDesiredCards = True
            if gotDesiredCards and type(avoidCTuples) is list and (tuple(sorted(crd7A)) in avoidCTuples or tuple(sorted(crd7B)) in avoidCTuples): gotDesiredCards = False

        paCRV = aRank[1]
        pbCRV = bRank[1]
        ha = aRank[0]
        hb = bRank[0]
        numRanks[aRank[0]]+=1
        numRanks[bRank[0]]+=1
        hS[aRank[0]] = aRank[-1]
        hS[bRank[0]] = bRank[-1]
        diff = paCRV-pbCRV
        wins = 0 if diff>0 else 1
        if diff==0: wins = 2 # a draw
        numWins[wins] += 1

        # convert cards tuples to ints
        crd7A = [PDeck.cti(c) for c in crd7A]
        crd7B = [PDeck.cti(c) for c in crd7B]

        # mask some table cards
        nMask = [0]
        if probNoMask is None: nMask = [5,2,1,0]
        elif random.random() > probNoMask: nMask = [5,2,1]
        random.shuffle(nMask)
        nMask = nMask[0]
        for ix in range(2+5-nMask,7): crd7A[ix] = 52

        # calc MontCarlo chances of winning for A
        nAWins = 0
        if diff > 0: nAWins = 1
        if diff == 0: nAWins = 0.5
        if nMonte > 0:
            gotCards = [c for c in crd7A if c!=52]
            for it in range(nMonte):
                nineCards = [] + gotCards
                newDeck = PDeck()
                for c in gotCards: newDeck.getECard(c)
                while len(nineCards) < 9: nineCards.append(PDeck.cti(newDeck.getCard()))
                a7c = nineCards[:7]
                b7c = nineCards[2:]
                aR = PDeck.cardsRank(a7c)
                bR = PDeck.cardsRank(b7c)
                if aR[1] > bR[1]: nAWins += 1
                if aR[1] == bR[1]: nAWins += 0.5
        mcAChance = nAWins / (nMonte+1)

        crd7AB.append(crd7A)            # 7 cards of A
        crd7BB.append(crd7B)            # 7 cards of B
        winsB.append(wins)              # who wins {0,1,2}
        rankAB.append(ha)               # rank of A
        rankBB.append(hb)               # rank ok B
        mcAChanceB.append(mcAChance)    # chances for A

    return {
        'crd7AB':       crd7AB,
        'crd7BB':       crd7BB,
        'winsB':        winsB,
        'rankAB':       rankAB,
        'rankBB':       rankBB,
        'mcAChanceB':   mcAChanceB,
        'numRanks':     numRanks,
        'numWins':      numWins}

if __name__ == "__main__":

    batch = prep2X7Batch(
        bs=         10,
        probNoMask= 0.5,
        nMonte=     100)
    crd7A = batch['crd7AB']
    mcAC = batch['mcAChanceB']

    for ix in range(len(crd7A)):
        for c in crd7A[ix]:
            if c!=52: print(PDeck.cts(c), end=' ')
        print(mcAC[ix])