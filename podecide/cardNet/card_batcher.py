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

import ptools.lipytools.little_methods as lim

from pologic.podeck import PDeck, ASC


# prepares batch of 2x 7cards with regression, MP ready
def prep2X7Batch(
        task=       None,   # needed by QMP, here passed avoid_tuples - list of sorted_card tuples (from test batch) to avoid in batch
        bs=         1000,   # batch size
        r_balance=  True,   # rank balance (forces to balance ranks)
        d_balance=  0.1,    # draw balance (False or fraction of draws)
        no_maskP=   None,   # probability of not masking (all cards are known), for None uses full random
        n_monte=    30,     # num of montecarlo samples for A win chance estimation
        asc: dict=  None,
        verb=       0):

    deck = PDeck() # since it is hard to give any object to method of subprocess...
    avoid_tuples = task

    b_cA, b_cB, b_wins, b_rA, b_rB, b_mAWP = [],[],[],[],[],[] # batches
    r_num = [0]*9 # ranks counter
    w_num = [0]*3 # wins counter

    iter = range(bs)
    if verb: iter = tqdm(iter)
    for _ in iter:
        deck.reset_deck()

        # look 4 the last freq rank
        n_min_rank = min(r_num)
        desired_rank = r_num.index(n_min_rank)

        desired_draw = False if d_balance is False else w_num[2] < d_balance * sum(w_num)

        cA = None           # seven cards of A
        cB = None           # seven cards of B
        rA = None           # rank A
        rB = None           # rank B
        got_allC = False    # got all (proper) cards
        while not got_allC:

            cA = deck.get7of_rank(desired_rank) if r_balance else [deck.get_card() for _ in range(7)] # 7 cards for A
            cB = [deck.get_card() for _ in range(2)] + cA[2:] # 2+5 cards for B

            # randomly swap hands of A and B (to avoid win bias)
            if random.random() > 0.5:
                temp = cA
                cA = cB
                cB = temp

            # get cards ranks
            rA = deck.cards_rank(cA)
            rB = deck.cards_rank(cB)

            rAV = rA[1]
            rBV = rB[1]
            rA = rA[0]
            rB = rB[0]

            if not desired_draw or (desired_draw and rAV==rBV): got_allC = True
            if got_allC and type(avoid_tuples) is list and (tuple(sorted(cA)) in avoid_tuples or tuple(sorted(cB)) in avoid_tuples): got_allC = False

        r_num[rA]+=1
        r_num[rB]+=1

        diff = rAV-rBV
        wins = 0 if diff>0 else 1
        if diff==0: wins = 2 # a draw
        w_num[wins] += 1

        # convert cards tuples to ints
        cA = [PDeck.cti(c) for c in cA]
        cB = [PDeck.cti(c) for c in cB]

        # mask some table cards
        nMask = [0]
        if no_maskP is None: nMask = [5,2,1,0]
        elif random.random() > no_maskP: nMask = [5,2,1]
        random.shuffle(nMask)
        nMask = nMask[0]
        for ix in range(2+5-nMask,7): cA[ix] = 52

        # estimate A win chance with montecarlo
        n_wins = 0
        if diff > 0: n_wins = 1
        if diff == 0: n_wins = 0.5
        if n_monte > 0:
            got_cards = {c for c in cA if c!=52} # set
            for it in range(n_monte):
                nine_cards = got_cards.copy()
                while len(nine_cards) < 9: nine_cards.add(int(52 * random.random())) # much faster than random.randrange(52), use numpy.random.randint(52, size=10...) for more
                nine_cards = sorted(nine_cards)
                if asc:
                    aR = asc[tuple(nine_cards[:7])]
                    bR = asc[tuple(nine_cards[2:])]
                else:
                    aR = PDeck.cards_rank(nine_cards[:7])[1]
                    bR = PDeck.cards_rank(nine_cards[2:])[1]
                if aR >  bR: n_wins += 1
                if aR == bR: n_wins += 0.5
        mcAChance = n_wins / (n_monte + 1)

        b_cA.append(cA)             # 7 cards of A
        b_cB.append(cB)             # 7 cards of B
        b_wins.append(wins)         # who wins {0,1,2}
        b_rA.append(rA)             # rank of A
        b_rB.append(rB)             # rank ok B
        b_mAWP.append(mcAChance)    # chances for A

    return {
        'cA':       b_cA,
        'cB':       b_cB,
        'wins':     b_wins,
        'rA':       b_rA,
        'rB':       b_rB,
        'mAWP':     b_mAWP,
        'r_num':    r_num,
        'w_num':    w_num}

# prepares tests batch
def get_test_batch(
        size :int,          # batch size
        mcs :int,           # n montecarlo samples
        with_ASC=    True): # with all seven cards (dict)

    fn = '_cache/s%d_m%d.batch'%(size,mcs)
    test_batch = lim.r_pickle(fn)
    if test_batch: print('\nGot test batch from file: %s'%fn)
    else:
        print('\nPreparing test batch (%d,%d)...'%(size,mcs))
        test_batch = prep2X7Batch(
            bs=         size,
            n_monte=    mcs,
            asc=        ASC('_cache/asc.dict') if with_ASC else None,
            verb=       1)
        lim.w_pickle(test_batch, fn)
    c_tuples = []
    for ix in range(size):
        c_tuples.append(tuple(sorted(test_batch['cA'][ix])))
        c_tuples.append(tuple(sorted(test_batch['cB'][ix])))
    print('Got %d of hands in test_batch' % len(c_tuples))
    c_tuples = dict.fromkeys(c_tuples, 1)
    print('of which %d is unique' % len(c_tuples))

    return test_batch, c_tuples


if __name__ == "__main__":

    """
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
    """
    get_test_batch(2000,100000)
    #get_test_batch(2000,10000000)