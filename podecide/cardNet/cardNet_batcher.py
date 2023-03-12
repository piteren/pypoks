import random
from tqdm import tqdm
from typing import Dict, Optional

from pypaq.lipytools.files import r_pickle, w_pickle
from pypaq.lipytools.pylogger import get_pylogger

from pologic.podeck import PDeck, ASC


# prepares batch of 2x 7cards with regression, MP ready
def prep2X7batch(
        deck: Optional[PDeck]=  None,
        task=                   None,   # needed by QMP, here passed avoid_tuples - list of sorted_card tuples (from test batch) to avoid in batch
        batch_size=             1000,   # batch size
        r_balance=              True,   # rank balance (forces to balance ranks)
        d_balance=              0.1,    # draw balance (False or fraction of draws)
        no_maskP=               None,   # probability of not masking (all cards are known), for None uses full random
        n_monte=                30,     # num of montecarlo samples for A win chance estimation
        asc: dict=              None,
        verbosity=              0
) -> Dict[str,list]:

    if not deck:
        deck = PDeck()

    avoid_tuples = task

    b_cA, b_cB, b_wins, b_rA, b_rB, b_mAWP = [],[],[],[],[],[] # batches
    rank_counter = [0]*9
    won_counter =  [0]*3

    iter = range(batch_size)
    if verbosity: iter = tqdm(iter)
    for _ in iter:
        deck.reset()

        # look 4 the last freq rank
        n_min_rank = min(rank_counter)
        desired_rank = rank_counter.index(n_min_rank)

        desired_draw = False if d_balance is False else won_counter[2] < d_balance * sum(won_counter)

        cards_7A = None     # seven cards of A
        cards_7B = None     # seven cards of B
        rank_A = None
        rank_B = None
        rank_A_value = 0
        rank_B_value = 0

        got_all_cards = False  # got all (proper) cards
        while not got_all_cards:

            cards_7A = deck.get7of_rank(desired_rank) if r_balance else [deck.get_card() for _ in range(7)] # 7 cards for A
            cards_7B = [deck.get_card() for _ in range(2)] + cards_7A[2:] # 2+5 cards for B

            # randomly swap hands of A and B (to avoid win bias)
            if random.random() > 0.5:
                temp = cards_7A
                cards_7A = cards_7B
                cards_7B = temp

            # get cards ranks
            rank_A, rank_A_value, _, _ = deck.cards_rank(cards_7A)
            rank_B, rank_B_value, _, _ = deck.cards_rank(cards_7B)

            if not desired_draw or (desired_draw and rank_A_value==rank_B_value):
                got_all_cards = True
            if got_all_cards and type(avoid_tuples) is list and (tuple(sorted(cards_7A)) in avoid_tuples or tuple(sorted(cards_7B)) in avoid_tuples):
                got_all_cards = False

        rank_counter[rank_A]+=1
        rank_counter[rank_B]+=1

        diff = rank_A_value-rank_B_value
        wins = 0 if diff>0 else 1
        if diff==0: wins = 2 # a draw
        won_counter[wins] += 1

        # convert cards tuples to ints
        cards_7A = [PDeck.cti(c) for c in cards_7A]
        cards_7B = [PDeck.cti(c) for c in cards_7B]

        # mask some table cards
        nMask = [0]
        if no_maskP is None: nMask = [5,2,1,0]
        elif random.random() > no_maskP: nMask = [5,2,1]
        random.shuffle(nMask)
        nMask = nMask[0]
        for ix in range(2+5-nMask,7): cards_7A[ix] = 52

        # estimate A win chance with montecarlo
        n_wins = 0
        if diff > 0: n_wins = 1
        if diff == 0: n_wins = 0.5
        if n_monte > 0:
            got_cards = {c for c in cards_7A if c!=52} # set
            for it in range(n_monte):
                nine_cards = got_cards.copy()
                while len(nine_cards) < 9: nine_cards.add(int(52 * random.random())) # much faster than random.randrange(52), use numpy.random.randint(52, size=10..) for more
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

        b_cA.append(cards_7A)             # 7 cards of A
        b_cB.append(cards_7B)             # 7 cards of B
        b_wins.append(wins)         # who wins {0,1,2}
        b_rA.append(rank_A)             # rank of A
        b_rB.append(rank_B)             # rank ok B
        b_mAWP.append(mcAChance)    # win chances for A

    return {
        'cards_A':      b_cA,
        'cards_B':      b_cB,
        'label_won':    b_wins,
        'label_rank_A': b_rA,
        'label_rank_B': b_rB,
        'prob_won_A':   b_mAWP,
        'rank_counter': rank_counter,
        'won_counter':  won_counter}

# prepares tests batch
def get_test_batch(
        size :int,         # batch size
        mcs :int,          # n montecarlo samples
        use_ASC=    True,  # with all seven cards (dict)
        logger=     None):

    if not logger: logger = get_pylogger()

    fn = f'_cache/s{size}_m{mcs}.batch'
    test_batch = r_pickle(fn)
    if test_batch:
        logger.info(f'got test batch from file: {fn}')
    else:
        logger.info(f'preparing test batch ({size},{mcs})..')
        test_batch = prep2X7batch(
            deck=       PDeck(),
            batch_size= size,
            n_monte=    mcs,
            asc=        ASC('_cache/asc.dict') if use_ASC else None,
            verbosity=  1)
        w_pickle(test_batch, fn)
    c_tuples = []
    for ix in range(size):
        c_tuples.append(tuple(sorted(test_batch['cards_A'][ix])))
        c_tuples.append(tuple(sorted(test_batch['cards_B'][ix])))
    logger.info(f'got {len(c_tuples)} of hands in test_batch, of which {len(set(c_tuples))} is unique')
    return test_batch, c_tuples


if __name__ == "__main__":

    #get_test_batch(2000, 10000)
     get_test_batch(2000, 100000)
    #get_test_batch(2000, 10000000)