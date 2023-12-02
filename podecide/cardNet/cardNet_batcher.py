from ompr.runner import RunningWorker, OMPRunner
from pypaq.lipytools.files import r_pickle, w_pickle
from pypaq.lipytools.pylogger import get_pylogger
import random
from tqdm import tqdm
from typing import Dict, Optional, List

from envy import CACHE_FD
from pologic.podeck import PDeck, ASC, monte_carlo_prob_won


def prep2X7batch(
        deck: Optional[PDeck]=      None,
        task=                       None,   # needed by OMPR, here passed avoid_tuples - list of sorted_card tuples (from test batch) to avoid in batch
        batch_size=                 1000,   # batch size
        r_balance=                  True,   # rank balance (forces to balance ranks)
        d_balance=                  0.1,    # draw balance (False or fraction of draws)
        no_maskP: Optional[float]=  None,   # probability of not masking (all cards are known), for None uses full random
        n_monte=                    30,     # num of montecarlo samples for A win chance estimation
        asc: ASC=                   None,
        verbosity=                  0
) -> Dict[str, List]:
    """ prepares batch of 2x 7cards
    MP ready """

    if not deck:
        deck = PDeck()

    avoid_tuples = task

    b_cA, b_cB, b_wins, b_rA, b_rB, b_mAWP = [],[],[],[],[],[] # batches
    rank_counter = [0]*9
    won_counter =  [0]*3

    it = range(batch_size)
    for _ in tqdm(it) if verbosity else it:
        deck.reset()

        # look for the last freq rank
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

            cards_7A = deck.get_7of_rank(desired_rank) if r_balance else [deck.get_card() for _ in range(7)] # 7 cards for A
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
        for ix in range(2+5-nMask,7):
            cards_7A[ix] = 52

        # A win prob
        mcAChance = monte_carlo_prob_won(
            cards=      cards_7A,
            n_samples=  n_monte,
            asc=        asc)

        b_cA.append(cards_7A)       # 7 cards of A
        b_cB.append(cards_7B)       # 7 cards of B
        b_wins.append(wins)         # who wins {0,1,2}
        b_rA.append(rank_A)         # rank of A
        b_rB.append(rank_B)         # rank ok B
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


class Batch2X7_RW(RunningWorker):

    def __init__(
            self,
            batch_size: int,
            n_monte: int):
        self.deck = PDeck()
        self.batch_size = batch_size
        self.n_monte = n_monte

    def process(self, **kwargs) -> Dict[str,List]:
        batch = prep2X7batch(
            deck=       self.deck,
            batch_size= self.batch_size,
            n_monte=    self.n_monte)
        batch.pop('rank_counter')
        batch.pop('won_counter')
        return batch


class OMPRunner_Batch2X7(OMPRunner):

    def __init__(
            self,
            rw_class=       Batch2X7_RW,
            batch_size=     1000,
            n_monte=        100,
            devices=        1.0,
            **kwargs):
        OMPRunner.__init__(
            self,
            rw_class=           rw_class,
            rw_init_kwargs=     {'batch_size':batch_size, 'n_monte':n_monte},
            devices=            devices,
            ordered_results=    False,
            **kwargs)


def _get_batches(
        logger,
        n_batches=  10000,
        batch_size= 1000,
        n_monte=    10,
        devices=    1.0,
) -> List:
    """ prepares list of batches """
    logger.info(f'preparing batches {n_batches} x({batch_size},{n_monte})..')
    ompr = OMPRunner_Batch2X7(
        batch_size= batch_size,
        n_monte=    n_monte,
        devices=    devices,
        logger=     logger)
    ompr.process(tasks=[{}] * n_batches)
    batches = ompr.get_all_results()
    ompr.exit()
    return batches


def get_train_batches(
        n_batches=  50000,
        batch_size= 1000,
        n_monte=    100,
        devices=    1.0,
        logger=     None,
        loglevel=   20,
) -> Dict:
    """ prepares dict with train batches """

    if not logger:
        logger = get_pylogger(name='get_train_batches', level=loglevel)

    fn = f'{CACHE_FD}/tr{n_batches}_s{batch_size}_m{n_monte}.batches'
    logger.info(f'Reading train batches from file: {fn} ..')
    batches = r_pickle(fn)
    if batches:
        logger.info(f'got train batches from file: {fn}')
    else:
        batches = _get_batches(
            logger=     logger,
            n_batches=  n_batches,
            batch_size= batch_size,
            n_monte=    n_monte,
            devices=    devices)
        w_pickle(batches, fn)

    return batches


def get_test_batch(
        batch_size :int,
        n_monte :int,
        devices=        1.0,
        logger=         None,
        loglevel=       20,
):
    """ prepares tests batch """

    if not logger:
        logger = get_pylogger(name='get_test_batch', level=loglevel)

    fn = f'{CACHE_FD}/s{batch_size}_m{n_monte}.batch'
    logger.info(f'Reading test batch from file: {fn} ..')
    test_batch = r_pickle(fn)
    if test_batch:
        logger.info(f'got test batch from file: {fn}')
    else:

        # prepare batches of size 1
        batches = _get_batches(
            logger=     logger,
            n_batches=  batch_size,
            batch_size= 1,
            n_monte=    n_monte,
            devices=    devices)

        test_batch = {k: [] for k in batches[0]}
        for b in batches:
            for k in test_batch:
                test_batch[k] += b[k]

        logger.info(f'writing the batch ({batch_size},{n_monte}) to {fn} ..')
        w_pickle(test_batch, fn)

    c_tuples = []
    for ix in range(batch_size):
        c_tuples.append(tuple(sorted(test_batch['cards_A'][ix])))
        c_tuples.append(tuple(sorted(test_batch['cards_B'][ix])))
    logger.info(f'got {len(c_tuples)} of hands in test_batch, of which {len(set(c_tuples))} are unique')
    return test_batch, c_tuples


if __name__ == "__main__":

    for nm in [
        10,
        100,
    ]:
        get_train_batches(n_monte=nm)

    for size,mcs in [
        (2000, 1000),
        (2000, 100000),
        (2000, 10000000),
    ]:
        get_test_batch(batch_size=size, n_monte=mcs)
