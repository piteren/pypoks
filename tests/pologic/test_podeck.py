import itertools
import random
import time
from tqdm import tqdm
import unittest

from pologic.podeck import PDeck, ASC, monte_carlo_prob_won


class TestPDeck(unittest.TestCase):

    def test_card_representation(self):
        for cix in range(53):
            print(f'{cix:2} {PDeck.cts(cix)} {PDeck.ctt(cix)}')


    def test_deck_simple(self):
        test_cases = [
            (['2h', '4h', '5c', '8d', 'Qd', 'Ks', 'As'], 0),
            (['3c', '4h', '5c', '8d', '8c', 'Qs', '2s'], 1),
            (['3c', '3h', '5c', '7d', '8c', '6h', '6s'], 2),
            (['5c', '5h', '5d', 'Jc', 'Ts', '3h', '2h'], 3),
            (['Ac', 'Ks', 'Qc', 'Jc', 'Ts', '3h', '2h'], 4),
            (['Ac', 'Ks', 'Qc', 'Qh', 'Jc', 'Ts', '2h'], 4),
            (['Ac', 'Ks', '2c', '3h', '4c', '5s', '2h'], 4),
            (['Ac', 'Kc', 'Qc', 'Qh', 'Jh', 'Tc', '2c'], 5),
            (['3c', '3h', '5c', '5d', '5h', '7h', '6s'], 6),
            (['3c', '3h', '5c', '5d', '5h', '7h', '3s'], 6),
            (['3c', '3h', '5d', '3d', 'Ts', '3s', '2h'], 7),
            (['Ac', 'Kc', 'Qc', 'Qh', 'Jc', 'Tc', '2c'], 8),
        ]
        for tc in test_cases:
            rank = PDeck.cards_rank(tc[0])
            print(rank[-1])
            self.assertEqual(rank[0], tc[1])


    def test_100(self):

        asc = ASC()

        for ix,cards in enumerate([
            ['4d', 'Ah', 'Ts', '8c', 'Jh', 'Qd', 'Ac'],
            ['Jd', '3h', '5d', '9h', '2d', 'As', 'Ad'],
            ['3d', '8c', '9d', 'Jd', 'Qd', 'Kh', '3h'],
            ['3s', 'Ah', '3c', '2s', '8h', '2d', '8d'],
            ['Td', '5d', '7h', 'Ad', '3s', '6c', '8d'],
            ['7s', '7h', 'Ah', 'Kd', '7d', 'Th', '5c'],
            ['7h', '5c', '6d', '3c', '9d', '5d', 'Qd'],
            ['Kc', 'Jc', '6d', '5c', 'Qc', '8d', '7c'],
            ['6h', '8d', 'Ks', '7c', '6d', '5h', '7h'],
            ['3h', '8d', 'Kh', '3c', 'Qc', '6d', '2d'],
            ['Jd', '3d', '5c', '7d', '6d', 'Ac', 'Th'],
            ['Kd', '4h', '6d', '8h', 'Jh', '6s', '5c'],
            ['Jh', 'Qd', '5c', '2c', '9h', 'Qc', '4s'],
            ['Jc', 'Qs', 'Kc', '2c', 'Jd', '6c', '7d'],
            ['2c', '7h', 'Th', '9s', '3h', '5d', '5s'],
            ['Qh', '3d', '3s', 'Qs', '3c', 'Qc', 'As'],
            ['2c', '8h', 'Ks', '5c', '7s', 'Jd', '5d'],
            ['9h', 'Js', '5h', '4s', '3c', 'Ks', '9c'],
            ['9h', '2c', '3s', '9d', '5s', 'Js', '2s'],
            ['4c', '7c', 'Jc', '5d', '6h', '7s', '4h'],
            ['2c', 'Ah', '6s', 'Jh', 'Ts', '4d', '6c'],
            ['Ad', '4s', '7d', 'Kd', 'Ts', '6h', '8h'],
            ['Td', '5d', '9c', '3c', 'Qh', '6d', '5h'],
            ['Kd', '3c', '5d', 'Ad', 'Js', 'Ks', '5c'],
            ['6s', '3h', '2s', '6h', '7d', 'Qd', '4c'],
            ['7c', 'Ac', '5h', 'Ad', '7d', 'Kc', '6c'],
            ['Kd', '7c', 'Ac', '2c', 'Js', '3c', '4s'],
            ['3h', '3s', '9s', 'Jh', 'Kh', '7d', '4s'],
            ['3s', '3h', '5h', '9h', '8d', '5c', 'Ah'],
            ['Jd', '5d', '4s', '8s', 'Kh', '6s', '5c'],
            ['9c', 'Qs', 'Ad', '7h', '6d', '9h', 'Js'],
            ['5s', 'Qd', '9s', '4d', 'Kh', '6s', '4c'],
            ['4d', 'Tc', 'Ad', 'Qs', '4h', '8d', 'Td'],
            ['5d', 'Kh', 'Ac', 'Kd', '7s', '8d', '2h'],
            ['8s', '5c', 'Qh', 'Th', 'Kh', '2d', '5h'],
            ['6d', '7h', 'Ac', '2c', '5d', 'Tc', '4s'],
            ['9c', '5d', 'Ac', 'Jd', '8s', '5h', '8d'],
            ['Ad', '3h', 'Js', 'Jh', 'Qs', '4d', '7s'],
            ['Qd', '2d', 'Ac', '8d', 'Jh', '3s', '8c'],
            ['4s', '8c', '7h', '3h', '9d', '7c', 'Js'],
            ['6d', '7d', '9h', '9c', 'Ad', 'Ac', 'As'],
            ['6s', 'Ks', 'Kc', '4c', '8c', '9d', 'Th'],
            ['3h', '2h', '8c', '9d', '9s', '3d', '5s'],
            ['3c', '7h', '2s', '2c', 'Jc', '4s', '9h'],
            ['3s', 'Qh', '2h', 'Kc', '8s', 'Jc', 'Js'],
            ['6d', '4c', 'Ac', 'Tc', '8c', '7s', '6c'],
            ['Qh', 'Kh', '2d', '9c', 'Th', 'Jh', '9d'],
            ['Qc', '9d', '4h', 'As', 'Ad', '3h', '4d'],
            ['Qc', 'Tc', '5s', 'Jc', 'Ac', '8s', '5d'],
            ['3d', '7c', '2h', '8d', '4d', '4c', 'Th'],
            ['Qd', '6c', 'Jh', '5d', '5h', '6h', 'Td'],
            ['Td', 'Kh', '7d', '7s', 'As', '7h', '8c'],
            ['6c', 'Kh', '5c', 'Qc', 'Jc', '8s', 'Th'],
            ['5c', 'Qh', 'Td', '2c', '9c', '6c', 'Ks'],
            ['7c', 'Kd', '9h', 'Qc', '6h', 'Jc', '6d'],
            ['5h', '7c', '4c', '9d', '4s', '7h', 'Ks'],
            ['9d', '4d', '7d', '6h', 'Kc', '6s', '4h'],
            ['Ah', '5h', '3d', '7s', '9h', '6c', 'Ac'],
            ['Ad', '2s', 'Qd', '8s', 'Ah', 'Td', 'Qh'],
            ['Th', 'Qd', '7c', '4s', '7h', 'Qs', '7s'],
            ['Qd', '6h', '8d', 'Qh', '3s', 'Jd', '5d'],
            ['Jh', 'Kc', 'As', '7s', '4h', 'Ah', '4c'],
            ['7s', '8c', '3c', 'Ts', '4h', '5s', '4s'],
            ['9c', '3d', 'Kc', '5s', 'Ks', '4d', '3s'],
            ['9d', '3d', '9c', 'Qc', '7d', 'Ah', '4h'],
            ['4c', '7s', '2c', '5d', 'Ah', 'Qh', '3s'],
            ['2s', '6s', 'Js', 'Ts', 'Qc', '5s', '8d'],
            ['Jd', '5d', 'As', '9h', '4h', '5h', '3c'],
            ['5c', '4h', '4d', 'Ad', '5s', '4s', '3c'],
            ['9d', '2c', 'Ks', 'Kc', '6h', 'Js', '9c'],
            ['8d', 'Ad', '9s', '2d', '3h', '7h', '8h'],
            ['2d', '9c', '5h', '7d', '8h', '2c', 'Th'],
            ['6s', 'Tc', '3d', '9d', '3s', '5s', 'Qh'],
            ['8s', 'Kh', '5d', '2h', 'Qc', 'Ac', '8h'],
            ['4c', 'Ah', '5d', 'Td', 'Jh', '9h', 'Th'],
            ['Kc', '6c', '5d', '8c', '7c', '5c', 'Ad'],
            ['8c', 'Qs', '9h', 'Qh', 'Tc', 'Jd', 'Kc'],
            ['8s', '4c', '4d', '5d', 'Ah', 'Kh', '6c'],
            ['3c', '8h', '5c', '3d', '3s', 'Tc', 'Jd'],
            ['3s', '5c', '6d', 'Td', '4c', '7s', 'As'],
            ['5c', '9h', 'Jh', '9c', 'Qc', 'Kd', 'Qs'],
            ['9s', 'Qc', '7c', '4c', 'Kc', '5s', 'Qh'],
            ['9c', '8d', '7s', '5d', 'Ts', 'Th', 'Qc'],
            ['5c', '6c', 'Qh', '2s', '3s', 'Ks', '5h'],
            ['5c', 'Ts', 'Kh', '8c', '3c', '3h', '9c'],
            ['8s', '6h', 'Td', 'Kd', '7c', '6c', 'Ts'],
            ['5s', '2h', 'Ah', 'Ks', 'Jc', 'Jd', 'Qh'],
            ['4d', '2h', '9h', '3d', 'Ac', 'Qd', '5s'],
            ['9c', '4c', '8c', '7d', '9s', '7h', '5c'],
            ['9h', '8c', 'Kh', 'Td', 'Ac', 'Th', 'Jc'],
            ['6s', '8h', '8s', '5h', '2s', '7s', 'Kc'],
            ['7d', 'Qh', '8s', 'As', 'Ts', 'Js', 'Jh'],
            ['4h', 'Jd', 'Ac', '4d', 'Td', '3d', '9c'],
            ['8d', 'As', '3c', '3d', '9c', '6d', 'Kd'],
            ['6c', 'Ah', '6h', '7h', '5s', 'Ks', 'Kc'],
            ['2s', '4c', '7d', '4s', '8d', '8h', '6s'],
            ['Kd', 'Kc', 'Jh', '4c', '2d', '2h', 'Ac'],
            ['4h', '7h', 'Ad', '2h', 'Ts', 'Ah', '9h'],
            ['Jd', '8h', '9d', '7d', '5s', '5h', '3h'],
            ['7d', '5h', 'Td', '8h', '9h', 'Ah', '4s'],
        ]):
            rank = PDeck.cards_rank(cards)
            cards_t = [PDeck.cti(c) for c in cards]
            cards_ts = tuple(sorted(cards_t))
            asc_rank_value = asc.cards_rank(cards_ts)
            self.assertTrue(rank[1] == asc_rank_value)
            print(f'{cards} {rank[0]} {rank[-1]}')


    def test_deck_random(self):
        dk = PDeck()
        for n in range(10000):
            rank = n%8
            cards = dk.get_7of_rank(rank)
            cr = PDeck.cards_rank(cards)
            if cr[0] != rank:
                print(cards,cr)
            self.assertEqual(cr[0],rank)


    def test_rank_speed(self, num_ask=int(1e6)):
        """ tests speed of ranking """

        tdeck = PDeck()
        scL = []
        print('\nPreparing cards..')
        for _ in tqdm(range(num_ask)):
            scL.append([tdeck.get_card() for _ in range(7)])
            tdeck.reset()

        x = int(num_ask/2)
        for c in scL[x:x+10]:
            print(c)

        s_time = time.time()
        for sc in scL:
            _ = PDeck.cards_rank_tuples(sc)
        e_time = time.time()

        print(f'time taken {e_time-s_time:.2f}sec')
        print(f'speed {int(num_ask/(e_time-s_time))}/sec')


    def test_ASC_ranks(self, num_ask=int(1e6)):
        """ compares speed of ASC and PDeck """

        print('\nPreparing combinations of 7 from 52 ..')
        comb_list = list(itertools.combinations([x for x in range(52)], 7))
        print(f'done!, got {len(comb_list)} combinations')
        x = 1235143
        for c in comb_list[x:x+10]:
            print(c)

        ask_cards = [comb_list[random.randint(0, len(comb_list))] for _ in range(num_ask)]

        asc = ASC()
        print(f'got ASC of len: {len(asc)}')
        s_time = time.time()
        for c in tqdm(ask_cards):
            _ = asc.cards_rank(c)
        print(f'speed {int(num_ask / (time.time() - s_time))}/sec')

        s_time = time.time()
        for c in tqdm(ask_cards):
            _ = PDeck.cards_rank(c)
        print(f'speed {int(num_ask / (time.time() - s_time))}/sec')


    def test_monte_carlo_prob_won(self):

        #asc = ASC()

        for cards in [
            [22, 23, 40, 10, 43, 52, 52],
            [22, 23, 40, 10, 43],
            [50, 51],
            [0, 14],
            [0, 1],
            [12, 45],
        ]:

            mx = 5
            print(list(PDeck.cts(c) for c in cards))
            for p in range(mx):
                n = 10**p
                s_time = time.time()
                prob = monte_carlo_prob_won(
                    cards=      cards,
                    n_samples=  n,
                    #asc=        asc,
                )
                print(f'{n:{mx}} {time.time() - s_time:7.3f}s {prob}')

