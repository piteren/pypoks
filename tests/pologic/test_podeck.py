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
            (['2H', '4H', '5C', '8D', 'DD', 'KS', 'AS'], 0),
            (['3C', '4H', '5C', '8D', '8C', 'DS', '2S'], 1),
            (['3C', '3H', '5C', '7D', '8C', '6H', '6S'], 2),
            (['5C', '5H', '5D', 'JC', 'TS', '3H', '2H'], 3),
            (['AC', 'KS', 'DC', 'JC', 'TS', '3H', '2H'], 4),
            (['AC', 'KS', 'DC', 'DH', 'JC', 'TS', '2H'], 4),
            (['AC', 'KS', '2C', '3H', '4C', '5S', '2H'], 4),
            (['AC', 'KC', 'DC', 'DH', 'JH', 'TC', '2C'], 5),
            (['3C', '3H', '5C', '5D', '5H', '7H', '6S'], 6),
            (['3C', '3H', '5C', '5D', '5H', '7H', '3S'], 6),
            (['3C', '3H', '5D', '3D', 'TS', '3S', '2H'], 7),
            (['AC', 'KC', 'DC', 'DH', 'JC', 'TC', '2C'], 8),
        ]
        for tc in test_cases:
            rank = PDeck.cards_rank(tc[0])
            print(rank[-1])
            self.assertEqual(rank[0], tc[1])


    def test_100(self):

        asc = ASC()

        for cards in [
            ['4D', 'AH', 'TS', '8C', 'JH', 'DD', 'AC'],
            ['JD', '3H', '5D', '9H', '2D', 'AS', 'AD'],
            ['3D', '8C', '9D', 'JD', 'DD', 'KH', '3H'],
            ['3S', 'AH', '3C', '2S', '8H', '2D', '8D'],
            ['TD', '5D', '7H', 'AD', '3S', '6C', '8D'],
            ['7S', '7H', 'AH', 'KD', '7D', 'TH', '5C'],
            ['7H', '5C', '6D', '3C', '9D', '5D', 'DD'],
            ['KC', 'JC', '6D', '5C', 'DC', '8D', '7C'],
            ['6H', '8D', 'KS', '7C', '6D', '5H', '7H'],
            ['3H', '8D', 'KH', '3C', 'DC', '6D', '2D'],
            ['JD', '3D', '5C', '7D', '6D', 'AC', 'TH'],
            ['KD', '4H', '6D', '8H', 'JH', '6S', '5C'],
            ['JH', 'DD', '5C', '2C', '9H', 'DC', '4S'],
            ['JC', 'DS', 'KC', '2C', 'JD', '6C', '7D'],
            ['2C', '7H', 'TH', '9S', '3H', '5D', '5S'],
            ['DH', '3D', '3S', 'DS', '3C', 'DC', 'AS'],
            ['2C', '8H', 'KS', '5C', '7S', 'JD', '5D'],
            ['9H', 'JS', '5H', '4S', '3C', 'KS', '9C'],
            ['9H', '2C', '3S', '9D', '5S', 'JS', '2S'],
            ['4C', '7C', 'JC', '5D', '6H', '7S', '4H'],
            ['2C', 'AH', '6S', 'JH', 'TS', '4D', '6C'],
            ['AD', '4S', '7D', 'KD', 'TS', '6H', '8H'],
            ['TD', '5D', '9C', '3C', 'DH', '6D', '5H'],
            ['KD', '3C', '5D', 'AD', 'JS', 'KS', '5C'],
            ['6S', '3H', '2S', '6H', '7D', 'DD', '4C'],
            ['7C', 'AC', '5H', 'AD', '7D', 'KC', '6C'],
            ['KD', '7C', 'AC', '2C', 'JS', '3C', '4S'],
            ['3H', '3S', '9S', 'JH', 'KH', '7D', '4S'],
            ['3S', '3H', '5H', '9H', '8D', '5C', 'AH'],
            ['JD', '5D', '4S', '8S', 'KH', '6S', '5C'],
            ['9C', 'DS', 'AD', '7H', '6D', '9H', 'JS'],
            ['5S', 'DD', '9S', '4D', 'KH', '6S', '4C'],
            ['4D', 'TC', 'AD', 'DS', '4H', '8D', 'TD'],
            ['5D', 'KH', 'AC', 'KD', '7S', '8D', '2H'],
            ['8S', '5C', 'DH', 'TH', 'KH', '2D', '5H'],
            ['6D', '7H', 'AC', '2C', '5D', 'TC', '4S'],
            ['9C', '5D', 'AC', 'JD', '8S', '5H', '8D'],
            ['AD', '3H', 'JS', 'JH', 'DS', '4D', '7S'],
            ['DD', '2D', 'AC', '8D', 'JH', '3S', '8C'],
            ['4S', '8C', '7H', '3H', '9D', '7C', 'JS'],
            ['6D', '7D', '9H', '9C', 'AD', 'AC', 'AS'],
            ['6S', 'KS', 'KC', '4C', '8C', '9D', 'TH'],
            ['3H', '2H', '8C', '9D', '9S', '3D', '5S'],
            ['3C', '7H', '2S', '2C', 'JC', '4S', '9H'],
            ['3S', 'DH', '2H', 'KC', '8S', 'JC', 'JS'],
            ['6D', '4C', 'AC', 'TC', '8C', '7S', '6C'],
            ['DH', 'KH', '2D', '9C', 'TH', 'JH', '9D'],
            ['DC', '9D', '4H', 'AS', 'AD', '3H', '4D'],
            ['DC', 'TC', '5S', 'JC', 'AC', '8S', '5D'],
            ['3D', '7C', '2H', '8D', '4D', '4C', 'TH'],
            ['DD', '6C', 'JH', '5D', '5H', '6H', 'TD'],
            ['TD', 'KH', '7D', '7S', 'AS', '7H', '8C'],
            ['6C', 'KH', '5C', 'DC', 'JC', '8S', 'TH'],
            ['5C', 'DH', 'TD', '2C', '9C', '6C', 'KS'],
            ['7C', 'KD', '9H', 'DC', '6H', 'JC', '6D'],
            ['5H', '7C', '4C', '9D', '4S', '7H', 'KS'],
            ['9D', '4D', '7D', '6H', 'KC', '6S', '4H'],
            ['AH', '5H', '3D', '7S', '9H', '6C', 'AC'],
            ['AD', '2S', 'DD', '8S', 'AH', 'TD', 'DH'],
            ['TH', 'DD', '7C', '4S', '7H', 'DS', '7S'],
            ['DD', '6H', '8D', 'DH', '3S', 'JD', '5D'],
            ['JH', 'KC', 'AS', '7S', '4H', 'AH', '4C'],
            ['7S', '8C', '3C', 'TS', '4H', '5S', '4S'],
            ['9C', '3D', 'KC', '5S', 'KS', '4D', '3S'],
            ['9D', '3D', '9C', 'DC', '7D', 'AH', '4H'],
            ['4C', '7S', '2C', '5D', 'AH', 'DH', '3S'],
            ['2S', '6S', 'JS', 'TS', 'DC', '5S', '8D'],
            ['JD', '5D', 'AS', '9H', '4H', '5H', '3C'],
            ['5C', '4H', '4D', 'AD', '5S', '4S', '3C'],
            ['9D', '2C', 'KS', 'KC', '6H', 'JS', '9C'],
            ['8D', 'AD', '9S', '2D', '3H', '7H', '8H'],
            ['2D', '9C', '5H', '7D', '8H', '2C', 'TH'],
            ['6S', 'TC', '3D', '9D', '3S', '5S', 'DH'],
            ['8S', 'KH', '5D', '2H', 'DC', 'AC', '8H'],
            ['4C', 'AH', '5D', 'TD', 'JH', '9H', 'TH'],
            ['KC', '6C', '5D', '8C', '7C', '5C', 'AD'],
            ['8C', 'DS', '9H', 'DH', 'TC', 'JD', 'KC'],
            ['8S', '4C', '4D', '5D', 'AH', 'KH', '6C'],
            ['3C', '8H', '5C', '3D', '3S', 'TC', 'JD'],
            ['3S', '5C', '6D', 'TD', '4C', '7S', 'AS'],
            ['5C', '9H', 'JH', '9C', 'DC', 'KD', 'DS'],
            ['9S', 'DC', '7C', '4C', 'KC', '5S', 'DH'],
            ['9C', '8D', '7S', '5D', 'TS', 'TH', 'DC'],
            ['5C', '6C', 'DH', '2S', '3S', 'KS', '5H'],
            ['5C', 'TS', 'KH', '8C', '3C', '3H', '9C'],
            ['8S', '6H', 'TD', 'KD', '7C', '6C', 'TS'],
            ['5S', '2H', 'AH', 'KS', 'JC', 'JD', 'DH'],
            ['4D', '2H', '9H', '3D', 'AC', 'DD', '5S'],
            ['9C', '4C', '8C', '7D', '9S', '7H', '5C'],
            ['9H', '8C', 'KH', 'TD', 'AC', 'TH', 'JC'],
            ['6S', '8H', '8S', '5H', '2S', '7S', 'KC'],
            ['7D', 'DH', '8S', 'AS', 'TS', 'JS', 'JH'],
            ['4H', 'JD', 'AC', '4D', 'TD', '3D', '9C'],
            ['8D', 'AS', '3C', '3D', '9C', '6D', 'KD'],
            ['6C', 'AH', '6H', '7H', '5S', 'KS', 'KC'],
            ['2S', '4C', '7D', '4S', '8D', '8H', '6S'],
            ['KD', 'KC', 'JH', '4C', '2D', '2H', 'AC'],
            ['4H', '7H', 'AD', '2H', 'TS', 'AH', '9H'],
            ['JD', '8H', '9D', '7D', '5S', '5H', '3H'],
            ['7D', '5H', 'TD', '8H', '9H', 'AH', '4S'],
        ]:
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

