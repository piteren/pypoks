import unittest

from pologic.podeck import PDeck


class TestPDeck(unittest.TestCase):

    def test_card_representation(self):
        for ci in range(53):
            print(f'{ci:2} {PDeck.cts(ci)} {PDeck.ctt(ci)}')


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
            self.assertEqual(rank[0],tc[1])

    def test_deck_random(self):
        dk = PDeck()
        for n in range(10000):
            rank = n%8
            cards = dk.get7of_rank(rank)
            cr = PDeck.cards_rank(cards)
            if cr[0] != rank:
                print(cards,cr)
            self.assertEqual(cr[0],rank)