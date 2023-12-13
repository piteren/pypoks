import time
from tqdm import tqdm
import unittest

from pologic.potable import PTable
from pologic.hand_history import HHistory

table_size = 3


class TestPTable(unittest.TestCase):

    def test_base(self):
        table = PTable(
            name=       'table',
            pl_ids=     [f'pl{ix}' for ix in range(table_size)])
        hh = table.run_hand()

        print(f'\nHHistory:\n{hh}')

        print(f'\nreadable events:')
        for e in hh.events:
            re = hh.readable_event(e)
            if re: print(re)

    def test_table_speed(self, n_hands=100000):
        table = PTable(name='table_speed', pl_ids=[f'pl{ix}' for ix in range(table_size)])
        stime = time.time()
        for _ in tqdm(range(n_hands)):
            table.run_hand()
        n_sec = time.time()-stime
        print(f'time taken: {n_sec:.1f}sec ({int(n_hands/n_sec)} h/s)')

    def test_run_with_hh(self):
        table = PTable(
            name=       'table',
            pl_ids=     [f'pl{ix}' for ix in range(table_size)])
        hh1 = table.run_hand()
        print(f'\nHHistory1:\n{hh1}')

        hh2 = table.run_hand(hh_given=hh1)
        print(f'\nHHistory2:\n{hh2}')

        for e1,e2 in zip(hh1.events[1:-1],hh2.events[1:-1]):
            self.assertEqual(e1, e2)

        # test for partial hand history
        hh3 = table.run_hand()
        print(f'\nHHistory3:\n{hh3}')
        hh3.events = hh3.events[:len(hh3.events)//2]

        hh4 = table.run_hand(hh_given=hh3)
        print(f'\nHHistory4:\n{hh4}')

