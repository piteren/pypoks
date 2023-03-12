import time
from tqdm import tqdm
import unittest

from pologic.potable import PTable


class TestPDeck(unittest.TestCase):

    def test_base(self):
        table = PTable(
            name=       'table',
            pl_ids=     [0,1,2],
            loglevel=   10)
        hh = table.run_hand()
        print(hh)


    def test_table_speed(self, n_hands=100000):
        table = PTable(name='table_speed', pl_ids=[0,1,2])
        stime = time.time()
        for _ in tqdm(range(n_hands)):
            table.run_hand()
            #hh = table.run_hand()
            #print('%s\n'%hh)
        n_sec = time.time()-stime
        print('time taken: %.1fsec (%d h/s)'%(n_sec, n_hands/n_sec))


    def test_table_history(self, n=3):
        table = PTable(name='table_history', pl_ids=[0,1,2])
        for _ in range(n):
            hh = table.run_hand()
            print(f'{hh}\n')