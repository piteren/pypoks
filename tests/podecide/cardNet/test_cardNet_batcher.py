import unittest

from podecide.cardNet.cardNet_batcher import prep2X7batch


class Test_cardNet_batch(unittest.TestCase):

    def test_base(self):
        batch = prep2X7batch(batch_size=5)
        for k in batch:
            print(f'{k}: ({len(batch[k])}) {batch[k]}')
