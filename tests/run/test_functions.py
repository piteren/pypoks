from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.files import prep_folder
import unittest

from envy import PyPoksException
from podecide.dmk import FolDMK
from run.functions import build_single_foldmk

TMP_MODELS_DIR = f'tests/podecide/_tmp/_models'
FolDMK.SAVE_TOPDIR = TMP_MODELS_DIR

logger = get_pylogger(
    name=           'test',
    level=          20,
    #flat_child=     True,
)


class TestFunctions(unittest.TestCase):

    def setUp(self) -> None:
        logger.info('cleaning TMP_MODELS_DIR')
        prep_folder(TMP_MODELS_DIR, flush_non_empty=True)

    def test_build_single_foldmk(self):
        build_single_foldmk(name='dmk_b', family='b', logger=logger)
        build_single_foldmk(name='dmk_b', family='b', logger=logger)
        build_single_foldmk(name='dmk_b', family='b', oversave=False, logger=logger)
        build_single_foldmk(name='dmk_c', family='b', logger=logger)
