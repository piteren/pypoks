from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.mptools import Que, QMessage
import unittest

from envy import PyPoksException, load_game_config
from podecide.dmk_motorch import DMK_MOTorch_PG
from podecide.dmk import RanDMK, FolDMK

TMP_MODELS_DIR = f'tests/podecide/_tmp/_models'
FolDMK.SAVE_TOPDIR = TMP_MODELS_DIR

logger = get_pylogger(
    name=       'test',
    level=      10,
    flat_child= True,
)

GAME_CONFIG = load_game_config(name='2players_2bets')
TBL_CFG = {k: GAME_CONFIG[k] for k in ['table_size', 'table_moves', 'table_cash_start']}


def lifecycle(name:str, family:str='g'):
    """ build, start, save, stop, close FolDMK """

    gm_que = Que()

    dmk = FolDMK(
        name=           name,
        family=         family,
        motorch_type=   DMK_MOTorch_PG,
        motorch_point=  {'device':None},
        logger=         logger,
        **TBL_CFG)
    dmk.que_to_gm = gm_que
    dmk.start()

    msg = gm_que.get()
    logger.debug(f'GM receives message from DMK, type: {msg.type}, data: {msg.data}')

    for msg_type in [
        'start_dmk_loop',
        'save_dmk',
        'stop_dmk_loop',
        'stop_dmk_process'
    ]:
        msg = QMessage(type=msg_type,data=None)
        dmk.que_from_gm.put(msg)
        msg = gm_que.get()
        logger.debug(f'GM receives message from DMK, type: {msg.type}, data: {msg.data}')


def build_from_point(
        name=           'dmka',
        motorch_type=   DMK_MOTorch_PG,
        family=         'g',
        loglevel=       20):
    dmk_point = {
        'name':         name,
        'motorch_type': motorch_type,
        'family':       family}
    dmk_point.update(TBL_CFG)
    FolDMK.build_from_point(dmk_point=dmk_point, loglevel=loglevel)


class TestRanDMK(unittest.TestCase):

    def test_base_init(self):
        RanDMK(
            name=           'test_randmk',
            table_size=     TBL_CFG['table_size'],
            table_moves=    TBL_CFG['table_moves'])


class TestFolDMK(unittest.TestCase):

    def setUp(self) -> None:
        prep_folder(TMP_MODELS_DIR, flush_non_empty=True)

    def test_init(self):
        FolDMK(
            name=           'foldmk_test',
            loglevel=       10,
            **TBL_CFG, )

    def test_save_point(self):
        dmk = FolDMK(
            name=           'foldmk_test',
            family=         'a',
            loglevel=       10,
            **TBL_CFG)
        dmk.save_point()
        print(dmk)
        self.assertTrue(dmk.family == 'a' and dmk.save_fn_pfx == 'dmk_dna')

    def test_save_dir(self):
        folder = f'{TMP_MODELS_DIR}/sub'
        dmk = FolDMK(
            name=           'foldmk_test',
            family=         'a',
            save_topdir=    folder,
            loglevel=       10,
            **TBL_CFG)
        dmk.save_point()
        print(dmk)
        self.assertTrue(dmk.family == 'a' and dmk.save_fn_pfx == 'dmk_dna' and dmk.save_topdir == folder)

    def test_save_load(self):
        dmk = FolDMK(
            name=           'foldmk_test',
            family=         'p',
            loglevel=       10,
            **TBL_CFG)
        dmk.save_point()
        dmk = FolDMK(name='foldmk_test')
        print(dmk)
        self.assertTrue(dmk.family == 'p')

    def test_build_from_point(self):
        build_from_point(name='dmk_a', loglevel=10)

    def test_lifecycle(self):
        lifecycle('foldmk_test')
        lifecycle('foldmk_test')

    def test_backup(self):
        dmk_name = 'foldmk_test'

        self.assertRaises(PyPoksException, FolDMK.save_policy_backup, dmk_name)
        self.assertRaises(PyPoksException, FolDMK.restore_policy_backup, dmk_name)

        build_from_point(name=dmk_name)

        FolDMK.save_policy_backup(dmk_name=dmk_name)
        FolDMK.restore_policy_backup(dmk_name=dmk_name)

    def test_copy(self):
        build_from_point(name='foldmk_test')

        FolDMK.copy_saved(
            name_src= 'foldmk_test',
            name_trg= 'foldmk_test_copy')
        dmk = FolDMK(name='foldmk_test_copy', assert_saved=True)
        self.assertTrue(dmk.name == 'foldmk_test_copy')

    def test_gx_saved(self):
        build_from_point(name='dmk1')
        build_from_point(name='dmk2')

        # base
        FolDMK.gx_saved(
            name_parentA=   'dmk1',
            name_parentB=   'dmk2',
            name_child=     'dmk3',
            loglevel=       10,
        )
        fc = FolDMK(name='dmk3', loglevel=10)
        print(fc)
        self.assertTrue(fc.parents == ['dmk1', 'dmk2'])

        # into subdir
        folder = f'{TMP_MODELS_DIR}/subdir'
        FolDMK.gx_saved(
            name_parentA=       'dmk1',
            name_parentB=       'dmk3',
            name_child=         'dmk4',
            save_topdir_child=  folder,
            do_gx_ckpt=         False)
        fd = FolDMK(name='dmk4', save_topdir=folder)
        print(fd)
        self.assertTrue(fd.parents == ['dmk1', ['dmk1', 'dmk2']] and fd.save_topdir == folder)

        # no ckpt
        FolDMK.gx_saved(
            name_parentA=   'dmk1',
            name_parentB=   'dmk2',
            name_child=     'dmk5',
            do_gx_ckpt=     False,
            loglevel=       10,
        )
        fc = FolDMK(name='dmk5', loglevel=10)
        print(fc)