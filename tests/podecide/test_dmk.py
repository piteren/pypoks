import unittest

from pypaq.lipytools.files import prep_folder
from pypaq.mpython.mptools import Que, QMessage

from podecide.dmk import RanDMK, NeurDMK, FolDMK

TMP_MODELS_DIR = f'_tmp/_models'


# FolDMK: build, save, close
def lifecycle(name:str, loglevel=20):

    gm_que = Que()

    dmk = FolDMK(
        name=           name,
        save_topdir=    TMP_MODELS_DIR,
        loglevel=       loglevel)

    dmk.que_to_gm = gm_que
    dmk.start()

    rmsg = gm_que.get()
    print(f'GM receives message from DMK, TYPE: {rmsg.type}, DATA: {rmsg.data}')

    for msg_type in [
        'start_dmk_loop',
        'save_dmk',
        'stop_dmk_loop',
        'stop_dmk_process'
    ]:
        message = QMessage(type=msg_type,data=None)
        dmk.que_from_gm.put(message)
        rmsg = gm_que.get()
        print(f'GM receives message from DMK, TYPE: {rmsg.type}, DATA: {rmsg.data}')


class TestRanDMK(unittest.TestCase):

    def test_base(self):
        RanDMK(name='test_randmk')


class TestFolDMK(unittest.TestCase):

    def setUp(self) -> None:
        prep_folder(TMP_MODELS_DIR, flush_non_empty=True)


    def test_init(self):
        FolDMK(name='foldmk_test')


    def test_save(self):
        dmk = FolDMK(
            name=           'foldmk_test',
            save_topdir=    TMP_MODELS_DIR)
        dmk.save()


    def test_lifecycle(self):
        lifecycle('foldmk_test', loglevel=10)


    def test_copy(self):
        lifecycle('foldmk_test')
        FolDMK.copy_saved(
            name_src=           'foldmk_test',
            name_trg=           'foldmk_test_copy',
            save_topdir_src=    TMP_MODELS_DIR)


    def test_gx(self):
        lifecycle('fa', loglevel=30)
        lifecycle('fb', loglevel=30)

        FolDMK.gx_saved(
            name_parent_main=           'fa',
            name_parent_scnd=           'fb',
            name_child=                 'fc',
            save_topdir_parent_main=    TMP_MODELS_DIR)

        fc = FolDMK(
            name=           'fc',
            save_topdir=    TMP_MODELS_DIR)
        print(fc)
        self.assertTrue(fc['parents'] == ['fa', 'fb'])