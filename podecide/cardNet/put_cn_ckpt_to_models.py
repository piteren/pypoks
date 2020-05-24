"""

 2020 (c) piteren

    puts cardNet ckpt for given nn dmk model name

"""

import os

from ptools.lipytools.little_methods import r_pickle
from ptools.neuralmess.dev_manager import nestarter
from ptools.neuralmess.base_elements import mrg_ckpts

from pypoks_envy import DMK_MODELS_FD, CN_MODELS_FD, get_cardNet_name


def put_cn_ckpts(dmk_name :str):

    nestarter(log_folder=None, devices=False, verb=0)

    dmk_FD = f'{DMK_MODELS_FD}/{dmk_name}/'

    file_mdict = r_pickle(f'{dmk_FD}mdict.dct')
    c_embW = file_mdict['c_embW']
    cn_name = get_cardNet_name(c_embW)

    cardNet_FD = f'{CN_MODELS_FD}/{cn_name}/'

    if not os.path.isdir(cardNet_FD): return False # there is no cardNet for this dmk

    mrg_ckpts(
        ckptA=          'enc_vars',
        ckptA_FD=       cardNet_FD,
        ckptB=          None,
        ckptB_FD=       None,
        ckptM=          'enc_vars',
        ckptM_FD=       dmk_FD,
        replace_scope=  dmk_name,
        verb=           0)
    return True
