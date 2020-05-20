"""

 2020 (c) piteren

    puts cn ckpts for given nn dmk model names

"""

from typing import List

from ptools.neuralmess.dev_manager import nestarter
from ptools.neuralmess.base_elements import mrg_ckpts

from pypoks_envy import DMK_MODELS_FD, CN_MODELS_FD


def put_cn_ckpts(
        c_embW :int,            # width of cards embedding (defines cards_net type)
        modelsL :List[str]):    # list of folder names of DMKs

    nestarter(log_folder=None, devices=False, verb=0)

    for mdl in modelsL:

        print(f'processing {mdl} ...')

        mrg_ckpts(
            ckptA=          'enc_vars',
            ckptA_FD=       f'{CN_MODELS_FD}/cardNet{c_embW}/',
            ckptB=          None,
            ckptB_FD=       None,
            ckptM=          'enc_vars',
            ckptM_FD=       f'{DMK_MODELS_FD}/{mdl}/',
            replace_scope=  mdl,
            verb=           0)
