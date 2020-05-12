"""

 2020 (c) piteren

    puts cn ckpts for all nndmk models

"""

import os
import shutil

from ptools.lipytools.little_methods import r_pickle
from ptools.neuralmess.dev_manager import nestarter
from ptools.neuralmess.base_elements import mrg_ckpts

nestarter(log_folder=None, devices=False, verb=0)

MODELS_FD = '_models'

models = os.listdir(MODELS_FD)
for mdl in models:

    print(f'processing {mdl} ...')

    md_file = f'{MODELS_FD}/{mdl}/mdict.dct'
    file_mdict = r_pickle(md_file)
    c_embW = file_mdict['c_embW']

    cn_name = f'cnet{c_embW}'

    for ckpt in [
        #'enc_vars',
        #'cnn_vars',
        #'opt_vars'
    ]: shutil.rmtree(f'{MODELS_FD}/{mdl}/{ckpt}')

    mrg_ckpts(
        ckptA=          'enc_vars',
        ckptA_FD=       f'_models_pretrained/cardNet/{cn_name}/',
        ckptB=          None,
        ckptB_FD=       None,
        ckptM=          'enc_vars',
        ckptM_FD=       f'{MODELS_FD}/{mdl}/',
        replace_scope=  mdl,
        verb=           0)
