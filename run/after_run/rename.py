"""
script to rename saved DMKs
"""
import shutil

from pypoks_envy import DMK_MODELS_FD
from run.functions import get_saved_dmks_names, copy_dmks


if __name__ == "__main__":

    dmks = get_saved_dmks_names()
    print(dmks)
    dmks_targets = [f'dmk{dn[4:6]}{dn[3]}{dn[7:]}' for dn in dmks]
    print(dmks_targets)

    copy_dmks(names_src=dmks, names_trg=dmks_targets)

    for dn in dmks:
        shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)