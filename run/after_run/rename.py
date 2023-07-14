import shutil

from envy import DMK_MODELS_FD
from run.functions import get_saved_dmks_names, copy_dmks


if __name__ == "__main__":

    dmks = get_saved_dmks_names()
    print(dmks)

    #dmks_targets = [f'{dn[:-4]}' for n, dn in enumerate(dmks)]
    #dmks_targets = [f'dmk00{dn[3]}{n:02}' for n,dn in enumerate(dmks)]
    dmks_targets = [f'dmk00a{n:02}' for n,dn in enumerate(dmks)]
    print(dmks_targets)

    copy_dmks(names_src=dmks, names_trg=dmks_targets)

    for dn in dmks:
        shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)