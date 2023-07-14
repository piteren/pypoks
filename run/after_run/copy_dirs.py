from envy import DMK_MODELS_FD
from run.functions import get_saved_dmks_names, copy_dmks


if __name__ == "__main__":

    from_dir = f'{DMK_MODELS_FD}/_pmt'
    to_dir = DMK_MODELS_FD

    names = get_saved_dmks_names(folder=from_dir)

    copy_dmks(
        names_src=          names,
        names_trg=          names,
        save_topdir_src=    from_dir,
        save_topdir_trg=    to_dir)