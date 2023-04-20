from pypaq.lipytools.files import list_dir, prep_folder
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.mpython.mpdecor import proc_return, proc_wait
from pypaq.pms.base import POINT, PSDD
from pypaq.pms.paspa import PaSpa
import shutil
from typing import List, Optional, Dict, Tuple, Union

from pypoks_base import PyPoksException
from pypoks_envy import DMK_MODELS_FD
from podecide.dmk import FolDMK
from podecide.dmk_module_pg import ProCNN_DMK_PG
from podecide.dmk_module_a2c import ProCNN_DMK_A2C
from podecide.games_manager import GamesManager_PTR



def get_fresh_dna(name:str, family:str) -> Dict[str,POINT]:

    if family not in 'abcd':
        raise PyPoksException(f'unknown family: {family}')

    motorch_psdd: PSDD = {
        'baseLR':           [1e-6,  3e-5],
        'do_clip':          (True,  False),
        'train_ce':         (True,  False)}

    motorch_point_common: POINT = {
        'name':                     name,
        'module_type':              ProCNN_DMK_PG,
        'save_topdir':              DMK_MODELS_FD,
        'cards_emb_width':          12,
        'load_cardnet_pretrained':  True,
        'psdd':                     motorch_psdd}

    foldmk_psdd: PSDD = {
        'argmax_prob':      [0.0,   1.0],
        'sample_nmax':      [1,     4],
        'sample_maxdist':   [0.0,   0.2],

        'upd_trigger':      [20000, 40000],

        'enable_pex':       (True,  False),
        'pex_max':          [0.01,  0.05],
        'prob_zero':        [0.0,   0.45],
        'prob_max':         [0.0,   0.45],
        'step_min':         [100,   10000],
        'step_max':         [10000, 100000],
        'pid_pex_fraction': [0.5,   1.0]}

    foldmk_point_common: POINT = {
        'name':         name,
        'family':       family,
        'trainable':    True,
        'psdd':         foldmk_psdd}

    # A2C
    if family in 'cd':
        motorch_point_common['module_type'] = ProCNN_DMK_A2C

    # wider network
    if family in 'bd':
        motorch_point_common['cards_emb_width'] = 24

    """
    # no sampling from prob
    if family == '...':
        foldmk_psdd.pop('argmax_prob')
        foldmk_point_common['argmax_prob'] = 1
    """

    foldmk_point = {}
    foldmk_point.update(foldmk_point_common)
    paspa_foldmk = PaSpa(psdd=foldmk_psdd)
    foldmk_point.update(paspa_foldmk.sample_point_GX())

    motorch_point = {}
    motorch_point.update(motorch_point_common)
    paspa_motorch = PaSpa(psdd=motorch_psdd)
    motorch_point.update(paspa_motorch.sample_point_GX())

    return {
        'foldmk_point':     foldmk_point,
        'motorch_point':    motorch_point}


def get_saved_dmks_names(folder:str=DMK_MODELS_FD) -> List[str]:
    all_dirs = list_dir(folder)['dirs']
    return [d for d in all_dirs if d.startswith('dmk')]

# builds single FolDMK, saves into folder
@proc_wait
def build_single_foldmk(name:str, family:str, logger=None):
    points = get_fresh_dna(name, family)
    FolDMK.from_points(**points, logger=logger)

# builds dmks, checks folder for present ones
def build_from_names(
        names: List[str],
        families: Union[str,List[str]],
        logger=     None,
        loglevel=   20):

    if not logger: logger = get_pylogger(level=loglevel)

    prep_folder(DMK_MODELS_FD)
    dmks_in_dir = get_saved_dmks_names()
    logger.info(f'got dmks: {dmks_in_dir} already in {DMK_MODELS_FD}')

    sub_logger = get_child(logger, change_level=10)
    for nm,fm in zip(names,families):
        if nm not in dmks_in_dir:
            logger.info(f'> building: {nm}, family: {fm}')
            build_single_foldmk(name=nm, family=fm, logger=sub_logger)

@proc_wait
def copy_dmks(
        names_src: List[str],
        names_trg: List[str],
        save_topdir_trg: Optional[str]= None,
        logger=                         None,
        loglevel=                       30):
    if not logger: logger = get_pylogger(level=loglevel)
    for ns, nt in zip(names_src, names_trg):
        FolDMK.copy_saved(
            name_src=           ns,
            name_trg=           nt,
            save_topdir_trg=    save_topdir_trg,
            logger=             logger)

# resets all existing DMKs names
def reset_all_names(prefix: str = 'dmk000_'):
    all_names = get_saved_dmks_names()
    counter = 0
    for nm in all_names:
        print(f'renaming {nm}..')
        FolDMK.copy_saved(
            name_src=   nm,
            name_trg=   f'{prefix}{counter}')
        counter += 1
        shutil.rmtree(f'{DMK_MODELS_FD}/{nm}', ignore_errors=True)

# helper to set family of all saved DMKs
def set_all_to_family(family='a'):
    for nm in get_saved_dmks_names():
        FolDMK.oversave_point(name=nm, family=family)


def run_GM(
        logger,
        dmk_point_PLL: Optional[List[Dict]]=        None,
        dmk_point_TRL: Optional[List[Dict]]=        None,
        num_loops: int=                             1,      # each loop starts new GM
        game_size: int=                             100000,
        dmk_n_players: int=                         60,
        sep_pairs: Optional[List[Tuple[str,str]]]=  None,
        sep_bvalue=                                 0.7,
) -> Dict:

    @proc_return # processes <- to clear mem
    def single_loop():
        gm = GamesManager_PTR(
            dmk_point_PLL=  dmk_point_PLL,
            dmk_point_TRL=  dmk_point_TRL,
            dmk_n_players=  dmk_n_players,
            use_fsexc=      False,
            logger=         logger)
        return gm.run_game(
            game_size=      game_size,
            publish_GM=     not dmk_point_TRL,
            sep_pairs=      sep_pairs,
            sep_bvalue=     sep_bvalue)

    dmk_results = None
    for _ in range(num_loops):
        loop_dmk_results = single_loop()

        # accumulate
        if dmk_results is None: dmk_results = loop_dmk_results
        else:
            for dn in dmk_results:
                dmk_results[dn]['wonH_IV'] += loop_dmk_results[dn]['wonH_IV']
                dmk_results[dn]['wonH'] += loop_dmk_results[dn]['wonH']

    return dmk_results