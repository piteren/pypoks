import random

from pypaq.lipytools.files import list_dir, prep_folder
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.mpython.mpdecor import proc_return, proc_wait
from pypaq.pms.base import POINT, PSDD
from pypaq.pms.paspa import PaSpa
import shutil
from typing import List, Optional, Dict, Tuple, Union

from envy import DMK_MODELS_FD, PyPoksException
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


def pretrain(
        n_dmk_total: int=       10,         # final total number of pretrained DMKs
        families=               'abcd',
        multi_pretrain: int=    3,          # total multiplication of DMKs for pretrain, for 1 there is no pretrain
        n_dmk_TR_group: int=    10,         # DMKs group size while TR
        game_size_TR=           100000,
        dmk_n_players_TR=       150,
        n_dmk_TS_group: int=    20,         # DMKs group size while TS
        game_size_TS=           100000,
        dmk_n_players_TS=       150,
        logger=                 None):

    if not logger:
        logger = get_pylogger(
            name=   'pretrain',
            folder= DMK_MODELS_FD,
            level=  20)
    logger.info(f'running pretrain for {n_dmk_total} DMKs from families: {families}, multi: {multi_pretrain}')

    ### 0. create DMKs

    fam = families * n_dmk_total * multi_pretrain
    fam = fam[:n_dmk_total * multi_pretrain]

    names = [f'dmk00{f}{ix:02}' for ix,f in enumerate(fam)]

    build_from_names(
        names=      names,
        families=   fam,
        logger=     logger)

    if multi_pretrain > 1:

        pub = {
            'publish_player_stats': False,
            'publish_pex':          False,
            'publish_update':       False,
            'publish_more':         False}

        ### 1. train DMKs in groups

        tr_names = [] + names
        random.shuffle(tr_names)
        tr_groups = []
        while tr_names:
            tr_groups.append(tr_names[:n_dmk_TR_group])
            tr_names = tr_names[n_dmk_TR_group:]
        for tg in tr_groups:
            run_GM(
                dmk_point_TRL=  [{'name':nm, 'motorch_point':{'device':n%2}, **pub} for n,nm in enumerate(tg)],
                game_size=      game_size_TR,
                dmk_n_players=  dmk_n_players_TR,
                logger=         logger)

        ### 2. test DMKs in groups

        dmk_results = {}
        ts_names = [] + names
        random.shuffle(ts_names)
        ts_groups = []
        while ts_names:
            ts_groups.append(ts_names[:n_dmk_TS_group])
            ts_names = ts_names[n_dmk_TS_group:]
        for tg in ts_groups:
            rgd = run_GM(
                dmk_point_PLL=  [{'name':nm, 'motorch_point':{'device':n%2}, **pub} for n,nm in enumerate(tg)],
                game_size=      game_size_TS,
                dmk_n_players=  dmk_n_players_TS,
                logger=         logger)
            dmk_results.update(rgd['dmk_results'])

        ### 3. select best, remove rest
        dmk_ranked = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in names]
        dmk_ranked = [e[0] for e in sorted(dmk_ranked, key=lambda x: x[1], reverse=True)]
        for dn in dmk_ranked[n_dmk_total:]:
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

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
        save_topdir_src: str=           DMK_MODELS_FD,
        save_topdir_trg: Optional[str]= None,
        logger=                         None,
        loglevel=                       30):
    if not logger: logger = get_pylogger(level=loglevel)
    for ns, nt in zip(names_src, names_trg):
        FolDMK.copy_saved(
            name_src=           ns,
            name_trg=           nt,
            save_topdir_src=    save_topdir_src,
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

@proc_return # processes <- to clear mem
def run_GM(
        logger,
        dmk_point_ref: Optional[List[Dict]]=        None,
        dmk_point_PLL: Optional[List[Dict]]=        None,
        dmk_point_TRL: Optional[List[Dict]]=        None,
        game_size: int=                             100000,
        dmk_n_players: int=                         60,
        sep_all_break: bool=                        False,
        sep_pairs: Optional[List[Tuple[str,str]]]=  None,
        sep_pairs_factor: float=                    0.9,
        sep_n_stdev: float=                         2.0,
        publish_GM: bool=                           False,
) -> Dict[str, Dict]:

    gm = GamesManager_PTR(
        dmk_point_ref=  dmk_point_ref,
        dmk_point_PLL=  dmk_point_PLL,
        dmk_point_TRL=  dmk_point_TRL,
        dmk_n_players=  dmk_n_players,
        logger=         logger)
    return gm.run_game(
        game_size=          game_size,
        publish_GM=         publish_GM,
        sep_all_break=      sep_all_break,
        sep_pairs=          sep_pairs,
        sep_pairs_factor=   sep_pairs_factor,
        sep_n_stdev=        sep_n_stdev)


def results_report(
        dmk_results: Dict[str, Dict],
        dmks: Optional[List[str]]=  None
) -> str:

    if not dmks:
        dmks = list(dmk_results.keys())
    dmk_rw = [(dn, dmk_results[dn]['last_wonH_afterIV']) for dn in dmks]
    dmk_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x: x[1], reverse=True)]

    res_nfo = ''
    for dn in dmk_ranked:
        wonH = dmk_results[dn]['last_wonH_afterIV']
        wonH_mstd = dmk_results[dn]['wonH_IV_mean_stdev']
        wonH_mstd_str = f'[{wonH_mstd:.2f}]'
        stats_nfo = ''
        for k in dmk_results[dn]["global_stats"]:
            v = dmk_results[dn]["global_stats"][k]
            stats_nfo += f'{k}:{v:4.1f} '
        res_nfo += f'{dn} : {wonH:6.2f} {wonH_mstd_str:7}    {stats_nfo}\n'

    return res_nfo