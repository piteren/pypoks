import random

from pypaq.lipytools.files import list_dir, prep_folder, r_json
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.mpython.mpdecor import proc_return
from pypaq.pms.base import POINT, PSDD
from pypaq.pms.paspa import PaSpa
import select
import shutil
import sys
from typing import List, Optional, Dict, Tuple, Union

from envy import DMK_MODELS_FD, TR_RESULTS_FP, PyPoksException
from pologic.game_config import GameConfig
from podecide.dmk import FolDMK
from podecide.dmk_motorch import DMK_MOTorch, DMK_MOTorch_PG, DMK_MOTorch_A2C, DMK_MOTorch_PPO
from podecide.game_manager import GameManager_PTR


def check_continuation() -> Dict:
    """ checks for run loops continuation """
    continuation = False
    tr_results = r_json(TR_RESULTS_FP)
    if tr_results:
        saved_n_loops = int(tr_results['loop_ix'])
        print(f'Do you want to continue with saved ({DMK_MODELS_FD}) {saved_n_loops} loops? ..waiting 10 sec (y/n, y-default)')
        i, o, e = select.select([sys.stdin], [], [], 10)
        if i and sys.stdin.readline().strip() == 'n':
            # clean out dmk folder
            shutil.rmtree(f'{DMK_MODELS_FD}', ignore_errors=True)
            print(f'cleaned out {DMK_MODELS_FD}')
        else:
            continuation = True
    return {
        'continuation':     continuation,
        'training_results': tr_results}


def dmk_name(
        loop_ix: int,
        family: str,
        counter: int,
        age: Optional[int]= None,
        is_ref: bool=       False,
) -> str:
    """ prepares DMK name from parameters"""
    age_nfo = f'_{age:03}' if age is not None else ''
    ref_nfo = 'R' if is_ref else ''
    return f'dmk{loop_ix:03}{family}{counter:02}{age_nfo}{ref_nfo}'


def get_saved_dmks_names(folder:Optional[str]=None) -> List[str]:
    if not folder: folder = FolDMK.SAVE_TOPDIR
    all_dirs = list_dir(folder)['dirs']
    return [d for d in all_dirs if d.startswith('dmk')]


def get_fresh_dna(game_config:GameConfig, name:str, family:str) -> POINT:

    # a - PG 12
    # b - PG 24
    # c - PG 12 small
    # d - A2C 12
    # p - PPO 12

    if family not in 'abcdp':
        raise PyPoksException(f'unknown family: {family}')

    motorch_psdd: PSDD = {
        'baseLR':                   [5e-6, 1e-5],
        #'nam_loss_coef':            (1.5, 2.3, 3.0, 5.0, 10.0, 20.0),
        #'entropy_coef':             (0.0, 0.01, 0.02, 0.05),
    }

    motorch_point_common: POINT = {
        'family':                   family,
        'cards_emb_width':          12,
        'n_lay':                    12,
        'load_cardnet_pretrained':  True,
        'psdd':                     motorch_psdd,
        'device':                   None,
    }

    if family == 'b':
        motorch_point_common['cards_emb_width'] = 24

    if family == 'c':
        motorch_point_common.update({
            'event_emb_width':      4,
            'float_feat_size':      4,
            'player_id_emb_width':  4,
            'player_pos_emb_width': 4,
            'n_lay':                4,
        })

    foldmk_psdd: PSDD = {
        'upd_trigger':              [20000, 40000] if family != 'p' else [40000, 80000],
        #'reward_share':             (None,  3,5,6),
    }

    foldmk_point_common: POINT = {
        'name':                     name,
        'family':                   family,
        'motorch_type':             DMK_MOTorch_PG,
        'trainable':                True,
        'psdd':                     foldmk_psdd,
    }
    if family == 'd':
        foldmk_point_common['motorch_type'] = DMK_MOTorch_A2C
    if family == 'p':
        foldmk_point_common['motorch_type'] = DMK_MOTorch_PPO

    foldmk_point = {
        'table_size':       game_config.table_size,
        'table_moves':      game_config.table_moves,
        'table_cash_start': game_config.table_cash_start}
    foldmk_point.update(foldmk_point_common)
    paspa_foldmk = PaSpa(psdd=foldmk_psdd, loglevel=30)
    foldmk_point.update(paspa_foldmk.sample_point_GX())

    motorch_point = {}
    motorch_point.update(motorch_point_common)
    paspa_motorch = PaSpa(psdd=motorch_psdd, loglevel=30)
    motorch_point.update(paspa_motorch.sample_point_GX())
    foldmk_point['motorch_point'] = motorch_point

    return foldmk_point


def build_single_foldmk(
        game_config: GameConfig,
        name:str,
        family:str,
        oversave=   True, # if DMK exists, True for oversave creates fresh DMK, otherwise existing DMK will be used (-> no new build)
        logger=     None,
        loglevel=   20,
) -> None:
    """ builds single FolDMK
    saves into folder """

    if not logger:
        logger = get_pylogger(name='build_single_foldmk', level=loglevel)
    logger.info(f'Building DMK {name}')

    if name in get_saved_dmks_names():
        if oversave:
            shutil.rmtree(f'{DMK_MODELS_FD}/{name}', ignore_errors=True)
            logger.info(f'> {name} was deleted from {DMK_MODELS_FD}')
        else:
            logger.info(f'> {name} exists, was left')
            return

    FolDMK.build_from_point(
        dmk_point=  get_fresh_dna( game_config=game_config, name=name, family=family),
        logger=     logger)


def pretrain(
        game_config: GameConfig,
        n_dmk_total: int,                   # final total number of pretrained DMKs
        families: str,
        multi_pretrain: int=    3,          # total multiplication of DMKs for pretrain, for 1 there is no pretrain
        n_dmk_TR_group: int=    10,         # DMKs group size while TR
        game_size_TR=           100000,
        n_tables=               1000,
        n_dmk_TS_group: int=    20,         # DMKs group size while TS
        game_size_TS=           100000,
        logger=                 None):
    """ prepares some pretrained DMKs """

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
        game_config=    game_config,
        names=          names,
        families=       fam,
        logger=         logger)

    if multi_pretrain > 1:

        pub = {
            'publish_player_stats': False,
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
            run_PTR_game(
                dmk_point_TRL=  [{'name':nm, 'motorch_point':{'device':n%2}, **pub} for n,nm in enumerate(tg)],
                game_size=      game_size_TR,
                n_tables=       n_tables,
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
            rgd = run_PTR_game(
                dmk_point_PLL=  [{'name':nm, 'motorch_point':{'device':n%2}, **pub} for n,nm in enumerate(tg)],
                game_size=      game_size_TS,
                n_tables=       n_tables,
                logger=         logger)
            dmk_results.update(rgd['dmk_results'])

        ### 3. select best, remove rest
        dmk_ranked = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in names]
        dmk_ranked = [e[0] for e in sorted(dmk_ranked, key=lambda x: x[1], reverse=True)]
        for dn in dmk_ranked[n_dmk_total:]:
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)


def build_from_names(
        game_config: GameConfig,
        names: List[str],
        families: Union[str,List[str]],
        oversave=   True,
        logger=     None,
        loglevel=   20):
    """ builds dmks, checks folder for present ones """

    if not logger: logger = get_pylogger(level=loglevel)

    prep_folder(DMK_MODELS_FD)
    dmks_in_dir = get_saved_dmks_names()
    logger.info(f'got dmks: {dmks_in_dir} already in {DMK_MODELS_FD}')

    sub_logger = get_child(logger, change_level=10)
    for nm,fm in zip(names,families):
        if nm not in dmks_in_dir:
            logger.info(f'> building: {nm}, family: {fm}')
            build_single_foldmk(
                game_config=    game_config,
                name=           nm,
                family=         fm,
                oversave=       oversave,
                logger=         sub_logger)


def copy_dmks(
        names_src: List[str],
        names_trg: List[str],
        save_topdir_src: str=           DMK_MODELS_FD,
        save_topdir_trg: Optional[str]= None,
        logger=                         None,
        loglevel=                       30):
    """ copies list of FolDMK
    wrapped into a subprocess to separate torch """
    if not logger:
        logger = get_pylogger(name='copy_dmks', level=loglevel)
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

# helper to multiply LR of all saved DDMK_MOTorch
def set_LR(mul=0.5):
    for nm in get_saved_dmks_names():
        point = DMK_MOTorch.load_point(name=nm)
        print(nm, point)
        DMK_MOTorch.oversave_point(name=nm, baseLR=point['baseLR']*mul)

@proc_return
def run_PTR_game(
        logger,
        game_config: GameConfig,
        name: Optional[str]=                        None,
        gm_loop: Optional[int]=                     None,
        dmk_point_refL: Optional[List[Dict]]=       None,
        dmk_point_PLL: Optional[List[Dict]]=        None,
        dmk_point_TRL: Optional[List[Dict]]=        None,
        game_size: int=                             100000,
        n_tables: int=                              1000,
        sep_all_break: bool=                        False,
        sep_pairs: Optional[List[Tuple[str,str]]]=  None,
        sep_pairs_factor: float=                    0.9,
        sep_n_stddev: float=                        1.0,
        publish: bool=                              True,
) -> Dict[str, Dict]:
    """ runs GM PTR game in a subprocess """
    gm = GameManager_PTR(
        game_config=    game_config,
        name=           name,
        gm_loop=        gm_loop,
        dmk_point_refL= dmk_point_refL,
        dmk_point_PLL=  dmk_point_PLL,
        dmk_point_TRL=  dmk_point_TRL,
        n_tables=       n_tables,
        logger=         logger)
    return gm.run_game(
        game_size=          game_size,
        publish=            publish,
        sep_all_break=      sep_all_break,
        sep_pairs=          sep_pairs,
        sep_pairs_factor=   sep_pairs_factor,
        sep_n_stddev=       sep_n_stddev)


if __name__ == "__main__":
    set_LR(mul=0.5)