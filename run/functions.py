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
from podecide.dmk import FolDMK
from podecide.dmk_motorch import DMK_MOTorch_PG, DMK_MOTorch_PPO
from podecide.games_manager import GamesManager_PTR


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


def get_fresh_dna(game_config: Dict, name:str, family:str) -> POINT:

    # a - PG
    # b - PPO

    if family not in 'ab':
        raise PyPoksException(f'unknown family: {family}')

    motorch_psdd: PSDD = {
        'baseLR':                   [1e-6,  1e-4] if family =='a' else [1e-8,  1e-5],
        'opt_alpha':                [0.7,   0.9],
        'opt_beta':                 [0.3,   0.7],
        'opt_amsgrad':              (True,  False),
        'train_ce':                 (True,  False),
        'reward_norm':              (True,  False),
        'nam_loss_coef':            (0.0, 0.2, 0.5, 1.0, 1.5),
    }
    if family == 'a':
        motorch_psdd.update({
            'use_rce':              (True, False),
            'gc_do_clip':           (True, False),
        })
    if family == 'b':
        motorch_psdd.update({
            'gc_max_clip':          (0.3, 0.5, 0.7),
            'gc_max_upd':           (1.1, 1.2, 1.5),
            'clip_coef':            (0.1, 0.2, 0.3),
            'entropy_coef':         (0.0, 0.005, 0.01),
            'minibatch_num':        (3,4,5,6,7),
            'n_epochs_ppo':         (1,2,3),
        })

    motorch_point_common: POINT = {
        'family':                   family,
        'cards_emb_width':          12,
        'n_lay':                    12,
        'load_cardnet_pretrained':  True,
        'psdd':                     motorch_psdd,
    }

    foldmk_psdd: PSDD = {
        'upd_trigger':              [20000, 50000] if family == 'a' else [40000, 80000],
        'reward_share':             (None,  3,5,6),
    }
    if family == 'a':
        foldmk_psdd.update({
                # ExaDMK
            'enable_pex':           (True, False),
            'pex_max':              [0.01,  0.05],
            'prob_zero':            [0.0,   0.45],
            'prob_max':             [0.0,   0.45],
            'step_min':             [100,   10000],
            'step_max':             [10000, 100000],
            'pid_pex_fraction':     [0.5,   1.0],
        })

    foldmk_point_common: POINT = {
        'name':                     name,
        'family':                   family,
        'motorch_type':             DMK_MOTorch_PG if family == 'a' else DMK_MOTorch_PPO,
        'trainable':                True,
        'enable_pex':               False,
        'psdd':                     foldmk_psdd,
    }

    foldmk_point = {k: game_config[k] for k in ['table_size', 'table_moves', 'table_cash_start']}
    foldmk_point.update(foldmk_point_common)
    paspa_foldmk = PaSpa(psdd=foldmk_psdd, loglevel=30)
    foldmk_point.update(paspa_foldmk.sample_point_GX())

    motorch_point = {}
    motorch_point.update(motorch_point_common)
    paspa_motorch = PaSpa(psdd=motorch_psdd, loglevel=30)
    motorch_point.update(paspa_motorch.sample_point_GX())
    foldmk_point['motorch_point'] = motorch_point

    # TODO: optimizers: torch.optim.RAdam torch.optim.RMSprop

    return foldmk_point


def build_single_foldmk(
        game_config: Dict,
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
        dmk_point=  get_fresh_dna(
            game_config=    game_config,
            name=           name,
            family=         family),
        logger=     logger)


def pretrain(
        game_config: Dict,
        n_dmk_total: int,                   # final total number of pretrained DMKs
        families: str,
        multi_pretrain: int=    3,          # total multiplication of DMKs for pretrain, for 1 there is no pretrain
        n_dmk_TR_group: int=    10,         # DMKs group size while TR
        game_size_TR=           100000,
        dmk_n_players_TR=       150,
        n_dmk_TS_group: int=    20,         # DMKs group size while TS
        game_size_TS=           100000,
        dmk_n_players_TS=       150,
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


def build_from_names(
        game_config: Dict,
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

@proc_return
def run_GM(
        logger,
        game_config: Dict,
        name: Optional[str]=                        None,
        dmk_point_refL: Optional[List[Dict]]=       None,
        dmk_point_PLL: Optional[List[Dict]]=        None,
        dmk_point_TRL: Optional[List[Dict]]=        None,
        game_size: int=                             100000,
        dmk_n_players: int=                         60,
        sep_all_break: bool=                        False,
        sep_pairs: Optional[List[Tuple[str,str]]]=  None,
        sep_pairs_factor: float=                    0.9,
        sep_n_stddev: float=                        1.0,
        publish: bool=                              True,
) -> Dict[str, Dict]:
    """ runs GM game in a subprocess """

    gm = GamesManager_PTR(
        game_config=    game_config,
        name=           name,
        dmk_point_refL= dmk_point_refL,
        dmk_point_PLL=  dmk_point_PLL,
        dmk_point_TRL=  dmk_point_TRL,
        dmk_n_players=  dmk_n_players,
        logger=         logger)
    return gm.run_game(
        game_size=          game_size,
        publish=            publish,
        sep_all_break=      sep_all_break,
        sep_pairs=          sep_pairs,
        sep_pairs_factor=   sep_pairs_factor,
        sep_n_stddev=       sep_n_stddev)


def results_report(
        dmk_results: Dict[str, Dict],
        dmks: Optional[List[str]]=  None,
) -> str:

    if not dmks:
        dmks = list(dmk_results.keys())

    res_nfo = ''
    for ix,dn in enumerate(sorted(dmks, key=lambda x: dmk_results[x]['last_wonH_afterIV'], reverse=True)):

        wonH = dmk_results[dn]['last_wonH_afterIV']
        wonH_IV_std = dmk_results[dn]['wonH_IV_stddev']
        wonH_mstd = dmk_results[dn]['wonH_IV_mean_stddev']
        wonH_mstd_str = f'[{wonH_IV_std:.2f}/{wonH_mstd:.2f}]'

        stats_nfo = ''
        for k in dmk_results[dn]["global_stats"]:
            v = dmk_results[dn]["global_stats"][k]
            stats_nfo += f'{k}:{v*100:4.1f} '

        wonH_diff = f"diff: {dmk_results[dn]['wonH_diff']:5.2f}" if 'wonH_diff' in dmk_results[dn] else ""
        sep = f" sepF: {dmk_results[dn]['separated_factor']:4.2f}" if 'separated_factor' in dmk_results[dn] else ""
        lifemark = f" {dmk_results[dn]['lifemark']}" if 'lifemark' in dmk_results[dn] else ""

        res_nfo += f'{ix:2} {dn:18} : {wonH:6.2f} {wonH_mstd_str:>12}  {stats_nfo}  {wonH_diff}{sep}{lifemark}\n'

    if res_nfo:
        res_nfo = res_nfo[:-1]
    return res_nfo