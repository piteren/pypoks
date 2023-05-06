from pypaq.lipytools.pylogger import get_pylogger
import shutil

from pypoks_envy import DMK_MODELS_FD
from run.functions import build_from_names, run_GM
from run_train_loop_V2 import CONFIG_INIT


def pretrain(
        n_fam_dmk=      3,
        families=       'abcd',
        game_size=      10000,
        dmk_n_players=  150,
        logger=         None):

    if not logger:
        logger = get_pylogger(
            name=   'pretrain',
            folder= DMK_MODELS_FD,
            level=  20)

    n_fam = len(families)

    logger.info(f'Running {n_fam} pretrain loops..')

    pub_TR = {
        'publish_player_stats': False,
        'publish_pex':          False,
        'publish_update':       False,
        'publish_more':         False}

    for f in families:

        initial_names = [f'dmk00{f}{ix:02}' for ix in range(n_fam*n_fam_dmk)]

        build_from_names(
            names=      initial_names,
            families=   f*len(initial_names),
            logger=     logger)

        dmk_results = run_GM(
            dmk_point_TRL=  [{'name':dn, 'motorch_point':{'device':n%2}, **pub_TR} for n,dn in enumerate(initial_names)],
            game_size=      game_size,
            dmk_n_players=  dmk_n_players,
            logger=         logger)['dmk_results']

        dmk_ranked = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in initial_names]
        dmk_ranked = [e[0] for e in sorted(dmk_ranked, key=lambda x: x[1], reverse=True)]
        for dn in dmk_ranked[n_fam_dmk:]:
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)


if __name__ == "__main__":

    pretrain(
        n_fam_dmk=      CONFIG_INIT['n_dmk_total'] // len(CONFIG_INIT['families']),
        families=       CONFIG_INIT['families'],
        game_size=      CONFIG_INIT['pretrain_game_size'],
        dmk_n_players=  CONFIG_INIT['dmk_n_players_TR'])