from pypaq.lipytools.pylogger import get_pylogger
import random

from pypoks_envy import MODELS_FD
from run.functions import build_from_names, get_saved_dmks_names, run_GM


if __name__ == "__main__":

    n_gpus = 2
    n_dmk = 10
    game_size = 100000

    logger = get_pylogger(
        name=   'simple_play',
        folder= MODELS_FD,
        level=  20)

    names = [dn for dn in get_saved_dmks_names() if '_old' not in dn]
    if not names:
        n_dmk = 10
        names = [f'dmk000_{ix}' for ix in range(n_dmk)]
        build_from_names(
            names=      names,
            families=   [random.choice('abc') for _ in range(n_dmk)],
            logger=     logger)
        logger.info(f'starting simple play of fresh {n_dmk} DMKs')
    else:
        logger.info(f'starting simple play saved DMKs: {names}')

    pub = {
        'publish_player_stats': True,
        'publish_pex':          False, # won't matter since PL does not pex
        'publish_update':       False,
        'publish_more':         False}

    dmk_results = run_GM(
        dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':n%n_gpus}, **pub} for n,dn in enumerate(names)],
        game_size=      game_size,
        dmk_n_players=  150,
        logger=         logger)['dmk_results']

    for dn in dmk_results:
        print(f'{dn} : {dmk_results[dn]}')