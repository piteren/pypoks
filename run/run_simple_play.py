from pypaq.lipytools.pylogger import get_pylogger, get_child

from envy import DMK_MODELS_FD
from run.functions import run_GM, build_from_names, copy_dmks, results_report



if __name__ == "__main__":

    # sample playing script

    n_dmk = 10
    game_size = 100000

    logger = get_pylogger(
        name=   'simple_play',
        folder= DMK_MODELS_FD,
        level=  20)

    names = [f'dmk000_{ix}' for ix in range(n_dmk)]

    build_from_names(
        names=      names,
        families=   ['a']*n_dmk,
        logger=     logger)

    names_ref = [f'{nm}_ref' for nm in names]
    copy_dmks(
        names_src=  names,
        names_trg=  names_ref,
        logger=     get_child(logger, change_level=10))

    pub_TS = {
        'publish_player_stats': False,
        'publish_pex':          False,
        'publish_update':       False,
        'publish_more':         False}

    for l in range(3):

        rgd = run_GM(
            dmk_point_ref=  [{'name':dn, 'motorch_point':{'device':0}, **pub_TS} for dn in names_ref],
            dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':1}, **pub_TS} for dn in names],
            game_size=      game_size,
            dmk_n_players=  150,
            logger=         logger)
        dmk_results = rgd['dmk_results']
        logger.info(f'Play game results:\n{results_report(dmk_results)}')