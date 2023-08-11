from pypaq.lipytools.pylogger import get_pylogger, get_child

from envy import DMK_MODELS_FD
from run.functions import run_GM, build_from_names, copy_dmks



if __name__ == "__main__":

    # sample testing script

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

    pub_TR = {
        'publish_player_stats': False,
        'publish_pex':          False,
        'publish_update':       True,
        'publish_more':         False}

    for l in range(3):

        rgd = run_GM(
            dmk_point_ref=  [{'name':dn, 'motorch_point':{'device':0}, 'fwd_stats_iv':1000, **pub_TR} for n,dn in enumerate(names_ref)],
            dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':1}, 'fwd_stats_iv':1000, **pub_TR} for n,dn in enumerate(names)],
            game_size=      game_size,
            dmk_n_players=  150,
            logger=         logger)
        dmk_results = rgd['dmk_results']

        dmk_ranked = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in dmk_results]
        dmk_ranked = [e[0] for e in sorted(dmk_ranked, key=lambda x: x[1], reverse=True)]

        res_nfo = ''
        for dn in dmk_ranked:
            stats_nfo = ''
            for k in dmk_results[dn]["global_stats"]:
                v = dmk_results[dn]["global_stats"][k]
                stats_nfo += f'{k}: {v:4.1f} '
            res_nfo += f'{dn:30} : {dmk_results[dn]["wonH_afterIV"][-1]:6.2f}    {stats_nfo}\n'
        logger.info(f'Play game results:\n{res_nfo}')