from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.files import r_json

from envy import DMK_MODELS_FD, RESULTS_FP
from run.functions import run_GM, copy_dmks



if __name__ == "__main__":

    # sample testing script

    game_size = 100000

    logger = get_pylogger(
        name=   'simple_test',
        folder= DMK_MODELS_FD,
        level=  20)

    loop_results = r_json(RESULTS_FP)
    refs_ranked = loop_results['refs_ranked']
    logger.info(f'Starting test game for {" ".join(refs_ranked)}')

    names = [nm[:-4] for nm in refs_ranked]
    copy_dmks(
        names_src=  refs_ranked,
        names_trg=  names,
        logger=     get_child(logger, change_level=10))

    pub_TS = {
        'publish_player_stats': False,
        'publish_pex':          False,
        'publish_update':       False,
        'publish_more':         False}

    for l in range(3):

        rgd = run_GM(
            dmk_point_ref=  [{'name':dn, 'motorch_point':{'device':0}, 'fwd_stats_iv':1000, **pub_TS} for dn in refs_ranked],
            dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':1}, 'fwd_stats_iv':1000, **pub_TS} for dn in names],
            game_size=      game_size,
            dmk_n_players=  150,
            sep_all_break=  True,
            sep_n_stdev=    1.0,
            logger=         logger)
        dmk_results = rgd['dmk_results']

        dmk_rw = [(dn, dmk_results[dn]['last_wonH_afterIV']) for dn in dmk_results]
        dmk_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x: x[1], reverse=True)]

        res_nfo = ''
        for dn in dmk_ranked:
            stats_nfo = ''
            for k in dmk_results[dn]["global_stats"]:
                v = dmk_results[dn]["global_stats"][k]
                stats_nfo += f'{k}: {v:4.1f} '
            res_nfo += f'{dn:30} : {dmk_results[dn]["last_wonH_afterIV"]:6.2f}    {stats_nfo}\n'
        logger.info(f'Test game results:\n{res_nfo}')