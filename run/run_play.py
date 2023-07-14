from pypaq.lipytools.pylogger import get_pylogger

from envy import DMK_MODELS_FD
from run.functions import get_saved_dmks_names, run_GM


if __name__ == "__main__":

    # INFO: although game size is quite big, game will be broken as soon as DMKs become separated
    sep_n_stdev = 1.0 # for deeper test use at least 2.0
    game_size = 2000000

    logger = get_pylogger(
        name=   'run_play',
        folder= DMK_MODELS_FD,
        level=  20)

    names = get_saved_dmks_names()
    logger.info(f'running play game for {len(names)} DMKs: {names}')

    pub = {
        'publish_player_stats': True,
        'publish_pex':          False, # won't matter since PL does not pex
        'publish_update':       False,
        'publish_more':         False}

    rgd = run_GM(
        dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':n%2}, **pub} for n,dn in enumerate(names)],
        game_size=      game_size,
        dmk_n_players=  (120 // len(names)) * 30,
        sep_all_break=  True,
        sep_n_stdev=    sep_n_stdev,
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