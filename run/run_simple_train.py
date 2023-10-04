from pypaq.lipytools.pylogger import get_pylogger

from envy import DMK_MODELS_FD
from run.functions import run_GM, build_from_names, results_report


# sample training script with one GM
if __name__ == "__main__":

    # presets
   #n_gpu, n_dmk, dmk_n_players, game_size = 0,  2, 2,   1000       # CPU min
   #n_gpu, n_dmk, dmk_n_players, game_size = 0,  2, 300, 1000000    # CPU medium
    n_gpu, n_dmk, dmk_n_players, game_size = 2, 10, 200, 1000000    # 2xGPU high

    logger = get_pylogger(
        name=   'simple_train',
        folder= DMK_MODELS_FD,
        level=  20)

    names = [f'dmk0{ix:02}' for ix in range(n_dmk)]

    build_from_names(
        names=      names,
        families=   ['a']*n_dmk,
        logger=     logger)

    pub_TR = {
        'publish_player_stats': True,
        'publish_pex':          True,
        'publish_update':       True,
        'publish_more':         True}

    dmk_pointL = [{
        'name':             dn,
        'motorch_point':    {'device':n % n_gpu if n_gpu else None},
        'won_iv':           1000,
        **pub_TR,
    } for n,dn in enumerate(names)]

    rgd = run_GM(
        dmk_point_TRL=  dmk_pointL,
        game_size=      game_size,
        dmk_n_players=  dmk_n_players,
        sep_n_stdev=    1.0,
        logger=         logger,
        publish_GM=     True)
    dmk_results = rgd['dmk_results']
    logger.info(f'simple train game results:\n{results_report(dmk_results)}')