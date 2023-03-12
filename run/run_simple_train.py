from pypaq.lipytools.pylogger import get_pylogger

from pypoks_envy import MODELS_FD
from run.functions import run_GM, build_from_names



if __name__ == "__main__":

    n_gpus = 2
    n_dmk = 10
    game_size = 100000

    logger = get_pylogger(
        name=   'train_little',
        folder= MODELS_FD,
        level=  20)

    build_from_names(
        names=      [f'dmk000_{ix}' for ix in range(n_dmk)],
        families=   ['a']*n_dmk,
        logger=     logger)

    pub_TR = {
        'publish_player_stats': False,
        'publish_pex':          False,
        'publish_update':       True,
        'publish_more':         False}

    run_GM(
        dmk_point_TRL=  [{'name':dn, 'motorch_point':{'device':n%n_gpus}, 'fwd_stats_iv':1000, **pub_TR} for n,dn in enumerate(names)],
        game_size=      game_size,
        dmk_n_players=  150,
        logger=         logger)