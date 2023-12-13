from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.files import r_json

from envy import DMK_MODELS_FD, TR_RESULTS_FP, load_game_config
from run.functions import run_GM, copy_dmks, results_report

PUB = {'publish_player_stats':False, 'publish_pex':False, 'publishFWD':False, 'publishUPD':False}


def run(game_size :int):
    """ sample testing script """

    logger = get_pylogger(
        name=   'simple_test',
        folder= DMK_MODELS_FD,
        level=  20)

    game_config = load_game_config(folder=DMK_MODELS_FD)

    loop_results = r_json(TR_RESULTS_FP)
    refs_ranked = loop_results['refs_ranked']
    logger.info(f'Starting test game for: {", ".join(refs_ranked)}')

    names = [nm[:-4] for nm in refs_ranked]
    copy_dmks(
        names_src=  refs_ranked,
        names_trg=  names,
        logger=     get_child(logger, change_level=10))

    for l in range(3):

        rgd = run_GM(
            game_config=    game_config,
            dmk_point_refL= [{'name':dn, 'motorch_point':{'device':0}, **PUB} for dn in refs_ranked],
            dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':1}, **PUB} for dn in names],
            game_size=      game_size,
            dmk_n_players=  150,
            sep_all_break=  True,
            sep_n_stddev=   1.0,
            logger=         logger)
        dmk_results = rgd['dmk_results']
        logger.info(f'Test game results:\n{results_report(dmk_results)}')


if __name__ == "__main__":
    run(100000)