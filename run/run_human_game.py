from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.files import r_json

from envy import TR_RESULTS_FP, DMK_MODELS_FD
from pologic.game_config import GameConfig
from podecide.game_manager import HumanGameManager
from run.functions import build_single_foldmk


if __name__ == "__main__":

    logger = get_pylogger(
        name=       f'human_game',
        folder=     DMK_MODELS_FD,
        level=      10,
        #flat_child= True,
    )
    logger.info("Human Game starts")

    loop_results = r_json(TR_RESULTS_FP)
    if loop_results:

        game_config = GameConfig.from_name(folder=DMK_MODELS_FD)
        n_ai_players = game_config.table_size - 1
        logger.info(f"> {game_config.table_size} table players")

        key = 'refs_ranked' if 'refs_ranked' in loop_results else 'dmk_ranked'
        dmks_ranked = loop_results[key]
        dmk_names_ai = dmks_ranked[:n_ai_players]   # bests from ranked
        #dmk_names_ai = [dmks_ranked[1]]             # if you want to use specific player/s
        logger.info(f'got players from loop results: {dmk_names_ai}')
    else:
        game_config = GameConfig.from_name('3players_2bets')
        n_ai_players = game_config.table_size - 1
        dmk_names_ai = [f'dmk{ix}' for ix in range(n_ai_players)]
        for nm in dmk_names_ai:
            build_single_foldmk(
                game_config=    game_config,
                name=           nm,
                family=         'b',
                oversave=       False,
                logger=         logger)

    gm = HumanGameManager(
        dmk_names=      dmk_names_ai,
        game_config=    game_config,
        logger=         logger,
        #debug_dmks=     True,
        debug_tables=   True,
    )
    gm.start_game()