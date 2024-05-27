from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.files import r_json

from envy import TR_RESULTS_FP, DMK_MODELS_FD
from pologic.game_config import GameConfig
from podecide.game_manager import TestGameManager


if __name__ == "__main__":

    logger = get_pylogger(
        name=       f'test_game',
        folder=     DMK_MODELS_FD,
        level=      10,
        #flat_child= True,
    )
    logger.info("test game starts..")

    loop_results = r_json(TR_RESULTS_FP)

    game_config = GameConfig.from_name(folder=DMK_MODELS_FD)
    n_ai_players = game_config.table_size - 1
    logger.info(f"> {game_config.table_size} table players")

    key = 'refs_ranked' if 'refs_ranked' in loop_results else 'dmk_ranked'
    dmks_ranked = loop_results[key]
    dmk_agent_name = dmks_ranked[0]
    logger.info(f'got players from loop results: {dmk_agent_name}')

    gm = TestGameManager(
        dmk_agent_name= dmk_agent_name,
        game_config=    game_config,
        logger=         logger,
        debug_tables=   True,
    )
    gm.run_test_game()
