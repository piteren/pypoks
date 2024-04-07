""" with this script *one hand of one DMK playing on a single table may be run
* actual number of played hands may be higher, since asynchronous nature of the framework """

from pypaq.lipytools.pylogger import get_pylogger

from envy import DMK_MODELS_FD
from run.functions import build_from_names
from pologic.game_config import GameConfig
from podecide.game_manager import GameManager_PTR


def run(game_config_name:str='3players_2bets', family='a'):

    logger = get_pylogger(
        name=       f'toy_game',
        folder=     DMK_MODELS_FD,
        level=      10,
        #flat_child= True,
    )
    logger.info(f'Starting a toy game')

    game_config = GameConfig.from_name(name=game_config_name, copy_to=DMK_MODELS_FD)
    logger.info(f'> game config name: {game_config.name}')

    dmk_name = f'dmk_{family}'

    build_from_names(
        game_config=    game_config,
        names=          [dmk_name],
        families=       [family],
        oversave=       True,
        logger=         logger)

    dmk_point = {
        'name': dmk_name,
        'motorch_point': {'device':None},
        'publish_player_stats': False, 'publishFWD': False, 'publishUPD': False}

    gm = GameManager_PTR(
        game_config=    game_config,
        name=           'GM_toy_game',
        gm_loop=        None,
        dmk_point_refL= [],
        dmk_point_PLL=  [dmk_point],
        dmk_point_TRL=  [],
        n_tables=       1,
        logger=         logger,
        debug_dmks=     True,
        debug_tables=   True)

    gm.run_game(game_size=1, sleep=0, publish=False)


if __name__ == "__main__":
    run()