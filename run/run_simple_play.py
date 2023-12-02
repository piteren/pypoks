from pypaq.lipytools.pylogger import get_pylogger, get_child

from envy import DMK_MODELS_FD, load_game_config
from run.functions import run_GM, build_from_names, copy_dmks, results_report

PUB = {'publish_player_stats':True, 'publish_pex':True, 'publishFWD':True, 'publishUPD':True}


def run(
        game_config_name: str,
        n_dmk: int,
        n_dmk_refs: int,
        n_tables: int,
        game_size: int,
):
    """ sample playing script """

    game_config = load_game_config(game_config_name)

    logger = get_pylogger(
        name=   'simple_play',
        folder= DMK_MODELS_FD,
        level=  20)

    names = [f'dmk_{ix:02}' for ix in range(n_dmk)]

    build_from_names(
        game_config=    game_config,
        names=          names,
        families=       ['a']*n_dmk,
        oversave=       False,
        logger=         logger)

    names_ref = [f'{nm}R' for nm in names[:n_dmk_refs]]
    if names_ref:
        copy_dmks(
            names_src=  names,
            names_trg=  names_ref,
            logger=     get_child(logger, change_level=10))

    dmk_n_players = n_tables // n_dmk * game_config['table_size'] if not names_ref else n_tables // n_dmk
    rgd = run_GM(
        game_config=    game_config,
        dmk_point_refL= [{'name':dn, 'motorch_point':{'device':0}, **PUB} for dn in names_ref],
        dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':1}, **PUB} for dn in names],
        game_size=      game_size,
        dmk_n_players=  dmk_n_players,
        logger=         logger)
    dmk_results = rgd['dmk_results']
    logger.info(f'Play game results:\n{results_report(dmk_results)}')


if __name__ == "__main__":
    run(
        game_config_name=   '2players_2bets',
        n_dmk=              30,
        n_dmk_refs=         6,
        n_tables=           100,#2000,
        game_size=          100000,
    )