from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.mpython.mpdecor import proc_wait
from typing import Optional

from envy import DMK_MODELS_FD
from run.functions import run_PTR_game, build_from_names, copy_dmks, get_saved_dmks_names
from run.after_run.reports import results_report
from pologic.game_config import GameConfig

PUB =     {'publish_player_stats':True,  'publishFWD':True,  'publishUPD':True}
PUB_REF = {'publish_player_stats':False, 'publishFWD':False, 'publishUPD':False}


@proc_wait
def run(game_config_name: Optional[str]=    None,   # must be given if not saved
        do_TR=                              True,   # True / False (TR/PL)
        n_tables=                           1000,
        game_size=                          1000000,
        n_gpu=                              2,
        use_saved=                          True,   # tries to use DMKs saved in DMK_MODELS_FD, creates new if no DMKs
            # parameters below are used only when DMKs are not saved before
        family=                             'a',
        n_dmk=                              20,
        n_refs=                             0,      # 0 or more, sets (copies) first n_refs from DMKs as refs
        ):
    """ with this script may be run simple game of:
    - DMKs may be present in a DMK_MODELS_FD folder or may be generated from scratch
    - TR / TS mode, with or without refs """

    mode = 'TR' if do_TR else 'PL'
    logger = get_pylogger(
        name=       f'simple_{mode}',
        folder=     DMK_MODELS_FD,
        level=      20,
        #flat_child= True,
    )
    sub_logger = get_child(logger)
    logger.info(f'Starting game {mode}')

    game_config = None
    dmk_names = []
    dmk_refs = []

    if use_saved:

        dmk_names = get_saved_dmks_names()
        dmk_refs = [dn for dn in dmk_names if dn.endswith('R')]
        for dn in dmk_refs:
            dmk_names.remove(dn)

        if dmk_names:

            game_config = GameConfig.from_name(folder=DMK_MODELS_FD)

            logger.info(f'using saved DMKs ({len(dmk_names)}): {dmk_names}')
            logger.info(f'using saved DMK refs ({len(dmk_refs)}): {dmk_refs}')
            logger.info(f'and game_config: {game_config}')

    if not dmk_names:

        game_config = GameConfig.from_name(name=game_config_name, copy_to=DMK_MODELS_FD)
        logger.info(f'> game config name: {game_config.name}')

        families = family * n_dmk
        families = families[:n_dmk]
        dmk_names = [f'dmk_{f}{ix:02}' for ix,f in enumerate(families)]

        build_from_names(
            game_config=    game_config,
            names=          dmk_names,
            families=       families,
            oversave=       False,
            logger=         logger)

        if n_refs:
            dmk_to_refs = dmk_names[:n_refs]
            dmk_refs = [f'{dn}R' for dn in dmk_to_refs]
            copy_dmks(
                names_src=  dmk_to_refs,
                names_trg=  dmk_refs,
                logger=     sub_logger)


    dmk_point_TRL = []
    dmk_point_PLL = []
    dmk_pointL = [
        {'name':dn, 'motorch_point':{'device':n%n_gpu if n_gpu else None}, **PUB}
        for n,dn in enumerate(dmk_names)]
    if do_TR: dmk_point_TRL = dmk_pointL
    else:     dmk_point_PLL = dmk_pointL

    dmk_point_refL = [
        {'name':dn, 'motorch_point':{'device':n%n_gpu if n_gpu else None}, **PUB_REF}
        for n, dn in enumerate(dmk_refs)]

    rgd = run_PTR_game(
        game_config=    game_config,
        name=           f'GM_{mode}',
        dmk_point_refL= dmk_point_refL,
        dmk_point_TRL=  dmk_point_TRL,
        dmk_point_PLL=  dmk_point_PLL,
        game_size=      game_size,
        n_tables=       n_tables,
        logger=         logger)
    dmk_results = rgd['dmk_results']
    logger.info(f'game results:\n{results_report(dmk_results)}')


if __name__ == "__main__":
    run(
        game_config_name=   '3players_2bets',
        #do_TR=              False,
        family=             'adp',
        n_dmk=              18,
        n_refs=             0,
        )