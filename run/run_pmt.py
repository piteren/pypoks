""" PMT - Periodical Masters Test
test of bests agents saved from time to time while training """

from pypaq.lipytools.pylogger import get_pylogger

from envy import DMK_MODELS_FD
from run.functions import get_saved_dmks_names, run_PTR_game
from run.after_run.reports import results_report
from pologic.game_config import GameConfig

PUB_NONE = {'publish_player_stats':False, 'publishFWD':False, 'publishUPD':False}

PMT_FD = f'{DMK_MODELS_FD}/_pmt'


if __name__ == "__main__":

    logger = get_pylogger(
        name=       'pmt',
        add_stamp=  False,
        folder=     DMK_MODELS_FD,
        level=      20,
        #flat_child= True,
    )

    game_config = GameConfig.from_name(folder=DMK_MODELS_FD)

    all_pmt = get_saved_dmks_names(PMT_FD)

    rgd = run_PTR_game(
        game_config=    game_config,
        dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':i%2}, 'save_topdir':PMT_FD, **PUB_NONE} for i,dn in enumerate(all_pmt)],
        game_size=      300000,
        n_tables=       1000,
        logger=         logger,
        publish=        False)
    pmt_results = rgd['dmk_results']

    print(f'PMT results:\n{results_report(pmt_results)}')