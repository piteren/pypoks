from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.mpython.mpdecor import proc_wait
from pypaq.lipytools.printout import stamp

from envy import DMK_MODELS_FD, load_game_config
from run.functions import run_GM, build_from_names, results_report, copy_dmks

PUB =     {'publish_player_stats':True,  'publish_pex':True,  'publishFWD':True,  'publishUPD':True}
PUB_REF = {'publish_player_stats':False, 'publish_pex':False, 'publishFWD':False, 'publishUPD':False}


# sample training function
@proc_wait
def run(game_config_name: str):

    # presets
   #n_gpu, n_dmk, dmk_n_players, game_size = 0,  2, 20,  10000      # CPU min
   #n_gpu, n_dmk, dmk_n_players, game_size = 0,  2, 300, 300000     # CPU medium
   #n_gpu, n_dmk, dmk_n_players, game_size = 2,  2, 100, 300000     # GPU medium
    n_gpu, n_dmk, dmk_n_players, game_size = 2,  8, 100, 1000000    # 2xGPU high

    family = 'a'
    do_TR = True#False # True / False
    n_refs = 0#6    # 0 or more

    logger = get_pylogger(
        name=       f'simple_train_{family}',
        folder=     DMK_MODELS_FD,
        level=      20,
        #flat_child= True,
    )
    sub_logger = get_child(logger)

    game_config = load_game_config(name=game_config_name, copy_to=DMK_MODELS_FD)

    #st = f'{stamp(month=False, day=False, letters=None)}_'
    st = ''
    names = [f'dmk_{st}{family}{ix:02}' for ix in range(n_dmk)]

    build_from_names(
        game_config=    game_config,
        names=          names,
        families=       [family]*n_dmk,
        oversave=       False,
        logger=         logger)

    dmk_refs = []
    if n_refs:
        dmk_to_ref = names[:n_refs]
        dmk_refs = [f'{dn}R' for dn in dmk_to_ref]
        copy_dmks(
            names_src=  dmk_to_ref,
            names_trg=  dmk_refs,
            logger=     sub_logger)

    dmk_point_TRL = []
    dmk_point_PLL = []
    dmk_pointL = [
        {'name':dn, 'motorch_point':{'device':n%n_gpu if n_gpu else None}, **PUB}
        for n,dn in enumerate(names)]
    if do_TR: dmk_point_TRL = dmk_pointL
    else:     dmk_point_PLL = dmk_pointL

    dmk_point_refL = [
        {'name':dn, 'motorch_point':{'device':n%n_gpu if n_gpu else None}, **PUB_REF}
        for n, dn in enumerate(dmk_refs)]

    rgd = run_GM(
        game_config=    game_config,
        name=           f'GM_{family}',
        dmk_point_refL= dmk_point_refL,
        dmk_point_TRL=  dmk_point_TRL,
        dmk_point_PLL=  dmk_point_PLL,
        game_size=      game_size,
        dmk_n_players=  dmk_n_players,
        logger=         logger)
    dmk_results = rgd['dmk_results']
    logger.info(f'simple train game results:\n{results_report(dmk_results)}')


if __name__ == "__main__":
    run('2players_2bets')