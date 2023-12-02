from pypaq.lipytools.files import r_json

from envy import RESULTS_FP, N_TABLE_PLAYERS
from podecide.games_manager import HuGamesManager
from run.functions import build_single_foldmk


if __name__ == "__main__":

    n_ai_players = N_TABLE_PLAYERS - 1

    loop_results = r_json(RESULTS_FP)
    if loop_results:
        refs_ranked = loop_results['refs_ranked']
        # dmk_names_ai = refs_ranked[:n_ai_players] # if you want best
        dmk_names_ai = [refs_ranked[1]]             # if you want to use specific player/s
    else:
        dmk_names_ai = [f'dmk{ix}' for ix in range(n_ai_players)]
        for nm in dmk_names_ai:
            build_single_foldmk(name=nm, family='b')

    gm = HuGamesManager(
        dmk_names=      dmk_names_ai,
        #debug_dmks=     True,
        #debug_tables=   True,
    )
    gm.start_games()
    gm.run_tk()
    gm.kill_games()