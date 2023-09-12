from pypaq.lipytools.files import r_json

from envy import RESULTS_FP
from podecide.games_manager import HuGamesManager
from run.functions import build_single_foldmk


if __name__ == "__main__":

    loop_results = r_json(RESULTS_FP)
    if loop_results:
        refs_ranked = loop_results['refs_ranked']
        best_dmk = refs_ranked[:2]
    else:
        best_dmk = ['dmk1','dmk2']
        for nm in best_dmk:
            build_single_foldmk(name=nm,family='a')

    gm = HuGamesManager(dmk_names=best_dmk, debug_tables=True)
    gm.start_games()
    gm.run_tk()
    gm.kill_games()