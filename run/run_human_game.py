from pypaq.lipytools.files import r_json

from envy import RESULTS_FP
from podecide.games_manager import HuGamesManager


if __name__ == "__main__":

    loop_results = r_json(RESULTS_FP)
    refs_ranked = loop_results['refs_ranked']
    best_dmk = refs_ranked[:2]

    gm = HuGamesManager(dmk_names=best_dmk, debug_tables=True)
    gm.start_games()
    gm.run_tk()
    gm.kill_games()