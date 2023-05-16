from pypaq.lipytools.files import r_json

from pypoks_envy import RESULTS_FP
from podecide.games_manager import HuGamesManager


if __name__ == "__main__":

    all_results = r_json(RESULTS_FP)
    last_loop = max([int(k) for k in all_results['loops'].keys()])
    dmk_ranked = all_results['loops'][str(last_loop)]
    best_dmk = dmk_ranked[:2]

    gm = HuGamesManager(dmk_names=best_dmk)
    gm.start_games()
    gm.run_tk()
    gm.kill_games()