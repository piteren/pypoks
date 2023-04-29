from pypaq.lipytools.files import r_json

from pypoks_envy import RESULTS_FP
from podecide.games_manager import HuGamesManager
from run.functions import get_saved_dmks_names


if __name__ == "__main__":

    saved_names = get_saved_dmks_names()
    all_results = r_json(RESULTS_FP)
    last_loop = max(all_results['loops'].keys())
    dmk_ranked = all_results['loops'][last_loop]
    best_dmk = dmk_ranked[:2]

    gm = HuGamesManager(dmk_names=best_dmk)
    gm.start_games()
    gm.run_tk()
    gm.kill_games()