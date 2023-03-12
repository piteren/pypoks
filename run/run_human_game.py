from pypaq.lipytools.files import r_json

from pypoks_envy import RESULTS_FP
from podecide.games_manager import HuGamesManager
from run.functions import get_saved_dmks_names


if __name__ == "__main__":

    saved_names = get_saved_dmks_names()
    all_results = r_json(RESULTS_FP)
    best_dmk = [(dn, all_results[dn]['rank'][-1]) for dn in saved_names if '_old' not in dn]
    best_dmk = sorted(best_dmk, key=lambda x:x[1])
    best_dmk = [e[0] for e in best_dmk[:2]]

    gm = HuGamesManager(dmk_names=best_dmk)
    gm.start_games()
    gm.run_tk()
    gm.kill_games()