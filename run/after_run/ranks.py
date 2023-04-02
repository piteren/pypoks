from pypaq.lipytools.files import r_json
from pypaq.lipytools.plots import two_dim_multi
from pypaq.lipytools.moving_average import MovAvg
from typing import List, Dict, Optional

from pypoks_envy import RESULTS_FP
from run.functions import get_saved_dmks_names


def get_ranks(
        dmk_TR: Optional[List[str]]=    None,
        all_results: Optional[Dict]=    None,
        mavg_factor=                    0.2,
) -> Dict:

    if dmk_TR is None:
        names_saved = get_saved_dmks_names()  # get all saved names
        dmk_TR = [dn for dn in names_saved if '_old' not in dn]  # get names of TR

    low_rank = len(dmk_TR)

    if all_results is None:
        all_results = r_json(RESULTS_FP)

    ranks = {dn: [low_rank] * int(dn[3:6]) + all_results[dn]['rank'] for dn in dmk_TR}
    ranks_smooth = {}
    for dn in ranks:
        ma = MovAvg(mavg_factor)
        ranks_smooth[dn] = [ma.upd(v) for v in ranks[dn]]

    len_ranks = max([len(v) for v in ranks.values()])
    families = set([all_results[dn]['family'] for dn in dmk_TR])
    ranks_fam = {f: [[] for _ in range(len_ranks)] for f in families}
    for dn in ranks:
        dnf = all_results[dn]['family']
        for ix,r in enumerate(ranks[dn]):
            ranks_fam[dnf][ix].append(r)
    for f in ranks_fam:
        for ix in range(len(ranks_fam[f])):
            ranks_fam[f][ix] = sum(ranks_fam[f][ix]) / len(ranks_fam[f][ix])

    return {
        'ranks':        ranks,
        'ranks_smooth': ranks_smooth,
        'ranks_fam':    ranks_fam}


if __name__ == "__main__":

    rd = get_ranks()

    two_dim_multi(
        ys=     list(rd['ranks'].values()),
        names=  list(rd['ranks'].keys()))

    two_dim_multi(
        ys=     list(rd['ranks_smooth'].values()),
        names=  list(rd['ranks_smooth'].keys()))

    two_dim_multi(
        ys=     list(rd['ranks_fam'].values()),
        names=  list(rd['ranks_fam'].keys()))