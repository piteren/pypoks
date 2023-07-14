from pypaq.lipytools.files import r_json
from pypaq.lipytools.plots import two_dim_multi
from pypaq.lipytools.moving_average import MovAvg
from typing import List, Dict, Optional

from envy import RESULTS_FP
from run.functions import get_saved_dmks_names


def get_ranks(
        all_results: Optional[Dict]=    None,
        mavg_factor=                    0.3,
) -> Dict:

    if all_results is None:
        all_results = r_json(RESULTS_FP)

    all_names = []
    for lix in all_results['loops']:
        all_names += all_results['loops'][lix]
    all_names = set(all_names)

    ranks = {dn: [] for dn in all_names}
    for lix in all_results['loops']:
        names_present = []
        for r,dn in enumerate(all_results['loops'][lix]):
            names_present.append(dn)
            ranks[dn].append(r)
        for dn in ranks:
            if dn not in names_present:
                ranks[dn].append(None)

    ranks_smooth = {}
    for dn in ranks:
        ranks_smooth[dn] = []
        ma = MovAvg(mavg_factor)
        for v in ranks[dn]:
            if v is None:
                ma = MovAvg(mavg_factor)
                ranks_smooth[dn].append(None)
            else:
                ranks_smooth[dn].append(ma.upd(v))

    """
    len_ranks = max([len(v) for v in ranks.values()])
    families = set([all_results[dn]['family'] for dn in dmk_TR])
    ranks_fam = {f: [[] for _ in range(len_ranks)] for f in families}
    for dn in ranks:
        dnf = all_results[dn]['family']
        for ix,r in enumerate(ranks[dn]):
            ranks_fam[dnf][ix].append(r)
    print(families)
    print(ranks_fam)
    for f in ranks_fam:
        for ix in range(len(ranks_fam[f])):
            ranks_fam[f][ix] = sum(ranks_fam[f][ix]) / len(ranks_fam[f][ix])
    """

    return {
        'ranks':        ranks,
        'ranks_smooth': ranks_smooth,
        #'ranks_fam':    ranks_fam
    }


if __name__ == "__main__":

    rd = get_ranks()
    print(rd)


    """
    two_dim_multi(
        ys=     list(rd['ranks'].values()),
        names=  list(rd['ranks'].keys()))

    two_dim_multi(
        ys=     list(rd['ranks_smooth'].values()),
        names=  list(rd['ranks_smooth'].keys()))

    two_dim_multi(
        ys=     list(rd['ranks_fam'].values()),
        names=  list(rd['ranks_fam'].keys()))
    """