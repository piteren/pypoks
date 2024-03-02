from pypaq.pms.base import POINT
from pypaq.pms.points_cloud import VPoint
from typing import Dict, Optional, List

from run.after_run.review_points import merged_point_in_psdd, points_nice_table


def results_report(
        dmk_results: Dict[str, Dict],
        dmks: Optional[List[str]]=  None,
) -> str:
    """ prepares DMKs results report
    pos DMK_name : wonH [wonH_IV_stddev/wonH_IV_mean_stddev] poker_stats wonH_diff sep lifemark """

    if not dmks:
        dmks = list(dmk_results.keys())

    res_nfo = ''
    for ix,dn in enumerate(sorted(dmks, key=lambda x: dmk_results[x]['last_wonH_afterIV'], reverse=True)):

        wonH = dmk_results[dn]['last_wonH_afterIV']
        wonH_IV_std = dmk_results[dn]['wonH_IV_stddev']
        wonH_mstd = dmk_results[dn]['wonH_IV_mean_stddev']
        wonH_mstd_str = f'[{wonH_IV_std:.2f}/{wonH_mstd:.2f}]'

        stats_nfo = ''
        for k in dmk_results[dn]["global_stats"]:
            v = dmk_results[dn]["global_stats"][k]
            stats_nfo += f'{k}:{v*100:4.1f} '

        wonH_diff = f"diff: {dmk_results[dn]['wonH_diff']:5.2f}" if 'wonH_diff' in dmk_results[dn] else ""
        sep = f" sepF: {dmk_results[dn]['separated_factor']:4.2f}" if 'separated_factor' in dmk_results[dn] else ""
        lifemark = f" {dmk_results[dn]['lifemark']}" if 'lifemark' in dmk_results[dn] else ""

        res_nfo += f'{ix:2} {dn:18} : {wonH:6.2f} {wonH_mstd_str:>12}  {stats_nfo}  {wonH_diff}{sep}{lifemark}\n'

    if res_nfo:
        res_nfo = res_nfo[:-1]
    return res_nfo


def nice_hpms_report(
        points_dmk: Dict[str, POINT],
        points_motorch: Dict[str, POINT],
        dmk_ranked: List[str],
) -> str:
    vpoints = [VPoint(
        point=  merged_point_in_psdd(points_dmk[dn], points_motorch[dn]),
        name=   dn) for dn in dmk_ranked]
    table = points_nice_table(vpoints, do_val=False)
    table_pos =  [f'   {table[0]}']
    table_pos += [f'{ix:2} {l}' for ix,l in enumerate(table[1:])]
    return '\n'.join(table_pos)