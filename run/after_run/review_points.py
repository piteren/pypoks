from pypaq.lipytools.files import r_pickle
from pypaq.pms.base import POINT
from pypaq.pms.paspa import PaSpa
from pypaq.pms.points_cloud import VPoint, points_nice_table

from run.functions import get_saved_dmks_names
from podecide.dmk import FolDMK
from podecide.dmk_motorch import DMK_MOTorch


def merged_point_in_psdd(pa:POINT, pb:POINT):
    space_mrg = PaSpa(psdd=pa['psdd'], loglevel=30) + PaSpa(psdd=pb['psdd'], loglevel=30)
    point = {}
    point.update(pa)
    point.update(pb)
    return {k: v for k, v in point.items() if k in space_mrg.axes}


def get_points_from_data(folder:str):

    points_data = r_pickle(f'{folder}/points.data')
    points_dmk =     points_data['points_dmk']
    points_motorch = points_data['points_motorch']

    dmk_names = sorted(list(points_dmk.keys()))

    return {dn: merged_point_in_psdd(points_dmk[dn], points_motorch[dn]) for dn in dmk_names}


def get_dmk_points(folder:str):

    dmk_names = sorted(get_saved_dmks_names(folder))

    points = {}
    for dn in dmk_names:
        point_dmk = FolDMK.load_point(name=dn, save_topdir=folder)
        point_motorch = DMK_MOTorch.load_point(name=dn, save_topdir=folder)
        points[dn] = merged_point_in_psdd(point_dmk, point_motorch)

    return points


if __name__ == "__main__":

    # TODO:
    #  needs to merge all psdd before
    #  same names in both folders


    dmk_sel = [
        'dmk067a00',
        'dmk055a02',
        'dmk124a00',
        'dmk082a02',
        'dmk142a01',
        'dmk151a01',
    ]

    points = {}
    for fd in [
        '_models/dmk_x36',
        #'_models/dmk_d2',
    ]:
        points.update(get_points_from_data(fd))
        points.update(get_dmk_points(fd))
        print(sorted(list(points.keys())))

    print()
    vpoints = [VPoint(point=points[pk], name=pk) for pk in dmk_sel]
    table = points_nice_table(vpoints, do_val=False)
    print('\n'.join(table))