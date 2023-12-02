from pypaq.lipytools.files import r_pickle
from pypaq.pms.points_cloud import VPoint
from pypaq.pms.paspa import PaSpa
from pypaq.pms.points_cloud import PointsCloud

from envy import DMK_MODELS_FD

DFD = DMK_MODELS_FD
#DFD = '_models/_dmk.ppo6.data'

points_data = r_pickle(f'{DFD}/points.data')
points_dmk =     points_data['points_dmk']
points_motorch = points_data['points_motorch']
scores =         points_data['scores']

dmk_names = list(points_dmk.keys())
print(f'\nDMKs: {dmk_names}')

sckL = []
for dn in dmk_names:
    sckL += list(scores[dn].keys())
sckL = list(set(sckL))
print(f'scores: {sckL}')

k = dmk_names[0]
psdd = points_motorch[k]['psdd']
paspa = PaSpa(psdd=psdd)
print(f'\n{paspa}')

for sck in [
    'rank_14',
    'rank_18',
]:
    print(f'\nscore key: {sck}')

    sel_scores = {k: scores[k][sck] for k in scores if sck in scores[k]}

    vpoints = []
    for k in sel_scores:
        point_trimmed = {p: points_motorch[k][p] for p in psdd}
        vpoints.append(VPoint(
            name=   k,
            point=  point_trimmed,
            value=  sel_scores[k]))
    vpoints.sort(key=lambda x:x.value, reverse=True)

    pcloud = PointsCloud(paspa=paspa)
    pcloud.update_cloud(vpoints=vpoints)

    print(pcloud)
    pcloud.plot(axes=['baseLR','gc_do_clip','entropy_coef'])


# TODO: locals()
# https://docs.python.org/3/library/functions.html#locals
# https://plotly.com/python-api-reference/generated/plotly.express.scatter_3d