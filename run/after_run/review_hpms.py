from pypaq.lipytools.files import r_pickle
from pypaq.pms.points_cloud import VPoint
from pypaq.pms.paspa import PaSpa
from pypaq.pms.points_cloud import PointsCloud

from envy import DMK_MODELS_FD

#DFD = DMK_MODELS_FD
#DFD = '_models/dmk_x36'
DFD = '_models/dmk_d2'

points_data = r_pickle(f'{DFD}/points.data')
points_dmk =     points_data['points_dmk']
points_motorch = points_data['points_motorch']
scores =         points_data['scores']

# TEMP rewrite scores
for dn in scores:
    scores_rew = {}
    for k in scores[dn]:
        n = int(k[5:])
        scores_rew[f'rank_{n:003}'] = scores[dn][k]
    scores[dn] = scores_rew

dmk_names = sorted(list(points_dmk.keys()))
dmk_names = [n for n in dmk_names if points_dmk[n]['family'] == 'a']
print(f'\nDMKs ({len(dmk_names)}): {dmk_names}')

sckL = []
for dn in dmk_names:
    sckL += list(scores[dn].keys())
sckL = sorted(list(set(sckL)))
print(f'scores ({len(sckL)}): {sckL}')

k = dmk_names[0]
psdd = points_motorch[k]['psdd']
paspa = PaSpa(psdd=psdd)
print(f'\n{paspa}')

#for sck in [f'rank_{ix:003}' for ix in (30,60,90,120,150)]:
for sck in [f'rank_{ix:003}' for ix in (20, 40, 60, 80)]:
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

    pcloud = PointsCloud(paspa=paspa, loglevel=30)
    pcloud.update_cloud(vpoints=vpoints)

    print(pcloud)
    pcloud.plot(axes=['baseLR','gc_do_clip','nam_loss_coef'])

# https://plotly.com/python-api-reference/generated/plotly.express.scatter_3d