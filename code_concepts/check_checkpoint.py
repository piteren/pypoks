from pypaq.lipytools.stats import msmx
import torch

#ckpt_path = '_models/dmk/dmk151a01/dmk151a01.pt'
ckpt_path = '../_models/dmk/dmk034b00/dmk034b00.pt'
#ckpt_path = '_models/cardNet/cardNet12/cardNet12.pt'

save_obj = torch.load(f=ckpt_path, map_location='cpu')

norms = {}
for k in save_obj['model_state_dict'].keys():
    norms[k] = float(save_obj['model_state_dict'][k].norm())

for k in norms:
    tns = save_obj["model_state_dict"][k]
    n = tns.numel()
    max = torch.max(tns)
    print(f'{k:100s}:{norms[k]:5.1f} {n:6} {max:.1f}')
print(msmx(list(norms.values()))['string'])
