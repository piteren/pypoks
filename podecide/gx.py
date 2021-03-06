"""

 2020 (c) piteren

"""

import os
import random
from typing import List

from ptools.neuralmess.base_elements import mrg_ckpts

from pypoks_envy import DMK_MODELS_FD

# genetic crossing for TF based checkpoints
def xross(
        ppl :List[tuple],           # population (ckpt, eval, family):
        shape :tuple,               # (number of parents (from top), number of children (from bottom))
        noiseF :float=  0.03,
        verb=           0):

    if verb > 0: print('\nXross (GA mixing)...')
    if verb > 1:
        for pv in ppl: print(f' >> {pv[0]:5s} : {pv[1]:6.2f} : {pv[2]}')

    # build population in families
    families = set([pv[2] for pv in ppl])
    fppl = {f: [] for f in families}
    for pv in ppl: fppl[pv[2]].append(pv)
    if None in fppl: fppl.pop(None) # None family is meant to not GAX

    f_parents_names = {}
    f_replace_names = {}
    for f in fppl:
        ppl = fppl[f]

        parents = ppl[:shape[0]]
        replace = ppl[-shape[1]:]
        parents_names = [p[0] for p in parents]
        replace_names = [r[0] for r in replace]
        f_parents_names[f] = parents_names
        f_replace_names[f] = replace_names
        if verb > 1:
            print(f' > family {f}')
            print(f' >> parents {parents_names}')
            print(f' >> replace {replace_names}')

        mfd = f'{DMK_MODELS_FD}/{parents_names[0]}'
        ckptL = [dI for dI in os.listdir(mfd) if os.path.isdir(os.path.join(mfd,dI))]
        ckptL.remove('opt_vars')

        # merge checkpoints
        mrg_dna = {name: [random.sample(parents_names,2), 0.2+0.6*random.random()] for name in replace_names}
        for name in mrg_dna:
            dmka_name =     mrg_dna[name][0][0]
            dmkb_name =     mrg_dna[name][0][1]
            rat =           mrg_dna[name][1]
            if verb > 0: print(f' > merging: {dmka_name} + {dmkb_name} >> {name} ({rat:.2f})')
            for ckpt in ckptL:
                mrg_ckpts(
                    ckptA =         ckpt,
                    ckptA_FD =      f'{DMK_MODELS_FD}/{dmka_name}/',
                    ckptB =         ckpt,
                    ckptB_FD =      f'{DMK_MODELS_FD}/{dmkb_name}/',
                    ckptM =         ckpt,
                    ckptM_FD =      f'{DMK_MODELS_FD}/{name}/',
                    replace_scope = name,
                    mrgF =          rat,
                    noiseF =        noiseF)

    return {
        'parents':  f_parents_names,
        'mixed':    f_replace_names}
