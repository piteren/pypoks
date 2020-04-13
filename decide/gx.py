"""

 2020 (c) piteren

"""

import os
import random
from typing import List

from putils.neuralmess.base_elements import mrg_ckpts

# genetic crossing for TF based checkpoints
def xross(
        ppl :List[tuple],           # population (ckpt,eval):
        n_par :int,                 # number of parents (from top)
        n_mix :int,                 # number of childs (from bottom)
        noiseF :float=  0.03,
        top_FD=         '_models/',
        verb=           0):

    if verb > 0: print('\nXross (GA mixing)...')
    if verb > 1:
        for pv in ppl: print(f' >> {pv[0]:5s} : {pv[1]:6.2f}')

    parents = ppl[:n_par]
    replace = ppl[-n_mix:]
    parents_names = [p[0] for p in parents]
    replace_names = [r[0] for r in replace]
    if verb > 1:
        print(' >> parents', parents_names)
        print(' >> replace', replace_names)

    mfd = top_FD + '/' + parents_names[0]
    ckptL = [dI for dI in os.listdir(mfd) if os.path.isdir(os.path.join(mfd,dI))]
    ckptL.remove('opt_vars')

    # merge checkpoints
    # TODO: use sampling with probability (eval/pos as a prob)
    mrg_dna = {name: [random.sample(parents_names,2), 0.2+0.6*random.random()] for name in replace_names}
    for name in mrg_dna:
        dmka_name =     mrg_dna[name][0][0]
        dmkb_name =     mrg_dna[name][0][1]
        rat =           mrg_dna[name][1]
        if verb > 0: print(f' > merging {dmka_name} + {dmkb_name} >> {name} ({rat:.2f})')
        for ckpt in ckptL:
            # TODO: add noise to variables
            mrg_ckpts(
                ckptA =         ckpt,
                ckptA_FD =      '_models/%s/' % dmka_name,
                ckptB =         ckpt,
                ckptB_FD =      '_models/%s/' % dmkb_name,
                ckptM =         ckpt,
                ckptM_FD =      '_models/%s/' % name,
                replace_scope = name,
                mrgF =          rat,
                noiseF =        noiseF)

    return {
        'parents':  parents_names,
        'mixed':    replace_names}
