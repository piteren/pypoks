import math
from pypaq.lipytools.files import w_json, w_pickle, r_pickle
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.printout import stamp
from pypaq.pms.config_manager import ConfigManager
from pypaq.pms.points_cloud import VPoint
import random
import shutil
import time
from torchness.tbwr import TBwr
from typing import List, Tuple

from envy import DMK_MODELS_FD, TR_CONFIG_FP, TR_RESULTS_FP
from pologic.game_config import GameConfig
from podecide.dmk import FolDMK
from podecide.dmk_motorch import DMK_MOTorch
from podecide.game_manager import separation_report, separated_factor
from run.functions import check_continuation, get_saved_dmks_names, run_PTR_game, copy_dmks, build_single_foldmk, \
    dmk_name
from run.after_run.reports import results_report
from run.after_run.review_points import merged_point_in_psdd, points_nice_table

# training config
LOOP_CONFIG = {
        # general
    'exit':                     False,      # exits loop (after train)
    'pause':                    False,      # pauses loop after test till Enter pressed
    'families':                 'a',        # active families (forced to be present in learners)
    'min_familyF':              0.5,        # forces every family to be present in learners by at least F of initial amount
    'ndmk_refs':                4,          # number of refs DMKs
    'ndmk_learners':            16,         # number of learners DMKs
    'group_size_TR':            20,         # learners are trained against refs in groups of this size
    'group_size_TS':            50,         # learners and refs are tested against refs in groups of this size
        # train / test
    'game_size_TR':             100000,
    'against_best':             3,          # every Nth loop train against only best ref
    'game_size_TS':             100000,
    'n_tables':                 1000,       # target number of tables (TR & TS)
        # replace / new
    'n_stddev':                 1.0,        # number of stddev that is considered to be valid separation distance
    'remove_key':               [4,1],      # [A,B] remove DMK if in last A+B life marks there are A -| and last is not +/
    'prob_fresh_dmk':           0.5,        # probability of 100% fresh DMK (otherwise GX)
    'prob_fresh_ckpt':          0.5,        # probability of fresh checkpoint (child from GX of point only, without GX of ckpt)
        # PMT (Periodical Masters Test)
    'n_loops_PMT':              5,          # do PMT every N loops
    'ndmk_PMT':                 10,         # max number of DMKs (masters) in PMT
}

PUB_REF = {'publish_player_stats':False, 'publishFWD':False, 'publishUPD':False}
PUB_TR =  {'publish_player_stats':False, 'publishFWD':False, 'publishUPD':True}
PUB_TS =  {'publish_player_stats':True,  'publishFWD':True,  'publishUPD':False}
PUB_PMT = {'publish_player_stats':False, 'publishFWD':False, 'publishUPD':False}

PMT_FD = f'{DMK_MODELS_FD}/_pmt'


def modify_lists_V1(
        dmk_learners,
        dmk_learners_aged,
        dmk_refs,
        dmk_results,
        cm,
        logger,
        sub_logger,
) -> Tuple[List[str], List[str]]:
    """ manages / modifies DMKs lists - V1 """

    ### prepare new list of learners

    # rank learners by last_wonH_afterIV
    dmk_rw = [(dn, dmk_results[dn]['last_wonH_afterIV']) for dn in dmk_learners_aged]
    learners_aged_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x:x[1], reverse=True)]

    # asir = aged + separated + improved -> ranked
    dmk_learners_asir = [
        dn for dn in learners_aged_ranked
        if dmk_results[dn]['separated_factor'] >= cm.n_stddev and dmk_results[dn]['wonH_diff'] > 0]
    logger.info(f'learners aged & separated & improved: {", ".join(dmk_learners_asir)}')

    dmk_learners_new = []
    for ix,dna in enumerate(dmk_learners_aged):
        dn = dmk_learners[ix]

        # replace learner by aged & separated & improved
        if dna in dmk_learners_asir:
            dmk_learners_new.append(dna)
        # leave old
        else:
            dmk_learners_new.append(dn)
            dmk_results[dn]['lifemark'] = dmk_results[dna]['lifemark'] # update lifemark of not improved

    ### remove bad lifemarks from learners

    dmk_learners_bad_lifemark = []
    for dn in dmk_learners_new:
        l_end = dmk_results[dn]['lifemark'][-sum(cm.remove_key):]
        if l_end.count('-') + l_end.count('|') >= cm.remove_key[0] and l_end[-1] not in '/+':
            dmk_learners_bad_lifemark.append(dn)

    if dmk_learners_bad_lifemark:
        logger.info(f'removing learners with bad lifemark: {", ".join(dmk_learners_bad_lifemark)}')
        dmk_learners_new = [dn for dn in dmk_learners_new if dn not in dmk_learners_bad_lifemark]

    ### eventually replace some refs by learners that improved

    dmk_rw = [(dn, dmk_results[dn]['last_wonH_afterIV']) for dn in dmk_refs]
    refs_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x:x[1], reverse=True)]

    # iterate over asi from the best
    dmk_add_to_refs = []
    for dn in dmk_learners_asir:

        replaces = None

        # first by age
        for dnr in refs_ranked:
            if dnr[:-4] == dn[:-3]:
                replaces = dnr
                break

        # then by sep
        if not replaces:
            for dnr in reversed(refs_ranked): # from bottom
                a_wonH = dmk_results[dn]['last_wonH_afterIV']
                b_wonH = dmk_results[dnr]['last_wonH_afterIV']
                if a_wonH > b_wonH and cm.n_stddev <= separated_factor(
                    a_wonH=             a_wonH,
                    a_wonH_mean_stddev= dmk_results[dn]['wonH_IV_mean_stddev'],
                    b_wonH=             b_wonH,
                    b_wonH_mean_stddev= dmk_results[dnr]['wonH_IV_mean_stddev']):
                    replaces = dnr
                    break

        if replaces:
            dmk_add_to_refs.append(dn)
            refs_ranked.remove(replaces)

    new_ref_names = []
    if dmk_add_to_refs:
        logger.info(f'adding to refs: {", ".join(dmk_add_to_refs)}')

        new_ref_names = [f'{dn}R' for dn in dmk_add_to_refs]
        copy_dmks(
            names_src=  dmk_add_to_refs,
            names_trg=  new_ref_names,
            logger=     sub_logger)

    # prepare sorted dmk_refs after all changes
    dmk_rw = [(dn, dmk_results[dn]['last_wonH_afterIV']) for dn in refs_ranked + new_ref_names]
    dmk_refs_new = [e[0] for e in sorted(dmk_rw, key=lambda x:x[1], reverse=True)]

    return dmk_refs_new, dmk_learners_new


def modify_lists_V2(
        dmk_learners,
        dmk_learners_aged,
        dmk_refs,
        dmk_results,
        cm,
        logger,
        sub_logger,
) -> Tuple[List[str], List[str]]:
    """ manages / modifies DMKs lists - V2 """

    ### prepare new list of learners

    dmk_learners_ar = sorted(dmk_learners_aged, key=lambda x:dmk_results[x]['last_wonH_afterIV'], reverse=True) # ar = aged + ranked
    dmk_learners_air = [dn for dn in dmk_learners_ar if dmk_results[dn]['wonH_diff'] > 0] # air = aged + improved + ranked
    logger.info(f'learners improved ({len(dmk_learners_air)}): {", ".join(dmk_learners_air)}')

    dmk_learners_sel_age = []
    for ix,dna in enumerate(dmk_learners_aged):
        dn = dmk_learners[ix]

        # replace learner by air
        if dna in dmk_learners_air:
            dmk_learners_sel_age.append(dna)
        # leave old
        else:
            dmk_learners_sel_age.append(dn)
            dmk_results[dn]['lifemark'] = dmk_results[dna]['lifemark'] # update lifemark of not improved

    ### remove bad lifemarks from learners

    dmk_learners_bad_lifemark = []
    for dn in dmk_learners_sel_age:
        elen = sum(cm.remove_key)
        lifemark_ending = dmk_results[dn]['lifemark'][-elen:]
        lifemark_ending2 = dmk_results[dn]['lifemark'][-2*elen:]
        cond_1 = lifemark_ending.count('-') + lifemark_ending.count('|') >= cm.remove_key[0] and lifemark_ending[-1] not in '/+'
        cond_2 = len(lifemark_ending2) >= 2*elen and '+' not in lifemark_ending2
        if cond_1 or cond_2:
            dmk_learners_bad_lifemark.append(dn)

    if dmk_learners_bad_lifemark:
        logger.info(f'removing learners with bad lifemark: {", ".join(dmk_learners_bad_lifemark)}')

    dmk_learners_new = [dn for dn in dmk_learners_sel_age if dn not in dmk_learners_bad_lifemark]

    ### prepare new refs list

    # select new refs without aged duplicates
    dmk_rw_to_refs = []
    _without_age = []
    for dn in sorted(
            dmk_learners_aged + dmk_learners + dmk_refs,
            key=        lambda x:dmk_results[x]['last_wonH_afterIV'],
            reverse=    True):
        dn_without_age = dn[:9]
        if dn_without_age not in _without_age:
            _without_age.append(dn_without_age)
            dmk_rw_to_refs.append(dn)
            if len(dmk_rw_to_refs) == cm.ndmk_refs:
                break
    logger.info(f'new refs from: {", ".join(dmk_rw_to_refs)}')

    dmk_refs_new = []
    for dn in dmk_rw_to_refs:
        if dn[-1] == 'R':
            dmk_refs_new.append(dn)
        else:
            ref_name = f'{dn}R'
            copy_dmks(
                names_src=  [dn],
                names_trg=  [ref_name],
                logger=     sub_logger)
            dmk_refs_new.append(ref_name)

    return dmk_refs_new, dmk_learners_new # INFO: dmk_refs_new is sorted here, dmk_learners_new NOT


def run(game_config_name: str):
    """ Trains DMKs against REFs using groups of learners(aged) and old.

    Below are descriptions of some structures used:

        tr_results = {
            'loop_ix': 5 (int),                                 <- number of loops performed yet
            'lifemarks': {
                'dmk1': '+|/++',
                ..
            },
            'refs_ranked': ['dmk4','dmk1', ..]
        }

        dmk_results = {
            'dmk01a00_03': {                                    # not-aged do not have all keys
                'wonH_IV':              [float,..]              <- wonH of interval
                'wonH_afterIV':         [float,..]              <- wonH after interval
                'wonH_IV_stddev':       float,
                'wonH_IV_mean_stddev':  float,
                'last_wonH_afterIV':    float,
                'family':               dmk.family,
                'trainable':            dmk.trainable,
                'age':                  dmk.age
                'separated_factor':     float
                'wonH_diff':            float                   <- wonH diff against not aged
                'lifemark':             '++'
            },
            ..
        }

    Loop procedure:
    1. Create DMKs
        fill up dmk_learners (new / GX)
        create dmk_refs <- only in the first loop
    2. Train (learners)
        copy learners to new age (+1)
        split dmk_learners into groups of group_size_TR
        train each group against dmk_refs
    3. Test (learners & refs)
        prepare list of DMKs to test
        split into groups of group_size_TS
        test may be broken with 'separated' condition
    4. Analyse & report results of learners and refs
    5. Manage (modify) DMKs lists (learners & refs)
    6. PMT evaluation """

    cc = check_continuation()
    tr_results = cc['training_results']

    tl_name = f'run_train_loop_refs_learners_{stamp(letters=None)}'
    logger = get_pylogger(
        name=       tl_name,
        add_stamp=  False,
        folder=     DMK_MODELS_FD,
        level=      20,
        #flat_child= True,
    )
    logger.info(f'{tl_name} starts..')
    sub_logger = get_child(logger)

    cm = ConfigManager(file_FP=TR_CONFIG_FP, config_init=LOOP_CONFIG, logger=logger)
    tbwr = TBwr(logdir=f'{DMK_MODELS_FD}/{tl_name}')

    if cc['continuation']:

        game_config = GameConfig.from_name(folder=DMK_MODELS_FD)
        logger.info(f'> game config name: {game_config.name}')

        saved_n_loops = int(tr_results['loop_ix'])
        logger.info(f'> continuing with saved {saved_n_loops} loops')
        loop_ix = saved_n_loops + 1

        dmk_refs = tr_results['refs_ranked']
        dmk_learners = list(tr_results['lifemarks'].keys())

        points_data = r_pickle(f'{DMK_MODELS_FD}/points.data')

    else:

        game_config = GameConfig.from_name(name=game_config_name, copy_to=DMK_MODELS_FD)
        logger.info(f'> game config name: {game_config.name}')

        loop_ix = 1
        dmk_refs = []
        dmk_learners = []

        tr_results = {
            'loop_ix':      loop_ix,
            'lifemarks':    {},
            'refs_ranked':  []}

        points_data = {'points_dmk':{}, 'points_motorch':{}, 'scores':{}}

    logger.info(f'> game config: {game_config}')

    while True:

        loop_stime = time.time()

        if cm.exit:
            logger.info('train loop exits')
            cm.exit = False
            break

        logger.info(f'\n ************** starts loop #{loop_ix} **************')

        tbwr.add(value=cm.game_size_TS, tag=f'loop/game_size_TS', step=loop_ix)
        tbwr.add(value=cm.game_size_TR, tag=f'loop/game_size_TR', step=loop_ix)

        #************************************************************************************************ 1. create DMKs

        # fill up dmk_learners (new / GX)
        cix = 0
        if len(dmk_learners) < cm.ndmk_learners:

            logger.info(f'building {cm.ndmk_learners - len(dmk_learners)} new DMKs (learners):')

            # look for forced families
            learners_families = {dn: FolDMK.load_point(name=dn)['family'] for dn in dmk_learners}
            families_present = ''.join(list(learners_families.values()))
            families_count = {fm: families_present.count(fm) for fm in cm.families}
            families_count = [(fm, families_count[fm]) for fm in families_count]
            force_to = int(cm.ndmk_learners / len(cm.families) * cm.min_familyF)
            families_forced = [fc[0] for fc in families_count if fc[1] < force_to]
            if families_forced: logger.info(f'families forced: {families_forced}')

            refs_families = {dn: FolDMK.load_point(name=dn)['family'] for dn in dmk_refs}

            while len(dmk_learners) < cm.ndmk_learners:

                # build new one from forced
                if families_forced:
                    family = families_forced.pop(0)
                    name_child = dmk_name(loop_ix=loop_ix, family=family, counter=cix, age=0, is_ref=False)
                    logger.info(f'> {name_child} <- fresh, forced from family {family}')
                    build_single_foldmk(
                        game_config=    game_config,
                        name=           name_child,
                        family=         family,
                        logger=         sub_logger)

                else:

                    pa = random.choices(dmk_refs, weights=list(reversed(range(len(dmk_refs)))))[0] if dmk_refs else None

                    # 100% fresh DMK from selected family
                    if random.random() < cm.prob_fresh_dmk or pa is None:
                        family = random.choice(cm.families)
                        name_child = dmk_name(loop_ix=loop_ix, family=family, counter=cix, age=0, is_ref=False)
                        logger.info(f'> {name_child} <- 100% fresh')
                        build_single_foldmk(
                            game_config=    game_config,
                            name=           name_child,
                            family=         family,
                            logger=         sub_logger)

                    # GX from refs
                    else:
                        family = refs_families[pa] if pa is not None else random.choice(cm.families)
                        name_child = dmk_name(loop_ix=loop_ix, family=family, counter=cix, age=0, is_ref=False)
                        other_fam = [dn for dn in dmk_refs if refs_families[dn] == family]
                        if len(other_fam) > 1:
                            other_fam.remove(pa)
                        if len(other_fam) > 1:
                            pb = random.choices(other_fam, weights=list(reversed(range(len(other_fam)))))[0]
                        else:
                            pb = other_fam[0]

                        ckpt_fresh = random.random() < cm.prob_fresh_ckpt
                        ckpt_fresh_info = ' (fresh ckpt)' if ckpt_fresh else ''
                        logger.info(f'> {name_child} = {pa} + {pb}{ckpt_fresh_info}')
                        FolDMK.gx_saved(
                            name_parentA=   pa,
                            name_parentB=   pb,
                            name_child=     name_child,
                            do_gx_ckpt=     not ckpt_fresh,
                            logger=         sub_logger)

                dmk_learners.append(name_child)
                points_data['points_dmk'][name_child[:9]] = FolDMK.load_point(name=name_child)
                points_data['points_motorch'][name_child[:9]] = DMK_MOTorch.load_point(name=name_child)
                cix += 1

        ### create dmk_refs

        # in the first loop copy them from learners
        new_refs = []
        if loop_ix == 1:
            dmk_refs_from_learners = dmk_learners[:cm.ndmk_refs]
            dmk_refs = [f'{dn}R' for dn in dmk_refs_from_learners]
            copy_dmks(
                names_src=  dmk_refs_from_learners,
                names_trg=  dmk_refs,
                logger=     sub_logger)
            new_refs = dmk_refs

        # eventually add new
        while len(dmk_refs) < cm.ndmk_refs:
            family = random.choice(cm.families)
            name_child = dmk_name(loop_ix=loop_ix, family=family, counter=cix, age=0, is_ref=True)
            cix += 1
            build_single_foldmk(
                game_config=    game_config,
                name=           name_child,
                family=         family,
                logger=         sub_logger)
            new_refs.append(name_child)
            dmk_refs.append(name_child)
        if new_refs:
            logger.info(f'created {len(new_refs)} refs DMKs: {", ".join(new_refs)}')

        logger.info(f'loop #{loop_ix} DMKs:')
        logger.info(f'> learners: ({len(dmk_learners)}) {", ".join(dmk_learners)}')
        logger.info(f'> refs:     ({len(dmk_refs)}) {", ".join(dmk_refs)}')

        #******************************************************************************************* 2. train (learners)

        # copy dmk_learners with age +1
        dmk_learners_aged = [f'{dn[:-3]}{int(dn[-3:])+1:03}' for dn in dmk_learners]
        copy_dmks(
            names_src=  dmk_learners,
            names_trg=  dmk_learners_aged,
            logger=     sub_logger)

        # create groups by evenly distributing DMKs
        n_groups = math.ceil(len(dmk_learners_aged) / cm.group_size_TR)
        tr_groups = [[] for _ in range(n_groups)]
        gix = 0
        for dn in dmk_learners_aged:
            fix = gix % n_groups
            tr_groups[fix].append(dn)
            gix += 1

        dmk_refs_sel = [] + dmk_refs
        best_copies = []
        if cm.against_best and loop_ix % cm.against_best == 0:

            best = dmk_refs[0]
            logger.info(f'will train against best ref: {best}')

            # multiply them to speed up training
            best_copies = [f'{best}_copy{ix}' for ix in range(cm.ndmk_refs - 1)]
            copy_dmks(
                names_src=  [best] * len(best_copies),
                names_trg=  best_copies,
                logger=     sub_logger)

            dmk_refs_sel = [best] + best_copies

        for trg in tr_groups:
            run_PTR_game(
                game_config=    game_config,
                name=           f'GM_TR{loop_ix:03}',
                gm_loop=        loop_ix,
                dmk_point_refL= [{'name':dn, 'motorch_point':{'device':i%2}, **PUB_REF} for i,dn in enumerate(dmk_refs_sel)],
                dmk_point_TRL=  [{'name':dn, 'motorch_point':{'device':i%2}, **PUB_TR}  for i,dn in enumerate(trg)],
                game_size=      cm.game_size_TR,
                n_tables=       cm.n_tables,
                logger=         logger)

        # eventually remove copies
        for dn in best_copies:
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        #************************************************************************************* 3. test (learners & refs)

        # INFO: if refs group has not changed since last loop -> some DMKs may not need to be tested

        ### prepare full list of DMKs to test

        sep_pairs = list(zip(dmk_learners, dmk_learners_aged))

        # if there are refs that are not present in learners,
        # those need to be tested also
        # so copy them without R (to test against those with R) and delete after
        dmk_refs_to_test = []
        for dnr in dmk_refs:
            if dnr[:-1] not in dmk_learners:
                dmk_refs_to_test.append(dnr)
        dmk_refs_copied_to_test = []
        if dmk_refs_to_test:
            dmk_refs_copied_to_test = [dn[:-1] for dn in dmk_refs_to_test]
            copy_dmks(
                names_src=  dmk_refs_to_test,
                names_trg=  dmk_refs_copied_to_test,
                logger=     sub_logger)

        # create groups by evenly distributing DMKs
        ndmk_TS = len(sep_pairs) * 2 + len(dmk_refs_copied_to_test)
        n_groups = math.ceil(ndmk_TS / cm.group_size_TS)
        ts_groups = [[] for _ in range(n_groups)]
        group_pairs = [[] for _ in range(n_groups)]
        gix = 0
        for e in sep_pairs + dmk_refs_copied_to_test:
            fix = gix % n_groups
            if type(e) is tuple:
                ts_groups[fix].append(e[0])
                ts_groups[fix].append(e[1])
                group_pairs[fix].append(e)
            else:
                ts_groups[fix].append(e)
            gix += 1

        speedL = []
        dmk_results = {}
        for tsg,pairs in zip(ts_groups,group_pairs):
            rgd = run_PTR_game(
                game_config=        game_config,
                gm_loop=            loop_ix,
                dmk_point_refL=     [{'name':dn, 'motorch_point':{'device': i % 2}, **PUB_REF} for i,dn in enumerate(dmk_refs)],
                dmk_point_PLL=      [{'name':dn, 'motorch_point':{'device': i%2}, **PUB_TS}     for i,dn in enumerate(tsg)],
                game_size=          cm.game_size_TS,
                n_tables=           cm.n_tables,
                sep_pairs=          pairs,
                sep_pairs_factor=   1.1, # disables sep pairs break
                logger=             logger,
                publish=            False)
            speedL.append(rgd['loop_stats']['speed'])
            dmk_results.update(rgd['dmk_results'])

        speed_TS = sum(speedL) / len(speedL)
        tbwr.add(value=speed_TS, tag=f'loop/speed_Hs', step=loop_ix)

        # instantiate results of refs
        for dn in dmk_refs:
            dmk_results[dn] = dmk_results[dn[:-1]]

        # delete dmk_refs_copied_to_test
        for dn in dmk_refs_copied_to_test:
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        #************************************************************************************** 4. analyse & log results

        sr = separation_report(dmk_results=dmk_results, sep_pairs=sep_pairs)

        # update dmk_results
        session_lifemarks = ''
        for ix,dna in enumerate(dmk_learners_aged):
            dn = dmk_learners[ix]
            dmk_results[dna]['separated_factor'] = sr['sep_pairs_stat'][ix]
            dmk_results[dna]['wonH_diff'] = dmk_results[dna]['wonH_afterIV'][-1] - dmk_results[dn]['wonH_afterIV'][-1]
            lifemark_upd = '/' if dmk_results[dna]['wonH_diff'] > 0 else '|'
            if dmk_results[dna]['separated_factor'] >= cm.n_stddev:
                lifemark_upd = '+' if dmk_results[dna]['wonH_diff'] > 0 else '-'
            lifemark_prev = tr_results['lifemarks'][dn] if dn in tr_results['lifemarks'] else ''
            dmk_results[dna]['lifemark'] = lifemark_prev + lifemark_upd
            session_lifemarks += lifemark_upd

        not_sep_count = session_lifemarks.count('|') + session_lifemarks.count('/')
        sep_factor = (len(session_lifemarks) - not_sep_count) / len(session_lifemarks)
        tbwr.add(value=sep_factor, tag=f'loop/sep_factor', step=loop_ix)

        logger.info(f'DMKs results:\n{results_report(dmk_results, dmks=dmk_learners_aged+dmk_refs)}')

        dmk_vp = sorted(dmk_learners_aged + dmk_refs, key=lambda x: dmk_results[x]['wonH_afterIV'], reverse=True)
        vpoints = [VPoint(
            point=merged_point_in_psdd(points_data['points_dmk'][dn[:9]], points_data['points_motorch'][dn[:9]]),
            name=dn) for dn in dmk_vp]
        table = points_nice_table(vpoints, do_val=False)
        table_pos = [f'   {table[0]}']
        table_pos += [f'{ix:2} {l}' for ix, l in enumerate(table[1:])]
        points_str = '\n'.join(table_pos)
        logger.info(f'DMKs POINTS:\n{points_str}')

        # add scores
        dmk_ranked = sorted(dmk_learners_aged + dmk_learners + dmk_refs, key= lambda x:dmk_results[x]['wonH_afterIV'], reverse=True)
        dmk_ranked_policies = []
        for dn in dmk_ranked:
            if dn[:9] not in dmk_ranked_policies:
                dmk_ranked_policies.append(dn[:9])
        n_dmk = len(dmk_ranked_policies)
        for ix,dn in enumerate(dmk_ranked_policies):
            if dn not in points_data['scores']:
                points_data['scores'][dn] = {}
            points_data['scores'][dn][f'rank_{loop_ix:003}'] = (n_dmk - ix) / n_dmk

        #********************************************************************************* 5. manage / modify DMKs lists

        dmk_refs_new, dmk_learners_new = modify_lists_V2(
            dmk_learners=       dmk_learners,
            dmk_learners_aged=  dmk_learners_aged,
            dmk_refs=           dmk_refs,
            dmk_results=        dmk_results,
            cm=                 cm,
            logger=             logger,
            sub_logger=         sub_logger)

        # clean out folder
        for dn in dmk_learners + dmk_learners_aged + dmk_refs:
            if dn not in dmk_learners_new + dmk_refs_new:
                shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        # update again for new refs
        for dn in dmk_refs_new:
            dmk_results[dn] = dmk_results[dn[:-1]]

        dmk_refs.sort(key=lambda x: dmk_results[x]['last_wonH_afterIV'], reverse=True) # sort here to properly calculate refs results gain TOP
        refs_results = [dmk_results[dn]['last_wonH_afterIV'] for dn in dmk_refs]
        refs_new_results = [dmk_results[dn]['last_wonH_afterIV'] for dn in dmk_refs_new]
        refs_gain_all = sum(refs_new_results) - sum(refs_results)
        refs_gain_top = refs_new_results[0] - refs_results[0]
        refs_diff = dmk_results[dmk_refs_new[0]]['last_wonH_afterIV'] - dmk_results[dmk_refs_new[-1]]['last_wonH_afterIV']
        refs_wonH_IV_stddev_avg = sum([dmk_results[dn]['wonH_IV_stddev'] for dn in dmk_refs_new]) / len(dmk_refs_new)

        tbwr.add(value=refs_gain_all,           tag=f'loop/refs_gain_all',           step=loop_ix)
        tbwr.add(value=refs_gain_top,           tag=f'loop/refs_gain_top',           step=loop_ix)
        tbwr.add(value=refs_diff,               tag=f'loop/refs_diff',               step=loop_ix)
        tbwr.add(value=refs_wonH_IV_stddev_avg, tag=f'loop/refs_wonH_IV_stddev_avg', step=loop_ix)

        tr_results = {
            'refs_ranked':  dmk_refs_new,
            'lifemarks':    {dn: dmk_results[dn]['lifemark'] for dn in dmk_learners_new},
            'loop_ix':      loop_ix}
        w_json(tr_results, TR_RESULTS_FP)

        # refs (NEW) poker stats
        dmk_sets = {
            'refs_best':    [dmk_refs_new[0]],
            'refs_avg':     dmk_refs_new}
        for dsn in dmk_sets:
            gsL = [dmk_results[dn]['global_stats'] for dn in dmk_sets[dsn]]
            gsa_avg = {k: [] for k in gsL[0]}
            for e in gsL:
                for k in e:
                    gsa_avg[k].append(e[k])
            for l,k in zip('abcdefghijklmnoprs'[:len(gsa_avg)], gsa_avg):
                gsa_avg[k] = sum(gsa_avg[k]) / len(gsa_avg[k])
                tbwr.add(
                    value=  gsa_avg[k],
                    tag=    f'loop_poker_stats_{dsn}/{l}.{k}',
                    step=   loop_ix)

        w_pickle(points_data, f'{DMK_MODELS_FD}/points.data')

        if cm.pause: input("press Enter to continue..")

        loop_time = (time.time() - loop_stime) / 60
        logger.info(f'loop {loop_ix} finished, refs_gain ALL:{refs_gain_all:.2f} TOP:{refs_gain_top:.2f}, time taken: {loop_time:.1f}min')
        tbwr.add(value=loop_time, tag=f'loop/loop_time', step=loop_ix)

        #********************************************************************************************* 6. PMT evaluation

        if loop_ix % cm.n_loops_PMT == 0:
            new_master = dmk_refs_new[0]
            pmt_name = f'{new_master[:-1]}_pmt{loop_ix:03}'
            copy_dmks(
                names_src=          [new_master],
                names_trg=          [pmt_name],
                save_topdir_trg=    PMT_FD,
                logger=             sub_logger)
            logger.info(f'copied {new_master} to PMT -> {pmt_name}')

            all_pmt = get_saved_dmks_names(PMT_FD)
            if len(all_pmt) > 2: # run test for at least 3 DMKs
                logger.info(f'PMT starts..')

                # create groups evenly distributing DMKs
                n_groups = math.ceil(len(all_pmt) / cm.group_size_TS)
                ts_groups = [[] for _ in range(n_groups)]
                gix = 0
                for dn in all_pmt:
                    fix = gix % n_groups
                    ts_groups[fix].append(dn)
                    gix += 1

                pmt_results = {}
                for tsg in ts_groups:
                    rgd = run_PTR_game(
                        game_config=    game_config,
                        dmk_point_refL= [{'name':dn, 'motorch_point':{'device':i%2}, **PUB_REF} for i,dn in enumerate(dmk_refs_new)],
                        dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':i%2}, 'save_topdir':PMT_FD, **PUB_PMT} for i,dn in enumerate(tsg)],
                        game_size=      cm.game_size_TS,
                        n_tables=       cm.n_tables,
                        sep_all_break=  True,
                        logger=         logger,
                        publish=        False)
                    pmt_results.update(rgd['dmk_results'])

                logger.info(f'PMT results:\n{results_report(pmt_results)}')

                # remove worst
                if len(all_pmt) == cm.ndmk_PMT:
                    dmk_rw = [(dn, pmt_results[dn]['last_wonH_afterIV']) for dn in pmt_results]
                    pmt_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x: x[1], reverse=True)]
                    dn = pmt_ranked[-1]
                    shutil.rmtree(f'{PMT_FD}/{dn}', ignore_errors=True)
                    logger.info(f'removed PMT: {dn}')

        dmk_refs = dmk_refs_new
        dmk_learners = dmk_learners_new
        loop_ix += 1


if __name__ == "__main__":
    run(
        #'2players_2bets'
        '2players_9bets'
    )