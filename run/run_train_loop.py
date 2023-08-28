"""
    This script trains a group of DMKs.
    DMKs are trained from a scratch with self-play game.
    Configuration is set to fully utilize power of 2 GPUs (11GB RAM).

    Below are descriptions of some structures used

        loops_results = {
            'loop_ix': 5 (int),                                 <- number of loops performed yet
            'lifemarks': {
                'dmk01a00_03': '+|/++',
                'dmk01a02_01': '-',
                ..
            }
        }

        dmk_ranked = ['dmka00_00','dmkb00_01',..]               <- ranking after last loop, updated in the loop after train
        dmk_old = ['dmka00_00_old','dmkb00_01_old',..]          <- just all _old names

        dmk_results = {
            'dmk01a00_03': {                                    # not-aged do not have all keys
                'wonH_IV':              [float,..]              <- wonH of interval
                'wonH_afterIV':         [float,..]              <- wonH after interval
                'wonH_IV_stdev':        float,
                'wonH_IV_mean_stdev':   float,
                'last_wonH_afterIV':    float,
                'family':               dmk.family,
                'trainable':            dmk.trainable,
                'age':                  dmk.age
                'separated':            0 or 1                  <- if separated against not aged
                'wonH_diff':            float                   <- wonH diff against not aged
                'lifemark':             '++'
            },
            ..
        }

    *** Some challenges of train_loop:

    1. sometimes test game breaks too quick because of separation factor, running longer would easily separate more DMKs, below an example:

        GM: 1.5min left:47.8min |--------------------| 3.1% 2109H/s (+1062Hpp) -- SEP:0.65[0.86]::0.80[1.00]2023-07-28 17:48:00,595 {    games_manager.py:391} p64325 INFO: > finished game (pairs separation factor: 0.80, game factor: 0.03)
        2023-07-28 17:48:04,801 {    games_manager.py:416} p64325 INFO: GM_PL_0728_1745_hFo finished run_game, avg speed: 1938.1H/s, time taken: 95.1sec
        2023-07-28 17:48:04,848 {   run_train_loop.py:317} p8824 INFO: DMKs train results:
         >  0(0.00) dmk01b09(1)     :  41.02 ::  77.14 s +
         >  1(0.10) dmk01b03(1)     :  40.73 ::  78.22 s +
         >  2(0.20) dmk01b08(1)     :  38.93 ::  77.49 s +
         >  3(0.30) dmk01b06(1)     :  36.06 ::  55.78 s +
         >  4(0.40) dmk01b00(1)     :  31.38 ::  58.40 s +
         >  5(0.50) dmk01b04(1)     :  26.97 ::  19.82 s +
         >  6(0.60) dmk01b05(1)     :  18.77 :: -12.65   |
         >  7(0.70) dmk01b01(1)     :   8.68 ::  48.70 s +
         >  8(0.80) dmk01b02(1)     :   4.96 ::  32.00 s +
         >  9(0.90) dmk01b07(1)     : -19.57 ::  13.72   |

        ..with this example dmk01b07 would be updated as separated +

"""
import math
from pypaq.lipytools.files import r_json, w_json
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.printout import stamp
from pypaq.pms.config_manager import ConfigManager
import random
import select
import shutil
import sys
import time
from torchness.tbwr import TBwr

from envy import DMK_MODELS_FD, PMT_FD, CONFIG_FP, RESULTS_FP
from podecide.dmk import FolDMK
from podecide.games_manager import separation_report, separated_factor
from run.functions import get_saved_dmks_names, run_GM, copy_dmks, build_single_foldmk, results_report

CONFIG_INIT = {
        # general
    'exit':                     False,      # exits loop (after train)
    'pause':                    False,      # pauses loop after test till Enter pressed
    # TODO: temp set ('b')
    'families':                 'a',        # active families (at least one DMK should be present)
    # TODO: temp set (10)
    'ndmk_refs':                5,         # number of refs DMKs (should fit on one GPU)
    # TODO: temp set (10)
    'ndmk_learners':            5,         # number of learners DMKs
    'ndmk_TR':                  5,          # learners are trained against refs in groups of this size
    'ndmk_TS':                  10,         # learners and refs are tested against refs in groups of this size
    'update_size':              100000,     # how much increase TS game size when needed
    'update_interval':          5,          # interval of TS game size update
        # train
    'game_size_TR':             100000,
    'dmk_n_players_TR':         150,        # number of players per DMK while TR
        # test
    'game_size_TS':             100000,
    'dmk_n_players_TS':         150,        # number of players per DMK while TS
    'sep_pairs_factor':         1.1,        # pairs separation break value, value > 1 disables break
    'sep_n_stdev':              1.0,        # separation won IV mean stdev factor
        # replace / new
    'remove_key':               [4,1],      # [A,B] remove DMK if in last A+B life marks there are A -|
    'prob_fresh_dmk':           0.8,        # probability of 100% fresh DMK
    'prob_fresh_ckpt':          0.8,        # probability of fresh checkpoint (child from GX of point only, without GX of ckpt)
        # PMT (Periodical Masters Test)
    'n_loops_PMT':              5,          # do PMT every N loops
    'ndmk_PMT':                 10}         # max number of DMKs (masters) in PMT


if __name__ == "__main__":

    # check for continuation
    continuation = False
    loops_results = r_json(RESULTS_FP)
    if loops_results:
        saved_n_loops = int(loops_results['loop_ix'])
        print(f'Do you want to continue with saved ({DMK_MODELS_FD}) {saved_n_loops} loops? ..waiting 10 sec (y/n, y-default)')
        i, o, e = select.select([sys.stdin], [], [], 10)
        if i and sys.stdin.readline().strip() == 'n':
            pass
        else:
            continuation = True

    # clean out dmk folder
    if not continuation:
        shutil.rmtree(f'{DMK_MODELS_FD}', ignore_errors=True)
        print(f'cleaned out {DMK_MODELS_FD}')

    tl_name = f'run_train_loop_{stamp()}'
    logger = get_pylogger(
        name=       tl_name,
        add_stamp=  False,
        folder=     DMK_MODELS_FD,
        level=      20)
    logger.info(f'train_loop {tl_name} starts..')

    cm = ConfigManager(file_FP=CONFIG_FP, config_init=CONFIG_INIT, logger=logger)
    tbwr = TBwr(logdir=f'{DMK_MODELS_FD}/{tl_name}')

    if continuation:

        saved_n_loops = int(loops_results['loop_ix'])
        logger.info(f'> continuing with saved {saved_n_loops} loops')

        loop_ix = saved_n_loops + 1
        since_last_update = int(loops_results['since_last_update'])

        saved_dmks = get_saved_dmks_names()
        dmk_refs = [dn for dn in saved_dmks if dn.endswith('_ref')]
        dmk_learners = [dn for dn in saved_dmks if dn not in dmk_refs]

    else:

        loop_ix = 1
        since_last_update = cm.update_interval

        dmk_refs = []
        dmk_learners = []

        loops_results = {
            'loop_ix':              loop_ix,
            'lifemarks':            {},
            'refs_ranked':          [],
            'since_last_update':    since_last_update}

    """
    DMKs are named with pattern: f'dmk{loop_ix:02}{family}{cix:02}_{age:02}' + optional '_ref'
    where:
        - cix   : index of DMK created in one loop
        - _ref  : is added to DMKs i refs group

    1. eventually create DMKs
        fill up dmk_learners (new / GX)
        create dmk_refs <- only in the first loop
            
    2. train (learners)
        copy learners to new age (+1)
        split dmk_learners into groups of ndmk_TR
        train each group against dmk_refs
    
    3. test (learners & refs)
        prepare list of DMKs to test
        split into groups of ndmk_TS
        test may be broken with 'separated' condition
    
    4. analyse / report results of learners and refs
        
    5. manage / modify DMKs lists (learners & refs)
    
    6. adjust TS game size
    
    7. PMT evaluation
    """
    while True:

        loop_stime = time.time()

        if cm.exit:
            logger.info('train loop exits')
            cm.exit = False
            break

        logger.info(f'\n ************** starts loop #{loop_ix} **************')

        tbwr.add(value=cm.game_size_TS, tag=f'loop/game_size_TS', step=loop_ix)
        tbwr.add(value=cm.game_size_TR, tag=f'loop/game_size_TR', step=loop_ix)

        #************************************************************************************* 1. eventually create DMKs

        # fill up dmk_learners (new / GX)
        if len(dmk_learners) < cm.ndmk_learners:

            logger.info(f'building {cm.ndmk_learners - len(dmk_learners)} new DMKs (learners):')

            learners_families = {dn: FolDMK.load_point(name=dn)['family'] for dn in dmk_learners}
            refs_families = {dn: FolDMK.load_point(name=dn)['family'] for dn in dmk_refs}

            # look for forced families
            families_present = ''.join(list(learners_families.values()))
            families_count = {fm: families_present.count(fm) for fm in cm.families}
            families_count = [(fm, families_count[fm]) for fm in families_count]
            families_forced = [fc[0] for fc in families_count if fc[1] < 1]
            if families_forced: logger.info(f'families forced: {families_forced}')

            cix = 0
            while len(dmk_learners) < cm.ndmk_learners:

                # build new one from forced
                if families_forced:
                    family = families_forced.pop()
                    name_child = f'dmk{loop_ix:02}{family}{cix:02}_00'
                    logger.info(f'> {name_child} <- fresh, forced from family {family}')
                    build_single_foldmk(
                        name=   name_child,
                        family= family,
                        logger= get_child(logger, change_level=10))

                else:

                    pa = random.choice(dmk_refs) if dmk_refs else None
                    family = refs_families[pa] if pa is not None else random.choice(cm.families)
                    name_child = f'dmk{loop_ix:02}{family}{cix:02}_00'

                    # 100% fresh DMK from selected family
                    if random.random() < cm.prob_fresh_dmk or pa is None:
                        logger.info(f'> {name_child} <- 100% fresh')
                        build_single_foldmk(
                            name=   name_child,
                            family= family,
                            logger= get_child(logger, change_level=10))

                    # TODO: check if it works now ..with refs
                    # GX from refs
                    else:
                        other_fam = [dn for dn in dmk_refs if refs_families[dn] == family]
                        if len(other_fam) > 1:
                            other_fam.remove(pa)
                        pb = random.choice(other_fam)

                        ckpt_fresh = random.random() < cm.prob_fresh_ckpt
                        ckpt_fresh_info = ' (fresh ckpt)' if ckpt_fresh else ''
                        name_child = f'dmk{loop_ix:02}{family}{cix:02}'
                        logger.info(f'> {name_child} = {pa} + {pb}{ckpt_fresh_info}')
                        FolDMK.gx_saved(
                            name_parent_main=           pa,
                            name_parent_scnd=           pb,
                            name_child=                 name_child,
                            save_topdir_parent_main=    DMK_MODELS_FD,
                            do_gx_ckpt=                 not ckpt_fresh,
                            logger=                     get_child(logger, change_level=10))

                dmk_learners.append(name_child)
                cix += 1

        # create dmk_refs (in first loop)
        if loop_ix == 1:

            dmk_refs_from_learners = dmk_learners[:cm.ndmk_refs]
            dmk_refs = [f'{dn}_ref' for dn in dmk_refs_from_learners]
            copy_dmks(
                names_src=  dmk_refs_from_learners,
                names_trg=  dmk_refs,
                logger=     get_child(logger, change_level=10))

            cix = len(dmk_learners)
            while len(dmk_refs) < cm.ndmk_refs:
                family = random.choice(cm.families)
                name_child = f'dmk{loop_ix:02}{family}{cix:02}_00_ref'
                cix += 1
                build_single_foldmk(
                    name=   name_child,
                    family= family,
                    logger= get_child(logger, change_level=10))
                dmk_refs.append(name_child)

            logger.info(f'created {len(dmk_refs)} refs DMKs: {dmk_refs}')

        logger.info(f'loop #{loop_ix} DMKs:')
        logger.info(f'> learners: ({len(dmk_learners)}) {", ".join(dmk_learners)}')
        logger.info(f'> refs:     ({len(dmk_refs)}) {", ".join(dmk_refs)}')

        #******************************************************************************************* 2. train (learners)

        # copy dmk_learners to new age
        dmk_learners_aged = [f'{dn[:-2]}{int(dn[-2:])+1:02}' for dn in dmk_learners]
        copy_dmks(
            names_src=  dmk_learners,
            names_trg=  dmk_learners_aged,
            logger=     get_child(logger, change_level=10))

        # create groups by evenly distributing DMKs
        n_groups = math.ceil(len(dmk_learners_aged) / cm.ndmk_TR)
        tr_groups = [[] for _ in range(n_groups)]
        gix = 0
        for dn in dmk_learners_aged:
            fix = gix % n_groups
            tr_groups[fix].append(dn)
            gix += 1

        pub_ref = {
            'publish_player_stats': False,
            'publish_pex':          False,
            'publish_update':       False,
            'publish_more':         False}
        pub_TR = {
            'publish_player_stats': False,
            'publish_pex':          False,
            'publish_update':       True,
            'publish_more':         False}
        for trg in tr_groups:
            run_GM(
                dmk_point_ref=  [{'name':dn, 'motorch_point':{'device':0}, **pub_ref} for dn in dmk_refs],
                dmk_point_TRL=  [{'name':dn, 'motorch_point':{'device':1}, **pub_TR}  for dn in trg],
                game_size=      cm.game_size_TR,
                dmk_n_players=  cm.dmk_n_players_TR,
                logger=         logger)

        #************************************************************************************* 3. test (learners & refs)
        
        # TODO: if refs group has not changed since last loop -> some DMKs may not need to be tested

        ### prepare full list of DMKs to test

        sep_pairs = list(zip(dmk_learners, dmk_learners_aged))

        # if there are refs that are not present in learners, those need to be tested also (and then deleted)
        dmk_refs_to_test = []
        for dn in dmk_refs:
            dn_test = dn[:-4]
            if dn_test not in dmk_learners:
                dmk_refs_to_test.append(dn)
        dmk_refs_copied_to_test = []
        if dmk_refs_to_test:
            dmk_refs_copied_to_test = [dn[:-4] for dn in dmk_refs_to_test]
            copy_dmks(
                names_src=  dmk_refs_to_test,
                names_trg=  dmk_refs_copied_to_test,
                logger=     get_child(logger, change_level=10))

        # create groups by evenly distributing DMKs
        ndmk_TS = len(sep_pairs) * 2 + len(dmk_refs_copied_to_test)
        n_groups = math.ceil(ndmk_TS / cm.ndmk_TS)
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

        pub = {
            'publish_player_stats': True,
            'publish_pex':          False, # won't matter since PL does not pex
            'publish_update':       False,
            'publish_more':         False}
        speedL = []
        dmk_results = {}
        for tsg,pairs in zip(ts_groups,group_pairs):
            rgd = run_GM(
                dmk_point_ref=      [{'name':dn, 'motorch_point':{'device': 0}, **pub_ref} for dn in dmk_refs],
                dmk_point_PLL=      [{'name':dn, 'motorch_point':{'device': 1}, **pub}     for dn in tsg],
                game_size=          cm.game_size_TS,
                dmk_n_players=      cm.dmk_n_players_TS,
                sep_pairs=          pairs,
                sep_pairs_factor=   cm.sep_pairs_factor,
                sep_n_stdev=        cm.sep_n_stdev,
                logger=             logger)
            speedL.append(rgd['loop_stats']['speed'])
            dmk_results.update(rgd['dmk_results'])

        speed_TS = sum(speedL) / len(speedL)
        tbwr.add(value=speed_TS, tag=f'loop/speed_Hs', step=loop_ix)

        # instantiate results of refs
        for dn in dmk_refs:
            dmk_results[dn] = dmk_results[dn[:-4]]

        # delete dmk_refs_copied_to_test
        for dn in dmk_refs_copied_to_test:
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        #************************************************************************************** 4. analyse & log results

        sr = separation_report(
            dmk_results=    dmk_results,
            n_stdev=        cm.sep_n_stdev,
            sep_pairs=      sep_pairs)

        # update dmk_results
        session_lifemarks = ''
        for ix,dna in enumerate(dmk_learners_aged):
            dn = dmk_learners[ix]
            dmk_results[dna]['separated'] = sr['sep_pairs_stat'][ix]
            dmk_results[dna]['wonH_diff'] = dmk_results[dna]['wonH_afterIV'][-1] - dmk_results[dn]['wonH_afterIV'][-1]
            lifemark_upd = '/' if dmk_results[dna]['wonH_diff'] > 0 else '|'
            if dmk_results[dna]['separated']:
                lifemark_upd = '+' if dmk_results[dna]['wonH_diff'] > 0 else '-'
            lifemark_prev = loops_results['lifemarks'][dn] if dn in loops_results['lifemarks'] else ''
            dmk_results[dna]['lifemark'] = lifemark_prev + lifemark_upd
            session_lifemarks += lifemark_upd

        not_sep_count = session_lifemarks.count('|') + session_lifemarks.count('/')
        sep_factor = (len(session_lifemarks) - not_sep_count) / len(session_lifemarks)
        tbwr.add(value=sep_factor, tag=f'loop/sep_factor', step=loop_ix)

        # rank learners by last_wonH_afterIV
        dmk_rw = [(dn, dmk_results[dn]['last_wonH_afterIV']) for dn in dmk_learners_aged]
        learners_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x:x[1], reverse=True)]

        res_nfo = f'learners results:\n'
        for dn in learners_ranked:
            wonH = dmk_results[dn]['last_wonH_afterIV']
            wonH_diff = dmk_results[dn]['wonH_diff']
            wonH_IV_std = dmk_results[dn]['wonH_IV_stdev']
            wonH_mstd = dmk_results[dn]['wonH_IV_mean_stdev']
            wonH_mstd_str = f'[{wonH_IV_std:.2f}/{wonH_mstd:.2f}]'
            sep = ' s' if dmk_results[dn]['separated'] else '  '
            lifemark = f' {dmk_results[dn]["lifemark"]}'
            stats_nfo = ''
            for k in dmk_results[dn]["global_stats"]:
                v = dmk_results[dn]["global_stats"][k]
                stats_nfo += f'{k}:{v:4.1f} '
            res_nfo += f'{dn:18} : {wonH:6.2f} {wonH_mstd_str:>12}  {stats_nfo}  d: {wonH_diff:6.2f}{sep}{lifemark}\n'
        logger.info(res_nfo)

        logger.info(f'refs results:\n{results_report(dmk_results, dmks=dmk_refs)}')

        #********************************************************************************* 5. manage / modify DMKs lists

        ### prepare new list of learners

        dmk_learners_asi = [dn for dn in learners_ranked if dmk_results[dn]['separated'] and dmk_results[dn]['wonH_diff']>0]
        logger.info(f'learners aged & separated & improved: {", ".join(dmk_learners_asi)}')

        dmk_learners_updated = []
        for ix,dna in enumerate(dmk_learners_aged):
            dn = dmk_learners[ix]

            # replace learner by aged & separated & improved
            if dna in dmk_learners_asi:
                dmk_learners_updated.append(dna)
            # save lifemark of not improved
            else:
                dmk_results[dn]['lifemark'] = dmk_results[dna]['lifemark']
                dmk_learners_updated.append(dn)

        ### remove bad lifemarks from learners

        dmk_learners_bad_lifemark = []
        for dn in dmk_learners_updated:
            lifemark_ending = dmk_results[dn]['lifemark'][-sum(cm.remove_key):]
            if lifemark_ending.count('-') + lifemark_ending.count('|') >= cm.remove_key[0] and lifemark_ending[-1] not in '/+':
                dmk_learners_bad_lifemark.append(dn)

        if dmk_learners_bad_lifemark:
            logger.info(f'removing learners with bad lifemark: {", ".join(dmk_learners_bad_lifemark)}')
            dmk_learners_updated = [dn for dn in dmk_learners_updated if dn not in dmk_learners_bad_lifemark]

        # clean out folder
        for dn in dmk_learners + dmk_learners_aged:
            if dn not in dmk_learners_updated:
                shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        dmk_learners = dmk_learners_updated

        loops_results['lifemarks'] = {}
        for dn in dmk_learners:
            loops_results['lifemarks'][dn] = dmk_results[dn]['lifemark']

        ### replace some refs by learners that improved

        dmk_rw = [(dn, dmk_results[dn]['last_wonH_afterIV']) for dn in dmk_refs]
        refs_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x: x[1], reverse=True)]

        dmk_add_to_refs = []
        for dn in reversed(dmk_learners_asi[:cm.ndmk_refs]): # reversed to start from worst

            replaced_by_age = None
            for dnr in refs_ranked:
                if dnr[:-7] == dn[:-3]:
                    dmk_add_to_refs.append(dn)
                    replaced_by_age = dnr
                    break

            if replaced_by_age:
                refs_ranked.remove(replaced_by_age)
            else:

                dnr = refs_ranked[-1]

                if separated_factor(
                    a_wonH=             dmk_results[dn]['last_wonH_afterIV'],
                    a_wonH_mean_stdev=  dmk_results[dn]['wonH_IV_mean_stdev'],
                    b_wonH=             dmk_results[dnr]['last_wonH_afterIV'],
                    b_wonH_mean_stdev=  dmk_results[dnr]['wonH_IV_mean_stdev'],
                    n_stdev=            cm.sep_n_stdev,
                ) >= 1:
                    dmk_add_to_refs.append(dn)
                    refs_ranked.remove(dnr)

        refs_gain = sum([dmk_results[dn]['wonH_diff'] for dn in dmk_add_to_refs])

        if dmk_add_to_refs:
            dmk_add_to_refs = list(reversed(dmk_add_to_refs)) # reverse it back
            logger.info(f'adding to refs: {", ".join(dmk_add_to_refs)}; +refs_gain: {refs_gain:.2f}')

            for dn in dmk_refs:
                if dn not in refs_ranked:
                    shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

            new_ref_names = [f'{dn}_ref' for dn in dmk_add_to_refs]
            copy_dmks(
                names_src=  dmk_add_to_refs,
                names_trg=  new_ref_names,
                logger=     get_child(logger, change_level=10))

            dmk_refs = refs_ranked + new_ref_names

        # prepare refs_ranked after all changes and save
        for dn in dmk_refs:
            dmk_results[dn] = dmk_results[dn[:-4]] # copy again for new refs
        dmk_rw = [(dn, dmk_results[dn]['last_wonH_afterIV']) for dn in dmk_refs]
        refs_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x: x[1], reverse=True)]
        loops_results['refs_ranked'] = refs_ranked

        refs_diff = dmk_results[refs_ranked[0]]['last_wonH_afterIV'] - dmk_results[refs_ranked[-1]]['last_wonH_afterIV']
        refs_wonH_IV_stdev_avg = sum([dmk_results[dn]['wonH_IV_stdev'] for dn in dmk_refs]) / cm.ndmk_refs

        tbwr.add(value=refs_gain,               tag=f'loop/refs_gain',              step=loop_ix)
        tbwr.add(value=refs_diff,               tag=f'loop/refs_diff',              step=loop_ix)
        tbwr.add(value=refs_wonH_IV_stdev_avg,  tag=f'loop/refs_wonH_IV_stdev_avg', step=loop_ix)

        dmk_sets = {
            'refs_best':    [refs_ranked[0]],
            'refs_avg':     refs_ranked}
        for dsn in dmk_sets:
            gsL = [dmk_results[dn]['global_stats'] for dn in dmk_sets[dsn]]
            gsa_avg = {k: [] for k in gsL[0]}
            for e in gsL:
                for k in e:
                    gsa_avg[k].append(e[k])
            for k in gsa_avg:
                gsa_avg[k] = sum(gsa_avg[k]) / len(gsa_avg[k])
                tbwr.add(
                    value=  gsa_avg[k],
                    tag=    f'loop_poker_stats_{dsn}/{k}',
                    step=   loop_ix)

        loops_results['loop_ix'] = loop_ix
        w_json(loops_results, RESULTS_FP)

        # *************************************************************************************** 6. adjust TS game size

        print('since_last_update:',since_last_update)
        if since_last_update > 0:
            since_last_update -= 1

        if since_last_update == 0:

            lifemarks_str = ''
            for dn in loops_results['lifemarks']:
                lifemarks_str += loops_results['lifemarks'][dn][-cm.update_interval:]
            print('lifemarks:', lifemarks_str)

            not_sep_count = lifemarks_str.count('|') + lifemarks_str.count('/')
            print(f'not_sep_count: {not_sep_count} / {len(lifemarks_str)}')
            if not_sep_count / len(lifemarks_str) >= 0.5:
                cm.game_size_TS += cm.update_size
                since_last_update = cm.update_interval
                logger.info(f'increased game_size_TS to {cm.game_size_TS}')


        if cm.pause: input("press Enter to continue..")

        loop_time = (time.time() - loop_stime) / 60
        logger.info(f'loop {loop_ix} finished, time taken: {loop_time:.1f}min')
        tbwr.add(value=loop_time, tag=f'loop/loop_time', step=loop_ix)

        #********************************************************************************************* 7. PMT evaluation

        if loop_ix % cm.n_loops_PMT == 0:
            new_master = refs_ranked[0]
            copy_dmks(
                names_src=          [new_master],
                names_trg=          [f'{new_master[:-4]}_pmt{loop_ix // cm.n_loops_PMT:02}'],
                save_topdir_trg=    PMT_FD,
                logger=             get_child(logger, change_level=10))
            logger.info(f'copied {new_master} to PMT')

            all_pmt = get_saved_dmks_names(PMT_FD)
            if len(all_pmt) > 2: # skips some
                logger.info(f'PMT starts..')

                # create groups by evenly distributing DMKs
                n_groups = math.ceil(len(all_pmt) / cm.ndmk_TS)
                ts_groups = [[] for _ in range(n_groups)]
                gix = 0
                for dn in all_pmt:
                    fix = gix % n_groups
                    ts_groups[fix].append(dn)
                    gix += 1

                pub_ref = {
                    'publish_player_stats': False,
                    'publish_pex':          False,
                    'publish_update':       False,
                    'publish_more':         False}
                pub = {
                    'publish_player_stats': True,
                    'publish_pex':          False,
                    'publish_update':       False,
                    'publish_more':         False}
                pmt_results = {}
                for tsg in ts_groups:
                    rgd = run_GM(
                        dmk_point_ref=  [{'name':dn, 'motorch_point':{'device':0}, **pub_ref} for dn in dmk_refs],
                        dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':1}, 'save_topdir':PMT_FD, **pub} for dn in all_pmt],
                        game_size=      cm.game_size_TS,
                        dmk_n_players=  cm.dmk_n_players_TS,
                        sep_all_break=  True,
                        logger=         logger)
                    pmt_results.update(rgd['dmk_results'])

                logger.info(f'PMT results:\n{results_report(pmt_results)}')

                # remove worst
                if len(all_pmt) == cm.ndmk_PMT:
                    dmk_rw = [(dn, pmt_results[dn]['last_wonH_afterIV']) for dn in pmt_results]
                    pmt_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x: x[1], reverse=True)]
                    dn = pmt_ranked[-1]
                    shutil.rmtree(f'{PMT_FD}/{dn}', ignore_errors=True)
                    logger.info(f'removed PMT: {dn}')

        loop_ix += 1