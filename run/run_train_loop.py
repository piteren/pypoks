"""
    This script trains a group of DMKs.
    DMKs are trained from a scratch with self-play game.
    Configuration is set to fully utilize power of 2 GPUs (11GB RAM).

    Below are descriptions of some structures used

        loops_results = {
            'loop_ix': 5 (int),                                 <- number of loops performed yet
            'lifemarks': {
                'dmk01a00_03': '++-',
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
from typing import List

from envy import DMK_MODELS_FD, PMT_FD, CONFIG_FP, RESULTS_FP
from podecide.dmk import FolDMK
from podecide.games_manager import stdev_with_none, separation_report
from run.functions import get_saved_dmks_names, run_GM, copy_dmks, build_single_foldmk
from run.after_run.ranks import get_ranks

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

    'game_size_upd':            100000,     # how much increase TR or TS game size when needed
    'min_sep':                  0.4,        # -> 0.7    min factor of DMKs separated, if lower TS game is increased
    'factor_TS_TR':             2,          # max factor TS_game_size/TR_game_size, if higher TR is increased
        # train
    # TODO: temp set (30K)
    'game_size_TR':             50000,
    'dmk_n_players_TR':         150,        # number of players per DMK while TR
        # test
    # TODO: temp set (30K)
    'game_size_TS':             50000,
    'dmk_n_players_TS':         150,        # number of players per DMK while TS
    'sep_pairs_factor':         0.8,        # -> 1.0    pairs separation break value
    # TODO: temp set (2.0)
    'sep_n_stdev':              1.0,        # separation won IV mean stdev factor
        # replace / new
    'rank_mavg_factor':         0.3,        # mavg_factor of rank_smooth calculation
    'safe_rank':                0.5,        # <0.0;1.0> factor of rank_smooth that is safe (not considered to be replaced, 0.6 means that top 60% of rank is safe)
    'remove_key':               [4,1],      # [A,B] remove DMK if in last A+B life marks there are A -|
    'prob_fresh_dmk':           0.8,        # -> 0.5    probability of 100% fresh DMK
    'prob_fresh_ckpt':          0.8,        # -> 0.5    probability of fresh checkpoint (child from GX of point only, without GX of ckpt)
        # PMT (Periodical Masters Test)
    'n_loops_PMT':              5,          # do PMT every N loops
    'n_dmk_PMT':                20}         # max number of DMKs (masters) in PMT

# TODO:
#  - keep and display wonH + std
#  - some ref may not be in learners, those need to be duplicated without _ref to test and then deleted
#  - age of GXed learner


if __name__ == "__main__":

    tl_name = f'run_train_loop_{stamp()}'
    logger = get_pylogger(
        name=       tl_name,
        add_stamp=  False,
        folder=     DMK_MODELS_FD,
        level=      20)

    logger.info(f'train_loop {tl_name} starts..')

    cm = ConfigManager(file_FP=CONFIG_FP, config_init=CONFIG_INIT, logger=logger)
    tbwr = TBwr(logdir=f'{DMK_MODELS_FD}/{tl_name}')

    # initial values
    loop_ix = 1
    dmk_learners = []
    dmk_refs = []

    # check for continuation
    loops_results = r_json(RESULTS_FP)
    if loops_results:

        saved_n_loops = loops_results['loop_ix']
        logger.info(f'Do you want to continue with saved (in {DMK_MODELS_FD}) {saved_n_loops} loops? ..waiting 10 sec (y/n, y-default)')

        i, o, e = select.select([sys.stdin], [], [], 10)
        if i and sys.stdin.readline().strip() == 'n':
            pass
        else:
            logger.info(f'> continuing with saved {saved_n_loops} loops')

            loop_ix = saved_n_loops + 1

            saved_dmks = get_saved_dmks_names()
            dmk_refs = [dn for dn in saved_dmks if dn.endswith('_ref')]
            dmk_learners = [dn for dn in saved_dmks if dn not in dmk_refs]

    else:
        loops_results = {'loop_ix': 0, 'lifemarks': {}}

    """
        new name pattern:
        f'dmk{loop_ix:02}{family}{cix:02}' -> f'dmk{loop_ix:02}{family}{cix:02}_{age:02}' + optional '_ref'

    1. eventually create missing dmk_learners (new / GX)
    2. train
        split dmk_learners into groups of ndmk_TR
        train each group against dmk_refs
        
        
    3. test trainable + _old, test has to
        - compare 'actual' vs '_old' and check which significantly improved or degraded
        - set actual ranking (masters, poor ones)
        test may be broken with 'separated' condition
    4. analyse results
        report
        roll back from _old those who not improved
        adjust TR / TS parameters     
    5. eventually remove poor & save all_results
    6. do PMT every N loop
    """
    while True:

        logger.info(f'\n ************** starts loop {loop_ix} **************')
        loop_stime = time.time()

        if cm.exit:
            logger.info('train loop exits')
            cm.exit = False
            break

        ### 1. eventually create missing dmk_learners (new / GX)

        if len(dmk_learners) < cm.ndmk_learners:

            logger.info(f'building {cm.ndmk_learners - len(dmk_learners)} new DMKs (learners):')

            ranked_families = {dn: FolDMK.load_point(name=dn)['family'] for dn in dmk_learners}

            # look for forced families
            families_present = ''.join(list(ranked_families.values()))
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
                    family = ranked_families[pa] if pa is not None else random.choice(cm.families)
                    name_child = f'dmk{loop_ix:02}{family}{cix:02}_00'

                    # 100% fresh DMK from selected family
                    if random.random() < cm.prob_fresh_dmk or pa is None:
                        logger.info(f'> {name_child} <- 100% fresh')
                        build_single_foldmk(
                            name=   name_child,
                            family= family,
                            logger= get_child(logger, change_level=10))

                    # TODO: check if it works with refs
                    # GX from refs
                    else:
                        other_fam = [dn for dn in dmk_refs if ranked_families[dn] == family]
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

        # create dmk_refs in first loop
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


        ### 2. train

        # copy dmk_learners to new age
        dmk_learners_aged = [f'{dn[:-2]}{int(dn[-2:])+1:02}' for dn in dmk_learners]
        copy_dmks(
            names_src=  dmk_learners,
            names_trg=  dmk_learners_aged,
            logger=     get_child(logger, change_level=10))

        # create groups
        tr_groups = []
        dmk_learners_copy = [] + dmk_learners_aged
        while dmk_learners_copy:
            tr_groups.append(dmk_learners_copy[:cm.ndmk_TR])
            dmk_learners_copy = dmk_learners_copy[cm.ndmk_TR:]

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

        ### 3. test
        
        # TODO: if refs group has not changed since last loop -> some DMKs may not need to be tested

        # prepare full list of DMKs for TS
        dmks_PLL = dmk_learners + dmk_learners_aged
        sep_pairs = list(zip(dmk_learners,dmk_learners_aged)) # TODO: will we use this concept still?
        dmk_refs_to_test = []
        for dn in dmk_refs:
            dn_test = dn[:-4]
            if dn_test not in dmks_PLL:
                dmk_refs_to_test.append(dn)
        dmk_refs_copied_to_test = []
        if dmk_refs_to_test:
            dmk_refs_copied_to_test = [dn[:-4] for dn in dmk_refs_to_test]
            copy_dmks(
                names_src=  dmk_refs_to_test,
                names_trg=  dmk_refs_copied_to_test,
                logger=     get_child(logger, change_level=10))
            dmks_PLL += dmk_refs_copied_to_test

        # create groups
        ts_groups = []
        while dmks_PLL:
            ts_groups.append(dmks_PLL[:cm.ndmk_TS])
            dmks_PLL = dmks_PLL[cm.ndmk_TS:]

        pub = {
            'publish_player_stats': True,
            'publish_pex':          False, # won't matter since PL does not pex
            'publish_update':       False,
            'publish_more':         False}
        speedL = []
        dmk_results = {}
        for tsg in ts_groups:
            rgd = run_GM(
                dmk_point_PLL=      [{'name':dn, 'motorch_point':{'device':n%2}, **pub} for n,dn in enumerate(tsg)],
                game_size=          cm.game_size_TS,
                dmk_n_players=      cm.dmk_n_players_TS,
                sep_pairs=          sep_pairs,
                sep_pairs_factor=   cm.sep_pairs_factor,
                sep_n_stdev=        cm.sep_n_stdev,
                logger=             logger)
            speedL.append(rgd['loop_stats']['speed'])
            dmk_results.update(rgd['dmk_results'])

        speed_TS = sum(speedL) / len(speedL)

        # delete dmk_refs_copied_to_test
        for dn in dmk_refs_copied_to_test:
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        ### 4. analyse results

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
            lifemark_upd = '|'
            if dmk_results[dna]['separated']:
                lifemark_upd = '+' if dmk_results[dna]['wonH_diff'] > 0 else '-'
            # TODO: ? remove old lifemark (dn) from loops_results
            lifemark_prev = loops_results['lifemarks'][dn] if dn in loops_results['lifemarks'] else ''
            dmk_results[dna]['lifemark'] = lifemark_prev + lifemark_upd
            session_lifemarks += lifemark_upd

        sep_factor = (len(session_lifemarks) - session_lifemarks.count('|')) / len(session_lifemarks)

        ### log results

        # prepare new rank(ed) ..sorted by wonH_afterIV
        dmk_rw = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in dmk_learners_aged]
        dmk_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x:x[1], reverse=True)]

        # TODO: ranks won't be needed anymore
        #ranks_smooth = get_ranks(all_results=all_results, mavg_factor=cm.rank_mavg_factor)['ranks_smooth']

        res_nfo = f'DMKs train results:\n'
        for pos,dn in enumerate(dmk_ranked):
            #rank_smth_norm = f'{ranks_smooth[dn][-1] / cm.n_dmk_total:4.2f}' if ranks_smooth[dn] else '----'
            wonH = dmk_results[dn]['last_wonH_afterIV']
            wonH_diff = dmk_results[dn]['wonH_diff']
            wonH_mstd = dmk_results[dn]['wonH_IV_mean_stdev']
            wonH_mstd_str = f'[{wonH_mstd:.2f}]'
            sep = ' s' if dmk_results[dn]['separated'] else '  '
            lifemark = f' {dmk_results[dn]["lifemark"]}'
            res_nfo += f' > {pos:>2} {dn} : {wonH:6.2f} {wonH_mstd_str:9} :: {wonH_diff:6.2f}{sep}{lifemark}\n'
        logger.info(res_nfo)

        ### TB log

        # TODO: refactor, show: refs_gain
        masters = dmk_ranked[:cm.n_dmk_master]
        masters_res = [dmk_results[dn]['wonH_old_diff'] for dn in masters if dmk_results[dn]['separated_old']]
        masters_gain = sum([v for v in masters_res if v > 0])
        masters_wonH_avg = sum([dmk_results[dn]['wonH_afterIV'][-1] for dn in masters]) / len(masters)
        masters_wonH_std_avg = sum([stdev_with_none(dmk_results[dn]['wonH_IV']) for dn in masters]) / len(masters)

        tbwr.add(value=masters_gain,            tag=f'loop/masters_gain',           step=loop_ix)
        tbwr.add(value=masters_wonH_avg,        tag=f'loop/masters_wonH_avg',       step=loop_ix)
        tbwr.add(value=masters_wonH_std_avg,    tag=f'loop/masters_wonH_std_avg',   step=loop_ix)
        tbwr.add(value=speed_TS,                tag=f'loop/speed_Hs',               step=loop_ix)
        tbwr.add(value=cm.game_size_TS,         tag=f'loop/game_size_TS',           step=loop_ix)
        tbwr.add(value=cm.game_size_TR,         tag=f'loop/game_size_TR',           step=loop_ix)
        tbwr.add(value=sep_factor,              tag=f'loop/sep_factor',             step=loop_ix)

        """
        # eventually increase game_size
        if sep_factor < cm.min_sep:
            cm.game_size_TS = cm.game_size_TS + cm.game_size_upd
        if cm.game_size_TS > cm.game_size_TR * cm.factor_TS_TR:
            cm.game_size_TR = cm.game_size_TR + cm.game_size_upd

        # search for significantly_learned, remove their _old
        significantly_learned = [dn for dn in dmk_ranked if dmk_results[dn]['lifemark'][-1] == '+']
        logger.info(f'got significantly_learned: {significantly_learned}')

        # eventually roll back some from _old
        roll_back_from_old = [dn for dn in dmk_ranked if dn not in significantly_learned]
        if roll_back_from_old:

            logger.info(f'rolling back: {roll_back_from_old}')
            copy_dmks(
                names_src=  [f'{nm}_old' for nm in roll_back_from_old],
                names_trg=  roll_back_from_old,
                logger=     get_child(logger, change_level=10))

            # update dmk_results of rolled back, then remove all _old
            for dn in roll_back_from_old:
                dmk_results[dn]['wonH_IV'] = dmk_results[f'{dn}_old']['wonH_IV']
                dmk_results[dn]['wonH_afterIV'] = dmk_results[f'{dn}_old']['wonH_afterIV']
                dmk_results[dn]['age'] -= 1

            # prepare new rank(ed) again (with rolled back)
            dmk_rw = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in dmk_ranked]
            dmk_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x: x[1], reverse=True)]

        dmk_sets = {
            'best':         [dmk_ranked[0]],
            'masters_avg':  dmk_ranked[:cm.n_dmk_master]}
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

        # finally update all_results
        all_results['loops'][str(loop_ix)] = dmk_ranked
        for dn in dmk_ranked:
            all_results['lifemarks'][dn] = dmk_results[dn]['lifemark']

        ranks_smooth = get_ranks(all_results=all_results, mavg_factor=cm.rank_mavg_factor)['ranks_smooth']

        # log again results if roll back has been done
        if roll_back_from_old:
            res_nfo = f'DMKs train results after roll back:\n'
            for pos,dn in enumerate(dmk_ranked):
                rank_smth_norm = f'{ranks_smooth[dn][-1] / cm.n_dmk_total:4.2f}' if ranks_smooth[dn] else '----'
                name_aged = f'{dn}({dmk_results[dn]["age"]})'
                wonH_afterIV = dmk_results[dn]['wonH_afterIV'][-1]
                res_nfo += f' > {pos:>2}({rank_smth_norm}) {name_aged:15s} : {wonH_afterIV:6.2f}\n'
            logger.info(res_nfo)

        ### 5. eventually remove poor and finally save all_results

        dmk_masters = dmk_ranked[:cm.n_dmk_master]
        dmk_poor = dmk_ranked[cm.n_dmk_master:]

        rank_candidates = [dn for dn in dmk_poor if ranks_smooth[dn][-1] > cm.n_dmk_total * cm.safe_rank]
        logger.info(f'rank_candidates: {rank_candidates}')

        lifemark_candidates = []
        for dn in dmk_poor:
            lifemark_ending = all_results['lifemarks'][dn][-sum(cm.remove_key):]
            if lifemark_ending.count('-') + lifemark_ending.count('|') >= cm.remove_key[0] and lifemark_ending[-1] != '+':
                lifemark_candidates.append(dn)
        logger.info(f'lifemark_candidates: {lifemark_candidates}')

        remove = set(rank_candidates) & set(lifemark_candidates)
        if remove:
            logger.info(f'DMKs to remove: {remove}')

            for dn in remove:
                shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)
                shutil.rmtree(f'{DMK_MODELS_FD}/{dn}_old', ignore_errors=True)
                dmk_ranked.remove(dn)

            all_results['loops'][str(loop_ix)] = dmk_ranked  # update with new dmk_ranked

        w_json(all_results, RESULTS_FP)

        ### 6. PMT evaluation

        if loop_ix % cm.n_loops_PMT == 0:

            # copy masters with new name
            logger.info(f'Running PMT..')
            new_master = dmk_ranked[0]
            copy_dmks(
                names_src=          [new_master],
                names_trg=          [f'{new_master}_pmt{loop_ix // cm.n_loops_PMT:03}'],
                save_topdir_trg=    PMT_FD,
                logger=             get_child(logger, change_level=10))
            logger.info(f'copied {new_master} to PMT')

            all_pmt = get_saved_dmks_names(PMT_FD)
            if len(all_pmt) > 2: # skips some

                pub = {
                    'publish_player_stats': True,
                    'publish_pex':          False,
                    'publish_update':       False,
                    'publish_more':         False}
                pmt_results = run_GM(
                    dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':n%2}, 'save_topdir':PMT_FD, **pub} for n,dn in enumerate(all_pmt)],
                    game_size=      cm.game_size_TS * 2,
                    dmk_n_players=  cm.dmk_n_players_TS,
                    sep_all_break=  True,
                    logger=         logger)['dmk_results']

                pmt_ranked = [(dn, pmt_results[dn]['wonH_afterIV'][-1]) for dn in pmt_results]
                pmt_ranked = [e[0] for e in sorted(pmt_ranked, key=lambda x: x[1], reverse=True)]

                pmt_nfo = 'PMT results (wonH):\n'
                pos = 1
                for dn in pmt_ranked:
                    name_aged = f'{dn}({pmt_results[dn]["age"]}){pmt_results[dn]["family"]}'
                    pmt_nfo += f' > {pos:>2} {name_aged:25s} : {pmt_results[dn]["wonH_afterIV"][-1]:6.2f}\n'
                    pos += 1
                logger.info(pmt_nfo)

                # remove worst
                if len(all_pmt) == cm.n_dmk_PMT:
                    dn = pmt_ranked[-1]["name"]
                    shutil.rmtree(f'{PMT_FD}/{dn}', ignore_errors=True)
                    logger.info(f'removed PMT: {dn}')

        if cm.pause: input("press Enter to continue..")

        loop_time = (time.time() - loop_stime) / 60
        logger.info(f'loop {loop_ix} finished, time taken: {loop_time:.1f}min')
        tbwr.add(value=loop_time, tag=f'loop/loop_time', step=loop_ix)

        loop_ix += 1
"""