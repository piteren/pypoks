from pypaq.lipytools.files import prep_folder, r_json, w_json
from pypaq.lipytools.plots import two_dim
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.stats import mam
from pypaq.pms.config_manager import ConfigManager
import random
import select
import shutil
import sys

from pypoks_envy import DMK_MODELS_FD, AGE_FD, PMT_FD, CONFIG_FP, RESULTS_FP
from podecide.dmk import FolDMK
from podecide.games_manager import separation_report
from run.functions import get_saved_dmks_names, build_from_names, run_GM, copy_dmks, build_single_foldmk
from run.after_run.ranks import get_ranks

CONFIG_INIT = {
    'exit':                 False,      # exits loop (after train)
    'pause':                False,      # pauses loop after test till Enter pressed
    'families':             'abc',      # active families
    'n_dmk_total':          10,         # total number of trainable DMKs (population size)  // 12x cN12 playing / GPU 6x cN12 training
    'n_dmk_master':         5,          # number of 'masters' DMKs (trainable are trained against them)
    'n_dmk_TR_group':       5,          # DMKs are trained with masters in groups of that size (group is build of n_dmk_TR_group + n_dmk_master)
    # train
    'game_size_TR':         200000,
    'dmk_n_players_TR':     150,
    # test
    'game_size_TS':         300000,
    'dmk_n_players_TS':     150,        # number of players per DMK while TS // 300 is ok, but loads a lot of RAM
    'sep_bvalue':           1.2,
    'n_stdev':              2.0,
    # children
    'remove_key':           [5,1],      # [A,B] remove DMK if in last A+B life marks there are A minuses
    'prob_fresh':           0.5,        # probability of fresh child (GX of point only, without GX of ckpt)
    # PMT (Periodical Masters Test)
    'n_loops_PMT':          10,         # do PMT every N loops
    'n_dmk_PMT':            20,         # max number of DMKs (masters) in PMT
}


if __name__ == "__main__":

    # check for continuation
    loop_ix = 1
    _continue = False
    saved_n_loops = None
    all_results = r_json(RESULTS_FP)
    if all_results:
        saved_n_loops = all_results['n_loops']
        print(f'Do you want to continue with saved (in {DMK_MODELS_FD}) {saved_n_loops} loops? ..waiting 10 sec (y/n, y-default)')
        i, o, e = select.select([sys.stdin], [], [], 10)
        if i and sys.stdin.readline().strip() == 'n':
            pass
        else:
            loop_ix = saved_n_loops + 1
            _continue = True

    # eventually flush folder
    if not _continue:
        prep_folder(DMK_MODELS_FD, flush_non_empty=True)

    logger = get_pylogger(
        name=   'train_loop_V2',
        folder= DMK_MODELS_FD,
        level=  20)

    logger.info(f'train_loop_V2 starts..')
    if _continue: logger.info(f'> continuing with saved {saved_n_loops} loops')
    else:         logger.info(f'> flushing DMK_MODELS_FD: {DMK_MODELS_FD}')

    cm = ConfigManager(file_FP=CONFIG_FP, config=CONFIG_INIT, logger=logger)
    config = cm()

    if not _continue:

        logger.info(f'Building {config["n_dmk_total"]} DMKs..')
        initial_names = [f'dmk000_{ix}' for ix in range(config['n_dmk_total'])]
        families = [random.choice(config['families']) for _ in initial_names]
        build_from_names(
            names=      initial_names,
            families=   families,
            logger=     logger)

        all_results = {dn: {
            'family':   f,
            'rank':     [],
            'lifemark': '',
        } for dn,f in zip(initial_names,families)}
        all_results['n_loops'] = 0

    """
    1. manage config
    2. duplicate trainable to _old
    3. train
        split n_dmk_total into groups of n_dmk_TR_group
        train each group against n_dmk_master (in first loop - random selected)
    4. test trainable + _old, test has to
        - compare 'actual' vs '_old' and check which significantly improved or degraded
        - set actual ranking (masters, poor ones)
        test may be broken with 'separated' condition
    5. analyse results
        report
        roll back from _old those who not improved
        adjust TR / TS parameters     
    6. do PMT every N loop
    7. eventually replace poor with a new
    """
    # TODO: automatically increase TR/TS game_size by 50K when condition meet:
    #  - TR when not enough DMKs is learning
    #  - TS when not separated enough
    while True:

        logger.info(f'\n ************** starts loop {loop_ix} **************')

        ### 1. manage config changes

        if loop_ix == 2:
            if config['sep_bvalue'] > 1:
                config = cm.update(sep_bvalue=0.9)
        config = cm.load() # eventually load new config

        # it is much clear to use variables in the code than the dict
        exit_loop =             config['exit']
        pause =                 config['pause']
        families =              config['families']
        n_dmk_total =           config['n_dmk_total']
        n_dmk_master =          config['n_dmk_master']
        n_dmk_TR_group =        config['n_dmk_TR_group']
        # train
        game_size_TR =          config['game_size_TR']
        dmk_n_players_TR =      config['dmk_n_players_TR']
        # test
        game_size_TS =          config['game_size_TS']
        dmk_n_players_TS =      config['dmk_n_players_TS']
        sep_bvalue=             config['sep_bvalue']
        n_stdev =               config['n_stdev']
        # children
        remove_key =            config['remove_key']
        prob_fresh =            config['prob_fresh']
        # PMT
        n_loops_PMT =           config['n_loops_PMT']
        n_dmk_PMT =             config['n_dmk_PMT']

        if exit_loop:
            logger.info('pypoks_train exits')
            cm.update(exit=False)
            break

        ### 2. prepare dmk_ranked, duplicate them to _old

        names_saved = get_saved_dmks_names() # get all saved names
        dmk_TR = [dn for dn in names_saved if '_old' not in dn] # get names of TR
        dmk_ranked = sorted(dmk_TR, key=lambda x:all_results[x]['rank'][-1] if all_results[x]['rank'] else n_dmk_total) # sort using last rank
        dmk_old = [f'{nm}_old' for nm in dmk_ranked]

        logger.info(f'got DMKs ranked: {dmk_ranked}, duplicating them to _old..')
        copy_dmks(
            names_src=  dmk_ranked,
            names_trg=  dmk_old,
            logger=     get_child(logger, change_level=10))

        ### 3. train

        # create groups
        tr_groups = []
        dmk_ranked_copy = [] + dmk_ranked
        while dmk_ranked_copy:
            tr_groups.append(dmk_ranked_copy[:n_dmk_TR_group])
            dmk_ranked_copy = dmk_ranked_copy[n_dmk_TR_group:]

        pub_PL = {
            'publish_player_stats': False,
            'publish_pex':          False,
            'publish_update':       False,
            'publish_more':         False}
        pub_TR = {
            'publish_player_stats': False,
            'publish_pex':          False,
            'publish_update':       True,
            'publish_more':         False}
        for tg in tr_groups:
            run_GM(
                dmk_point_PLL=  [{'name':f'{dn}_old', 'motorch_point':{'device':0}, **pub_PL} for dn in dmk_ranked[:n_dmk_master]], # we take old here since GM cannot have same player name in PPL nad TRL
                dmk_point_TRL=  [{'name':nm, 'motorch_point':{'device':1}, **pub_TR} for nm in tg],
                game_size=      game_size_TR,
                dmk_n_players=  dmk_n_players_TR,
                logger=         logger)

        ### 4. test

        pub = {
            'publish_player_stats': True,
            'publish_pex':          False, # won't matter since PL does not pex
            'publish_update':       False,
            'publish_more':         False}

        sep_pairs = [(dn, f'{dn}_old') for dn in dmk_ranked]

        dmk_results = run_GM(
            dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device': n % 2}, **pub} for n,dn in enumerate(dmk_ranked + dmk_old)],
            game_size=      game_size_TS,
            dmk_n_players=  dmk_n_players_TS,
            sep_pairs=      sep_pairs,
            sep_bvalue=     sep_bvalue,
            logger=         logger)

        ### 5. analyse results

        # age report
        age = [dmk_results[dn]['age'] for dn in dmk_ranked]
        min_age, avg_age, max_age = mam(age)
        logger.info(f'age MIN:{min_age} AVG:{avg_age:.1f} MAX:{max_age}')
        two_dim(age, name=f'age_{loop_ix:03}', save_FD=AGE_FD)

        # separation
        sr = separation_report(
            dmk_results=    dmk_results,
            n_stdev=        n_stdev,
            sep_pairs=      sep_pairs)
        logger.info(f'separation ALL normalized count:   {sr["sep_nc"]:.3f}')
        logger.info(f'separation pairs normalized count: {sr["sep_pairs_nc"]:.3f}')
        # update dmk_results
        for ix,dn in enumerate(dmk_ranked):
            dmk_results[dn]['separated_old'] = sr['sep_pairs_stat'][ix]
            dmk_results[dn]['wonH_old_diff'] = dmk_results[dn]['wonH_afterIV'][-1] - dmk_results[f'{dn}_old']['wonH_afterIV'][-1]
            lifemark_upd = '+' if dmk_results[dn]['separated_old'] and dmk_results[dn]['wonH_old_diff'] > 0 else '-'
            dmk_results[dn]['lifemark'] = all_results[dn]['lifemark'] + lifemark_upd

        # log results
        res_nfo = f'DMKs train results:\n'
        dmk_ranked_TR = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in dmk_ranked]
        dmk_ranked_TR = [e[0] for e in sorted(dmk_ranked_TR, key=lambda x:x[1], reverse=True)]
        pos = 1
        for dn in dmk_ranked_TR:
            nm_aged = f'{dn}({dmk_results[dn]["age"]}){dmk_results[dn]["family"]}'
            wonH = dmk_results[dn]['wonH_afterIV'][-1]
            wonH_old_diff = dmk_results[dn]['wonH_old_diff']
            sep_nfo = ' s' if dmk_results[dn]['separated_old'] else '  '
            lifemark_nfo = f' {dmk_results[dn]["lifemark"]}'
            res_nfo += f' > {pos:>2} {nm_aged:15s} : {wonH:6.2f} :: {wonH_old_diff:6.2f}{sep_nfo}{lifemark_nfo}\n'
            pos += 1
        logger.info(res_nfo)

        # search for significantly_learned, remove their _old
        significantly_learned = [dn for dn in dmk_ranked if dmk_results[dn]['lifemark'][-1] == '+']
        logger.info(f'got significantly_learned: {significantly_learned}')

        # search for those who needs to be rolled back from _old
        roll_back_from_old = [dn for dn in dmk_ranked if dn not in significantly_learned]
        logger.info(f'rolling back: {roll_back_from_old}')
        copy_dmks(
            names_src=  [f'{nm}_old' for nm in roll_back_from_old],
            names_trg=  roll_back_from_old,
            logger=     get_child(logger, change_level=10))

        # prepare new rank (including results of rolled back) and save in all_results
        dmk_ranked = significantly_learned + [f'{dn}_old' for dn in roll_back_from_old]
        dmk_ranked = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in dmk_ranked]
        dmk_ranked = [e[0] for e in sorted(dmk_ranked, key=lambda x: x[1], reverse=True)]
        dmk_ranked = [dn if '_old' not in dn else dn[:-4] for dn in dmk_ranked]
        logger.info(f'after loop dmk_ranked: {dmk_ranked}')
        for ix,dn in enumerate(dmk_ranked):
            all_results[dn]['rank'].append(ix)
            all_results[dn]['lifemark'] = dmk_results[dn]['lifemark']
        all_results['n_loops'] = loop_ix

        ### 6. PMT evaluation

        if loop_ix % n_loops_PMT == 0:

            # copy masters with new name
            logger.info(f'Running PMT..')
            new_master = dmk_ranked[0]
            copy_dmks(
                names_src=          [new_master],
                names_trg=          [f'{new_master}_pmt{loop_ix//n_loops_PMT:03}'],
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
                    dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device': n % 2}, 'save_topdir':PMT_FD, **pub} for n,dn in enumerate(all_pmt)],
                    game_size=      game_size_TS,
                    dmk_n_players=  dmk_n_players_TS,
                    logger=         logger)

                pmt_ranked = [(dn, pmt_results[dn]['wonH_afterIV'][-1]) for dn in pmt_results]
                pmt_ranked = [e[0] for e in sorted(pmt_ranked, key=lambda x: x[1], reverse=True)]

                pmt_nfo = 'PMT results (wonH):\n'
                pos = 1
                for dn in pmt_ranked:
                    nm_aged = f'{dn}({pmt_results[dn]["age"]}){pmt_results[dn]["family"]}'
                    pmt_nfo += f' > {pos:>2} {nm_aged:25s} : {pmt_results[dn]["wonH_afterIV"][-1]:5.2f}\n'
                    pos += 1
                logger.info(pmt_nfo)

                # remove worst
                if len(all_pmt) == n_dmk_PMT:
                    dn = pmt_ranked[-1]["name"]
                    shutil.rmtree(f'{PMT_FD}/{dn}', ignore_errors=True)
                    logger.info(f'removed PMT: {dn}')

        ### 7. eventually replace poor with a new

        dmk_masters = dmk_ranked[:n_dmk_master]
        dmk_poor = dmk_ranked[n_dmk_master:]

        ranks_smooth = get_ranks()['ranks_smooth']
        rank_candidates = [dn for dn in dmk_poor if ranks_smooth[dn][-1] > n_dmk_total*0.6]
        logger.info(f'rank_candidates: {rank_candidates}')

        lifemark_candidates = []
        for dn in dmk_poor:
            lifemark_ending = all_results[dn]["lifemark"][-sum(remove_key):]
            if lifemark_ending.count('-') >= remove_key[0]:
                lifemark_candidates.append(dn)
        logger.info(f'lifemark_candidates: {lifemark_candidates}')

        remove = set(rank_candidates) & set(lifemark_candidates)
        logger.info(f'DMKs to remove: {remove}')
        if remove:

            for dn in remove:
                shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)
                shutil.rmtree(f'{DMK_MODELS_FD}/{dn}_old', ignore_errors=True)
                dmk_results.pop(dn)

            # check forced family
            families_count = {fm: 0 for fm in families}
            for dn in dmk_results:
                families_count[dmk_results[dn]['family']] += 1
            families_count = [(fm, families_count[fm]) for fm in families_count]
            families_count.sort(key=lambda x: x[1])
            min_count = n_dmk_total / len(families) / 2
            family = families_count[0][0] if families_count[0][1] < min_count else None

            for cix in range(len(remove)):

                name_child = f'dmk{loop_ix:03}_{cix}'

                # build new one
                if family is not None:
                    logger.info(f'> {name_child} forced from family {family}')
                    build_single_foldmk(
                        name=   name_child,
                        family= family,
                        logger= get_child(logger, change_level=10))

                # GX from parents
                else:
                    pa = random.choice(dmk_masters)
                    family = dmk_results[pa]['family']
                    pb = random.choice([dn for dn in dmk_masters if dmk_results[dn]['family'] == family])

                    ckpt_fresh = random.random() < prob_fresh
                    ckpt_fresh_info = ' (fresh ckpt)' if ckpt_fresh else ''
                    logger.info(f'> {name_child} = {pa} + {pb}{ckpt_fresh_info}')
                    FolDMK.gx_saved(
                        name_parent_main=   pa,
                        name_parent_scnd=   pb,
                        name_child=         name_child,
                        do_gx_ckpt=         not ckpt_fresh,
                        logger=             get_child(logger, change_level=10))

                all_results[name_child] = {'family':family, 'rank': [], 'lifemark':''}
                family = None # reset family for next child loop

        w_json(all_results, RESULTS_FP)

        if pause: input("press Enter to continue..")

        loop_ix += 1