from pypaq.lipytools.files import prep_folder, r_json, w_json
from pypaq.lipytools.plots import two_dim
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.stats import mam
from pypaq.lipytools.printout import stamp
from pypaq.pms.config_manager import ConfigManager
from torchness.tbwr import TBwr
import random
import select
import shutil
import sys

from pypoks_envy import DMK_MODELS_FD, AGE_FD, PMT_FD, CONFIG_FP, RESULTS_FP
from podecide.dmk import FolDMK
from podecide.games_manager import separation_report
from run.pretrain_for_loop import pretrain
from run.functions import get_saved_dmks_names, run_GM, copy_dmks, build_single_foldmk
from run.after_run.ranks import get_ranks

CONFIG_INIT = {
        # general
    'pretrain_game_size':       100000,
    'exit':                     False,      # exits loop (after train)
    'pause':                    False,      # pauses loop after test till Enter pressed
    'families':                 'abcd',     # active families, at least one representative should be present
    'n_dmk_total':              12,         # total number of trainable DMKs (population size)  // 12x cN12 playing / GPU 6x cN12 training
    'n_dmk_master':             6,          # number of 'masters' DMKs (trainable are trained against them)
    'n_dmk_TR_group':           6,          # DMKs are trained with masters in groups of that size (group is build of n_dmk_TR_group + n_dmk_master)
    'game_size_upd':            100000,     # how much increase game size of TR or TS when needed
    'max_notsep':               0.3,        # max factor of DMKs not separated, if higher TS game is increased
        # train
    'game_size_TR':             100000,
    'dmk_n_players_TR':         150,
        # test
    'game_size_TS':             100000,
    'dmk_n_players_TS':         150,        # number of players per DMK while TS // 300 is ok, but loads a lot of RAM
    'sep_pairs_factor':         1.0,        # pairs separation break value
    'sep_n_stdev':              2.0,
        # replace / new
    'rank_mavg_factor':         0.3,        # mavg_factor of rank_smooth calculation
    'safe_rank':                0.5,        # <0.0;1.0> factor fo rank_smooth that is safe (not considered to be replaced, 0.6 means that top 60% of rank is safe)
    'remove_key':               [2,0],      # -> [3,0]  [A,B] remove DMK if in last A+B life marks there are A -|
    'prob_fresh':               0.7,        # probability of fresh checkpoint (child from GX of point only, without GX of ckpt)
        # PMT (Periodical Masters Test)
    'n_loops_PMT':              10,         # do PMT every N loops
    'n_dmk_PMT':                20}         # max number of DMKs (masters) in PMT

"""
    # below are descriptions of structures used
    
    all_results = {
        'loops': {
            1:  ['dmka00_00','dmkb00_01',..],               <- ranking after 1st loop
            2:  ['dmka00_01','dmkb00_04',..],               <- ranking after 2nd loop
            ..},
        'lifemarks': {
            'dmka00_00': '++',
            'dmka00_01': '|-',
            ..}
            
    dmk_ranked = ['dmka00_00','dmkb00_01',..]               <- ranking after last loop, updated in the loop after train
    dmk_old = ['dmka00_00_old','dmkb00_01_old',..]          <- just all _old names
    
    dmk_results = {
        'dmka00_00': {                                      # _old do not have all keys and are removed later
            'wonH_IV':          [float,..]                  <- wonH of interval
            'wonH_afterIV':     [float,..]                  <- wonH after interval
            'family':           dmk.family,
            'trainable':        dmk.trainable,
            'age':              dmk.age
            'separated_old':    0 or 1
            'wonH_old_diff':    float
            'lifemark':         '++'
        },
        ..}
    
"""



if __name__ == "__main__":

    # check for continuation
    loop_ix = 1
    _continue = False
    saved_n_loops = None
    all_results = r_json(RESULTS_FP)
    if all_results:
        saved_n_loops = len(all_results['loops'])
        print(f'Do you want to continue with saved (in {DMK_MODELS_FD}) {saved_n_loops} loops? ..waiting 10 sec (y/n, y-default)')
        i, o, e = select.select([sys.stdin], [], [], 10)
        if i and sys.stdin.readline().strip() == 'n':
            pass
        else:
            loop_ix = saved_n_loops + 1
            _continue = True

    # eventually flush folder
    if not _continue:
        prep_folder(
            path=               DMK_MODELS_FD,
            #flush_non_empty=    True
        )

    tl_name = f'train_loop_V2_{stamp()}'

    logger = get_pylogger(
        name=       tl_name,
        add_stamp=  False,
        folder=     DMK_MODELS_FD,
        level=      20)

    logger.info(f'train_loop_V2 starts..')
    if _continue: logger.info(f'> continuing with saved {saved_n_loops} loops')
    else:         logger.info(f'> flushing DMK_MODELS_FD: {DMK_MODELS_FD}')

    cm = ConfigManager(file_FP=CONFIG_FP, config_init=CONFIG_INIT, logger=logger)

    if not _continue:

        logger.info(f'Building {cm.n_dmk_total} DMKs..')

        """
        pretrain(
            n_fam_dmk=      cm.n_dmk_total // len(cm.families),
            families=       cm.families,
            game_size=      cm.pretrain_game_size,
            dmk_n_players=  cm.dmk_n_players_TR,
            logger=         logger)
        """

        all_results = {'loops':{}, 'lifemarks':{}}

    # TODO: prep folder?
    tbwr = TBwr(logdir=f'{DMK_MODELS_FD}/{tl_name}')

    """
    1. duplicate trainable to _old
    2. train
        split n_dmk_total into groups of n_dmk_TR_group
        train each group against n_dmk_master (in first loop - random selected)
    3. test trainable + _old, test has to
        - compare 'actual' vs '_old' and check which significantly improved or degraded
        - set actual ranking (masters, poor ones)
        test may be broken with 'separated' condition
    4. analyse results
        report
        roll back from _old those who not improved
        adjust TR / TS parameters     
    5. eventually replace poor with a new / GX
    6. do PMT every N loop
    """
    while True:

        logger.info(f'\n ************** starts loop {loop_ix} **************')

        config = cm.load() # eventually load new config from file

        if cm.exit:
            logger.info('train_loop_V2 exits')
            cm.exit = False
            break

        ### 1. prepare dmk_ranked, duplicate them to _old

        dmk_ranked = all_results['loops'][loop_ix-1] if loop_ix>1 else get_saved_dmks_names()
        logger.info(f'got DMKs ranked: {dmk_ranked}, duplicating them to _old..')
        dmk_old = [f'{nm}_old' for nm in dmk_ranked]
        copy_dmks(
            names_src=  dmk_ranked,
            names_trg=  dmk_old,
            logger=     get_child(logger, change_level=10))

        ### 2. train

        # create groups
        tr_groups = []
        dmk_ranked_copy = [] + dmk_ranked
        while dmk_ranked_copy:
            tr_groups.append(dmk_ranked_copy[:cm.n_dmk_TR_group])
            dmk_ranked_copy = dmk_ranked_copy[cm.n_dmk_TR_group:]

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
                dmk_point_PLL=  [{'name':f'{dn}_old', 'motorch_point':{'device':0}, **pub_PL} for dn in dmk_ranked[:cm.n_dmk_master]], # we take old here since GM cannot have same player name in PPL nad TRL
                dmk_point_TRL=  [{'name':nm, 'motorch_point':{'device':1}, **pub_TR} for nm in tg],
                game_size=      cm.game_size_TR,
                dmk_n_players=  cm.dmk_n_players_TR,
                logger=         logger)

        ### 3. test

        pub = {
            'publish_player_stats': True,
            'publish_pex':          False, # won't matter since PL does not pex
            'publish_update':       False,
            'publish_more':         False}

        sep_pairs = [(dn, f'{dn}_old') for dn in dmk_ranked]

        dmks_PLL = dmk_ranked + dmk_old
        random.shuffle(dmks_PLL) # shuffle to randomly distribute among GPUs
        dmk_results = run_GM(
            dmk_point_PLL=      [{'name':dn, 'motorch_point':{'device':n%2}, **pub} for n,dn in enumerate(dmks_PLL)],
            game_size=          cm.game_size_TS,
            dmk_n_players=      cm.dmk_n_players_TS,
            sep_pairs=          sep_pairs,
            sep_pairs_factor=   cm.sep_pairs_factor,
            sep_n_stdev=        cm.sep_n_stdev,
            logger=             logger)

        ### 4. analyse results

        # separation
        sr = separation_report(
            dmk_results=    dmk_results,
            n_stdev=        cm.sep_n_stdev,
            sep_pairs=      sep_pairs)
        logger.info(f'separation ALL normalized count:   {sr["sep_nc"]:.3f}')
        logger.info(f'separation pairs normalized count: {sr["sep_pairs_nc"]:.3f}')
        # update dmk_results
        session_lifemarks = ''
        for ix,dn in enumerate(dmk_ranked):
            dmk_results[dn]['separated_old'] = sr['sep_pairs_stat'][ix]
            dmk_results[dn]['wonH_old_diff'] = dmk_results[dn]['wonH_afterIV'][-1] - dmk_results[f'{dn}_old']['wonH_afterIV'][-1]
            lifemark_upd = '|'
            if dmk_results[dn]['separated_old']:
                lifemark_upd = '+' if dmk_results[dn]['wonH_old_diff'] > 0 else '-'
            lifemark_prev = all_results['lifemarks'][dn] if dn in all_results['lifemarks'] else ''
            dmk_results[dn]['lifemark'] = lifemark_prev + lifemark_upd
            session_lifemarks += lifemark_upd

        # eventually increase game_size
        if loop_ix in [10,20]:
            cm.game_size_TR = cm.game_size_TR + cm.game_size_upd
        count_notsep = session_lifemarks.count('|')
        if count_notsep > cm.n_dmk_total * cm.max_notsep:
            cm.game_size_TS = cm.game_size_TS + cm.game_size_upd

        # log results
        res_nfo = f'DMKs train results:\n'
        dmk_ranked = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in dmk_ranked]
        dmk_ranked = [e[0] for e in sorted(dmk_ranked, key=lambda x:x[1], reverse=True)]
        all_results['loops'][loop_ix] = dmk_ranked  # update with new dmk_ranked
        ranks_smooth = get_ranks(all_results=all_results, mavg_factor=cm.rank_mavg_factor)['ranks_smooth']
        pos = 0
        for dn in dmk_ranked:
            rank = f'{ranks_smooth[dn][-1] / cm.n_dmk_total:4.2f}' if ranks_smooth[dn] else '----'
            nm_aged = f'{dn}({dmk_results[dn]["age"]})'
            wonH = dmk_results[dn]['wonH_afterIV'][-1]
            wonH_old_diff = dmk_results[dn]['wonH_old_diff']
            sep_nfo = ' s' if dmk_results[dn]['separated_old'] else '  '
            lifemark_nfo = f' {dmk_results[dn]["lifemark"]}'
            res_nfo += f' > {pos:>2}({rank}) {nm_aged:15s} : {wonH:6.2f} :: {wonH_old_diff:6.2f}{sep_nfo}{lifemark_nfo}\n'
            pos += 1
        logger.info(res_nfo)

        # log global_stats to TB
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
                    tag=    f'global_stats/{dsn}_{k}',
                    step=   loop_ix)

        # search for significantly_learned, remove their _old
        significantly_learned = [dn for dn in dmk_ranked if dmk_results[dn]['lifemark'][-1] == '+']
        logger.info(f'got significantly_learned: {significantly_learned}')

        # roll back some from _old
        roll_back_from_old = [dn for dn in dmk_ranked if dn not in significantly_learned]
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
        for dn in dmk_old:
            dmk_results.pop(dn)

        # log again results after roll back
        res_nfo = f'DMKs train results after roll back:\n'
        dmk_ranked = [(dn, dmk_results[dn]['wonH_afterIV'][-1]) for dn in dmk_ranked]
        dmk_ranked = [e[0] for e in sorted(dmk_ranked, key=lambda x: x[1], reverse=True)]
        all_results['loops'][loop_ix] = dmk_ranked  # update with new dmk_ranked
        ranks_smooth = get_ranks(all_results=all_results, mavg_factor=cm.rank_mavg_factor)['ranks_smooth']
        pos = 0
        for dn in dmk_ranked:
            rank = f'{ranks_smooth[dn][-1] / cm.n_dmk_total:4.2f}' if ranks_smooth[dn] else '----'
            nm_aged = f'{dn}({dmk_results[dn]["age"]})'
            wonH = dmk_results[dn]['wonH_afterIV'][-1]
            res_nfo += f' > {pos:>2}({rank}) {nm_aged:15s} : {wonH:6.2f}\n'
            pos += 1
        logger.info(res_nfo)

        all_results['loops'][loop_ix] = dmk_ranked
        for dn in dmk_ranked:
            all_results['lifemarks'][dn] = dmk_results[dn]['lifemark']

        ### 5. eventually replace poor with a new

        dmk_masters = dmk_ranked[:cm.n_dmk_master]
        dmk_poor = dmk_ranked[cm.n_dmk_master:]

        ranks_smooth = get_ranks(all_results=all_results, mavg_factor=cm.rank_mavg_factor)['ranks_smooth']
        rank_candidates = [dn for dn in dmk_poor if ranks_smooth[dn][-1] > cm.n_dmk_total * cm.safe_rank]
        logger.info(f'rank_candidates: {rank_candidates}')

        lifemark_candidates = []
        for dn in dmk_poor:
            lifemark_ending = all_results['lifemarks'][dn][-sum(cm.remove_key):]
            if lifemark_ending.count('-') + lifemark_ending.count('|') >= cm.remove_key[0]:
                lifemark_candidates.append(dn)
        logger.info(f'lifemark_candidates: {lifemark_candidates}')

        remove = set(rank_candidates) & set(lifemark_candidates)
        logger.info(f'DMKs to remove: {remove}')

        if remove:

            for dn in remove:
                shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)
                shutil.rmtree(f'{DMK_MODELS_FD}/{dn}_old', ignore_errors=True)
                dmk_results.pop(dn)
                dmk_ranked.remove(dn)

            # look for forced families
            families_count = {fm: 0 for fm in cm.families}
            for dn in dmk_results:
                families_count[dmk_results[dn]['family']] += 1
            families_count = [(fm, families_count[fm]) for fm in families_count]
            families_forced = [fc[0] for fc in families_count if fc[1] < 1]
            if families_forced: logger.info(f'families forced: {families_forced}')

            for cix in range(len(remove)):

                # build new one from forced
                if families_forced:
                    family = families_forced.pop()
                    name_child = f'dmk{loop_ix:02}{family}{cix:02}'
                    logger.info(f'> {name_child} forced from family {family}')
                    build_single_foldmk(
                        name=   name_child,
                        family= family,
                        logger= get_child(logger, change_level=10))

                # GX from parents
                else:
                    pa = random.choice(dmk_masters)
                    family = dmk_results[pa]['family']
                    name_child = f'dmk{loop_ix:02}{family}{cix:02}'
                    pb = random.choice([dn for dn in dmk_masters if dmk_results[dn]['family'] == family])

                    ckpt_fresh = random.random() < cm.prob_fresh
                    ckpt_fresh_info = ' (fresh ckpt)' if ckpt_fresh else ''
                    logger.info(f'> {name_child} = {pa} + {pb}{ckpt_fresh_info}')
                    FolDMK.gx_saved(
                        name_parent_main=           pa,
                        name_parent_scnd=           pb,
                        name_child=                 name_child,
                        save_topdir_parent_main=    DMK_MODELS_FD,
                        do_gx_ckpt=                 not ckpt_fresh,
                        logger=                     get_child(logger, change_level=10))

                dmk_ranked.append(name_child)

            all_results['loops'][loop_ix] = dmk_ranked # update with new dmk_ranked

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
                    game_size=      cm.game_size_TS,
                    dmk_n_players=  cm.dmk_n_players_TS,
                    sep_all_break=  True,
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
                if len(all_pmt) == cm.n_dmk_PMT:
                    dn = pmt_ranked[-1]["name"]
                    shutil.rmtree(f'{PMT_FD}/{dn}', ignore_errors=True)
                    logger.info(f'removed PMT: {dn}')

        if cm.pause: input("press Enter to continue..")

        loop_ix += 1