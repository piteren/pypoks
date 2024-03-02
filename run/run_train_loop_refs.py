from pypaq.lipytools.files import w_json
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.printout import stamp
from pypaq.pms.config_manager import ConfigManager
from pypaq.lipytools.files import w_pickle, r_pickle
import random
import shutil

from envy import DMK_MODELS_FD, TR_CONFIG_FP, TR_RESULTS_FP
from run.functions import check_continuation, run_PTR_game, build_single_foldmk, copy_dmks, dmk_name, get_saved_dmks_names
from run.after_run.reports import results_report, nice_hpms_report
from pologic.game_config import GameConfig
from podecide.dmk import FolDMK
from podecide.dmk_motorch import DMK_MOTorch

LOOP_CONFIG = {
        # general
    'exit_after':               None,       # exits loop for given int
    'pause':                    False,      # pauses loop after test till Enter pressed
    'families':                 'a',        # active families (forced to be present)
    'n_dmk':                    20,         # number of DMKs
    'n_dmk_refs':               0,          # number of refs DMKs
    'n_gpu':                    2,
    'n_tables':                 1000,       # target number of tables (for any game: TR, PMT)
    'game_size_TR':             100000,
        # remove / new DMKs
    'do_removal':               False,
    'remove_n_loops':           5,          # perform removal attempt every N loops
    'remove_tail':              4,          # number of DMKs taken into removal pipeline
    'remove_n_dmk':             1,          # target number of DMKs to remove
    'remove_old':               20,         # remove DMKs that are at least such old
    'prob_fresh_dmk':           0.5,        # probability of 100% fresh DMK (otherwise GX)+
    'prob_fresh_ckpt':          0.5,        # probability of fresh checkpoint (GX of POINT only, without GX of ckpt)
        # Periodical Masters Test
    'n_loops_PMT':              10,         # do PMT every N loops
    'ndmk_PMT':                 10,         # max number of DMKs (masters) in PMT
    'game_size_PMT':            300000,
}

PUB_TR =   {'publish_player_stats':True,  'publishFWD':True,  'publishUPD':True}
PUB_NONE = {'publish_player_stats':False, 'publishFWD':False, 'publishUPD':False}

PMT_FD = f'{DMK_MODELS_FD}/_pmt'


def run(game_config_name:str, use_saved_dmks=True, del_removed_dmks=False):
    """ Trains DMKs in a loop.
    - refs are selected every loop from the top of DMKs
    - remove policy: every N loops remove some old from the bottom """

    cc = check_continuation()
    tr_results = cc['training_results']

    tl_name = f'run_train_loop_refs_{stamp()}'
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

    if cc['continuation']:

        game_config = GameConfig.from_name(folder=DMK_MODELS_FD)
        logger.info(f'> game config name: {game_config.name}')

        saved_n_loops = int(tr_results['loop_ix'])
        logger.info(f'> continuing with saved {saved_n_loops} loops')
        loop_ix = saved_n_loops + 1

        dmk_ranked = tr_results['dmk_ranked']

        points_data = r_pickle(f'{DMK_MODELS_FD}/points.data')

    else:

        game_config = GameConfig.from_name(name=game_config_name, copy_to=DMK_MODELS_FD)
        logger.info(f'> game config name: {game_config.name}')

        dmk_ranked = []
        if use_saved_dmks:
            dmks_saved = get_saved_dmks_names(DMK_MODELS_FD)
            if dmks_saved:
                families = [FolDMK.load_point(name=dn)['family'] for dn in dmks_saved]
                dmk_ranked = [dmk_name(
                    loop_ix=    0,
                    family=     family,
                    counter=    ix) for ix,family in enumerate(families)]
                logger.info(f'> got saved DMKs: {dmks_saved}, renaming them to {dmk_ranked}')
                copy_dmks(
                    names_src=  dmks_saved,
                    names_trg=  dmk_ranked,
                    logger=     sub_logger)
                for dn in dmks_saved:
                    shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        loop_ix = 1

        points_data = {
            'points_dmk':       {dn: FolDMK.load_point(name=dn) for dn in dmk_ranked},
            'points_motorch':   {dn: DMK_MOTorch.load_point(name=dn) for dn in dmk_ranked},
            'scores':           {}}

    while True:

        if cm.exit_after == loop_ix - 1:
            logger.info('train loop exits')
            break

        logger.info(f'\n ************** starts loop #{loop_ix} **************')

        # eventually remove some DMKs, if there is too much
        while len(dmk_ranked) > cm.n_dmk:
            dn = dmk_ranked.pop()
            logger.info(f'removing {dn} ({len(dmk_ranked)}/{cm.n_dmk})')
            shutil.rmtree(f'{PMT_FD}/{dn}', ignore_errors=True)

        # eventually build new
        if len(dmk_ranked) < cm.n_dmk:

            logger.info(f'building new DMKs:')

            # look for forced families
            dmk_families = {dn: FolDMK.load_point(name=dn)['family'] for dn in dmk_ranked}
            families_present = ''.join(list(dmk_families.values()))
            families_count = {fm: families_present.count(fm) for fm in cm.families}
            families_count = [(fm, families_count[fm]) for fm in families_count]
            families_forced = [fc[0] for fc in families_count if fc[1] < 1]
            if families_forced: logger.info(f'families forced: {families_forced}')

            cix = 0
            dmk_new = []
            while len(dmk_ranked + dmk_new) < cm.n_dmk:

                # build new one from forced
                if families_forced:
                    family = families_forced.pop()
                    name_child = dmk_name(loop_ix=loop_ix, family=family, counter=cix)
                    logger.info(f'> {name_child} <- fresh, forced from family {family}')
                    build_single_foldmk(
                        game_config=    game_config,
                        name=           name_child,
                        family=         family,
                        logger=         get_child(logger, change_level=10))

                else:

                    pa = random.choices(dmk_ranked, weights=list(reversed(range(len(dmk_ranked)))))[0] if len(dmk_ranked) > 1 else None

                    # 100% fresh DMK from selected family
                    if random.random() < cm.prob_fresh_dmk or not pa or dmk_families[pa] not in cm.families:
                        family = random.choice(cm.families)
                        name_child = dmk_name(loop_ix=loop_ix, family=family, counter=cix)
                        logger.info(f'> {name_child} <- 100% fresh')
                        build_single_foldmk(
                            game_config=    game_config,
                            name=           name_child,
                            family=         family,
                            logger=         get_child(logger, change_level=10))

                    # GX
                    else:
                        family = dmk_families[pa]
                        name_child = dmk_name(loop_ix=loop_ix, family=family, counter=cix)
                        other_from_family = [dn for dn in dmk_ranked if dmk_families[dn] == family]
                        if len(other_from_family) > 1:
                            other_from_family.remove(pa)
                        if len(other_from_family) > 1:
                            pb = random.choices(other_from_family, weights=list(reversed(range(len(other_from_family)))))[0]
                        else:
                            pb = other_from_family[0]

                        ckpt_fresh = random.random() < cm.prob_fresh_ckpt
                        ckpt_fresh_info = ' (fresh ckpt)' if ckpt_fresh else ''
                        logger.info(f'> {name_child} = {pa} + {pb}{ckpt_fresh_info}')
                        FolDMK.gx_saved(
                            name_parentA=   pa,
                            name_parentB=   pb,
                            name_child=     name_child,
                            do_gx_ckpt=     not ckpt_fresh,
                            logger=         get_child(logger))

                points_data['points_dmk'][name_child] = FolDMK.load_point(name=name_child)
                points_data['points_motorch'][name_child] = DMK_MOTorch.load_point(name=name_child)
                dmk_new.append(name_child)
                cix += 1

            dmk_ranked += dmk_new

        ### train

        # prepare refs
        dmk_ranked_to_ref = dmk_ranked[:cm.n_dmk_refs]
        dmk_refs = [f'{dn}R' for dn in dmk_ranked_to_ref]
        copy_dmks(
            names_src=  dmk_ranked_to_ref,
            names_trg=  dmk_refs,
            logger=     sub_logger)

        dmk_point_refL = [
            {'name':dn, 'motorch_point':{'device':n % cm.n_gpu if cm.n_gpu else None}, **PUB_NONE}
            for n,dn in enumerate(dmk_refs)]
        dmk_point_TRL = [
            {'name':dn, 'motorch_point':{'device':n % cm.n_gpu if cm.n_gpu else None}, **PUB_TR}
            for n,dn in enumerate(dmk_ranked)]

        rgd = run_PTR_game(
            game_config=    game_config,
            name=           f'GM_{loop_ix:003}',
            gm_loop=        loop_ix,
            dmk_point_refL= dmk_point_refL,
            dmk_point_TRL=  dmk_point_TRL,
            game_size=      cm.game_size_TR,
            n_tables=       cm.n_tables,
            logger=         logger)
        dmk_results = rgd['dmk_results']
        dmk_ranked = sorted(dmk_ranked, key=lambda x: dmk_results[x]['last_wonH_afterIV'], reverse=True)

        logger.info(f'train results:\n{results_report(dmk_results)}')
        # logger.info(f'DMKs POINTS:\n{nice_hpms_report(points_data["points_dmk"], points_data["points_motorch"], dmk_ranked)}')

        # delete refs
        for dn in dmk_refs:
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        # eventually remove some
        if cm.do_removal and loop_ix % cm.remove_n_loops == 0:

            remove_candids = dmk_ranked[-cm.remove_tail:]       # take the tail
            remove_candids = list(reversed(remove_candids))     # first from the worst
            remove_candids.sort(key= lambda x: int(x[3:6]))     # next from the oldest
            remove_candids = remove_candids[:cm.remove_n_dmk]   # finally trim
            remove_candids = [dn for dn in remove_candids if (loop_ix - int(dn[3:6])) >= cm.remove_old]

            if remove_candids:
                logger.info(f'removing {len(remove_candids)} DMKs: {", ".join(remove_candids)}')
            for dn in remove_candids:
                dmk_ranked.remove(dn)
                if del_removed_dmks:
                    shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        # add scores
        for ix,dn in enumerate(dmk_ranked):
            if dn not in points_data['scores']:
                points_data['scores'][dn] = {}
            points_data['scores'][dn][f'rank_{loop_ix:003}'] = (cm.n_dmk - ix) / cm.n_dmk

        w_pickle(points_data, f'{DMK_MODELS_FD}/points.data')

        tr_results = {'loop_ix':loop_ix, 'dmk_ranked':dmk_ranked}
        w_json(tr_results, TR_RESULTS_FP)

        # PMT evaluation
        if loop_ix % cm.n_loops_PMT == 0:
            new_master = dmk_ranked[0]
            pmt_name = f'{new_master}_pmt{loop_ix:03}'
            copy_dmks(
                names_src=          [new_master],
                names_trg=          [pmt_name],
                save_topdir_trg=    PMT_FD,
                logger=             sub_logger)
            logger.info(f'copied {new_master} to PMT -> {pmt_name}')

            all_pmt = get_saved_dmks_names(PMT_FD)
            if len(all_pmt) > 2: # run test for at least 3 DMKs
                logger.info(f'PMT starts..')

                rgd = run_PTR_game(
                    game_config=    game_config,
                    dmk_point_PLL=  [{'name':dn, 'motorch_point':{'device':i%2}, 'save_topdir':PMT_FD, **PUB_NONE} for i,dn in enumerate(all_pmt)],
                    game_size=      cm.game_size_PMT,
                    n_tables=       cm.n_tables,
                    sep_all_break=  True,
                    sep_n_stddev=   2.0,
                    logger=         logger,
                    publish=        False)
                pmt_results = rgd['dmk_results']

                logger.info(f'PMT results:\n{results_report(pmt_results)}')

                # remove worst
                if len(all_pmt) == cm.ndmk_PMT:
                    dmk_rw = [(dn, pmt_results[dn]['last_wonH_afterIV']) for dn in pmt_results]
                    pmt_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x: x[1], reverse=True)]
                    dn = pmt_ranked[-1]
                    shutil.rmtree(f'{PMT_FD}/{dn}', ignore_errors=True)
                    logger.info(f'removed PMT: {dn}')

        loop_ix += 1


if __name__ == "__main__":
    run('3players_2bets')