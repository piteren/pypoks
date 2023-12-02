from pypaq.lipytools.files import w_json
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.printout import stamp
from pypaq.pms.config_manager import ConfigManager
from pypaq.lipytools.files import w_pickle, r_pickle
import random
import shutil

from envy import DMK_MODELS_FD, TR_CONFIG_FP, TR_RESULTS_FP, load_game_config, get_game_config_name
from run.functions import check_continuation, run_GM, build_from_names, build_single_foldmk, results_report, copy_dmks, dmk_name
from podecide.dmk import FolDMK
from podecide.dmk_motorch import DMK_MOTorch

TR_CONFIG = {
        # general
    'exit':                     False,      # exits loop (after train)
    'pause':                    False,      # pauses loop after test till Enter pressed
    'families':                 'ab',       # active families (forced to be present)
    'n_dmk':                    20,         # number of DMKs
    'n_dmk_refs':               6,          # number of refs DMKs
    'game_size_TR':             100000,
    'n_tables':                 2000,       # target number of tables
    'n_gpu':                    2,
        # replace / new
    'remove_n_loops':           3,          # perform remove attempt every N loops
    'removeF':                  0.2,        # factor of bottom DMKs considered to be removed
    'remove_old':               4,          # remove DMKs that are at least such old
    'prob_fresh_dmk':           0.5,        # probability of 100% fresh DMK (otherwise GX)
    'prob_fresh_ckpt':          0.5,        # probability of fresh checkpoint (child from GX of point only, without GX of ckpt)
}

PUB_TR =  {'publish_player_stats':True,  'publish_pex':True,  'publishFWD':True,  'publishUPD':True}
PUB_REF = {'publish_player_stats':False, 'publish_pex':False, 'publishFWD':False, 'publishUPD':False}


def run(game_config_name: str):
    """ Trains DMKs in a loop.
    Configuration is set to utilize power of 2 GPUs (11GB RAM) strong CPU and ~200GB RAM.

    - there are no old / new and no PL game
    - refs are selected every loop from the top of DMKs
    - baseline remove policy - every N loops remove some bottom DMKs """

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
    logger.info(f'train_loop {tl_name} starts..')
    sub_logger = get_child(logger)

    cm = ConfigManager(file_FP=TR_CONFIG_FP, config_init=TR_CONFIG, logger=logger)

    if cc['continuation']:

        game_config_name = get_game_config_name(DMK_MODELS_FD)
        game_config = load_game_config(game_config_name)

        saved_n_loops = int(tr_results['loop_ix'])
        logger.info(f'> continuing with saved {saved_n_loops} loops of game config {game_config_name}')
        loop_ix = saved_n_loops + 1

        dmk_ranked = tr_results['dmk_ranked']

        points_data = r_pickle(f'{DMK_MODELS_FD}/points.data')

    else:

        game_config = load_game_config(name=game_config_name, copy_to=DMK_MODELS_FD)

        loop_ix = 1

        families = (cm.families * cm.n_dmk)[:cm.n_dmk]
        dmk_ranked = [
            dmk_name(
                loop_ix=    loop_ix,
                family=     family,
                counter=    ix) for ix,family in enumerate(families)]
        build_from_names(
            game_config=    game_config,
            names=          dmk_ranked,
            families=       families,
            oversave=       False,
            logger=         logger)
        random.shuffle(dmk_ranked)  # shuffle initial names to break "patterns" on GPU etc

        points_data = {
            'points_dmk':       {dn: FolDMK.load_point(name=dn) for dn in dmk_ranked},
            'points_motorch':   {dn: DMK_MOTorch.load_point(name=dn) for dn in dmk_ranked},
            'scores':           {}}

    while True:

        if cm.exit:
            logger.info('train loop exits')
            cm.exit = False
            break

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

                    pa = random.choices(dmk_ranked, weights=list(reversed(range(len(dmk_ranked)))))[0]

                    # 100% fresh DMK from selected family
                    if random.random() < cm.prob_fresh_dmk:
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
            {'name':dn, 'motorch_point':{'device':n % cm.n_gpu if cm.n_gpu else None}, **PUB_REF}
            for n,dn in enumerate(dmk_refs)]
        dmk_point_TRL = [
            {'name':dn, 'motorch_point':{'device':n % cm.n_gpu if cm.n_gpu else None}, **PUB_TR}
            for n,dn in enumerate(dmk_ranked)]

        rgd = run_GM(
            name=           f'GM_{loop_ix:003}',
            game_config=    game_config,
            dmk_point_refL= dmk_point_refL,
            dmk_point_TRL=  dmk_point_TRL,
            game_size=      cm.game_size_TR,
            dmk_n_players=  cm.n_tables // len(dmk_ranked),
            logger=         logger)
        dmk_results = rgd['dmk_results']
        logger.info(f'train results:\n{results_report(dmk_results)}')

        # delete refs
        for dn in dmk_refs:
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        dmk_ranked = sorted(dmk_ranked, key=lambda x:dmk_results[x]['last_wonH_afterIV'], reverse=True)

        # eventually remove some
        if loop_ix % cm.remove_n_loops == 0:

            n_remove = int(len(dmk_ranked) * cm.removeF)
            remove_candids = dmk_ranked[-n_remove:]
            remove_candids = [dn for dn in remove_candids if (loop_ix - int(dn[3:6])) >= cm.remove_old]

            if remove_candids:
                logger.info(f'removing {len(remove_candids)} DMKs: {", ".join(remove_candids)}')
            for dn in remove_candids:
                dmk_ranked.remove(dn)
                shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

        # add scores
        for ix,dn in enumerate(dmk_ranked):
            if dn not in points_data['scores']:
                points_data['scores'][dn] = {}
            points_data['scores'][dn][f'rank_{loop_ix}'] = (cm.n_dmk - ix) / cm.n_dmk

        w_pickle(points_data, f'{DMK_MODELS_FD}/points.data')

        tr_results = {
            'loop_ix':      loop_ix,
            'lifemarks':    {
                dn: dmk_results[dn]['lifemark'] if 'lifemark' in dmk_results[dn] else ''
                for dn in dmk_results},
            'dmk_ranked':   dmk_ranked}
        w_json(tr_results, TR_RESULTS_FP)

        loop_ix += 1


if __name__ == "__main__":
    run('2players_2bets')