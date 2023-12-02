from pypaq.lipytools.files import w_json
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.printout import stamp
from pypaq.pms.config_manager import ConfigManager
from pypaq.lipytools.files import w_pickle
import random
import shutil
from typing import Dict, List

from envy import DMK_MODELS_FD, TR_CONFIG_FP, TR_RESULTS_FP, load_game_config
from run.functions import run_GM, build_from_names, build_single_foldmk, results_report, dmk_name
from podecide.dmk import FolDMK
from podecide.dmk_motorch import DMK_MOTorch

TR_CONFIG = {
        # general
    'exit':                     False,      # exits loop (after train)
    'pause':                    False,      # pauses loop after test till Enter pressed
    'families':                 'ab',       # active families (forced to be present)
    'n_dmk':                    20,         # number of DMKs
    'game_size_TR':             200000,
    'n_tables':                 2000,       # target number of tables
    'n_gpu':                    2,
        # replace / new
    'remove_key':               [2,0],      # [A,B] (remove DMK if in last A+B life marks there are A - and last is not +
    'n_remove':                 4,          # number of candidates from the bottom to remove
    'prob_fresh_dmk':           0.5,        # probability of 100% fresh DMK (otherwise GX)
    'prob_fresh_ckpt':          0.5,        # probability of fresh checkpoint (child from GX of point only, without GX of ckpt)
}

PUB_TR =  {'publish_player_stats':True, 'publish_pex':True, 'publishFWD':True, 'publishUPD':True}


def run(game_config_name: str):
    """ Trains DMKs in a loop.

    In this scrip lifemark has different meaning than in run_train_loop_refs.py,
    here + means DMK won in this loop more than in previous loop.

    - baseline remove policy """

    tl_name = f'run_train_loop_{stamp()}'
    logger = get_pylogger(
        name=       tl_name,
        add_stamp=  False,
        folder=     DMK_MODELS_FD,
        level=      20,
        #flat_child= True,
    )
    logger.info(f'train loop {tl_name} for game config: {game_config_name} starts..')
    sub_logger = get_child(logger)

    game_config = load_game_config(name=game_config_name, copy_to=DMK_MODELS_FD)

    cm = ConfigManager(file_FP=TR_CONFIG_FP, config_init=TR_CONFIG, logger=logger)

    loop_ix = 1
    families = (cm.families * cm.n_dmk)[:cm.n_dmk]
    dmk_names = [
        dmk_name(
            loop_ix=    loop_ix,
            family=     family,
            counter=    ix)
        for ix,family in enumerate(families)]

    build_from_names(
        game_config=    game_config,
        names=          dmk_names,
        families=       families,
        oversave=       False,
        logger=         logger)

    points_data = {
        'points_dmk':       {dn: FolDMK.load_point(name=dn) for dn in dmk_names},
        'points_motorch':   {dn: DMK_MOTorch.load_point(name=dn) for dn in dmk_names},
        'scores':           {}}

    dmk_wonH: Dict[str,List[float]] = {} # DMK loops wonH
    dmk_lifemarks: Dict[str,str] = {}
    dmk_ranked: List[str] = dmk_names
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
                        logger=         sub_logger)

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
                            logger=         sub_logger)

                    # GX
                    else:
                        family = dmk_families[pa]
                        name_child = dmk_name(loop_ix=loop_ix, family=family, counter=cix)
                        other_from_family = [dn for dn in dmk_ranked if dmk_families[dn] == family]
                        if len(other_from_family) > 1:
                            other_from_family.remove(pa)
                        pb = random.choices(other_from_family, weights=list(reversed(range(len(other_from_family)))))[0]

                        ckpt_fresh = random.random() < cm.prob_fresh_ckpt
                        ckpt_fresh_info = ' (fresh ckpt)' if ckpt_fresh else ''
                        logger.info(f'> {name_child} = {pa} + {pb}{ckpt_fresh_info}')
                        FolDMK.gx_saved(
                            name_parentA=   pa,
                            name_parentB=   pb,
                            name_child=     name_child,
                            do_gx_ckpt=     not ckpt_fresh,
                            logger=         sub_logger)

                points_data['points_dmk'][name_child] = FolDMK.load_point(name=name_child)
                points_data['points_motorch'][name_child] = DMK_MOTorch.load_point(name=name_child)
                dmk_new.append(name_child)
                cix += 1

            dmk_ranked += dmk_new

        dmk_pointL = [
            {'name':dn, 'motorch_point':{'device':n % cm.n_gpu if cm.n_gpu else None}, **PUB_TR}
            for n,dn in enumerate(dmk_ranked)]

        rgd = run_GM(
            game_config=    game_config,
            name=           f'GM_{loop_ix:003}',
            dmk_point_TRL=  dmk_pointL,
            game_size=      cm.game_size_TR,
            dmk_n_players=  (cm.n_tables * game_config['table_size']) // len(dmk_pointL),
            #sep_n_stddev=   1.0,
            logger=         logger)
        dmk_results = rgd['dmk_results']

        for dn in dmk_results:

            if dn not in dmk_wonH:
                dmk_wonH[dn] = []
            dmk_wonH[dn].append(dmk_results[dn]['last_wonH_afterIV'])

            if len(dmk_wonH[dn]) > 1:

                if dn not in dmk_lifemarks:
                    dmk_lifemarks[dn] = ''
                wonH_diff = dmk_wonH[dn][-1] - dmk_wonH[dn][-2]
                dmk_lifemarks[dn] += '+' if wonH_diff > 0 else '-'

                dmk_results[dn]['wonH_diff'] = wonH_diff
                dmk_results[dn]['lifemark'] = dmk_lifemarks[dn]

        logger.info(f'train results:\n{results_report(dmk_results)}')

        # prepare new ranked
        dmk_rw = [(dn, dmk_results[dn]['last_wonH_afterIV']) for dn in dmk_results]
        dmk_ranked = [e[0] for e in sorted(dmk_rw, key=lambda x: x[1], reverse=True)]

        # add scores
        for ix,dn in enumerate(dmk_ranked):
            if dn not in points_data['scores']:
                points_data['scores'][dn] = {}
            points_data['scores'][dn][f'rank_{loop_ix}'] = (cm.n_dmk - ix) / cm.n_dmk

        w_pickle(points_data, f'{DMK_MODELS_FD}/points.data')

        # eventually remove worst
        remove = []
        for dn in dmk_ranked[-cm.n_remove:]:
            if 'lifemark' in dmk_results[dn]:
                lifemark_ending = dmk_results[dn]['lifemark'][-sum(cm.remove_key):]
                if lifemark_ending.count('-') >= cm.remove_key[0] and lifemark_ending[-1] != '+':
                    remove.append(dn)
        for dn in remove:
            logger.info(f'removing {dn} <- DMK with bad lifemark')
            dmk_ranked.remove(dn)
            shutil.rmtree(f'{DMK_MODELS_FD}/{dn}', ignore_errors=True)

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