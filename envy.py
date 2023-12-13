from pypaq.lipytools.files import list_dir
import shutil
from typing import List, Dict, Optional
import yaml


# table states
TBL_STT = {
    0:  'idle',
    1:  'preflop',
    2:  'flop',
    3:  'turn',
    4:  'river',
    5:  'showdown'}

# stats implemented, monitored (TB) and used (player data)
PLAYER_STATS_USED = (
        # preflop
    'VPIP',     # voluntarily put money into pot
    'PFR',      # preflop raise
    '3BET',     # preflop first re-raise
    # '4BET',     # preflop second re-raise
    'ATS',      # attempt to steal
    'FTS',      # fold to steal
        # flop
    'CB',       # flop CB (for preflop aggressor)
    'CBFLD',    # flop fold to CB (for preflop not-aggressor)
    'DNK',      # flop donk bet (for preflop not-aggressor)
        # postflop
    'pAGG',     # postflop aggression
        # showdown
    'WTS',      # went to showdown
    'W$SD',     # won the showdown
        # global
    # 'AFq',      # global aggression freq
    'HF',       # folded the hand
)

# folders / files / paths
CACHE_FD =       '_cache'
ASC_FP =        f'{CACHE_FD}/asc.dict'

MODELS_FD =      '_models'
CN_MODELS_FD =  f'{MODELS_FD}/cardNet'
DMK_MODELS_FD = f'{MODELS_FD}/dmk'

# some training files
TR_CONFIG_FP =  f'{DMK_MODELS_FD}/training.cfg'
TR_RESULTS_FP = f'{DMK_MODELS_FD}/training_results.json'

DEBUG_MODE =    True # True allows more information to be shown while playing HumanGame


def get_pos_names(table_size:int) -> List[str]:
    """ returns names of positions at table """
    if table_size == 2: return ['BTN','BB']
    if table_size == 3: return ['SB','BB','BTN']
    if table_size == 6: return ['SB','BB','UTG','MP','CO','BTN']
    if table_size == 9: return ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CO','BTN']
    raise PyPoksException('not supported number of table players')


def get_cardNet_name(cards_emb_width:int):
    return f'cardNet{cards_emb_width}'


def get_game_config_name(folder:str) -> str:
    files = list_dir(folder)['files']
    config_names = [fn[:-8] for fn in files if fn.endswith('_gc.yaml')]
    if len(config_names) == 0:
        raise PyPoksException('there is no config_file in given folder!')
    if len(config_names) > 1:
        raise PyPoksException('there are many config_files in given folder!')
    return config_names[0]


def load_game_config(
        name: Optional[str]=    None,
        folder: Optional[str]=  'game_configs',
        copy_to: Optional[str]= None,
) -> Dict:
    """ loads game config with given name from folder,
    optionally copies it to copy_to folder, if given
    """

    if name is None and folder is None:
        raise PyPoksException('name or folder must be given!')

    if not name:
        name = get_game_config_name(folder)

    with open(f"{folder}/{name}_gc.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if copy_to is not None:
        shutil.copyfile(f'{folder}/{name}_gc.yaml', f'{copy_to}/{name}_gc.yaml')

    return config


class PyPoksException(Exception):
    pass
