from typing import List


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

GAME_CONFIGS_FD = 'game_configs'

MODELS_FD =      '_models'
CN_MODELS_FD =  f'{MODELS_FD}/cardNet'
DMK_MODELS_FD = f'{MODELS_FD}/dmk'

# some training files
TR_CONFIG_FP =  f'{DMK_MODELS_FD}/training.cfg'
TR_RESULTS_FP = f'{DMK_MODELS_FD}/training_results.json'

DEBUG_MODE =    True # True allows more information to be shown while playing HumanGame


class PyPoksException(Exception):
    pass


def get_pos_names(table_size:int) -> List[str]:
    """ returns names of positions at table """
    if table_size == 2: return ['BTN','BB']
    if table_size == 3: return ['SB','BB','BTN']
    if table_size == 6: return ['SB','BB','UTG','MP','CO','BTN']
    if table_size == 9: return ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CO','BTN']
    raise PyPoksException('not supported number of table players')


def get_cardNet_name(cards_emb_width:int):
    return f'cardNet{cards_emb_width}'