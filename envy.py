from typing import List

DEBUG_MODE = True # True allows more information to be shown while playing HumanGame

# poker game settings
N_TABLE_PLAYERS =   3   # supported are 2,3,6,9
TABLE_CASH_START =  500 # player starts every hand with 500
TABLE_CASH_SB =     2
TABLE_CASH_BB =     5

# table states names
TBL_STT = {
    0:  'idle',
    1:  'preflop',
    2:  'flop',
    3:  'turn',
    4:  'river',
    5:  'showdown'}

# possible table moves [(name, Optional[preflop,flop])]
TBL_MOV = [
    ('CCK',),           # check
    ('FLD',),           # fold
    ('CLL',),           # call
    #('BRM',),           # bet/raise MIN
    ('BR1', 2.5, 0.6),  # bet/raise small
    ('BR2', 4.0, 1.0),  # bet/raise large
    #('BRA',)             # bet/raise ALL IN
]

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

# DMK won interval
# number of hands for which won $ is accumulated and then processed
# it is an important constant for DMK, WonMan & GamesManager
# since produces data used to monitor separation, stddev, progress -> training process
WON_IV = 1000

# folders
MODELS_FD =          '_models'
DMK_MODELS_FD =     f'{MODELS_FD}/dmk'
CN_MODELS_FD =      f'{MODELS_FD}/cardNet'
DMK_POINT_PFX =      'dmk_dna'

ARCHIVE_FD =        f'{DMK_MODELS_FD}/_archive'     # here are stored DMKs removed from the loop (because of poor performance)
PMT_FD =            f'{DMK_MODELS_FD}/_pmt'         # here are stored PMT
CONFIG_FP =         f'{DMK_MODELS_FD}/pypoks.cfg'
RESULTS_FP =        f'{DMK_MODELS_FD}/loops_results.json'


# returns names of positions at table
def get_pos_names(n_table_players=N_TABLE_PLAYERS) -> List[str]:
    if n_table_players == 2: return ['BTN','BB']
    if n_table_players == 3: return ['SB','BB','BTN']
    if n_table_players == 6: return ['SB','BB','UTG','MP','CO','BTN']
    if n_table_players == 9: return ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CO','BTN']
    raise PyPoksException('not supported number of table players')


def get_cardNet_name(c_embW :int):
    return f'cardNet{c_embW}'


class PyPoksException(Exception):
    pass
