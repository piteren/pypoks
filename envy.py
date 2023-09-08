DEBUG_MODE = True # True allows more information to be shown while playing HumanGame


# poker game settings
N_TABLE_PLAYERS =   3       # supported are 2,3,6,9
TABLE_CASH_START =  500     # player starts every hand with 500
TABLE_CASH_SB =     2
TABLE_CASH_BB =     5


# possible table moves 0..N-1 {IX: (name, ..)}
TBL_MOV = {
    0:  ('C/F', None),      # check/fold
    1:  ('CLL', None),      # call
    2:  ('BRS', 2.5, 0.6),  # bet/raise small
    3:  ('BRL', 4.0, 1.0),  # bet/raise large
}
TBL_MOV_R = {TBL_MOV[k][0]: k for k in TBL_MOV} # {name: IX}

# folders
MODELS_FD =          '_models'
DMK_MODELS_FD =     f'{MODELS_FD}/dmk'
CN_MODELS_FD =      f'{MODELS_FD}/cardNet'
DMK_POINT_PFX =      'dmk_dna'

ARCHIVE_FD =        f'{DMK_MODELS_FD}/_archive'     # here are stored DMKs removed from the loop (because of poor performance)
PMT_FD =            f'{DMK_MODELS_FD}/_pmt'         # here are stored PMT
CONFIG_FP =         f'{DMK_MODELS_FD}/pypoks.cfg'
RESULTS_FP =        f'{DMK_MODELS_FD}/loops_results.json'

DMK_STATS_IV = 1000 # DMK (player) stats interval size, it is quite important constant for DMK, StatsManager & GamesManager


def get_pos_names(n_table_players=N_TABLE_PLAYERS):
    if n_table_players not in (2,3,6,9):
        raise PyPoksException('not supported number of table players')
    if n_table_players == 2: return ['BTN','BB']
    if n_table_players == 3: return ['SB','BB','BTN']
    if n_table_players == 6: return ['SB','BB','UTG','MP','CT','BTN']
    if n_table_players == 9: return ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']


def get_cardNet_name(c_embW :int):
    return f'cardNet{c_embW}'


class PyPoksException(Exception):
    pass
