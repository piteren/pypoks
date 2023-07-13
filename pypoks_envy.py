DEBUG_MODE = True # True allows more information to be shown


# poker game settings
N_TABLE_PLAYERS =   3
TABLE_CASH_START =  500 # player starts every hand with 500
TABLE_SB =          2
TABLE_BB =          5

TBL_MOV = {
    0:  ('C/F', None),      # check/fold
    1:  ('CLL', None),      # call
    2:  ('BRS', 2.5, 0.6),  # bet/raise small
    3:  ('BRL', 4.0, 1.0),  # bet/raise large
    #4:  ('BR6', 6/3),  # bet/raise 6/3 pot
    #5:  ('BR8', 8/3),  # bet/raise 8/3 pot
    # 4:'BRA'   # all-in
}
TBL_MOV_R = {TBL_MOV[k][0]: k for k in TBL_MOV}

POS_NMS = {
    2:  ['BTN','BB'],
    3:  ['SB','BB','BTN'],
    6:  ['SB','BB','UTG','MP','CT','BTN'],
    9:  ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']}
POS_NMS_R = {POS_NMS[9][ix]: ix for ix in range(9)}

# folders
MODELS_FD =          '_models'
DMK_MODELS_FD =     f'{MODELS_FD}/dmk'
CN_MODELS_FD =      f'{MODELS_FD}/cardNet'
DMK_POINT_PFX =      'dmk_dna'

ARCHIVE_FD =        f'{DMK_MODELS_FD}/_archive'     # here are stored DMKs removed from the loop (because of poor performance)
PMT_FD =            f'{DMK_MODELS_FD}/_pmt'         # here are stored PMT
CONFIG_FP =         f'{DMK_MODELS_FD}/pypoks.cfg'
RESULTS_FP =        f'{DMK_MODELS_FD}/all_results.json'

DMK_STATS_IV = 1000 # DMK (player) stats interval size, it is quite important constant for DMK, StatsManager & GamesManager

def get_cardNet_name(c_embW :int):
    return f'cardNet{c_embW}'
