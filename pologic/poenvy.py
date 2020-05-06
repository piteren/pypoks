"""

 2020 (c) piteren

"""

DEBUG_MODE = True # True allows more information to be shown

N_TABLE_PLAYERS = 3

TABLE_CASH_START = 500 # player starts every hand with 500
# no ante yet (...and ever probably)
TABLE_SB = 2
TABLE_BB = 5

# supported table moves (moves of player supported by table)
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

# position names for table sizes
POS_NMS = {
    2:  ['SB','BB'],
    3:  ['SB','BB','BTN'],
    6:  ['SB','BB','UTG','MP','CT','BTN'],
    9:  ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']}

# position names reversed
POS_NMS_R = {POS_NMS[9][ix]: ix for ix in range(9)}

# by now 'shorthanded' & side flop is not supported (but it is not needed since 500 @start of each hand)