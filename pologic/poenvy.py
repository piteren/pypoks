"""

 2020 (c) piteren

"""

# supported table moves (moves of player supported by table)
TBL_MOV = {
    0:  'C/F',  # check/fold
    1:  'CLL',  # call
    2:  'BR5',  # bet/raise 0.5
    3:  'BR8'}  # bet/raise 0.8
    # 4:'BRA'   # all-in

# position names for table sizes
POS_NMS = {
    2:  ['SB','BB'],
    3:  ['SB','BB','BTN'],
    6:  ['SB','BB','UTG','MP','CT','BTN'],
    9:  ['SB','BB','UTG1','UTG2','MP1','MP2','HJ','CT','BTN']}

N_TABLE_PLAYERS = 3

TABLE_CASH_START = 500 # player starts every hand with 500
# no ante yet (...and ever probably)
TABLE_SB = 2
TABLE_BB = 5

# by now 'shorthanded' & side flop is not supported (but it is not needed since 500 @start of each hand)