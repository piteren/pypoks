from pypaq.lipytools.pylogger import get_pylogger
from typing import List, Union, Dict, Optional

from envy import PLAYER_STATS_USED, get_pos_names, PyPoksException
from pologic.hand_history import STATE

"""
baseline values of stats to start with
for each number of table players
for their meaning: check PLAYER_STATS_USED
for their definition: check PStatsEx.STATS_RECIPE
"""
# TODO: fill with good values
INITIAL_STATS = {
                #2      #3      #6      #9      number of table players
    'VPIP':    [0.531,  0.331,  0.274,  0.192],
    'PFR':     [0.502,  0.303,  0.256,  0.171],
    '3BET':    [0.022,  0.022,  0.022,  0.022],
    '4BET':    [0.022,  0.022,  0.022,  0.022],
    'ATS':     [0.555,  0.555,  0.555,  0.555],
    'FTS':     [0.555,  0.555,  0.555,  0.555],
    'CB':      [0.777,  0.777,  0.777,  0.777],
    'CBFLD':   [0.222,  0.222,  0.222,  0.222],
    'DNK':     [0.111,  0.111,  0.111,  0.111],
    'pAGG':    [0.666,  0.666,  0.666,  0.666],
    'WTS':     [0.333,  0.333,  0.333,  0.333],
    'W$SD':    [0.555,  0.555,  0.555,  0.555],
    'AFq':     [0.666,  0.666,  0.666,  0.666],
    'HF':      [0.555,  0.555,  0.555,  0.555],
}


class PStatsEx:
    """ Player Stats Extractor """

    # recipe how to calculate stats with __interval_counts (val = a / b but only if b != 0)
    STATS_RECIPE = {
        'VPIP':  ('nVPIP',             'n_my_preflop'),
        'PFR':   ('nPFR',              'n_my_preflop'),
        '3BET':  ('n3BET',             'n_my_preflop'),
        '4BET':  ('n4BET',             'n_my_preflop'),
        'ATS':   ('nATS',              'nATScould'),
        'FTS':   ('nFTS',              'nATSfaced'),
        'CB':    ('nCB',               'nCBcould'),
        'CBFLD': ('nFLDCB',            'nFLDCBcould'),
        'DNK':   ('nDNK',              'nDNKcould'),
        'pAGG':  ('nBRpostflop',       'nMOVpostflop'),
        'WTS':   ('n_my_showdown_fp',  'n_flop_seen'),
        'W$SD':  ('n_my_showdown_won', 'n_my_showdown'),
        'AFq':   ('nBR',               'nMOV'),
        'HF':    ('nHF',               'n_my_preflop'),
    }

    def __init__(
            self,
            player: Union[str,int],                     # player name in states
            table_size: int,
            table_moves: List,
            use_initial=                        True,   # for True starts with INITIAL_STATS
            initial_override: Optional[Dict]=   None,   # optional dict with stats to be overriden
            initial_size=                       100,    # weight of INITIAL_STATS
            upd_freq=                           100,    # how often player stats are updated (every N of hands)
            logger=                             None,
            loglevel=                           20,
    ):
        if not logger:
            logger = get_pylogger(name='PStatsEx', level=loglevel)
        self.logger = logger

        for sname in PLAYER_STATS_USED:
            if sname not in INITIAL_STATS:
                raise PyPoksException(f'Stat {sname} not in INITIAL_STATS!')
            if sname not in PStatsEx.STATS_RECIPE:
                raise PyPoksException(f'Stat {sname} not in PStatsEx.STATS_RECIPE!')

        self.player = player
        self.table_moves = table_moves
        self.n_hands = 0 # global hands counter
        self.__table_pos_names = get_pos_names(table_size)

        # player stats {k: [value, n]} n - number of events / weight
        self.__stats = {k: [0.0,   0] for k in PLAYER_STATS_USED}
        if use_initial:
            ix_is = 0 # 2 players
            if len(self.__table_pos_names) == 3: ix_is = 1
            if len(self.__table_pos_names) == 6: ix_is = 2
            if len(self.__table_pos_names) == 9: ix_is = 3
            for k in self.__stats:
                self.__stats[k] = [INITIAL_STATS[k][ix_is], initial_size]
        if initial_override:
            for sname in initial_override:
                if sname not in self.__stats:
                    raise PyPoksException(f'Unknown initial stat given: {sname}!')
                self.__stats[sname] = [initial_override[sname], initial_size]

        self.__upd_freq = upd_freq
        self.__n_interval_hands = 0

        self.__hen = {} # current hand events notes 
        self.__interval_counts = {} # my interval events, are build with hand events notes

        self.__reset_hand_notes()
        self.__reset_interval()

    def __reset_hand_notes(self):
        
        self.__hen = {
            
                ### preflop
            'my_preflop':        False, # I have done at least one move
            'is_preflop':        True,  # is it preflop now
            'nBRpreflop':        0,     # number of preflop BR moves (from all)
            'firstBRpreflop':    None,  # player name (Union[str,int]) TODO: ?? some before limped to BB
            'playerPOS':         {},    # {player: POS name (str)}
            'preflop_aggressor': None,  # player name (Union[str,int]) of preflop aggressor TODO: ?? 1. BB (flat)called by other, 2. BRA of other to my BR without legal bet size -> then I CLL
                # VPIP / PFR
            'haveVPIP':          False, # I have VPIP
            'havePFR':           False, # I have PFR
                # 3/4 BET
            '3BETpreflop':       False, # I have 3BET preflop
            '4BETpreflop':       False, # I have 4BET preflop
                # steal
            'ATScould':          False, # I had ATS possibility (my position in [CO,BTN,SB] and no other BR before)
            'ATSplayer':         None,  # player name (Union[str,int]) who attempted to steal
            'facedATS':          False, # I faced ATS (my POS is SB or BB and someone ATS)
            'foldedATS':         False, # I folded while facing ATS
    
                ### flop
            'my_flop':           False, # I have done at least one flop move
            'is_flop':           False, # is it flop now
            'nMOVflop':          0,     # number of flop moves (from all)
            'nBRflop':           0,     # number of flop BR moves (from all)
            'firstBRflop':       None,  # player name (Union[str,int])
                # CB / DNK
            'couldCB':           False, # I was preflop aggressor and could BR first on the flop
            'haveCB':            False, # I have CB flop <- I was preflop aggressor and BR first on the flop
            'have_seenCB':       False, # I was faced flop CB (I was preflop not-aggressor and villain bet flop before me)
            'folded2CB':         False, # I was preflop not-aggressor and have folded to CB flop
            'couldDNK':          False, # I could donk bet flop
            'haveDNK':           False, # I have donk bet flop
    
                ### postflop
            'nMOVpostflop_my':   0,     # number of my postflop moves
            'nBRpostflop_my':    0,     # number of my postflop aggressive (BR) moves
    
                ### showdown
            'was_showdown':      False, # hand finished with showdown (but: I could FLD before, ..so it could be not mine)
    
                ### global
            'nBRmy':             0,     # number of my aggressive (BR) moves
            'nMOVmy':            0,     # number of all my MOV
            'won':               0,     # $ I won
            'FLD':               False, # I folded
        }

    def __reset_interval(self):
        self.__interval_counts = {
                # preflop
            'n_my_preflop':      0, # number of preflops where I MOVed (consider such hand: I was in BB and everybody folded to me -> I have no MOV -> all preflop stats are not counted)
            'nVPIP':             0, # number of VPIP hands
            'nPFR':              0, # number of PFR hands
            'n3BET':             0, # number of preflop 3Bets
            'n4BET':             0, # number of preflop 4Bets 
            'nATScould':         0, # number of my possible ATS (my position in [CO,BTN,SB] and no other BR before)
            'nATS':              0, # number of ATS (BR from ATSPOS when no other BR before)
            'nATSfaced':         0, # number of my faced ATS (my POS is SB or BB and someone ATS)
            'nFTS':              0, # number of my FLD to faced ATS
                # flop
            'n_flop_seen':       0, # number of my seen flops
            'nCBcould':          0, # number of my possible flop CB (I was preflop aggressor and no one has bet flop before)
            'nCB':               0, # number of my CB flop
            'nFLDCBcould':       0, # number of my faced flop CB (I was preflop not-aggressor)
            'nFLDCB':            0, # number of my folds to CB flop
            'nDNKcould':         0, # number of hands where I could DNK bet
            'nDNK':              0, # number of hands where I have DNK bet
                # showdown
            'n_my_showdown':     0, # number of hands where I have seen showdown
            'n_my_showdown_fp':  0, # number of hands where I have seen showdown and have seen flop
            'n_my_showdown_won': 0, # number of hands where I have seen showdown and WON
                # postflop
            'nMOVpostflop':      0, # number of postflop moves
            'nBRpostflop':       0, # number of aggressive (BR) postflop moves
                # global
            'nBR':               0, # number of aggressive moves (BR) - total
            'nMOV':              0, # number of MOV - total
            'nHF':               0, # number of folded hands
        }
        self.__n_interval_hands = 0

    def process_states(self, states:List[STATE]):

        for s in states:

            snm = s[0]

            if snm == 'TST':

                # flop (preflop finished)
                if s[1][0] == 2:
                    self.__hen['is_preflop'] = False
                    self.__hen['is_flop'] = True

                # turn (flop finished)
                if s[1][0] == 3:
                    self.__hen['is_flop'] = False

                if s[1][0] == 5:
                    self.__hen['was_showdown'] = True

            # POS
            if snm == 'POS':
                self.__hen['playerPOS'][s[1][0]] = self.__table_pos_names[s[1][1]]

            # move
            if snm == 'MOV':

                mv = s[1][1]
                move_name = self.table_moves[mv][0]
                
                if self.__hen['is_preflop']:
                    
                    if 'BR' in move_name:
                    
                        self.__hen['nBRpreflop'] += 1

                        if self.__hen['firstBRpreflop'] is None:
                            self.__hen['firstBRpreflop'] = s[1][0]

                        self.__hen['preflop_aggressor'] = s[1][0]  # set preflop aggressor (last that has BR preflop)
                        
                        if self.__hen['nBRpreflop'] == 1 and self.__hen['playerPOS'][s[1][0]] in ['CO','BTN','SB']:
                            self.__hen['ATSplayer'] = s[1][0]
                    
                if self.__hen['is_flop']:
                    
                    self.__hen['nMOVflop'] += 1

                    if 'BR' in move_name:

                        self.__hen['nBRflop'] += 1

                        # save first betting player name
                        if self.__hen['firstBRflop'] is None:
                            self.__hen['firstBRflop'] = s[1][0]

                # my move
                if s[1][0] == self.player:

                    self.__hen['nMOVmy'] += 1
                    self.__hen['my_preflop'] = True

                    # late POS defense
                    if self.__hen['is_preflop']:

                        if self.__hen['playerPOS'][self.player] in ['CO','BTN','SB'] and self.__hen['firstBRpreflop'] in [None, self.player]:
                            self.__hen['ATScould'] = True
                        
                        # faced ATS (someone tries to steal, and I am in defense POS)
                        if self.__hen['playerPOS'][self.player] in ['SB','BB']:
                            if self.__hen['ATSplayer'] is not None and self.__hen['nBRpreflop'] == 1:
                                self.__hen['facedATS'] = True

                    if self.__hen['is_flop']:

                        self.__hen['my_flop'] = True

                        # I am preflop aggressor
                        if self.__hen['preflop_aggressor'] == self.player:

                            # no BR before -> I could CB
                            if self.__hen['firstBRflop'] in [None, self.player]:
                                self.__hen['couldCB'] = True

                            # I have done CB
                            if self.__hen['firstBRflop'] == self.player:
                                self.__hen['haveCB'] = True

                        # I am preflop not-aggressor
                        else:
                            
                            # it is my MOV, and it is first on flop
                            if self.__hen['nMOVflop'] == 1:
                                self.__hen['couldDNK'] = True
                            
                            # there was one BR yet (CB) form someone else -> I'm facing CB
                            if self.__hen['nBRflop'] == 1 and self.__hen['firstBRflop'] != self.player:
                                self.__hen['have_seenCB'] = True
                        
                    # my postflop MOV
                    if not self.__hen['is_preflop']:
                        self.__hen['nMOVpostflop_my'] += 1
                        
                    if move_name == 'FLD':

                        self.__hen['FLD'] = True

                        # FTS
                        if self.__hen['is_preflop'] and self.__hen['playerPOS'][self.player] in ['SB','BB']:
                            if self.__hen['ATSplayer'] is not None and self.__hen['nBRpreflop'] == 1:
                                self.__hen['foldedATS'] = True

                        # other player CB and I FLD
                        if self.__hen['is_flop'] and self.__hen['nBRflop'] == 1 and self.__hen['preflop_aggressor'] != self.player:
                            self.__hen['folded2CB'] = True

                    if move_name == 'CLL':

                        if self.__hen['is_preflop']:
                            self.__hen['haveVPIP'] = True

                    if 'BR' in move_name:

                        self.__hen['nBRmy'] += 1

                        if self.__hen['is_preflop']:

                            self.__hen['haveVPIP'] = True
                            self.__hen['havePFR'] = True

                            # my preflop 3BET
                            if self.__hen['nBRpreflop'] == 2:
                                self.__hen['3BETpreflop'] = True

                            # my preflop 4BET
                            if self.__hen['nBRpreflop'] == 3:
                                self.__hen['4BETpreflop'] = True

                        else:
                            self.__hen['nBRpostflop_my'] += 1

                        if self.__hen['is_flop']:
                            if self.__hen['preflop_aggressor'] != self.player and self.__hen['nMOVflop'] == 1:
                                self.__hen['haveDNK'] = True

            # my results -> hand finished
            if snm == 'PRS' and s[1][0] == self.player:

                self.__hen['won'] = s[1][1]

                ### preflop
                if self.__hen['my_preflop']:
                    self.__interval_counts['n_my_preflop'] += 1
                if self.__hen['haveVPIP']:
                    self.__interval_counts['nVPIP'] += 1
                if self.__hen['havePFR']:
                    self.__interval_counts['nPFR'] += 1
                if self.__hen['3BETpreflop']:
                    self.__interval_counts['n3BET'] += 1
                if self.__hen['4BETpreflop']:
                    self.__interval_counts['n4BET'] += 1
                if self.__hen['ATScould']:
                    self.__interval_counts['nATScould'] += 1
                if self.__hen['ATSplayer'] == self.player:
                    self.__interval_counts['nATS'] += 1
                if self.__hen['facedATS']:
                    self.__interval_counts['nATSfaced'] += 1
                if self.__hen['foldedATS']:
                    self.__interval_counts['nFTS'] += 1

                ### flop
                if self.__hen['my_flop']:
                    self.__interval_counts['n_flop_seen'] += 1
                # CB
                if self.__hen['couldCB']:
                    self.__interval_counts['nCBcould'] += 1
                if self.__hen['haveCB']:
                    self.__interval_counts['nCB'] += 1
                # DNK
                if self.__hen['couldDNK']:
                    self.__interval_counts['nDNKcould'] += 1
                if self.__hen['haveDNK']:
                    self.__interval_counts['nDNK'] += 1
                # CBFLD
                if self.__hen['have_seenCB']:
                    self.__interval_counts['nFLDCBcould'] += 1
                if self.__hen['folded2CB']:
                    self.__interval_counts['nFLDCB'] += 1

                ### postflop
                self.__interval_counts['nMOVpostflop'] += self.__hen['nMOVpostflop_my']
                self.__interval_counts['nBRpostflop'] += self.__hen['nBRpostflop_my']

                ### showdown
                if self.__hen['was_showdown'] and not self.__hen['FLD']:
                    self.__interval_counts['n_my_showdown'] += 1
                if self.__hen['was_showdown'] and not self.__hen['FLD'] and self.__hen['my_flop']:
                    self.__interval_counts['n_my_showdown_fp'] += 1
                if self.__hen['was_showdown'] and not self.__hen['FLD'] and self.__hen['won'] > 0:
                    self.__interval_counts['n_my_showdown_won'] += 1

                ### global
                if self.__hen['FLD']:
                    self.__interval_counts['nHF'] += 1
                self.__interval_counts['nBR'] += self.__hen['nBRmy']
                self.__interval_counts['nMOV'] += self.__hen['nMOVmy']

                # update & reset
                self.n_hands += 1
                self.__n_interval_hands += 1

                self.logger.debug('### hand notes:')
                for k in self.__hen:
                    if k not in ['is_preflop','playerPOS','is_flop']:
                        self.logger.debug(f"> {k:18}: {self.__hen[k]}")

                self.__reset_hand_notes()

                # time to update stats
                if self.__n_interval_hands == self.__upd_freq:

                    for k in self.__stats:
                        # (nh,nc) if nc != 0 -> stat = nh/nc -> add it with weight of nc -> update weight
                        nhk,nck = PStatsEx.STATS_RECIPE[k]
                        nh,nc = self.__interval_counts[nhk], self.__interval_counts[nck]
                        if nc:
                            self.__stats[k][0] = (self.__stats[k][0] * self.__stats[k][1] + nh) / (self.__stats[k][1] + nc)
                            self.__stats[k][1] += nc

                    self.__reset_interval()

    @property
    def player_stats(self) -> Dict:
        return {k: self.__stats[k][0] for k in self.__stats}

    # returns detailed str with player stats (with support)
    def __str__(self):
        s = f'player {self.player} played {self.n_hands} hands:\n'
        for k in self.__stats:
            s += f'{k:5} : {self.__stats[k][0]*100:4.1f}% ({self.__stats[k][1]})\n'
        return s[:-1]