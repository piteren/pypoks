"""

 2020 (c) piteren

 StatsManager works for DMK collects stats of all his players and calculates average DMK stats
    > SM uses n_hands as a counter (number of hands performed by ALL table players of DMK)
    > is an essential component of aware DMK / GamesManager since prepares data useful to make higher level decisions

"""

import tensorflow as tf
import time
from typing import List


# Stats Manager (for DMK)
class StatsMNG:

    def __init__(
            self,
            name :str,
            p_addrL :list,              # list of (unique) player ids, used as keys in dicts
            start_hand= 0,              # number of hand to start with
            stats_iv=   1000,           # interval (n_hands) for putting stats on TB
            acc_won_iv= (100000,200000),# should be multiplication of stats_iv
            verb=       0):

        self.verb = verb
        self.stats_iv = stats_iv
        for v in acc_won_iv: assert v % stats_iv == 0
        self.speed =        None # speed of running in H/s
        self.won_save =     {start_hand: 0} # {n_hand: $won} saved while putting to TB (for stats_iv), it will grow but won't be big...
        self.acc_won =      {k: 0 for k in acc_won_iv} # $won/hand in ranges of acc_won_iv
        self.stats =        {} # stats of DMK (for all players)
        self.chsd =         {pID: None for pID in p_addrL} # current hand stats data (per player)
        self.is_BB =        {pID: False for pID in p_addrL} # BB position of player at table {pID: True/False}
        self.is_preflop =   {pID: True for pID in p_addrL} # preflop indicator of player at table {pID: True/False}

        self.reset_stats(start_hand)
        for pID in self.chsd: self.__reset_chsd(pID)

        self.summ_writer = tf.summary.FileWriter(logdir='_models/' + name, flush_secs=10)
        self.stime = time.time()

    # resets stats (DMK)
    def reset_stats(self, start_hand=None):
        """
        by now implemented stats:
          VPIP  - Voluntarily Put $ In Pot %H: %H player puts money in pot (SB and BB do not count)
          PFR   - Preflop Raise: %H player raises preflop
          HF    - Hands Folded: %H player folds
          AGG   - Postflop Aggression Frequency %: (totBet + totRaise) / anyMove *100
        """
        if not start_hand: start_hand = 0
        self.stats = {  # [total, interval]
            'nH':       [start_hand,0],     # n hands played (clock)
            '$':        [0,0],              # $ won
            'nVPIP':    [0,0],              # n hands VPIP
            'nPFR':     [0,0],              # n hands PFR
            'nHF':      [0,0],              # n hands folded
            'nPM':      [0,0],              # n moves postflop
            'nAGG':     [0,0]}              # n aggressive moves postflop

    # resets self.chsd for player (per player stats)
    def __reset_chsd(
            self,
            pID):

        self.chsd[pID] = {
            'VPIP':     False,
            'PFR':      False,
            'HF':       False,
            'nPM':      0,      # num of postflop moves
            'nAGG':     0}

    # updates self.chsd with given player move
    def __upd_chsd(
            self,
            pID,
            move :str):

        if move == 'C/F': self.chsd[pID]['HF'] = True
        if self.is_preflop[pID]:
            if move == 'CLL' and not self.is_BB[pID] or 'BR' in move: self.chsd[pID]['VPIP'] = True
            if 'BR' in move: self.chsd[pID]['PFR'] = True
        else:
            self.chsd[pID]['nPM'] += 1
            if 'BR' in move: self.chsd[pID]['nAGG'] += 1

    # puts prepared stats to TB
    def __push_TB(self):

        speed_summ = tf.Summary(value=[tf.Summary.Value(tag=f'reports/speed(H/s)', simple_value=self.speed)])
        self.summ_writer.add_summary(speed_summ, self.stats['nH'][0])

        for k in self.acc_won:
            if self.stats['nH'][0] >= k:
                acw_summ = tf.Summary(value=[tf.Summary.Value(tag=f'sts_acc_won/{k}', simple_value=self.acc_won[k])])
                self.summ_writer.add_summary(acw_summ, self.stats['nH'][0])

        won_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/0.$wonT', simple_value=self.stats['$'][0])])
        vpip = self.stats['nVPIP'][1] / self.stats['nH'][1] * 100
        vpip_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/1.VPIP', simple_value=vpip)])
        pfr = self.stats['nPFR'][1] / self.stats['nH'][1] * 100
        pfr_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/2.PFR', simple_value=pfr)])
        agg = self.stats['nAGG'][1] / self.stats['nPM'][1] * 100 if self.stats['nPM'][1] else 0
        agg_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/3.AGG', simple_value=agg)])
        ph = self.stats['nHF'][1] / self.stats['nH'][1] * 100
        ph_summ = tf.Summary(value=[tf.Summary.Value(tag='sts/4.HF', simple_value=ph)])
        self.summ_writer.add_summary(won_summ, self.stats['nH'][0])
        self.summ_writer.add_summary(vpip_summ, self.stats['nH'][0])
        self.summ_writer.add_summary(pfr_summ, self.stats['nH'][0])
        self.summ_writer.add_summary(agg_summ, self.stats['nH'][0])
        self.summ_writer.add_summary(ph_summ, self.stats['nH'][0])

    # extracts stats from player states
    def take_states(
            self,
            pID,
            states :List[list]):

        for s in states:
            if s[0] == 'TST':                                                   # table state changed
                if s[1] == 'preflop':           self.is_preflop[pID] =  True
                if s[1] == 'flop':              self.is_preflop[pID] = False
            if s[0] == 'POS' and s[1][0] == 0:  self.is_BB[pID] = s[1][1]=='BB' # position
            if s[0] == 'MOV' and s[1][0] == 0:  self.__upd_chsd(pID, s[1][1])   # move received
            if s[0] == 'PRS' and s[1][0] == 0:                                  # final hand results

                my_reward = s[1][1]
                for ti in [0,1]:
                    self.stats['nH'][ti] += 1
                    self.stats['$'][ti] += my_reward

                    # update self.stats with self.chsd
                    if self.chsd[pID]['VPIP']:  self.stats['nVPIP'][ti] += 1
                    if self.chsd[pID]['PFR']:   self.stats['nPFR'][ti] += 1
                    if self.chsd[pID]['HF']:    self.stats['nHF'][ti] += 1
                    self.stats['nPM'][ti] +=    self.chsd[pID]['nPM']
                    self.stats['nAGG'][ti] +=   self.chsd[pID]['nAGG']

                self.__reset_chsd(pID)

                # put stats on TB
                if self.stats['nH'][1] == self.stats_iv:

                    self.speed = self.stats_iv/(time.time()-self.stime)
                    self.stime = time.time()

                    hand_num = self.stats['nH'][0]
                    self.won_save[hand_num] = self.stats['$'][0]
                    for k in self.acc_won:
                        if hand_num-k in self.won_save:
                            self.acc_won[k] = (self.won_save[hand_num]-self.won_save[hand_num-k])/k

                    self.__push_TB()

                    for key in self.stats.keys(): self.stats[key][1] = 0 # reset interval values