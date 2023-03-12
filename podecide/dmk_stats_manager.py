"""

 2020 (c) piteren

 StatsManager
    is an essential component of aware DMK / GamesManager since prepares data useful to make higher level decisions
    - supports DMK:
        - collects/builds DMK stats using given player states
            - all players stats
            - calculates average DMK stats and returns

    by now supports stats:
        player:
          VPIP  - Voluntarily Put $ In Pot %H: %H player puts money in pot (SB and BB do not count)
          PFR   - Preflop Raise: %H player raises preflop
          HF    - Hands Folded: %H player folds
          nPM   - num of postflop moves
          AGG   - Postflop Aggression Frequency %: (totBet + totRaise) / anyMove *100


"""

import time
from typing import List, Dict, Optional


# StatsManager (for DMK), manages / builds stats, publishes
class StatsManager:

    def __init__(
            self,
            name: str,          # takes name after DMK
            pids: List[str],    # list of (unique) DMK player ids
            stats_iv: int):     # stats interval (n_hands DMK receives stats dict after process_states())

        self.name = name

        self.hand_cg = 0            # hand clock global
        self.hand_ci = 0            # hand clock interval

        self.stats_iv = stats_iv

        # DMK poker stats for interval - accumulated (average for all players of one DMK)
        self.stats = {
            'won':      0,  # won $

            'nVPIP':    0,  # n hands VPIP
            'nPFR':     0,  # n hands PFR
            'nHF':      0,  # n hands folded

            'nPM':      0,  # n moves postflop
            'nAGG':     0}  # n aggressive moves postflop

        self.chsd =         {pid: None  for pid in pids} # current hand stats data (per player)
        self.is_BB =        {pid: False for pid in pids} # BB position of player at table {pid: True/False}
        self.is_preflop =   {pid: True  for pid in pids} # preflop indicator of player at table {pid: True/False}
        for pid in self.chsd:
            self.__reset_chsd(pid)

        self.speed = None # running speed H/s
        self.stime = time.time()

    # resets self.chsd for player (per player stats)
    def __reset_chsd(self, pid:str):
        self.chsd[pid] = {
            'VPIP':     False,  # VPIP in current hand
            'PFR':      False,  # PFR  in current hand
            'HF':       False,  # folded in current hand
            'nPM':      0,      # number of postflop moves in current hand
            'nAGG':     0}      # number of aggressive moves in current hand
        self.is_BB[pid] = False
        self.is_preflop[pid] = True

    # updates self.chsd with given player move
    def __upd_chsd(self, pid:str, move:str):
        if move == 'C/F': self.chsd[pid]['HF'] = True
        if self.is_preflop[pid]:
            if move == 'CLL' and not self.is_BB[pid] or 'BR' in move: self.chsd[pid]['VPIP'] = True
            if 'BR' in move: self.chsd[pid]['PFR'] = True
        else:
            self.chsd[pid]['nPM'] += 1
            if 'BR' in move: self.chsd[pid]['nAGG'] += 1

    # builds stats from player states, returns dictionary with stats to publish (once every self.stats_iv)
    def process_states(self, pid: str, states :List[list]) -> Optional[Dict]:

        statsD = None
        for s in states:
            if s[0] == 'TST':                                                       # table state changed
                if s[1][0] == 'preflop':        self.is_preflop[pid] = True
                if s[1][0] == 'flop':           self.is_preflop[pid] = False
            if s[0] == 'POS' and s[1][0] == 0:  self.is_BB[pid] = s[1][1] == 'BB'   # position
            if s[0] == 'MOV' and s[1][0] == 0:  self.__upd_chsd(pid, s[1][1])       # move received
            if s[0] == 'PRS' and s[1][0] == 0:                                      # final hand results

                my_reward = s[1][1]
                self.hand_cg += 1
                self.hand_ci += 1

                self.stats['won'] += my_reward

                # update self.stats with self.chsd
                if self.chsd[pid]['VPIP']:  self.stats['nVPIP'] += 1
                if self.chsd[pid]['PFR']:   self.stats['nPFR'] += 1
                if self.chsd[pid]['HF']:    self.stats['nHF'] += 1
                self.stats['nPM'] +=    self.chsd[pid]['nPM']
                self.stats['nAGG'] +=   self.chsd[pid]['nAGG']

                self.__reset_chsd(pid)

                # update some stats after interval
                if self.hand_ci == self.stats_iv:

                    self.speed = self.stats_iv/(time.time()-self.stime)
                    self.stime = time.time()

                    # prepare stats dict
                    statsD = {
                        '0.VPIP':       self.stats['nVPIP'] / self.hand_ci * 100,
                        '1.PFR':        self.stats['nPFR'] / self.hand_ci * 100,
                        '2.AGG':        self.stats['nAGG'] / self.stats['nPM'] * 100 if self.stats['nPM'] else 0,
                        '3.HF':         self.stats['nHF'] / self.hand_ci * 100,
                        'speed(H/s)':   self.speed,
                        'won':          self.stats['won']}

                    # reset interval values
                    self.hand_ci = 0
                    for key in self.stats.keys():
                        self.stats[key] = 0

        return statsD