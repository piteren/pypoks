import time
from typing import List, Dict, Optional, Tuple


# WonManager (for DMK), manages DMK hands interval and won $ values
class WonMan:

    def __init__(self, won_iv:int):

        self.won_iv = won_iv    # won interval (n_hands DMK receives won data after process_states())

        # counters (interval,global)
        self.n_hands =  [0,0]   # number of hands
        self.won =      [0,0]   # total won $

        self.stime = time.time()

    # updates global values with interval, then resets interval
    def __merge_interval_to_global(self):
        self.n_hands[1] += self.n_hands[0]
        self.n_hands[0] = 0
        self.won[1] += self.won[0]
        self.won[0] = 0

    # extracts hand won from player states, returns dictionary with stats to publish (but only once every self.stats_iv)
    def process_states(self, states:List[Tuple]) -> Optional[Dict]:

        for s in states:

            # my final hand results
            if s[0] == 'PRS' and s[1][0] == 0:

                self.n_hands[0] += 1
                self.won[0] += s[1][1]

                # update after interval
                if self.n_hands[0] == self.won_iv:

                    # prepare won dict
                    wonD = {
                        'speed(H/s)':   self.won_iv/(time.time()-self.stime),
                        'wonH':         self.won[0] / self.won_iv}

                    self.stime = time.time()
                    self.__merge_interval_to_global()

                    return wonD

        return None

    def get_global_nhands(self) -> int:
        return self.n_hands[0] + self.n_hands[1]