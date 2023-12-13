import numpy as np
from typing import List, Optional


class GameState:
    """
        keeps information (data) from a point of time in a game, when player:
        - receives state_data from a table
        - is supposed to do a move      (no move is also a move)
        - gets reward for that move     (no reward is also a reward)
    """

    def __init__(self, state_orig_data):
        self.state_orig_data=          state_orig_data  # orig values of state (any object)
        self.allowed_moves: Optional[List[bool]]= None  # list of (player) allowed moves
        self.moves_cash:    Optional[List[int]]=  None  # cash for allowed moves, used only by TK for GUI
        self.probs:         Optional[np.ndarray]= None  # probabilities of moves, updated by _calc_probs
        self.move:          Optional[int]=        None  # move (selected)
        self.reward:        Optional[float]=      None  # reward (for move), direct
        self.reward_sh:     Optional[float]=      None  # reward (for move), shared (among all previous unrewarded moves)

    def __str__(self):
        return str(self.state_orig_data)