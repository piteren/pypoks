"""

 2019 (c) piteren

"""

import random
import tensorflow as tf
import time

# player possible moves
PLR_MVS = {
    0:      'C/F',
    1:      'CLL',
    2:      'B/R',
    3:      'ALL'}


class PPlayer:

    def __init__(
            self,
            name=       'pl0',
            dMK=        None):      # decisionMaker

        self.verbLev = 1
        self.pMsg = True
        self.name = name
        self.dMK = dMK
        self.nextStateUpdate = 0 # number of hand state to_update_from next
        # managed ONLY by table
        self.table = None
        self.hand = None
        self.cash = 0
        self.wonLast = 0
        self.wonTotal = 0
        self.cHandCash = 0 # current hand cash (amount put by player on current hand)
        self.cRiverCash = 0 # current river cash (amount put by player on current river yet)

        self.summWriter = None
        if self.name == 'pl0':
            self.summWriter = tf.summary.FileWriter(logdir='_nnTB/pyp_'+time.strftime("%Y.%m.%d_%H.%M.%S")[5:-3], flush_secs=10)
        self.counter = 0

        if self.verbLev: print('(player)%s created' % self.name)

    # makes player decision (having table status ...and any other info)
    def makeMove(self):

        # calculate possible moves and cash based on table state and hand history
        possibleMoves = {x: True for x in range(4)} # by now all
        if self.table.cashToCall - self.cRiverCash == 0: possibleMoves[1] = False # cannot call (already called)
        if self.cash < 2*self.table.cashToCall: possibleMoves[2] = False # cannot bet/raise
        if self.cash == self.table.cashToCall - self.cRiverCash: possibleMoves[1] = False # cannot call (just allin)
        possibleMovesCash = {
            0:  0,
            1:  self.table.cashToCall - self.cRiverCash,
            2:  2*self.table.cashToCall - self.cRiverCash if self.table.cashToCall else self.table.cash // 2, # by now simple
            3:  self.cash}
        possibleMoves = [key for key in possibleMoves.keys() if possibleMoves[key]]

        # dMK move
        if self.dMK:
            currentHandH = self.table.hands[-1]
            stateChanges = currentHandH[self.nextStateUpdate:]
            self.nextStateUpdate = len(currentHandH)
            selectedMove = self.dMK.mDec(stateChanges, possibleMoves)
        # random move
        else:
            random.shuffle(possibleMoves)
            selectedMove = possibleMoves[0]

        return selectedMove, possibleMovesCash[selectedMove]

    # method called by table to inform player about reward for last hand
    # player may update move making alg here
    def getReward(
            self,
            reward: int):

        self.nextStateUpdate = 0 # TODO: not the best place to reset this counter ...player should know new hand starts
        if self.dMK: self.dMK.getReward(reward)

        if self.summWriter and self.counter % 1000 == 0:
            accSumm = tf.Summary(value=[tf.Summary.Value(tag=self.name+'/wonTot', simple_value=self.wonTotal)])
            self.summWriter.add_summary(accSumm, self.counter)
        self.counter += 1
