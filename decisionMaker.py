"""

 2019 (c) piteren

"""

from multiprocessing import Process, Queue
import numpy as np
import random
import tensorflow as tf
import time

from pLogic.pDeck import PDeck


# basic implementation of DMK (random sampling)
class DMK:

    def __init__(
            self,
            name :str,          # name should be unique (@table)
            nPl=        100,    # number of managed players
            nMoves=     4,      # number of (all) moves supported by DM, has to match table/player
            randMove=   0.2,    # how often move will be sampled from random
            runTB=      False):

        self.name = name
        self.nMoves = nMoves
        self.randMove = randMove

        self.nPl = nPl
        # variables below store data for FWD and BWD batches and state evaluation, are permanently updated (by encState, runUpdate...)
        self.lDMR = {ix: [] for ix in range(self.nPl)}  # list of dicts {'decState': 'move': 'reward':}
        self.preflop = {ix: True for ix in range(self.nPl)} # preflop indicator
        self.plsHpos = {ix: [] for ix in range(self.nPl)} # positions @table of players from self.pls
        self.psblMoves = {ix: [] for ix in range(self.nPl)} # possible moves save
        self.nH = 0 # number of hands

        self.runTB = runTB
        self.summWriter = tf.summary.FileWriter(logdir='_nnTB/' + self.name, flush_secs=10) if runTB else None

        self.repTime = time.time()
        self.stsV = 1000 # stats interval
        self.sts = {} # stats
        self.cHSdata = {ix: None for ix in range(self.nPl)} # current hand data for stats per player
        self._resetSTS()
        for pIX in range(self.nPl): self._resetCSHD(pIX)

    # resets self.cHSdata for player
    def _resetCSHD(
            self,
            pIX):

        self.cHSdata[pIX] = {
            'VPIP':     False,
            'PFR':      False,
            'SH':       False,
            'nPM':      0,
            'nAGG':     0}

    # resets stats
    def _resetSTS(self):
        """
        by now implemented stats:
          VPIP - Voluntarily Put $ in Pot %H; how many hands (%) player put money in pot (SB and BB do not count)
          PFR - Preflop Raise %H; how many hands (%) player raised preflop
          SH - Stacked Hands; %H where player stacked
          AGG - Postflop Aggression Frequency %; (totBet + totRaise) / anyMove *100, only postflop
        """
        self.sts = {  # [total,interval]
            'nH':       [0,0],  # n hands played
            '$':        [0,0],  # $ won
            'nVPIP':    [0,0],  # n hands with VPIP
            'nPFR':     [0,0],  # n hands with PFR
            'nSH':      [0,0],  # n hands stacked
            'nPM':      [0,0],  # n moves postflop
            'nAGG':     [0,0]}  # n aggressive moves postflop

    """
    # resets knowledge, stats, name of DMK
    def resetME(self, newName=None):

        if newName: self.name = newName
        self._resetSTS()
        if self.runTB: self.summWriter = tf.summary.FileWriter(logdir='_nnTB/' + self.name, flush_secs=10)
    """

    # table state encoder (into decState form - readable by DMC.getProbs method)
    # + updates some info (pos, pre/postflop tableState) in self variables
    def _encState(
            self,
            pIX :int,
            tableStateChanges :list):

        for state in tableStateChanges:
            key = list(state.keys())[0]

            # update positions of players @table for new hand, enter preflop
            if key == 'playersPC':
                newPos = [0]*len(state[key])
                for ix in range(len(state[key])):
                    newPos[state[key][ix][0]] = ix
                self.plsHpos[pIX] = newPos
                self.preflop[pIX] = True
            # enter postflop
            if key == 'newTableCards' and len(state[key]) == 3: self.preflop[pIX] = False

            # get reward and update
            if key == 'winnersData':

                self.nH += 1
                """
                if self.nH % 1000 == 0:
                    print(' >>> DMK speed: %.1fsec/%d hands' % (time.time() - self.repTime, 1000))
                    self.repTime = time.time()
                """

                myReward = 0
                for el in state[key]:
                    if el['pIX'] == 0: myReward = el['won']

                # update reward backward
                for ix in reversed(range(len(self.lDMR[pIX]))):
                    if self.lDMR[pIX][ix]['reward'] is None: self.lDMR[pIX][ix]['reward'] = myReward  # update reward
                    else: break

                for ti in [0,1]:
                    self.sts['nH'][ti] += 1
                    self.sts['$'][ti] += myReward

                    # update self.sts with self.cHSdata
                    if self.cHSdata[pIX]['VPIP']:    self.sts['nVPIP'][ti] += 1
                    if self.cHSdata[pIX]['PFR']:     self.sts['nPFR'][ti] += 1
                    if self.cHSdata[pIX]['SH']:      self.sts['nSH'][ti] += 1
                    self.sts['nPM'][ti] += self.cHSdata[pIX]['nPM']
                    self.sts['nAGG'][ti] += self.cHSdata[pIX]['nAGG']
                self._resetCSHD(pIX)

                # sts
                if self.sts['nH'][1] == self.stsV:

                    # reporting
                    if self.summWriter:
                        won = tf.Summary(value=[tf.Summary.Value(tag='sts/0_$wonT', simple_value=self.sts['$'][0])])
                        vpip = self.sts['nVPIP'][1] / self.sts['nH'][1] * 100
                        vpip = tf.Summary(value=[tf.Summary.Value(tag='sts/1_VPIP', simple_value=vpip)])
                        pfr = self.sts['nPFR'][1] / self.sts['nH'][1] * 100
                        pfr = tf.Summary(value=[tf.Summary.Value(tag='sts/2_PFR', simple_value=pfr)])
                        agg = self.sts['nAGG'][1] / self.sts['nPM'][1] * 100 if self.sts['nPM'][1] else 0
                        agg = tf.Summary(value=[tf.Summary.Value(tag='sts/3_AGG', simple_value=agg)])
                        sh = self.sts['nSH'][1] / self.sts['nH'][1] * 100
                        sh = tf.Summary(value=[tf.Summary.Value(tag='sts/4_sh', simple_value=sh)])
                        self.summWriter.add_summary(won, self.sts['nH'][0])
                        self.summWriter.add_summary(vpip, self.sts['nH'][0])
                        self.summWriter.add_summary(pfr, self.sts['nH'][0])
                        self.summWriter.add_summary(agg, self.sts['nH'][0])
                        self.summWriter.add_summary(sh, self.sts['nH'][0])

                    # reset interval values
                    for key in self.sts.keys():
                        self.sts[key][1] = 0

                self.runUpdate()

        # custom implementation should add further decState preparation
        decState = None
        return decState

    # returns probabilities of for move in form of [(pIX,probs)...] or None
    def _calcProbs(
            self,
            pIX,
            decState):

        """
        here goes custom implementation with:
         - more than random
         - multi player calculation >> probs will be calculated in batches
        """
        # equal probs
        return [(pIX, [1/self.nMoves]*self.nMoves)]

    # updates current hand data for stats based on move performing
    def _updMoveStats(
            self,
            pIX,    # player index
            move):  # player move

        if move == 3: self.cHSdata[pIX]['SH'] = True
        if self.preflop[pIX]:
            if move == 1 and self.plsHpos[pIX][0] != 1 or move > 1: self.cHSdata[pIX]['VPIP'] = True
            if move > 1: self.cHSdata[pIX]['PFR'] = True
        else:
            self.cHSdata[pIX]['nPM'] += 1
            if move > 1: self.cHSdata[pIX]['nAGG'] += 1

    # makes decisions based on stateChanges - selects move from possibleMoves using calculated probabilities
    # returns decisions in form of [(pIX,move)...] or None
    # TODO: prep mDec in tournament mode (no sampling from distribution, but max)
    def _makeDec(
            self,
            pIX :int,
            decState,
            possibleMoves :list):

        pProbsL = self._calcProbs(pIX, decState)
        self.psblMoves[pIX] = possibleMoves # save possible moves

        # players probabilities list will be returned from time to time, same for decisions
        decs = None
        if pProbsL is not None:
            decs = []
            for pProbs in pProbsL:
                pIX, probs = pProbs
                if random.random() < self.randMove: probs = [1/self.nMoves] * self.nMoves

                probMask = [int(pM) for pM in self.psblMoves[pIX]]
                probs = probs * np.asarray(probMask)
                if np.sum(probs) > 0: probs = probs / np.sum(probs)
                else: probs = [1/self.nMoves] * self.nMoves

                move = np.random.choice(np.arange(self.nMoves), p=probs) # sample from probs
                decs.append((pIX, move))

                # save state and move (for updates etc.)
                self.lDMR[pIX].append({
                    'decState': decState,
                    'move':     move,
                    'reward':   None})

                self._updMoveStats(pIX, move)  # stats

        return decs

    # takes player data and wraps stateEncoding+makingDecision
    def procPLData(
            self,
            pAddr,
            stateChanges,
            possibleMoves):

        dix, pix = pAddr
        decState = self._encState(pix, stateChanges)  # encode table state with DMK encoder
        decs = self._makeDec(pix, decState, possibleMoves) if possibleMoves is not None else None
        return decs

    # runs update of DMK based saved decStates, moves and rewards
    def runUpdate(self): pass

# Base-Neural-DMK (for neuralGraph)
class BNDMK(DMK):

    def __init__(
            self,
            session :tf.Session,
            gFND,               # graph function dictionary
            nPl=        100,
            randMove=   0.01,
            nH4UP=      1000):  # target number of moves 4 1 update

        super().__init__(
            name=       gFND['scope'],
            nPl=        nPl,
            randMove=   randMove,
            runTB=      True)

        self.session = session

        self.inC =          gFND['inC']
        self.inMT =         gFND['inMT']
        self.inV =          gFND['inV']
        self.wV =           gFND['wV']
        self.move =         gFND['move']
        self.reward =       gFND['reward']
        self.inState =      gFND['inState']
        singleZeroState =   gFND['singleZeroState']
        self.probs =        gFND['probs']
        self.finState =     gFND['finState']
        self.optimizer =    gFND['optimizer']
        self.loss =         gFND['loss']
        self.gN =           gFND['gN']
        vars =              gFND['vars']
        optVars =           gFND['optVars']

        zeroState = self.session.run(singleZeroState)
        self.lastFwdState = {ix: zeroState  for ix in range(self.nPl)} # netState after last fwd
        self.lastUpdState = {ix: zeroState  for ix in range(self.nPl)} # netState after last update
        self.myCards =      {ix: None       for ix in range(self.nPl)} # player+table cards
        self.decStates =    {}

        self.session.run(tf.initializers.variables(var_list=vars+optVars))

        self.nH4UP = nH4UP

    """
    def resetME(self, newName=None):
        super().resetME(newName)
        self.session.run(tf.initializers.global_variables()
    """

    # prepares state in form of nn input
    def _encState(
            self,
            pIX: int,
            tableStateChanges: list):

        super()._encState(pIX, tableStateChanges)

        inMT = []  # list of moves (2)
        inV = []  # list of vectors (2)
        for state in tableStateChanges:
            key = list(state.keys())[0]

            if key == 'playersPC':
                myCards = state[key][self.plsHpos[pIX][0]][1]
                self.myCards[pIX] = [PDeck.cti(card) for card in myCards]

            if key == 'newTableCards':
                tCards = state[key]
                self.myCards[pIX] += [PDeck.cti(card) for card in tCards]

            if key == 'moveData':
                who = state[key]['pIX'] # who moved
                if who: # my index is 0, so do not include my moves

                    inMT.append(int(state[key]['plMove'][0]))  # move type

                    vec = np.zeros(shape=self.wV)

                    vec[0] = who
                    vec[1] = self.plsHpos[pIX][who] # what position

                    vec[2] = state[key]['tBCash'] / 1500
                    vec[3] = state[key]['pBCash'] / 500
                    vec[4] = state[key]['pBCHandCash'] / 500
                    vec[5] = state[key]['pBCRiverCash'] / 500
                    vec[6] = state[key]['bCashToCall'] / 500
                    vec[7] = state[key]['tACash'] / 1500
                    vec[8] = state[key]['pACash'] / 500
                    vec[9] = state[key]['pACHandCash'] / 500
                    vec[10] = state[key]['pACRiverCash'] / 500

                    inV.append(vec)

        inC = [] + self.myCards[pIX]
        while len(inC) < 7: inC.append(52) # pad cards
        while len(inMT) < 2*(len(self.plsHpos[pIX])-1): inMT.append(4) # pad moves
        while len(inV) < 2*(len(self.plsHpos[pIX])-1): inV.append(np.zeros(shape=self.wV)) # pad vectors
        nnInput = [inC], [inMT], [inV] # seq of values (seq because of shape)
        """
        print()
        for el in nnInput:
            ls = el[0]
            if type(ls[0]) is np.ndarray:
                for e in ls:
                    print(e.tolist())
            else: print(ls)
        #"""
        return nnInput

    # calculates probs with NN for batches
    def _calcProbs(
            self,
            pIX,
            decState):

        self.decStates[pIX] = decState

        pProbsL = None
        if len(self.decStates) == self.nPl // 3: # TODO: hardcoded amount!

            pIXsl = sorted(list(self.decStates.keys())) # sorted list of pIX that will be processed

            inCbatch = []
            inMTbatch = []
            inVbatch = []
            statesBatch = []
            for pIX in pIXsl:
                inCseq, inMTseq, inVseq = self.decStates[pIX]
                inCbatch.append(inCseq)
                inMTbatch.append(inMTseq)
                inVbatch.append(inVseq)
                statesBatch.append(self.lastFwdState[pIX])

            feed = {
                self.inC:       inCbatch,
                self.inMT:      inMTbatch,
                self.inV:       inVbatch,
                self.inState:   statesBatch}

            fetches = [self.probs, self.finState]
            probs, fwdStates = self.session.run(fetches, feed_dict=feed)

            probs = np.reshape(probs, newshape=(probs.shape[0],probs.shape[-1])) # TODO: maybe smarter way

            pProbsL = []
            for ix in range(fwdStates.shape[0]):
                pIX = pIXsl[ix]
                pProbsL.append((pIX,probs[ix]))
                self.lastFwdState[pIX] = fwdStates[ix]

            self.decStates = {}

        return pProbsL

    # runs update of net
    def runUpdate(self):

        nM = [len(self.lDMR[ix]) for ix in range(self.nPl)] # number of saved moves per player (not all rewarded)
        avgNM = int(sum(nM)/len(nM)) # avg

        # do update
        updFREQ = self.nH4UP / (self.nPl / 2)
        if avgNM > updFREQ:
            #print('min med max nM', min(nM), avgNM, max(nM))
            nR = [0]* self.nPl # factual num of rewarded moves
            for pix in range(self.nPl):
                for mix in reversed(range(len(self.lDMR[pix]))):
                    if self.lDMR[pix][mix]['reward'] is not None:
                        nR[pix] = mix
                        break
            avgNR = int(sum(nR)/len(nR))
            if avgNR: # exclude 0 case
                #print('min med max nR', min(nR), avgNR, max(nR))
                uPIX = [ix for ix in range(self.nPl) if nR[ix] >= avgNR]
                #print('len(uPIX)', len(uPIX))
                #print('upd size:',len(uPIX)*avgNR)

                # build batches of data
                inCbatch = []
                inMTbatch = []
                inVbatch = []
                moveBatch = []
                rewBatch = []
                for pix in uPIX:
                    inCseq = []
                    inMTseq = []
                    inVseq = []
                    moveSeq = []
                    rewSeq = []
                    for nM in range(avgNR):
                        mDict = self.lDMR[pix][nM]
                        decState = mDict['decState']
                        inCseq += decState[0]
                        inMTseq += decState[1]
                        inVseq += decState[2]
                        moveSeq.append(mDict['move'])
                        #rew = 1 if mDict['reward'] > 0 else -1
                        #if mDict['reward'] == 0: rew = 0
                        rew = mDict['reward']
                        rewSeq.append(rew)
                    inCbatch.append(inCseq)
                    inMTbatch.append(inMTseq)
                    inVbatch.append(inVseq)
                    moveBatch.append(moveSeq)
                    rewBatch.append(rewSeq)
                statesBatch = [self.lastUpdState[ix] for ix in uPIX]  # build directly from dict of Upd states

                """
                inCbatch = np.asarray(inCbatch)
                inMTbatch = np.asarray(inMTbatch)
                inVbatch = np.asarray(inVbatch)
                moveBatch = np.asarray(moveBatch)
                rewBatch = np.asarray(rewBatch)
                statesBatch = np.asarray(statesBatch)
                print(inCbatch.shape)
                print(inMTbatch.shape)
                print(inVbatch.shape)
                print(moveBatch.shape)
                print(rewBatch.shape)
                print(statesBatch.shape)
                """

                feed = {
                    self.inC:       inCbatch,
                    self.inMT:      inMTbatch,
                    self.inV:       inVbatch,
                    self.move:      moveBatch,
                    self.reward:    rewBatch,
                    self.inState:   statesBatch}

                fetches = [self.optimizer, self.loss, self.gN, self.finState]
                _, loss, gN, updStates = self.session.run(fetches, feed_dict=feed)

                for ix in range(len(uPIX)):
                    pIX = uPIX[ix]
                    self.lastUpdState[pIX] = updStates[ix] # save states
                    self.lDMR[pIX] = self.lDMR[pIX][avgNR:] # trim

                print('%s :%4d: loss %.3f gN %.3f' % (self.name, len(uPIX)*avgNR, loss, gN))
                if self.summWriter:
                    losssum = tf.Summary(value=[tf.Summary.Value(tag='gph/loss', simple_value=loss)])
                    gNsum = tf.Summary(value=[tf.Summary.Value(tag='gph/gN', simple_value=gN)])
                    self.summWriter.add_summary(losssum, self.sts['nH'][0])
                    self.summWriter.add_summary(gNsum, self.sts['nH'][0])