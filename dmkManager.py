"""

 2019 (c) piteren

"""

from multiprocessing import Queue
import random

from pLogic.pTable import PTable

class DMKManager:

    def __init__(
            self,
            dMKs :list,
            verbLev=    0):

        self.verbLev = verbLev
        self.dMKs = dMKs

        pAddr = []
        for dix in range(len(dMKs)):
            for pix in range(dMKs[dix].nPl):
                pAddr.append((dix,pix))

        self.pOQue = Queue() # players(table) output que (one common 4 all)
        self.pIQues = {pA: Queue() for pA in pAddr} # player(table) input ques

        tplCount = 3 # hardcoded number of players per table
        nTables = len(self.pIQues) // tplCount
        random.shuffle(pAddr)
        self.tables = []
        for tix in range(nTables):
            tpIQues = {}
            for pix in range(tplCount):
                pA = pAddr[tix*tplCount+pix]
                tpIQues[pA] = self.pIQues[pA]
            table = PTable(
                pIQues=     tpIQues,
                pOQue=      self.pOQue,
                name=       'tbl%d'%tix)
            self.tables.append(table)

    def runGames(self):
        for tbl in self.tables: tbl.start() # start tables

        while True:
            pAddr, stateChanges, possibleMoves = self.pOQue.get() # wait for player data
            dix, pix = pAddr
            decState = self.dMKs[dix].encState(pix, stateChanges) # encode table state with DMK encoder

            # this is ask for move
            if possibleMoves is not None:
                dec = self.dMKs[dix].mDec(pix, decState, possibleMoves)
                # got decisions for some players
                if dec is not None:
                    for d in dec:
                        pIX, move = d
                        self.pIQues[(dix,pIX)].put(move)
