"""

 2019 (c) piteren

 DMK Manager connects DMKs with tables
 - takes list of DMK, puts their players on tables and runs game

"""

from multiprocessing import Queue
import random

from pLogic.pTable import PTable

class DMKManager:

    def __init__(
            self,
            dMKs :list,
            pMsg=       False,
            verbLev=    0):

        self.verbLev = verbLev
        self.dMKs = dMKs

        # prepare player addresses
        pAddr = []
        for dix in range(len(dMKs)):
            for pix in range(dMKs[dix].nPl):
                pAddr.append((dix,pix))

        self.pOQue = Queue() # players(table) output que (1 4 all)
        self.pIQues = {pA: Queue() for pA in pAddr} # player(table) input ques (1 per player)

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
                name=       'tbl%d'%tix,
                pMsg=       pMsg,
                verbLev=    self.verbLev)
            self.tables.append(table)

    def runGames(self):

        for tbl in self.tables: tbl.start() # start tables

        while True:
            pAddr, stateChanges, possibleMoves = self.pOQue.get() # wait for player data
            dix, pix = pAddr # resolve address
            dec = self.dMKs[dix].procPLData(pAddr, stateChanges, possibleMoves)
            # split dec among ques
            if dec is not None:
                for d in dec:
                    pIX, move = d
                    self.pIQues[(dix,pIX)].put(move)
