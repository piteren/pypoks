"""

 2019 (c) piteren

 DMK Manager connects DMKs with tables
 - takes list of DMK, puts their players on tables and runs game

"""

from multiprocessing import Queue
import random
import time

from pologic.potable import PTable

class DMKManager:

    def __init__(
            self,
            dmkL :list,         # list of decision_makers
            tpl_count=  3,      # number of players per table (table size)
            pmsg=       False,
            verb=       0):

        na_players = sum([d.n_players for d in dmkL])
        assert na_players % tpl_count == 0, 'ERR: number of players is wrong!'

        self.verb = verb
        self.dmkL = dmkL

        # prepare player addresses (tuples)
        pl_addrL = []
        for dix in range(len(dmkL)):                # dmk ix
            for pix in range(dmkL[dix].n_players):  # pl ix
                pl_addrL.append((dix,pix))

        # player ques
        self.pl_out_que = Queue()                           # output que (one for all players)
        self.pl_in_queD = {pa: Queue() for pa in pl_addrL}  # input ques (one per player)

        # create tables
        self.tables = []
        random.shuffle(pl_addrL)
        for tix in range(len(self.pl_in_queD)//tpl_count):
            table_in_queD = {}
            for pix in range(tpl_count):
                pa = pl_addrL[tix*tpl_count+pix]
                table_in_queD[pa] = self.pl_in_queD[pa]
            table = PTable(
                pi_ques=    table_in_queD,
                po_que=     self.pl_out_que,
                name=       'tbl%d'%tix,
                pmsg=       pmsg,
                verb=       self.verb)
            self.tables.append(table)

    # runs game loop (manages ques in loop)
    def run_games(
            self,
            nMtoPH=     5,      # None or 0 for no print
            nMtoReset=  100):   # None or 0 for no reset $

        print('DMKMan running games, starting %d tables...'%len(self.tables))
        for tbl in self.tables: tbl.start() # start table
        print('tables started!')

        lPHtime = time.time()
        lRStime = lPHtime

        while True:
            pAddr, stateChanges, possibleMoves = self.pl_out_que.get() # wait for player data
            dix, pix = pAddr # resolve address
            dec = self.dmkL[dix].procPLData(pAddr, stateChanges, possibleMoves)

            # split dec among ques
            if dec is not None:

                # take random dmk and print one hand
                if nMtoPH and (time.time()-lPHtime)/60 > nMtoPH:
                    nD = random.randrange(len(self.dmkL))
                    self.dmkL[nD].storeNextHand = True
                    lPHtime = time.time()

                # reset user $ won
                if nMtoReset and (time.time()-lRStime)/60 > nMtoReset:
                    for dmk in self.dmkL: dmk.sts['$'][0] = 0
                    lRStime = time.time()

                for d in dec:
                    pIX, move = d
                    self.pl_in_queD[(dix, pIX)].put(move)

