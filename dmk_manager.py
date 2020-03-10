"""

 2019 (c) piteren

 DMK Manager connects DMKs with tables
 - takes list of DMK, puts their players on tables and runs game

"""

from multiprocessing import Queue
import os
import random
import time

from putils.neuralmess.dev_manager import nestarter
from putils.neuralmess.base_elements import mrg_ckpts

from pologic.potable import PTable
from decision_maker import BaNeDMK
from neural_graphs import lstm_GFN, cnn_GFN, cnnCE_GFN

class DMKManager:

    def __init__(
            self,
            fwd_func,
            n_dmk :int=     5,      # number of dmks
            n_players :int= 30,     # number of players per dmk
            verb=           0):

        self.verb = verb
        self.fwd_func = fwd_func
        self.n_players = n_players

        self.dmkIX = 0 # index of next dmk
        dmk_names = self._get_new_names(n_dmk)

        self.dmks = {dmk_name: BaNeDMK(
            fwd_func=   self.fwd_func,
            mdict=      {'name':dmk_name},
            n_players=  self.n_players) for dmk_name in dmk_names}
        self.tables = []
        self.pl_out_que = None  # output que (one for all players, here players put states for DMK)
        self.pl_in_queD = {}    # input ques (one per player, here  dMK puts addressed decisions for players)
        self._build_game_env()

    # returns list with new names
    def _get_new_names(self, n :int):
        dmk_names = ['cng%d' % dIX for dIX in range(self.dmkIX, self.dmkIX + n)]
        self.dmkIX += n
        return dmk_names

    # builds game envy based on self.dmks (dictionary)
    def _build_game_env(
            self,
            tpl_count=  3): # number of players per table (table size)

        # close everything
        for pa in self.pl_in_queD: self.pl_in_queD[pa].put('game_end')
        print('game ended')
        for table in self.tables:
            table.join()
            table.close()
        print('tables closed')
        for pa in self.pl_in_queD: self.pl_in_queD[pa].join()
        print('pl ques joined')
        if self.pl_out_que: self.pl_out_que.join()
        print('pl out que joined')

        # prepare player addresses (tuples)
        pl_addrL = []
        for dk in self.dmks:
            for pix in range(self.dmks[dk].n_players):  # pl ix
                pl_addrL.append((dk,pix))

        # create player ques
        self.pl_out_que = Queue()
        self.pl_in_queD = {pa: Queue() for pa in pl_addrL}

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
                pmsg=       False,
                verb=       self.verb)
            self.tables.append(table)

    # runs game loop (manages ques in loop)
    def run_games(
            self,
            n_mprn=     5,      # n minutes to print hand
            n_mgxc=     1):     # n minutes to apply genetic crossing

        print('DMKMan running games, starting %d tables...'%len(self.tables))
        for tbl in self.tables: tbl.start() # start table
        print('tables started!')

        p_time = time.time()
        g_time = p_time

        while True:
            pAddr, stateChanges, possibleMoves = self.pl_out_que.get() # wait for player data
            dk, pix = pAddr # resolve address
            dec = self.dmks[dk].proc_player_data(pAddr, stateChanges, possibleMoves)

            # split dec among ques
            if dec is not None:

                # take random dmk and print one hand
                if n_mprn and (time.time() - p_time)/60 > n_mprn:
                    dks = list(self.dmks.keys())
                    dk = random.choice(dks)
                    self.dmks[dk].store_next_hand = True
                    p_time = time.time()

                # GXC
                if n_mgxc and (time.time() - g_time)/60 > n_mgxc:
                    g_time = time.time()
                    self._gxc()

                for d in dec:
                    pIX, move = d
                    self.pl_in_queD[(dk, pIX)].put(move)

    # genetic crossing
    def _gxc(
            self,
            xcF=    0.5):

        print('GXC')

        dmkl = [self.dmks[k] for k in self.dmks]
        dmk_xc = sorted(dmkl, key= lambda x: x.stats['$'][0], reverse=True)
        for dmk in dmk_xc: print(dmk.stats['$'][0])
        dmk_xc = dmk_xc[:int(len(dmk_xc)*xcF)]

        for dmk in dmk_xc:
            dmk.save()
            dmk.stats['$'][0] = 0

        dmk_names = [dmk.name for dmk in dmk_xc]
        print(dmk_names)

        mfd = '_models/'+dmk_names[0]
        ckptL = [dI for dI in os.listdir(mfd) if os.path.isdir(os.path.join(mfd,dI))]
        ckptL.remove('opt_vars')
        print(ckptL)

        new_dmk_names = self._get_new_names(len(self.dmks)-len(dmk_names))
        print(new_dmk_names)
        for name in new_dmk_names: os.mkdir('_models/'+name)

        # merge checkpoints
        mrg_dna = {name: [random.sample(dmk_xc,2),random.random()] for name in new_dmk_names}
        for name in mrg_dna:
            dmka = mrg_dna[name][0][0]
            dmkb = mrg_dna[name][0][0]
            rat = mrg_dna[name][1]
            for ckpt in ckptL:
                mrg_ckpts(
                    ckptA=      ckpt,
                    ckptA_FD=   '_models/%s/'%dmka.name,
                    ckptB=      ckpt,
                    ckptB_FD=   '_models/%s/'%dmkb.name,
                    ckptM=      ckpt,
                    ckptM_FD=   '_models/%s/'%name,
                    mrgF=       rat)
        new_dmks = [BaNeDMK(
            fwd_func=   self.fwd_func,
            mdict=      {'name':dmk_name},
            n_players=  self.n_players) for dmk_name in new_dmk_names]
        self.dmks = dmk_xc + new_dmks
        self.dmks = {dmk.name: dmk for dmk in self.dmks}
        self._build_game_env()

if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    dmkm = DMKManager(
        fwd_func=   cnnCE_GFN,
        n_dmk=      5,
        n_players=  9,
        verb=       0)
    dmkm.run_games()