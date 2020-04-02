"""

 2019 (c) piteren

 DMK Manager connects DMKs with tables
 - takes list of DMK, puts their players on tables and runs game

"""

from multiprocessing import Queue
import os
import random
import time
from tqdm import tqdm

#from putils.neuralmess.base_elements import mrg_ckpts

from pologic.potable import QPTable
from decide.decision_maker import RDMK, NeurDMK
from decide.neural.neural_graphs import cnnCEM_GFN

class DMKManager:

    def __init__(
            self,
            n_dmk :int=     5,      # number of dmks
            n_players :int= 30,     # number of players per dmk
            dec_trigger=    0.9,    # factor of tables waiting for decision to run deciding method
            verb=           0):

        self.verb = verb
        self.n_players = n_players
        self.dec_trigger = dec_trigger
        self.n_waiting_tbl = 0

        # neural
        """
        self.dmks = {
            'dmk%d'%dix: NDMK(
                fwd_func=   cnnCEM_GFN,
                name=       'dmk%d'%dix,
                n_players=  n_players,
                verb=       self.verb) for dix in range(n_dmk)}
        """

        # random
        self.dmks = {
            'ldmk%d'%dix: RDMK(
                name=       'ldmk%d'%dix,
                n_players=  n_players,
                verb=       self.verb) for dix in range(n_dmk)}

        self.tables = []
        self.pl_out_que = None  # output que (one for all players, here players put states for DMK)
        self.pl_in_queD = {}    # input ques (one per player, here dmk puts addressed decisions for players)
        self._build_game_env()

    # builds game envy based on self.dmks (dictionary)
    def _build_game_env(
            self,
            tpl_count=  3): # number of players per table (table size)

        # close everything
        if self.tables:
            for pa in self.pl_in_queD: self.pl_in_queD[pa].put('game_end')
            print('old game ended')

            n_tended = 0
            while n_tended < len(self.tables):
                msg = self.pl_out_que.get()
                if type(msg) is str and msg[:3] == 'fin':
                    n_tended += 1

            for table in tqdm(self.tables): table.join()
            print('tables joined')

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
            table = QPTable(
                pi_ques=    table_in_queD,
                po_que=     self.pl_out_que,
                name=       'tbl%d'%tix,
                verb=       self.verb)
            self.tables.append(table)

    def run_games(
            self,
            n_mprn=     2,      # n minutes to print hand
            n_mgxc=     30):    # n minutes to apply genetic crossing

        print('DMKMan running games...')

        print(' > starting %d tables'%len(self.tables))
        for tbl in tqdm(self.tables): tbl.start() # start table
        print(' >> tables started!')

        self.n_waiting_tbl = 0
        while True:
            player_data = self.pl_out_que.get()
            dk, pIX = player_data['id'] # resolve address

            # player sends states
            if 'state_changes' in player_data:
                self.dmks[dk].take_states(pIX, player_data['state_changes'])

            # player asks for move
            if 'possible_moves' in player_data:
                self.dmks[dk].take_possible_moves(pIX, player_data['possible_moves'])
                self.n_waiting_tbl += 1 # every player that sends request for move locks one table
                if self.n_waiting_tbl >= len(self.tables) * self.dec_trigger:
                    dmk_updating = sorted(list(self.dmks.values()), key=lambda x: x.num_waiting(), reverse=True)[0]  # take dmk with the highest number of waiting decisions
                    decL = dmk_updating.make_decisions()
                    if self.verb>1: print(' > waiting tables: %d, decisions made: %d'%(self.n_waiting_tbl,len(decL)))
                    for d in decL:
                        pIX, move = d
                        self.pl_in_queD[(dmk_updating.name, pIX)].put(move)
                    self.n_waiting_tbl -= len(decL) # every decision unlocks one table

    def run_loop(self):

        print('DMKMan running loop...')

        print(' > starting %d tables'%len(self.tables))
        for tbl in tqdm(self.tables): tbl.start() # start table
        print(' >> tables started!')

        decision = [0,1,2,3]  # hardcoded to speed-up
        ix = 0
        stime = time.time()
        while True:
            player_data = self.pl_out_que.get()
            dk, pIX = player_data['id'] # resolve address

            # player asks for move
            if 'possible_moves' in player_data:
                possible_moves = player_data['possible_moves']

                pm_probs = [int(pm) for pm in possible_moves]
                dec = random.choices(decision, weights=pm_probs)[0]

                self.pl_in_queD[(dk, pIX)].put(dec)
                ix += 1
                if ix==10000:
                    ix = 0
                    print('speed: %d/sec'%(10000/(time.time()-stime)))
                    stime = time.time()


"""
class DMKManager1P:

    def __init__(
            self,
            n_dmk :int=     5,      # number of dmks
            n_players :int= 30,     # number of players per dmk
            dec_trigger=    0.9,    # factor of tables waiting for decision to run deciding method
            verb=           0):

        self.verb = verb
        self.n_players = n_players
        self.dec_trigger = dec_trigger
        self.n_waiting_tbl = 0

        self.dmks = {
            'dmk%d'%dix: NDMK(
                fwd_func=   cnnCEM_GFN,
                name=       'dmk%d'%dix,
                n_players=  n_players,
                verb=       self.verb) for dix in range(n_dmk)}

        self.tables = []
        self.pl_out_que = None  # output que (one for all players, here players put states for DMK)
        self.pl_in_queD = {}    # input ques (one per player, here dmk puts addressed decisions for players)
        self._build_game_env()

    # builds game envy based on self.dmks (dictionary)
    def _build_game_env(
            self,
            tpl_count=  3): # number of players per table (table size)

        # close everything
        if self.tables:
            for pa in self.pl_in_queD: self.pl_in_queD[pa].put('game_end')
            print('old game ended')

            n_tended = 0
            while n_tended < len(self.tables):
                msg = self.pl_out_que.get()
                if type(msg) is str and msg[:3] == 'fin':
                    n_tended += 1

            for table in tqdm(self.tables): table.join()
            print('tables joined')

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
            table = QPTable(
                pi_ques=    table_in_queD,
                po_que=     self.pl_out_que,
                name=       'tbl%d'%tix,
                verb=       self.verb)
            self.tables.append(table)

    def run_games(
            self,
            n_mprn=     2,      # n minutes to print hand
            n_mgxc=     30):    # n minutes to apply genetic crossing

        print('DMKMan running games...')

        print(' > starting %d tables'%len(self.tables))
        for tbl in tqdm(self.tables): tbl.start() # start table
        print(' >> tables started!')

        self.n_waiting_tbl = 0
        while True:
            player_data = self.pl_out_que.get()
            dk, pIX = player_data['id'] # resolve address

            # player sends states
            if 'state_changes' in player_data:
                self.dmks[dk].take_states(pIX, player_data['state_changes'])

            # player asks for move
            if 'possible_moves' in player_data:
                self.dmks[dk].take_possible_moves(pIX, player_data['possible_moves'])
                self.n_waiting_tbl += 1 # every player that sends request for move locks one table
                if self.n_waiting_tbl >= len(self.tables) * self.dec_trigger:
                    dmk_updating = sorted(list(self.dmks.values()), key=lambda x: x.num_waiting(), reverse=True)[0]  # take dmk with the highest number of waiting decisions
                    decL = dmk_updating.make_decisions()
                    if self.verb>1: print(' > waiting tables: %d, decisions made: %d'%(self.n_waiting_tbl,len(decL)))
                    for d in decL:
                        pIX, move = d
                        self.pl_in_queD[(dmk_updating.name, pIX)].put(move)
                    self.n_waiting_tbl -= len(decL) # every decision unlocks one table

    def run_loop(self):

        print('DMKMan running loop...')

        print(' > starting %d tables'%len(self.tables))
        for tbl in tqdm(self.tables): tbl.start() # start table
        print(' >> tables started!')

        decision = [0,1,2,3]  # hardcoded to speed-up
        ix = 0
        stime = time.time()
        while True:
            player_data = self.pl_out_que.get()
            dk, pIX = player_data['id'] # resolve address

            # player asks for move
            if 'possible_moves' in player_data:
                possible_moves = player_data['possible_moves']

                pm_probs = [int(pm) for pm in possible_moves]
                dec = random.choices(decision, weights=pm_probs)[0]

                self.pl_in_queD[(dk, pIX)].put(dec)
                ix += 1
                if ix==10000:
                    ix = 0
                    print('speed: %d/sec'%(10000/(time.time()-stime)))
                    stime = time.time()

    # runs game loop (manages ques in loop)
    def run_games_old(
            self,
            n_mprn=     2,      # n minutes to print hand
            n_mgxc=     30):     # n minutes to apply genetic crossing

        print('DMKMan running games...')

        print(' > starting %d tables'%len(self.tables))
        for tbl in tqdm(self.tables): tbl.start() # start table
        print(' >> tables started!')

        self.n_waiting_tbl = 0
        p_time = time.time()
        g_time = p_time
        while True:
            player_data = self.pl_out_que.get()
            dk, pIX = player_data['id'] # resolve address

            # player sends states
            if 'state_changes' in player_data:
                self.dmks[dk].take_states(pIX, player_data['state_changes'])

            # player asks for move ('possible_moves' in player_data)
            if 'possible_moves' in player_data:
                self.n_waiting_tbl += 1
                self.dmks[dk].take_possible_moves(pIX, player_data['possible_moves'])
                if self.n_waiting_tbl >= len(self.tables)*self.dec_trigger:
                    dmk_updating = sorted(list(self.dmks.values()), key= lambda x: x.num_waiting(), reverse=True)[0] # take dmk with the highest number of waiting decisions

                    dec = dmk_updating.get_decisions() # TODO: remember of case 0 decisions returned, DMK should loop while decisions returned
                    print('@@@ nw dec',self.n_waiting_tbl,dmk_updating.name,len(dec))
                    self.n_waiting_tbl -= len(dec)

                    # split dec among ques of updating dmk
                    for d in dec:
                        pIX, move = d
                        print(pIX, move)
                        self.pl_in_queD[(dmk_updating.name, pIX)].put(move)

            # take random dmk and print one hand
            if n_mprn and (time.time() - p_time) / 60 > n_mprn:
                dk = random.choice(list(self.dmks.keys()))
                self.dmks[dk].store_next_hand = True
                p_time = time.time()

            if n_mgxc and (time.time()-g_time)/60 > n_mgxc: break # GXC loop break
"""

"""
class DMKManager:

    def __init__(
            self,
            fwd_func,
            n_dmk :int=     5,      # number of dmks
            n_players :int= 30,     # number of players per dmk
            dec_trigger=    0.9,    # factor of tables waiting for decision to run deciding method
            verb=           0):

        self.verb = verb
        self.fwd_func = fwd_func
        self.n_players = n_players
        self.dec_trigger = dec_trigger
        self.n_waiting_tbl = 0

        self.dmkIX = 0 # index of next dmk
        dmk_names = self._get_new_names(n_dmk)
        self._init_folders(dmk_names)

        self.dmks = {dmk_name: BaNeDMK(
            fwd_func=   self.fwd_func,
            mdict=      {'name':dmk_name},
            n_players=  self.n_players) for dmk_name in dmk_names}
        self.tables = []
        self.pl_out_que = None  # output que (one for all players, here players put states for DMK)
        self.pl_in_queD = {}    # input ques (one per player, here dmk puts addressed decisions for players)
        self._build_game_env()

    # returns list with new names
    def _get_new_names(self, n :int):
        dmk_names = ['cng%d' % dIX for dIX in range(self.dmkIX, self.dmkIX + n)]
        self.dmkIX += n
        return dmk_names

    # inits folders for first players
    @staticmethod
    def _init_folders(nameL):

        s_ckpt_FD = '_models/_CKPTs/'
        ckptL = [dI for dI in os.listdir(s_ckpt_FD) if os.path.isdir(os.path.join(s_ckpt_FD,dI))]

        for name in nameL:
            os.mkdir('_models/' + name)
            for ckpt in ckptL:
                mrg_ckpts(
                    ckptA=          ckpt,
                    ckptA_FD=       '_models/_CKPTs/',
                    ckptB=          None,
                    ckptB_FD=       None,
                    ckptM=          ckpt,
                    ckptM_FD=       '_models/%s/'%name,
                    replace_scope=  name)

    # builds game envy based on self.dmks (dictionary)
    def _build_game_env(
            self,
            tpl_count=  3): # number of players per table (table size)

        # close everything
        if self.tables:
            for pa in self.pl_in_queD: self.pl_in_queD[pa].put('game_end')
            print('old game ended')

            n_tended = 0
            while n_tended < len(self.tables):
                msg = self.pl_out_que.get()
                if type(msg) is str and msg[:3] == 'fin':
                    n_tended += 1

            for table in tqdm(self.tables): table.join()
            print('tables joined')

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
            table = QPTable(
                pi_ques=    table_in_queD,
                po_que=     self.pl_out_que,
                name=       'tbl%d'%tix,
                verb=       self.verb)
            self.tables.append(table)

    # runs game loop (manages ques in loop)
    def run_games(
            self,
            n_mprn=     2,      # n minutes to print hand
            n_mgxc=     30):     # n minutes to apply genetic crossing

        print('DMKMan running games...')
        gxcIX = 0

        while True:
            print(' > starting %d tables (GXC %d)...'%(len(self.tables),gxcIX))
            for tbl in tqdm(self.tables): tbl.start() # start table
            print(' >> tables started!')

            self.n_waiting_tbl = 0
            p_time = time.time()
            g_time = p_time
            while True:
                player_data = self.pl_out_que.get()
                dk, pIX = player_data['id'] # resolve address

                # player sends states
                if 'state_changes' in player_data:
                    self.dmks[dk].take_states(pIX, player_data['state_changes'])

                # player asks for move ('possible_moves' in player_data)
                else:
                    self.n_waiting_tbl += 1
                    self.dmks[dk].take_possible_moves(pIX, player_data['possible_moves'])
                    if self.n_waiting_tbl >= len(self.tables)*self.dec_trigger:
                        dmk_updating = sorted(list(self.dmks.values()), key= lambda x: x.num_waiting(), reverse=True)[0] # take dmk with the highest number of waiting decisions

                        dec = dmk_updating.get_decisions() # TODO: remember of case 0 decisions returned, DMK should loop while decisions returned
                        self.n_waiting_tbl -= len(dec)

                        # split dec among ques of updating dmk
                        for d in dec:
                            pIX, move = d
                            self.pl_in_queD[(dmk_updating.name, pIX)].put(move)

                # take random dmk and print one hand
                if n_mprn and (time.time() - p_time) / 60 > n_mprn:
                    dk = random.choice(list(self.dmks.keys()))
                    self.dmks[dk].store_next_hand = True
                    p_time = time.time()

                if n_mgxc and (time.time()-g_time)/60 > n_mgxc: break # GXC loop break

            self._gxc()
            gxcIX += 1

    # genetic crossing
    def _gxc(self, xcF=0.5):

        print('GXC (mixing)...')

        dmkl = [self.dmks[k] for k in self.dmks]
        dmk_xc = sorted(dmkl, key= lambda x: x.stats['$'][0], reverse=True)
        for dmk in dmk_xc: print('%10s %d'%(dmk.name,dmk.stats['$'][0]))
        dmk_xc = dmk_xc[:int(len(dmk_xc)*xcF)]

        # save best and close all
        for dmk in dmk_xc:
            dmk.save()
            dmk.stats['$'][0] = 0
        for k in self.dmks: self.dmks[k].close()

        xc_dmk_names = [dmk.name for dmk in dmk_xc]
        print(xc_dmk_names)
        mfd = '_models/'+xc_dmk_names[0]
        ckptL = [dI for dI in os.listdir(mfd) if os.path.isdir(os.path.join(mfd,dI))]
        ckptL.remove('opt_vars')

        new_dmk_names = self._get_new_names(len(self.dmks)-len(xc_dmk_names))
        for name in new_dmk_names: os.mkdir('_models/'+name)

        # merge checkpoints
        mrg_dna = {name: [[dmk.name for dmk in random.sample(dmk_xc,2)],random.random()] for name in new_dmk_names}
        print(' % mrg_dna:')
        for key in sorted(list(mrg_dna.keys())): print('%10s %s'%(key,mrg_dna[key]))
        for name in mrg_dna:
            dmka_name = mrg_dna[name][0][0]
            dmkb_name = mrg_dna[name][0][1]
            rat = mrg_dna[name][1]
            for ckpt in ckptL:
                mrg_ckpts(
                    ckptA=          ckpt,
                    ckptA_FD=       '_models/%s/'%dmka_name,
                    ckptB=          ckpt,
                    ckptB_FD=       '_models/%s/'%dmkb_name,
                    ckptM=          ckpt,
                    ckptM_FD=       '_models/%s/'%name,
                    replace_scope=  name,
                    mrgF=           rat)

        # create new dmks
        new_dmks = [BaNeDMK(
            fwd_func=   self.fwd_func,
            mdict=      {'name':dmk_name},
            n_players=  self.n_players) for dmk_name in xc_dmk_names + new_dmk_names]
        self.dmks = {dmk.name: dmk for dmk in new_dmks}

        self._build_game_env()
"""