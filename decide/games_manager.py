"""

 2020 (c) piteren

 GamesManager is responsible for:
 - starting / stopping games
 - making GX (and other general policy) decisions

"""

from multiprocessing import Queue
import random
import time
from tqdm import tqdm

from decide.decision_maker import RProDMK, NeurDMK
from decide.neural.neural_graphs import cnnCEM_GFN
from pologic.potable import QPTable
from decide.gx import xross

# manages DMKs, tables, games
class GamesManager:

    def __init__(
            self,
            n_dmk=          14,
            dmk_players=    150,
            stats_iv=       5000,
            acc_won_iv=     (100000,200000),
            verb=           0):

        self.verb = verb

        self.in_que = Queue() # here receives data from DMKs

        self.tpl_count = 3 # hardcoded
        assert (n_dmk * dmk_players) % self.tpl_count == 0
        self.tables = [] # list of tables

        self.gx_iv = acc_won_iv[-2]

        # create DMK dictionary
        # RProDMK(name='dmk%d' % ix, n_players=dmk_players) for ix in range(n_dmk) # random
        self.dmkD = {
            f'dmk{ix}': NeurDMK(
                gm_que=         self.in_que,
                fwd_func=       cnnCEM_GFN,
                device=         None, # CPU
                name=           f'dmk{ix}',
                n_players=      dmk_players,
                pmex=           0.2,
                suex=           0.0,
                stats_iv=       stats_iv,
                acc_won_iv=     acc_won_iv,
                verb=           self.verb) for ix in range(n_dmk)}

    # creates tables using (ques of) DMKs
    def _create_tables(self):

        # build dict of ques tuples for all players: (IN,OUT)
        # - take them fom DMKs, shuffle and distribute among tables players
        ques = {}
        for dmk in self.dmkD.values():
            dmk_iq = dmk.dmk_in_que
            pl_iqD = dmk.pl_in_queD
            for k in pl_iqD:
                ques[k] = (pl_iqD[k],dmk_iq)
        ques_keys = list(ques.keys())
        random.shuffle(ques_keys)

        # create tables
        tables = []
        table_queD = {}
        for k in ques_keys:
            table_queD[k] = ques[k]
            if len(table_queD) == self.tpl_count:
                table = QPTable(
                    gm_que=     self.in_que,
                    pl_ques=    table_queD,
                    name=       f'tbl{len(tables)}',
                    verb=       0)
                tables.append(table)
                table_queD = {}
        return tables

    # starts tables
    def _start_tables(self):
        print('Starting tables...')
        for tbl in tqdm(self.tables): tbl.start()
        print(f' > started {len(self.tables)} tables!')

    # stops tables
    def _stop_tables(self):
        print('Stopping tables...')
        for table in self.tables: table.in_que.put('stop')
        for _ in tqdm(self.tables): self.in_que.get()
        print(' > all tables stopped!')

    # starts DMKs
    def _start_dmks(self):
        print('Starting DMKs...')
        for dmk in tqdm(self.dmkD.values()): dmk.start()
        print(f' > started {len(self.dmkD)} DMKs!')

    # stops DMKs
    def _stop_dmks(self):
        print('Stopping DMKs...')
        for dmk in self.dmkD.values(): dmk.in_que.put('stop')
        for _ in tqdm(self.dmkD): self.in_que.get()
        print(' > all DMKs stopped!')

    # runs processed games
    def run_games(
            self,
            gx_loop_sh= (3,1),  # shape of GXA while loop
            gx_exit=    True,   # perform GXA after loop exit
            gx_limit=   None):  # number of GAX to perform

        self.tables = self._create_tables()
        self._start_tables()
        self._start_dmks()

        n_sec_iv = 30 # number of seconds between reporting
        gx_counter = 0
        stime = time.time()
        gx_time = stime
        while True:
            time.sleep(n_sec_iv)

            # get reports
            reports = {}
            for dmk in self.dmkD.values(): dmk.in_que.put('send_report')
            for _ in self.dmkD:
                report = self.in_que.get()
                reports[report[0]] = report[2]

            if self.verb > 0:
                nh = [r['n_hand'] for r in reports.values()]
                print(f' GM:{(time.time()-gx_time)/60:4.1f}min, nH: {min(nh)}-{max(nh)}')

            do_gx = True
            for dmk_name in reports:
                if reports[dmk_name]['n_hand'] < self.gx_iv*(gx_counter+1):
                    do_gx = False
                    break

            if do_gx:

                gx_counter += 1
                if self.verb > 0: print(f' GM: starting GX ({gx_counter})')

                # save all
                for dmk in self.dmkD.values(): dmk.in_que.put('save_model')
                for _ in self.dmkD:
                    sr = self.in_que.get()
                    #print(sr)

                # sort DMKs
                gx_list = []
                for dmk_name in reports:
                    gx_list.append((
                        dmk_name,
                        reports[dmk_name]['acc_won'][self.gx_iv]))
                gx_list = sorted(gx_list, key= lambda x: x[1], reverse=True)

                if gx_limit and gx_counter == gx_limit:
                    gx_last_list = gx_list  # save last list for return
                    break

                xres = xross(gx_list, n_par=gx_loop_sh[0], n_mix=gx_loop_sh[1], verb=self.verb+1)

                for dmk_name in xres['mixed']: self.dmkD[dmk_name].in_que.put('reload_model')
                for _ in xres['mixed']: print(self.in_que.get())

                gx_time = time.time()

        self._stop_tables()
        self._stop_dmks()

        if gx_exit:
            size = int(len(gx_last_list)/2)
            xross(gx_last_list, n_par=size, n_mix=size, verb=2)

        return gx_last_list
