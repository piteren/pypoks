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
from pologic.potable import QPTable
from decide.gx import xross

# manages DMKs, tables, games
class GamesManager:

    def __init__(
            self,
            dmk_dna :dict,
            acc_won_iv,
            verb=           0):

        self.verb = verb
        if self.verb > 0: print('\n *** GamesManager *** starting...')

        self.in_que = Queue() # here receives data from DMKs

        self.tpl_count = 3 # hardcoded

        assert sum([dmk_dna[n]['n_players'] for n in dmk_dna]) % self.tpl_count == 0
        self.tables = [] # list of tables

        self.gx_iv = acc_won_iv[-2]

        # save families
        self.dmk_families = {name: dmk_dna[name]['family'] for name in dmk_dna}
        for name in dmk_dna: dmk_dna[name].pop('family')

        # create DMK dictionary
        self.dmkD = {
            name: NeurDMK(
                gm_que=         self.in_que,
                name=           name,
                acc_won_iv=     acc_won_iv,
                **dmk_dna[name]
            ) for name in dmk_dna}

    # creates tables using (ques of) DMKs
    def _create_tables(self):

        # TODO: create mixed tables (mix families)

        # build dict of ques tuples for all players: (IN,OUT)
        # - take them fom DMKs, shuffle and distribute among tables players
        ques = {}
        for dmk in self.dmkD.values():
            pl_iqD = dmk.pl_in_queD
            for k in pl_iqD:
                ques[k] = (pl_iqD[k], dmk.dmk_in_que)
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
                    verb=       self.verb-1)
                tables.append(table)
                table_queD = {}
        return tables

    # starts tables
    def _start_tables(self):
        if self.verb > 0: print('Starting tables...')
        for tbl in tqdm(self.tables): tbl.start()
        if self.verb > 0: print(f' > started {len(self.tables)} tables!')

    # stops tables
    def _stop_tables(self):
        if self.verb > 0: print('Stopping tables...')
        for table in self.tables: table.in_que.put('stop')
        for _ in tqdm(self.tables): self.in_que.get()
        if self.verb > 0: print(' > all tables stopped!')

    # starts DMKs
    def _start_dmks(self):
        if self.verb > 0: print('Starting DMKs...')
        for dmk in tqdm(self.dmkD.values()): dmk.start()
        if self.verb > 0: print(f' > started {len(self.dmkD)} DMKs!')

    # stops DMKs
    def _stop_dmks(self):
        if self.verb > 0: print('Stopping DMKs...')
        for dmk in self.dmkD.values(): dmk.in_que.put('stop')
        for _ in tqdm(self.dmkD): self.in_que.get()
        if self.verb > 0: print(' > all DMKs stopped!')

    # runs processed games
    def run_games(
            self,
            gx_loop_sh= (3,1),  # shape of GXA while loop
            gx_exit_sh= (3,3),  # shape of GXA after loop exit
            gx_limit=   None):  # number of GAX to perform

        self.tables = self._create_tables()
        self._start_tables()
        self._start_dmks()

        n_sec_iv = 30 # number of seconds between reporting
        gx_counter = 0
        stime = time.time()
        gx_time = stime
        last_nhs = 0
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
                if self.verb > 0:
                    now_nhs = sum([r['n_hand'] for r in reports.values()])
                    print(f' GM: {int((now_nhs-last_nhs)/(time.time()-gx_time))}H/s, starting GX:{gx_counter}')
                    last_nhs = now_nhs

                # save all
                for dmk in self.dmkD.values(): dmk.in_que.put('save_model')
                for _ in self.dmkD: self.in_que.get()

                # sort DMKs
                gx_list = []
                for dmk_name in reports:
                    gx_list.append((
                        dmk_name,
                        reports[dmk_name]['acc_won'][self.gx_iv],
                        self.dmk_families[dmk_name]))
                gx_list = sorted(gx_list, key= lambda x: x[1], reverse=True)

                if gx_limit and gx_counter == gx_limit:
                    gx_last_list = gx_list  # save last list for return
                    break

                if gx_loop_sh:
                    xres = xross(gx_list, shape=gx_loop_sh, verb=self.verb+1)

                    for f in xres['mixed']:
                        for dmk_name in xres['mixed'][f]: self.dmkD[dmk_name].in_que.put('reload_model')
                        for _ in xres['mixed'][f]:
                            rel = self.in_que.get()
                            print(f'{rel[0]} {rel[1]} (family {f})')

                gx_time = time.time()

        self._stop_tables()
        self._stop_dmks()

        if gx_exit_sh: xross(gx_last_list, shape=gx_exit_sh, verb=2)

        return gx_last_list
