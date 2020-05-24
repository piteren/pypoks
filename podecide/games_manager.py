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

from ptools.lipytools.little_methods import prep_folder

from pypoks_envy import MODELS_FD, DMK_MODELS_FD
from pologic.poenvy import N_TABLE_PLAYERS
from pologic.potable import QPTable
from podecide.gx import xross
from podecide.cardNet.put_cn_ckpt_to_models import put_cn_ckpts


# manages DMKs, tables, games
class GamesManager:

    def __init__(
            self,
            dmk_dna :dict,
            use_pretrained_cn=  True,
            acc_won_iv=         (100000,200000),
            verb=               1):

        self.verb = verb
        if self.verb > 0: print('\n *** GamesManager *** stars...')

        prep_folder(MODELS_FD)
        prep_folder(DMK_MODELS_FD)

        self.use_pretrained_cn = use_pretrained_cn

        self.in_que = Queue() # here receives data from DMKs and tables

        self.gx_iv = acc_won_iv[-2]

        # create DMK dictionary
        self.dmkD = {}
        for name in dmk_dna:
            dmk_type = dmk_dna[name].pop('dmk_type')
            self.dmkD[name] = dmk_type(
                gm_que=     self.in_que,
                name=       name,
                acc_won_iv= acc_won_iv,
                **dmk_dna[name])

        assert sum([self.dmkD[name].n_players for name in self.dmkD]) % N_TABLE_PLAYERS == 0

        self.families = set([self.dmkD[name].family for name in self.dmkD])

        self.tables = []  # list of tables

    # creates tables using (ques of) DMKs, method uses family information to mix players
    def _create_tables(self):

        # build dict of lists of player ques tuples: {family: [(p_addr,in,out)]}
        fam_ques = {fam: [] for fam in self.families}
        for dmk in self.dmkD.values():
            pl_iqD = dmk.pl_in_queD
            for k in pl_iqD:
                fam_ques[dmk.family].append((k, pl_iqD[k], dmk.dmk_in_que))

        for fam in fam_ques: random.shuffle(fam_ques[fam])

        # base alg to put all ques into one list (...for mixed tables)
        quesLL = [fam_ques[fam] for fam in fam_ques]    # list of lists
        quesL = []                                      # target flat list
        qLIX = 0
        while quesLL:
            quesL.append(quesLL[qLIX].pop())
            if not quesLL[qLIX]: quesLL.pop(qLIX)       # remove empty list
            qLIX += 1                                   # increase
            if qLIX >= len(quesLL): qLIX = 0            # reset

        # create tables
        tables = []
        table_ques = []
        while quesL:
            table_ques.append(quesL.pop())
            if len(table_ques) == N_TABLE_PLAYERS:
                table = QPTable(
                    name=       f'tbl{len(tables)}',
                    gm_que=     self.in_que,
                    pl_ques=    {t[0]: (t[1],t[2]) for t in table_ques},
                    verb=       self.verb-1)
                tables.append(table)
                table_ques = []
        return tables

    # starts tables
    def _start_tables(self):
        if self.verb > 0: print('Starting tables...')
        for tbl in tqdm(self.tables): tbl.start()
        if self.verb > 0: print(' > tables init...')
        for _ in self.tables: self.in_que.get()
        if self.verb > 0: print(f' > started {len(self.tables)} tables!')

    # stops tables
    def _stop_tables(self):
        if self.verb > 0: print('Stopping tables...')
        for table in self.tables: table.in_que.put('stop')
        for _ in tqdm(self.tables): self.in_que.get()
        if self.verb > 0: print(' > all tables exited!')

    # starts DMKs
    def _start_dmks(self):
        if self.verb > 0: print('Starting DMKs...')
        for dmk in tqdm(self.dmkD.values()): dmk.start()
        if self.verb > 0: print(' > initializing...')
        for _ in tqdm(self.dmkD): self.in_que.get()
        if self.verb > 0: print(' > DMKs initialized!')

        if self.use_pretrained_cn:
            num_done = 0
            for name in self.dmkD:
                print(f' >> putting pretrained cardNet for {name}', end='')
                done = put_cn_ckpts(name)
                if done:
                    self.dmkD[name].in_que.put('reload_model')
                    num_done += 1
                    print(' ...done!')
                else: print(' - cardNet not found!')
            for _ in range(num_done):
                rel = self.in_que.get()
                print(f' >>> after cardNet {rel[0]} {rel[1]}')

        for dmk in self.dmkD.values(): dmk.in_que.put('start_game') # synchronizes DMKs a bit...
        for _ in self.dmkD: self.in_que.get()
        if self.verb > 0: print(f' > started {len(self.dmkD)} DMKs!')

    # stops DMKs
    def _stop_dmks(self):
        if self.verb > 0: print('Stopping DMKs...')

        for dmk in self.dmkD.values(): dmk.in_que.put('stop_game')
        for _ in tqdm(self.dmkD): self.in_que.get()
        if self.verb > 0: print(' > all DMKs games stopped!')

        for dmk in self.dmkD.values(): dmk.in_que.put('stop_dmk')
        for _ in tqdm(self.dmkD): self.in_que.get()
        if self.verb > 0: print(' > all DMKs exited!')


    def start_games(self):
        self.tables = self._create_tables()
        self._start_tables()
        self._start_dmks()


    def stop_games(self):
        self._stop_tables()
        self._stop_dmks()

    # an alternative way of stopping all subprocesses (dmks & tables)
    def kill_games(self):
        if self.verb > 0: print('Killing games...')
        for dmk in self.dmkD.values(): dmk.kill()
        for table in self.tables: table.kill()

    # runs processed games with gx
    def run_gx_games(
            self,
            gx_loop_sh= (3,1),  # shape of GXA while loop
            gx_exit_sh= (3,3),  # shape of GXA after loop exit
            gx_limit=   10):    # number of GAX to perform, for None: no limit

        self.start_games()

        stime = time.time()
        gx_time = stime

        last_gx_hand = {}
        gx_counter = 0
        n_sec_iv = 30  # number of seconds between reporting
        while True:

            # first get reports
            reports = {}
            for dmk in self.dmkD.values(): dmk.in_que.put('send_report')
            for _ in self.dmkD:
                report = self.in_que.get()
                reports[report[0]] = report[2]
            # then build last_gx_hand (at first loop)
            if not last_gx_hand:
                last_gx_hand = {dmk_name: reports[dmk_name]['n_hands'] for dmk_name in reports}

            if self.verb > 0:
                nh = [r['n_hands'] for r in reports.values()]
                print(f' GM:{(time.time()-gx_time)/60:4.1f}min, nH: {min(nh)}-{max(nh)}')

            do_gx = True
            for dmk_name in reports:
                if reports[dmk_name]['n_hands'] < last_gx_hand[dmk_name] + self.gx_iv:
                    do_gx = False
                    break

            if do_gx:

                gx_counter += 1

                if self.verb > 0:
                    last_nhs = sum(last_gx_hand.values())
                    now_nhs = sum([r['n_hands'] for r in reports.values()])
                    print(f' GM: {int((now_nhs-last_nhs)/(time.time()-gx_time))}H/s, starting GX:{gx_counter}')

                for dmk_name in reports: last_gx_hand[dmk_name] = reports[dmk_name]['n_hands']

                # save all
                for dmk in self.dmkD.values(): dmk.in_que.put('save_model')
                for _ in self.dmkD: self.in_que.get()

                # sort DMKs
                gx_list = []
                for dmk_name in reports:
                    gx_list.append((
                        dmk_name,
                        reports[dmk_name]['acc_won'][self.gx_iv],
                        self.dmkD[dmk_name].family))
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

            time.sleep(n_sec_iv)

        self.stop_games()

        if gx_exit_sh: xross(gx_last_list, shape=gx_exit_sh, verb=2)

        return gx_last_list
