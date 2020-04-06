"""

 2020 (c) piteren

"""

from multiprocessing import Queue
import random
from tqdm import tqdm

from putils.neuralmess.dev_manager import nestarter

from decide.decision_maker import ProDMK, NeurDMK
from decide.neural.neural_graphs import cnnCEM_GFN
from pologic.potable import QPTable

# class responsible for running / stopping games
class GamesManager:

    def __init__(
            self,
            n_dmk=          14,
            dmk_players=    150):

        tpl_count = 3 # hardcoded
        assert (n_dmk * dmk_players) % tpl_count == 0

        # create DMKs
        #self.dmkL = [ProDMK(name='dmk%d' % ix, n_players=dmk_players) for ix in range(n_dmk)]
        self.dmkL = [NeurDMK(
            fwd_func=       cnnCEM_GFN,
            device=         None, # CPU
            name=           'dmk%d'%ix,
            n_players=      dmk_players,
            pmex=           0.2,
            suex=           0.0,
            stats_iv=       10000,
            verb=           1) for ix in range(n_dmk)]

        # get all ques into one dict
        ques = {}
        for dmk in self.dmkL:
            pl_oq = dmk.pl_out_que
            pl_iqD = dmk.pl_in_queD
            for k in pl_iqD:
                ques[k] = (pl_iqD[k],pl_oq)

        # create tables
        ques_keys = list(ques.keys())
        random.shuffle(ques_keys)
        self.tables = []
        table_queD = {}
        for k in ques_keys:
            table_queD[k] = ques[k]
            if len(table_queD) == tpl_count:
                table = QPTable(
                    pl_ques=    table_queD,
                    name=       'tbl%d' % len(self.tables),
                    # verb=       self.verb,
                )
                self.tables.append(table)
                table_queD = {}

    # runs processed games
    def run_games(self):
        print('Starting DMKs...')
        for dmk in tqdm(self.dmkL): dmk.start()
        print('Starting tables...')
        for tbl in tqdm(self.tables): tbl.start()



if __name__ == "__main__":

    nestarter('_log', custom_name='dmk_games')

    gm = GamesManager()
    gm.run_games()