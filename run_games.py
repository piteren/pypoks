"""

 2020 (c) piteren

"""

#from putils.neuralmess.dev_manager import nestarter

from decide.dmk_manager import DMKManager
#from decide.neural.neural_graphs import cnnCE_GFN


if __name__ == "__main__":

    #nestarter('_log', custom_name='dmk_games')

    dmkm = DMKManager(
        n_dmk=      7,
        n_players=  60,
        verb=       2)
    dmkm.run_games(
        n_mprn=     5,
        n_mgxc=     60)