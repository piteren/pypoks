"""

 2019 (c) piteren

"""

from multiprocessing import Queue

from decisionMaker import DecisionMaker, SNdmk

class DMKman:

    def __init__(
            self,
            dmk,
            players :list):

        iQue = Queue()
        oQue = Queue()
        self.dmk = dmk(iQue,oQue)

        manyQues = {}
        for ix in range(len(players)):
            manyQues[ix] = Queue()
            players[ix].pix = ix
            players[ix].iQue = iQue
            players[ix].mQue = manyQues[ix]
