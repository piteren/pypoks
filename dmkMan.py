"""

 2019 (c) piteren

"""

from multiprocessing import Queue

from decisionMaker import DecisionMaker, SNdmk

class DMKman:

    def __init__(
            self,
            dmk,
            nInst :int):

        self.iQue = Queue()
        self.oQue = Queue()
        self.dmk = dmk(self.iQue, self.oQue)
        self.nInst = nInst
        self.manyQues = {ix: Queue() for ix in range(self.nInst)}

    # collect results and put them on proper mQues
    def run(self):
        while True:
            ix, res = self.oQue.get()
            self.manyQues[ix].put(res)
