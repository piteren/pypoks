"""

 2019 (c) piteren

 many processes to one WITHOUT external sync

"""

from multiprocessing import Process, Queue

class many(Process):

    def __init__(
            self,
            id,
            oQue :Queue,    # out que (many)
            rQue :Queue):   # return que (many)

        super().__init__()
        self.id = id
        self.oQue = oQue
        self.rQue = rQue

    def run(self):
        tix = 0
        while True:
            self.oQue.put([self.id,tix])
            tix += 1
            done = self.rQue.get()
            print('proces id %d got %s' %(self.id, done))


class one(Process):

    def __init__(
            self,
            iQue,       # input que (one)
            oQuesD):    # output ques dict (one)

        super().__init__()
        self.iQue = iQue
        self.oQuesD = oQuesD

    def run(self):
        nDone = 0
        tList = []
        while True:
            if len(tList) == 50:
                for ts in tList:
                    ix, task = ts
                    self.oQuesD[ix].put('done')
                    nDone += 1
                    print(nDone)
                tList = []
            tList.append(self.iQue.get())

oneQue = Queue()
manyQues = {ix: Queue() for ix in range(100)}

op = one(oneQue, manyQues)
op.start()

mProc = [many(ix,oneQue,manyQues[ix]) for ix in range(100)]
for mp in mProc: mp.start()
