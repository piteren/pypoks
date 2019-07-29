"""

 2019 (c) piteren

 many processes to one with external sync (loop)

"""

from multiprocessing import Process, Queue

class many(Process):

    def __init__(
            self,
            id,
            oQue :Queue,
            mQue :Queue):

        super().__init__()
        self.id = id
        self.oQue = oQue
        self.mQue = mQue

    def run(self):
        tix = 0
        while True:
            self.oQue.put([self.id,tix])
            tix += 1
            done = self.mQue.get()
            print('proces id %d got %s' %(self.id, done))


class one(Process):

    def __init__(
            self,
            oQue,
            rQue):

        super().__init__()
        self.oQue = oQue
        self.rQue = rQue

    def run(self):
        while True:
            ix, task = self.oQue.get()
            self.rQue.put([ix, 'done'])


oneQue = Queue()
retQue = Queue()
op = one(oneQue, retQue)
op.start()

manyQues = {ix: Queue() for ix in range(100)}
mProc = [many(ix,oneQue,manyQues[ix]) for ix in range(100)]
for mp in mProc: mp.start()

zix = 0
while True:
    ix, done = retQue.get()
    manyQues[ix].put(done)
    print(zix)
    zix += 1
