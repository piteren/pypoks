import time
from multiprocessing import Process, Queue
import numpy as np
import os
import psutil


def mem(pid):
    return psutil.Process(pid).memory_info().rss / 1024 ** 2 # MB

class SubProc(Process):

    def __init__(
            self,
            data):
        self.data = data
        Process.__init__(self, target=self._run)

    def _run(self):
        print('started')
        self.data = self.data / 2
        time.sleep(2)
        print(self.data.shape)
        print('finished')
        pass

par_pid = os.getpid()
print(f'parent pid: {par_pid}')
arr = np.random.rand(10000,10000)
#arrB = np.random.rand(10000,10000)
print(f'parent mem: {mem(par_pid)}')

spL = [SubProc(arr) for _ in range(50)]
for sp in spL: sp.start()
for sp in spL:
    sp_pid = sp.pid
    print(f'sp pid: {sp_pid}')
    print(f'sp mem: {mem(sp_pid)}')
arr = arr / 2
#arrB = arrB / 2
print(f'parent mem: {mem(par_pid)}')