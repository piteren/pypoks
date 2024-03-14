import os
import torch
from multiprocessing import Process
#from torch.multiprocessing import Process, set_start_method


class TProces(Process):

    def __init__(self):
        Process.__init__(self)

    def run(self):
        inp = torch.rand((10,20), device='cuda')
        print(f'in run {os.getpid()} {inp.device}')
        pass

if __name__ == '__main__':

    # set_start_method('spawn', force=True)

    print(f'main pid: {os.getpid()}')
    print(f'in main {os.getpid()} {torch.rand((10,20), device="cpu").device}')
    """
    Until torch CUDA in the main process won't be initialized with (for example) the line below,
    everything will work fine and CUDA will be properly initialized in subprocesses.
    To initialize cuda here (below) and in the subprocess p torch multiprocessing has to be used.
    
    Conclusion:
        if you are not going to use torch multiprocessing avoid initialization of the CUDA in the main process.
    """
    #print(f'in main {os.getpid()} {torch.rand((10,20), device="cuda").device}')

    p = TProces()
    p.start()