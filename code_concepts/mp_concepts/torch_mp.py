import numpy as np
import torch
#from multiprocessing import Process
from torch.multiprocessing import Process, set_start_method
from torchness.base_elements import TNS, DTNS
from torchness.motorch import MOTorch, Module
from torchness.layers import LayDense



class NNModule(Module):

    def __init__(self, **kwargs):

        Module.__init__(self, **kwargs)

        self.dns = LayDense(
            in_features=    200,
            out_features=   200,
            activation=     torch.nn.ReLU)

    def forward(self, x:TNS) -> DTNS:
        return {'out':self.dns(x)}

    def loss(self, x) -> DTNS:
        pass


class TProces(Process):

    def __init__(self, model):
        Process.__init__(self)
        self.model = model

    def run(self):
        inp = np.random.random((10, 200))
        for _ in range(100000):
            inp = self.model(inp)['out']
            inp = inp.detach()
            #torch.cuda.empty_cache()



if __name__ == '__main__':

    set_start_method('spawn', force=True)

    models = [MOTorch(module_type=NNModule, device=-1) for _ in range(5)]

    processes = [TProces(m) for m in models]
    for p in processes:
        p.start()