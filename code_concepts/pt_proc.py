import os
import time
import torch

print(os.getpid())
a = torch.tensor(10)
print(a)
a.to('cuda')
print(a)
time.sleep(10)