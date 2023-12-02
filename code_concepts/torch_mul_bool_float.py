import torch

ta = torch.rand(15)
tb = ta > 0.2
print(ta)
print(tb)
print(ta * tb)