from torchness.grad_clipping import clip_grad_norm_
import torch



vec = torch.rand(100)
grad = torch.rand(100)
grad_orig = torch.clone(grad)
vec.grad = grad
print(grad)
print(grad.norm())

norm = clip_grad_norm_(vec, 1)

print(norm)
print(grad)
print(grad.norm())

print(grad_orig / grad)

