import numpy as np

rng = np.random.default_rng()

n = 4
probs = [0.8, 0.1, 0.05, 0.05]
print(sum(probs))

nc = {ix:0 for ix in range(n)}
n_samples = 100
for _ in range(n_samples):
    choice = rng.choice(n, p=probs)
    nc[choice] += 1


for k in nc:
    print(f'{k}: {nc[k]} {nc[k]/n_samples:.4f}')