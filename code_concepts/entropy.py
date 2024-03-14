import numpy as np

for w in range(3,15):
    probs = np.asarray([1/w]*w)
    entropy = (-probs * np.log2(probs)).sum()
    perplexity = 2**entropy
    print(f'{w:2} {entropy:.2f} {perplexity:.2f}')
