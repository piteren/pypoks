import torch

logits = (torch.rand(2) - 0.5) * 10 # unnormalized log probabilities (NN output)
print(f'NN logits:  {logits}')

dist = torch.distributions.Categorical(logits=logits)
print(dist)
print(f'logits:     {dist.logits}')
logits_n = logits - logits.logsumexp(dim=-1, keepdim=True)
print(f'logits_n:   {logits_n}')
print(f'probs:      {dist.probs}')
logits2 = torch.log2(dist.probs)
print(f'logits2:    {logits2}')
print(f'entropy     {dist.entropy()}')
entropy_ = (-dist.probs * dist.logits).sum()
print(f'entropy_    {entropy_}')
print(f'perplexity: {dist.perplexity()}')