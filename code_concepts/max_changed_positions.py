import random

# what is maximum accumulated position change for shuffled list of elements


def diff(s: dict):
    d = 0
    for e in s:
        d += abs(s[e][-1]-s[e][-2])
    return d

num = 10
some = list(range(num))
pos = {ix: [ix] for ix in some}

diffs = []
for _ in range(100):
    random.shuffle(some)
    random.shuffle(some)
    for ix,e in enumerate(some):
        pos[e].append(ix)
    diffs.append(diff(pos))

print(max(diffs), min(diffs), diffs)
