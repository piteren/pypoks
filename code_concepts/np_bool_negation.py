import numpy as np

probs = [0.1, 0.2, 0.4, 0.3]

allowed = [False, True, True, False]
allowed_arr = np.asarray(allowed)
print(allowed_arr)

not_allowed_arr = ~allowed_arr
print(not_allowed_arr)

probs_arr = np.asarray(probs)
print(probs_arr)

print(sum(probs_arr * not_allowed_arr))