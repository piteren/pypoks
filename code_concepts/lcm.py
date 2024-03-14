import math

n_all_players = 1000 * 2
multiplier = math.lcm(2,3)
print(multiplier)

n_all_players += multiplier - n_all_players % multiplier
print(n_all_players)
print(n_all_players / multiplier)