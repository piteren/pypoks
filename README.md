# PYPOKS
**reinforcement training of AI agent @poker game environment with TF @python
(solve problem of making good/optimal decisions with very limited observation, many similarities to autonomous driving or stock market trading)**

project objectives:
**1st order:**
- implement reinforcement agent training with TF (@python)
- high performance training with multiprocessing
- add genetic algorithm
**2nd order:**
- prepare informational project web page
- build online demo (game interface for tests/demo)
- look for supporting entities
- optional: build HQ classifier for human/AI game decisions

by now I have build:
* simple poker environment and quite fast game algorithm
* simple: random, *hardcoded, RNN(LSTM) poker players (deciding algorithms)
* simple reinforcement procedure that works
* simple logging with console + TF.TB

but this is just beginning and there are still a lot of exciting things to do...

if you are interested in collaboration please email me: tojestprzedmalpa@gmail.com




**\*\*\*\*\* (19.07.26) \*\*\*\*\***
By now simplified poker game algorithm:
* no ante
* constant SB(2), BB(5) and table startCash(500)
* every hand starts with player cash = table.startCash
* 3 players on table
* simplified betting sizes (predefined possible sizes)

Those assumptions imply 4 possible moves: C/F, CLL, B/R, ALL

Table generates hand history (kind of dictionary).
Table player translates table history into player (perspective) history - replaces players with indexes 0..n, where 0 is always self

this implementation runs about 10H/s (F+B)


**\*\*\*\*\* (19.08.07) \*\*\*\*\***
Implemented tables as separate processes with neural DMK supporting many players.
Each DMK makes decisions with one NN(LSTM) and runs many players on many tables.
In forward DMK amkes decision for 1/3 of its players in single pass. In backward runs single batch of 1000 moves
TF runs from (single) main process.
this implementation runs about 830-960H/s (F+B) on GPU and 600H/s on CPU