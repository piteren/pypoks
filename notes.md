***
##### **TODO:**

https://pymotw.com/2/multiprocessing/communication.html - communication between processes

http://www.adaltas.com/en/2019/01/09/applying-deep-reinforcement-learning-poker/ - poker reinforcement

https://pypi.org/project/poker/ - python framework for poker related operations
https://github.com/worldveil/deuces - pure Python poker hand evaluation library

https://www.data-blogger.com/2017/11/01/pokerbot-create-your-poker-ai-bot-in-python/
https://briancaffey.github.io/2018/01/02/checking-poker-hands-with-python.html
https://medium.com/@andreasthiele/building-your-own-no-limit-texas-holdem-poker-bot-in-python-cd9919302c1c
https://www.datacamp.com/community/tutorials/python-probability-tutorial

http://www.pokerology.com/lessons/math-and-probability/

***
### **(19.07.26)**  

By now simplified poker game algorithm:
- no ante
- constant SB(2), BB(5) and table startCash(500)
- every hand starts with player cash = table.startCash
- 3 players on table
- simplified betting sizes (predefined possible sizes)

Those assumptions imply 4 possible moves: C/F, CLL, B/R, ALL

Table generates hand history (kind of dictionary).
Table player translates table history into player (perspective) history - replaces players with indexes 0..n, where 0 is always self

this implementation runs about 10H/s (F+B)

***
### **(19.08.07)**  

Implemented tables as separate processes with neural DMK supporting many players.
Each DMK makes decisions with one NN(LSTM) and runs many players on many tables.
In forward DMK makes decision for 1/3 of its players in single pass. In backward runs single batch of 1000 moves
TF runs from (single) main process.
this implementation runs about 830-960H/s (F+B) on GPU and 600H/s on CPU

***
### **(19.09.02)**

Implemented Card Network.
CN is a FF NN build to calculate cards embeddings and compute efficient hands representations.
Batch preparation is implemented with multiprocessing and builds balanced (among hands ranks) batches.
Every hand (7 cards) is processed with dense network to build cards representation.
2 representations are concatenated and classified with single layer for winner.

Problems:
 - high gradients
 - lack of final convergence

TODO:
 - warmup
 - histograms
 - rank classifier