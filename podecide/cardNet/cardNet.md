## cardNet

**cardNet** (CN) is a neural network responsible for representing and evaluating poker hands.
CN encodes 7 given cards (2 player and 5 table cards) into a single tensor.
This tensor stores all the information about the cards, their strength, rank, and other valuable properties.
Cards are represented with embeddings.
- CN12 (card embedding width: 12) is an 8-layer TNS encoder to an 84 width tensor.
- CN24 (card embedding width: 24) is an 8-layer TNS encoder to a 168 width tensor.

CN is trained in a supervised scheme, and then the pretrained CN is used by the DMK NN module
while training Agents with a Reinforcement Learning (RL) algorithm.

To pretrain **cardNet**, run:

```
$ python podecide/cardNet/cardNet_train.py
```
The training of cardNet uses a single GPU and will take about 2 hours on a single GTX1080.
You can also skip the training of **cardNet**; Reinforcement Learning will run without it.