## cardNet

cardNet (CN) is a NN responsible for poker cards (hands) evaluation. CN encodes  7 given cards (2 player and 5 table cards) into one tensor. This tensor stores all the information about cards, its strength, rank and other valuable properties. Cards are represented with embeddings.
- CN12 (card embedding width: 12) is 8 layer TNS encoder to 84 width tensor 
- CN24 (card embedding width: 24) is 8 layer TNS encoder to 168 width tensor

CN is trained in supervised training scheme, and then pretrained CN is used by DMK NN module while training agents with reinforcement learining (RL).