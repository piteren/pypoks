![](pypoks_logo.png)

###Deep Reinforcement Learning (RL) with neural network (NN) agent @poker game environment with Python & TensorFlow(TF)

Machine Learining (ML) areas:
- DRL + NN to make good decisions (building the strategy)
- limited observation data and noisy input
- efficient environment (data) representation for NN & RL 
- backpropagation & high poker variance
- genetic algorithms (GA) implementation  

tech scope:
- advanced neural networks architectures @TensorFlow
- data processing with Python
- Python Multiprocessing & TF (many processes, many GPUs and a lot of data for parallel computing)
- GA with TF

if you are interested in collaboration please email [me](mailto:me@piotrniewinski.com)

#
###Setup:


* Create virtualenv with python 3.6
```
$ virtualenv -p python3.6 venv
```
* Activate it
```
$ source venv/bin/activate
```
* Install requirements
```
$ pip install -r requirements.txt
```
* Install spacy en model
```
$ python -m spacy download en
```

* You will also need tkinter for GUI (pypoks_human_game.py), please install it for python3.6

#
###Reinforcement Learning
To run reinforcement learning (NN training):

* If you want to use pretrained cardNet (sppeds-up reinforcement learning) you have to train cardNet by running:
```
$ python podecide/cardNet/cardNet_train.py
```
cardNet training uses single GPU.
You can also skip traing cardNet. Pypoks reinforcement learning will run without it.

* Run pypoks_training.py, the process is configured for about 50 cores and 100GB RAM (no GPU needed)
```
$ python pypoks_training.py
```
Check Tensorboard (--logdir="_models") for some stats of training and poker game.

#
### Test
* Run human game with trained AI player
```
$ python pypoks_human_game.py
```