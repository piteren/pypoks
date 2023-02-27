![](pypoks_logo.png)

### Deep Reinforcement Learning (RL) with neural network (NN) agent @poker game environment with Python & TensorFlow(TF)

Machine Learining (ML) areas:
- DRL + NN to make good decisions (building the strategy)
- limited observation data and noisy input
- efficient environment (data) representation for NN & RL 
- backpropagation & high poker variance
- genetic algorithms (GA) implementation  

tech scope:
- advanced neural networks architectures @TensorFlow
- data processing with Python
- Python Multiprocessing & TensorFlow (many processes, many GPUs and a lot of data for parallel computing)
- Genetic Algorithms with TensorFlow based neural models

if you are interested in collaboration please email [me](mailto:me@piotrniewinski.com)

#
###Setup:

_To run training scripts you will need about 50 CPU cores, 80GB RAM and a single GPU system. If you are not going to use pretrained cardNet, which is optional, you will not need a GPU._

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
* Init and update ptools submodule
```
$ git submodule init
$ git submodule update
```

* You will also need tkinter for GUI (pypoks_human_game.py), please install it for python3.6

#
###Reinforcement Learning
This repo is configured for reinforcement learning of limit texas holdem poker with 3 players. You can change configuration or please contact me if you have any questions. To run reinforcement learning (NN training):

* If you want to use pretrained cardNet (sppeds-up reinforcement learning) you have to train cardNet by running:
```
$ python podecide/cardNet/cardNet_train.py
```
Training of cardNet uses single GPU and will take about 1 hour on GTX1080.
You can also skip training of cardNet. Pypoks reinforcement learning will run without it.

* Run pypoks_training.py
```
$ python pypoks_training.py
```
_In case of_ "OSError: [Errno 24] Too many open files" _You may need to increase open files limit before:_
```
$ ulimit -n 65535
```

After the training check Tensorboard (--logdir="_models") for some stats of the reinforcement process.

#
### Test
* Run human game with trained AI player
```
$ python pypoks_human_game.py
```
