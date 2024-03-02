![](images/pypoks_logo.png)

## Deep Reinforcement Learning (DRL) with neural network (NN) agent @poker game environment with Python & PyTorch

### Machine Learning areas:
- DRL (NN Agent) to make good decisions (building the strategy)
- limited observation data, noisy input
- efficient environment (data) representation for NN & RL 
- backpropagation & high poker variance
- genetic algorithms (GA) implementation  

### Tech scope:
- advanced neural networks architectures @PyTorch
- data processing with Python
- Python Multiprocessing & PyTorch (many processes, many GPUs and a lot of data for parallel computing)
- GA with PyTorch

![](images/pypoks_ques.png)

---

### Setup

The project may be set up with python=<3.11. Install requirements from ```requirements.txt```

To run training scripts you will need about 50 CPU cores, 120GB RAM and 2x GPU system.<br>
You may just play `run_human_game.py` with trained agents downloaded from [here](https://drive.google.com/file/d/1e4QEdch2SVgloQjSNzftAohn_Y_lji-U/view?usp=sharing)
To play human game with agents you will also need `tkinter` for GUI, please install it.
For instructions how to install tkinter for python 3.11 please go to gui/tkinter folder.


### Training
This code is preconfigured for reinforcement learning of limit texas holdem poker.<br>

Poker agent - **Decision Maker (DMK)** - consist of part responsible for preparation of card representations - **cardNet**.<br>
**cardNet** may be optionally pretrained with supervised learning. To pretrain **cardNet** run:

```
$ python podecide/cardNet/cardNet_train.py
```
Training of cardNet uses single GPU and will take about 1 hour on GTX1080.
You can also skip training of **cardNet**, reinforcement learning will run without it.

To train poker agent **(DMK)** from a scratch run:

```
$ python run/run_train_loop.py
```

This script will train a set of agents with RL self-play. Script is preconfigured with many options that will fit for system with 2x GPUs (11GB).
Trained agents available to download with a link above took about 5 days to train.<br>

While training, you may check the progress with TensorBoard (run `run_TB.sh`)

![](images/pypoksTB.png)

In case of `OSError: [Errno 24] Too many open files` You may need to increase open files limit: `$ ulimit -n 65535`

### Human Game - playing with trained agents

To play a game with trained agents:
```
$ python run/run_human_game.py
```
![](images/pypoks_HDMK.png)

Allowed moves are defined with table_moves in game_config yaml file.

While playing debug of a game is logged to terminal. You can always check what cards played each agent.

![](images/terminal_HDMK.png)
