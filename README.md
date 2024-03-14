![](images/pypoks_logo.png)

## Deep Reinforcement Learning (DRL) with Neural Network (NN) based Agent at NL Texas Holdem Poker game environment with Python & PyTorch

### research scope of the project (ML/RL):
- testbed for different RL concepts like PG, A3C, PPO and their modifications
- efficient NN Agent (PyTorch based) architecture details
- asynchronous self-play: multi-GPU, many sub-processes, hundreds of tables at once
- efficient environment events (data) representation (multiplayer, many bets)
- efficient process (and sub-processes) monitoring
- Genetic Algorithms (GA) for policies (with PyTorch)
- high (poker) variance & backpropagation
- high (poker) variance & policy evaluation

![](images/pypoks_ques.png)


### How to read the docs

In some sub-folders there are separate readmies (.md), please follow them for the more detailed concepts
of the code from the sub-folders. 

---
### Setup

The project may be set up with python=<3.11. Install requirements from ```requirements.txt```

To run training scripts you will need about 50 CPU cores, 120GB RAM and 2x GPU system.<br>
You may just play `run_human_game.py` with trained agents downloaded from [here](https://drive.google.com/file/d/1e4QEdch2SVgloQjSNzftAohn_Y_lji-U/view?usp=sharing)
To play human game with agents you will also need `tkinter` for GUI, please install it.
For instructions how to install tkinter for python 3.11 please go to gui/tkinter folder.


### Training

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

Allowed moves are defined in ```/game_configs``` yaml file.

While playing, a debug of a game is logged to the terminal - you always can check cards played by each agent.

![](images/terminal_HDMK.png)
