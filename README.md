![](images/pypoks_logo.png)

### Deep Reinforcement Learning (DRL) with neural network (NN) agent @poker game environment with Python & PyTorch

Machine Learning areas:
- DRL (NN Agent) to make good decisions (building the strategy)
- limited observation data, noisy input
- efficient environment (data) representation for NN & RL 
- backpropagation & high poker variance
- genetic algorithms (GA) implementation  

tech scope:
- advanced neural networks architectures @PyTorch
- data processing with Python
- Python Multiprocessing & PyTorch (many processes, many GPUs and a lot of data for parallel computing)
- GA with PyTorch

---

### Setup

To run training scripts you will need about 50 CPU cores, 120GB RAM and 2x GPU system.<br>
You may just play `run_human_game.py` with pretrained agents downloaded from here:<br>
https://drive.google.com/file/d/1QPW_wA-hX0YQUy4PNNFak1vu1jQ0k_cC/view?usp=share_link - to be extracted in project main folder.<br>
To play human game with agents you will also need `tkinter` for GUI, please install it.


### Training
This code is preconfigured for reinforcement learning of limit texas holdem poker with 3 players.<br>
To train **cardNet** - supervised learning of poker hands model - run:

```
$ python podecide/cardNet/cardNet_train.py
```
Training of cardNet uses single GPU and will take about 1 hour on GTX1080.
You can also skip training of cardNet, **pypoks** reinforcement learning will run without it.

To train poker agents from a scratch run:

```
$ python run/run_pretrain.py
```
and next:
```
$ python run/run_train_loop_V2.py
```

Those scripts are preconfigured with many options that will fit for system with 2x GPUs (11GB).
Pretrained agents (available to download with a link above) took about 30 hours to train.<br>

While training, you may check the progress with TensorBoard (run `run_TB.sh`)

![](images/pypoksTB.png)

In case of `OSError: [Errno 24] Too many open files` You may need to increase open files limit: `$ ulimit -n 65535`

### Play with pretrained agents

To play a game with pretrained AI players:
```
$ python run/run_human_game.py
```
![](images/pypoks_HDMK.png)

There are 4 actions possible:
- C/F - check / fold
- CLL - call
- BRS - bet or raise (small)
- BRL - bet or raise (large)

While playing debug of a game is logged to terminal. You can always check what cards played each agent.

![](images/terminal_HDMK.png)