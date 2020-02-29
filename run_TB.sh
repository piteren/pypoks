#!/bin/bash
pkill tensorboard
nohup tensorboard --logdir="$PWD/_models" &
