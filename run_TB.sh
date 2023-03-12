#!/bin/bash
pkill tensorboard
nohup tensorboard --logdir="$PWD/_models/dmk" >/dev/null 2>&1 &