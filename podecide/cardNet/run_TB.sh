#!/bin/bash
pkill tensorboard
nohup tensorboard --logdir="$PWD/../../_models/cardNet" >/dev/null 2>&1 &