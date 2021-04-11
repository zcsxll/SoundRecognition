#!/bin/bash

if [ $# != 1 ]; then
    echo "usage: $0 <log dir>"
    exit
fi

tensorboard --logdir $1 --port 1417
