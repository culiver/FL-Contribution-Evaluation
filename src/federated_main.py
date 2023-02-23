#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
import utils

from trainer import FedAvg

def main():
    start_time = time.time()
    args = args_parser()
    exp_details(args)

    checkpoint = utils.checkpoint(args)

    if args.trainer == 'fedavg':
        t = FedAvg(args, checkpoint)
        t.train()


    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

if __name__ == '__main__':
    main()
    
