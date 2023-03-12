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
# from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, plot_acc
import utils

from trainer.FedAvg import FedAvg
from trainer.KA import KA
import models
import json

def main():
    start_time = time.time()
    args = args_parser()
    exp_details(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = utils.checkpoint(args)

    os.makedirs('../save/objects', exist_ok=True)

    if not args.plot_only:
        if args.trainer == 'fedavg':
            _model = models.Model(args, checkpoint)
            t = FedAvg(args, _model, checkpoint)
            t.train()
        if args.trainer == 'ka':
            _model = models.Model(args, checkpoint)
            t = KA(args, _model, checkpoint)
            t.train()

    plot_acc(args)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

if __name__ == '__main__':
    main()
    
