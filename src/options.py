#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # Aggregate Mode
    parser.add_argument('--trainer', type=str, default='fedavg', help='trainer name')

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--iter_num', type=int, default=10,
                        help="number of data distributions")
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_ka', type=float, default=2e-4,
                        help='learning rate of ka')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--rm_type', type=int, default=0,
                        help='remove by what (default: 0 is random, 1 is remove high, 2 is remove low)')
    parser.add_argument('--rm_step', type=int, default=5,
                        help='remove by what (default: 0 is random, 1 is remove high, 2 is remove low)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    
    parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu_id', type=int, default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpu', type=bool, default=True, help="To use cuda")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--optimizer_ka', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # Log specifications
    parser.add_argument('--reset', action='store_true',
                    help='reset the training')
    parser.add_argument('--save', type=str, default='test',
                        help='file name to save')
    parser.add_argument('--load', type=str, default='',
                        help='file name to load')
    parser.add_argument('--resume', type=int, default=0,
                        help='resume from specific checkpoint')
    parser.add_argument('--save_models', action='store_true',
                        help='save all intermediate models')
    parser.add_argument('--print_every', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_results', action='store_true',
                        help='save output results')
    parser.add_argument('--save_gt', action='store_true',
                        help='save low-resolution and high-resolution images together')
    parser.add_argument('--plot_only', action='store_true',
                        help='only use log to plot')

    # Knowledge amalgamation
    parser.add_argument('--ka_bs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--ka_ep', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--kd_loss_weight', type=float, default=1,
                        help="review kd loss weight")
                        

    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=20,
                        help='number of threads for data loading')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=1,
                        help='number of GPUs')
    # parser.add_argument('--seed', type=int, default=1,
                        # help='random seed')

    args = parser.parse_args()
    return args
