#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import torch
import time
import datetime
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, NamedTuple, Optional, Sequence, Tuple
from pydvl.value import compute_least_core_values, LeastCoreMode
from pydvl.utils import (
    Dataset,
)

def save_acc(args, converged_accuracy):
    # Saving the accuracy of converged federated learning training:
    file_name = '../save/{}_{}_{}_{}_iid[{}]_E[{}]_B[{}]_iterNum[{}]/acc_rmType[{}].json'.\
        format(args.trainer, args.dataset, args.model, args.epochs, args.iid,
            args.local_ep, args.local_bs, args.iter_num, args.rm_type)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    print(converged_accuracy)
    with open(file_name, 'w') as f:
        json.dump(converged_accuracy, f)


def plot_acc(args, acc_curves=[0, 1, 2]):
    curve_color = ['r', 'g', 'b']
    rm_info = {0:{'color_mean':'r', 'color_std':'lightcoral', 'label':'rmRandom'}, 1:{'color_mean':'g', 'color_std':'lightgreen', 'label':'rmHigh'}, 2:{'color_mean':'b', 'color_std':'lightblue', 'label':'rmLow'}}

    plt.figure()
    x = list(range(0, args.num_users, args.rm_step))

    for i in acc_curves:
        log_file = '../save/{}_{}_{}_{}_iid[{}]_E[{}]_B[{}]_iterNum[{}]/acc_rmType[{}].json'.\
            format(args.trainer, args.dataset, args.model, args.epochs, args.iid, args.local_ep, args.local_bs, args.iter_num, i)

        if not os.path.isfile(log_file):
            continue

        with open(log_file) as f:
            acc_list = np.array(json.load(f))
        means = np.mean(acc_list, axis=0)
        stds = np.std(acc_list, axis=0)

        plt.plot(x, means, label=rm_info[i]['label'], color=rm_info[i]['color_mean'])
        plt.fill_between(x=x, y1=means-stds, y2=means+stds, alpha=0.3, color=rm_info[i]['color_std'])

    plt.title('Comparison of Remove High, Remove Low and Remove Randomly')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Number of Excluded users')

    plt.legend()

    img_name = os.path.join(os.path.dirname(log_file), 'acc_curve.png')
    plt.savefig(img_name)


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = []
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = json.load(self.get_path('fed_log.json'))['score']
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            # os.system('rm -rf ' + self.dir)
            os.system('rm ' + os.path.join(self.dir, 'log.txt'))
            os.system('rm ' + os.path.join(self.dir, 'config.txt'))
            os.system('rm ' + os.path.join(self.dir, 'fed_log.json'))
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        os.makedirs(self.get_path('results-{}'.format(args.dataset)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, model, epoch, is_best=False):
        model.save(self.get_path('model'), epoch, is_best=is_best)
        with open(self.get_path('fed_log.json'), 'w', newline='') as jsonfile:
            json.dump({'score':self.log}, jsonfile)

    def add_log(self, score):
        self.log.append(score)

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

class Prototype_set():
    def __init__(self, clientPrototypes):
        self.prototypes = clientPrototypes
        self.indices = np.arange(len(self.prototypes))
        self.data_names = self.indices
    def __len__(self):    
        return len(self.prototypes)

class Utility_Func:
    def __init__(
        self,
        serverPrototype,
        clientPrototypes,
        client_data_nums,
    ):
        self.serverPrototype = serverPrototype
        self.data = Prototype_set(clientPrototypes)
        self.client_data_nums = client_data_nums
        
    def __call__(self, indices):
        utility: float = self._utility(indices)
        return utility

    def scorer(self, indices, metric='cosine'):
        if metric == 'cosine':
            # Merge selected clients
            weights = self.client_data_nums[np.array(indices)] / self.client_data_nums[np.array(indices)].sum()
            clientPrototype = (self.data.prototypes[indices, :] * weights[:, np.newaxis, np.newaxis]).sum(axis=0)
            # Normalization
            serverPrototype = self.serverPrototype / np.linalg.norm(self.serverPrototype, axis=1, keepdims=True)
            exist_indexes = np.linalg.norm(clientPrototype, axis=1) > 0
            clientPrototype[exist_indexes] = clientPrototype[exist_indexes] / np.linalg.norm(clientPrototype[exist_indexes], axis=1, keepdims=True)

            # Calculate score
            distMatrix = serverPrototype @ clientPrototype.T
            nearHit  = np.diagonal(distMatrix).copy()
            np.fill_diagonal(distMatrix, -1)
            nearMiss = distMatrix.max(axis=0)
            score = (nearHit - nearMiss).mean() / 4 + 0.5
            return score

    def _utility(self, indices) -> float:
        if len(indices) == 0:
            return 0.0
            
        return self.scorer(indices)


def contribution_eval(serverPrototype, clientPrototypes, client_data_nums, metric='cosine', budget=200, exact=True):
    if metric == 'cosine':
        # serverPrototype = serverPrototype / np.linalg.norm(serverPrototype, axis=1, keepdims=True)
        # for clientPrototype in clientPrototypes:
        #     # Norm the prototype to Vector
        #     exist_indexes = np.linalg.norm(clientPrototype, axis=1) > 0
        #     clientPrototype[exist_indexes] = clientPrototype[exist_indexes] / np.linalg.norm(clientPrototype[exist_indexes], axis=1, keepdims=True)
        #     distMatrix = serverPrototype @ clientPrototype.T
        #     nearHit  = np.diagonal(distMatrix).copy()
        #     np.fill_diagonal(distMatrix, -1)
        #     nearMiss = distMatrix.max(axis=0)
        #     score = (nearHit - nearMiss).mean()
        #     scores.append(score)
        # clientPrototypes = Dataset.from_arrays(
        #     clientPrototypes,
        #     np.zeros(clientPrototypes.shape[0]),
        #     train_size=0,
        # )
        utility = Utility_Func(serverPrototype, clientPrototypes, client_data_nums)
        if exact:
            values = compute_least_core_values(
                u=utility,
                mode=LeastCoreMode.Exact,
                progress=True,
            )
        else:
            values = compute_least_core_values(
                u=utility,
                mode=LeastCoreMode.MonteCarlo,
                n_iterations=budget,
                n_jobs=1,
            )
    return values.values