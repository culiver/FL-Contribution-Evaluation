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

# def plot_acc(args):
#     file_names = ['../save/objects/Acc_{}_{}_{}_iid[{}]_E[{}]_B[{}]_rmType[{}].json'.\
#             format(args.dataset, args.model, args.epochs, args.iid,
#                 args.local_ep, args.local_bs, i) for i in range(2)]

#     data_random, data_high, data_low = [], [], []

#     for i in range(10):

#         file_name = path + 'random_{}.pkl'.format(i)
#         with open(file_name, 'rb') as f:
#             train_accuracy = pickle.load(f)
#             # train_loss, train_accuracy, idxs_exluded_users, runtime, score = pickle.load(f)
#         data_random.append(train_accuracy)

#         file_name = path + 'low_{}.pkl'.format(i)
#         with open(file_name, 'rb') as f:
#             train_accuracy = pickle.load(f)
#             # train_loss, train_accuracy, idxs_exluded_users, runtime, score = pickle.load(f)
#         data_low.append(train_accuracy)


#         file_name = path + 'high_{}.pkl'.format(i)
#         with open(file_name, 'rb') as f:
#             train_accuracy = pickle.load(f)
#             # train_loss, train_accuracy, idxs_exluded_users, runtime, score = pickle.load(f)
#         data_high.append(train_accuracy)


#     means_random = np.mean(data_random, axis=0)
#     stds_random = np.std(data_random, axis=0)

#     means_low = np.mean(data_low, axis=0)
#     stds_low = np.std(data_low, axis=0)

#     means_high = np.mean(data_high, axis=0)
#     stds_high = np.std(data_high, axis=0)

#     x = list(range(0,10))

#     plt.figure()

#     plt.plot(x,means_random,label='Remove randomly', color = 'r')
#     plt.plot(x,means_high,label='Remove High', color = 'g')
#     plt.plot(x,means_low,label='Remove Low', color = 'b')

#     plt.fill_between(x=x, y1=means_random-stds_random, y2=means_random+stds_random, alpha=0.2, color="lightcoral")
#     plt.fill_between(x=x, y1=means_high-stds_high, y2=means_high+stds_high, alpha=0.2, color="lightgreen")
#     plt.fill_between(x=x, y1=means_low-stds_low, y2=means_low+stds_low, alpha=0.2, color="lightblue")

#     plt.title('Comparison of Remove High, Remove Low and Remove Randomly')
#     plt.ylabel('Average Accuracy')
#     plt.xlabel('Number of Excluded users')

#     plt.legend()

#     plt.show()

#     plt.savefig(path+'.png')


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
