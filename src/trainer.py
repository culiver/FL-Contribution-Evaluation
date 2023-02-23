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

import matplotlib
import matplotlib.pyplot as plt

class FedAvg():
    def __init__(self, args, ckp):
        self.args = args
        self.ckp = ckp
        path_project = os.path.abspath('..')
        self.writer = SummaryWriter('../logs')

    def train(self):
        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        device = 'cuda' if self.args.gpu else 'cpu'

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = get_dataset(self.args)

        # BUILD MODEL
        if self.args.model == 'cnn':
            # Convolutional neural netork
            if self.args.dataset == 'mnist':
                global_model = CNNMnist(args=self.args)
            elif self.args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist(args=self.args)
            elif self.args.dataset == 'cifar':
                global_model = CNNCifar(args=self.args)

        elif self.args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,
                                dim_out=self.args.num_classes)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        self.ckp.write_log(str(global_model))

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        self.train_loss, self.train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0

        for epoch in tqdm(range(self.args.epochs)):
            local_weights, local_losses = [], []
            self.ckp.write_log(f'\n | Global Training Round : {epoch+1} |\n')

            global_model.train()
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

            for idx in idxs_users:
                local_model = LocalUpdate(args=self.args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=self.writer)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            self.train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(self.args.num_users):
                local_model = LocalUpdate(args=self.args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=self.writer)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            self.train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                self.ckp.write_log(f' \nAvg Training Stats after {epoch+1} global rounds:')
                self.ckp.write_log(f'Training Loss : {np.mean(np.array(self.train_loss))}')
                self.ckp.write_log('Train Accuracy: {:.2f}% \n'.format(100*self.train_accuracy[-1]))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(self.args, global_model, test_dataset)

        self.ckp.write_log(f' \n Results after {self.args.epochs} global rounds of training:')
        self.ckp.write_log("|---- Avg Train Accuracy: {:.2f}%".format(100*self.train_accuracy[-1]))
        self.ckp.write_log("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        # Saving the objects train_loss and train_accuracy:
        file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(self.args.dataset, self.args.model, self.args.epochs, self.args.frac, self.args.iid,
                self.args.local_ep, self.args.local_bs)

        with open(file_name, 'wb') as f:
            pickle.dump([self.train_loss, self.train_accuracy], f)

    def test(self):
        pass

    def plot(self):
        # PLOTTING (optional)
        matplotlib.use('Agg')

        # Plot Loss curve
        plt.figure()
        plt.title('Training Loss vs Communication rounds')
        plt.plot(range(len(self.train_loss)), self.train_loss, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                    format(self.args.dataset, self.args.model, self.args.epochs, self.args.frac,
                           self.args.iid, self.args.local_ep, self.args.local_bs))
        
        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title('Average Accuracy vs Communication rounds')
        plt.plot(range(len(self.train_accuracy)), self.train_accuracy, color='k')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')
        plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                    format(self.args.dataset, self.args.model, self.args.epochs, self.args.frac,
                           self.args.iid, self.args.local_ep, self.args.local_bs))

    # def terminate(self):
    #     if self.args.test_only:
    #         self.test()
    #         return True
    #     else:
    #         epoch = self.optimizer.get_last_epoch() + 1
    #         return epoch >= self.args.epochs

