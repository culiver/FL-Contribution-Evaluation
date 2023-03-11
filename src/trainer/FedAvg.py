import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from options import args_parser
from update import LocalUpdate, test_inference
# from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

import matplotlib
import matplotlib.pyplot as plt
import json

class FedAvg():
    def __init__(self, args, my_model, ckp):
        self.args = args
        self.model = copy.deepcopy(my_model)
        self.init_model = copy.deepcopy(my_model)
        self.ckp = ckp
        path_project = os.path.abspath('..')
        self.writer = SummaryWriter('../logs')

    def computeServerPrototype(self, num_class=10):
        """ Returns prototype.
        """
        new_model = self.model
        new_model.eval()

        class_features = {}
        label_nums = {}
        prototypes = []

        device = 'cuda' if self.args.gpu else 'cpu'
        testloader = DataLoader(self.test_dataset, batch_size=128,
                                shuffle=False)

        numData = torch.tensor(len(testloader.dataset))

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            features = new_model(images, is_feat=True, preact=True)[0][-1]
            features = features.view(features.size(0), -1)
            
            for i in range(features.shape[0]):
                label = labels[i].item()
                feature_vector = features[i].detach().cpu().numpy()
                feat_size = feature_vector.shape
                class_features[label] = class_features.get(label, np.zeros(feat_size)) + feature_vector
                label_nums[label] = label_nums.get(label, 0) + 1

        # Compute the mean feature vector for each class
        for label in range(num_class):
            if label in class_features:
                prototypes.append(class_features[label] / label_nums[label])
            else:
                prototypes.append(np.zeros(feat_size))

        torch.cuda.empty_cache()

        return np.array(prototypes)
    
    def contribution_eval(self, serverPrototype, clientPrototypes, metric='cosine'):
        scores = []
        if metric == 'cosine':
            serverPrototype = serverPrototype / np.linalg.norm(serverPrototype, axis=1, keepdims=True)
            for clientPrototype in clientPrototypes:
                exist_indexes = np.linalg.norm(clientPrototype, axis=1) > 0
                clientPrototype[exist_indexes] = clientPrototype[exist_indexes] / np.linalg.norm(clientPrototype[exist_indexes], axis=1, keepdims=True)
                distMatrix = serverPrototype @ clientPrototype.T
                nearHit  = np.diagonal(distMatrix).mean()
                np.fill_diagonal(distMatrix, 0)
                nearMiss = distMatrix.max(axis=1)
                score = (nearHit - nearMiss).mean()
                scores.append(score)
        return np.array(scores)

    def train_remove_clients(self, exclude_nums, scores, rm_high):
        print('='*20)
        print('Start remove clients!')
        print('='*20)
        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        device = 'cuda' if self.args.gpu else 'cpu'

        for exclude_num in exclude_nums:
            remain_num = self.args.num_users - exclude_num
            print("Number of clients: {}".format(remain_num))
            if rm_high:
                idxs_users = np.argpartition(scores, remain_num)[:remain_num]
            else:
                idxs_users = np.argpartition(scores, -remain_num)[-remain_num:]

            self.model = copy.deepcopy(self.init_model)
            self.model.to(device)
            self.model.train()
            # copy weights
            global_weights = self.model.state_dict()

            for epoch in tqdm(range(self.args.epochs)):
                local_weights, local_losses = [], []
                self.ckp.write_log(f'\n | Global Training Round : {epoch+1} |\n')

                self.model.train()
                m = max(int(self.args.frac * self.args.num_users), 1)

                for idx in idxs_users:
                    local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                            idxs=self.user_groups[idx], logger=self.writer)
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(self.model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))

                # update global weights
                global_weights = average_weights(local_weights)

                # update global weights
                self.model.load_state_dict(global_weights)

                loss_avg = sum(local_losses) / len(local_losses)

            # Test inference after completion of training
            test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)
            self.converged_accuracy[-1].append(test_acc)
            self.ckp.write_log("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    def train(self, iter_num=1):
        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        device = 'cuda' if self.args.gpu else 'cpu'
        self.converged_accuracy = []
        self.ckp.write_log(str(self.model))

        for iter in range(iter_num):
            print('='*20)
            print('Start training!')
            print('='*20)
            self.converged_accuracy.append([])
            # load dataset and user groups
            self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)

            # Set the model to train and send it to device.
            self.model = copy.deepcopy(self.init_model)
            self.model.to(device)
            self.model.train()
            self.model.load(
                self.ckp.get_path('model'),
                pre_train=self.args.pre_train,
                resume=self.args.resume,
                cpu=self.args.cpu
            )

            # copy weights
            global_weights = self.model.state_dict()

            # Training
            self.train_loss, self.train_accuracy = [], []
            val_acc_list, net_list = [], []
            cv_loss, cv_acc = [], []
            print_every = 2
            val_loss_pre, counter = 0, 0

            # epoch_start = len(self.ckp.log)

            for epoch in tqdm(range(self.args.epochs)):
                local_weights, local_losses = [], []
                self.ckp.write_log(f'\n | Global Training Round : {epoch+1} |\n')

                self.model.train()
                m = max(int(self.args.frac * self.args.num_users), 1)
                idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

                for idx in idxs_users:
                    local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                            idxs=self.user_groups[idx], logger=self.writer)
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(self.model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))

                # update global weights
                global_weights = average_weights(local_weights)

                # update global weights
                self.model.load_state_dict(global_weights)

                loss_avg = sum(local_losses) / len(local_losses)
                self.train_loss.append(loss_avg)

                # Calculate avg training accuracy over all users at every epoch
                list_acc, list_loss = [], []
                self.model.eval()
                for idx in range(self.args.num_users):
                    local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                            idxs=self.user_groups[idx], logger=self.writer)
                    acc, loss = local_model.inference(model=self.model)
                    list_acc.append(acc)
                    list_loss.append(loss)
                self.train_accuracy.append(sum(list_acc)/len(list_acc))

                test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)

                # print global training loss after every 'i' rounds
                if (epoch+1) % print_every == 0:
                    self.ckp.write_log(f' \nAvg Training Stats after {epoch+1} global rounds:')
                    self.ckp.write_log(f'Training Loss : {np.mean(np.array(self.train_loss))}')
                    self.ckp.write_log('Train Accuracy: {:.2f}%'.format(100*self.train_accuracy[-1]))
                    self.ckp.write_log("|---- Test Accuracy: {:.2f} \n%".format(100*test_acc))

                self.ckp.add_log(test_acc)
                self.ckp.save(self.model, epoch, is_best=(self.ckp.log.index(max(self.ckp.log)) == epoch))

            # Test inference after completion of training
            test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)
            
            self.converged_accuracy[-1].append(test_acc)

            self.ckp.write_log(f' \n Results after {self.args.epochs} global rounds of training:')
            self.ckp.write_log("|---- Avg Train Accuracy: {:.2f}%".format(100*self.train_accuracy[-1]))
            self.ckp.write_log("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Saving the objects train_loss and train_accuracy:
            file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
                format(self.args.dataset, self.args.model, self.args.epochs, self.args.frac, self.args.iid,
                    self.args.local_ep, self.args.local_bs)

            with open(file_name, 'wb') as f:
                pickle.dump([self.train_loss, self.train_accuracy], f)
        

            if self.args.rm_type != 0:
                serverPrototype = self.computeServerPrototype()
                clientPrototypes = []

                for idx in range(self.args.num_users):
                    local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                            idxs=self.user_groups[idx], logger=self.writer)
                    prototype = local_model.computeLocalPrototype(model=self.model)
                    clientPrototypes.append(prototype)
                
                self.contributions = self.contribution_eval(serverPrototype, clientPrototypes)
            else:
                self.contributions = np.zeros(self.args.num_users)

            if self.args.rm_type == 1:
                self.train_remove_clients(exclude_nums=[i for i in range(self.args.rm_step,self.args.num_users,self.args.rm_step)], scores=self.contributions, rm_high=True)
            else:
                self.train_remove_clients(exclude_nums=[i for i in range(self.args.rm_step,self.args.num_users,self.args.rm_step)], scores=self.contributions, rm_high=False)

        # Saving the accuracy of converged federated learning training:
        file_name = '../save/objects/Acc_{}_{}_{}_iid[{}]_E[{}]_B[{}]_rmType[{}].json'.\
            format(self.args.dataset, self.args.model, self.args.epochs, self.args.iid,
                self.args.local_ep, self.args.local_bs, self.args.rm_type)
        print(self.converged_accuracy)
        with open(file_name, 'w') as f:
            json.dump(self.converged_accuracy, f)


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

