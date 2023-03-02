import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
# from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from torch.utils.data import DataLoader, Dataset

import matplotlib
import matplotlib.pyplot as plt

from kd import hcl

class KA():
    def __init__(self, args, my_model, ckp):
        self.args = args
        self.model = my_model
        self.ckp = ckp
        path_project = os.path.abspath('..')
        self.writer = SummaryWriter('../logs')
        self.device = 'cuda' if args.gpu else 'cpu'

    def amalgamate(self, test_dataset, local_weights):
        m = max(int(self.args.frac * self.args.num_users), 1)
        self.ka_loss = []
        print_every = 1
        self.criterion_CE = nn.NLLLoss().to(self.device)
        
        # Build Bridges

        # Dataset
        testloader = DataLoader(test_dataset, batch_size=self.args.ka_bs, shuffle=False)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
            optimizer_fc = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
            optimizer_fc = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.ka_ep):
            batch_loss = []
            # Make sure all param can be updated
            for name, param in self.model.named_parameters():
                param.requires_grad = True

            for batch_idx, (images, labels) in enumerate(testloader):
                losses = {}
                images, labels = images.to(self.device), labels.to(self.device)

                teacher_list = []
                for w in local_weights:
                    teacher_list.append(copy.deepcopy(self.model))
                    teacher_list[-1].load_state_dict(w)
                
                s_features, pred = self.model(images, is_feat = True, preact=True)

                t_features_list = []
                t_pred_list = []
                for teacher in teacher_list:
                    t_features, t_pred = teacher(images, is_feat=True, preact=True)
                    t_features_list.append(t_features)
                    t_pred_list.append(t_pred)
                
                # Calculate the score of teachers and find best
                all_logits = torch.stack(t_pred_list)
                best_model_idx = torch.argmax(all_logits[:, torch.arange(pred.shape[0]), labels], dim=0)

                # Group teachers layer by layer and select the best based on index
                best_t_features = []
                for l in range(len(t_features_list[0])):
                    t_feature = []
                    for t in range(len(t_features_list)):
                        t_feature.append(t_features_list[t][l])
                    t_feature = torch.stack(t_feature, dim=1)
                    t_feature = t_feature[torch.arange(pred.shape[0]), best_model_idx]
                    best_t_features.append(t_feature)

                feature_kd_loss = hcl(s_features, best_t_features)
                losses['review_kd_loss'] = feature_kd_loss * self.args.kd_loss_weight

                self.model.zero_grad()
                loss = sum(losses.values())
                self.ka_loss.append(loss.item())

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| KA Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images),
                        len(testloader.dataset),
                        100. * batch_idx / len(testloader), loss.item()))
            
            # Finetuning fc layer to check result
            for name, param in self.model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for batch_idx, (images, labels) in enumerate(testloader):
                losses = {}
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion_CE(log_probs, labels)
                loss.backward()
                optimizer_fc.step()

            test_acc, test_loss = test_inference(self.args, self.model, test_dataset)
            
            # print global training loss after every 'i' rounds
            if (iter+1) % print_every == 0:
                self.ckp.write_log(f' \nAvg Training Stats after {iter+1} KA rounds:')
                self.ckp.write_log(f'Training Loss : {np.mean(np.array(self.ka_loss))}')
                self.ckp.write_log("|---- Test Accuracy: {:.2f}%\n".format(100*test_acc))

            self.ckp.add_log(test_acc)
            self.ckp.save(self.model, iter, is_best=(self.ckp.log.index(max(self.ckp.log)) == iter))

    def train(self):
        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        device = 'cuda' if self.args.gpu else 'cpu'

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = get_dataset(self.args)

        # Set the model to train and send it to device.
        self.model.to(device)
        self.model.train()
        self.model.load(
            self.ckp.get_path('model'),
            pre_train=self.args.pre_train,
            resume=self.args.resume,
            cpu=self.args.cpu
        )
        self.ckp.write_log(str(self.model))

        # copy weights
        global_weights = self.model.state_dict()

        # Training
        self.train_loss, self.train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0


        # Start Local ERM
        local_weights, local_losses = [], []

        self.model.train()
        m = max(int(self.args.frac * self.args.num_users), 1)
        idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=self.args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=self.writer)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(self.model), global_round=0)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # Knowledge Amalgamation
        self.amalgamate(test_dataset, local_weights)



    def test(self):
        pass



