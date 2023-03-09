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

from kd import hcl, Bridge
from ckt import CKTModule

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
        self.criterion_L1 = nn.L1Loss().to(self.device)
        self.criterion_L1_none = nn.L1Loss(reduction='none').to(self.device)
        self.softmax = torch.nn.Softmax(dim=0)
        
        # Build Bridges
        # bridges = [Bridge([6, 16]).to(self.device).train() for i in range(len(local_weights))]
        ckt_modules = nn.ModuleList([])
        for c in [10, 20]: # mnist
        # for c in [6, 16]: # cifar
            ckt_modules.append(CKTModule(channel_t=c, channel_s=c, channel_h=c//2, n_teachers=len(local_weights)).to(self.device))
        # Dataset
        testloader = DataLoader(test_dataset, batch_size=self.args.ka_bs, shuffle=False)

        # Set optimizer for the local updates
        if self.args.optimizer_ka == 'sgd':
            optimizer = torch.optim.SGD(list(self.model.parameters())+list(ckt_modules.parameters()), lr=self.args.lr_ka,
                                        momentum=0.5)
        elif self.args.optimizer_ka == 'adam':
            optimizer = torch.optim.Adam(list(self.model.parameters())+list(ckt_modules.parameters()), lr=self.args.lr_ka,
                                         weight_decay=1e-4)

        teacher_list = []
        for w in local_weights:
            teacher_list.append(copy.deepcopy(self.model).eval())
            teacher_list[-1].load_state_dict(w)

        self.model.eval()
        for iter in range(self.args.ka_ep):
            batch_loss = []
            # Make sure all param can be updated
            for name, param in self.model.named_parameters():
                param.requires_grad = True

            for batch_idx, (images, labels) in enumerate(testloader):
                losses = {}
                images, labels = images.to(self.device), labels.to(self.device)

                features_from_student, pred = self.model(images, is_feat = True, preact=True)

                t_features_list = []
                s_features_list = []
                t_pred_list = []
                for idx, teacher in enumerate(teacher_list):
                    with torch.no_grad():
                        t_features, t_pred = teacher(images, is_feat=True, preact=True)
                        t_features_list.append(t_features)
                        t_pred_list.append(t_pred)

                # Calculate the score of teachers and find best
                all_logits = torch.stack(t_pred_list) # (t, b, c)
                best_model_idx = torch.argmax(all_logits[:, torch.arange(pred.shape[0]), labels], dim=0)
                weight_t = self.softmax(all_logits[:, torch.arange(pred.shape[0]), labels]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                golden_pred = torch.stack(t_pred_list, dim=1)[torch.arange(pred.shape[0]), best_model_idx]
                features_from_teachers = []
                for layer in range(len(t_features_list[0])):
                    features_from_teachers.append([t_features_list[i][layer] for i in range(len(local_weights))])

                PFE_loss, PFV_loss = 0., 0.
                for i, (s_features, t_features) in enumerate(zip(features_from_student, features_from_teachers)):
                    t_proj_features, t_recons_features, s_proj_features = ckt_modules[i](t_features, s_features)
                    t_proj_features = torch.stack(t_proj_features)
                    feat_dist = self.criterion_L1_none(s_proj_features.unsqueeze(0).expand_as(t_proj_features), t_proj_features)
                    PFE_loss += (feat_dist * weight_t.expand_as(feat_dist)).mean()
                    PFV_loss += 0.05 * self.criterion_L1(torch.stack(t_recons_features), torch.stack(t_features))

                T_loss = self.criterion_L1(pred, golden_pred)
                loss = T_loss + PFE_loss + PFV_loss

                self.model.zero_grad()
                self.ka_loss.append(loss.item())

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| KA Epoch : {} | [{}/{} ({:.0f}%)]\tLoss_T: {:.6f} Loss_PFE: {:.6f} Loss_PFV: {:.6f}'.format(
                        iter, batch_idx * len(images),
                        len(testloader.dataset),
                        100. * batch_idx / len(testloader), T_loss.item(), PFE_loss.item(), PFV_loss.item()))
            

            test_acc, test_loss = test_inference(self.args, self.model, test_dataset)
            
            # print global training loss after every 'i' rounds
            if (iter+1) % print_every == 0:
                self.ckp.write_log(f' \nAvg Training Stats after {iter+1} KA rounds:')
                self.ckp.write_log(f'Training Loss : {np.mean(np.array(self.ka_loss))}')
                self.ckp.write_log("|---- Test Accuracy: {:.2f}%\n".format(100*test_acc))

            self.ckp.add_log(test_acc)
            self.ckp.save(self.model, iter, is_best=(self.ckp.log.index(max(self.ckp.log)) == iter))
            # self.model.train()
            # print(self.model.training)

        model_temp = copy.deepcopy(self.model)
        for idx, w in enumerate(local_weights):
            model_temp.load_state_dict(w)
            test_acc, test_loss = test_inference(self.args, model_temp, test_dataset)
            self.ckp.write_log("|---- Client {} Test Accuracy: {:.2f}%\n".format(idx, 100*test_acc))


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



