import os
import copy
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details, save_acc
from utils import contribution_eval

class Central():
    def __init__(self, args, my_model, ckp):
        self.args = args
        self.model = copy.deepcopy(my_model)
        self.init_model = copy.deepcopy(my_model)
        self.ckp = ckp
        path_project = os.path.abspath('..')
        self.writer = SummaryWriter('../logs')
        self.device = 'cuda' if args.gpu else 'cpu'

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
            if remain_num == self.args.num_users:
                idxs_users = np.arange(self.args.num_users)
            elif rm_high:
                idxs_users = np.argpartition(scores, remain_num)[:remain_num]
            else:
                idxs_users = np.argpartition(scores, -remain_num)[-remain_num:]

            self.model = copy.deepcopy(self.init_model)
            self.model.to(device)
            self.model.train()
            # copy weights
            global_weights = self.model.state_dict()

            for epoch in range(self.args.epochs):
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

            save_acc(self.args, self.converged_accuracy)

    def get_contributions(self):
        if self.args.rm_type != 0:
            serverPrototype = self.computeServerPrototype()
            clientPrototypes = []
            client_data_nums = []
            for idx in range(self.args.num_users):
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                        idxs=self.user_groups[idx], logger=self.writer)
                prototype = local_model.computeLocalPrototype(model=self.model)
                clientPrototypes.append(prototype)
                client_data_nums.append(len(self.user_groups[idx]))
            
            self.contributions = contribution_eval(serverPrototype, np.array(clientPrototypes), np.array(client_data_nums))
        else:
            self.contributions = np.zeros(self.args.num_users)