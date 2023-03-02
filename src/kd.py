#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class Bridge(nn.Module):
    def __init__(self, channels=[]):
        super(Bridge, self).__init__()

        t_models = []
        s_models = []

        for c in channels:
            t_models.append(nn.Conv2d(c, c, 1))
            s_models.append(nn.Conv2d(c, c, 1))

        self.t_models = nn.ModuleList(t_models)
        self.s_models = nn.ModuleList(s_models)

    def forward(self, feat, type='s'):
        
        models = self.s_models if type == 's' else self.t_models
        output = []
        for i in range(len(feat)):
            output.append(models[i](feat[i]))
        
        return output

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all