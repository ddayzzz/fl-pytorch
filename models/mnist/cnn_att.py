# -*- coding: utf-8 -*-
from itertools import chain
import copy
import torch
from torch import nn
from models.mnist.cnn import Model as CNNModel


class FeatureFuse(nn.Module):
    """ Activation attention Layer"""
    def __init__(self, policy):
        super().__init__()
        if policy == 'multi':
            self.fuse = nn.Parameter(torch.zeros(1, 64, 1, 1))
        elif policy == 'single':
            self.fuse = nn.Parameter(torch.zeros(1))
        elif policy == 'conv':
            self.fuse = nn.Conv2d(64 * 2, 64, kernel_size=1)
        else:
            raise ValueError('Invalid attention policy.')

        self.policy = policy

    def forward(self, x, y):
        """
            inputs :
                x, y: input feature maps (B X C X W X H)
                x from the local model, y from the global one
            returns :
                out : fused feature map
        """
        if self.policy in ['multi', 'single']:
            out = self.fuse * y + (1 - self.fuse) * x
        else:
            out = torch.cat((x, y), dim=1)
            out = self.fuse(out)

        return out


class Model(CNNModel):
    '''MNIST model with attention components'''
    def __init__(self, image_size, options):
        '''
            policy: which attention component to use.
        '''
        super().__init__(image_size, options)
        # 注意力网络
        operator = options['operator']
        self.latest_global_feature = copy.deepcopy(self.features)
        for p in self.latest_global_feature.parameters():
            p.requires_grad = False
        self.attn = FeatureFuse(policy=operator)

    def set_global_model(self, model: nn.Module, device):
        """
        将当前的全局模型复制一份,
        :param model:
        :return:
        """
        global_feature = copy.deepcopy(model.features).to(device)
        for p in global_feature.parameters():
            p.requires_grad = False
        self.latest_global_feature = global_feature

    def forward(self, x):
        x = x.view((x.size(0), 1, self.image_size, self.image_size))
        with torch.no_grad():
            global_feature = self.latest_global_feature(x)
        x = self.features(x)
        out = self.attn(x, global_feature)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
