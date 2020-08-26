from models.utils.resnet import resnet56
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, options):
        super(Model, self).__init__()
        self.resnet = resnet56(pretrained=False, class_num=100)

    def forward(self, x):
        x = self.resnet(x)
        return x


