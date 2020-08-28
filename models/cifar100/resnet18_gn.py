from models.utils.resnet_gn import resnet18
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, options, group_norm=2):
        super(Model, self).__init__()
        # 参照TTF 中的设定, group 为2
        self.resnet = resnet18(pretrained=False, num_classes=100, group_norm=group_norm)

    def forward(self, x):
        x = self.resnet(x)
        return x


