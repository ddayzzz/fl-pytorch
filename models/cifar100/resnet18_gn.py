from models.utils.resnet_gn import resnet18
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, options):
        super(Model, self).__init__()
        group_norm = 2
        """
        tfa.layers.GroupNormalization(
    groups: int = 2,
    axis: int = -1,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    beta_initializer: tfa.rnn.cell.Constraint = 'zeros',
    gamma_initializer: tfa.rnn.cell.Constraint = 'ones',
    beta_regularizer: tfa.rnn.cell.Constraint = None,
    gamma_regularizer: tfa.rnn.cell.Constraint = None,
    beta_constraint: tfa.rnn.cell.Constraint = None,
    gamma_constraint: tfa.rnn.cell.Constraint = None,
    **kwargs
)
        """
        # 参照TTF 中的设定, group 为2(tff中只修改axis)
        self.resnet = resnet18(pretrained=False, num_classes=100, group_norm=group_norm)

    def forward(self, x):
        x = self.resnet(x)
        return x


