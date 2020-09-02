from models.utils.resnet_gn import resnet18
from torch import nn


class Model(nn.Module):

    def __init__(self, options, group_norm=2):
        super(Model, self).__init__()
        # 参照TTF 中的设定, group 为2
        self.resnet = resnet18(pretrained=False, num_classes=100, num_channels_per_group=group_norm)

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__ == '__main__':
    model = Model(None).cuda()
    s = model.state_dict()
    from torchsummary import summary
    summary(model, (3, 24, 24))


