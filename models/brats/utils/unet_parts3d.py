# 作者：玖零猴
# 链接：https://zhuanlan.zhihu.com/p/138463933
# 来源：知乎
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

from torch import nn
from torch import cat


def batch_norm3d(in_dim):
    return nn.BatchNorm3d(in_dim, track_running_stats=False, affine=True)


class DoubleConv3D(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(DoubleConv3D, self).__init__()
        #
        inter_channels = out_channels if in_channels > out_channels else out_channels //2

        layers = [
            nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        if batch_norm:
            layers.insert(1, batch_norm3d(inter_channels))
            layers.insert(len(layers)-1, batch_norm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)


class Down3D(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Down3D, self).__init__()
        self.pub = DoubleConv3D(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        return x, self.pool(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(Up3D, self).__init__()
        self.pub = DoubleConv3D(in_channels // 2  +in_channels, out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        #c1 = (x1.size(2) - x.size(2)) // 2
        #c2 = (x1.size(3) - x.size(3)) // 2
        #x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        x = cat((x, x1), dim=1)
        x = self.pub(x)
        return x