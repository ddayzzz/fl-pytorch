import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


class Model(nn.Module):

    def __init__(self, num_classes, options):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=0.25, inplace=False),
            nn.Flatten(),
            nn.Linear(in_features=12 * 12 * 64, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(in_features=128, out_features=num_classes)
        )
        for m in self.layers.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                xavier_uniform_(m.weight)
                zeros_(m.bias)

    def forward(self, x):
        out = self.layers(x)
        return out


if __name__ == '__main__':
    from torchsummary import summary
    x = torch.rand([10, 1, 28, 28]).cuda()
    model = Model(62, None).cuda()
    y = model(x)
    print(model)
    print(y)
    print(summary(model, (1, 28, 28)))

