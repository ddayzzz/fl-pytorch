import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from models.utils.same_padding_conv import Conv2d


class Model(nn.Module):

    def __init__(self, num_classes, image_size, options):
        super(Model, self).__init__()
        self.input_shape = (1, image_size * image_size)
        self.num_classes = num_classes
        self.image_size = image_size
        # 创建参数, 这里必须使用 SAME padding, 保持卷积的不变
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=1)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc2 = nn.Linear(2048, self.num_classes)
        # 初始化, tf2(keras) 和 tf1 都是这种初始化的方式 glorot_uniform/xavier_uniform, bias 则是全零
        xavier_uniform_(self.conv1.weight)
        xavier_uniform_(self.conv2.weight)
        xavier_uniform_(self.fc1.weight)
        xavier_uniform_(self.fc2.weight)
        zeros_(self.conv1.bias)
        zeros_(self.conv2.bias)
        zeros_(self.fc1.bias)
        zeros_(self.fc2.bias)


    def forward(self, x):
        x = x.view(-1, 1, self.image_size, self.image_size)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    x = torch.rand([10, 1, 28, 28])
    model = Model(10, 28, None)
    print(model.fc1.weight.data_ptr(), model.fc1.weight)
    t = model.state_dict()
    fc1w = t['fc1.weight']
    a = torch.rand(fc1w.shape)
    a.add_(fc1w)
    print(a.data_ptr(), a)
    print(fc1w.data_ptr(), fc1w)
    fc1w.copy_(a)
    print(fc1w.data_ptr(), fc1w)

