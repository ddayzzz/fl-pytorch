from torch.nn import Module, Conv3d, BatchNorm3d, init
from models.brats.utils.unet_parts3d import Down3D, Up3D


class Model(Module):

    def __init__(self, init_channels, class_nums, options, batch_norm=True, sample=True):
        super(Model, self).__init__()
        # init_channels = 4
        # class_nums = 3
        # batch_norm = True
        # sample = True
        self.init_channels = init_channels
        self.class_nums = class_nums
        self.batch_norm = batch_norm
        self.sample = sample
        self.en1 = Down3D(init_channels, 64, batch_norm)
        self.en2 = Down3D(64, 128, batch_norm)
        self.en3 = Down3D(128, 256, batch_norm)
        self.en4 = Down3D(256, 512, batch_norm)

        self.up3 = Up3D(512, 256, batch_norm, sample)
        self.up2 = Up3D(256, 128, batch_norm, sample)
        self.up1 = Up3D(128, 64, batch_norm, sample)
        self.con_last = Conv3d(64, class_nums, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x = self.en1(x)
        x2, x = self.en2(x)
        x3, x = self.en3(x)
        x4, _ = self.en4(x)

        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.con_last(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv3d):
                init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    import torch
    from torchsummary import summary
    torch.cuda.set_device(2)
    model1 = Model(4, 3, options=None).cuda()
    # model2 = Model(4, 3).to('cuda:6')
    x = torch.rand((1, 4, 32, 64, 64)).cuda()
    y1 = model1(x)
    print(summary(model1, (4, 32, 64, 64), device='cuda'))