# 使用全部四种模态以及 HGG 和 LGG 的数据集
import json
import os
import numpy as np
import torch
import torch.utils.data
from dataset.read_data import DatasetInfo


class BRATSAllModDataset(torch.utils.data.Dataset):

    def __init__(self, images, masks):
        self.images = images
        self.labels = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.labels[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        # num_slice, h, w , c = npimage.shape
        # [num_slice/DEPTH/BLOCK, H, W, C]
        # CONV3D: [C,DEPTH,H,W]
        npimage = npimage.transpose((3, 0, 1, 2))
        # 针对输入数据的不同的模态设计不同的 ground truth, 在预处理的时候已经处理了这3个通道

        nplabel = npmask.transpose((3, 0, 1, 2)).astype(np.float32)
        npimage = npimage.astype(np.float32)

        return npimage, nplabel


def make_dataset(options) -> DatasetInfo:
    root = os.path.dirname(os.path.realpath(__file__))
    data = os.path.join(root, 'data', 'configs')
    cfg = os.path.join(data, options['brats_config'] + '.json')
    print(">>> BRATS load config: ", cfg)
    with open(cfg) as fp:
        cfg = json.load(fp)
    train_ds = dict()
    for k, v in cfg['train'].items():
        # v: {pid: [ [train...], [test...]]}
        images = []
        masks = []
        for pid, item in v.items():
            images.extend(item[0])
            masks.extend(item[1])
        train_ds[k] = BRATSAllModDataset(images, masks)

    test_ds = dict()
    for k, v in cfg['test'].items():
        # v: {pid: [ [image...], [mask...]]}
        images = []
        masks = []
        for pid, item in v.items():
            images.extend(item[0])
            masks.extend(item[1])
        test_ds[k] = BRATSAllModDataset(images, masks)
    # 机构名
    train_clients = list(cfg['train'].keys())
    test_clients = list(cfg['test'].keys())
    return DatasetInfo(
        train_users=train_clients,
        test_users=test_clients,
        train_data=train_ds,
        test_data=test_ds,
        validation_data=None

    )


def show_image(config_name):
    from matplotlib import pyplot as plt
    import torch
    from torchvision.utils import make_grid
    ds = make_dataset({'brats_config': config_name})
    print('包含的训练的机构: ')
    for ins in ds.train_users:
        print(ins)
    print('包含的测试的机构: ')
    for ins in ds.test_users:
        print(ins)
    #
    images = []
    masks = []
    for ins in ds.train_users[:4]:
        store = ds.train_data[ins]
        for i in range(4):
            img, mask = store[i]
            # [patch, c, h , w]
            img = img[0, 0, :, :]
            images.append(torch.from_numpy(img))
            # masks.append(torch.from_numpy(mask))
    images = torch.stack(images).unsqueeze(1)
    print(images.shape)
    # masks = torch.stack(masks)
    to_show = make_grid(tensor=images, nrow=4)
    print(to_show.shape)
    plt.figure()
    plt.imshow(np.transpose(to_show.numpy(), [1, 2, 0]))
    plt.show()


if __name__ == '__main__':
    show_image('2018train_2019test')
