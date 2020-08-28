import os
import torch
import numpy as np
from dataset.cifar100.get_tff_format import CIFAR100TFFVersion as _CIFAR100TFFVersion, DATA
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CIFAR100Dataset(Dataset):

    def __init__(self, pixel, labels, corase_labels, transform=None):
        """
        京对应的
        :param data:
        :param labels:
        """
        super(CIFAR100Dataset, self).__init__()
        self.pixel = pixel
        self.labels = labels
        self.corase_labels = corase_labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.pixel[index], self.labels[index]

        if self.transform is not None:
            data = self.transform(data)
        return data, target


def create_dataset(dataset: _CIFAR100TFFVersion, trans):
    cid_to_dataset = dict()
    for c in dataset.client_ids:
        cid_to_dataset[c] = CIFAR100Dataset(*dataset[c], transform=trans)
    return dataset.client_ids, cid_to_dataset


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def make_data(options):
    data_prefix = os.path.join(DATA, 'data')
    # 可能需要进行处理, 默认图像大小为 32, 原文中没有对图像进行 disort(random flip)
    crop_size = options['cifar100_image_size']
    # tff 中使用的 per_image_standard
    CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # TODO 目标大小24, 应该不会调用 padding
        transforms.RandomCrop(crop_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        # 根据 tff 给出的实验设置
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_client_data = _CIFAR100TFFVersion(data_prefix=data_prefix, is_train=True)
    test_client_data = _CIFAR100TFFVersion(data_prefix=data_prefix, is_train=False)
    train_clients, train_data = create_dataset(train_client_data, trans=train_transform)
    test_clients, test_data = create_dataset(test_client_data, trans=valid_transform)
    return train_clients, train_data, test_clients, test_data


if __name__ == '__main__':
    data_prefix = os.path.join(DATA, 'data')
    # 可能需要进行处理, 默认图像大小为 32, 原文中没有对图像进行 disort(random flip)
    crop_size = 24
    # tff 中使用的 per_image_standard
    CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(crop_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_client_data = _CIFAR100TFFVersion(data_prefix=data_prefix, is_train=True)
    test_client_data = _CIFAR100TFFVersion(data_prefix=data_prefix, is_train=False)
    train_clients, train_data = create_dataset(train_client_data, trans=train_transform)
    test_clients, test_data = create_dataset(test_client_data, trans=valid_transform)

    from matplotlib import pyplot as plt
    from torchvision.utils import make_grid

    def show_image(client_data, ds):
        images = []
        for cid in client_data.client_ids[:32]:
            x, label = ds[cid][0]
            print(x.shape, label)
            # 选择一张即可
            images.append(x)
        images = torch.stack(images, dim=0)
        print(images.shape, images.dtype)
        to_show = make_grid(tensor=images, nrow=4)
        plt.figure()
        plt.imshow(np.transpose(to_show.numpy(), [1, 2, 0]))
        plt.show()

    show_image(train_client_data, train_data)
    show_image(test_client_data, test_data)





