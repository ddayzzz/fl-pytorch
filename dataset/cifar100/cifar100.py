import os
import torch
from dataset.cifar100.get_tff_format import CIFAR100TFFVersion as _CIFAR100TFFVersion, DATA
from torch.utils.data import Dataset


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


def create_dataset(dataset: _CIFAR100TFFVersion):
    cid_to_dataset = dict()
    for c in dataset.client_ids:
        cid_to_dataset[c] = CIFAR100Dataset(*dataset[c], transform=None)
    return dataset.client_ids, cid_to_dataset


def make_data():
    data_prefix = os.path.join(DATA, 'data')
    # 可能需要进行处理

    train_client_data = _CIFAR100TFFVersion(data_prefix=data_prefix, is_train=True)
    test_client_data = _CIFAR100TFFVersion(data_prefix=data_prefix, is_train=False)
    train_clients, train_data = create_dataset(train_client_data)
    test_clients, test_data = create_dataset(test_client_data)
    return train_clients, train_data, test_clients, test_data





