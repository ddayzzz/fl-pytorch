import os
import torch
from dataset.emnist.get_tff_format import EMNISTTFFVersion as _EMNISTTFFVersion, DATA
from torch.utils.data import Dataset


class EMNISTDataset(Dataset):

    def __init__(self, pixel, labels, transform=None):
        """
        京对应的
        :param data:
        :param labels:
        """
        super(EMNISTDataset, self).__init__()
        self.pixel = pixel
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.pixel[index], self.labels[index]

        if self.transform is not None:
            data = self.transform(data)
        return data, target


def create_dataset(dataset: _EMNISTTFFVersion):
    cid_to_dataset = dict()
    for c in dataset.client_ids:
        cid_to_dataset[c] = EMNISTDataset(*dataset[c], transform=None)
    return dataset.client_ids, cid_to_dataset


def make_data(only_digits):
    data_prefix = os.path.join(DATA, 'data')
    # 可能需要进行处理, 按照tff的叙述, emnist 已经经过处理
    train_client_data = _EMNISTTFFVersion(data_prefix=data_prefix, is_train=True, only_digitis=only_digits)
    test_client_data = _EMNISTTFFVersion(data_prefix=data_prefix, is_train=False, only_digitis=only_digits)
    train_clients, train_data = create_dataset(train_client_data)
    test_clients, test_data = create_dataset(test_client_data)
    return train_clients, train_data, test_clients, test_data





