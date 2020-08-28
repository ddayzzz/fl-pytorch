# 参考 https://github.com/tensorflow/federated/blob/v0.16.1/tensorflow_federated/python/simulation/datasets/cifar100.py#L23-L96
import os
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import hashlib
import tarfile
import h5py
import collections
import numpy as np
from PIL import Image


DATA = os.path.dirname(os.path.realpath(__file__))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def extract_archive_tar_bz2(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with tarfile.open(from_path, 'r:bz2') as tar:
        tar.extractall(path=to_path)

    if remove_finished:
        os.remove(from_path)


def calculate_sha256(fpath, chunk_size=1024 * 1024):
    sha256 = hashlib.sha256()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            sha256.update(chunk)
    result = sha256.hexdigest()
    print(f'File : {fpath}, sha256 is {result}')
    return result


def check_integrity(fpath, sha256=None):
    if not os.path.isfile(fpath):
        return False
    if sha256 is None:
        return True
    return calculate_sha256(fpath) == sha256


class CIFAR100TFFVersion(Dataset):

    FILE_INFO = {
        'fed_cifar100.tar.bz2': {'url': 'https://storage.googleapis.com/tff-datasets-public/fed_cifar100.tar.bz2',
                                 'train': 'fed_cifar100_train.h5',
                                 'test': 'fed_cifar100_test.h5',
                                 'sha256': 'e8575e22c038ecef1ce6c7d492d7abee7da13b1e1ba9b70a7fc18531ba7590de'}
    }

    def __init__(self, data_prefix, is_train=True):
        """
        The dataset is downloaded and cached locally. If previously downloaded, it tries to load the dataset from cache.
        The dataset is derived from the CIFAR-100 dataset. The training and testing examples are partitioned across 500
        and 100 clients (respectively). No clients share any data samples, so it is a true partition of CIFAR-100. The
        train clients have string client IDs in the range [0-499], while the test clients have string client IDs in the
        range [0-99]. The train clients form a true partition of the CIFAR-100 training split, while the test clients
        form a true partition of the CIFAR-100 testing split.
        The data partitioning is done using a hierarchical Latent Dirichlet Allocation (LDA) process, referred to as the
        Pachinko Allocation Method (PAM). This method uses a two-stage LDA process, where each client has an associated
        multinomial distribution over the coarse labels of CIFAR-100, and a coarse-to-fine label multinomial distribution
        for that coarse label over the labels under that coarse label. The coarse label multinomial is drawn from a symmetric
        Dirichlet with parameter 0.1, and each coarse-to-fine multinomial distribution is drawn from a symmetric Dirichlet
        with parameter 10. Each client has 100 samples. To generate a sample for the client, we first select a coarse
        label by drawing from the coarse label multinomial distribution, and then draw a fine label using the coarse-to-fine
        multinomial distribution. We then randomly draw a sample from CIFAR-100 with that label (without replacement).
        If this exhausts the set of samples with this label, we remove the label from the coarse-to-fine multinomial and
        renormalize the multinomial distribution.
        """
        self.data_key = 'train' if is_train else 'test'
        self.data_prefix = data_prefix
        self.tff_raw_data = mkdir(os.path.join(data_prefix, 'tff_raw_data'))
        data_path = self.check_file(filename='fed_cifar100.tar.bz2')
        self.h5_data, self.client_ids = self.load_data(data_path)
        # self.per_image_standard = per_image_standard
        # 访问方式: examples['examples'][client_ids[0]]['image'].value
        # 开始处理数据
        # self.train_labels, self.train_coarse_label, self.train_image = train_examples.labels, train_examples.coarse_label, train_examples.image
        # self.test_labels, self.test_coarse_label, self.test_image = test_examples.labels, test_examples.coarse_label, test_examples.image

    def check_file(self, filename):
        target_file = os.path.join(self.tff_raw_data, self.FILE_INFO[filename][self.data_key])
        if not os.path.exists(target_file):
            # 需要处理文件
            tar_file = os.path.join(self.tff_raw_data, filename)
            if not check_integrity(tar_file, self.FILE_INFO[filename]['sha256']):
                info = self.FILE_INFO[filename]
                download_url(info['url'], root=self.tff_raw_data, filename=filename)
            # 解压 tar.bz2, 需要额外的处理
            extract_archive_tar_bz2(tar_file, self.tff_raw_data)
        # 加载数据
        return target_file

    def load_data(self, filename):
        h5_file = h5py.File(filename, "r")
        client_ids = sorted(list(h5_file['examples'].keys()))
        return h5_file, client_ids

    def create_dataset_for_client(self, client_id):
        data = collections.OrderedDict((name, ds[()]) for name, ds in sorted(
            self.h5_data['examples'][client_id].items()))
        x, label, corase_label = data['image'], data['label'], data['coarse_label']
        return x, label, corase_label

    def create_dataset_for_all_client(self):
        size = 0
        data = collections.OrderedDict()
        for cid in self.client_ids:
            x, label, corase_label = self.create_dataset_for_client(cid)
            size += len(label)
            data[cid] = (x, label, corase_label)
        return data

    def __len__(self):
        return len(self.client_ids)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.create_dataset_for_client(self.client_ids[item])
        else:
            return self.create_dataset_for_client(item)



if __name__ == '__main__':
    data_prefix = DATA + '/data'
    cifar100_train = CIFAR100TFFVersion(data_prefix=data_prefix, is_train=True)
    cifar100_test = CIFAR100TFFVersion(data_prefix=data_prefix, is_train=False)
    print(cifar100_train.client_ids)
    print(len(cifar100_train))
    print(cifar100_test.client_ids)
    print(len(cifar100_test))
    from matplotlib import pyplot as plt
    import torch
    from torchvision.utils import make_grid
    images = []
    for cid in cifar100_train.client_ids[:32]:
        x, label, cor_label = cifar100_train[cid]
        print(np.shape(x), label, cor_label)
        # 选择一张即可
        images.append(torch.from_numpy(np.transpose(x[0, :, :, :], (2, 0, 1))))
    images = torch.stack(images, dim=0)
    print(images.shape)
    to_show = make_grid(tensor=images, nrow=4)
    plt.figure()
    plt.imshow(np.transpose(to_show.numpy(), [1, 2, 0]))
    plt.show()

