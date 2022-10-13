# 参考 https://github.com/tensorflow/federated/blob/v0.16.1/tensorflow_federated/python/simulation/datasets/cifar100.py#L23-L96
import os
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import hashlib
import tarfile
import h5py
import collections
import numpy as np

DATA = os.path.dirname(os.path.realpath(__file__))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def extract_archive_tar_bz2(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with tarfile.open(from_path, 'r:bz2') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=to_path)

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


class EMNISTTFFVersion(Dataset):
    FILE_INFO = {
        'fed_emnist_digitsonly': {
            'url': 'https://storage.googleapis.com/tff-datasets-public/fed_emnist_digitsonly.tar.bz2',
            'train': 'fed_emnist_digitsonly_train.h5',
            'test': 'fed_emnist_digitsonly_test.h5',
            'sha256': '55333deb8546765427c385710ca5e7301e16f4ed8b60c1dc5ae224b42bd5b14b'},
        'fed_emnist': {
            'url': 'https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2',
            'train': 'fed_emnist_train.h5',
            'test': 'fed_emnist_test.h5',
            'sha256': 'fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0'}
    }

    def __init__(self, data_prefix, is_train=True, only_digitis=True, image_transform=None):
        """
        Downloads and caches the dataset locally. If previously downloaded, tries to
    load the dataset from cache.

  This dataset is derived from the Leaf repository
  (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
  dataset, grouping examples by writer. Details about Leaf were published in
  "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

  *Note*: This dataset does not include some additional preprocessing that
  MNIST includes, such as size-normalization and centering.
  In the Federated EMNIST data, the value of 1.0
  corresponds to the background, and 0.0 corresponds to the color of the digits
  themselves; this is the *inverse* of some MNIST representations,
  e.g. in [tensorflow_datasets]
  (https://github.com/tensorflow/datasets/blob/master/docs/datasets.md#mnist),
  where 0 corresponds to the background color, and 255 represents the color of
  the digit.

  Data set sizes:

  *only_digits=True*: 3,383 users, 10 label classes

  -   train: 341,873 examples
  -   test: 40,832 examples

  *only_digits=False*: 3,400 users, 62 label classes

  -   train: 671,585 examples
  -   test: 77,483 examples

  Rather than holding out specific users, each user's examples are split across
  _train_ and _test_ so that all users have at least one example in _train_ and
  one example in _test_. Writers that had less than 2 examples are excluded from
  the data set.
        """
        filename_prefix = 'fed_emnist_digitsonly' if only_digitis else 'fed_emnist'
        self.only_digits = only_digitis
        self.data_key = 'train' if is_train else 'test'
        self.data_prefix = data_prefix
        self.tff_raw_data = mkdir(os.path.join(data_prefix, 'tff_raw_data'))
        data_path = self.check_file(filename_prefix=filename_prefix)
        self.examples, self.client_ids = self.load_data(data_path)
        size = 0
        data = collections.OrderedDict()
        for cid in self.client_ids:
            x, label = self.examples['examples'][cid]['pixels'].value, self.examples['examples'][cid][
                'label'].value
            # 默认已经转换为 float32
            # N, H, W -> N, C, H, W
            x = np.expand_dims(x, axis=1)
            size += len(label)
            # label 默认为 int32, torch 需要 int64
            data[cid] = (x, label.astype(np.int64))
        self.client_wise_data = data
        self.size = size
        # 访问方式: examples['examples'][client_ids[0]]['image'].value
        # 开始处理数据
        # self.train_labels, self.train_coarse_label, self.train_image = train_examples.labels, train_examples.coarse_label, train_examples.image
        # self.test_labels, self.test_coarse_label, self.test_image = test_examples.labels, test_examples.coarse_label, test_examples.image

    def check_file(self, filename_prefix):
        filename = filename_prefix + '.tar.bz2'
        target_file = os.path.join(self.tff_raw_data, self.FILE_INFO[filename_prefix][self.data_key])
        if not os.path.exists(target_file):
            # 需要处理文件
            tar_file = os.path.join(self.tff_raw_data, filename)
            if not check_integrity(tar_file, self.FILE_INFO[filename_prefix]['sha256']):
                info = self.FILE_INFO[filename_prefix]
                download_url(info['url'], root=self.tff_raw_data, filename=filename)
            # 解压 tar.bz2, 需要额外的处理
            extract_archive_tar_bz2(tar_file, self.tff_raw_data)
        # 加载数据
        return target_file

    def load_data(self, filename):
        h5_file = h5py.File(filename, "r")
        client_ids = sorted(list(h5_file['examples'].keys()))
        return h5_file, client_ids

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.client_wise_data[self.client_ids[item]]
        else:
            return self.client_wise_data[item]


if __name__ == '__main__':
    data_prefix = DATA + '/data'
    emnist_all_train = EMNISTTFFVersion(data_prefix=data_prefix, is_train=True, only_digitis=False)
    emnist_all_test = EMNISTTFFVersion(data_prefix=data_prefix, is_train=False, only_digitis=False)
    print(emnist_all_train.client_ids)
    print(len(emnist_all_train))
    print(emnist_all_test.client_ids)
    print(len(emnist_all_test))
    from matplotlib import pyplot as plt
    import torch
    from torchvision.utils import make_grid

    images = []
    for cid in emnist_all_train.client_ids[:64]:
        x, label = emnist_all_train[cid]
        print(np.shape(x), label)
        # 选择一张即可
        images.append(torch.from_numpy(x[0, :, :, :]))
    images = torch.stack(images, dim=0)
    print(images.shape)
    to_show = make_grid(tensor=images, nrow=8)
    plt.figure()
    plt.imshow(np.transpose(to_show.numpy(), [1, 2, 0]))
    plt.show()
