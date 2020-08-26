import pickle
from collections import defaultdict
import os
import json
from collections import namedtuple
from dataset.dataset_wrapper import MiniDataset
from dataset.shakespeare.shakespeare import Shakespeare
from dataset.sent140.sent140 import Sent140


__all__ = ['read_leaf', 'read_from_file', 'read_torch_dataset']
DATASET_WRAPPER = {
    'shakespeare': Shakespeare,
    'sent140': Sent140
}

DatasetInfo = namedtuple('DatasetInfo', ['train_users', 'test_users', 'train_data', 'test_data', 'validation_data'])


def wrap_dataset(dataset_name, data, options):
    wrapper = DATASET_WRAPPER.get(dataset_name, MiniDataset)
    return wrapper(data, options)

def _read_dir_leaf(data_dir):
    print('>>> Read data from:', data_dir)
    clients = []
    groups = []
    # 如果 dict 对象不存在时候, 不raise一个KeyError
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_leaf(dataset_name, options, train_data_dir, test_data_dir) -> DatasetInfo:
    train_clients, train_groups, train_data = _read_dir_leaf(train_data_dir)
    test_clients, test_groups, test_data = _read_dir_leaf(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups
    # name -> Dataset
    user_train_data = dict(
        [(k, wrap_dataset(dataset_name, data_dict, options)) for k, data_dict in train_data.items()])
    user_test_data = dict(
        [(k, wrap_dataset(dataset_name, data_dict, options)) for k, data_dict in test_data.items()])
    return DatasetInfo(
        train_users=train_clients,
        test_users=test_clients,
        train_data=user_train_data,
        test_data=user_test_data,
        validation_data=None
    )

def _load_data(fp, ext: str):
    if ext.lower() == '.pkl':
        cdata = pickle.load(fp)
    elif ext.lower() == '.json':
        cdata = json.load(fp)
    else:
        raise ValueError('不支持的类型: {}'.format(ext))
    return cdata


def read_from_file(dataset_name, options, train_data_dir, test_data_dir, sub_data=None) -> DatasetInfo:
    """
    解析数据
    :param train_data_dir: 训练数据目录, 自动读取 pkl
    :param test_data_dir: 测试数据目录, 自动读取 pkl
    :return: clients的编号(按照升序), groups, train_data, test_data (两者均为dict, 键是 client 的编号; 映射为 x_index 表示索引, 这个依赖于原始数据集)
    """
    ext = os.path.splitext(sub_data)[-1]
    clients = []
    groups = []
    train_data_index = {}
    test_data_index = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(ext)]
    if sub_data is not None:
        assert sub_data in train_files
        train_files = [sub_data]

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = _load_data(inf, ext)
        # 所有的用户
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        # user_data 是一个字典
        train_data_index.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(ext)]
    if sub_data is not None:
        assert sub_data in test_files
        test_files = [sub_data]

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = _load_data(inf, ext)
        test_data_index.update(cdata['user_data'])

    clients = list(sorted(train_data_index.keys()))
    # name -> Dataset
    user_train_data = dict([(k, wrap_dataset(dataset_name, data_dict, options)) for k, data_dict in train_data_index.items()])
    user_test_data = dict([(k, wrap_dataset(dataset_name, data_dict, options)) for k, data_dict in test_data_index.items()])
    return DatasetInfo(
        train_users=clients,
        test_users=clients,
        train_data=user_train_data,
        test_data=user_test_data,
        validation_data=None
    )


def read_torch_dataset(dataset_name, options) -> DatasetInfo:
    """
    直接从某个类中读取完整的数据集
    :param dataset_name:
    :return:
    """
    if dataset_name == 'cifar100':
        from dataset.cifar100.cifar100 import make_data
        train_clients, train_data, test_clients, test_data = make_data(options=options)
        ds = DatasetInfo(
            train_data=train_data,
            train_users=train_clients,
            test_data=test_data,
            test_users=test_clients,
            validation_data=None
        )
    elif dataset_name == 'emnist':
        from dataset.emnist.emnist_tff import make_data
        # 使用62分类
        train_clients, train_data, test_clients, test_data = make_data(only_digits=True)
        ds = DatasetInfo(
            train_data=train_data,
            train_users=train_clients,
            test_data=test_data,
            test_users=test_clients,
            validation_data=None
        )
    else:
        raise NotImplemented
    return ds