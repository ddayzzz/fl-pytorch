import os
import pickle
import json


def _load_data(fp, ext: str):
    if ext.lower() == '.pkl':
        cdata = pickle.load(fp)
    elif ext.lower() == '.json':
        cdata = json.load(fp)
    else:
        raise ValueError('不支持的类型: {}'.format(ext))
    return cdata


def read_pkl(train_data_dir, test_data_dir, sub_data=None):
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
    return clients, groups, train_data_index, test_data_index