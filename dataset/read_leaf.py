from collections import defaultdict
import os
import json


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


def read_leaf(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = _read_dir_leaf(train_data_dir)
    test_clients, test_groups, test_data = _read_dir_leaf(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data