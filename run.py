import numpy as np
import importlib
import torch
import os
import random
from collections import namedtuple
from dataset.read_leaf import read_leaf
from dataset.read_pkl import read_pkl
from utils.data_utils import MiniDataset
from config import DATASETS, TRAINERS, MODEL_CONFIG, DATASET_WRAPPER
from config import base_options, add_dynamic_options
from dataset.statistical_info import print_stats


DatasetInfo = namedtuple('DatasetInfo', ['users', 'groups', 'train_data', 'test_data', 'validation_data', 'dataset_wrapper'])


def read_options():
    parser = base_options()
    parser = add_dynamic_options(parser)
    parsed = parser.parse_args()
    options = parsed.__dict__
    # 设置种子
    os.environ['PYTHONHASHSEED'] = str(options['seed'])
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    random.seed(1234 + options['seed'])
    if options['device'].startswith('cuda'):
        torch.cuda.manual_seed_all(123 + options['seed'])
        torch.backends.cudnn.deterministic = True  # cudnn


    # 读取数据集
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)
    # 数据集的包装器
    dataset_wrapper = DATASET_WRAPPER.get(dataset_name, MiniDataset)

    # 将配置的参数添加到测试文件中
    model_cfg_key ='.'.join((dataset_name, options['model']))
    model_cfg = MODEL_CONFIG.get(model_cfg_key)

    # 加载选择的 solver 类
    trainer_path = 'trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # 加载模型类
    model_path = 'models.{0}.{1}'.format(dataset_name, options['model'])
    mod = importlib.import_module(model_path)
    model_obj = getattr(mod, 'Model')(**model_cfg)

    # 打印参数
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> 参数:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, model_obj, trainer_class, dataset_wrapper, dataset_name, sub_data


def main():


    # 解析参数
    options, model_obj, trainer_class, dataset_wrapper, dataset_name, sub_data = read_options()

    # 数据的文件始终在其父目录
    dataset_prefix = os.path.realpath(options['data_prefix'])
    #
    is_leaf = options['leaf']
    if is_leaf:
        train_path = os.path.join(dataset_prefix, 'data', dataset_name, 'data', 'train')
        test_path = os.path.join(dataset_prefix, 'data', dataset_name, 'data', 'test')
        df = read_leaf(train_path, test_path)
    else:
        train_path = os.path.join(dataset_prefix, 'dataset', dataset_name, 'data', 'train')
        test_path = os.path.join(dataset_prefix, 'dataset', dataset_name, 'data', 'test')
        df = read_pkl(train_path, test_path, sub_data=sub_data)
    # df: train_clients, train_groups, train_data, test_data
    all_data_info = DatasetInfo(users=df[0], groups=df[1], train_data=df[2], test_data=df[3], validation_data=None, dataset_wrapper=dataset_wrapper)
    # 输出数据的信息
    print_stats(all_data_info.train_data, all_data_info.users, title='>>> 训练数据信息')
    print_stats(all_data_info.test_data, all_data_info.users, title='>>> 测试数据信息')
    # 调用solver
    trainer = trainer_class(options, all_data_info, model_obj)
    trainer.train()


if __name__ == '__main__':
    main()
