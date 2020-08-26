import numpy as np
import os
import time
import collections
from typing import OrderedDict, Dict, List, Any, Union, Iterable, Tuple
import abc
import torch
from torch.utils.data import Dataset
from torch import nn, optim
import pandas as pd
from clients.base_client import BaseClient
from utils.metrics import Metrics
from utils.flops_counter import get_model_complexity_info
from dataset.read_data import DatasetInfo
import copy


class FedBase(abc.ABC):

    def __init__(self, options, model: nn.Module, dataset_info, append2metric=None):
        """
        定义联邦学习的基本的服务器, 这里的模型是在所有的客户端之间共享使用
        :param options: 参数配置
        :param model: 模型
        :param dataset: 数据集参数
        :param optimizer: 优化器
        :param criterion: 损失函数类型(交叉熵,Dice系数等等)
        :param worker: Worker 实例
        :param append2metric: 自定义metric
        """
        self.options = options
        self.model, self.flops, self.params_num, self.model_bytes = self.setup_model(model=model)
        self.device = options['device']
        # 记录总共的训练数据
        client_info = self.setup_clients(dataset_info=dataset_info)
        # TODO orderdict 可以这么写
        self.train_client_ids, self.train_clients = (client_info['train_clients'][0], list(client_info['train_clients'][1].values()))
        self.test_client_ids, self.test_clients = (client_info['test_clients'][0], list(client_info['test_clients'][1].values()))
        self.num_train_clients = len(self.train_client_ids)
        self.num_test_clients = len(self.test_client_ids)

        self.num_epochs = options['num_epochs']
        self.num_rounds = options['num_rounds']
        self.clients_per_round = options['clients_per_round']
        self.save_every_round = options['save_every']
        self.eval_on_train_every_round = options['eval_on_train_every']
        self.eval_on_test_every_round = options['eval_on_test_every']
        self.eval_on_validation_every_round = options['eval_on_validation_every']
        # 使用 client 的API
        self.global_model = model
        self.global_model.train()
        self.name = '_'.join(['', f'wn[{options["clients_per_round"]}]', f'num_train[{self.num_train_clients}]', f'num_test[{self.num_test_clients}]'])
        self.metrics = Metrics(options=options, name=self.name, append2suffix=append2metric, result_prefix=options['result_prefix'])
        self.quiet = options['quiet']

    def setup_model(self, model):
        dev = self.options['device']
        model = model.to(dev)
        # input_shape = model.input_shape
        # input_type = model.input_type if hasattr(model, 'input_type') else None
        # flops, params_num, model_bytes = \
        #     get_model_complexity_info(model, input_shape, input_type=input_type, device=dev)
        # return model, flops, params_num, model_bytes
        return model, None, None, None

    def create_clients_group(self, users: Iterable[Any], train_or_test_dataset_obj: Dict[Any, Dataset], dataset_type) -> OrderedDict[Any, BaseClient]:
        """

        :param dataset_info: {client_id: xxx}
        :return:
        """
        all_clients = collections.OrderedDict()
        for user in users:
            c = BaseClient(id=user, dataset=train_or_test_dataset_obj[user], dataset_type=dataset_type, options=self.options)
            all_clients[user] = c
        return all_clients

    def setup_clients(self, dataset_info: Union[DatasetInfo]) -> Dict[Any, Tuple[List[Any], OrderedDict[Any, BaseClient]]]:
        """
        :param dataset_info:
        :return: 分为 train, test 或者其他的客户端. 第一个元素为客户端的 id,
        """
        result = dict()
        if isinstance(dataset_info, DatasetInfo):
            if dataset_info.train_data is not None:
                result['train_clients'] = (dataset_info.train_users, self.create_clients_group(users=dataset_info.train_users, train_or_test_dataset_obj=dataset_info.train_data, dataset_type='train'))
            if dataset_info.test_data is not None:
                result['test_clients'] = (dataset_info.test_users, self.create_clients_group(users=dataset_info.test_users, train_or_test_dataset_obj=dataset_info.test_data, dataset_type='test'))
            if dataset_info.validation_data is not None:
                raise NotImplementedError
            return result
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def aggregate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    def select_clients(self, round_i, clients_per_rounds) -> List[BaseClient]:
        """
        选择客户端, 采用的均匀的无放回采样
        :param round:
        :param num_clients:
        :return:
        """
        np.random.seed(round_i)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.train_clients, clients_per_rounds, replace=False).tolist()

    def aggregate_parameters_weighted(self, solns: List[Dict[str, torch.Tensor]], num_samples: List[int]) -> Dict[str, torch.Tensor]:
        """
        加权聚合模型
        :param solns: 这些参数保存在CPU中
        :param num_samples:
        :return:
        """
        # 产生新的数据
        arr = np.asarray(num_samples, dtype=np.float32)
        factors = arr / np.sum(arr)
        result = dict()
        for p_name in solns[0].keys():
            new = torch.zeros_like(solns[0][p_name])
            for factor, sol in zip(factors, solns):
                # TODO inplace, factor * 当前参数, 如果参数为不是浮点数, 那么这个可能会出现问题. 需要格外注意在 buffer 和 parameter
                #  中出现了整数类型的参数, 例如 batc norm , track_running_stats = True, 就会在 buffer 中创建一个 long
                new.add_(sol[p_name], alpha=factor)
            result[p_name] = new
        return result

    def eval_on(self, round_i, clients: Iterable[BaseClient], client_type):
        df = pd.DataFrame(columns=['client_id', 'mean_acc', 'mean_loss', 'num_samples'])

        num_samples = []
        losses = []
        correct_num = []
        for c in clients:
            # 设置网络
            stats = c.test(self.global_model)

            num_samples.append(stats['num_samples'])
            losses.append(stats['loss_meter'].sum)
            correct_num.append(stats['acc_meter'].sum)
            #
            df = df.append({'client_id': c.id, 'mean_loss': stats['loss_meter'].avg, 'mean_acc': stats['acc_meter'].avg, 'num_samples': stats['num_samples'], }, ignore_index=True)

        # ids = [c.id for c in self.clients]
        # groups = [c.group for c in self.clients]
        all_sz = sum(num_samples)
        mean_loss = sum(losses) / all_sz
        mean_acc = sum(correct_num) / all_sz
        #
        if not self.quiet:
            print(f'Round {round_i}, eval on "{client_type}" client mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        # round_i, on_which, filename, other_to_logger
        self.metrics.update_eval_stats(round_i=round_i, on_which=client_type, other_to_logger={'acc': mean_acc, 'loss': mean_loss}, df=df)

    def solve_epochs(self, round_i, clients: Iterable[BaseClient], num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        self.global_model.train()
        num_samples = []
        losses = []
        correct_num = []

        solns = []
        for c in clients:
            model_to_client = copy.deepcopy(self.global_model).to(self.device)
            # 保存信息
            stat, state_dict = c.solve_epochs(round_i, model_to_client, num_epochs, hide_output=self.quiet)

            num_samples.append(stat['num_samples'])
            losses.append(stat['loss_meter'].sum)
            correct_num.append(stat['acc_meter'].sum)
            #
            solns.append(state_dict)
            # 写入测试的相关信息
            # self.metrics.update_commu_stats(round_i, flop_stat)

        mean_loss = sum(losses) / sum(num_samples)
        mean_acc = sum(correct_num) / sum(num_samples)

        stats = {
            'acc': mean_acc, 'loss': mean_loss,
        }
        if not self.quiet:
            print(f'Round {round_i}, train metric mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        self.metrics.update_train_stats(round_i, stats)
        return solns, num_samples

