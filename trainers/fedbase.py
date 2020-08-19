import numpy as np
import os
import time
import collections
from typing import OrderedDict, Dict, List
import abc
import torch
from torch import nn, optim
import pandas as pd
from clients.base_client import BaseClient
from utils.metrics import Metrics
from utils.flops_counter import get_model_complexity_info


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
        self.clients = self.setup_clients(dataset_info=dataset_info)
        self.num_epochs = options['num_epochs']
        self.num_rounds = options['num_rounds']
        self.clients_per_round = options['clients_per_round']
        self.save_every_round = options['save_every']
        self.eval_on_train_every_round = options['eval_on_train_every']
        self.eval_on_test_every_round = options['eval_on_test_every']
        self.eval_on_validation_every_round = options['eval_on_validation_every']
        self.num_clients = len(self.clients)
        # 使用 client 的API
        self.latest_model = self.get_latest_model()
        self.name = '_'.join(['', f'wn{options["clients_per_round"]}', f'tn{self.num_clients}'])
        self.metrics = Metrics(clients=self.clients, options=options, name=self.name, append2suffix=append2metric, result_prefix=options['result_prefix'])
        self.quiet = options['quiet']

    def setup_model(self, model):
        dev = self.options['device']
        model = model.to(dev)
        input_shape = model.input_shape
        input_type = model.input_type if hasattr(model, 'input_type') else None
        flops, params_num, model_bytes = \
            get_model_complexity_info(model, input_shape, input_type=input_type, device=dev)
        return model, flops, params_num, model_bytes

    def get_latest_model(self):
        return self.clients[0].get_model_state_dict()

    def setup_clients(self, dataset_info):
        if len(dataset_info.groups) == 0:
            groups = [None for _ in dataset_info.users]
        else:
            groups = dataset_info.groups

        all_clients = []
        for user, group in zip(dataset_info.users, groups):

            tr = dataset_info.dataset_wrapper(dataset_info.train_data[user], options=self.options)
            te = dataset_info.dataset_wrapper(dataset_info.test_data[user], options=self.options)
            if dataset_info.validation_data is not None:
                va = dataset_info.dataset_wrapper(dataset_info.validation_data[user], options=self.options)
            else:
                va = None

            c = BaseClient(id=user, options=self.options, train_dataset=tr, test_dataset=te, model=self.model, validation_dataset=va)
            all_clients.append(c)
        return all_clients

    def set_latest_model(self, client: BaseClient):
        client.set_model_state_dict(self.latest_model)

    @abc.abstractmethod
    def aggregate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    def select_clients(self, round_i, num_clients):
        """
        选择客户端, 采用的均匀的无放回采样
        :param round:
        :param num_clients:
        :return:
        """
        num_clients = min(num_clients, self.num_clients)
        np.random.seed(round_i)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def aggregate_parameters_weighted(self, solns: List[OrderedDict[str, torch.Tensor]], num_samples: List[int]) -> OrderedDict[str, torch.Tensor]:
        """
        聚合模型
        :param solns: 列表.
        :param kwargs:
        :return: 聚合后的参数
        """
        lastes = collections.OrderedDict()
        for p_name in solns[0].keys():
            new = torch.zeros_like(solns[0][p_name])
            sz = 0
            for num_sample, sol in zip(num_samples, solns):
                new += sol[p_name] * num_sample
                sz += num_sample
            new /= sz
            lastes[p_name] = new
        return lastes


    def eval_on(self, round_i, clients, use_test_data=False, use_train_data=False, use_val_data=False):
        assert use_test_data + use_train_data + use_val_data == 1, "不能同时设置"
        df = pd.DataFrame(columns=['client_id', 'mean_acc', 'mean_loss', 'num_samples'])

        num_samples = []
        losses = []
        correct_num = []
        for c in clients:
            # 设置网络
            self.set_latest_model(client=c)
            if use_test_data:
                stats =c.test(c.test_dataset_loader)
            elif use_train_data:
                stats = c.test(c.train_dataset_loader)
            elif use_val_data:
                stats = c.test(c.validation_dataset_loader)

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
        if use_test_data:
            fn, on = 'test_at_round_{}.csv'.format(round_i), 'test'
        elif use_train_data:
            fn, on = 'train_at_round_{}.csv'.format(round_i), 'train'
        elif use_val_data:
            fn, on = 'validation_at_round_{}.csv'.format(round_i), 'validation'
        #
        if not self.quiet:
            print(f'Round {round_i}, eval on "{on}" dataset mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        # round_i, on_which, filename, other_to_logger
        self.metrics.update_eval_stats(round_i=round_i, on_which=on, other_to_logger={'acc': mean_acc, 'loss': mean_loss}, df=df)

    def solve_epochs(self, round_i, clients, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        num_samples = []
        losses = []
        correct_num = []

        solns = []
        for c in clients:
            self.set_latest_model(client=c)
            # 保存信息
            stat = c.solve_epochs(round_i, c.train_dataset_loader, num_epochs, hide_output=self.quiet)

            num_samples.append(stat['num_samples'])
            losses.append(stat['loss_meter'].sum)
            correct_num.append(stat['acc_meter'].sum)
            #
            soln = c.get_model_state_dict()
            solns.append(soln)
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

