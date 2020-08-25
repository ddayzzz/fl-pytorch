import torch
import time
import numpy as np
from typing import OrderedDict, Dict, List, Any, Union, Iterable, Tuple
import collections
import abc
from optimizers.pgd import PerturbedGradientDescent
from clients.base_client import BaseClient
from trainers.fedbase import FedBase
from torch.nn import Module
from torch.utils.data import Dataset


def get_model_parameters_list(model: Module) -> List[torch.Tensor]:
    p = [p.detach().clone() for p in model.parameters()]
    return p


def set_model_parameters_list(model: Module, params: List[torch.Tensor]):
    for p, src in zip(model.parameters(), params):
        p.data.copy_(src.data)


class FedProxClient(BaseClient):

    def __init__(self, id, model: Module, dataset, dataset_type, options):
        super(FedProxClient, self).__init__(id, model, dataset, dataset_type, options)

    def create_optimizer(self):
        if self.dataset_type == 'train':
            return PerturbedGradientDescent(params=self.model.parameters(),
                                            weight_decay=self.options['wd'],
                                            mu=self.options['mu'],
                                            lr=self.options['lr'])
        else:
            return None

    def get_model_parameters_list(self) -> List[torch.Tensor]:
        return get_model_parameters_list(self.model)

    def set_model_parameters_list(self, params: List[torch.Tensor]):
        set_model_parameters_list(self.model, params)


class FedProx(FedBase):

    def __init__(self, options, dataset_info, model):
        print('>>> Using FedProx')
        self.drop_rate = options['drop_rate']
        self.mu = options['mu']
        a = f'mu_{options["mu"]}_dp_{[options["drop_rate"]]}'
        super(FedProx, self).__init__(options=options, model=model, dataset_info=dataset_info, append2metric=a)

    def get_latest_model(self):
        return get_model_parameters_list(self.model)

    def set_latest_model(self):
        # client和 fedbase 都共用了模型
        set_model_parameters_list(self.model, self.latest_model)

    def create_clients_group(self, users: Iterable[Any], train_or_test_dataset_obj: Dict[Any, Dataset], dataset_type) -> OrderedDict[Any, FedProxClient]:
        all_clients = collections.OrderedDict()
        for user in users:
            c = FedProxClient(id=user, model=self.model, dataset=train_or_test_dataset_obj[user], dataset_type=dataset_type, options=self.options)
            all_clients[user] = c
        return all_clients

    def select_clients(self, round_i, clients_per_rounds) -> List[int]:
        """
        这个返回client的索引而非对象
        :param round_i:
        :param num_clients: 选择的客户端的数量
        :return:
        """
        np.random.seed(round_i)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.num_train_clients, clients_per_rounds, replace=False).tolist()

    def aggregate_parameters_weighted(self, solns: List[List[torch.Tensor]], num_samples: List[int]) -> List[torch.Tensor]:
        """
        聚合模型
        :param solns: 列表.
        :param kwargs:
        :return: 聚合后的参数
        """
        factors = np.asarray(num_samples) / sum(num_samples)
        result = dict()
        num_params = len(solns[0])
        for p_name in range(num_params):
            new = torch.zeros_like(solns[0][p_name])
            for factor, sol in zip(factors, solns):
                new.add_(sol[p_name], alpha=factor)
            result.append(new)
        return result

    def aggregate(self, solns, num_samples):
        # return self.aggregate_parameters_weighted(solns, num_samples)
        result = self.aggregate_parameters_weighted(solns, num_samples)
        del solns
        del self.latest_model
        return result

    def solve_epochs(self,round_i, clients: List[int], num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        selected_client_indices = clients
        activated_clients_indices = np.random.choice(selected_client_indices,
                                                     round(self.clients_per_round * (1 - self.drop_rate)),
                                                     replace=False)
        num_samples = []
        tot_corrects = []
        losses = []

        solns = []
        for c_index in selected_client_indices:
            if c_index in activated_clients_indices:
                # 正常运行
                epoch = num_epochs
            else:
                # 需要变化 epoch 的客户端
                epoch = np.random.randint(low=1, high=num_epochs)
            c = self.train_clients[c_index]
            # 设置优化器的参数
            c.optimizer.set_old_weights(old_weights=self.latest_model)
            # 同步为最新的模型
            self.set_latest_model()
            # 保存信息
            stat = c.solve_epochs(round_i, c.train_dataset_loader, epoch, hide_output=self.quiet)
            tot_corrects.append(stat['acc_meter'].sum)
            num_samples.append(stat['num_samples'])
            losses.append(stat['loss_meter'].sum)
            #
            solns.append(c.get_model_parameters_list())
            # 写入测试的相关信息
            # self.metrics.update_commu_stats(round_i, flop_stat)

        mean_loss = sum(losses) / sum(num_samples)
        mean_acc = sum(tot_corrects) / sum(num_samples)

        stats = {
            'acc': mean_acc, 'loss': mean_loss,
        }
        if not self.quiet:
            print(f'Round {round_i}, train metric mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        self.metrics.update_train_stats(round_i, stats)
        return solns, num_samples

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')
            selected_client_indices = self.select_clients(round_i=round_i, clients_per_rounds=self.clients_per_round)
            solns, num_samples = self.solve_epochs(round_i=round_i, clients=selected_client_indices)
            self.latest_model = self.aggregate(solns, num_samples)
            # eval on test
            if round_i % self.eval_on_test_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.test_clients, client_type='test')

            if round_i % self.eval_on_train_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.train_clients, client_type='train')

            if round_i % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()
        self.metrics.write()