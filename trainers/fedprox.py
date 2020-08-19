import torch
import time
import numpy as np
from typing import List
from optimizers.pgd import PerturbedGradientDescent
from clients.base_client import BaseClient
from trainers.fedbase import FedBase


class FedProxClient(BaseClient):

    def __init__(self, id, model, train_dataset, test_dataset, options):
        super(FedProxClient, self).__init__(id, model, train_dataset, test_dataset, options, validation_dataset=None)

    def create_optimizer(self):
        return PerturbedGradientDescent(params=self.model.parameters(),
                                        weight_decay=self.options['wd'],
                                        mu=self.options['mu'],
                                        lr=self.options['lr'])

    def get_model_parameters_list(self) -> List[torch.Tensor]:
        p = [p.detach().clone() for p in self.model.parameters()]
        return p

    def set_model_parameters_list(self, params: List[torch.Tensor]):
        for p, src in zip(self.model.parameters(), params):
            p.data.copy_(src.data)


class FedProx(FedBase):

    def __init__(self, options, dataset_info, model):
        print('>>> Using FedProx')
        self.drop_rate = options['drop_rate']
        self.mu = options['mu']
        a = f'mu_{options["mu"]}_dp_{[options["drop_rate"]]}'
        super(FedProx, self).__init__(options=options, model=model, dataset_info=dataset_info, append2metric=a)

    def get_latest_model(self):
        return self.clients[0].get_model_parameters_list()

    def set_latest_model(self, client: FedProxClient):
        client.set_model_parameters_list(self.latest_model)

    def setup_clients(self, dataset_info):
        if len(dataset_info.groups) == 0:
            groups = [None for _ in dataset_info.users]
        else:
            groups = dataset_info.groups
        all_clients = []
        for user, group in zip(dataset_info.users, groups):
            tr = dataset_info.dataset_wrapper(dataset_info.train_data[user], options=self.options)
            te = dataset_info.dataset_wrapper(dataset_info.test_data[user], options=self.options)
            c = FedProxClient(id=user, options=self.options, train_dataset=tr, test_dataset=te, model=self.model)
            all_clients.append(c)
        return all_clients

    def select_clients(self, round_i, num_clients):
        """
        这个返回client的索引而非对象
        :param round_i:
        :param num_clients: 选择的客户端的数量
        :return:
        """
        num_clients = min(num_clients, self.num_clients)
        np.random.seed(round_i)  # 确定每一轮次选择相同的客户端(用于比较不同算法在同一数据集下的每一轮的客户端不变)
        return np.random.choice(self.num_clients, num_clients, replace=False).tolist()

    def aggregate_parameters_weighted(self, solns: List[List[torch.Tensor]], num_samples: List[int]) -> List[torch.Tensor]:
        """
        聚合模型
        :param solns: 列表.
        :param kwargs:
        :return: 聚合后的参数
        """
        lastes = list()
        num_params = len(solns[0])
        for p_name in range(num_params):
            new = torch.zeros_like(solns[0][p_name])
            sz = 0
            for num_sample, sol in zip(num_samples, solns):
                new += sol[p_name] * num_sample
                sz += num_sample
            new /= sz
            lastes.append(new)
        return lastes

    def aggregate(self, solns, num_samples):
        return self.aggregate_parameters_weighted(solns, num_samples)

    def solve_epochs(self,round_i, clients, num_epochs=None):
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
            c = self.clients[c_index]
            # 设置优化器的参数
            c.optimizer.set_old_weights(old_weights=self.latest_model)
            c.set_model_parameters_list(self.latest_model)
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
            selected_client_indices = self.select_clients(round_i=round_i, num_clients=self.clients_per_round)
            solns, num_samples = self.solve_epochs(round_i=round_i, clients=selected_client_indices)
            self.latest_model = self.aggregate(solns, num_samples)
            # eval on test
            if round_i % self.eval_on_test_every_round == 0:
                self.eval_on(use_test_data=True, round_i=round_i, clients=self.clients)
            if round_i % self.eval_on_train_every_round == 0:
                self.eval_on(use_train_data=True, round_i=round_i, clients=self.clients)

            if round_i % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()
        self.metrics.write()