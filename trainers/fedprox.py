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
from utils.metrics import Meter
import tqdm
import copy


class FedProxClient(BaseClient):

    def __init__(self, id, dataset, dataset_type, options):
        super(FedProxClient, self).__init__(id, dataset, dataset_type, options)

    def create_optimizer(self, model: Module):
        if self.dataset_type == 'train':
            return PerturbedGradientDescent(params=model.parameters(),
                                            weight_decay=self.options['wd'],
                                            mu=self.options['mu'],
                                            lr=self.options['lr'])
        else:
            return None

    def solve_epochs_with_global(self, round_i, model: Module, global_model: Module, num_epochs, hide_output: bool = False) -> Tuple[Dict[str, Union[int, Meter]], Dict[str, torch.Tensor]]:
        loss_meter = Meter()
        acc_meter = Meter()
        num_all_samples = 0
        optimizer = self.create_optimizer(model)
        # TODO 直接引用上一次的 global 模型, 避免复制, optimizer 的  step 中也是不记录梯度的
        optimizer.set_old_weights(old_weights=[p for p in global_model.parameters()])

        model.train()

        with tqdm.trange(num_epochs, disable=hide_output) as t:

            for epoch in t:
                t.set_description(f'Client: {self.id}, Round: {round_i}, Epoch :{epoch}')
                for batch_idx, (X, y) in enumerate(self.dataset_loader):
                    # from IPython import embed
                    X, y = X.to(self.device), y.to(self.device)

                    optimizer.zero_grad()
                    pred = model(X)

                    loss = self.criterion(pred, y)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                    optimizer.step()

                    correct_sum = self.count_correct(pred, y)
                    num_samples = y.size(0)
                    num_all_samples += num_samples
                    loss_meter.update(loss.item(), n=num_samples)
                    acc_meter.update(correct_sum.item() / num_samples, n=num_samples)
                    if (batch_idx % 10 == 0):
                        # 纯数值, 这里使用平均的损失
                        t.set_postfix(mean_loss=loss.item())
        # 返回参数
        result = {
            'loss_meter': loss_meter,
            'acc_meter': acc_meter,
            'num_samples': num_all_samples
        }
        state_dict = model.state_dict()
        # 输出相关的参数
        return result, state_dict


class FedProx(FedBase):

    def __init__(self, options, dataset_info, model):
        print('>>> Using FedProx')
        self.drop_rate = options['drop_rate']
        self.mu = options['mu']
        a = f'mu_{options["mu"]}_dp_{[options["drop_rate"]]}'
        super(FedProx, self).__init__(options=options, model=model, dataset_info=dataset_info, append2metric=a)

    def create_clients_group(self, users: Iterable[Any], train_or_test_dataset_obj: Dict[Any, Dataset], dataset_type) -> OrderedDict[Any, FedProxClient]:
        all_clients = collections.OrderedDict()
        for user in users:
            c = FedProxClient(id=user, dataset=train_or_test_dataset_obj[user], dataset_type=dataset_type, options=self.options)
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

    def aggregate(self, solns, num_samples):
        # return self.aggregate_parameters_weighted(solns, num_samples)
        new_state = self.aggregate_parameters_weighted(solns, num_samples)
        self.global_model.load_state_dict(new_state)

    def solve_epochs(self,round_i, clients: List[int], num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs
        self.global_model.train()
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
            # 同步为最新的模型
            model_to_client = copy.deepcopy(self.global_model).to(self.device)
            # 保存信息
            stat, state_dict = c.solve_epochs_with_global(round_i, model_to_client, self.global_model, epoch, hide_output=self.quiet)
            tot_corrects.append(stat['acc_meter'].sum)
            num_samples.append(stat['num_samples'])
            losses.append(stat['loss_meter'].sum)
            #
            solns.append(state_dict)
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
            self.aggregate(solns, num_samples)
            # eval on test
            if round_i % self.eval_on_test_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.test_clients, client_type='test')

            if round_i % self.eval_on_train_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.train_clients, client_type='train')

            if round_i % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()
        self.metrics.write()