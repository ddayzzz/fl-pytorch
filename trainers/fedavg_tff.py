from trainers.fedbase import FedBase
from clients.base_client import BaseClient
import torch
from torch.utils.data import Dataset
import copy
from typing import Iterable, Dict, List, Tuple, Union, OrderedDict, Any
from utils.metrics import Meter
from torch.nn import Module
import numpy as np
import tqdm
import collections


STATE_TYPE = Dict[str, torch.Tensor]


class TFFClient(BaseClient):

    def __init__(self, id, dataset, dataset_type, options):
        super(TFFClient, self).__init__(id, dataset, dataset_type, options)

    def solve_epochs_delta(self, round_i, model: Module, global_state: STATE_TYPE, num_epochs, hide_output: bool = False) -> Tuple[Dict[str, Union[int, Meter]], STATE_TYPE]:
        loss_meter = Meter()
        acc_meter = Meter()
        num_all_samples = 0
        optimizer = self.create_optimizer(model)

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
        # 计算差值 latest  - init
        for k, v in state_dict.items():
            v.sub_(global_state[k])
        # 输出相关的参数
        return result, state_dict


class FedAvgTFF(FedBase):

    def __init__(self, options, dataset_info, model):
        """
        类似于 Tensorflow-federated 形式的FedAvg
        区别在于其更新的方式是增量更新的形式 https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_2
        :param options:
        :param dataset_info:
        :param model:
        """
        self.server_lr = options['server_lr']
        self.client_lr = options['lr']
        a = 'client_lr[{}]_server_lr[{}]'.format(self.client_lr, self.server_lr)
        super(FedAvgTFF, self).__init__(options=options, model=model, dataset_info=dataset_info, append2metric=a)

    def create_clients_group(self, users: Iterable[Any], train_or_test_dataset_obj: Dict[Any, Dataset], dataset_type) -> OrderedDict[Any, TFFClient]:
        all_clients = collections.OrderedDict()
        for user in users:
            c = TFFClient(id=user, dataset=train_or_test_dataset_obj[user], dataset_type=dataset_type,
                           options=self.options)
            all_clients[user] = c
        return all_clients

    def aggregate_parameters_weighted(self, solns: Tuple[STATE_TYPE, List[STATE_TYPE]], num_samples: List[int]):
        """
        这里同样需要使用服务器的 optimizer 进行计算. server 使用GD即可, 将 delta 视为 grad, grad 需要为负数
        tff 中对应的代码
        grads_and_vars = tf.nest.map_structure(
        lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
        tf.nest.flatten(model_weights.trainable))
        server_optimizer.apply_gradients(grads_and_vars, name='server_update')
        :param solns:
        :param num_samples:
        :return:
        """
        # θ^-, 为加权聚合后的模型参数, solns[1] 客户端的参数delta的列表
        weighted_params = super(FedAvgTFF, self).aggregate_parameters_weighted(solns=solns[1], num_samples=num_samples)
        # 这里使用的新的参数, 调用 global_model 的 state_dict(其使用 detach, 使用 in-place操作)
        global_state = solns[0]
        for k, v in global_state.items():
            # θ = θ - server_lr * (-1.0 * θ^-)
            v.add_(weighted_params[k].mul_(-1.0), alpha=-self.server_lr)


    def aggregate(self, solns, num_samples):
        # 直接聚合
        self.aggregate_parameters_weighted(solns, num_samples)
        # solns[0] 为 每一个 round 中, global model 的 state_dict, 用 inplace更新

    def solve_epochs(self, round_i, clients: Iterable[TFFClient], num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs
        self.global_model.train()
        num_samples = []
        losses = []
        correct_num = []

        solns = []
        global_state_dict = self.global_model.state_dict()
        for c in clients:
            model_to_client = copy.deepcopy(self.global_model).to(self.device)
            # 保存信息
            stat, delta = c.solve_epochs_delta(round_i, model=model_to_client, global_state=global_state_dict, num_epochs=num_epochs, hide_output=self.quiet)

            num_samples.append(stat['num_samples'])
            losses.append(stat['loss_meter'].sum)
            correct_num.append(stat['acc_meter'].sum)
            # 计算模型之间的差值
            solns.append(delta)
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
        return (global_state_dict, solns), num_samples

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')

            selected_clients = self.select_clients(round_i=round_i, clients_per_rounds=self.clients_per_round)

            solns, num_samples = self.solve_epochs(round_i, clients=selected_clients)

            # 这里直接修改模型
            self.aggregate(solns, num_samples)
            # eval on test
            if round_i % self.eval_on_test_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.test_clients, client_type='test')

            if round_i % self.eval_on_train_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.train_clients, client_type='train')

            if round_i > 0 and round_i % self.save_every_round == 0:
                self.save(round_i)
                self.metrics.write()

        self.metrics.write()
        self.save(self.num_rounds)