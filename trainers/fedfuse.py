import torch
import time
import tqdm
import numpy as np
from typing import List
from torch.nn import Module
from torch.utils.data import Dataset
from typing import OrderedDict, Dict, List, Any, Union, Iterable, Tuple
import collections
from clients.base_client import BaseClient, get_model_parameters_dict, set_model_parameters_dict
from trainers.fedbase import FedBase
from utils.metrics import Meter
import pandas as pd


class FedFuseClient(BaseClient):

    def __init__(self, id, model: Module, dataset, dataset_type, options):
        super(FedFuseClient, self).__init__(id, model, dataset, dataset_type, options)
        self.fuse_operator = options['operator']
        # attention 层的参数
        self.client_fuse_operator = [p.data.clone() for k, p in self.model.attn.named_parameters()]

    def create_optimizer(self):
        from torch import optim
        if self.dataset_type == 'train':
            opt = optim.SGD(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.options['lr'],
                            momentum=self.options['momentum'],
                            weight_decay=self.options['wd'])
            return opt
        else:
            return None

    def backup_attention(self, mu=0.9):
        copied = [p.data.clone() for p in self.model.attn.parameters()]
        if self.fuse_operator in ['multi', 'single']:
            # 需要手动 weight_decay
            # TODO 这两个 operator 都是一个参数, λ
            copied = [(1 - mu) * copied[0] + mu * self.client_fuse_operator[0]]
        self.client_fuse_operator = copied

    def recover_client_data(self):
        for p, src in zip(self.model.attn.parameters(), self.client_fuse_operator):
            p.data.copy_(src.data)


class FedFuse(FedBase):

    def __init__(self, options, dataset_info, model):
        self.operator = options['operator']
        a = f'policy[{self.operator}]'
        print('>>> Using FedFuse, use operator :', self.operator)
        super(FedFuse, self).__init__(options=options, model=model, dataset_info=dataset_info, append2metric=a)

    def set_latest_model(self):
        set_model_parameters_dict(self.model, self.latest_model)

    def get_latest_model(self) -> Dict[str, torch.Tensor]:
        return get_model_parameters_dict(self.model)

    def create_clients_group(self, users: Iterable[Any], train_or_test_dataset_obj: Dict[Any, Dataset], dataset_type) -> OrderedDict[Any, FedFuseClient]:
        all_clients = collections.OrderedDict()
        for user in users:
            c = FedFuseClient(id=user, model=self.model, dataset=train_or_test_dataset_obj[user], dataset_type=dataset_type, options=self.options)
            all_clients[user] = c
        return all_clients

    def aggregate(self, solns, num_samples):
        return self.aggregate_parameters_weighted(solns, num_samples)

    def solve_epochs(self,round_i, clients: Iterable[FedFuseClient], num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        num_samples = []
        losses = []
        correct_num = []

        solns = []
        for i, c in enumerate(clients):
            # 设置全局的参数
            self.set_latest_model()

            if i == 0:
                # TODO 第一次设置就行, 模型中都是 deepcopy
                # 设置 latest model, 这个参数不会被更新, 当前的模型正好是全局的参数
                self.model.set_global_model(self.model, device=self.device)

            c.recover_client_data()
            stat = c.solve_epochs(round_i, num_epochs, hide_output=self.quiet)

            num_samples.append(stat['num_samples'])
            losses.append(stat['loss_meter'].sum)
            correct_num.append(stat['acc_meter'].sum)
            #
            soln = c.get_model_parameters_dict()
            solns.append(soln)
            # 保存当前客户端的 fuse 或者 conv 算子的参数
            c.backup_attention()

        mean_loss = sum(losses) / sum(num_samples)
        mean_acc = sum(correct_num) / sum(num_samples)

        stats = {
            'acc': mean_acc, 'loss': mean_loss,
        }
        if not self.quiet:
            print(f'Round {round_i}, train metric mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        self.metrics.update_train_stats(round_i, stats)
        return solns, num_samples

    def eval_on(self, round_i, clients: Iterable[FedFuseClient], client_type):
        df = pd.DataFrame(columns=['client_id', 'mean_acc', 'mean_loss', 'num_samples'])

        num_samples = []
        losses = []
        correct_num = []
        for i, c in enumerate(clients):
            # 设置网络
            self.set_latest_model()
            if i == 0:
                # 设置当前最新的模型, global 模型设置一次即可, 这个值是不会变的
                self.model.set_global_model(self.model, device=self.device)
            c.recover_client_data()
            stats = c.test()

            num_samples.append(stats['num_samples'])
            losses.append(stats['loss_meter'].sum)
            correct_num.append(stats['acc_meter'].sum)
            #
            df = df.append({'client_id': c.id, 'mean_loss': stats['loss_meter'].avg, 'mean_acc': stats['acc_meter'].avg,
                            'num_samples': stats['num_samples'], }, ignore_index=True)

        # ids = [c.id for c in self.clients]
        # groups = [c.group for c in self.clients]
        all_sz = sum(num_samples)
        mean_loss = sum(losses) / all_sz
        mean_acc = sum(correct_num) / all_sz
        #
        if not self.quiet:
            print(
                f'Round {round_i}, eval on "{client_type}" client mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        # round_i, on_which, filename, other_to_logger
        self.metrics.update_eval_stats(round_i=round_i, on_which=client_type,
                                       other_to_logger={'acc': mean_acc, 'loss': mean_loss}, df=df)

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')

            selected_clients = self.select_clients(round_i=round_i, clients_per_rounds=self.clients_per_round)
            # TODO 这里不修改签名
            solns, num_samples = self.solve_epochs(round_i, clients=selected_clients)

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