import tqdm
import torch
import collections
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from utils.metrics import Meter
from typing import Dict, Union


class BaseClient(object):

    def __init__(self, id, model: nn.Module, train_dataset, test_dataset, options: dict, validation_dataset=None):
        """

        :param id:
        :param model:
        :param train_dataset:
        :param test_dataset:
        :param options:
        :param validation_dataset:
        """
        self.id = id
        self.options = options
        # 这个必须是客户端相关的
        self.num_train_dataset = len(train_dataset)
        self.num_test_dataset = len(test_dataset)
        if validation_dataset is not None:
            self.num_validation_dataset = len(validation_dataset)
        self.num_workers = options['num_workers']
        self.batch_size = options['batch_size']
        self.device = options['device']
        self.model = model
        #
        self.criterion = self.create_criterion()
        self.optimizer = self.create_optimizer()
        #
        self.train_dataset_loader = self.create_data_loader(train_dataset, dataset_type='train')
        self.test_dataset_loader = self.create_data_loader(test_dataset, dataset_type='test')
        if validation_dataset is not None:
            self.validation_dataset_loader = self.create_data_loader(validation_dataset, dataset_type='valid')

    def __repr__(self):
        return 'BaseClient <id={}>'.format(self.id)

    def create_data_loader(self, dataset, dataset_type):
        shuffle = dataset_type == 'train'
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True)

    def create_criterion(self):
        return nn.CrossEntropyLoss(reduction='mean').to(self.device)

    def create_optimizer(self):
        opt = optim.SGD(self.model.parameters(), lr=self.options['lr'], momentum=0.5, weight_decay=self.options['wd'])
        return opt

    def get_model_parameters_dict(self) -> Dict[str, torch.Tensor]:
        result = dict()
        for k, v in self.model.named_parameters():
            result[k] = v.detach().clone()
        return result

    def set_model_paramters_dict(self, parameters_dict : Dict[str, torch.Tensor]):
        # torch 自带 copy 的方法
        for k, v in self.model.named_parameters():
            # 见数据复制到 model 的位置, 二者是不共享位置的
            v.data.copy_(parameters_dict[k])


    def count_correct(self, pred, targets):
        _, predicted = torch.max(pred, 1)
        correct = predicted.eq(targets).sum()
        return correct

    def test(self, dataset_loader) -> Dict[str, Union[int, Meter]]:
        self.model.eval()

        loss_meter = Meter()
        acc_meter = Meter()
        num_all_samples = 0

        with torch.no_grad():

            for batch_idx, (X, y) in enumerate(dataset_loader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, y)
                correct = self.count_correct(pred, y)
                #
                num_samples = y.size(0)
                loss_meter.update(loss.item(), n=num_samples)
                acc_meter.update(correct.item() / num_samples, n=num_samples)
                num_all_samples += num_samples

        return {
            'loss_meter': loss_meter,
            'acc_meter': acc_meter,
            'num_samples': num_all_samples
        }

    def solve_epochs(self, round_i, data_loader, num_epochs, hide_output: bool = False) -> Dict[str, Union[int, Meter]]:
        loss_meter = Meter()
        acc_meter = Meter()
        num_all_samples = 0
        self.model.train()

        with tqdm.trange(num_epochs, disable=hide_output) as t:

            for epoch in t:
                t.set_description(f'Client: {self.id}, Round: {round_i}, Epoch :{epoch}')
                for batch_idx, (X, y) in enumerate(data_loader):
                    # from IPython import embed
                    X, y = X.to(self.device), y.to(self.device)

                    self.optimizer.zero_grad()
                    pred = self.model(X)

                    loss = self.criterion(pred, y)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                    self.optimizer.step()

                    correct_sum = self.count_correct(pred, y)
                    num_samples = y.size(0)
                    num_all_samples += num_samples
                    loss_meter.update(loss.item(), n=num_samples)
                    acc_meter.update(correct_sum.item() / num_samples, n=num_samples)
                    if (batch_idx % 10 == 0):
                        # 纯数值, 这里使用平均的损失
                        t.set_postfix(mean_loss=loss.item())

        # 输出相关的参数
        return {
            'loss_meter': loss_meter,
            'acc_meter': acc_meter,
            'num_samples': num_all_samples
        }





