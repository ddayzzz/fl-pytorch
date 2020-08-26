import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch import nn
from utils.metrics import Meter
from typing import Dict, Union, Tuple
import copy


class BaseClient(object):

    def __init__(self, id, dataset: Dataset, dataset_type: str, options: dict):
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
        self.num_dataset = len(dataset)
        self.num_workers = options['num_workers']
        self.batch_size = options['train_batch_size'] if dataset_type == 'train' else options['test_batch_size']
        self.device = options['device']
        self.dataset_type = dataset_type
        #
        self.criterion = self.create_criterion()
        # TODO, 待验证, 每个客户端创建一个 optimizer 的对象可能会造成内存泄露!!
        # self.optimizer = self.create_optimizer()
        #
        self.dataset_loader = self.create_data_loader(dataset)

    def __repr__(self):
        return 'BaseClient <id={} dataset_type={}>'.format(self.id, self.dataset_type)

    def create_data_loader(self, dataset) -> DataLoader:
        shuffle = self.dataset_type == 'train'
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True)

    def create_criterion(self):
        return nn.CrossEntropyLoss(reduction='mean').to(self.device)

    def create_optimizer(self, model: nn.Module):
        if self.dataset_type == 'train':
            opt = optim.SGD(model.parameters(), lr=self.options['lr'], momentum=self.options['momentum'],
                            weight_decay=self.options['wd'])
            return opt
        else:
            return None

    def count_correct(self, pred, targets):
        _, predicted = torch.max(pred, 1)
        correct = predicted.eq(targets).sum()
        return correct

    def test(self, model: nn.Module) -> Dict[str, Union[int, Meter]]:
        model.eval()

        loss_meter = Meter()
        acc_meter = Meter()
        num_all_samples = 0

        with torch.no_grad():

            for batch_idx, (X, y) in enumerate(self.dataset_loader):
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
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

    def solve_epochs(self, round_i, model: nn.Module, num_epochs, hide_output: bool = False) -> Tuple[Dict[str, Union[int, Meter]], Dict[str, torch.Tensor]] :
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
        state_dict = model.state_dict()
        # 输出相关的参数
        return {
            'loss_meter': loss_meter,
            'acc_meter': acc_meter,
            'num_samples': num_all_samples
        }, state_dict