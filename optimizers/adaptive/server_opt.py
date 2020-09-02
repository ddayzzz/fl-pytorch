import torch
from torch.nn import Module
import abc
from typing import Dict, Optional, Callable
from torch.optim.optimizer import Optimizer, required
import math


def exponential_learning_rate_decay(base_lr, decay_rate, decay_steps, staircase):
    def call(round_i):
        if staircase:
            return base_lr * (decay_rate ** (round_i // decay_steps))
        else:
            return base_lr * (decay_rate ** (round_i / decay_steps))
    return call

def inverse_linear_decay_learning_rate(base_lr, decay_rate, decay_steps, staircase):
    def call(round_i):
        if staircase:
            return base_lr / (1.0 + decay_rate * (round_i // decay_steps))
        else:
            return base_lr / (1.0 + decay_rate * (round_i / decay_steps))
    return call

def inverse_sqrt_decay_learning_rate(base_lr, decay_rate, decay_steps, staircase):
    def call(round_i):
        if staircase:
            return base_lr / math.sqrt(1.0 + decay_rate * (round_i // decay_steps))
        else:
            return base_lr / math.sqrt(1.0 + decay_rate * (round_i / decay_steps))
    return call


def warmup_learning_rate(base_lr: float, warmup_steps: int, fn: Callable[[int], float]) -> Callable[[int], float]:
    if warmup_steps is None or warmup_steps <= 0:
        def call(round_i) -> float:
            return fn(round_i)
        return call
    else:
        def warmup_and_decay_fn(round_num) -> float:
            warmedup_value = base_lr * (round_num + 1) / warmup_steps
            if round_num < warmup_steps:
                return warmedup_value
            else:
                return fn(round_num - warmup_steps)
        return warmup_and_decay_fn


class AdaptiveOptimizer(Optimizer):
    """
    基于论文 Adaptive Federated Optimization实现的服务端的优化器
    """

    def __init__(self, global_model: Module,
                 lr=required,
                 lr_decay_policy: str='constant',
                 decay_rate=None,
                 decay_steps=None,
                 staircase=False,
                 warmup_steps=None,
                 more_defaults: Optional[Dict]=None,
                 weight_decay=0,
                 partial_weight_decay=False):
        """
        创建对象
        :param global_model: 模型
        :param lr:
        :param momentum:
        :param dampening:
        :param weight_decay:
        :param nesterov:
        """
        if lr_decay_policy == 'constant':
            scheduler = warmup_learning_rate(lr, warmup_steps=warmup_steps, fn=lambda _: lr)
        elif lr_decay_policy == 'exp_decay':
            scheduler = warmup_learning_rate(lr, warmup_steps, exponential_learning_rate_decay(lr, decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase))
        elif lr_decay_policy == 'inv_lin':
            scheduler = warmup_learning_rate(lr, warmup_steps, inverse_linear_decay_learning_rate(lr, decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase))
        elif lr_decay_policy == 'inv_sqrt':
            scheduler = warmup_learning_rate(lr, warmup_steps, inverse_sqrt_decay_learning_rate(lr, decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase))
        else:
            raise ValueError(
                'Unrecognized schedule type {!s}'.format(lr_decay_policy))
        self.scheduler = scheduler
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        # 获取模型参数的相关信息, 记录了当前参数的名称, 方便从计算后的 weight diff 中找到对应的参数名
        param_names = []
        params = []
        weight_decay_mask = []
        for k, v in global_model.named_parameters():
            param_names.append(k)
            params.append(v)
            # 检查相关的参数是否需要 wd
            if getattr(v, 'enable_weight_decay', False):
                weight_decay_mask.append(True)
            else:
                if partial_weight_decay:
                    weight_decay_mask.append(False)
                else:
                    weight_decay_mask.append(True)
        defaults.update(param_names=param_names, weight_decay_mask=weight_decay_mask)
        if more_defaults is not None:
            defaults.update(more_defaults)
        super(AdaptiveOptimizer, self).__init__(params, defaults)

    def step_lr_scheduler(self, round_i):
        lr = self.scheduler(round_i)
        for p in self.param_groups:
            p['lr'] = lr

    def get_current_lr(self):
        return self.param_groups[0]['lr']

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplemented('Must implement the optimizer steps')

    @abc.abstractmethod
    def step_pseudo_grads(self, pseudo_grads: Dict[str, torch.Tensor]):
        """
        :param pseudo_grads: 加权聚合后的模型的参数(在联邦学习中包括了buffer和param)
        :return:
        """
        pass


if __name__ == '__main__':
    scheduler = warmup_learning_rate(1.0, 10, exponential_learning_rate_decay(1.0, decay_rate=0.9,
                                                                                       decay_steps=10,
                                                                                       staircase=False))
    lr0 = scheduler(0)
    lr1 = scheduler(9)
    lr2 = scheduler(20)
    print(lr0, lr1, lr2)