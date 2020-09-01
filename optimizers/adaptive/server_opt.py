import torch
from torch.nn import Module
import abc
from typing import Dict, Optional
from torch.optim.optimizer import Optimizer, required
from torch.nn import Conv2d



class ServerOptimizer(Optimizer):
    """
    基于论文 Adaptive Federated Optimization实现的服务端的优化器
    """

    def __init__(self, global_model: Module, lr=required, more_defaults: Optional[Dict]=None, weight_decay=0, partial_weight_decay=False):
        """
        创建对象
        :param global_model: 模型
        :param lr:
        :param momentum:
        :param dampening:
        :param weight_decay:
        :param nesterov:
        """
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
        super(ServerOptimizer, self).__init__(params, defaults)

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