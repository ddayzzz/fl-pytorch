import torch
from torch.nn import Module, Parameter
from torch.optim.optimizer import required
from optimizers.adaptive.server_opt import AdaptiveOptimizer, Dict
import warnings


class AdaptiveSGD(AdaptiveOptimizer):
    """
    基于论文Adaptive Federated Optimization实现的服务端的SGD优化器. 如果使用了动量, 那么这个优化算法则被称为 FEDAVGM, 如果动量为0, 则
    退化为普通的 FEDAVG
    """

    def __init__(self, global_model: Module, lr=required, lr_decay_policy: str='constant',
                 decay_rate=None,
                 decay_steps=None,
                 staircase=False,
                 warmup_steps=None,
                 momentum: float=0, dampening=0, weight_decay=0.0, nesterov=False, partial_weight_decay=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AdaptiveSGD, self).__init__(global_model, lr=lr,
                                          more_defaults=dict(momentum=momentum, dampening=dampening, nesterov=nesterov),
                                          weight_decay=weight_decay,
                                          partial_weight_decay=partial_weight_decay,
                                          lr_decay_policy=lr_decay_policy,
                                          decay_rate=decay_rate,
                                          decay_steps=decay_steps,
                                          staircase=staircase,
                                          warmup_steps=warmup_steps)

    def __setstate__(self, state):
        super(AdaptiveSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            weight_decay_mask = group['weight_decay_mask']
            # param_names = group['param_names']
            for param_index, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0 and weight_decay_mask[param_index]:
                    d_p = d_p.add(p, alpha=weight_decay)
                    # print('APPLY WD TO ', param_names[param_index])
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        # buffer中克隆出当前的初始的梯度
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        # v_{t+1} = v_t * momentum + grad
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss

    @torch.no_grad()
    def step_pseudo_grads(self, pseudo_grads: Dict[str, torch.Tensor]):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            param_names = group['param_names']
            weight_decay_mask = group['weight_decay_mask']
            for param_index, p in enumerate(group['params']):
                d_p = pseudo_grads.get(param_names[param_index], None)
                if d_p is None:
                    # 模型的参数可能存在冻结的情况
                    continue
                if weight_decay != 0 and weight_decay_mask[param_index]:
                    d_p = d_p.add(p, alpha=weight_decay)
                    # print('APPLY WD TO ', param_names[param_index])
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        # buffer中克隆出当前的初始的梯度
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        # v_{t+1} = v_t * momentum + grad
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])


if __name__ == '__main__':
    # 经过验证, 这个方法和 tensorflow 2.3 的效果相同, keras 的 L2 API中没有除以2
    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.param = Parameter(torch.ones((1, )))

        def forward(self, data):
            return self.param * 1.0
    model = Model()
    opt = AdaptiveSGD(model, lr=0.1, weight_decay=0.01, momentum=0.9)
    var0 = model.param.item()
    loss = (model(None) ** 2) / 2.0
    opt.zero_grad()
    loss.backward()
    opt.step()
    var1 = model.param.item()
    print(f'var0-var1: {var0 - var1}')

    opt.zero_grad()
    loss = (model(None) ** 2) / 2.0
    loss.backward()
    opt.step()

    var2 = model.param.item()
    print(f'var1-var2: {var1 - var2}')

