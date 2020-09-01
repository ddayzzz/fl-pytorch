import torch
import math
from torch.nn import Module, Parameter
from torch.optim.optimizer import required
from optimizers.adaptive.server_opt import ServerOptimizer, Dict
import warnings


class AdaptiveAdam(ServerOptimizer):
    """
    基于论文Adaptive Federated Optimization实现的服务端的SGD优化器. 如果使用了动量, 那么这个优化算法则被称为 FEDAVGM, 如果动量为0, 则
    退化为普通的 FEDAVG
    """

    def __init__(self, global_model: Module, lr=required, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay: float = 0.0, amsgrad=False, partial_weight_decay=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        super(AdaptiveAdam, self).__init__(global_model, lr=lr,
                                           weight_decay=weight_decay,
                                           more_defaults=dict(eps=eps, betas=betas, amsgrad=amsgrad),
                                           partial_weight_decay=partial_weight_decay)

    def __setstate__(self, state):
        super(AdaptiveAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay_mask = group['weight_decay_mask']
            for param_index, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0 and weight_decay_mask[param_index]:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    @torch.no_grad()
    def step_pseudo_grads(self, pseudo_grads: Dict[str, torch.Tensor]):

        for group in self.param_groups:
            weight_decay_mask = group['weight_decay_mask']
            param_names = group['param_names']
            for param_index, p in enumerate(group['params']):
                grad = pseudo_grads.get(param_names[param_index], None)
                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # 接下来实现的是论文 Algorithm 1 的步骤, 而 tensorflow 则使用的 epsilon hat 的版本. 即eps的初始化为 1e-7的版本
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                # t=t+1
                state['step'] += 1
                # bias-corrected first moment estimate 的分母
                bias_correction1 = 1 - beta1 ** state['step']
                # bias-corrected second moment estimate 的分母
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0 and weight_decay_mask[param_index]:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)


class AdaptiveAdamTF(ServerOptimizer):
    """
    基于论文Adaptive Federated Optimization实现的服务端的SGD优化器. 如果使用了动量, 那么这个优化算法则被称为 FEDAVGM, 如果动量为0, 则
    退化为普通的 FEDAVG
    """

    def __init__(self, global_model: Module, lr=required, betas=(0.9, 0.999), eps=1e-7,
                 weight_decay: float = 0.0, amsgrad=False, partial_weight_decay=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        super(AdaptiveAdamTF, self).__init__(global_model, lr=lr,
                                           weight_decay=weight_decay,
                                           more_defaults=dict(eps=eps, betas=betas, amsgrad=amsgrad),
                                           partial_weight_decay=partial_weight_decay)

    def __setstate__(self, state):
        super(AdaptiveAdamTF, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay_mask = group['weight_decay_mask']
            for param_index, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                # amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # if amsgrad:
                    #     # Maintains max of all exp. moving avg. of sq. grad. values
                    #     state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m_t_minus_1, v_minus_1 = state['m'], state['v']
                # if amsgrad:
                #     max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                alpha_t = group['lr'] * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])

                if group['weight_decay'] != 0 and weight_decay_mask[param_index]:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                m_t_minus_1.mul_(beta1).add_(grad, alpha=1 - beta1)
                v_minus_1.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                #     # Use the max. for normalizing running avg. of gradient
                #     denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                # else:
                #     denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                denom = v_minus_1.sqrt().add_(group['eps'])

                p.addcdiv_(m_t_minus_1, denom, value=-alpha_t)

        return loss

    @torch.no_grad()
    def step_pseudo_grads(self, pseudo_grads: Dict[str, torch.Tensor]):

        for group in self.param_groups:
            weight_decay_mask = group['weight_decay_mask']
            param_names = group['param_names']
            for param_index, p in enumerate(group['params']):
                grad = pseudo_grads.get(param_names[param_index], None)
                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                m_t_minus_1, v_minus_1 = state['m'], state['v']

                beta1, beta2 = group['betas']

                state['step'] += 1
                alpha_t = group['lr'] * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])

                if group['weight_decay'] != 0 and weight_decay_mask[param_index]:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                m_t_minus_1.mul_(beta1).add_(grad, alpha=1 - beta1)
                v_minus_1.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = v_minus_1.sqrt().add_(group['eps'])

                p.addcdiv_(m_t_minus_1, denom, value=-alpha_t)


if __name__ == '__main__':
    # 经过验证, 这个方法和 tensorflow 2.3 的效果相同, keras 的 L2 API中没有除以2
    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.param = Parameter(torch.ones((1, )))

        def forward(self, data):
            return self.param * 1.0
    model = Model()
    opt = AdaptiveAdamTF(model, lr=0.1, weight_decay=0.01, eps=0.1)
    var0 = model.param.item()
    for i in range(1, 5):
        opt.zero_grad()
        loss = (model(None) ** 2) / 2.0
        loss.backward()
        opt.step()
        var1 = model.param.item()
        print('diff: ', var0 - var1)
        var0 = var1


