from trainers.fedbase import FedBase
from clients.segmentation_client import SegmentationClient
import torch
from torch.utils.data import Dataset
import copy
from typing import Iterable, Dict, List, Tuple, Union, OrderedDict, Any
from utils.metrics import Meter
from torch.nn import Module
import pandas as pd
import tqdm
import collections
from torch import optim
from optimizers.adaptive.sgd import AdaptiveSGD

STATE_TYPE = Dict[str, torch.Tensor]


class TFFClient(SegmentationClient):

    def __init__(self, id, dataset, dataset_type, options):
        super(TFFClient, self).__init__(id, dataset, dataset_type, options)

    def create_optimizer(self, model: Module):
        if self.dataset_type == 'train':
            # opt = optim.SGD(model.parameters(), lr=self.options['client_lr'], momentum=0, weight_decay=self.options['wd'])
            # 客户端默认都有使用不带有 momentum 的 SGD
            opt = AdaptiveSGD(model, lr=self.options['client_lr'], momentum=0, weight_decay=self.options['wd'],
                              partial_weight_decay=False,
                              lr_decay_policy=self.options['lr_decay_policy'],
                              decay_steps=self.options['lr_decay_steps'],
                              decay_rate=self.options['lr_decay_rate'],
                              staircase=self.options['lr_staircase'],
                              warmup_steps=self.options['lr_warmup_steps']
                              )
            return opt
        else:
            return None

    def solve_epochs_delta(self, round_i, model: Module, global_state: STATE_TYPE, num_epochs,
                           hide_output: bool = False) -> Tuple[Dict[str, Union[int, Meter]], Dict[str, torch.Tensor]]:
        loss_meter = Meter()
        dice_meter = Meter('dice_coeff')
        num_all_samples = 0
        optimizer = self.create_optimizer(model)
        optimizer.step_lr_scheduler(round_i=round_i)
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
                    #
                    activated = torch.sigmoid(pred)
                    dice_coeff = self.compute_dice_coefficient(activated, y)
                    #
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                    optimizer.step()
                    num_samples = y.size(0)
                    num_all_samples += num_samples
                    loss_meter.update(loss.item(), n=num_samples)
                    dice_meter.update(dice_coeff.item(), n=num_samples)
                    if (batch_idx % 10 == 0):
                        # 纯数值, 这里使用平均的损失
                        t.set_postfix(mean_loss=loss.item())
        # 返回参数
        state_dict = model.state_dict()
        result = {
            'loss_meter': loss_meter,
            'dice_coeff_meter': dice_meter,
            'num_samples': num_all_samples,
            'lr': optimizer.get_current_lr()
        }
        # 计算差值 latest  - init
        for k, v in state_dict.items():
            v.sub_(global_state[k])
        # 输出相关的参数
        return result, state_dict


class AdaptiveOptimization(FedBase):

    def __init__(self, options, dataset_info, model):
        """
        类似于 Tensorflow-federated 形式的FedAvg
        区别在于其更新的方式是增量更新的形式 https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_2
        :param options:
        :param dataset_info:
        :param model:
        """
        self.server_lr = options['server_lr']
        self.client_lr = options['client_lr']
        self.wd = options['wd']
        server_optimizer_name = options['server_optimizer']
        self.adaptive_momentum = options['adaptive_momentum']
        target_optimizer = None
        target_opt_name = None
        opt_suffix = ''
        if server_optimizer_name == 'sgd':
            # FedAvg 和 FedAvgM
            if self.adaptive_momentum > 0:
                target_opt_name = 'fedavgm'
                opt_suffix = 'momentum_[{}]'.format(self.adaptive_momentum)
            else:
                target_opt_name = 'fedavg'
            target_optimizer = lambda: AdaptiveSGD(self.global_model, lr=self.server_lr,
                                                   momentum=self.adaptive_momentum, weight_decay=self.wd,
                                                   partial_weight_decay=False,
                                                   lr_decay_policy=options['lr_decay_policy'],
                                                   decay_steps=options['lr_decay_steps'],
                                                   decay_rate=options['lr_decay_rate'],
                                                   staircase=options['lr_staircase'],
                                                   warmup_steps=options['lr_warmup_steps'])
        elif server_optimizer_name == 'adam':
            from optimizers.adaptive.adam import AdaptiveAdamTF
            epsilon = options['adaptive_epsilon']
            target_opt_name = 'fedadam'
            target_optimizer = lambda: AdaptiveAdamTF(self.global_model, lr=self.server_lr, weight_decay=self.wd,
                                                      partial_weight_decay=False, eps=epsilon,
                                                      lr_decay_policy=options['lr_decay_policy'],
                                                      decay_steps=options['lr_decay_steps'],
                                                      decay_rate=options['lr_decay_rate'],
                                                      staircase=options['lr_staircase'],
                                                      warmup_steps=options['lr_warmup_steps']
                                                      )
            opt_suffix = f'epsilon[{epsilon}]'
        else:
            raise NotImplemented
        a = 'client_lr[{}]_server_lr[{}]_server_opt[{}]_wd[{}]_{}'.format(self.client_lr, self.server_lr,
                                                                          target_opt_name, self.wd, opt_suffix)
        super(AdaptiveOptimization, self).__init__(options=options, model=model, dataset_info=dataset_info,
                                                   append2metric=a)
        print('>>> Use ', target_opt_name, ' as server optimizer, its params: ', opt_suffix)
        self.server_optimizer = target_optimizer()

    def create_clients_group(self, users: Iterable[Any], train_or_test_dataset_obj: Dict[Any, Dataset], dataset_type) -> \
    OrderedDict[Any, TFFClient]:
        all_clients = collections.OrderedDict()
        for user in users:
            c = TFFClient(id=user, dataset=train_or_test_dataset_obj[user], dataset_type=dataset_type,
                          options=self.options)
            all_clients[user] = c
        return all_clients

    def aggregate_parameters_weighted(self, solns: List[STATE_TYPE], num_samples: List[int]):
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
        weighted_params = super(AdaptiveOptimization, self).aggregate_parameters_weighted(solns=solns,
                                                                                          num_samples=num_samples)
        for p, v in weighted_params.items():
            v.mul_(-1.0)
        # 这里使用的新的参数, 调用 global_model 的 state_dict(其使用 detach, 使用 in-place操作)
        # global_state = solns[0]
        # for k, v in global_state.items():
        #     # θ = θ - server_lr * (-1.0 * θ^-)
        #     v.add_(weighted_params[k].mul_(-1.0), alpha=-self.server_lr)
        # 这列改为使用 server optimizer
        self.server_optimizer.step_pseudo_grads(pseudo_grads=weighted_params)

    def aggregate(self, solns, num_samples):
        # 直接聚合
        self.aggregate_parameters_weighted(solns, num_samples)
        # solns[0] 为 每一个 round 中, global model 的 state_dict, 用 inplace更新

    def eval_on(self, round_i, clients: Iterable[TFFClient], client_type):
        df = pd.DataFrame(columns=['client_id', 'mean_dice_coeff', 'mean_loss', 'num_samples'])

        num_samples = []
        losses = []
        dice_coef = []
        for c in clients:
            # 设置网络
            stats = c.test(self.global_model)

            num_samples.append(stats['num_samples'])
            losses.append(stats['loss_meter'].sum)
            dice_coef.append(stats['dice_coeff_meter'].sum)
            #
            df = df.append({'client_id': c.id, 'mean_loss': stats['loss_meter'].avg, 'mean_dice_coeff': stats['dice_coeff_meter'].avg,
                            'num_samples': stats['num_samples'], }, ignore_index=True)

        # ids = [c.id for c in self.clients]
        # groups = [c.group for c in self.clients]
        all_sz = sum(num_samples)
        mean_loss = sum(losses) / all_sz
        mean_dice_coeff = sum(dice_coef) / all_sz
        #
        if not self.quiet:
            print(
                f'Round {round_i}, eval on "{client_type}" client\n'
                f'\tmean loss: {mean_loss:.5f}\n'
                f'\tmean dice coeff: {mean_dice_coeff:.5f}\n')

        # round_i, on_which, filename, other_to_logger
        self.metrics.update_eval_stats(round_i=round_i, on_which=client_type,
                                       other_to_logger={'dice_coef': mean_dice_coeff, 'loss': mean_loss}, df=df)

    def solve_epochs(self, round_i, clients: Iterable[TFFClient], num_epochs=None) -> Tuple[List[STATE_TYPE], List[int]]:
        server_lr = self.server_optimizer.get_current_lr()
        if num_epochs is None:
            num_epochs = self.num_epochs
        self.global_model.train()
        num_samples = []
        losses = []
        #
        dice_coef = []

        solns = []
        global_state_dict = self.global_model.state_dict()
        for c in clients:
            model_to_client = copy.deepcopy(self.global_model).to(self.device)
            # 保存信息
            stat, delta = c.solve_epochs_delta(round_i, model=model_to_client, global_state=global_state_dict,
                                               num_epochs=num_epochs, hide_output=self.quiet)

            num_samples.append(stat['num_samples'])
            losses.append(stat['loss_meter'].sum)
            dice_coef.append(stat['dice_coeff_meter'].sum)
            # 计算模型之间的差值
            solns.append(delta)
            # 写入测试的相关信息
            # self.metrics.update_commu_stats(round_i, flop_stat)
        client_lr = stat['lr']
        mean_loss = sum(losses) / sum(num_samples)
        mean_dice_coef = sum(dice_coef) / sum(num_samples)

        stats = {
            'dice_coef': mean_dice_coef, 'loss': mean_loss,
            'client_lr': client_lr, 'server_lr': server_lr
        }
        if not self.quiet:
            print(f'Round {round_i}, train metric\n'
                  f'\tmean loss: {mean_loss:.5f}\n'
                  f'\tmean dice coef: {mean_dice_coef:.5f}')
        self.metrics.update_train_stats(round_i, stats)
        # 修改server 的 lr
        self.server_optimizer.step_lr_scheduler(round_i=round_i)
        return solns, num_samples

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

            if round_i != 0 and round_i % self.eval_on_train_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.train_clients, client_type='train')

            if round_i > 0 and round_i % self.save_every_round == 0:
                self.save(round_i)
                self.metrics.write()

        self.metrics.write()
        self.save(self.num_rounds)
