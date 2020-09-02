from clients.base_client import BaseClient, Dataset, Meter, DataLoader
from typing import Dict, Union, Tuple
import tqdm
import torch
from torch.nn import Module
from utils.losses import BCEDiceLoss

class SegmentationClient(BaseClient):

    def __init__(self, id, dataset: Dataset, dataset_type: str, options: dict):
        super(SegmentationClient, self).__init__(id=id, dataset=dataset, dataset_type=dataset_type, options=options)

    def __repr__(self):
        return 'SegmentationClient <id={} dataset_type={}>'.format(self.id, self.dataset_type)

    def create_criterion(self):
        return BCEDiceLoss().to(self.device)

    @staticmethod
    def compute_dice_coefficient(pred: torch.Tensor, ground_truth: torch.Tensor, smooth: float=1e-5):
        """
        计算 dice coefficient
        :param pred: 激活后值
        :param ground_truth: Mask(0-1)
        :return:
        """
        # 样本的数量
        # 输入的样本格式应该是: [batch_size, patch_size, 4, H, W]
        num = ground_truth.size(0)
        x = pred.view(num, -1)
        target = ground_truth.view(num, -1)
        intersection = (x * target)
        dice = (2. * intersection.sum(1) + smooth) / (x.sum(1) + target.sum(1) + smooth)
        # 在 batch 取得均值
        return dice.sum() / num

    def test(self, model: Module) -> Dict[str, Union[int, Meter]]:
        model.eval()

        loss_meter = Meter('bce_dice_loss')
        dice_meter = Meter('dice_coeff')
        num_all_samples = 0

        with torch.no_grad():

            for batch_idx, (X, y) in enumerate(self.dataset_loader):
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                activated = torch.sigmoid(pred)
                loss = self.criterion(pred, y)
                #
                dice_coeff = self.compute_dice_coefficient(activated, ground_truth=y)
                #
                num_samples = y.size(0)
                loss_meter.update(loss.item(), n=num_samples)
                dice_meter.update(dice_coeff.item(), n=num_samples)
                num_all_samples += num_samples

        return {
            'loss_meter': loss_meter,
            'dice_coeff_meter': dice_meter,
            'num_samples': num_all_samples
        }

    def solve_epochs(self, round_i, model: Module, num_epochs, hide_output: bool = False) -> Tuple[Dict[str, Union[int, Meter]], Dict[str, torch.Tensor]] :
        loss_meter = Meter()
        dice_meter = Meter('dice_coeff')
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
        # 输出相关的参数
        return {
            'loss_meter': loss_meter,
            'dice_coeff_meter': dice_meter,
            'num_samples': num_all_samples
        }, state_dict