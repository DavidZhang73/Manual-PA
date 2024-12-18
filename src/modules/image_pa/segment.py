from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics import MetricCollection

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False)

from src.metrics import AP, MIoU
from src.models import SegmentNet
from src.utils import colors


@dataclass
class BatchData:
    shape_id: list[int]
    view_id: list[int]
    step_id: list[int]
    part_ids: list[list[int]]
    total_parts_cnt: list[int]
    img: torch.Tensor | None = None  # a tensor of B x C x H x W
    pts: list[torch.Tensor] | None = None  # a list of B tensors of shape N_i x 1000 x 3
    masks: list[torch.Tensor] | None = None  # a list of B tensors of shape N_i x 224 x 224
    similar_parts_cnt: torch.Tensor | None = None  # a tensor of B x N_i x 1
    bbox_size: list[torch.Tensor] | None = None  # a list of B tensors of shape N_i x 3
    ins_one_hot: torch.Tensor | None = None  # a tensor of B x N_i x 20

    @property
    def batch_size(self):
        return len(self.shape_id)

    @property
    def max_parts_cnt(self):
        """max(N_i)"""
        return max(self.total_parts_cnt)

    @property
    def batch_total_parts_cnt(self):
        """sum(N_i)"""
        return sum(self.total_parts_cnt)

    # Padded tensors
    @property
    def pts_padded(self):
        """Returns a padded tensor of shape B x max(N_i) x 1000 x 3"""
        return torch.nn.utils.rnn.pad_sequence(self.pts, batch_first=True, padding_value=0)

    @property
    def masks_padded(self):
        """Returns a padded tensor of shape B x max(N_i) x 224 x 224"""
        return torch.nn.utils.rnn.pad_sequence(self.masks, batch_first=True, padding_value=0)

    @property
    def similar_parts_cnt_padded(self):
        """Returns a padded tensor of shape B x max(N_i) x 1"""
        return torch.nn.utils.rnn.pad_sequence(self.similar_parts_cnt, batch_first=True, padding_value=0)

    @property
    def bbox_size_padded(self):
        """Returns a padded tensor of shape B x max(N_i) x 3"""
        return torch.nn.utils.rnn.pad_sequence(self.bbox_size, batch_first=True, padding_value=0)

    @property
    def ins_one_hot_padded(self):
        """Returns a padded tensor of shape B x max(N_i) x 20"""
        return torch.nn.utils.rnn.pad_sequence(self.ins_one_hot, batch_first=True, padding_value=0)

    # Packed tensors
    @property
    def img_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 3 x H x W"""
        imgs = []
        for i, cnt in enumerate(self.total_parts_cnt):
            imgs.append(self.img[i].repeat(cnt, 1, 1, 1))
        return torch.cat(imgs, dim=0)

    @property
    def pts_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 1000 x 3"""
        return torch.cat(self.pts, dim=0)

    @property
    def masks_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 224 x 224"""
        return torch.cat(self.masks, dim=0)

    @property
    def similar_parts_cnt_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 1"""
        return torch.cat(self.similar_parts_cnt, dim=0)

    @property
    def bbox_size_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 3"""
        return torch.cat(self.bbox_size, dim=0)

    @property
    def ins_one_hot_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 20"""
        return torch.cat(self.ins_one_hot, dim=0)


class ModelOutput:
    def __init__(self, data: BatchData, raw_output: tuple[torch.Tensor, ...], linear_assignment_fn: callable):
        self.data = data
        # Perform the Hungarian Matching algorithm to match the ground truth masks to the predicted masks.
        # Note that we prepend a mask to represent the background.
        self.matched_pred_masks: list[torch.Tensor] = []  # a list of B tensors of shape (1 + N_i) x H x W
        self.matched_gt_masks: list[torch.Tensor] = []  # a list of B tensors of shape (1 + N_i) x H x W
        for i in range(data.batch_size):
            pred_mask = raw_output[i][:-1]
            gt_mask = data.masks[i].float()

            simlar_cnt = data.similar_parts_cnt[i]
            matched_gt_ids, matched_pred_ids = linear_assignment_fn(
                gt_mask,
                pred_mask,
                simlar_cnt,
            )

            matched_pred_mask = torch.cat([raw_output[i][-1].unsqueeze(0), pred_mask[matched_pred_ids]], dim=0)
            matched_gt_mask = torch.cat([torch.zeros_like(gt_mask[0]).unsqueeze(0), gt_mask[matched_gt_ids]], dim=0)

            self.matched_pred_masks.append(matched_pred_mask)
            self.matched_gt_masks.append(matched_gt_mask)


Conf = namedtuple(
    "Conf",
    [
        "img_size",
        "ins_dim",
        "pointnet_emd_dim",
        "normalize",
        "device",
    ],
)


class ImagePASegmentNetModule(pl.LightningModule):
    def __init__(
        self,
        # model params
        pointnet_emd_dim: int = 512,
        max_num_parts: int = 20,
        max_num_similar_parts: int = 20,
        img_size: int = 224,
        # visualization
        vis_shape_ids: list[int] = [
            # PartNet
            ## Chair
            3069,
            43941,
            42975,
            38098,
            ## Table
            24686,
            21572,
            25309,
            19314,
            ## Storage
            46703,
            48834,
            46127,
            47808,
            # Ikea Manual
            1000,
            1001,
            1002,
            1003,
        ],
        vis_every_n_epoch: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        conf = Conf(
            img_size=img_size,
            ins_dim=max_num_similar_parts,
            pointnet_emd_dim=pointnet_emd_dim,
            normalize=False,  # we have already normalized the data
            device="cpu",
        )
        self.model = SegmentNet(
            conf=conf,
            partleaf_cnt=max_num_parts,
        )
        metrics = MetricCollection({"AP": AP(), "mIoU": MIoU()}, compute_groups=False)
        self.validate_metrics = metrics.clone(prefix="validate/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Visualization
        self.first_visualization = True  # used for logging ground truth masks only once

    def forward(self, data: BatchData):
        output: tuple[torch.Tensor, ...] = self.model(
            img=data.img_packed,  # sum(N_i) x 3 x H x W
            pc=data.pts_packed,  # sum(N_i) x 1000 x 3
            ins_feat=data.ins_one_hot_packed,  # sum(N_i) x 20
            part_cnt=data.total_parts_cnt,  # list of N_i
        )
        return output  # a tuple of B tensors of shape (N_i+1) x H x W, which has been softmaxed

    def calculate_loss(self, output: ModelOutput):
        loss = torch.tensor(0.0, device=self.device)
        for i in range(output.data.batch_size):
            matched_pred_mask = output.matched_pred_masks[i][1:]  # exclude the background
            matched_gt_mask = output.matched_gt_masks[i][1:]  # exclude the background
            matched_mask_loss = self.model.get_mask_loss(matched_pred_mask, matched_gt_mask)
            loss += matched_mask_loss.mean()
        loss = loss / output.data.batch_size
        self.log(f"{self.stage}/loss", loss.item())
        return loss

    @torch.no_grad()
    def calculate_metrics(self, output: ModelOutput):
        metrics = getattr(self, f"{self.stage}_metrics")
        for i in range(output.data.batch_size):
            matched_pred_mask = output.matched_pred_masks[i]
            matched_gt_mask = output.matched_gt_masks[i]

            pred_mask_onehot = F.one_hot(
                matched_pred_mask.argmax(dim=0),
                matched_pred_mask.shape[0],
            ).permute(2, 0, 1)
            gt_mask_onehot = matched_gt_mask.int()
            metrics.update(pred_mask_onehot, gt_mask_onehot)

    @torch.no_grad()
    def _visualize_mask(self, mask: torch.Tensor):
        """Visualize the mask using PIL.

        Args:
            mask (torch.Tensor): A tensor of shape C x H x W, where C is the number of classes including background.
                This mask can be either logits (probabilities) or one-hot encoded.

        Returns:
            PIL.Image: The mask image.
        """
        C, H, W = mask.shape
        mask = mask.cpu()
        mask_indices = torch.argmax(mask, dim=0)
        mask_img = np.zeros((H, W, 3), dtype=np.uint8)
        for class_idx in torch.unique(mask_indices).int():
            mask_img[mask_indices == class_idx] = colors[class_idx]
        return Image.fromarray(mask_img)

    @torch.no_grad()
    def visualize(self, output: ModelOutput):
        if self.current_epoch % self.hparams.vis_every_n_epoch != 0:
            return
        for i, shape_id in enumerate(output.data.shape_id):
            if shape_id in self.hparams.vis_shape_ids:
                # log ground truth mask only once and not for training
                if self.first_visualization and self.stage != "train":
                    gt_mask = output.matched_gt_masks[i]
                    self.logger.log_image(
                        key=f"{self.stage}/gt_mask/{shape_id}",
                        images=[self._visualize_mask(gt_mask)],
                        step=self.current_epoch,
                    )
                pred_mask = output.matched_pred_masks[i]
                self.logger.log_image(
                    key=f"{self.stage}/pred_mask/{shape_id}",
                    images=[self._visualize_mask(pred_mask)],
                    step=self.current_epoch,
                )

    def training_step(self, batch, batch_idx):
        data = BatchData(**batch)
        raw_output = self.forward(data)
        output = ModelOutput(data=data, raw_output=raw_output, linear_assignment_fn=self.model.linear_assignment)
        loss = self.calculate_loss(output)
        self.calculate_loss(output)
        self.visualize(output)
        return loss

    def validation_step(self, batch, batch_idx):
        data = BatchData(**batch)
        raw_output = self.forward(data)
        output = ModelOutput(data=data, raw_output=raw_output, linear_assignment_fn=self.model.linear_assignment)
        self.calculate_loss(output)
        self.calculate_metrics(output)
        self.visualize(output)

    def test_step(self, batch, batch_idx):
        data = BatchData(**batch)
        raw_output = self.forward(data)
        output = ModelOutput(data=data, raw_output=raw_output, linear_assignment_fn=self.model.linear_assignment)
        self.calculate_metrics(output)
        self.visualize(output)

    def _log_metrics_on_epoch_end(self):
        metrics = getattr(self, f"{self.stage}_metrics", None)
        if metrics is not None:
            self.log_dict(metrics.compute())
            metrics.reset()

    def on_train_epoch_end(self):
        self._log_metrics_on_epoch_end()

    def on_validation_epoch_end(self):
        self._log_metrics_on_epoch_end()
        self.first_visualization = False

    def on_test_epoch_start(self):
        self.first_visualization = True  # reset the flag for test-after-fit

    def on_test_epoch_end(self):
        self._log_metrics_on_epoch_end()

    @property
    def stage(self):
        return str(self.trainer.state.stage.value)
