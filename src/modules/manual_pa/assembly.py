import json
import os
from dataclasses import dataclass
from operator import itemgetter
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch3d.transforms import quaternion_apply
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from torch.nn import functional as F
from torchmetrics import MetricCollection
from transformers import AutoModel

from src.metrics.permutation import KendallTau
from src.models.manual_pa.mask_decoder import MaskDecoder

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False)

from src.metrics import ChamferDistance, PartAccuracy, SuccessRate
from src.models import ManualPAAssemblyNet


@dataclass
class BatchData:
    shape_id: list[int]
    view_ids: list[int]
    step_ids: list[int]
    part_ids: list[list[str]]
    total_parts_cnt: list[int]
    step_part_indices: list[list[int]] | None = None  # a list of B lists of P_i
    img: list[torch.Tensor] | None = None  # a list of B tensors of shape (1, 3, H, W)
    imgs: list[torch.Tensor] | None = None  # a list of B tensors of shape (S_i, 3, H, W)
    pts: list[torch.Tensor] | None = None  # a list of B tensors of shape (P_i, 1000, 3)
    masks: list[torch.Tensor] | None = (
        None  # a list of B tensors of shape (S_i, P_i, 224, 224), P_i is padded to max(P_i)
    )
    similar_parts_cnt: list[torch.Tensor] | None = None  # a list of B tensors of shape (P_i, 1)
    bbox_size: list[torch.Tensor] | None = None  # a list of B tensors of shape (P_i, 3)
    parts_cam_dof: list[torch.Tensor] | None = (
        None  # a list of B tensors of shape (S_i, P_i, 7), P_i is padded to max(P_i)
    )
    ins_one_hot: list[torch.Tensor] | None = None  # a list of B tensors of shape (P_i, 20)
    order: list[torch.Tensor] | None = None  # a list of B tensors of shape (P_i,)
    group: list[torch.Tensor] | None = None  # a list of B tensors of shape (P_i,)

    @property
    def batch_size(self):
        return len(self.shape_id)

    @property
    def pts_packed(self):
        """Pack into a tensor of shape (sum(P_i), 1000, 3)"""
        return torch.cat(self.pts)

    @property
    def order_one_hot(self):
        """One hot encoding of order in the shape of (P_i, 20)"""
        return [F.one_hot(o, 20) for o in self.order]

    @property
    def group_one_hot(self):
        """One hot encoding of group in the shape of (P_i, 20)"""
        return [F.one_hot(g, 20) for g in self.group]


class ModelOutput:
    def __init__(
        self,
        data: BatchData,
        pred_poses: tuple[torch.Tensor, ...],
        pred_masks: list[torch.Tensor],
        pred_ious: list[torch.Tensor],
        permutation_matrices_soft: list[torch.Tensor],
        permutation_matrices_hard: list[torch.Tensor],
        contrastive_matrix: torch.Tensor,
        linear_assignment_fn: callable,
    ):
        self.data = data
        self.pred_poses = pred_poses
        self.pred_masks = pred_masks
        self.pred_ious = pred_ious
        self.permutation_matrices_soft = permutation_matrices_soft
        self.permutation_matrices_hard = permutation_matrices_hard
        self.contrastive_matrix = contrastive_matrix

        self.linear_assignment_fn = linear_assignment_fn

        self.matched_gt_poses: list[torch.Tensor] = []  # a list of B tensors of shape (P_i, 7)
        self.matched_pred_poses: list[torch.Tensor] = []  # a list of B tensors of shape (P_i, 7)

        self.matched_part_ids: list[list[str]] = []  # a list of B lists of P_i strings

        self.matched_gt_ids: list[torch.Tensor] = []
        self.matched_pred_ids: list[torch.Tensor] = []

    def perform_hungarian_matching(self):
        matched_gt_ids_list = []
        matched_pred_ids_list = []
        # Perform the Hungarian Matching algorithm.
        for i in range(self.data.batch_size):
            pred_poses = self.pred_poses[i]  # (P_i, 7)
            gt_poses = self.data.parts_cam_dof[i][
                -1
            ]  # (P_i, 7)  # NOTE: For now, we only use the last step's pose for GT
            simlar_cnt = self.data.similar_parts_cnt[i]  # (P_i, 1)
            pts = self.data.pts[i]  # (P_i, 1000, 3)
            matched_gt_ids, matched_pred_ids = self.linear_assignment_fn(simlar_cnt, pts, pred_poses, gt_poses)

            matched_gt_ids_list.append(matched_gt_ids)
            matched_pred_ids_list.append(matched_pred_ids)

        return matched_gt_ids_list, matched_pred_ids_list

    def apply_matching(self, matched_gt_ids_list: list, matched_pred_ids_list: list):
        self.matched_gt_ids = matched_gt_ids_list
        self.matched_pred_ids = matched_pred_ids_list
        for i, (matched_pred_ids, matched_gt_ids) in enumerate(zip(matched_pred_ids_list, matched_gt_ids_list)):
            pred_poses = self.pred_poses[i]  # (P_i, 7)
            gt_poses = self.data.parts_cam_dof[i][
                -1
            ]  # (P_i, 7)  # NOTE: For now, we only use the last step's pose for GT

            # self.matched_gt_poses.append(gt_poses[matched_gt_ids])
            # self.matched_pred_poses.append(pred_poses[matched_pred_ids])

            self.matched_gt_poses.append(gt_poses[matched_pred_ids])
            self.matched_pred_poses.append(pred_poses[matched_gt_ids])

            self.matched_part_ids.append([self.data.part_ids[i][j] for j in matched_pred_ids])

    @property
    def matched_pred_centers_packed(self):
        """Pack into a tensor of shape (sum(P_i), 3)"""
        return torch.cat([pose[:, :3] for pose in self.matched_pred_poses])

    @property
    def matched_gt_centers_packed(self):
        """Pack into a tensor of shape (sum(P_i), 3)"""
        return torch.cat([pose[:, :3] for pose in self.matched_gt_poses])

    @property
    def matched_pred_quats_packed(self):
        """Pack into a tensor of shape (sum(P_i), 4)"""
        return torch.cat([pose[:, 3:] for pose in self.matched_pred_poses])

    @property
    def matched_gt_quats_packed(self):
        """Pack into a tensor of shape (sum(P_i), 4)"""
        return torch.cat([pose[:, 3:] for pose in self.matched_gt_poses])

    @property
    def matched_transformed_pred_pts(self):
        """Return a list of B tensors of shape (P_i, 1000, 3)"""
        return [
            quaternion_apply(pose[:, 3:].unsqueeze(1), pts) + pose[:, :3].unsqueeze(1)
            for pts, pose in zip(
                self.data.pts,
                self.matched_pred_poses,
            )
        ]

    @property
    def matched_transformed_gt_pts(self):
        """Return a list of B tensors of shape (P_i, 1000, 3)"""
        return [
            quaternion_apply(pose[:, 3:].unsqueeze(1), pts) + pose[:, :3].unsqueeze(1)
            for pts, pose in zip(
                self.data.pts,
                self.matched_gt_poses,
            )
        ]

    @property
    def matched_gt_masks(self):
        """Return a list of B tensors of shape (P_i, 224, 224)"""
        return [
            mask[0][self.matched_pred_ids[i]] for i, mask in enumerate(self.data.masks)
        ]  # TODO: Handle multiple steps

    # TODO: Refactor the following
    def serialize(self):
        """Return a dict in the following format:
        <shape_id>:
            <view>:
                <step>:
                    <part_id>: 4x4 matrix
        """
        ret = {}
        for i in range(self.data.batch_size):
            shape_id = self.data.shape_id[i]
            view_id = self.data.view_ids[i][0]  # TODO: Handle multiple
            step_id = self.data.step_ids[i][0]  # TODO: Handle multiple
            # part_ids = self.matched_part_ids[i]
            part_ids = self.data.part_ids[i]
            for j, part_id in enumerate(part_ids):
                ret.setdefault(shape_id, {}).setdefault(view_id, {}).setdefault(step_id, {})[part_id] = (
                    self.matched_pred_poses[i][j].cpu().tolist()
                )
        return ret


class ModelOutputs:
    def __init__(
        self,
        data: BatchData,
        pred_poses: tuple[torch.Tensor, ...],
        pred_masks: list[torch.Tensor],
        pred_ious: list[torch.Tensor],
        permutation_matrices_soft: list[torch.Tensor],
        permutation_matrices_hard: list[torch.Tensor],
        contrastive_matrix: torch.Tensor,
        linear_assignment_fn: callable,
    ):
        num_layers = pred_poses[0].shape[0]
        self.outputs = [
            ModelOutput(
                data,
                [o[i] for o in pred_poses],
                [o[i] for o in pred_masks],
                [o[i] for o in pred_ious],
                permutation_matrices_soft,
                permutation_matrices_hard,
                contrastive_matrix,
                linear_assignment_fn,
            )
            for i in range(num_layers)
        ]
        # Perform Hungarian Matching on the last layer
        matched_gt_ids_list, matched_pred_ids_list = self.outputs[-1].perform_hungarian_matching()
        # Apply the matching to all layers
        for output in self.outputs:
            output.apply_matching(matched_gt_ids_list, matched_pred_ids_list)

    @property
    def last_layer_output(self):
        return self.outputs[-1]

    def __getitem__(self, index):
        return self.outputs[index]

    def __len__(self):
        return len(self.outputs)

    def __iter__(self):
        return iter(self.outputs)


class ManualPAAssemblyNetModule(pl.LightningModule):
    def __init__(
        self,
        # model
        ## EXPERIMENTAL
        step_concate_instance_one_hot: bool = False,
        part_concate_instance_one_hot: bool = False,
        part_concate_group_one_hot: bool = False,
        add_positional_encoding: bool = True,
        step_add_positional_encoding: bool = True,
        part_add_positional_encoding: bool = True,
        use_rope: bool = True,
        vision_token_mode: Literal["all", "all_wo_cls", "cls_only", "max_pooling", "center16"] = "all",
        diagram_global_feature: bool = False,
        part_global_feature: bool = False,
        order_mode: Literal["raw", "random", "learned", "gt"] = "gt",
        permutation_tau: float = 0.75,
        permutation_n_iter: int = 3,
        step_diagram_feature_diff: bool = False,
        decoder_pe_mode: Literal["input_add", "layer_concat", "layer_add"] = "input_add",
        generator_type: Literal["parallel", "autoregressive"] = "parallel",
        ## pointnet encoder
        pointnet_encoder: Literal["pointnet", "pointnetlite", "pointnetgroup"] = "pointnetlite",
        pointnet_dim: int = 1024,
        ## vision encoder
        vision_encoder: Literal["facebook/dinov2-small", "facebook/dinov2-base"] = "facebook/dinov2-small",
        freeze_vision_encoder: bool = False,
        ## transformer decoder
        d_model: int = 768,
        num_layers: int = 8,
        aux_loss: bool = False,
        # loss weights
        loss_weight_center: float = 1.0,
        loss_weight_quat: float = 20.0,
        loss_weight_l2_rot: float = 1.0,
        loss_weight_shape_chamfer: float = 20.0,
        loss_weight_mask: float = 0.0,
        loss_weight_permutation: float = 0.0,
        # Evaluation
        eval_assembly: bool = True,
        eval_permutation: bool = False,
        force_eval: bool = False,  # HACK: for overfitting to one sample
        # Persistence
        save_poses: bool = True,
        save_metrics: bool = True,
        save_orders: bool = False,
        # Load from checkpoint
        ckpt_path: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        if pointnet_encoder == "pointnet":
            from src.models.manual_pa.pointnet import PointNetFeat
        elif pointnet_encoder == "pointnetlite":
            from src.models.manual_pa.pointnet_lite import (
                PointNetLiteFeat as PointNetFeat,
            )
        elif pointnet_encoder == "pointnetgroup":
            from src.models.manual_pa.pointnet_group import (
                PointNetGroupFeat as PointNetFeat,
            )

        pointnet_encoder = PointNetFeat(out_features=pointnet_dim)
        vision_encoder = AutoModel.from_pretrained(vision_encoder)
        mask_decoder = MaskDecoder()
        self.model = ManualPAAssemblyNet(
            pointnet_encoder=pointnet_encoder,
            vision_encoder=vision_encoder,
            mask_decoder=mask_decoder,
            freeze_vision_encoder=freeze_vision_encoder,
            d_model=d_model,
            num_layers=num_layers,
            step_concate_instance_one_hot=step_concate_instance_one_hot,
            part_concate_instance_one_hot=part_concate_instance_one_hot,
            part_concate_group_one_hot=part_concate_group_one_hot,
            add_positional_encoding=add_positional_encoding,
            step_add_positional_encoding=step_add_positional_encoding,
            part_add_positional_encoding=part_add_positional_encoding,
            use_rope=use_rope,
            vision_token_mode=vision_token_mode,
            diagram_global_feature=diagram_global_feature,
            part_global_feature=part_global_feature,
            order_mode=order_mode,
            permutation_tau=permutation_tau,
            permutation_n_iter=permutation_n_iter,
            step_diagram_feature_diff=step_diagram_feature_diff,
            decoder_pe_mode=decoder_pe_mode,
            generator_type=generator_type,
        )

        # Metrics
        if eval_assembly:
            metrics = MetricCollection(
                {
                    "SCD": ChamferDistance(),
                    "PA": PartAccuracy(),
                    # "CA": ConnectivityAccuracy(),
                    "SR": SuccessRate(),
                },
                compute_groups=False,
            )
            self.validate_metrics = metrics.clone(prefix="validate/")
            self.test_metrics = metrics.clone(prefix="test/")
        if eval_permutation:
            permutation_metrics = MetricCollection(
                {
                    "KD": KendallTau(),
                },
                compute_groups=False,
            )
            self.validate_permutation_metrics = permutation_metrics.clone(prefix="validate/")
            self.test_permutation_metrics = permutation_metrics.clone(prefix="test/")

        # Persistence
        if save_metrics:
            self.persample_metrics = metrics.clone()
            self.predicted_metrics = {}
        if save_poses:
            self.predicted_poses = {}
        if save_orders:
            self.predicted_orders = {}

        # Load from checkpoint
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)

    def forward(self, data: BatchData):
        if self.hparams.force_eval:
            self.model.eval()
        (
            pred_poses,  # a tuple of B tensors of shape (num_layers, P_i, 7)
            pred_masks,  # a list of B tensors of shape (num_layers, num_steps, num_parts, 64, 64)
            pred_ious,  # a list of B tensors of shape (num_layers, num_steps, num_parts, 1)
            permutation_matrices_soft,  # a list of B tensors of shape (P_i, P_i)
            permutation_matrices_hard,  # a list of B tensors of shape (P_i, P_i)
            contrastive_matrix,  # a tensor of shape (sum_{P_i}, sum_{P_i})
        ) = self.model(
            step_imgs=data.imgs,
            part_pts=data.pts,
            instance_one_hot=data.order_one_hot,
            group_one_hot=data.group_one_hot,
        )
        return ModelOutputs(
            data=data,
            pred_poses=pred_poses,
            pred_masks=pred_masks,
            pred_ious=pred_ious,
            permutation_matrices_soft=permutation_matrices_soft,
            permutation_matrices_hard=permutation_matrices_hard,
            contrastive_matrix=contrastive_matrix,
            linear_assignment_fn=self.model.linear_assignment,
        )

    def _calculate_loss(self, output: ModelOutput):
        # center loss
        center_loss = (
            0
            if self.hparams.loss_weight_center == 0
            else self.model.get_center_loss(
                output.matched_pred_centers_packed,
                output.matched_gt_centers_packed,
            ).mean()
        )

        # quat loss
        quat_loss = (
            0
            if self.hparams.loss_weight_quat == 0
            else self.model.get_quat_loss(
                output.data.pts_packed,
                output.matched_pred_quats_packed,
                output.matched_gt_quats_packed,
            ).mean()
        )

        # l2 rotation loss
        l2_rot_loss = (
            0
            if self.hparams.loss_weight_l2_rot == 0
            else self.model.get_l2_rotation_loss(
                output.data.pts_packed,
                output.matched_pred_quats_packed,
                output.matched_gt_quats_packed,
            ).mean()
        )

        # shape chamfer loss
        shape_chamfer_loss = (
            0
            if self.hparams.loss_weight_shape_chamfer == 0
            else self.model.get_shape_chamfer_loss(
                output.data.pts_packed,
                output.matched_pred_quats_packed,
                output.matched_gt_quats_packed,
                output.matched_pred_centers_packed,
                output.matched_gt_centers_packed,
                output.data.total_parts_cnt,
            ).mean()
        )

        # mask loss
        mask_loss = (
            0
            if self.hparams.loss_weight_mask == 0
            else self.model.get_mask_loss(output.pred_masks, output.matched_gt_masks).mean()
        ) + 1

        # permutation loss
        if self.hparams.loss_weight_permutation == 0:
            permutation_loss = 0
        else:
            # Sample-Level Contrastive Loss
            # permutation_loss = torch.tensor(0.0, device=self.device)
            # for i in range(output.data.batch_size):
            #     pred_permutation_matrix = output.permutation_matrices_soft[i]
            #     gt_permutation_matrix = output.data.order_one_hot[i][:, : pred_permutation_matrix.shape[0]].float()
            #     permutation_loss += (
            #         F.cross_entropy(pred_permutation_matrix, output.data.order[i])
            #         + F.cross_entropy(
            #             pred_permutation_matrix.t(),
            #             torch.argmax(gt_permutation_matrix, dim=0),
            #         )
            #     ) / 2
            # permutation_loss /= output.data.batch_size

            # Sample-Level Contrastive Loss with Similar Parts Removed
            permutation_loss = torch.tensor(0.0, device=self.device)
            for i in range(output.data.batch_size):
                pred_permutation_matrix = output.permutation_matrices_soft[i]

                # 1. Sort the contrastive matrix
                orders = output.data.order[i]
                sorted_contrastive_matrix = pred_permutation_matrix[:, orders]

                # 2. Get group ids
                groups = []
                current_id = 0
                total_count = 0
                while total_count < output.data.similar_parts_cnt[i].shape[0]:
                    current_count = output.data.similar_parts_cnt[i][total_count].item()
                    total_count += current_count
                    for _ in range(current_count):
                        groups.append(current_id)
                    current_id += 1
                groups = torch.tensor(groups, device=self.device)

                # 3. Sort the groups w.r.t. the order
                ordered_groups = torch.zeros_like(groups)
                ordered_groups.scatter_(0, orders, groups)

                # 4. Get mask
                unique_groups = ordered_groups.unique()
                mask = torch.zeros_like(ordered_groups, dtype=torch.bool)
                for group in unique_groups:
                    indices = (ordered_groups == group).nonzero(as_tuple=True)[0]
                    if indices.numel() > 0:
                        random_index = indices[torch.randint(len(indices), (1,))]
                        mask[random_index] = True

                # 5. Apply the mask
                sorted_contrastive_matrix = sorted_contrastive_matrix[mask][:, mask]

                # 4. Compute the loss
                target = torch.arange(sorted_contrastive_matrix.shape[0], device=self.device)
                permutation_loss += (
                    F.cross_entropy(sorted_contrastive_matrix, target)
                    + F.cross_entropy(sorted_contrastive_matrix.t(), target)
                ) / 2
            permutation_loss /= output.data.batch_size

            # Batch-Level Contrastive Loss
            # current_count = 0
            # orders = []
            # for i in range(output.data.batch_size):
            #     orders.append(output.data.order[i] + current_count)
            #     current_count += output.data.order[i].shape[0]
            # orders = torch.cat(orders)
            # orders_one_hot = F.one_hot(orders, orders.shape[0])
            # permutation_loss = (
            #     F.cross_entropy(output.contrastive_matrix, orders)
            #     + F.cross_entropy(output.contrastive_matrix.t(), torch.argmax(orders_one_hot, dim=0))
            # ) / 2

            # Batch-Level Contrastive Loss with Similar Parts Removed
            # 1. Get the aggregated order
            # current_count = 0
            # orders = []
            # for i in range(output.data.batch_size):
            #     orders.append(output.data.order[i] + current_count)
            #     current_count += output.data.order[i].shape[0]
            # orders = torch.cat(orders)

            # # 2. Sort the contrastive matrix
            # sorted_contrastive_matrix = output.contrastive_matrix[:, orders]

            # # 3. Filter out similar parts
            # ## 3.1 Get aggregated group ids
            # groups = []
            # current_id = 0
            # for i in range(output.data.batch_size):
            #     total_count = 0
            #     while total_count < output.data.similar_parts_cnt[i].shape[0]:
            #         current_count = output.data.similar_parts_cnt[i][total_count].item()
            #         total_count += current_count
            #         for _ in range(current_count):
            #             groups.append(current_id)
            #         current_id += 1
            # groups = torch.tensor(groups, device=self.device)
            # ## 3.2 Sort the groups w.r.t. the aggregated order
            # ordered_groups = torch.zeros_like(groups)
            # ordered_groups.scatter_(0, orders, groups)

            # ## 3.3 Filter out similar parts and generate a indics mask
            # unique_groups = ordered_groups.unique()
            # mask = torch.zeros_like(ordered_groups, dtype=torch.bool)
            # for group in unique_groups:
            #     indices = (ordered_groups == group).nonzero(as_tuple=True)[0]
            #     if indices.numel() > 0:
            #         random_index = indices[torch.randint(len(indices), (1,))]
            #         mask[random_index] = True

            # ## 3.4 Apply the mask
            # sorted_contrastive_matrix = sorted_contrastive_matrix[mask][:, mask]

            # # 4. Compute the loss
            # target = torch.arange(sorted_contrastive_matrix.shape[0], device=self.device)
            # permutation_loss = (
            #     F.cross_entropy(sorted_contrastive_matrix, target)
            #     + F.cross_entropy(sorted_contrastive_matrix.t(), target)
            # ) / 2

        return dict(
            center_loss=center_loss,
            quat_loss=quat_loss,
            l2_rot_loss=l2_rot_loss,
            shape_chamfer_loss=shape_chamfer_loss,
            mask_loss=mask_loss,
            permutation_loss=permutation_loss,
        )

    def calculate_loss(self, outputs: ModelOutputs):
        loss = torch.tensor(0.0, device=self.device)
        batch_size = outputs.last_layer_output.data.batch_size
        if not self.hparams.aux_loss or not self.training:
            outputs = [outputs.last_layer_output]
        for i, output in enumerate(outputs):
            loss_dict = self._calculate_loss(output)
            (
                center_loss,
                quat_loss,
                l2_rot_loss,
                shape_chamfer_loss,
                mask_loss,
                permutation_loss,
            ) = itemgetter(
                "center_loss",
                "quat_loss",
                "l2_rot_loss",
                "shape_chamfer_loss",
                "mask_loss",
                "permutation_loss",
            )(loss_dict)
            loss_layer = (
                self.hparams.loss_weight_center * center_loss
                + self.hparams.loss_weight_quat * quat_loss
                + self.hparams.loss_weight_l2_rot * l2_rot_loss
                + self.hparams.loss_weight_shape_chamfer * shape_chamfer_loss
                + self.hparams.loss_weight_mask * mask_loss
                + self.hparams.loss_weight_permutation * permutation_loss
            )
            if i == len(outputs) - 1:  # Only log the last layer
                self.log(f"{self.stage}/center_loss", center_loss, batch_size=batch_size)
                self.log(f"{self.stage}/quat_loss", quat_loss, batch_size=batch_size)
                self.log(f"{self.stage}/l2_rot_loss", l2_rot_loss, batch_size=batch_size)
                self.log(
                    f"{self.stage}/shape_chamfer_loss",
                    shape_chamfer_loss,
                    batch_size=batch_size,
                )
                self.log(f"{self.stage}/mask_loss", mask_loss, batch_size=batch_size)
                self.log(
                    f"{self.stage}/permutation_loss",
                    permutation_loss,
                    batch_size=batch_size,
                )

            loss += loss_layer / len(outputs)
        self.log(f"{self.stage}/loss", loss, batch_size=batch_size)
        return loss

    @torch.no_grad()
    def calculate_metrics(self, output: ModelOutput):
        metrics = getattr(self, f"{self.stage}_metrics", None)
        permutation_metrics = getattr(self, f"{self.stage}_permutation_metrics", None)
        for i in range(output.data.batch_size):
            if metrics is not None:
                metrics.update(
                    output.matched_transformed_pred_pts[i],
                    output.matched_transformed_gt_pts[i],
                )
            if permutation_metrics is not None:
                permutation_metrics.update(
                    torch.argmax(output.permutation_matrices_hard[i], dim=-1),
                    output.data.order[i],
                )

    def training_step(self, batch, batch_idx):
        data = BatchData(**batch)
        self.log("train/bs", data.batch_size, batch_size=data.batch_size)
        outputs = self.forward(data)
        loss = self.calculate_loss(outputs)
        return loss

    def validation_step(self, batch, batch_idx):
        data = BatchData(**batch)
        outputs = self.forward(data)
        self.calculate_loss(outputs)
        self.calculate_metrics(outputs.last_layer_output)

    def test_step(self, batch, batch_idx):
        data = BatchData(**batch)
        outputs = self.forward(data)
        output = outputs.last_layer_output
        self.calculate_metrics(output)
        # Persistence
        for i in range(output.data.batch_size):
            shape_id = data.shape_id[i]
            view_id = data.view_ids[i][0]  # TODO: Handle multiple
            step_id = data.step_ids[i][0]  # TODO: Handle multiple

            if self.hparams.save_metrics:
                m = self.persample_metrics(
                    output.matched_transformed_pred_pts[i],
                    output.matched_transformed_gt_pts[i],
                )
                for metric_key, metric_value in m.items():
                    self.predicted_metrics.setdefault(shape_id, {}).setdefault(view_id, {}).setdefault(step_id, {})[
                        metric_key
                    ] = metric_value.item()
                self.persample_metrics.reset()
            if self.hparams.save_poses:
                self.predicted_poses.update(output.serialize())
            if self.hparams.save_orders:
                self.predicted_orders[shape_id] = (
                    torch.argmax(output.permutation_matrices_hard[i], dim=-1).cpu().tolist()
                )

    def _log_metrics_on_epoch_end(self):
        metrics: MetricCollection = getattr(self, f"{self.stage}_metrics", None)
        if metrics is not None:
            self.log_dict(metrics.compute())
            metrics.reset()
        permutation_metrics: MetricCollection = getattr(self, f"{self.stage}_permutation_metrics", None)
        if permutation_metrics is not None:
            self.log_dict(permutation_metrics.compute())
            permutation_metrics.reset()

    def on_train_epoch_end(self):
        self._log_metrics_on_epoch_end()

    def on_validation_epoch_end(self):
        self._log_metrics_on_epoch_end()

    def on_test_epoch_end(self):
        self._log_metrics_on_epoch_end()
        # Persistence
        log_dir = self.trainer.log_dir  # this broadcasts the directory
        if (
            self.trainer.logger is not None
            and self.trainer.logger.name is not None
            and self.trainer.logger.version is not None
        ):
            log_dir = os.path.join(log_dir, self.trainer.logger.name, str(self.trainer.logger.version))
        if self.hparams.save_poses:
            output_pathname = os.path.join(log_dir, "pred_poses.json")
            print("Saving pose predictions to:", output_pathname)
            with open(output_pathname, "w") as f:
                json.dump(self.predicted_poses, f)
        if self.hparams.save_metrics:
            output_pathname = os.path.join(log_dir, "pred_metrics.json")
            print("Saving metrics to:", output_pathname)
            with open(output_pathname, "w") as f:
                json.dump(self.predicted_metrics, f)
        if self.hparams.save_orders:
            output_pathname = os.path.join(log_dir, "pred_orders.json")
            print("Saving orders to:", output_pathname)
            with open(output_pathname, "w") as f:
                json.dump(self.predicted_orders, f)

    @property
    def stage(self):
        return str(self.trainer.state.stage.value)
