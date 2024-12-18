import io
import json
import os
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch3d.transforms import quaternion_apply
from torchmetrics import MetricCollection

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False)

from src.metrics import ChamferDistance, PartAccuracy, SuccessRate
from src.models import AssemblyNet
from src.utils import colors


@dataclass
class BatchData:
    shape_id: list[int]
    view_id: list[int]
    step_id: list[int]
    part_ids: list[list[str]]
    total_parts_cnt: list[int]
    img: torch.Tensor | None = None  # a tensor of B x 3 x H x W
    pts: list[torch.Tensor] | None = None  # a list of B tensors of shape N_i x 1000 x 3
    masks: list[torch.Tensor] | None = None  # a list of B tensors of shape N_i x 224 x 224
    similar_parts_cnt: torch.Tensor | None = None  # a tensor of B x N_i x 1
    bbox_size: list[torch.Tensor] | None = None  # a list of B tensors of shape N_i x 3
    ins_one_hot: list[torch.Tensor] | None = None  # a list of B tensors of shape N_i x 20
    similar_parts_edge_indices: list[torch.Tensor] | None = None  # a list of B tensors of shape M x 2
    parts_cam_dof: list[torch.Tensor] | None = None  # a list of B tensors of shape N_i x 7
    step_part_indices: list[list[int]] | None = None  # a list of B lists of N_i

    @property
    def batch_size(self):
        return len(self.shape_id)

    @property
    def max_parts_cnt(self):
        """max(N_i)"""
        return max(self.total_parts_cnt)

    @property
    def step_parts_cnt(self):
        """N_i"""
        return [len(step_part_indices) for step_part_indices in self.step_part_indices]

    @property
    def batch_total_parts_cnt(self):
        """sum(N_i)"""
        return sum(self.total_parts_cnt)

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
    def step_pts_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 1000 x 3"""
        return torch.cat(
            [pts[step_part_indices] for pts, step_part_indices in zip(self.pts, self.step_part_indices)], dim=0
        )

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
        self.matched_pred_masks: list[torch.Tensor] = []  # a list of B tensors of shape N_i x H x W
        self.matched_gt_masks: list[torch.Tensor] = []  # a list of B tensors of shape N_i x H x W

        self.matched_pred_centers1: list[torch.Tensor] = []  # a list of B tensors of shape N_i x 3
        self.matched_pred_centers2: list[torch.Tensor] = []  # a list of B tensors of shape N_i x 3
        self.matched_gt_centers: list[torch.Tensor] = []  # a list of B tensors of shape N_i x 3

        self.matched_pred_quats1: list[torch.Tensor] = []  # a list of B tensors of shape N_i x 4
        self.matched_pred_quats2: list[torch.Tensor] = []  # a list of B tensors of shape N_i x 4
        self.matched_gt_quats: list[torch.Tensor] = []  # a list of B tensors of shape N_i x 4

        self.matched_gt_pts: list[torch.Tensor] = []  # a list of B tensors of shape N_i x 1000 x 3
        self.matched_pred_pts: list[torch.Tensor] = []  # a list of B tensors of shape N_i x 1000 x 3

        self.matched_part_ids: list[list[str]] = []  # a list of B lists of N_i strings

        cur_parts_cnt = 0
        for i in range(data.batch_size):
            pred_mask = raw_output[0][i]
            gt_mask = data.masks[i].float()
            step_part_indices = data.step_part_indices[i]

            simlar_cnt = data.similar_parts_cnt[i]
            parts_cnt = data.total_parts_cnt[i]
            pts = data.pts[i]
            gt_center = data.parts_cam_dof[i][:, :3]
            gt_quat = data.parts_cam_dof[i][:, 3:]

            center1 = raw_output[1][cur_parts_cnt : cur_parts_cnt + parts_cnt]
            quat1 = raw_output[2][cur_parts_cnt : cur_parts_cnt + parts_cnt]
            center2 = raw_output[3][cur_parts_cnt : cur_parts_cnt + parts_cnt]
            quat2 = raw_output[4][cur_parts_cnt : cur_parts_cnt + parts_cnt]
            cur_parts_cnt += parts_cnt

            matched_gt_ids, matched_pred_ids = linear_assignment_fn(
                gt_mask,
                pred_mask,
                simlar_cnt,
                pts,
                gt_center,
                gt_quat,
                center2,
                quat2,
            )

            # Filter out parts by step_part_indices
            matched_pred_ids = [j for j in matched_pred_ids if j in step_part_indices]
            matched_gt_ids = [j for j in matched_gt_ids if j in step_part_indices]

            self.matched_pred_masks.append(pred_mask[matched_pred_ids])
            self.matched_gt_masks.append(gt_mask[matched_gt_ids])

            self.matched_pred_centers1.append(center1[matched_pred_ids])
            self.matched_pred_centers2.append(center2[matched_pred_ids])
            self.matched_gt_centers.append(gt_center[matched_gt_ids])

            self.matched_pred_quats1.append(quat1[matched_pred_ids])
            self.matched_pred_quats2.append(quat2[matched_pred_ids])
            self.matched_gt_quats.append(gt_quat[matched_gt_ids])

            self.matched_gt_pts.append(pts[matched_gt_ids])
            self.matched_pred_pts.append(pts[matched_pred_ids])

            self.matched_part_ids.append([data.part_ids[i][j] for j in matched_pred_ids])

    @property
    def matched_transformed_gt_pts(self):
        """Returns a list of B tensors of shape N_i x 1000 x 3"""
        ret = []
        for i in range(self.data.batch_size):
            gt_pts = self.matched_gt_pts[i]
            gt_center = self.matched_gt_centers[i].unsqueeze(1)  # for boardcasting
            gt_quat = self.matched_gt_quats[i].unsqueeze(1)  # for boardcasting
            transformed_gt_pts = quaternion_apply(gt_quat, gt_pts) + gt_center
            ret.append(transformed_gt_pts)
        return ret

    @property
    def matched_transformed_pred_pts(self):
        """Returns a list of B tensors of shape N_i x 1000 x 3"""
        ret = []
        for i in range(self.data.batch_size):
            pred_pts = self.matched_pred_pts[i]
            pred_center = self.matched_pred_centers2[i].unsqueeze(1)  # for boardcasting
            pred_quat = self.matched_pred_quats2[i].unsqueeze(1)  # for boardcasting
            transformed_pred_pts = quaternion_apply(pred_quat, pred_pts) + pred_center
            ret.append(transformed_pred_pts)
        return ret

    @property
    def matched_pred_masks_packed(self):
        """Returns a packed tensor of shape sum(N_i) x H x W"""
        return torch.cat(self.matched_pred_masks, dim=0)

    @property
    def matched_gt_masks_packed(self):
        """Returns a packed tensor of shape sum(N_i) x H x W"""
        return torch.cat(self.matched_gt_masks, dim=0)

    @property
    def matched_pred_centers1_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 3"""
        return torch.cat(self.matched_pred_centers1, dim=0)

    @property
    def matched_pred_centers2_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 3"""
        return torch.cat(self.matched_pred_centers2, dim=0)

    @property
    def matched_gt_centers_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 3"""
        return torch.cat(self.matched_gt_centers, dim=0)

    @property
    def matched_pred_quats1_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 4"""
        return torch.cat(self.matched_pred_quats1, dim=0)

    @property
    def matched_pred_quats2_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 4"""
        return torch.cat(self.matched_pred_quats2, dim=0)

    @property
    def matched_gt_quats_packed(self):
        """Returns a packed tensor of shape sum(N_i) x 4"""
        return torch.cat(self.matched_gt_quats, dim=0)

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
            view_id = self.data.view_id[i]
            step_id = self.data.step_id[i]
            # part_ids = self.matched_part_ids[i]
            part_ids = self.data.part_ids[i]
            centers = self.matched_pred_centers2[i].cpu()
            quats = self.matched_pred_quats2[i].cpu()
            for j, part_id in enumerate(part_ids):
                ret.setdefault(shape_id, {}).setdefault(view_id, {}).setdefault(step_id, {})[part_id] = torch.cat(
                    [centers[j], quats[j]]
                ).tolist()
        return ret


Conf = namedtuple(
    "Conf",
    [
        "img_size",
        "ins_dim",
        "pointnet_emd_dim",
        "resnet_feat_dim",
        "pretrain_resnet",
        "device",
    ],
)


class ImagePAAssemblyNetModule(pl.LightningModule):
    def __init__(
        self,
        # model params
        pretrained_seg_model_ckpt_pathname: str = "",
        # loss weights
        loss_weight_center: float = 1.0,
        loss_weight_quat: float = 20.0,
        loss_weight_l2_rot: float = 1.0,
        loss_weight_shape_chamfer: float = 20.0,
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
        vis_every_n_epoch: int = 10,
        force_eval: bool = False,  # HACK: for overfitting to one sample
    ):
        super().__init__()
        self.save_hyperparameters()

        conf = Conf(
            img_size=224,
            ins_dim=20,
            pointnet_emd_dim=512,
            resnet_feat_dim=512,
            pretrain_resnet=False,
            device="cpu",
        )
        self.model = AssemblyNet(
            conf=conf,
            sem_cnt=20,
        )
        metrics = MetricCollection(
            {
                "SCD": ChamferDistance(),
                "PA": PartAccuracy(),
                # "CA": ConnectivityAccuracy(),
                "SR": SuccessRate(),
            },
            compute_groups=False,
        )
        if pretrained_seg_model_ckpt_pathname:
            state_dict = {}
            for key, value in torch.load(pretrained_seg_model_ckpt_pathname, map_location="cpu")["state_dict"].items():
                state_dict[key.replace("model.", "")] = value
            self.model.mask_net.load_state_dict(state_dict)

        # freeze the mask_net
        # for param in self.model.mask_net.parameters():
        #     param.requires_grad = False

        self.validate_metrics = metrics.clone(prefix="validate/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Visualization
        self.first_visualization = True  # used for logging ground truth masks only once

        # Persistence
        self.persample_metrics = metrics.clone()
        self.predicted_poses = {}
        self.predicted_metrics = {}

    def forward(self, data: BatchData):
        if self.hparams.force_eval:
            self.model.eval()
        raw_output: tuple[torch.Tensor, ...] = self.model(
            img=data.img_packed,  # sum(N_i) x 3 x H x W
            pc=data.pts_packed,  # sum(N_i) x 1000 x 3
            ins_feat=data.ins_one_hot_packed,  # sum(N_i) x 20
            part_cnt=data.total_parts_cnt,  # list of N_i
            equiv_edge_indices=data.similar_parts_edge_indices,  # list of M_i x 2
        )
        # pred_masks: a list of B tensors. Each tensor has shape N_i x H x W.
        # center: a tensor of sum(N_i) x 3
        # quat: a tensor of sum(N_i) x 4
        # center2: a tensor of sum(N_i) x 3
        # quat2: a tensor of sum(N_i) x 4
        return ModelOutput(data=data, raw_output=raw_output, linear_assignment_fn=self.model.linear_assignment)

    def calculate_loss(self, output: ModelOutput):
        loss = torch.tensor(0.0, device=self.device)

        # center loss
        center_loss = (
            self.model.get_center_loss(
                output.matched_pred_centers1_packed,
                output.matched_gt_centers_packed,
            )
            + self.model.get_center_loss(
                output.matched_pred_centers2_packed,
                output.matched_gt_centers_packed,
            )
        ).mean()
        loss += center_loss * self.hparams.loss_weight_center
        self.log(f"{self.stage}/center_loss", center_loss.item())

        # quat loss
        quat_loss = (
            self.model.get_quat_loss(
                output.data.step_pts_packed,
                output.matched_pred_quats1_packed,
                output.matched_gt_quats_packed,
            )
            + self.model.get_quat_loss(
                output.data.step_pts_packed,
                output.matched_pred_quats2_packed,
                output.matched_gt_quats_packed,
            )
        ).mean()
        loss += quat_loss * self.hparams.loss_weight_quat
        self.log(f"{self.stage}/quat_loss", quat_loss.item())

        # l2 rotation loss
        l2_rot_loss = (
            self.model.get_l2_rotation_loss(
                output.data.step_pts_packed,
                output.matched_pred_quats1_packed,
                output.matched_gt_quats_packed,
            )
            + self.model.get_l2_rotation_loss(
                output.data.step_pts_packed,
                output.matched_pred_quats2_packed,
                output.matched_gt_quats_packed,
            )
        ).mean()
        loss += l2_rot_loss * self.hparams.loss_weight_l2_rot
        self.log(f"{self.stage}/l2_rot_loss", l2_rot_loss.item())

        # shape chamfer loss
        shape_chamfer_loss = (
            self.model.get_shape_chamfer_loss(
                output.data.step_pts_packed,
                output.matched_pred_quats1_packed,
                output.matched_gt_quats_packed,
                output.matched_pred_centers1_packed,
                output.matched_gt_centers_packed,
                output.data.step_parts_cnt,
            )
            + self.model.get_shape_chamfer_loss(
                output.data.step_pts_packed,
                output.matched_pred_quats2_packed,
                output.matched_gt_quats_packed,
                output.matched_pred_centers2_packed,
                output.matched_gt_centers_packed,
                output.data.step_parts_cnt,
            )
        ).mean()
        loss += shape_chamfer_loss * self.hparams.loss_weight_shape_chamfer
        self.log(f"{self.stage}/shape_chamfer_loss", shape_chamfer_loss.item())

        # mask loss
        # NOTE: This loss is for logging purposes only.
        with torch.no_grad():
            mask_loss = []
            for i in range(output.data.batch_size):
                matched_pred_mask = output.matched_pred_masks[i]
                matched_gt_mask = output.matched_gt_masks[i]
                _mask_loss = self.model.get_mask_loss(matched_pred_mask, matched_gt_mask)
                mask_loss.append(_mask_loss.mean())
            mask_loss = torch.stack(mask_loss).mean()
            self.log(f"{self.stage}/mask_loss", mask_loss.item())

        self.log(f"{self.stage}/loss", loss.item())
        return loss

    @torch.no_grad()
    def calculate_metrics(self, output: ModelOutput):
        metrics = getattr(self, f"{self.stage}_metrics", None)
        for i in range(output.data.batch_size):
            metrics.update(output.matched_transformed_pred_pts[i], output.matched_transformed_gt_pts[i])

    @torch.no_grad()
    def _plot_part_point_clouds(self, part_point_clouds: torch.Tensor):
        """
        Plots 3D point clouds with each part assigned a different color.

        Args:
            part_point_clouds (torch.Tensor): A tensor of shape (N, P, 3) where:
                N: number of parts,
                P: number of points in each part.
        """
        # Ensure the tensor is on CPU and convert to numpy for plotting
        part_point_clouds = part_point_clouds.cpu().numpy()

        # Set up the 3D plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Plot each part with a different color
        for i in range(part_point_clouds.shape[0]):
            color = colors[1:][i % (len(colors) - 1)]
            color = [color / 255 for color in color]
            part = part_point_clouds[i]
            ax.scatter(part[:, 0], part[:, 1], part[:, 2], s=2, color=color)

        # Set the view similar to Blender's default (Right view)
        ax.view_init(elev=90, azim=-90)

        # disable the axis
        ax.axis("off")

        # convert to PIL image
        buf = io.BytesIO()
        ax.margins(0)  # Set margins to zero
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)

        return img

    @torch.no_grad()
    def visualize(self, output: ModelOutput):
        if self.current_epoch % self.hparams.vis_every_n_epoch != 0:
            return
        for i, shape_id in enumerate(output.data.shape_id):
            if shape_id in self.hparams.vis_shape_ids:
                # log ground truth mask only once and not for training
                if self.first_visualization and self.stage != "train":
                    gt_pts = output.matched_transformed_gt_pts[i]
                    self.logger.log_image(
                        key=f"{self.stage}/gt_pts/{shape_id}",
                        images=[self._plot_part_point_clouds(gt_pts)],
                        step=self.current_epoch,
                    )
                pred_pts = output.matched_transformed_pred_pts[i]
                self.logger.log_image(
                    key=f"{self.stage}/pred_pts/{shape_id}",
                    images=[self._plot_part_point_clouds(pred_pts)],
                    step=self.current_epoch,
                )

    def flatten_stepwise_batchdata(self, batch):
        """Flatten the stepwise batch data into a single batch data"""
        shape_id = batch["shape_id"]
        view_ids = batch["view_ids"]
        step_ids = batch["step_ids"]
        step_counts = [len(step_ids[i]) for i in range(len(step_ids))]
        batch_size = len(shape_id)

        new_batch = {}
        for key in list(batch.keys()):
            if key == "view_ids":
                new_data = []
                for i in range(batch_size):
                    new_data.extend(view_ids[i])
                new_batch["view_id"] = new_data
            elif key == "step_ids":
                new_data = []
                for i in range(batch_size):
                    new_data.extend(step_ids[i])
                new_batch["step_id"] = new_data
            elif key == "img":
                new_data = []
                for i in range(batch_size):
                    new_data.append(batch[key][i])
                new_batch[key] = torch.cat(new_data, dim=0)
            elif key in ["masks", "parts_cam_dof", "step_part_indices"]:
                new_data = []
                for i in range(batch_size):
                    for j in range(step_counts[i]):
                        new_data.append(batch[key][i][j])
                new_batch[key] = new_data
            else:
                new_data = []
                for i in range(batch_size):
                    for j in range(step_counts[i]):
                        new_data.append(batch[key][i])
                new_batch[key] = new_data
        return new_batch

    def training_step(self, batch, batch_idx):
        data = BatchData(**self.flatten_stepwise_batchdata(batch))
        output = self.forward(data)
        loss = self.calculate_loss(output)
        self.visualize(output)
        return loss

    def validation_step(self, batch, batch_idx):
        data = BatchData(**self.flatten_stepwise_batchdata(batch))
        output = self.forward(data)
        self.calculate_loss(output)
        self.calculate_metrics(output)
        self.visualize(output)

    def test_step(self, batch, batch_idx):
        data = BatchData(**self.flatten_stepwise_batchdata(batch))
        output = self.forward(data)
        self.calculate_metrics(output)
        self.visualize(output)
        # Persistence
        for i in range(output.data.batch_size):
            shape_id = data.shape_id[i]
            view_id = data.view_id[i]
            step_id = data.step_id[i]

            m = self.persample_metrics(output.matched_transformed_pred_pts[i], output.matched_transformed_gt_pts[i])
            for metric_key, metric_value in m.items():
                self.predicted_metrics.setdefault(shape_id, {}).setdefault(view_id, {}).setdefault(step_id, {})[
                    metric_key
                ] = metric_value.item()
            self.persample_metrics.reset()
            self.predicted_poses.update(output.serialize())

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
        # Persistence
        log_dir = self.trainer.log_dir  # this broadcasts the directory
        if (
            self.trainer.logger is not None
            and self.trainer.logger.name is not None
            and self.trainer.logger.version is not None
        ):
            log_dir = os.path.join(log_dir, self.trainer.logger.name, str(self.trainer.logger.version))
        output_pathname = os.path.join(log_dir, "pred_poses.json")
        print("Saving pose predictions to:", output_pathname)
        with open(output_pathname, "w") as f:
            json.dump(self.predicted_poses, f)
        output_pathname = os.path.join(log_dir, "pred_metrics.json")
        print("Saving metrics to:", output_pathname)
        with open(output_pathname, "w") as f:
            json.dump(self.predicted_metrics, f)

    @property
    def stage(self):
        return str(self.trainer.state.stage.value)
