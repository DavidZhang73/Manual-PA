from typing import Literal

import numpy as np
import pytorch3d.loss
import torch
from einops import rearrange, reduce
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import quaternion_apply as qrot
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from src.models.manual_pa.rope import RotaryEmbedding

from .transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False)


def pytorch3d_chamfer_distance(xyz1, xyz2, transpose=False, sqrt=False, eps=1e-12):
    """Chamfer distance

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.
        sqrt (bool): whether to square root distance
        eps (float): to safely sqrt

    Returns:
        dist1 (torch.Tensor): (b, n1)
        dist2 (torch.Tensor): (b, n2)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)  # (b, n1, 3)
        xyz2 = xyz2.transpose(1, 2)  # (b, n2, 3)

    # Calculate Chamfer Distance using PyTorch3D
    (dist1, dist2), _ = pytorch3d.loss.chamfer_distance(xyz1, xyz2, batch_reduction=None, point_reduction=None)

    if sqrt:
        dist1 = torch.sqrt(dist1 + eps)
        dist2 = torch.sqrt(dist2 + eps)

    return dist1, dist2


class PoseHead(nn.Module):
    """Adapted from Image-PA"""

    def __init__(self, in_features: int):
        super().__init__()
        self.mlp = nn.Linear(in_features, 256)
        self.trans = nn.Linear(256, 3)
        self.quat = nn.Linear(256, 4)
        self.quat.bias.data.zero_()

    def forward(self, feat):
        feat = torch.relu(self.mlp(feat))
        trans = self.trans(feat)
        quat_bias = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=feat.device, dtype=feat.dtype)
        quat = self.quat(feat) + quat_bias
        quat = quat / (1e-12 + quat.pow(2).sum(dim=-1, keepdim=True)).sqrt()
        out = torch.cat([trans, quat], dim=-1)
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding"""

    def __init__(
        self,
        d_model: int,
        # dropout,
        max_len: int = 1000,
    ):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        pe = torch.zeros((max_len, d_model))
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, dim: int = 1):
        """Add positional encoding to the input tensor.

        Args:
            X (torch.Tensor): The input tensor to add PE to, of shape (num, num_tokens, dim).
            dim (int, optional): The dim to add PE to. Defaults to 1, can be 0 or 1.

        Returns:
            torch.Tensor: The tensor with PE added.
        """
        pe = self.pe[: x.size(dim)]
        if dim == 0:
            pe = pe.unsqueeze(1)
        elif dim == 1:
            pe = pe.unsqueeze(0)
        else:
            raise ValueError("dim must be 0 or 1")
        return pe


class ManualPAAssemblyNet(nn.Module):
    def __init__(
        self,
        # EXPERIMENTAL
        step_concate_instance_one_hot: bool,
        part_concate_instance_one_hot: bool,
        part_concate_group_one_hot: bool,
        add_positional_encoding: bool,
        step_add_positional_encoding: bool,
        part_add_positional_encoding: bool,
        use_rope: bool,
        vision_token_mode: Literal["all", "all_wo_cls", "cls_only", "max_pooling", "center16"],
        diagram_global_feature: bool,
        part_global_feature: bool,
        order_mode: Literal["raw", "random", "learned", "gt"],
        permutation_tau: float,
        permutation_n_iter: int,
        step_diagram_feature_diff: bool,
        decoder_pe_mode: Literal["input_add", "layer_concat", "layer_add"],
        generator_type: Literal["parallel", "autoregressive"],
        # Model
        pointnet_encoder: nn.Module,
        vision_encoder: AutoModel,
        mask_decoder: nn.Module,
        freeze_vision_encoder: bool = False,
        # Transformer Decoder
        d_model: int = 768,
        num_layers: int = 3,
    ):
        super().__init__()

        # Encoders
        self.pointnet_encoder = pointnet_encoder
        pointnet_encoder_linear_in_features = (
            pointnet_encoder.out_features
        )  # NOTE: `out_features` only works for our custom PointNet
        pointnet_encoder_linear_out_features = d_model
        vision_encoder_linear_in_features = (
            vision_encoder.config.hidden_size
        )  # NOTE: `config` only works for huggingface
        vision_encoder_linear_out_features = d_model
        if step_concate_instance_one_hot:
            # vision_encoder_linear_in_features += 20
            vision_encoder_linear_out_features -= 20
        if part_concate_instance_one_hot:
            # pointnet_encoder_linear_in_features += 20
            pointnet_encoder_linear_out_features -= 20
        if part_concate_group_one_hot:
            # pointnet_encoder_linear_in_features += 20
            pointnet_encoder_linear_out_features -= 20
        if diagram_global_feature:
            vision_encoder_linear_in_features *= 2
        if part_global_feature:
            pointnet_encoder_linear_in_features *= 2
        self.pointnet_encoder_linear = nn.Linear(
            pointnet_encoder_linear_in_features, pointnet_encoder_linear_out_features
        )
        self.step_concate_instance_one_hot = step_concate_instance_one_hot
        self.part_concate_instance_one_hot = part_concate_instance_one_hot
        self.concate_group_one_hot = part_concate_group_one_hot
        self.step_add_positional_encoding = step_add_positional_encoding
        self.part_add_positional_encoding = part_add_positional_encoding
        self.vision_encoder = vision_encoder
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        self.vision_encoder_linear = nn.Linear(vision_encoder_linear_in_features, vision_encoder_linear_out_features)

        self.add_positional_encoding = add_positional_encoding
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Decoder
        rope = None
        if use_rope:
            rope = RotaryEmbedding(d_model // 8)
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            rope=rope,
            layer_concat=decoder_pe_mode == "layer_concat",
            layer_add=decoder_pe_mode == "layer_add",
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            # norm=nn.LayerNorm(d_model),
        )

        # Permutation
        if order_mode == "learned":
            self.permutation_tau = permutation_tau
            # self.permutation_generator = GumbelSinkhornPermutation(
            #     tau=permutation_tau,
            #     n_iter=permutation_n_iter,
            # )
        if step_diagram_feature_diff:
            self.step_diagram_difference_mlp = nn.Linear(d_model, d_model)

        if order_mode == "learned" or step_diagram_feature_diff:
            self.register_buffer("blank_step_diagram", torch.load("src/models/manual_pa/blank_step_diagram.pt"))

        # Heads
        self.pose_head = PoseHead(d_model)

        # Mask Decoder
        self.mask_decoder = mask_decoder

        self.vision_token_mode = vision_token_mode
        self.diagram_global_feature = diagram_global_feature
        self.part_global_feature = part_global_feature
        self.order_mode = order_mode
        self.step_diagram_feature_diff = step_diagram_feature_diff
        self.decoder_pe_mode = decoder_pe_mode
        self.generator_type = generator_type

        if step_diagram_feature_diff:
            self.step_diagram_feature_diff_mlp = nn.Linear(d_model * 2, d_model)

        if order_mode == "learned":
            self.step_diagram_difference_mlp = nn.Linear(d_model, d_model)
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                rope=rope,
            )
            # self.step_diagram_sa_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.part_sa_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _get_positional_encoding(
        self,
        features: list[torch.Tensor],
        permutation_matrices: list[torch.Tensor] | None = None,
    ):
        """Get positional encoding to the feature tensor.

        # 1. Get positional encoding to the num_tokens dimension of the feature tensor.
        2. Get positional encoding to the num (num_part, num_step) dimension of the feature tensor.

        Args:
            features (list[torch.Tensor]): a tuple of B tensors of shape (num, num_tokens, dim)
            permutation_matrices (list[torch.Tensor], optional): a list of B tensors of shape (num, num). Defaults to
                None.
        """
        ret = []
        for i in range(len(features)):
            pe = self.positional_encoding(features[i], dim=0)
            if permutation_matrices is not None:
                pe = torch.einsum("ij,jkm->ikm", permutation_matrices[i], pe)
            ret.append(pe)
            # features[i] = features[i] + self.positional_encoding(features[i], dim=1)
        return ret

    def forward_step_imgs(self, step_imgs: list[torch.Tensor]):
        step_imgs_lengths = [x.size(0) for x in step_imgs]
        packed_step_imgs = torch.cat(step_imgs)
        packed_step_imgs_features = self.vision_encoder(packed_step_imgs).last_hidden_state
        if self.vision_token_mode == "all":
            pass
        elif self.vision_token_mode == "all_wo_cls":
            packed_step_imgs_features = packed_step_imgs_features[:, 1:]  # Remove the first token (CLS)
        elif self.vision_token_mode == "cls_only":
            packed_step_imgs_features = packed_step_imgs_features[:, :1]  # Only use the first token (CLS)
        elif self.vision_token_mode == "max_pooling":
            packed_step_imgs_features, _ = packed_step_imgs_features.max(dim=1)
        elif self.vision_token_mode == "center16":
            packed_step_imgs_features = packed_step_imgs_features[:, 1:]
            packed_step_imgs_features = packed_step_imgs_features.index_select(
                1,
                torch.tensor(
                    [102, 103, 104, 105, 118, 119, 120, 121, 134, 135, 136, 137, 150, 151, 152, 153],
                    device=packed_step_imgs_features.device,
                ),
            )
        else:
            raise ValueError(f"Invalid vision_token_mode: {self.vision_token_mode}")
        if self.diagram_global_feature:
            step_imgs_features = list(torch.split(packed_step_imgs_features, step_imgs_lengths, dim=0))
            for idx, step_imgs_feature in enumerate(step_imgs_features):
                diagram_global_feature, _ = step_imgs_feature.max(dim=0)
                step_imgs_features[idx] = torch.cat(
                    [
                        step_imgs_feature,
                        diagram_global_feature.unsqueeze(0).expand(step_imgs_feature.size(0), -1, -1),
                    ],
                    dim=-1,
                )
            packed_step_imgs_features = torch.cat(step_imgs_features)
        packed_step_imgs_features = self.vision_encoder_linear(packed_step_imgs_features)
        if self.step_diagram_feature_diff:
            blank_step_diagram_feature = self.vision_encoder(self.blank_step_diagram.unsqueeze(0)).last_hidden_state
            blank_step_diagram_feature = self.vision_encoder_linear(blank_step_diagram_feature)
            step_imgs_features = list(torch.split(packed_step_imgs_features, step_imgs_lengths, dim=0))
            for idx, step_imgs_feature in enumerate(step_imgs_features):
                previous_features = torch.cat([blank_step_diagram_feature, step_imgs_feature[:-1]], dim=0)
                # Normal concat
                step_imgs_features[idx] = self.step_diagram_feature_diff_mlp(
                    torch.cat([step_imgs_feature, previous_features], dim=-1)
                    # step_imgs_feature - previous_features
                )
                # Multi-Head concat
            return step_imgs_features
        if self.step_concate_instance_one_hot:
            instance_one_hot = [
                F.one_hot(
                    torch.arange(length, device=packed_step_imgs_features.device),
                    num_classes=20,
                )
                for length in step_imgs_lengths
            ]
            packed_instance_one_hot = (
                torch.cat(instance_one_hot).unsqueeze(1).repeat(1, packed_step_imgs_features.shape[1], 1)
            )
            packed_step_imgs_features = torch.cat([packed_step_imgs_features, packed_instance_one_hot], dim=-1)
        step_imgs_features = torch.split(packed_step_imgs_features, step_imgs_lengths, dim=0)
        return list(step_imgs_features)

    def forward_part_pts(
        self,
        part_pts: list[torch.Tensor],
        instance_one_hot: list[torch.Tensor],
        group_one_hot: list[torch.Tensor],
    ):
        part_pts_lengths = [x.size(0) for x in part_pts]
        packed_part_pts = torch.cat(part_pts)
        packed_part_pts_features, *other = self.pointnet_encoder(packed_part_pts)
        # TODO: decouple `other`, to have a standard api for all models, including the
        # 1. feature
        # 2. positional embedding
        # 3. loss for STN
        if self.part_global_feature:
            part_pts_features = list(torch.split(packed_part_pts_features, part_pts_lengths, dim=0))
            for idx, part_pts_feature in enumerate(part_pts_features):
                part_global_feature, _ = part_pts_feature.max(dim=0)  # num_groups x d_pointnet
                part_pts_features[idx] = torch.cat(
                    [
                        part_pts_feature,
                        part_global_feature.unsqueeze(0).expand(part_pts_feature.size(0), -1, -1),
                    ],
                    dim=-1,
                )
            packed_part_pts_features = torch.cat(part_pts_features)
        packed_part_pts_features = self.pointnet_encoder_linear(packed_part_pts_features)
        if self.part_concate_instance_one_hot:
            # cumsum along columns
            # instance_one_hot = [torch.cumsum(one_hot, dim=0) for one_hot in instance_one_hot]
            packed_instance_one_hot = (
                torch.cat(instance_one_hot).unsqueeze(1).repeat(1, packed_part_pts_features.shape[1], 1)
            )
            packed_part_pts_features = torch.cat([packed_part_pts_features, packed_instance_one_hot], dim=-1)
        if self.concate_group_one_hot:
            packed_group_one_hot = torch.cat(group_one_hot).unsqueeze(1).repeat(1, packed_part_pts_features.shape[1], 1)
            packed_part_pts_features = torch.cat([packed_part_pts_features, packed_group_one_hot], dim=-1)
        part_pts_features = torch.split(packed_part_pts_features, part_pts_lengths, dim=0)
        return list(part_pts_features)

    def _flatten_pad_features(self, features: list[torch.Tensor]):
        flattened_features = [rearrange(x, "len tokens dim -> (len tokens) dim") for x in features]
        feature_lengths = [x.shape[0] for x in flattened_features]
        padded_features = nn.utils.rnn.pad_sequence(flattened_features)
        padded_mask = torch.zeros(len(features), padded_features.shape[0], device=padded_features.device)
        for i, length in enumerate(feature_lengths):
            padded_mask[i, length:] = 1
        padded_mask = padded_mask.bool()
        return padded_features, padded_mask, feature_lengths

    def forward_decoder(
        self,
        step_imgs_features: tuple[torch.Tensor],
        part_pts_features: tuple[torch.Tensor],
        step_imgs_pe: list[torch.Tensor] | None,
        part_pts_pe: list[torch.Tensor] | None,
        gt_orders: list[torch.Tensor] | None = None,
    ):
        # Transformer Decoder
        if self.generator_type == "parallel":
            padded_step_imgs_features, padded_step_imgs_mask, _ = self._flatten_pad_features(step_imgs_features)
            padded_part_pts_features, padded_part_pts_mask, part_pts_feature_lengths = self._flatten_pad_features(
                part_pts_features
            )
            # expand the positional encoding
            padded_part_pts_pe = None
            padded_step_imgs_pe = None
            if self.add_positional_encoding:
                num_vision_tokens = step_imgs_features[0].shape[1]
                step_imgs_pe = [x.expand(-1, num_vision_tokens, -1) for x in step_imgs_pe]
                num_part_tokens = part_pts_features[0].shape[1]
                part_pts_pe = [x.expand(-1, num_part_tokens, -1) for x in part_pts_pe]
                padded_step_imgs_pe, _, _ = self._flatten_pad_features(step_imgs_pe)
                padded_part_pts_pe, _, _ = self._flatten_pad_features(part_pts_pe)

                if self.decoder_pe_mode == "input_add":
                    if self.step_add_positional_encoding:
                        # HACK: comment this line to disable positional encoding for steps
                        padded_step_imgs_features = padded_step_imgs_features + padded_step_imgs_pe
                    if self.part_add_positional_encoding:
                        # HACK: comment this line to disable positional encoding for parts
                        padded_part_pts_features = padded_part_pts_features + padded_part_pts_pe
                    pass
                elif self.decoder_pe_mode == "layer_concat":
                    pass
                elif self.decoder_pe_mode == "layer_add":
                    pass
                else:
                    raise ValueError(f"Invalid decoder_pe_mode: {self.decoder_pe_mode}")
            output = self.transformer_decoder(
                tgt=padded_part_pts_features,
                memory=padded_step_imgs_features,
                tgt_key_padding_mask=padded_part_pts_mask,
                memory_key_padding_mask=padded_step_imgs_mask,
                # tgt_pe=padded_part_pts_pe,  # HACK: comment this line to disable positional encoding for parts
                # memory_pe=padded_step_imgs_pe,  # HACK: comment this line to disable positional encoding for steps
            )  # (num_layers, max_num_parts, batch_size, d_model)
            # Reformat the output
            last_hidden_state = []
            for i, length in enumerate(part_pts_feature_lengths):
                last_hidden_state.append(output[:, :length, i])
        elif self.generator_type == "autoregressive":
            last_hidden_state = []
            for step_img_feature, part_pts_feature, gt_order in zip(
                step_imgs_features, part_pts_features, gt_orders
            ):  # for each shape in the batch
                num_steps = step_img_feature.shape[0]
                tgt = part_pts_feature
                outputs = []
                for i in range(num_steps):
                    output = self.transformer_decoder(
                        tgt=tgt,  # (num_parts, 1, d_model)
                        memory=step_img_feature[i].unsqueeze(1),  # (num_tokens, 1, d_model)
                    )  # (num_layers, num_parts, 1, d_model)
                    last_output = output[:, -1, 0].unsqueeze(1)  # (num_layers, 1, d_model)
                    outputs.append(last_output)
                    last_layer_last_output = last_output[-1].unsqueeze(1)  # (1, 1, d_model)
                    tgt = torch.cat([tgt, last_layer_last_output], dim=0)
                _last_hidden_state = torch.stack(outputs, dim=1).squeeze(2)  # (num_layers, num_parts, d_model)
                _last_hidden_state = _last_hidden_state.index_select(1, torch.argsort(gt_order))
                last_hidden_state.append(_last_hidden_state)

        else:
            raise ValueError(f"Invalid generator_type: {self.generator_type}")

        return last_hidden_state

    def forward_pose_head(self, decoder_output: list[torch.Tensor]):
        packed_decoder_output = torch.cat(decoder_output, dim=1)
        pose_pred = self.pose_head(packed_decoder_output)
        return pose_pred.split([x.shape[1] for x in decoder_output], dim=1)

    # def forward_mask_decoder(self, step_imgs_features: list[torch.Tensor], decoder_output: list[torch.Tensor]):
    #     ret_masks = []
    #     ret_ious = []
    # for (
    #     img_features,  # (num_steps, num_tokens, d_model)
    #     part_feature,  # (num_layers, num_parts, d_model)
    # ) in zip(step_imgs_features, decoder_output):
    #     num_layers = part_feature.shape[0]
    #     num_parts = part_feature.shape[1]

    #     step_masks = []
    #     step_ious = []
    #     for step_img_features in img_features:
    #         step_img_features = step_img_features[1:]  # Remove the first token (CLS)  # (num_tokens - 1, d_model)
    #         step_img_pe = self.positional_encoding(step_img_features, dim=0).squeeze(1)  # (num_tokens - 1, d_model)
    #         part_feature = part_feature.view(-1, part_feature.shape[-1])  # (num_layers * num_parts, d_model)
    #         masks, ious = self.mask_decoder(
    #             step_img_features,
    #             step_img_pe,
    #             part_feature,
    #         )  # (num_layers * num_parts, 64, 64), (num_layers * num_parts, 1)
    #         masks = masks.view(num_layers, 1, num_parts, 64, 64)
    #         ious = ious.view(num_layers, 1, num_parts, 1)
    #         step_masks.append(masks)
    #         step_ious.append(ious)
    #     ret_masks.append(torch.cat(step_masks, dim=1))  # (num_layers, num_steps, num_parts, 64, 64)
    #     ret_ious.append(torch.cat(step_ious, dim=1))  # (num_layers, num_steps, num_parts, 1)
    # return ret_masks, ret_ious

    def forward(
        self,
        step_imgs: list[torch.Tensor],  # a list of B tensors of shape (num_steps, 3, H, W)
        part_pts: list[torch.Tensor],  # a list of B tensors of shape (num_parts, num_points, 3)
        instance_one_hot: list[torch.Tensor],  # a list of B tensors of shape (num_parts, 20)
        group_one_hot: list[torch.Tensor],  # a list of B tensors of shape (num_parts, 20)
    ):
        step_imgs_features = self.forward_step_imgs(
            step_imgs
        )  # a list of B tensors of shape (num_steps, num_tokens, d_model)
        part_pts_features = self.forward_part_pts(
            part_pts, instance_one_hot, group_one_hot
        )  # a list of B tensors of shape (num_parts, num_groups, d_model)
        permutation_matrices_soft = None
        permutation_matrices_hard = None
        contrastive_matrix = None
        if self.add_positional_encoding:
            if self.order_mode == "raw":
                permutation_matrices_hard = []
                for part_pts_feature in part_pts_features:
                    num_parts = part_pts_feature.shape[0]
                    permutation_matrices_hard.append(
                        torch.eye(num_parts, device=part_pts_feature.device, dtype=torch.float)
                    )
            elif self.order_mode == "random":
                permutation_matrices_hard = []
                for part_pts_feature in part_pts_features:
                    num_parts = part_pts_feature.shape[0]
                    permutation_matrices_hard.append(
                        F.one_hot(
                            torch.randperm(num_parts, device=part_pts_feature.device),
                        ).float()
                        if self.training
                        else torch.eye(num_parts, device=part_pts_feature.device, dtype=torch.float)
                    )
            elif self.order_mode == "learned":
                permutation_matrices_hard = []
                permutation_matrices_soft = []
                # forward the blank step diagram
                blank_step_diagram_feature = self.vision_encoder(self.blank_step_diagram.unsqueeze(0)).last_hidden_state
                blank_step_diagram_feature = self.vision_encoder_linear(blank_step_diagram_feature)
                # aggregate for contrastive learning
                _step_imgs_features_all = []
                _part_pts_features_all = []
                lengths = []
                for step_imgs_feature, part_pts_feature in zip(step_imgs_features, part_pts_features):
                    # previous_features = torch.cat([blank_step_diagram_feature, step_imgs_feature[:-1]], dim=0)
                    # step_imgs_feature = self.step_diagram_difference_mlp(step_imgs_feature - previous_features)
                    # step_imgs_feature = self.step_diagram_difference_mlp(step_imgs_feature + previous_features)
                    # step_imgs_feature = self.step_diagram_difference_mlp(
                    #     torch.cat([step_imgs_feature, previous_features], dim=-1)
                    # )
                    # step_imgs_feature = self.step_diagram_difference_mlp(
                    #     torch.cat([step_imgs_feature, step_imgs_feature - previous_features], dim=-1)
                    # )
                    # step_imgs_feature = self.step_diagram_difference_mlp(
                    #     torch.cat(
                    #         [
                    #             torch.abs(step_imgs_feature - previous_features),
                    #             # torch.sum(step_imgs_feature * previous_features, dim=-1, keepdim=True),
                    #         ],
                    #         dim=-1,
                    #     )
                    # )

                    max_pooled_step_imgs_feature = step_imgs_feature.max(dim=1).values
                    max_pooled_part_pts_feature = part_pts_feature.max(dim=1).values

                    # EXPERIMENTAL
                    # max_pooled_step_imgs_feature = step_imgs_feature.max(dim=1).values
                    # max_pooled_part_pts_feature = part_pts_feature.max(dim=1).values
                    # max_pooled_step_imgs_feature = self.step_diagram_sa_encoder(
                    #     max_pooled_step_imgs_feature.unsqueeze(1)
                    # ).squeeze(1)
                    # max_pooled_part_pts_feature = self.part_sa_encoder(
                    #     max_pooled_part_pts_feature.unsqueeze(1)
                    # ).squeeze(1)
                    # step_imgs_feature = step_imgs_feature.view(-1, step_imgs_feature.shape[-1])
                    # # step_imgs_pe = self._get_positional_encoding([step_imgs_feature])[0]
                    # src = torch.cat(
                    #     [step_imgs_feature.unsqueeze(1), part_pts_feature],
                    #     dim=0,
                    # )
                    # concatenated_features = self.step_diagram_sa_encoder(src).squeeze(1)
                    # step_imgs_feature = concatenated_features[: len(step_imgs_feature)]
                    # part_pts_feature = concatenated_features[len(step_imgs_feature) :]

                    # step_imgs_feature = step_imgs_feature.view(
                    #     part_pts_feature.shape[0], -1, step_imgs_feature.shape[-1]
                    # )
                    ### One Encoder for both
                    # src = torch.cat(
                    #     [max_pooled_step_imgs_feature.unsqueeze(1), max_pooled_part_pts_feature.unsqueeze(1)], dim=0
                    # )
                    # concatenated_features = self.step_diagram_sa_encoder(src).squeeze(1)
                    # max_pooled_step_imgs_feature = concatenated_features[: len(step_imgs_feature)]
                    # max_pooled_part_pts_feature = concatenated_features[len(step_imgs_feature) :]
                    ### One Encoder for parts only
                    # src = max_pooled_part_pts_feature.unsqueeze(1)
                    # src_pos = self._get_positional_encoding([src])[0]
                    # max_pooled_part_pts_feature = self.part_sa_encoder(src + src_pos).squeeze(1)
                    # EXPERIMENTAL

                    _step_imgs_features_all.append(max_pooled_step_imgs_feature)
                    _part_pts_features_all.append(max_pooled_part_pts_feature)
                    lengths.append(max_pooled_part_pts_feature.shape[0])
                step_imgs_features_all = torch.cat(_step_imgs_features_all, dim=0)
                part_pts_features_all = torch.cat(_part_pts_features_all, dim=0)
                contrastive_matrix = (part_pts_features_all @ step_imgs_features_all.t()) / self.permutation_tau

                # Start of Optimal Transport
                for _ in range(3):
                    contrastive_matrix = contrastive_matrix - contrastive_matrix.logsumexp(dim=-1, keepdim=True)
                    contrastive_matrix = contrastive_matrix - contrastive_matrix.logsumexp(dim=-2, keepdim=True)
                # End of Optimal Transport

                for length in lengths:
                    permutation_matrix = contrastive_matrix[:length, :length]
                    permutation_matrices_soft.append(permutation_matrix)
                    with torch.no_grad():
                        row, col = linear_sum_assignment(-permutation_matrix.detach().cpu())
                        permutation_matrix_hard = coo_matrix((np.ones_like(row), (row, col))).toarray()
                        permutation_matrix_hard = (
                            torch.from_numpy(permutation_matrix_hard).float().to(permutation_matrix.device)
                        )
                        permutation_matrices_hard.append(permutation_matrix_hard)

            elif self.order_mode == "gt":
                permutation_matrices_hard = []
                for one_hot in instance_one_hot:
                    num_parts = one_hot.shape[0]
                    permutation_matrices_hard.append(one_hot[:, :num_parts].float())
            (
                step_imgs_pe,
                part_pts_pe,
            ) = (
                self._get_positional_encoding(step_imgs_features),
                self._get_positional_encoding(part_pts_features, permutation_matrices_hard),
            )
        else:
            step_imgs_pe = None
            part_pts_pe = None
        decoder_output = self.forward_decoder(
            step_imgs_features,
            part_pts_features,
            step_imgs_pe,
            part_pts_pe,
            [torch.argmax(one_hot, dim=-1) for one_hot in instance_one_hot],
        )  # a list of B tensors of shape (num_layers, num_parts x num_groups, d_model)
        # Max pooling along the groups dimension
        decoder_output_pooled = []
        for output in decoder_output:
            decoder_output_pooled.append(
                reduce(
                    output,
                    "layers (parts groups) dim -> layers parts dim",
                    "max",
                    groups=part_pts_features[0].shape[1],
                )
            )
        pred_poses = self.forward_pose_head(
            decoder_output_pooled
        )  # a list of B tensors of shape (num_layers, num_parts, 7)
        # (
        #     pred_masks,  # a list of B tensors of shape (num_layers, num_steps, num_parts, 64, 64)
        #     pred_ious,  # a list of B tensors of shape (num_layers, num_steps, num_parts, 1)
        # ) = self.forward_mask_decoder(
        #     step_imgs_features,
        #     decoder_output_pooled,
        # )
        # return pred_poses, pred_masks, pred_ious
        return pred_poses, [], [], permutation_matrices_soft, permutation_matrices_hard, contrastive_matrix

    @torch.no_grad
    def linear_assignment(
        self,
        similar_cnt: torch.Tensor,  # a tensor of shape (P_i, 1)
        pts: torch.Tensor,  # a tensor of shape (P_i, 1000, 3)
        pred_poses: torch.Tensor,  # a tensor of shape (P_i, 7)
        gt_poses: torch.Tensor,  # a tensor of shape (P_i, 7)
    ):
        ret1 = []
        ret2 = []
        num_parts = similar_cnt.shape[0]
        t = 0
        while t < num_parts:
            cnt = similar_cnt[t].item()
            if cnt == 1:
                # If there is only one part, we don't need to do linear assignment
                ret1.append(t)
                ret2.append(t)
                t += 1
                continue
            cur_pts = pts[t : t + cnt]  # (cnt, 1000, 3)
            cur_pred_poses = pred_poses[t : t + cnt].unsqueeze(1)  # (cnt, 1, 7)
            cur_gt_poses = gt_poses[t : t + cnt].unsqueeze(1)  # (cnt, 1, 7)

            # Transform the points
            cur_pred_pts = qrot(cur_pred_poses[..., 3:], cur_pts) + cur_pred_poses[..., :3]  # (cnt, 1000, 3)
            cur_gt_pts = qrot(cur_gt_poses[..., 3:], cur_pts) + cur_gt_poses[..., :3]  # (cnt, 1000, 3)

            # Compute the distance matrix
            cur_pred_pts = cur_pred_pts.unsqueeze(1).repeat(1, cnt, 1, 1).view(-1, 1000, 3)  # (cnt * cnt, 1000, 3)
            cur_gt_pts = cur_gt_pts.unsqueeze(0).repeat(cnt, 1, 1, 1).view(-1, 1000, 3)  # (cnt * cnt, 1000, 3)
            dist1, dist2 = pytorch3d_chamfer_distance(cur_pred_pts, cur_gt_pts, transpose=False)
            dist_mat = (dist1.mean(1) + dist2.mean(1)).view(cnt, cnt)  # (cnt, cnt)
            dist_mat = torch.clamp(
                dist_mat, max=1
            )  # TODO: investigate the threshold # TODO: disable this for inference?

            # Linear assignment
            rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())
            for i in rind:
                ret1.append(t + i)
            for i in cind:
                ret2.append(t + i)

            t += cnt

        return ret1, ret2

    def get_center_loss(self, center1, center2):
        loss_per_data = (center1 - center2).pow(2).sum(dim=1)
        return loss_per_data

    def get_quat_loss(self, pts, quat1, quat2):
        num_point = pts.shape[1]

        pts1 = qrot(quat1.unsqueeze(1).repeat(1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(1).repeat(1, num_point, 1), pts)

        dist1, dist2 = pytorch3d_chamfer_distance(pts1, pts2, transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        return loss_per_data

    def get_l2_rotation_loss(self, pts, quat1, quat2):
        num_point = pts.shape[1]

        pts1 = qrot(quat1.unsqueeze(1).repeat(1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(1).repeat(1, num_point, 1), pts)

        loss_per_data = (pts1 - pts2).pow(2).sum(dim=2).mean(dim=1)

        return loss_per_data

    def get_shape_chamfer_loss(self, pts, quat1, quat2, center1, center2, part_cnt):
        num_point = pts.shape[1]
        part_pcs1 = qrot(quat1.unsqueeze(1).repeat(1, num_point, 1), pts) + center1.unsqueeze(1).repeat(1, num_point, 1)
        part_pcs2 = qrot(quat2.unsqueeze(1).repeat(1, num_point, 1), pts) + center2.unsqueeze(1).repeat(1, num_point, 1)
        t = 0
        shape_pcs1 = []
        shape_pcs2 = []
        for cnt in part_cnt:
            cur_shape_pc1 = part_pcs1[t : t + cnt].view(1, -1, 3)
            cur_shape_pc2 = part_pcs2[t : t + cnt].view(1, -1, 3)
            with torch.no_grad():
                # idx1 = furthest_point_sample(cur_shape_pc1, 2048).long()[0]
                idx1 = sample_farthest_points(cur_shape_pc1, K=2048)[1][0]
                # idx2 = furthest_point_sample(cur_shape_pc2, 2048).long()[0]
                idx2 = sample_farthest_points(cur_shape_pc2, K=2048)[1][0]
            shape_pcs1.append(cur_shape_pc1[:, idx1])
            shape_pcs2.append(cur_shape_pc2[:, idx2])
            t += cnt
        shape_pcs1 = torch.cat(shape_pcs1, dim=0)  # numshapes x 2048 x 3
        shape_pcs2 = torch.cat(shape_pcs2, dim=0)  # numshapes x 2048 x 3

        dist1, dist2 = pytorch3d_chamfer_distance(shape_pcs1, shape_pcs2, transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

        return loss_per_data

    def get_mask_loss(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor):
        loss_per_data = []
        for (
            pred_mask,  # (1, num_parts, 64, 64)  # TODO: support multiple steps
            gt_mask,  # (num_parts, 224, 224)
        ) in zip(pred_masks, gt_masks):
            pred_mask = pred_mask.sigmoid()
            # resize pred_mask to the size of gt_mask
            pred_mask = F.interpolate(
                pred_mask,
                size=gt_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            gt_mask = gt_mask.float()
            inter = (pred_mask * gt_mask).mean(dim=[1, 2])
            union = pred_mask.mean(dim=[1, 2]) + gt_mask.mean(dim=[1, 2]) - inter
            loss = -inter / (union + 1e-12)
            loss_per_data.append(loss.mean())
        return torch.stack(loss_per_data)


if __name__ == "__main__":
    # from src.models.manual_pa.pointnet import PointNetFeat
    # from src.models.manual_pa.pointnet_group import PointNetGroupFeat as PointNetFeat
    # from src.models.manual_pa.pointnet_lite import PointNetLiteFeat as PointNetFeat

    # pointnet_encoder = PointNetFeat()
    # vision_encoder = AutoModel.from_pretrained("facebook/dinov2-small")

    # model = ManualPAAssemblyNet(pointnet_encoder, vision_encoder)
    # part_pts = [
    #     torch.randn(5, 1000, 3),
    # ]
    # step_imgs = [
    #     torch.randn(1, 3, 224, 224),
    #     torch.randn(1, 3, 224, 224),
    #     torch.randn(1, 3, 224, 224),
    # ]
    # output = model(step_imgs, part_pts)
    # print([item.shape for item in output])
    pe = PositionalEncoding(768)
    x = torch.randn(32, 256, 768)
    x = pe(x, dim=0)
    x = pe(x, dim=1)
    print(x.shape)
