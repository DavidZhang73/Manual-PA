import json
import os
import random
from typing import Callable, Literal

import numpy as np
import pytorch_lightning as pl
import torch
from iopath import PathManager
from PIL import Image, ImageChops
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io.ply_io import _load_ply
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms import functional as F
from tqdm import tqdm


class OrderRandPermutation(nn.Module):
    def __init__(self, beta: float = 0.6):
        super().__init__()
        self.beta = beta

    def matching(self, alpha: torch.Tensor):
        row, col = linear_sum_assignment(-alpha)
        permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
        return torch.from_numpy(permutation_matrix).float().to(alpha.device)

    def forward(self, x: torch.Tensor):
        n = x.size(0)
        identity = torch.eye(n, device=x.device)
        noise = torch.rand_like(identity)
        log_alpha = identity * self.beta + noise
        P = self.matching(log_alpha)
        return x[P.argmax(dim=-1)]


class OrderGumbelSinkhornRandPermutation(nn.Module):
    def __init__(self, alpha: float, n_iters: int = 10):
        super().__init__()
        self.alpha = alpha
        self.n_iters = n_iters

    def gumbel_noise(self, shape, eps=1e-20):
        """Generate Gumbel noise."""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def sinkhorn(self, log_alpha):
        """Apply Sinkhorn normalization to log_alpha."""
        for _ in range(self.n_iters):
            log_alpha = log_alpha - log_alpha.logsumexp(dim=-1, keepdim=True)
            log_alpha = log_alpha - log_alpha.logsumexp(dim=-2, keepdim=True)
        return log_alpha.exp()

    def matching(self, alpha: torch.Tensor):
        """Negate the probability matrix to serve as cost matrix and solve the linear sum assignment problem.

        Args:
            alpha (torch.Tensor): The N x N probability matrix.

        Returns:
            torch.Tensor: The N x N permutation matrix.
        """
        row, col = linear_sum_assignment(-alpha)
        # Create the permutation matrix.
        permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
        return torch.from_numpy(permutation_matrix).float().to(alpha.device)

    def forward(self, x: torch.Tensor):
        """Apply Gumbel-Sinkhorn permutation to order."""
        n = x.size(0)
        # Create an initial identity matrix
        identity = torch.eye(n, device=x.device)

        # Add Gumbel noise to the identity matrix
        noise = self.gumbel_noise(identity.shape)
        # noise = torch.rand_like(identity)
        log_alpha = identity + noise * self.alpha

        # Apply Sinkhorn to get doubly stochastic matrix
        P = self.sinkhorn(log_alpha)
        P = self.matching(P)

        # Apply the permutation to the order vector
        return x[P.argmax(dim=-1)]


class IMADataset(Dataset):
    """Pytorch Dataset for ikea Manual Assembly dataset."""

    def __init__(
        self,
        dataset_path: str = "data/ikea-manual-assembly",
        source: Literal["partnet", "ikea-manual"] = "partnet",
        type: Literal["synthetic", "synthetic-stepwise", "real"] = "synthetic",
        split_pathname: str = "splits/image_pa/all.train.txt",
        data_features: list[
            Literal[
                "img",  # rgb image of the 2D line drawing, png format with alpha channel
                "imgs",  # rgb image of the 2D line drawing, png format with alpha channel
                "pts",  # point clouds of the N parts
                "masks",  # masks of the N parts
                "meshes",  # meshes of the N parts
                "similar_parts_cnt",  # number of similar parts of the N parts
                "similar_parts_edge_indices",  # edge indices of the similar parts
                "bbox_size",  # bounding box size of the N parts
                "ins_one_hot",  # instance one hot encoding of the N parts
                "order",  # order of the N parts
                "group",  # group of the N parts
                "parts_cam_dof",  # Camera 6DoF of the N parts
                "part_to_shape_matrix",  # transformation matrix from normalized parts to shape
                "template_features",  # template image features of the N parts
            ]
        ] = ["img", "pts", "masks", "similar_parts_cnt", "bbox_size", "ins_one_hot"],
        image_transform: Callable = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),  # image transform for the rgb image
        order_transform: Callable | None = None,  # order transform for the order of the parts
        num_views: int = 24,  # number of views for each shape, set to 1 for validation and testing.
        # Note that we will randomly sample one view if num_views > 1.
        step_mode: Literal["lastonly", "all"] = "all",  # "lastonly": only the last step, "all": all steps
        template_features_folder: str = "template_features/dinov2_1024",
        depth_folder: str = "depths/depth_anything_v2_224",
        max_parts_cnt: int = 20,  # the maximum number of parts in across all datasets.
        order_pathname: str
        | None = None,  # HACK: path name to predicted pred_orders.json file. Set to None to use the ground truth order.
        load_shape_id_to_part_cnt_map: bool = False,  # build shape_id_to_part_cnt_map for IMABatchSampler for training.
    ):
        self.data_features = data_features
        self.image_transform = image_transform
        self.order_transform = order_transform
        self.template_features_folder = template_features_folder
        self.depth_folder = depth_folder
        self.data_path = os.path.join(dataset_path, source, type)
        self.step_mode = step_mode
        self.num_views = num_views
        self.max_parts_cnt = max_parts_cnt
        self.order_pathname = order_pathname

        with open(os.path.join(self.data_path, split_pathname)) as f:
            shape_ids = [int(line.strip()) for line in f.readlines()]
        self.shape_ids = shape_ids

        if load_shape_id_to_part_cnt_map:
            self.shape_id_to_part_cnt_map = {}
            for shape_id in shape_ids:
                meta_path = os.path.join(self.data_path, "data", str(shape_id), "meta.json")
                with open(meta_path) as f:
                    meta = json.load(f)
                self.shape_id_to_part_cnt_map[shape_id] = len(meta.keys())

    def __len__(self):
        return len(self.shape_ids)

    def __getitem__(self, idx: int):
        ret = {}
        shape_id = self.shape_ids[idx]
        # get ground truth poses
        with open(os.path.join(self.data_path, "data", str(shape_id), "poses.json")) as f:
            gt_poses = json.load(f)
        _step_ids = sorted([int(step_id) for step_id in gt_poses["0"].keys()])
        if self.step_mode == "lastonly":
            step_ids = [_step_ids[-1]]
        else:
            step_ids = _step_ids
        if self.num_views > 1:
            view_ids = [random.randint(0, self.num_views - 1) for _ in range(len(step_ids))]
        else:
            view_ids = [0 for _ in range(len(step_ids))]
        # get meta data of this shape
        meta_path = os.path.join(self.data_path, "data", str(shape_id), "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        part_ids = sorted(meta.keys())
        equiv_edge_indices = []  # for similar_parts_edge_indices
        # sort part_ids and convert it into a list
        # such that single parts come first, followed by multi-parts groups
        single_parts = [part_id for part_id in part_ids if len(meta[part_id]["similar_parts"]) == 1]
        other_parts = [part_id for part_id in part_ids if len(meta[part_id]["similar_parts"]) > 1]
        new_part_ids = sorted(single_parts)
        t = len(single_parts)
        for part_id in other_parts:
            if part_id not in new_part_ids:
                cur_cnt = len(meta[part_id]["similar_parts"])
                new_part_ids.extend(sorted(meta[part_id]["similar_parts"]))
                for t1 in range(t, t + cur_cnt):
                    for t2 in range(t, t + cur_cnt):
                        if t1 != t2:
                            equiv_edge_indices.append([t1, t2])
                t += cur_cnt
        part_ids = new_part_ids

        step_part_indices = []
        for step_id in step_ids:
            step_part_indices.append(
                [part_ids.index(part_id) for part_id in part_ids if part_id in gt_poses["0"][str(step_id)].keys()]
            )

        # Basic info
        ret = dict(
            shape_id=shape_id,
            view_ids=view_ids,
            step_ids=step_ids,
            part_ids=part_ids,
            step_part_indices=step_part_indices,
            total_parts_cnt=len(part_ids),
        )

        # Predicted Order
        if self.order_pathname is not None:
            with open(self.order_pathname) as f:
                pred_orders = json.load(f)

        for data_feature in self.data_features:
            if "img" == data_feature:
                imgs = []
                for view_id, step_id in zip(view_ids, step_ids):
                    img_path = os.path.join(
                        self.data_path,
                        "data",
                        str(shape_id),
                        "imgs",
                        f"{view_id:03d}_{step_id:03d}.png",
                    )
                    _img = Image.open(img_path)
                    img = _img
                    # if the image has an alpha channel, use it as mask, and paste it on a white canvas
                    if _img.mode == "RGBA":
                        img = Image.new("RGB", _img.size, (255, 255, 255))
                        img.paste(_img, mask=_img.split()[3])
                    imgs.append(self.image_transform(img))
                ret["img"] = torch.stack(imgs)
                # S x 3 x 224 x 224 of type torch.float32

            # elif "imgs" == data_feature:  # HACK: stepwise
            #     imgs = []
            #     view_id = view_ids[0]
            #     for step_id in _step_ids:
            #         img_path = os.path.join(
            #             self.data_path,
            #             "data",
            #             str(shape_id),
            #             "imgs",
            #             f"{view_id:03d}_{step_id:03d}.png",
            #         )
            #         _img = Image.open(img_path)
            #         img = _img
            #         # if the image has an alpha channel, use it as mask, and paste it on a white canvas
            #         if _img.mode == "RGBA":
            #             img = Image.new("RGB", _img.size, (255, 255, 255))
            #             img.paste(_img, mask=_img.split()[3])
            #         imgs.append(self.image_transform(img))
            #     ret["imgs"] = torch.stack(imgs)
            #     # S x 3 x 224 x 224 of type torch.float32

            elif "imgs" == data_feature:  # stepwise, only the difference
                imgs = []
                view_id = view_ids[0]
                previous_img = Image.new("RGB", (224, 224), (255, 255, 255))
                for step_id in _step_ids:
                    img_path = os.path.join(
                        self.data_path,
                        "data",
                        str(shape_id),
                        "imgs",
                        f"{view_id:03d}_{step_id:03d}.png",
                    )
                    _img = Image.open(img_path)
                    img = _img
                    # if the image has an alpha channel, use it as mask, and paste it on a white canvas
                    if _img.mode == "RGBA":
                        img = Image.new("RGB", _img.size, (255, 255, 255))
                        img.paste(_img, mask=_img.split()[3])
                    img_diff = ImageChops.difference(img, previous_img)
                    # img_diff = ImageChops.invert(img_diff.convert("L")).convert("RGB")
                    imgs.append(self.image_transform(img_diff))
                    previous_img = img
                ret["imgs"] = torch.stack(imgs)
                # S x 3 x 224 x 224 of type torch.float32

            elif "pts" == data_feature:
                pt_path = os.path.join(self.data_path, "data", str(shape_id), "point_clouds_normalized")
                pts = []
                path_manager = PathManager()
                for part_id in part_ids:
                    part_pathname = os.path.join(pt_path, f"{part_id}.ply")
                    data = _load_ply(part_pathname, path_manager=path_manager)
                    pts.append(data.verts)
                ret["pts"] = torch.stack(pts)
                # N x 1000 x 3 of type torch.float32

            elif "masks" == data_feature:
                mask_path = os.path.join(self.data_path, "data", str(shape_id), "masks")
                mask_imgs = []
                for view_id, step_id in zip(view_ids, step_ids):
                    masks = []
                    for part_id in part_ids:
                        mask_pathname = os.path.join(mask_path, f"{view_id:03d}_{step_id:03d}_{part_id}.png")
                        mask = Image.open(mask_pathname).convert("L")
                        # convert to tensor
                        mask = T.ToTensor()(mask).bool()
                        masks.append(mask[0])
                    mask_imgs.append(torch.stack(masks))
                ret["masks"] = torch.stack(mask_imgs)
                # S x N x 224 x 224 of type torch.bool, False -> background, True -> part

            elif "meshes" == data_feature:
                mesh_path = os.path.join(self.data_path, "data", str(shape_id), "objs_normalized")
                mesh_pathnames = []
                for part_id in part_ids:
                    mesh_pathnames.append(os.path.join(mesh_path, f"{part_id}.obj"))
                ret["meshes"] = load_objs_as_meshes(mesh_pathnames, load_textures=False)

            elif "similar_parts_cnt" == data_feature:
                similar_parts_cnt = []
                for part_id in part_ids:
                    part_info = meta[part_id]
                    similar_parts_cnt.append([len(part_info["similar_parts"])])
                ret["similar_parts_cnt"] = torch.tensor(similar_parts_cnt)
                # N x 1 of type torch.int64

            elif "similar_parts_edge_indices" == data_feature:
                ret["similar_parts_edge_indices"] = torch.tensor(equiv_edge_indices).long()
                # M x 2 of type torch.int64

            elif "bbox_size" == data_feature:
                bbox_size = []
                for part_id in part_ids:
                    part_info = meta[part_id]
                    bbox_size.append(part_info["bbox_size"])
                ret["bbox_size"] = torch.tensor(bbox_size)
                # N x 3 of type torch.float32

            elif "ins_one_hot" == data_feature:
                # number of parts x number of similar parts
                ins_one_hot = torch.zeros(len(part_ids), self.max_parts_cnt)
                for part_id in part_ids:
                    count = 0
                    part_info = meta[part_id]
                    similar = part_info["similar_parts"]
                    for s in similar:
                        idx = part_ids.index(s)
                        ins_one_hot[idx, count] = 1
                        ins_one_hot[idx, len(similar)] = -1
                        count += 1
                ret["ins_one_hot"] = ins_one_hot
                # N x 20 of type torch.float32

            elif "order" == data_feature:
                order = []
                if self.order_pathname is not None:
                    order = pred_orders[str(shape_id)]
                else:
                    raw_order_pathname = os.path.join(self.data_path, "data", str(shape_id), "order.json")
                    with open(raw_order_pathname) as f:
                        raw_order = json.load(f)
                    for part_id in part_ids:
                        order.append(raw_order.index(part_id))
                order = torch.tensor(order, dtype=torch.int64)
                if self.order_transform is not None:
                    order = self.order_transform(order)
                ret["order"] = order
                # N of type torch.int64

            elif "group" == data_feature:
                group_map = {}
                group_id = 0
                for part_id in part_ids:
                    if part_id in group_map:
                        continue
                    similar_parts = meta[part_id]["similar_parts"]
                    group_map[part_id] = group_id
                    for similar_part in similar_parts:
                        if similar_part not in group_map:
                            group_map[similar_part] = group_id
                    group_id += 1
                group = [group_map[part_id] for part_id in part_ids]
                ret["group"] = torch.tensor(group, dtype=torch.int64)
                # N of type torch.int64

            elif "parts_cam_dof" == data_feature:
                parts_cam_dof = []
                for view_id, step_id in zip(view_ids, step_ids):
                    cam_dofs = []
                    for part_id in part_ids:
                        if part_id not in gt_poses[str(view_id)][str(step_id)]:
                            # padding with zeros
                            cam_dofs.append(
                                torch.tensor(
                                    [
                                        0,
                                        0,
                                        0,
                                        1,
                                        0,
                                        0,
                                        0,
                                    ]
                                )
                            )
                            continue
                        cam_dofs.append(torch.tensor(gt_poses[str(view_id)][str(step_id)][part_id]))
                    parts_cam_dof.append(torch.stack(cam_dofs).float())

                ret["parts_cam_dof"] = torch.stack(parts_cam_dof)
                # S x N x 7 of type torch.float32 [tx, ty, tz, qw, qx, qy, qz], padding with zeros

            elif "part_to_shape_matrix" == data_feature:
                matrices = []
                for part_id in part_ids:
                    part_info = meta[part_id]
                    matrices.append(torch.tensor(part_info["part_to_shape_matrix"]))
                ret["part_to_shape_matrix"] = torch.stack(matrices)
                # N x 4 x 4 of type torch.float32

            elif "template_features" == data_feature:
                template_features_path = os.path.join(
                    self.data_path, "data", str(shape_id), self.template_features_folder
                )
                template_features = []
                for part_id in part_ids:
                    template_feature_pathname = os.path.join(template_features_path, f"{part_id}.pt")  # 42 x 1024
                    template_feature = torch.load(template_feature_pathname)
                    template_features.append(template_feature)
                ret["template_features"] = torch.stack(template_features)
                # N x 42 x 1024 of type torch.float32

            elif "depth" == data_feature:
                depth_path = os.path.join(self.data_path, "data", str(shape_id), self.depth_folder)
                depth_imgs = []
                for view_id, step_id in zip(view_ids, step_ids):
                    depth_img_pathname = os.path.join(depth_path, f"{view_id:03d}_{step_id:03d}.png")
                    depth_img = Image.open(depth_img_pathname).convert("L")
                    depth_imgs.append(F.to_tensor(depth_img))
                ret["depth"] = torch.stack(depth_imgs)
                # S x 224 x 224 of type torch.float32

            else:
                raise ValueError(f"Invalid data_feature: {data_feature}")

        return ret


class IMABatchSampler(Sampler):
    """Pytorch Sampler for ikea Manual Assembly dataset.

    We sample a batch that has roughly the same number of parts, instead of the same number of shapes.
    """

    def __init__(
        self,
        shape_ids: list[int],
        shape_id_to_part_cnt_map: dict[int, int],
        parts_per_batch: int = 64,
        shuffle: bool = True,
    ):
        shape_ids_indices = list(range(len(shape_ids)))
        shuffled_shape_ids = shape_ids
        if shuffle:
            random.shuffle(shape_ids_indices)
            shuffled_shape_ids = [shape_ids[i] for i in shape_ids_indices]

        self.batches = []
        i = 0
        while i < len(shuffled_shape_ids):
            batch = []
            batch_part_cnt = 0
            for shape_id in shuffled_shape_ids[i:]:
                part_cnt = shape_id_to_part_cnt_map[shape_id]
                if part_cnt + batch_part_cnt <= parts_per_batch:
                    batch.append(shape_ids_indices[i])
                    batch_part_cnt += part_cnt
                    i += 1
                else:
                    break
            self.batches.append(batch)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


class IMADataModule(pl.LightningDataModule):
    """Pytorch Lightning DataModule for ikea Manual Assembly dataset."""

    def __init__(
        self,
        dataset_path: str = "data/ikea-manual-assembly",
        train_source: Literal["partnet", "ikea-manual"] = "partnet",
        val_source: Literal["partnet", "ikea-manual"] = "partnet",
        test_source: Literal["partnet", "ikea-manual"] = "partnet",
        train_type: Literal["synthetic", "synthetic-stepwise", "real"] = "synthetic",
        val_type: Literal["synthetic", "synthetic-stepwise", "real"] = "synthetic",
        test_type: Literal["synthetic", "synthetic-stepwise", "real"] = "synthetic",
        train_split_pathname: str = "splits/image_pa/all.train.txt",
        val_split_pathname: str = "splits/image_pa/all.val.txt",
        test_split_pathname: str = "splits/image_pa/all.test.txt",
        data_features: list[
            Literal[
                "img",
                "imgs",
                "pts",
                "masks",
                "meshes",
                "similar_parts_cnt",
                "similar_parts_edge_indices",
                "bbox_size",
                "ins_one_hot",
                "order",
                "group",
                "parts_cam_dof",
                "part_to_shape_matrix",
                "template_features",
                "depth",
            ]
        ] = [
            "img",
            "pts",
            "masks",
            "similar_parts_cnt",
            "bbox_size",
            "ins_one_hot",
        ],
        normalize_img: bool = True,
        augment_order: bool = False,
        augment_order_alpha: float = 0.3,
        template_features_folder: str = "template_features/dinov2_1024",
        depth_folder: str = "depths/depth_anything_v2_224",
        train_num_views: int = 24,
        step_mode: Literal["lastonly", "all"] = "all",
        order_pathname: str | None = None,
        # Sampler
        max_parts_per_batch: int = 64,
        shuffle: bool = True,
        # Data Loader
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["_class_path"]
        )  # HACK: https://github.com/Lightning-AI/pytorch-lightning/issues/20182

    def setup(self, stage: Literal["fit", "validate", "test"] | None = None):
        transforms = [T.ToTensor()]
        if self.hparams.normalize_img:
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        image_transform = T.Compose(transforms)
        if stage in ["fit"] or stage is None:
            self.train_dataset = IMADataset(
                dataset_path=self.hparams.dataset_path,
                source=self.hparams.train_source,
                type=self.hparams.train_type,
                split_pathname=self.hparams.train_split_pathname,
                data_features=self.hparams.data_features,
                image_transform=image_transform,
                order_transform=OrderGumbelSinkhornRandPermutation(self.hparams.augment_order_alpha)
                if self.hparams.augment_order
                else None,
                template_features_folder=self.hparams.template_features_folder,
                depth_folder=self.hparams.depth_folder,
                num_views=self.hparams.train_num_views,
                step_mode=self.hparams.step_mode,
                order_pathname=self.hparams.order_pathname,
                load_shape_id_to_part_cnt_map=True,
            )
        if stage in ["fit", "validate"] or stage is None:
            self.val_dataset = IMADataset(
                dataset_path=self.hparams.dataset_path,
                source=self.hparams.val_source,
                type=self.hparams.val_type,
                split_pathname=self.hparams.val_split_pathname,
                data_features=self.hparams.data_features,
                image_transform=image_transform,
                template_features_folder=self.hparams.template_features_folder,
                depth_folder=self.hparams.depth_folder,
                step_mode=self.hparams.step_mode,
                order_pathname=self.hparams.order_pathname,
                num_views=1,
            )
        if stage in ["test"] or stage is None:
            self.test_dataset = IMADataset(
                dataset_path=self.hparams.dataset_path,
                source=self.hparams.test_source,
                type=self.hparams.test_type,
                split_pathname=self.hparams.test_split_pathname,
                data_features=self.hparams.data_features,
                image_transform=image_transform,
                # order_transform=OrderGumbelSinkhornRandPermutation(self.hparams.augment_order_alpha)
                # if self.hparams.augment_order
                # else None,  # HACK for testing only
                template_features_folder=self.hparams.template_features_folder,
                depth_folder=self.hparams.depth_folder,
                step_mode=self.hparams.step_mode,
                order_pathname=self.hparams.order_pathname,
                num_views=1,
            )

    def collate_fn(self, batch):
        ret = {}
        for key in batch[0].keys():
            ret[key] = [data[key] for data in batch]
        return ret

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=IMABatchSampler(
                shape_ids=self.train_dataset.shape_ids,
                shape_id_to_part_cnt_map=self.train_dataset.shape_id_to_part_cnt_map,
                parts_per_batch=self.hparams.max_parts_per_batch,
                shuffle=self.hparams.shuffle,
            ),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn=self.collate_fn,
        )
