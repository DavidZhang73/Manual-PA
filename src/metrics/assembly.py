import torch
from pytorch3d.loss import chamfer_distance
from torchmetrics import Metric


class ChamferDistance(Metric):
    higher_is_better = False

    def __init__(self, scale_factor: float = 1e3):
        """Chamfer distance metric.

        Args:
            scale_factor (float, optional): Scaling factor for simplicity. Defaults to 1e3.
        """
        super().__init__()
        self.add_state("cd", default=torch.zeros(1), dist_reduce_fx="mean")
        self.add_state("count", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum")
        self.scale_factor = scale_factor

    def update(self, pred_pcs: torch.Tensor, target_pcs: torch.Tensor):
        """Update the metric with the given point clouds.

        Args:
            pred_pcs (torch.Tensor): N x P x 3 point cloud
            target_pcs (torch.Tensor): N x P x 3 point cloud
        """
        pred_pcs = pred_pcs.view(1, -1, 3)
        target_pcs = target_pcs.view(1, -1, 3)
        cd = chamfer_distance(pred_pcs, target_pcs)
        self.count += 1
        self.cd += cd[0]

    def compute(self):
        return self.cd / self.count * self.scale_factor


class PartAccuracy(Metric):
    higher_is_better = True

    def __init__(self, threshold: float = 0.01):
        """Part accuracy metric.

        A part is considered accurate if the chamfer distance between the predicted and target point clouds is less than
        the threshold.

        Args:
            threshold (float, optional): Epsilon threshold for part accuracy. Defaults to 0.01.
        """
        super().__init__()
        self.add_state("pa", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, pred_pcs: torch.Tensor, target_pcs: torch.Tensor):
        """Update the metric with the given point cloud for each part.

        Args:
            pred_pcs (torch.Tensor): N x P x 3 point clouds
            target_pcs (torch.Tensor): N x P x 3 point clouds
        """
        pa = 0
        for pred_pc, target_pc in zip(pred_pcs, target_pcs):
            cd = chamfer_distance(pred_pc.unsqueeze(0), target_pc.unsqueeze(0))[0]
            if cd <= self.threshold:
                pa += 1
        self.count += 1
        self.pa += pa / pred_pcs.shape[0]

    def compute(self):
        return self.pa / self.count


class ConnectivityAccuracy(Metric):
    higher_is_better = True

    def __init__(self, threshold: float = 0.01):
        """Connectivity accuracy metric.

        A connection is considered accurate if

        Args:
            threshold (float, optional): Epsilon threshold for connectivity accuracy. Defaults to 0.01.
        """
        super().__init__()
        self.add_state("ca", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, pred_pcs: torch.Tensor, target_pcs: torch.Tensor):
        """Update the metric with the given point cloud for each part.

        Args:
            pred_pcs (torch.Tensor): N x P x 3 point clouds
            target_pcs (torch.Tensor): N x P x 3 point clouds
        """
        # TODO: Implement connectivity accuracy
        raise NotImplementedError

    def compute(self):
        return self.ca / self.count


class SuccessRate(Metric):
    higher_is_better = True

    def __init__(self, threshold: float = 0.01):
        """Success rate metric.

        A shape is considered successful if the part accuracy for all parts is greater than the threshold.

        Args:
            threshold (float, optional): Epsilon threshold for success rate. Defaults to 0.01.
        """
        super().__init__()
        self.add_state("sr", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, pred_pcs: torch.Tensor, target_pcs: torch.Tensor):
        """Update the metric with the given point cloud for each part.

        Args:
            pred_pcs (torch.Tensor): N x P x 3 point clouds
            target_pcs (torch.Tensor): N x P x 3 point clouds
        """
        self.count += 1
        for pred_pc, target_pc in zip(pred_pcs, target_pcs):
            cd = chamfer_distance(pred_pc.unsqueeze(0), target_pc.unsqueeze(0))[0]
            if cd > self.threshold:
                return
        self.sr += 1

    def compute(self):
        return self.sr / self.count
