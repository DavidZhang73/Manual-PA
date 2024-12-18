import torch
from torchmetrics import Metric
from torchmetrics.functional.segmentation import mean_iou


class MIoU(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("score", default=torch.zeros(1), dist_reduce_fx="mean")
        self.add_state("count", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Computes the mean IoU score.

        Args:
            preds (torch.Tensor): An one-hot encoded tensor of shape (C, H, W), where C is the number of classes
                including the background as the first class.
            target (torch.Tensor): A one-hot encoded tensor of shape (C, H, W).
        """
        score = mean_iou(
            preds.unsqueeze(0),
            target.unsqueeze(0),
            num_classes=target.shape[0],
            include_background=False,
            per_class=False,
        ).mean()
        self.score += score
        self.count += 1

    def compute(self):
        return self.score / self.count


class AP(Metric):
    def __init__(self, iou_thresholds: list[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
        super().__init__()
        self.iou_thresholds = iou_thresholds
        self.add_state("score", default=torch.zeros(1), dist_reduce_fx="mean")
        self.add_state("count", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Computes the average precision score.

        Args:
            preds (torch.Tensor): A one-hot encoded tensor of shape (C, H, W), where C is the number of classes
                including the background as the first class.
            target (torch.Tensor): A one-hot encoded tensor of shape (C, H, W).
        """
        for iou_threshold in self.iou_thresholds:
            tp, fp, fn = 0, 0, 0
            for c in range(1, preds.shape[0]):  # Loop over each class, ignore the background class
                pred_class = preds[c]
                target_class = target[c]

                # Compute intersection and union
                intersection = (pred_class * target_class).sum().item()
                union = pred_class.sum().item() + target_class.sum().item() - intersection

                # Compute IoU
                iou = intersection / union if union > 0 else 0

                if iou >= iou_threshold:
                    tp += 1  # True positive
                else:
                    if pred_class.sum().item() > 0:
                        fp += 1  # False positive
                    if target_class.sum().item() > 0:
                        fn += 1  # False negative
            self.score += tp / (tp + fp) if (tp + fp) > 0 else 0
            self.count += 1

    def compute(self):
        return self.score / self.count
