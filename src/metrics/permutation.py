import torch
from torchmetrics import Metric
from torchmetrics.functional import kendall_rank_corrcoef


class KendallTau(Metric):
    higher_is_better = True

    def __init__(self):
        """Kendall's tau metric."""
        super().__init__()
        self.add_state("kendall_tau", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update the metric with the given predictions and targets.

        Args:
            preds (torch.Tensor): Predictions of shape (N).
            target (torch.Tensor): Targets of shape (N).
        """
        kt = kendall_rank_corrcoef(preds, target)
        self.kendall_tau += kt
        self.count += 1

    def compute(self):
        return self.kendall_tau / self.count


if __name__ == "__main__":
    m = KendallTau()
    m.update(torch.tensor([1, 2, 3]), torch.tensor([3, 2, 1]))
    m.update(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    # m.update(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    print(m.compute())
