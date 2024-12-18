# A lite version of PointNet, without STN
# Adapted from https://github.com/AntheaLi/3DPartAssembly/blob/9c93b659ac4bdb3807069e4911544b367115e091/exps/exp_assemble/models/model.py#L166

import torch
import torch.nn as nn


class PointNetLiteFeat(nn.Module):
    def __init__(self, out_features: int = 1024):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, out_features, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(out_features)

        self.mlp1 = nn.Linear(out_features, out_features)
        self.bn6 = nn.BatchNorm1d(out_features)

        self.out_features = out_features

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]

        x = torch.relu(self.bn6(self.mlp1(x)))
        return (x.unsqueeze(1),)


if __name__ == "__main__":
    batch_size = 32
    num_points = 1024
    model = PointNetLiteFeat()
    x = torch.rand(batch_size, num_points, 3)
    y = model(x)
    print(y.shape)
