import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


class MultiscaleInitializer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.num_branches = len(num_channels)
        self.conv = nn.Sequential(nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, padding=1),
                                  nn.GroupNorm(4, num_channels[0]), nn.ReLU(),
                                  nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1))
        self.low_convs = nn.ModuleList([None])
        for branch_index in range(1, self.num_branches):
            self.low_convs.append(nn.Sequential(
                nn.Conv2d(num_channels[branch_index - 1], num_channels[branch_index], kernel_size=3, padding=1,
                          stride=2), nn.ReLU(),
                nn.Conv2d(num_channels[branch_index], num_channels[branch_index], kernel_size=1)))

    def forward(self, x):
        z1_init = [self.conv(x)]
        for branch_index in range(1, self.num_branches):
            z1_init.append(self.low_convs[branch_index](z1_init[-1]))
        return z1_init


class SequenceInitializer(nn.Module):
    def __init__(self, nfeat, d_embed):
        super().__init__()
        self.window_size = 3
        self.tcn = nn.Sequential(nn.Conv1d(nfeat, 200, kernel_size=self.window_size),
                                 nn.ReLU(),
                                 nn.Conv1d(200, d_embed, kernel_size=1))

    def forward(self, x):
        # Assume x has shape (bsz x C x seq_len)
        return self.tcn(F.pad(x, (self.window_size - 1, 0)))


class Embedding4Initializer(Module):
    def __init__(self, in_features, out_features):
        super(Embedding4Initializer, self).__init__()
        self.in_features = in_features
        self.m = out_features

        self.Net = nn.Sequential(
            nn.Linear(self.in_features, 32),
            nn.Linear(32, self.m),
            nn.ReLU(),
        )


    def forward(self, U):
        # U_B = torch.spmm(U, self.B)  # U*B
        U_B = self.Net(U)
        return U_B
