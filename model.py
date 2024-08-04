import torch
import torch.nn as nn


class Inception3D(nn.Module):
    def __init__(self, in_channels=3, unit=2):
        super(Inception3D, self).__init__()

        self.unit = unit
        self.out_channels = 0

        # frames = (frames' - kernel + 2 * padding)/stride + 1
        # height = (height' - kernel + 2 * padding)/stride + 1
        # width = (width' - kernel + 2 * padding)/stride + 1
        # params = kernel_height * kernel_width * kernel_depth * in_channels * out_channels + biases(out_channels)

        # 1x1x1 conv output:
        # frames = (frames' - 1 + 2 * 0)/1 + 1 = frames'
        # height = (height' - 1 + 2 * 0)/1 + 1 = height'
        # width = (width' - 1 + 2 * 0)/1 + 1 = width'
        # params = 32
        self.branch1x1 = nn.Conv3d(in_channels, self.unit * 4, kernel_size=1)
        self.out_channels += self.unit * 4

        # 1x1x1 conv -> 3x3x3 conv output:
        # frames = (frames' - 3 + 2 * 1)/1 + 1 = frames'
        # height = (height' - 3 + 2 * 1)/1 + 1 = height'
        # width = (width' - 3 + 2 * 1)/1 + 1 = width'
        # params = 48
        # params = 5200
        self.branch3x3 = nn.Sequential(
            nn.Conv3d(in_channels, self.unit * 6, kernel_size=1),
            nn.Conv3d(self.unit * 6, self.unit * 8, kernel_size=3, padding=1)
        )
        self.out_channels += self.unit * 8

        # 1x1x1 conv -> 5x5x5 conv output:
        # frames = (frames' - 5 + 2 * 2)/1 + 1 = frames'
        # height = (height' - 5 + 2 * 2)/1 + 1 = height'
        # width = (width' - 5 + 2 * 2)/1 + 1 = width'
        # params = 8
        # params = 1004
        self.branch5x5 = nn.Sequential(
            nn.Conv3d(in_channels, self.unit, kernel_size=1),
            nn.Conv3d(self.unit, self.unit * 2, kernel_size=5, padding=2)
        )
        self.out_channels += self.unit * 2

        # 3x3x3 max pooling -> 1x1x1 conv output:
        # frames = (frames' - 3 + 2 * 1)/1 + 1 = frames'
        # height = (height' - 3 + 2 * 1)/1 + 1 = height'
        # width = (width' - 3 + 2 * 1)/1 + 1 = width'
        # params = 0
        # params = 16
        self.branch_pool = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, self.unit * 2, kernel_size=1)
        )
        self.out_channels += self.unit * 2

        # params = 33
        self.merge = nn.Conv3d(in_channels=self.out_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        _outputs = [branch1x1, branch3x3, branch5x5, branch_pool]

        x = torch.cat(_outputs, dim=1)
        x = self.merge(x)

        return x


class TemporalAttention(nn.Module):
    def __init__(self, out_frames):
        super(TemporalAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(out_frames, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)

        x = torch.sigmoid(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_frames):
        super(SpatialAttention, self).__init__()

        self.pool = nn.AvgPool3d(kernel_size=(in_frames, 1, 1))

    def forward(self, x):
        attention_map = self.pool(x)

        attention_map = torch.sigmoid(attention_map)

        return attention_map


class MLP(nn.Module):
    def __init__(self, frames, height, width):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(frames * height * width, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


class Model(nn.Module):
    def __init__(self, frames, height, width, unit, in_channels=3):
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.frames = frames
        self.height = height
        self.width = width
        self.unit = unit

        self.inception_3d_model = Inception3D(in_channels=self.in_channels, unit=self.unit)
        self.temporal_attention_model = TemporalAttention(self.frames)
        self.spatial_attention_model = SpatialAttention(self.frames)
        self.mlp_model = MLP(self.frames, self.height, self.width)

    def forward(self, x):
        f = self.inception_3d_model(x)
        ta = self.temporal_attention_model(f)

        broadcast_ta = ta.expand(-1, -1, -1, f.size(3), f.size(4))
        ft = f * broadcast_ta

        sa = self.spatial_attention_model(ft)

        broadcast_sa = sa.expand(-1, -1, ft.size(2), -1, -1)
        fts = ft * broadcast_sa

        o = self.mlp_model(fts)

        return o
