import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEATURES_NUM, MOVE_PLANES_NUM, MOVE_LABELS_NUM


class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        # 畳み込み層の後ろがBatchNormalization層の場合はbiasは省略.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = F.relu(out + x)

        return out


class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks=10, channels=192, fcl=256):
        """

        Args:
            blocks (int, optional): num of resnet blocks. Defaults to 10.
            channels (int, optional): num of channels. Defaults to 192.
            fcl (int, optional): num of linear layer's node for value network.
                Defaults to 256.
        """
        super(PolicyValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=FEATURES_NUM, out_channels=channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        # policy head
        # 出力層に全結合ではなく畳み込みを使うのは同じ位置の重みを共有することで学習効率を高められるため.
        self.policy_conv = nn.Conv2d(in_channels=channels, out_channels=MOVE_PLANES_NUM,
                                     kernel_size=1, bias=False)
        self.policy_bias = Bias(MOVE_LABELS_NUM)  # 座標事にことなるバイアスを加える.

        # value head
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MOVE_PLANES_NUM,
                                     kernel_size=1, bias=False)
        self.value_bn1 = nn.BatchNorm2d(MOVE_PLANES_NUM)
        self.value_fc1 = nn.Linear(MOVE_LABELS_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.norm1(x))

        # resnet blocks
        x = self.blocks(x)

        # policy head
        policy = self.policy_conv(x)
        policy = self.policy_bias(torch.flatten(policy, 1))

        # value head
        value = F.relu(self.value_bn1(self.value_conv1(x)))
        value = F.relu(self.value_fc1(torch.flatten(value, 1)))
        value = self.value_fc2(value)

        return policy, value
