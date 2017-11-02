from __future__ import print_function
import torch
import torch.nn as nn
from util import TBConv2d


class LeNet_5(nn.Module):
    def __init__(self, threshold=0, scale=False, clamp=False):
        super(LeNet_5, self).__init__()
        # 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        # 20 x 24 x 24
        self.bn_conv1 = nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 20 x 12 x 12
        self.bin_conv2 = TBConv2d(20, 50, kernel_size=5, stride=1, padding=0, threshold=threshold, scale=scale, clamp=clamp)
        # 50 x 8 x 8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 50 x 4 x 4
        self.bin_ip1 = TBConv2d(50, 500, kernel_size=4, stride=1, padding=0, threshold=threshold, scale=scale, clamp=clamp)
        # 500 x 1 x 1
        self.ip2 = nn.Linear(500, 10)
        # 10

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        return

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.bin_conv2(x)
        x = self.pool2(x)
        x = self.bin_ip1(x)
        x = x.view(x.size(0), 500)
        x = self.ip2(x)
        return x
