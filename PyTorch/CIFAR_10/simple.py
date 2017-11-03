import torch.nn as nn
import torch
import torch.nn.functional as F
from util import TBConv2d


class Net(nn.Module):
    def __init__(self, threshold=0, scale=False, clamp=False):
        super(Net, self).__init__()
        self.xnor = nn.Sequential(  # 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=False), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 32 x 16 x 16

            TBConv2d(32, 32, kernel_size=5, stride=1, padding=2, threshold=threshold, scale=scale, clamp=clamp,
                is_relu=False), nn.Sigmoid(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 32 x 8 x 8

            TBConv2d(32, 64, kernel_size=5, stride=1, padding=2, threshold=threshold, scale=scale, clamp=clamp,
                is_relu=False), nn.Sigmoid(), nn.AvgPool2d(kernel_size=3, stride=2, padding=1),  # 64 x 4 x 4

            nn.Conv2d(64, 10, kernel_size=4, stride=1, padding=0), )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x
