from __future__ import print_function
import torch
import torch.nn as nn
from util import Conv2dTB


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = Conv2dTB(28 * 28, 4096)
        self.fc2 = Conv2dTB(4096, 4096)
        self.fc3 = Conv2dTB(4096, 4096)
        self.fc4 = Conv2dTB(4096, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28, 1, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(-1, 10)
        return x
