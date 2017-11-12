import torch.nn as nn
from util import Conv2dTB


class Net(nn.Module):
    alpha = 1

    def __init__(self, threshold=0.6, scale=False, clamp=False):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            oup = int(oup * self.alpha)
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            inp = int(inp * self.alpha)
            oup = int(oup * self.alpha)
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                # nn.BatchNorm2d(inp),
                # nn.ReLU(inplace=True),

                # nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                Conv2dTB(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * self.alpha), 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
