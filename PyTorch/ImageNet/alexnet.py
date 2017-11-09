import torch
import os
import torch.nn as nn
from util import ConvBlock

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, threshold=0.6, scale=False, clamp=False):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # 3 x 227 x 227
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            # 96 x 55 x 55
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 96 x 27 x 27
            ConvBlock(96, 256, kernel_size=5, stride=1, padding=2, groups=1, threshold=threshold, scale=scale,
                clamp=clamp),
            # 256 x 27 x 27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256 x 13 x 13
            ConvBlock(256, 384, kernel_size=3, stride=1, padding=1, threshold=threshold, scale=scale,
                clamp=clamp),
            # 384 x 13 x 13
            ConvBlock(384, 384, kernel_size=3, stride=1, padding=1, groups=1, threshold=threshold),
            # 384 x 13 x 13
            ConvBlock(384, 256, kernel_size=3, stride=1, padding=1, groups=1, threshold=threshold, scale=scale,
                clamp=clamp),
            # 256 x 13 x 13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256 x 6 x 6
            ConvBlock(256, 4096, kernel_size=6, stride=1, padding=0, threshold=threshold, scale=scale, clamp=clamp),
            # 4096 x 1 x 1
            nn.Dropout(),
            ConvBlock(4096, 4096, kernel_size=1, stride=1, padding=0, threshold=threshold, scale=scale, clamp=clamp),
            # 4096 x 1 x 1
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0),
            # 1000 x 1 x 1
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'ImageNet/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model
