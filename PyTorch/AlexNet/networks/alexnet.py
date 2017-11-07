import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from util import ConvBlock

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, threshold=0, scale=False, clamp=False):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBlock(96, 256, kernel_size=5, stride=1, padding=2, groups=1, threshold=threshold, scale=scale,
                clamp=clamp),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBlock(256, 384, kernel_size=3, stride=1, padding=1, threshold=threshold, scale=scale, clamp=clamp),
            ConvBlock(384, 384, kernel_size=3, stride=1, padding=1, groups=1, threshold=threshold, scale=scale,
                clamp=clamp),
            ConvBlock(384, 256, kernel_size=3, stride=1, padding=1, groups=1, threshold=threshold, scale=scale,
                clamp=clamp), nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBlock(256, 4096, kernel_size=6, threshold=threshold, scale=scale, clamp=clamp),
            nn.Dropout(),
            ConvBlock(4096, 4096, kernel_size=1, threshold=threshold, scale=scale, clamp=clamp),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
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
