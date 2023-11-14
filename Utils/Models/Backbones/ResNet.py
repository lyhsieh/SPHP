import sys
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.models as models
import functools
import math


class ResNet(nn.Module):
    def __init__(self, layers, in_channels=3, pretrained=True):
        assert layers in [18, 34, 50, 101, 152]
        super().__init__()
        #pretrained_model = models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        pretrained_model = models.__dict__['resnet{}'.format(layers)](weights='ResNet%d_Weights.DEFAULT'%layers)
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = {
            'l0': x0,
            'l1': x1,
            'l2': x2,
            'l3': x3,
            'l4': x4
        }

        return out
    
    def preforward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        return x
