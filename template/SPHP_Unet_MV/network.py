'''
Unet using traditional convolution
'''
import sys
import torch
from torch import nn
sys.path.append('../..')
import Utils


def Block(ch1, ch2):
    m = nn.Sequential(
        nn.Conv2d(ch1, ch2, 3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.LeakyReLU(),
        nn.Conv2d(ch2, ch2, 3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.LeakyReLU()
    )
    return m


class MyModel(Utils.BaseModule):
    '''
    define model
    '''
    def __init__(self, save_path):
        super().__init__(save_path)
        in_ch = 2
        plane = [32, 64, 128, 256]
        self.downConv1 = Block(in_ch, plane[0])
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.downConv2 = Block(plane[0], plane[1])
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.downConv3 = Block(plane[1], plane[2])
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.downConv4 = Block(plane[2], plane[3])
        self.deconv3 = nn.ConvTranspose2d(plane[3], plane[2], 2, stride=2, bias=False)
        self.upConv3 = Block(plane[3], plane[2])
        self.deconv2 = nn.ConvTranspose2d(plane[2], plane[1], 2, stride=2, bias=False)
        self.upConv2 = Block(plane[2], plane[1])
        self.deconv1 = nn.ConvTranspose2d(plane[1], plane[0], 2, stride=2, bias=False)
        self.upConv1 = Block(plane[1], plane[0])
        self.output_layer = nn.Conv2d(plane[0], 13, 1)

    def forward(self, x):

        x1 = self.downConv1(x)

        x2 = self.maxpool1(x1)
        x2 = self.downConv2(x2)

        x3 = self.maxpool2(x2)
        x3 = self.downConv3(x3)

        x4 = self.maxpool3(x3)
        x4 = self.downConv4(x4)

        dx3 = self.deconv3(x4)
        ux3 = torch.cat((dx3, x3), 1)
        ux3 = self.upConv3(ux3)

        dx2 = self.deconv2(ux3)
        ux2 = torch.cat((dx2, x2), 1)
        ux2 = self.upConv2(ux2)

        dx1 = self.deconv1(ux2)
        ux1 = torch.cat((dx1, x1), 1)
        ux1 = self.upConv1(ux1)

        output = self.output_layer(ux1)

        return output
