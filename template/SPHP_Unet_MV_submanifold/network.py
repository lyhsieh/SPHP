'''
Unet using sparse convolution
'''
import torch
import torch.nn as nn
import Utils
import sparseconvnet as scn
import numpy as np


use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
device = 'cuda' if use_cuda else 'cpu'


def weights_init(m):
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def Conv(in_ch, out_ch):
    net = scn.Sequential(
        scn.SubmanifoldConvolution(dimension=2,
                                   nIn=in_ch,
                                   nOut=out_ch,
                                   filter_size=3,
                                   bias=False),
        scn.ReLU()
    )
    return net


def DownConv(in_ch, out_ch):
    net = scn.Sequential(
        scn.SubmanifoldConvolution(dimension=2,
                                   nIn=in_ch,
                                   nOut=out_ch,
                                   filter_size=3,
                                   bias=False),
        scn.ReLU(),
        scn.MaxPooling(dimension=2, pool_size=2, pool_stride=2)
    )
    return net


def UpConv(in_ch, out_ch):
    net = scn.Sequential(
        scn.TransposeConvolution(dimension=2,
                                 nIn=in_ch,
                                 nOut=out_ch,
                                 filter_size=3,
                                 filter_stride=2,
                                 bias=False),
        scn.ReLU()
    )
    return net


def Block(ch1, ch2):
    m = scn.Sequential(
        scn.SubmanifoldConvolution(dimension=2,
                                   nIn=ch1,
                                   nOut=ch2,
                                   filter_size=3,
                                   bias=False),
        scn.BatchNormLeakyReLU(ch2),
        scn.SubmanifoldConvolution(dimension=2,
                                   nIn=ch2,
                                   nOut=ch2,
                                   filter_size=3,
                                   bias=False),
        scn.BatchNormLeakyReLU(ch2),
    )
    return m


def ConcatAndUpConvolution(plane):
    m = scn.Sequential()
    m.add(scn.MaxPooling(dimension=2, pool_size=2, pool_stride=2))
    m.add(Block(plane[0], plane[1]))

    if len(plane) > 2:
        m.add(
            scn.ConcatTable().add(
                scn.Identity()).add(
                ConcatAndUpConvolution(plane[1:])
            )
        )
        m.add(scn.JoinTable())
        m.add(Block(plane[2], plane[1]))

    m.add(
        scn.Deconvolution(2, plane[1], plane[0], 2, 2, False)
    )

    return m


def Unet(in_ch, plane):
    m = scn.Sequential()
    m.add(Block(in_ch, plane[0]))
    m.add(
        scn.ConcatTable().add(
            scn.Identity()).add(
            ConcatAndUpConvolution(plane)
        )
    )
    m.add((scn.JoinTable()))
    m.add(Block(plane[1], plane[0]))
    m.add(scn.SubmanifoldConvolution(dimension=2,
                                     nIn=plane[0],
                                     nOut=13,
                                     filter_size=1,
                                     bias=False))
    return m


class MyModel(Utils.BaseModule):
    '''
    define model
    '''
    def __init__(self, save_path):
        super().__init__(save_path)
        self.U = scn.Sequential(
            Unet(2, [32, 64, 128, 256]),
            scn.SparseToDense(2, 13)
        )

    def forward(self, x):
        sparseModel = self.U
        inputSpatialSize = sparseModel.input_spatial_size(torch.LongTensor([288, 384]))
        x = x.cpu()
        position = np.where(x.sum(axis=1) != 0)
        locations = torch.LongTensor(list(zip(position[1], position[2], position[0])))
        features = torch.FloatTensor(x[position[0], :, position[1], position[2]])
        input_layer = scn.InputLayer(2, inputSpatialSize.tolist())
        input = input_layer([locations, features.to(device)])
        out = sparseModel(input.to(device))

        return out
