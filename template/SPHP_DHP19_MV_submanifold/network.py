'''
Sparse DHP19 network
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        scn.SubmanifoldConvolution(dimension=2, nIn=in_ch, nOut=out_ch, filter_size=3, bias=False),
        scn.ReLU()
    )

    return net

def DownConv(in_ch, out_ch):
    net = scn.Sequential(
        scn.SubmanifoldConvolution(dimension=2, nIn=in_ch, nOut=out_ch, filter_size=3, bias=False),
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

class MyModel(Utils.BaseModule):
    '''
    define network
    '''
    def __init__(self, save_path):
        super().__init__(save_path)
        self.dhp19 = scn.Sequential(
            DownConv(2, 16),
            Conv(16, 32),
            Conv(32, 32),
            DownConv(32, 32),
            Conv(32, 64),
            Conv(64, 64),
            Conv(64, 64),
            Conv(64, 64),
            UpConv(64, 32),
            Conv(32, 32),
            Conv(32, 32),
            Conv(32, 32),
            Conv(32, 32),
            UpConv(32, 16),
            Conv(16, 16),
            Conv(16, 16),
            scn.SubmanifoldConvolution(dimension=2, nIn=16, nOut=13, filter_size=3, bias=False),
            scn.ReLU(),
            scn.SparseToDense(2, 13)
        )


    def forward(self, x):
        sparseModel = self.dhp19
        # for input size (288, 384)
        inputSpatialSize = sparseModel.input_spatial_size(torch.LongTensor([291, 387]))
        x = x.cpu()

        position = np.where(x.sum(axis=1) != 0)
        locations = torch.LongTensor(list(zip(position[1], position[2], position[0])))
        features = torch.FloatTensor(x[position[0], :, position[1], position[2]])
        input_layer = scn.InputLayer(2, inputSpatialSize.tolist())
        input = input_layer([locations, features.to(device)])

        out = sparseModel(input.to(device))
        return out[..., :288, :384]
