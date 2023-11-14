from torch import nn
import Utils


def weights_init(m):
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

def Conv(in_ch, out_ch, dilation):
    net = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation,
                  stride=1,  dilation=dilation, bias=False),
        nn.ReLU()
    )
    return net

def DownConv(in_ch, out_ch, dilation):
    net = nn.Sequential(
        nn.Conv2d(in_ch,out_ch, kernel_size=3, padding=1, stride=1,
                  dilation=dilation, bias=False),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    return net

def UpConv(in_ch, out_ch, dilation):
    net = nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1,
                           output_padding=1, dilation=dilation, bias=False),
        nn.ReLU()
    )
    return net

class MyModel(Utils.BaseModule):
    '''
    define network
    '''
    def __init__(self, save_path):
        super().__init__(save_path)

        self.layer_1_3 = nn.Sequential(
            DownConv(3, 16, dilation=1),
            Conv(16, 32, dilation=1),
            Conv(32, 32, dilation=1),
        )
        self.layer_4_8 = nn.Sequential(
            DownConv(32, 32, dilation=2),
            Conv(32, 64, dilation=2),
            Conv(64, 64, dilation=2),
            Conv(64, 64, dilation=2),
            Conv(64, 64, dilation=2),
        )
        self.layer_9_13 = nn.Sequential(
            UpConv(64, 32, dilation=2),
            Conv(32, 32, dilation=2),
            Conv(32, 32, dilation=2),
            Conv(32, 32, dilation=2),
            Conv(32, 32, dilation=2)
        )
        self.layer_14_17 = nn.Sequential(
            UpConv(32, 16, dilation=1),
            Conv(16, 16, dilation=1),
            Conv(16, 16, dilation=1),
            nn.Conv2d(16, 13, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer_1_3(x)
        x = self.layer_4_8(x)
        x = self.layer_9_13(x)
        x = self.layer_14_17(x)
        return x
