import torch.nn as nn
import torch.nn.functional as F


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):
    def __init__(self):

        super(Net,self).__init__()

        # Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3), padding = 1, bias = False), # 32>32 | 3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3), padding = 1, bias = False), # 32>32 | 5
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3), padding = 1, bias = False), # 32>32 | 7
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding = 2, stride = 2, dilation = 2, bias = False), # 32>16 | 11
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding = 1, bias = False), # 16>16 | 15
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3), padding = 1, bias = False), # 16>16 | 19
            nn.BatchNorm2d(32),
            nn.ReLU(),
            depthwise_separable_conv(32,32), # 16>16 | 23
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding = 2, stride = 2, dilation = 2, bias = False), # 16>8 | 31
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding = 1, bias = False), # 8>8 | 39
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3), padding = 1, bias = False), # 8>8 | 47
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3), padding = 1, bias = False), # 8>8 | 55
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),padding = 0, bias = False), # 8>8 | 63
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(3,3),padding = 0, bias = False), # 8>8 | 71
            nn.BatchNorm2d(16)
            # nn.ReLU()
        )

        # Global average pooling layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding = 0, bias = False),
        )

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)

        x = self.gap(x)
        x = x.view(-1,10)

        return F.log_softmax(x,dim = -1)