import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=(3,3), stride = 1, padding = 1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,kernel_size=(3,3), stride = 1, padding = 1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.convblock1(x)
        return x

class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet,self).__init__()

        self.prep_layer = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=True),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=True),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.resblock1 = ResBlock(128, 128)

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,bias=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,bias=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.resblock2 = ResBlock(512, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.prep_layer(x)
        out = self.layer1(out)
        res1 = self.resblock1(out)
        out = out + res1
        out = self.layer2(out)
        out = self.layer3(out)
        res2 = self.resblock2(out)
        out = out + res2
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return F.log_softmax(out,dim = -1)