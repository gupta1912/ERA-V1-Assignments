import torch
import torch.nn as nn


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool='max'):
        super(ContractingBlock, self).__init__()
        
        self.conv1 = self.conv_block(in_channels, out_channels)
        self.conv2 = self.conv_block(out_channels, out_channels)
        
        self.pool_layer = self.get_pooling_layer(pool, out_channels)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def get_pooling_layer(self, pool, out_channels):
        if pool == 'max':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            return nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        skip = x2  # store the output for the skip connection

        x3 = self.pool_layer(x2)
        
        return x3, skip

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='transpose', r=True):
        super(ExpandingBlock, self).__init__()

        self.conv1 = self.conv_block(in_channels, out_channels, r)
        self.conv2 = self.conv_block(out_channels, out_channels, r)
        
        self.upsample = self.get_upsampling_layer(in_channels, out_channels, mode)
        
    def conv_block(self, in_channels, out_channels, r=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        ]
        if r:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def get_upsampling_layer(self, in_channels, out_channels, mode):
        if mode == 'transpose':
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat((x, skip), dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, mode='transpose', relu=False, pool='max'):
        super(UNet, self).__init__()

        # Contracting Blocks
        self.contract1 = ContractingBlock(in_channels, 64, pool)
        self.contract2 = ContractingBlock(64, 128, pool)
        self.contract3 = ContractingBlock(128, 256, pool)
        self.contract4 = ContractingBlock(256, 512, pool)

        # Expanding Blocks
        self.expand1 = ExpandingBlock(512, 256, mode)
        self.expand2 = ExpandingBlock(256, 128, mode)
        self.expand3 = ExpandingBlock(128, 64, mode, relu)

        # Final Convolutional Layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        _, x = self.contract4(x)

        # Expanding path
        x = self.expand1(x, skip3)
        x = self.expand2(x, skip2)
        x = self.expand3(x, skip1)

        x = self.final_conv(x)
        return x