import torch.nn as nn
import torch.nn.functional as F


def normalization_layer(n, norm_arg):

    if n == "BN":
        return(nn.BatchNorm2d(norm_arg[0]))
    elif n == "LN":
        return nn.GroupNorm(1, norm_arg[0])
    elif n == "GN":
        return nn.GroupNorm(norm_arg[0]//2, norm_arg[0])
    else:
        raise ValueError('Valid options are BN/LN/GN')

dropout_value = 0.005

class Net(nn.Module):
    def __init__(self, n_type='BN'):
        self.n_type = n_type
        super(Net,self).__init__()

        # Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [16, 30, 30])
        )

        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 28, 28])
        )

        # Transition Block 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1),padding = 0, bias = False), 
            nn.MaxPool2d(2, 2)
        )

        # Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [16, 12, 12])
        )

        # Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 10, 10])
        )

        # Convolution Block 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 10, 10])
        )

        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1),padding = 0, bias = False), 
            nn.MaxPool2d(2, 2)
        )

        # Convolution Block 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [16, 5, 5])
        )

        # Convolution Block 7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 5, 5])
        )

        # Convolution Block 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 5, 5])
        )

        # Global average pooling layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5),
            nn.Conv2d(in_channels=32,out_channels=10,kernel_size=(1,1),padding = 0, bias = False),
        )

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transblock1(x)

        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x + self.convblock5(x)
        x = self.transblock2(x)

        x = self.convblock6(x)
        x = self.convblock7(x)
        x = x + self.convblock8(x)

        x = self.gap(x)
        x = x.view(-1,10)

        return F.log_softmax(x,dim = -1)
    

# Assignment 7 models

class Neta(nn.Module):
    def __init__(self):
        super(Neta, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24, RF = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, RF = 6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 12, RF = 6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 10, RF = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 8, RF = 14

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 6, RF = 18
        
        # Global average pooling layer
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding = 0, bias = False)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class Netb(nn.Module):
    def __init__(self, dropout_value=0.05):
        super(Netb, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24, RF = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 24, RF = 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, RF = 6
        

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 10, RF = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 8, RF = 14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6, RF = 18
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 4, RF = 22
        
        # Global average pooling layer
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding = 0, bias = False),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class Netc(nn.Module):
    def __init__(self, dropout_value=0.05):
        super(Netc, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24, RF = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 24, RF = 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, RF = 6
        

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 10, RF = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 8, RF = 14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6, RF = 18
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 4, RF = 22
        
        # Global average pooling layer
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(in_channels=10,out_channels=10,kernel_size=(1,1),padding = 0, bias = False)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

# Assignment 6 model

class Net_s6(nn.Module):
    def __init__(self, dropout_value=0.005):
        super(Net_s6, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3) 
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3) 
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(16, 8, 1)
        self.bn4 = nn.BatchNorm2d(8)
        self.conv5 = nn.Conv2d(8, 12, 3)
        self.bn5 = nn.BatchNorm2d(12)
        self.conv6 = nn.Conv2d(12,16,3)
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16,16,3)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.Conv2d(16,16,3)
        self.bn8 = nn.BatchNorm2d(16)
        self.conv9 = nn.Conv2d(16,10,1)
        self.avg = nn.AvgPool2d(3)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.dropout(self.bn1(F.relu(self.conv1(x))))#28 - 26 - 3
        x = self.dropout(self.bn2(F.relu(self.conv2(x))))#26 - 24 - 5
        x = self.dropout(self.bn3(F.relu(self.conv3(x))))#24 - 22 - 7
        x = self.pool(x)#22 - 11 - 8
        x = self.dropout(self.bn4(F.relu(self.conv4(x))))#11 - 11 - 8
        x = self.dropout(self.bn5(F.relu(self.conv5(x))))#11 - 9 - 12
        x = self.dropout(self.bn6(F.relu(self.conv6(x))))#9 - 7 - 16
        x = self.dropout(self.bn7(F.relu(self.conv7(x))))#7 - 5 - 20
        x = self.dropout(self.bn8(F.relu(self.conv8(x))))#5 - 3 - 24
        x = F.relu(self.conv9(x))#3 - 3 - 24
        x = self.avg(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)