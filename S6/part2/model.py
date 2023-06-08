import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dropout_value=0.005):
        super(Net, self).__init__()
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