import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    #This defines the structure of the NN.
    #Total Params changes after adding bias =True/False
    def __init__(self):
        super(Net, self).__init__()
        # r_in:1, n_in:28, j_in:1, s:1, r_out:3, n_out:26, j_out:1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
        # r_in:3, n_in:26, j_in:1, s:1, r_out:5, n_out:24, j_out:1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        # r_in:5, n_in:24, j_in:1, s:2, r_out:6, n_out:12, j_out:2
        #maxpool
        # r_in:6, n_in:12, j_in:2, s:1, r_out:10, n_out:10, j_out:2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
        # r_in:10, n_in:10, j_in:2, s:1, r_out:14, n_out:8, j_out:2
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        # r_in:14, n_in:8, j_in:2, s:2, r_out:16, n_out:4, j_out:4
        #maxpool
        
        self.fc1 = nn.Linear(4096, 50, bias=False)
        self.fc2 = nn.Linear(50, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)