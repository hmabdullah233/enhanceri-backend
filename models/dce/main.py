import torch
import torch.nn as nn

class DCEBlock(nn.Module):
    def __init__(self, channels):
        super(DCEBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 24, 3, 1, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)

class DCE_Net(nn.Module):
    def __init__(self):
        super(DCE_Net, self).__init__()
        self.dce_block = DCEBlock(3)

    def forward(self, x):
        A = self.dce_block(x)
        R = torch.zeros_like(x)
        for i in range(8):
            R = R + A[:, i:i+1, :, :] * (torch.pow(R - 1, 2) - R)
        return R
