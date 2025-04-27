import torch
import torch.nn as nn

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, num_feat, num_block, num_grow_ch, scale):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.body = nn.Sequential(
            *[nn.Conv2d(num_feat, num_feat, 3, padding=1) for _ in range(num_block)]
        )
        self.head = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.tail = nn.Conv2d(num_feat, num_out_ch, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
