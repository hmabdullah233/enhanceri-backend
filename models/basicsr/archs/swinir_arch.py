import torch
import torch.nn as nn

class SwinIR(nn.Module):
    def __init__(self, upscale, in_chans, img_size, window_size, img_range, depths, embed_dim, num_heads, mlp_ratio, upsampler, resi_connection):
        super(SwinIR, self).__init__()
        self.conv = nn.Conv2d(in_chans, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
