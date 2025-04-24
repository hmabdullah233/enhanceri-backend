import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_relu=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(out_channels)]
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels, use_relu=False)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(4)])
        self.decoder = nn.Sequential(
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x
