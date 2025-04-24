import torch.nn as nn

class EnlightenGAN(nn.Module):
    def __init__(self):
        super(EnlightenGAN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
