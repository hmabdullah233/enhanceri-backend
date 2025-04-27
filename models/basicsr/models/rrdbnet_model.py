import torch.nn as nn

class RRDBNetModel(nn.Module):
    def __init__(self, model):
        super(RRDBNetModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)
