import torch.nn as nn
import torch

class DCEUNet(nn.Module):
    def __init__(self):
        super(DCEUNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

def enhance_image(input_tensor):
    model = DCEUNet()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://github.com/Li-Chongyi/Zero-DCE/releases/download/v1.0/zero_dce.pth',
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ))
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output
