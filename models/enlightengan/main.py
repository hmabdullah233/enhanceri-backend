import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models.enlightengan.model import EnlightenGAN

def apply_enlightengan(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    input_tensor = transform(pil_img).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EnlightenGAN().to(device)
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://github.com/arsenyinfo/enlightenGAN-pytorch/releases/download/v1.0/enlightengan.pth',
        map_location=device
    ))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor.to(device))[0].cpu()

    output_img = output.squeeze().permute(1, 2, 0).numpy()
    output_img = (output_img * 255).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
