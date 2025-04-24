import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models.dce.dce_model import enhance_image

def apply_dce(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    input_tensor = transform(pil_img).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enhanced = enhance_image(input_tensor.to(device))

    output = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()
    output = (output * 255).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
