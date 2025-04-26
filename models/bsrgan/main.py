import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from models.basicsr.archs.rrdbnet_arch import RRDBNet
from models.basicsr.utils.download_util import load_file_from_url

def apply_bsrgan(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4).to(device)
    model.eval()

    model_path = load_file_from_url(
        url='https://github.com/cszn/BSRGAN/releases/download/v1.0/BSRGAN.pth',
        model_dir='weights', progress=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = to_pil_image(img_rgb)
    tensor = to_tensor(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze().clamp_(0, 1).cpu()
    output_img = output.permute(1, 2, 0).numpy() * 255
    output_img = output_img.astype(np.uint8)
    return cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
