import cv2
import numpy as np
import torch
from models.codeformer.basicsr.archs.swinir_arch import SwinIR
from models.codeformer.basicsr.utils.download_util import load_file_from_url
from torchvision.transforms.functional import to_tensor, to_pil_image

def apply_swinir(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    model_path = load_file_from_url(
        url='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/SwinIR_Medium.pth',
        model_dir='weights',
        progress=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = to_pil_image(img_rgb)
    tensor = to_tensor(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze().clamp_(0, 1).cpu()
    output_img = output.permute(1, 2, 0).numpy() * 255
    output_img = output_img.astype(np.uint8)
    return cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
