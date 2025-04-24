import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from basicsr.archs.codeformer_arch import CodeFormer

def apply_codeformer(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8,
                       n_layers=9, connect_list=['32', '64', '128', '256']).to(device)
    model.eval()

    # Load pretrained weights
    checkpoint = torch.hub.load_state_dict_from_url(
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        map_location=device
    )
    model.load_state_dict(checkpoint['params_ema'])

    # Preprocess image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = to_pil_image(img_rgb)
    tensor = to_tensor(pil_img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        restored, _ = model(tensor, w=0.7, adain=True)

    # Postprocess
    restored = restored.squeeze().clamp_(0, 1).cpu()
    output_img = restored.permute(1, 2, 0).numpy() * 255
    output_img = output_img.astype(np.uint8)
    return cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
