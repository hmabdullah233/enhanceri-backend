import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def apply_realesrgan(img):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output, _ = upsampler.enhance(img_rgb)
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
