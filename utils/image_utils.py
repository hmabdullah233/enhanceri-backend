import cv2
import os
import torch
import numpy as np

from models.realesrgan.main import apply_realesrgan
from models.codeformer.main import apply_codeformer
from models.bsrgan.main import apply_bsrgan
from models.swinir.main import apply_swinir
from models.hdrnet.main import apply_hdrnet
from models.whitebox.main import apply_whitebox

def full_enhance(input_path, output_path):
    img = cv2.imread(input_path)

    img = apply_codeformer(img)
    img = apply_bsrgan(img)
    img = apply_realesrgan(img)
    img = apply_swinir(img)
    img = apply_hdrnet(img)
    img = apply_whitebox(img)

    cv2.imwrite(output_path, img)
