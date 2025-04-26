import cv2
import os
import torch
import numpy as np

from models.realesrgan.main import apply_realesrgan
from models.bsrgan.main import apply_bsrgan
from models.swinir.main import apply_swinir
from models.enlightengan.main import apply_enlightengan
from models.dce.main import apply_dce

def full_enhance(input_path, output_path):
    img = cv2.imread(input_path)

    img = apply_bsrgan(img)
    img = apply_realesrgan(img)
    img = apply_swinir(img)
    img = apply_enlightengan(img)
    img = apply_dce(img)

    cv2.imwrite(output_path, img)
