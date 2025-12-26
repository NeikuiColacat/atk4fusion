import numpy as np
import torch
import pathlib
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def get_palette():
    return np.array(
        [
            [0, 0, 0],
            [174, 199, 232],
            [152, 223, 138],
            [31, 119, 180],
            [255, 187, 120],
            [188, 189, 34],
            [140, 86, 75],
            [255, 152, 150],
            [214, 39, 40],
            [197, 176, 213],
            [148, 103, 189],
            [196, 156, 148],
            [23, 190, 207],
            [247, 182, 210],
            [219, 219, 141],
            [255, 127, 14],
            [158, 218, 229],
            [44, 160, 44],
            [112, 128, 144],
            [227, 119, 194],
            [82, 84, 163],
            [100, 85, 144],
            [178, 76, 76],
            [248, 156, 116],
            [146, 53, 53],
            [105, 100, 100],
            [118, 60, 40],
            [76, 76, 153],
            [60, 143, 113],
            [171, 71, 188],
            [30, 100, 230],
            [180, 200, 70],
            [92, 175, 236],
            [204, 204, 204],
            [138, 44, 28],
            [194, 155, 97],
            [143, 169, 86],
            [136, 45, 23],
            [224, 102, 13],
            [163, 38, 168],
        ],
        dtype=np.uint8,
    )


def de_norm(img: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img.detach()

    mean = torch.tensor(mean, device=img.device, dtype=img.dtype).reshape(-1, 3, 1, 1)
    std = torch.tensor(std, device=img.device, dtype=img.dtype).reshape(-1, 3, 1, 1)
    img = img * std + mean
    return img.clamp(0, 1)

def save_rgb(
    img_without_norm: torch.Tensor, file_name: str, save_path="output/adv_results"
):
    img_without_norm = img_without_norm.detach()

    if img_without_norm.shape[0] == 1:
        img_without_norm = img_without_norm.squeeze(dim=0)

    assert img_without_norm.dim() == 3

    save_path = save_path + "/" + file_name
    save_image(img_without_norm, save_path)

def save_adv_results(
    model,
    img_adv,
    modal_xs,
    file_prefix="sample",
    save_path="output/adv_results",
):

    img_adv_vis = de_norm(img_adv)
    save_rgb(img_adv_vis, f"{file_prefix}.png", save_path)

    logits = model(img_adv, modal_xs)
    preds = logits.argmax(dim=1)

    palette = get_palette()
    preds_color = torch.tensor(palette, dtype=torch.uint8)[preds]   # [B,H,W,3]

    preds_color = preds_color.permute(0,3,1,2).float() / 255.0
    save_image(preds_color, "f{save_path}/{file_prefix}_pred.png")
