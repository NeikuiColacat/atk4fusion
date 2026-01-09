import numpy as np
import torch
import pathlib
import matplotlib.pyplot as plt
import os
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

def save_all_results(
    model,
    images,
    img_adv, 
    modal_xs,
    mean_depth,
    idx=0,
    save_path="output/adv_results"
):
    """
    保存所有对抗攻击结果，包括4种分割可视化和对抗图片
    
    Args:
        model: 分割模型
        images: 原始图片 [B, 3, H, W]
        img_adv: 插入patch后的图片 [B, 3, H, W]
        modal_xs: 深度图 [B, 3, H, W]
        mean_depth: 平均深度图 [B, 3, H, W]
        idx: batch索引，用于文件命名
        save_path: 保存路径
    """
    # 创建子文件夹
    clean_path = os.path.join(save_path, "clean")
    adv_path = os.path.join(save_path, "adv")
    clean_no_depth_path = os.path.join(save_path, "clean_no_depth")
    adv_no_depth_path = os.path.join(save_path, "adv_no_depth")
    adv_img_path = os.path.join(save_path, "adv_images")
    
    for path in [clean_path, adv_path, clean_no_depth_path, adv_no_depth_path, adv_img_path]:
        os.makedirs(path, exist_ok=True)
    
    # 1. 获取4种logits
    with torch.no_grad():
        logits_clean = model(images, modal_xs)
        logits_adv = model(img_adv, modal_xs)
        logits_clean_no_depth = model(images, mean_depth)
        logits_adv_no_depth = model(img_adv, mean_depth)
    
    # 2. 转换为彩色分割图
    palette = get_palette()
    
    def logits_to_color(logits):
        preds = logits.argmax(dim=1)  # [B, H, W]
        preds_color = torch.tensor(palette, device=preds.device, dtype=torch.uint8)[preds]  # [B,H,W,3]
        return preds_color.permute(0, 3, 1, 2).float() / 255.0  # [B,3,H,W]
    
    pred_clean = logits_to_color(logits_clean)
    pred_adv = logits_to_color(logits_adv)
    pred_clean_no_depth = logits_to_color(logits_clean_no_depth)
    pred_adv_no_depth = logits_to_color(logits_adv_no_depth)
    
    # 3. 保存每张图片的结果到不同文件夹
    B = images.shape[0]
    for i in range(B):
        filename = f"batch{idx}_img{i}.png"
        
        # 保存4种分割结果到各自文件夹
        save_image(pred_clean[i], os.path.join(clean_path, filename))
        save_image(pred_adv[i], os.path.join(adv_path, filename))
        save_image(pred_clean_no_depth[i], os.path.join(clean_no_depth_path, filename))
        save_image(pred_adv_no_depth[i], os.path.join(adv_no_depth_path, filename))
        
        # 保存对抗图片（反归一化）到单独文件夹
        img_adv_vis = de_norm(img_adv[i:i+1])
        save_image(img_adv_vis[0], os.path.join(adv_img_path, filename))

def save_mIoU_log(
    mIoU_clean,
    mIoU_adv,
    mIoU_clean_no_depth,
    mIoU_adv_no_depth,
    log_path="output/adv_results/mIoU_log.txt"
):
    """
    将4种情况的 mIoU 保存到日志文件
    
    Args:
        mIoU_clean: clean 的 mIoU
        mIoU_adv: adv 的 mIoU
        mIoU_clean_no_depth: clean no depth 的 mIoU
        mIoU_adv_no_depth: adv no depth 的 mIoU
        idx: batch 索引
        log_path: 日志文件路径
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 追加写入模式
    with open(log_path, "a") as f:
        f.write(f"  clean mIoU:          {mIoU_clean:.4f}\n")
        f.write(f"  adv mIoU:            {mIoU_adv:.4f}\n")
        f.write(f"  clean_no_depth mIoU: {mIoU_clean_no_depth:.4f}\n")
        f.write(f"  adv_no_depth mIoU:   {mIoU_adv_no_depth:.4f}\n")
        f.write("-" * 50 + "\n")