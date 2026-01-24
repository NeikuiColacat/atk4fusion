import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


#---------------------------------------------------------------------
# PRINTABILITY LOSS
# Measures how well patch colors can be reproduced by a printer
# Uses a predefined set of printable colors from printer color charts
#---------------------------------------------------------------------
def get_printable_colors(levels: int = 6, grayscale_levels: int = 17):
    """
    Build a printer-friendly discrete RGB palette in [0,1].

    Default: "web-safe" style grid with 6 levels per channel (216 colors),
    plus an extra grayscale ramp for smoother tonal coverage.
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    if grayscale_levels < 2:
        raise ValueError("grayscale_levels must be >= 2")

    # Web-safe-ish quantization levels: 0, 0.2, 0.4, 0.6, 0.8, 1.0 (when levels=6)
    lv = torch.linspace(0.0, 1.0, steps=levels, dtype=torch.float32)

    # All RGB combinations: (levels^3, 3)
    grid = torch.cartesian_prod(lv, lv, lv)

    # Extra grayscale ramp improves neutral tones / printer stability
    g = torch.linspace(0.0, 1.0, steps=grayscale_levels, dtype=torch.float32)
    grays = torch.stack([g, g, g], dim=1)

    colors = torch.cat([grid, grays], dim=0)

    # Remove duplicates (grays already exist in grid) and keep deterministic order via unique-sort
    colors = torch.unique(colors, dim=0)

    return colors


def printability_loss(patch, printable_colors=None):
    """
    Computes the Non-Printability Score (NPS) loss for a patch.
    
    The loss measures how far each pixel is from the closest printable color.
    Lower loss = more printable.
    
    Args:
        patch: Tensor of shape (N, C, H, W), values in [0, 1]
        printable_colors: Tensor of shape (num_colors, 3), RGB in [0,1]
        
    Returns:
        Scalar loss value
    """
    device = patch.device
    
    if printable_colors is None:
        printable_colors = get_printable_colors()
    printable_colors = printable_colors.to(device)  # (num_colors, 3)
    
    # Ensure patch values are in [0, 1]
    patch = torch.clamp(patch, 0.0, 1.0)
    
    n, c, h, w = patch.shape
    
    # Reshape: (N, C, H, W) -> (N*H*W, C)
    patch_flat = patch.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (N*H*W, 3)
    
    # Compute L2 distance from each pixel to each printable color
    # patch_flat: (N*H*W, 1, 3), printable_colors: (1, num_colors, 3)
    patch_expanded = patch_flat.unsqueeze(1)  # (N*H*W, 1, 3)
    colors_expanded = printable_colors.unsqueeze(0)  # (1, num_colors, 3)
    
    distances = torch.norm(patch_expanded - colors_expanded, dim=2)  # (N*H*W, num_colors)
    
    # Get minimum distance to any printable color for each pixel
    min_distances, _ = torch.min(distances, dim=1)  # (N*H*W,)
    
    # Return mean of squared minimum distances
    return torch.mean(min_distances ** 2)


def smoothness_loss(patch):
    """
    Computes total variation loss for a patch to encourage smoothness.
    Reduces high-frequency noise for better printability.
    
    Args:
        patch: Tensor of shape (N, C, H, W)
        
    Returns:
        Scalar smoothness loss value
    """
    # Horizontal difference
    diff_h = patch[:, :, :, :-1] - patch[:, :, :, 1:]
    # Vertical difference  
    diff_v = patch[:, :, :-1, :] - patch[:, :, 1:, :]
    
    return torch.mean(diff_h ** 2) + torch.mean(diff_v ** 2)


class Loss_Manager(nn.Module):
    def __init__(
        self,
        img: torch.Tensor,
        depth: torch.Tensor,
        label: torch.Tensor,
        patch_gen,
        model,
        lambda_print: float = 100,      # Printability loss weight
        lambda_smooth: float = 100,    # Smoothness loss weight
    ):
        super().__init__()
        self.img = img
        self.depth = depth
        self.label = label
        self.model = model
        self.patch_gen = patch_gen 

        self.gamma = -1
        
        # Printability and smoothness loss weights
        self.lambda_print = lambda_print
        self.lambda_smooth = lambda_smooth
        
        # Pre-compute printable colors once
        self.printable_colors = get_printable_colors().to(img.device)

    def get_loss(self, logits, mask):
        B, C, H, W = logits.shape

        pred_flat = torch.argmax(logits, dim=1).reshape(-1)
        # Flatten tensors
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        labels_flat = self.label.reshape(-1)  # (B*H*W,)
        mask_flat = mask.reshape(-1)  # (B*H*W,)

        # Get predicted classes

        # Create two label copies for separate loss computation
        labels_misc = labels_flat.clone()  # For incorrectly classified pixels
        labels_no_misc = labels_flat.clone()  # For correctly classified pixels

        bg_mask = labels_flat == 255
        labels_misc[bg_mask] = 255
        labels_no_misc[bg_mask] = 255

        # Correctly classified pixels → ignore in misc loss
        labels_misc[labels_flat == pred_flat] = 255

        # Incorrectly classified pixels → ignore in no_misc loss
        labels_no_misc[labels_flat != pred_flat] = 255


        # Ignore patch region in both losses
        labels_misc[mask_flat == 1] = 255
        labels_no_misc[mask_flat == 1] = 255


        # Compute dynamic gamma if needed
        if self.gamma == -1:
            num_no_misc = torch.sum(labels_no_misc != 255)
            num_total = labels_flat.size(0) - torch.sum(mask_flat) - torch.sum(bg_mask)
            gamma = num_no_misc.float() / num_total.float()
        else:
            gamma = self.gamma

        # Compute separate losses
        loss_no_misc = F.cross_entropy(
            logits_flat, labels_no_misc, ignore_index=255, reduction="mean"
        )
        loss_misc = F.cross_entropy(
            logits_flat, labels_misc, ignore_index=255, reduction="mean"
        )

        # Combine losses with gamma weighting
        loss = gamma * loss_no_misc + (1 - gamma) * loss_misc

        return loss 

    def forward(self):

        img_adv = self.patch_gen()
        logits = self.model(img_adv, self.depth)  # [N, C, H, W]
        mask = self.patch_gen.mask                 # [N, 1, H, W]
        adv_loss = self.get_loss(logits , mask)


        depth_none = torch.full_like(self.depth , self.depth.mean())
        logits_without_depth = self.model(img_adv , depth_none)
        loss_withou_depth = self.get_loss(logits_without_depth , mask)
        
        # Get raw patch [0,1] for regularization losses
        patch = self.patch_gen.patch  # (N, 3, H, W), values in [0, 1]
        
        # Printability loss: encourages printer-reproducible colors
        print_loss = printability_loss(patch, self.printable_colors)
        
        # Smoothness loss: reduces high-frequency noise
        smooth_loss = smoothness_loss(patch)
        
        # Total loss = adversarial objective + regularization
        total_loss = (- adv_loss + loss_withou_depth 
                      + self.lambda_print * print_loss 
                      + self.lambda_smooth * smooth_loss)

        # print(print_loss.item() , smooth_loss.item() , adv_loss.item()) 
        return total_loss



def get_mIoU_sklearn(logits: torch.Tensor, label: torch.Tensor, patch_mask: torch.Tensor, ignore_index: int = 255):
    """
    计算单个样本的 mIoU，支持动态类别数。
    - logits: (B, C, H, W) 模型输出
    - label: (B, H, W) 真实标签
    - patch_mask: (B, H, W) 补丁区域，1 表示补丁区域（需过滤）
    - ignore_index: 忽略的标签值（默认 255）
    """

    # 1. 转换数据类型并展平
    # 先在 GPU 上做 argmax，减少传回 CPU 的数据量
    pred = torch.argmax(logits, dim=1).reshape(-1).cpu().numpy()
    label = label.reshape(-1).cpu().numpy()
    patch_mask = patch_mask.reshape(-1).cpu().numpy()

    # 2. 掩码过滤：只保留不是 ignore_index 且不在补丁区域的像素
    valid_mask = (label != ignore_index) & (patch_mask == 0)
    pred = pred[valid_mask]
    label = label[valid_mask]

    # 3. 动态类别集合
    # 提取当前样本中存在的所有类别标签
    classes = np.unique(np.concatenate([label, pred]))

    # 4. 构建混淆矩阵
    conf_mat = confusion_matrix(label, pred, labels=classes)

    # 5. 计算 IoU 指标
    # TP: 真正例, FP: 假正例, FN: 假负例
    TP = np.diag(conf_mat)
    FP = conf_mat.sum(axis=0) - TP
    FN = conf_mat.sum(axis=1) - TP

    # 计算每个类别的 IoU (1e-6 用于防止除以 0)
    IoU = TP / (TP + FP + FN + 1e-6)

    # 6. 计算所有存在类别的平均值
    mIoU = IoU.mean()

    return IoU, mIoU*100
