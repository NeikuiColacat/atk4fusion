import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .NYUv2_img_with_patch import PatchGeneratorPadding
from sklearn.metrics import confusion_matrix
from torch import Tensor


# #---------------------------------------------------------------------
# # PRINTABILITY LOSS
# # Measures how well patch colors can be reproduced by a printer
# # Uses a predefined set of printable colors from printer color charts
# #---------------------------------------------------------------------
# def get_printable_colors(levels: int = 6, grayscale_levels: int = 17):
#     """
#     Build a printer-friendly discrete RGB palette in [0,1].

#     Default: "web-safe" style grid with 6 levels per channel (216 colors),
#     plus an extra grayscale ramp for smoother tonal coverage.
#     """
#     if levels < 2:
#         raise ValueError("levels must be >= 2")
#     if grayscale_levels < 2:
#         raise ValueError("grayscale_levels must be >= 2")

#     # Web-safe-ish quantization levels: 0, 0.2, 0.4, 0.6, 0.8, 1.0 (when levels=6)
#     lv = torch.linspace(0.0, 1.0, steps=levels, dtype=torch.float32)

#     # All RGB combinations: (levels^3, 3)
#     grid = torch.cartesian_prod(lv, lv, lv)

#     # Extra grayscale ramp improves neutral tones / printer stability
#     g = torch.linspace(0.0, 1.0, steps=grayscale_levels, dtype=torch.float32)
#     grays = torch.stack([g, g, g], dim=1)

#     colors = torch.cat([grid, grays], dim=0)

#     # Remove duplicates (grays already exist in grid) and keep deterministic order via unique-sort
#     colors = torch.unique(colors, dim=0)

#     return colors


# def printability_loss(patch, printable_colors=None):
#     """
#     Computes the Non-Printability Score (NPS) loss for a patch.
    
#     The loss measures how far each pixel is from the closest printable color.
#     Lower loss = more printable.
    
#     Args:
#         patch: Tensor of shape (N, C, H, W), values in [0, 1]
#         printable_colors: Tensor of shape (num_colors, 3), RGB in [0,1]
        
#     Returns:
#         Scalar loss value
#     """
#     device = patch.device
    
#     if printable_colors is None:
#         printable_colors = get_printable_colors()
#     printable_colors = printable_colors.to(device)  # (num_colors, 3)
    
#     # Ensure patch values are in [0, 1]
#     patch = torch.clamp(patch, 0.0, 1.0)
    
#     n, c, h, w = patch.shape
    
#     # Reshape: (N, C, H, W) -> (N*H*W, C)
#     patch_flat = patch.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (N*H*W, 3)
    
#     # Compute L2 distance from each pixel to each printable color
#     # patch_flat: (N*H*W, 1, 3), printable_colors: (1, num_colors, 3)
#     patch_expanded = patch_flat.unsqueeze(1)  # (N*H*W, 1, 3)
#     colors_expanded = printable_colors.unsqueeze(0)  # (1, num_colors, 3)
    
#     distances = torch.norm(patch_expanded - colors_expanded, dim=2)  # (N*H*W, num_colors)
    
#     # Get minimum distance to any printable color for each pixel
#     min_distances, _ = torch.min(distances, dim=1)  # (N*H*W,)
    
#     # Return mean of squared minimum distances
#     return torch.mean(min_distances ** 2)


# def smoothness_loss(patch):
#     """
#     Computes total variation loss for a patch to encourage smoothness.
#     Reduces high-frequency noise for better printability.
    
#     Args:
#         patch: Tensor of shape (N, C, H, W)
        
#     Returns:
#         Scalar smoothness loss value
#     """
#     # Horizontal difference
#     diff_h = patch[:, :, :, :-1] - patch[:, :, :, 1:]
#     # Vertical difference  
#     diff_v = patch[:, :, :-1, :] - patch[:, :, 1:, :]
    
#     return torch.mean(diff_h ** 2) + torch.mean(diff_v ** 2)


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
        # self.lambda_print = lambda_print
        # self.lambda_smooth = lambda_smooth
        
        # Pre-compute printable colors once
        # self.printable_colors = get_printable_colors().to(img.device)

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


        depth_none = torch.full_like(self.depth , self.depth.mean().item())
        logits_without_depth = self.model(img_adv , depth_none)
        loss_withou_depth = self.get_loss(logits_without_depth , mask)
        
        # Get raw patch [0,1] for regularization losses
        patch = self.patch_gen.patch  # (N, 3, H, W), values in [0, 1]
        
        # Printability loss: encourages printer-reproducible colors
        # print_loss = printability_loss(patch, self.printable_colors)
        
        # Smoothness loss: reduces high-frequency noise
        # smooth_loss = smoothness_loss(patch)
        
        # Total loss = adversarial objective + regularization
        total_loss = -adv_loss + loss_withou_depth
                    #   + self.lambda_print * print_loss 
                    #   + self.lambda_smooth * smooth_loss)

        # print(print_loss.item() , smooth_loss.item() , adv_loss.item()) 
        return total_loss



class LossManagerPadding(nn.Module):
    def __init__(
        self,
        patch_gen : PatchGeneratorPadding,
        model : nn.Module,
    ):
        super().__init__()
        self.model: nn.Module = model
        self.patch_gen: PatchGeneratorPadding = patch_gen 
        self.gamma = -1

    def get_loss(self, logits: Tensor, label: Tensor, mask: Tensor):
        B, C, H, W = logits.shape

        pred_flat = torch.argmax(logits, dim=1).reshape(-1)
        # Flatten tensors
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        labels_flat = label.reshape(-1)  # (B*H*W,)

        mask = mask.expand(label.shape[0], -1, -1, -1)
        mask_flat = mask.reshape(-1)  # (B*H*W,)

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

        depth, label, mask = (
            self.patch_gen.modal_xs_padded,
            self.patch_gen.label_padded,
            self.patch_gen.mask,
        )

        logits = self.model(img_adv, depth)  # [N, C, H, W]
        adv_loss = self.get_loss(logits, label, mask)

        depth_none = torch.full_like(depth, depth.mean().item())
        logits_without_depth = self.model(img_adv , depth_none)
        loss_withou_depth = self.get_loss(logits_without_depth, label, mask)
        
        total_loss = -adv_loss + loss_withou_depth
        return total_loss