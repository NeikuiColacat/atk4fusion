import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss_Manager(nn.Module):
    def __init__(
        self,
        img: torch.Tensor,
        depth: torch.Tensor,
        label: torch.Tensor,
        patch_gen,
        model,
    ):
        super().__init__()
        self.img = img
        self.depth = depth
        self.label = label
        self.model = model
        self.patch_gen = patch_gen 

    def get_loss(self , logits , mask) :

        B, C, H, W = logits.shape
    
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        labels_flat = self.label.view(-1)  # (B*H*W,)
        mask_flat = mask.view(-1)  # (B*H*W,)
        
        labels_flat = labels_flat.clone()
        labels_flat[mask_flat == 1] = 255
        
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=255, reduction="mean")
        
        return loss
        

    def forward(self):

        ######################################

        img_adv = self.patch_gen()
        logits = self.model(img_adv, self.depth)  # [N, C, H, W]
        mask = self.patch_gen.mask                 # [N, 1, H, W]
        adv_loss = self.get_loss(logits , mask)


        depth_none = torch.full_like(self.depth , self.depth.mean())
        logits_without_depth = self.model(img_adv , depth_none)
        loss_withou_depth = self.get_loss(logits_without_depth , mask)
        
        return - adv_loss