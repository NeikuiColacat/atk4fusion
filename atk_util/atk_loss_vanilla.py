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

        self.gamma = -1

    # def get_loss(self , logits , mask) :

    #     B, C, H, W = logits.shape

    #     logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    #     labels_flat = self.label.view(-1)  # (B*H*W,)
    #     mask_flat = mask.view(-1)  # (B*H*W,)
        
    #     labels_flat = labels_flat.clone()
    #     labels_flat[mask_flat == 1] = 255
        
    #     loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=255, reduction="mean")
        
    #     return loss
    
    def get_loss(self, logits, mask):
        """
        Calculate loss with separate weighting for correctly and incorrectly classified pixels.
        
        Args:
            logits: Model output [B, C, H, W]
            mask: Patch mask [B, 1, H, W] where 1 indicates patch region
            
        Returns:
            Combined loss value
        """
        B, C, H, W = logits.shape

        # Flatten tensors
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        labels_flat = self.label.view(-1)  # (B*H*W,)
        mask_flat = mask.view(-1)  # (B*H*W,)

        # Get predicted classes
        pred = torch.argmax(logits_flat, dim=1).detach()

        # Create two label copies for separate loss computation
        labels_misc = labels_flat.clone()  # For incorrectly classified pixels
        labels_no_misc = labels_flat.clone()  # For correctly classified pixels

        # Correctly classified pixels → ignore in misc loss
        labels_misc[labels_flat == pred] = 255

        # Incorrectly classified pixels → ignore in no_misc loss
        labels_no_misc[labels_flat != pred] = 255

        # Ignore patch region in both losses
        labels_misc[mask_flat == 1] = 255
        labels_no_misc[mask_flat == 1] = 255

        # Compute dynamic gamma if needed
        if self.gamma == -1:
            num_no_misc = torch.sum(labels_no_misc != 255)
            num_total = labels_flat.size(0) - torch.sum(mask_flat)
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


        # depth_none = torch.full_like(self.depth , self.depth.mean())
        # logits_without_depth = self.model(img_adv , depth_none)
        # loss_withou_depth = self.get_loss(logits_without_depth , mask)
        
        return - adv_loss