import torch
from torch import nn
import torch.nn.functional as F


class PatchGenerator(nn.Module):
    def __init__(self, img: torch.Tensor, 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.top = self.left = 256 
        self.patch_h = self.patch_w = 128 
        
        self.img = img
        self.get_mask_from_stages()

        device = self.img.device
        dtype = self.img.dtype
        self.mean = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)

        init_tensor = torch.rand((1 , 3, self.patch_h, self.patch_w), dtype=dtype, device=device)
        init_tensor = torch.clamp(init_tensor, 0, 1)

        self.patch = nn.Parameter(init_tensor)
    
    def get_mask_from_stages(self):
        top , left = self.top , self.left
        h , w = self.patch_h , self.patch_w

        mask_orig = torch.zeros_like(self.img)
        mask_orig[:, :, top : top + h, left : left + w] = 1
        self.mask = mask_orig[:,0,:,:]

    def insert_patch(self, img, patch):
        """
        the input resolution : NYUDepthv2 , 480 x 640
        """

        _, _, h, w = patch.shape
        
        top = self.top
        left = self.left
        img_adv = img.clone()
        img_adv[:, :, top:top+h, left:left+w] = patch

        return img_adv 

    def forward(self):
        patch = torch.clamp(self.patch, 0, 1)
        patch_norm = (patch - self.mean) / self.std
        img_adv = self.insert_patch(self.img , patch_norm) 
        return img_adv 