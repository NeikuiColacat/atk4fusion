import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class PatchGenerator(nn.Module):
    def __init__(self, img: torch.Tensor, 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.top = self.left = 256 
        self.patch_h = self.patch_w  = 200 
        
        self.img = img
        self.batch_size = img.shape[0]
        self.get_mask()

        device = self.img.device
        dtype = self.img.dtype
        self.mean = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)

        init_tensor = torch.rand((self.batch_size , 3, self.patch_h, self.patch_w), dtype=dtype, device=device)
        init_tensor = torch.clamp(init_tensor, 0, 1)

        self.patch = nn.Parameter(init_tensor)
    
    def get_mask(self):
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

class PatchGeneratorPadding(nn.Module):
    def __init__(
        self,
        pad: int,
        img: Tensor,
        label: Tensor,
        modal_xs: Tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ) -> None:
        super().__init__()

        self.pad: int = pad
        device = self.device = img.device
        dtype = self.dtype = img.dtype

        self.mean: Tensor = torch.tensor(
            mean, device=device, dtype=dtype
        ).reshape(1, -1, 1, 1)

        self.std: Tensor = torch.tensor(
            std, device=device, dtype=dtype
        ).reshape(1, -1, 1, 1)
        self.update_batch(img, label, modal_xs)


        _, _, H, W = self.img.shape

        self.patch = nn.Parameter(
            torch.rand(1, 3, H + 2 * pad, W + 2 * pad, device=device, dtype=dtype)
        )
        self.mask: Tensor = torch.ones(
            1, 1, H + 2 * pad, W + 2 * pad, device=device, dtype=dtype
        )
        self.mask[:, :, pad : pad + H, pad : pad + W] = 0

    def update_batch(self, img: Tensor, label: Tensor, modal_xs: Tensor) -> None:
        if hasattr(self , "img"):
            assert img.shape[1:] == self.img.shape[1:]

        self.img: Tensor = img
        self.label: Tensor = label
        self.modal_xs : Tensor = modal_xs

        self.img_padded: Tensor = F.pad(
            self.img, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0
        )
        self.label_padded: Tensor = F.pad(
            self.label,
            (self.pad, self.pad, self.pad, self.pad),
            mode="constant",
            value=255,
        )
        self.modal_xs_padded: Tensor = F.pad(
            self.modal_xs,
            (self.pad, self.pad, self.pad, self.pad),
            mode="constant",
            value=self.modal_xs.mean().item(),
        )

    def forward(self) -> Tensor:
        patch_normalized: Tensor = torch.clamp(self.patch, 0, 1)
        patch_normalized: Tensor = (patch_normalized - self.mean) / self.std

        img_adv: Tensor = (
            self.img_padded * (1 - self.mask) + patch_normalized * self.mask
        )

        return img_adv