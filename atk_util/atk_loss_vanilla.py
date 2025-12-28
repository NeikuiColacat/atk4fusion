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
            num_total = labels_flat.size(0) - torch.sum(mask_flat)
            gamma = num_no_misc.float() / num_total.float()
        else:
            gamma = self.gamma

        # Compute separate losses
        loss_no_misc = F.cross_entropy(
            logits_flat, labels_no_misc, ignore_index=255, reduction="sum"
        )
        loss_misc = F.cross_entropy(
            logits_flat, labels_misc, ignore_index=255, reduction="sum"
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

def get_confusion_matrix(pred: torch.Tensor, label: torch.Tensor , patch_mask : torch.Tensor):
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    patch_mask = patch_mask.reshape(-1) != 1

    pred = pred[patch_mask] 
    label = label[patch_mask]

    mask = label != 255
    pred = pred[mask]
    label = label[mask]

    classes = torch.unique(torch.cat([label , pred]))
    num_classes = len(classes)

    classes_mp = {j.item(): i for i, j in enumerate(classes)}

    pred_mp = torch.tensor([classes_mp[i.item()] for i in pred])
    label_mp = torch.tensor([classes_mp[i.item()] for i in label])

    pred_pair_num = pred_mp * num_classes + label_mp 
    confusion_matrix = torch.bincount(pred_pair_num , minlength=num_classes**2).reshape(num_classes , num_classes)
    return confusion_matrix 


def get_mIoU(logits : torch.Tensor , label : torch.Tensor , mask : torch.Tensor) :
    pred = torch.argmax(logits, dim=1)

    conf_mat = get_confusion_matrix(pred, label, mask)

    TP = torch.diag(conf_mat) 
    FP = conf_mat.sum(0) - TP 
    FN = conf_mat.sum(1) - TP 
    IoU = TP / (TP + FP + FN + 1e-6) # 避免除零 
    mIoU = IoU.mean().item() 
    return IoU, mIoU



    


