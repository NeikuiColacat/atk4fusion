import torch 
import numpy as np
from torch import Tensor 
from sklearn.metrics import confusion_matrix

def get_mIoU_sklearn(
    logits: torch.Tensor,
    label: torch.Tensor,
    patch_mask: torch.Tensor,
    ignore_index: int = 255,
) -> tuple:
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

class StreamingMIoU:
    """
    This is for calculate mIoU for all the validation dataset
    """

    def __init__(self, num_classes : int , ignore_label:int = 255 , device = "cuda"):
        self.num_classes: int = num_classes
        self.ignore_label:int = ignore_label
        self.device: str = device
        self.conf_matrix: torch.Tensor = torch.zeros(
            num_classes, num_classes, device=device
        )

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        patch_mask: torch.Tensor,
        ignore_index: int = 255,
    ) -> None:
        pred: Tensor = logits.argmax(dim=1)
        keep: Tensor = (label != ignore_index) & (patch_mask == 0)

        tmp_conf_matrix: Tensor = torch.bincount(
            label[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

        self.conf_matrix += tmp_conf_matrix

    @torch.no_grad()
    def compute(self) -> float:
        ious: Tensor = self.conf_matrix.diag() / (
            self.conf_matrix.sum(0) + self.conf_matrix.sum(1) - self.conf_matrix.diag()
        )

        ious[ious.isnan()] = 0
        miou: float = ious.mean().item()

        return round(miou * 100, 2)
