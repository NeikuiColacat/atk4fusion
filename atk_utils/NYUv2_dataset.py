"""
NYUv2 Dataset API
独立的 NYUDepthv2 数据集加载器，不依赖原作者的 Engine

Usage:
    from atk_util.NYUv2_dataset import get_NYUv2_train_loader, get_NYUv2_val_loader
    
    train_loader = get_NYUv2_train_loader(batch_size=8)
    val_loader = get_NYUv2_val_loader(batch_size=1)
    
    for batch in val_loader:
        rgb = batch["data"]        # [B, 3, 480, 640]
        depth = batch["modal_x"]   # [B, 3, 480, 640]
        label = batch["label"]     # [B, 480, 640]
"""

import os
import cv2
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List


# ============== 路径配置 ==============
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
DEFAULT_DATASET_DIR = os.path.join(_PROJECT_ROOT, "datasets", "NYUDepthv2")
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "DFormer.yaml")


# ============== NYUv2 数据集配置 ==============
NYUV2_CONFIG = {
    "dataset_name": "NYUDepthv2",
    "num_classes": 40,
    "num_train_imgs": 795,
    "num_eval_imgs": 654,
    "image_height": 480,
    "image_width": 640,
    "background": 255,
    "rgb_format": ".jpg",
    "gt_format": ".png",
    "x_format": ".png",
    "gt_transform": True,  # label - 1
    "x_is_single_channel": True,
    "norm_mean": np.array([0.485, 0.456, 0.406]),
    "norm_std": np.array([0.229, 0.224, 0.225]),
    "depth_norm_mean": np.array([0.48, 0.48, 0.48]),
    "depth_norm_std": np.array([0.28, 0.28, 0.28]),
    "class_names": [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
        "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves",
        "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling",
        "books", "refridgerator", "television", "paper", "towel", "shower curtain",
        "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp",
        "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop",
    ],
}


# ============== 图像预处理工具 ==============
def normalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """归一化图像"""
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return img


def random_mirror(rgb: np.ndarray, gt: np.ndarray, depth: np.ndarray) -> Tuple:
    """随机水平翻转"""
    if np.random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        depth = cv2.flip(depth, 1)
    return rgb, gt, depth


def random_scale(rgb: np.ndarray, gt: np.ndarray, depth: np.ndarray, 
                 scales: List[float]) -> Tuple:
    """随机缩放"""
    scale = np.random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return rgb, gt, depth, scale


def random_crop(img: np.ndarray, crop_size: Tuple[int, int], 
                pad_value: float = 0) -> np.ndarray:
    """随机裁剪到指定大小"""
    h, w = img.shape[:2]
    crop_h, crop_w = crop_size
    
    # 如果图像小于裁剪大小，先 padding
    if h < crop_h or w < crop_w:
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        if len(img.shape) == 3:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, 
                                     cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
        else:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_CONSTANT, value=pad_value)
        h, w = img.shape[:2]
    
    # 随机选择裁剪位置
    start_h = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
    start_w = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
    
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]


# ============== 预处理类 ==============
class TrainPreprocess:
    """训练数据预处理"""
    
    def __init__(self, 
                 norm_mean: np.ndarray,
                 norm_std: np.ndarray,
                 depth_norm_mean: np.ndarray,
                 depth_norm_std: np.ndarray,
                 image_height: int = 480,
                 image_width: int = 640,
                 scale_array: List[float] = None):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.depth_norm_mean = depth_norm_mean
        self.depth_norm_std = depth_norm_std
        self.crop_size = (image_height, image_width)
        self.scale_array = scale_array or [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    
    def __call__(self, rgb: np.ndarray, gt: np.ndarray, 
                 depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 随机翻转
        rgb, gt, depth = random_mirror(rgb, gt, depth)
        
        # 随机缩放
        rgb, gt, depth, _ = random_scale(rgb, gt, depth, self.scale_array)
        
        # 归一化
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        depth = normalize(depth, self.depth_norm_mean, self.depth_norm_std)
        
        # 随机裁剪
        # 需要同步裁剪 rgb, gt, depth
        h, w = rgb.shape[:2]
        crop_h, crop_w = self.crop_size
        
        # padding if needed
        if h < crop_h or w < crop_w:
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)
            rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            gt = cv2.copyMakeBorder(gt, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            depth = cv2.copyMakeBorder(depth, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = rgb.shape[:2]
        
        start_h = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        start_w = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
        
        rgb = rgb[start_h:start_h + crop_h, start_w:start_w + crop_w]
        gt = gt[start_h:start_h + crop_h, start_w:start_w + crop_w]
        depth = depth[start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        # HWC -> CHW
        rgb = rgb.transpose(2, 0, 1)
        depth = depth.transpose(2, 0, 1)
        
        return rgb, gt, depth


class ValPreprocess:
    """验证数据预处理"""
    
    def __init__(self,
                 norm_mean: np.ndarray,
                 norm_std: np.ndarray,
                 depth_norm_mean: np.ndarray,
                 depth_norm_std: np.ndarray):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.depth_norm_mean = depth_norm_mean
        self.depth_norm_std = depth_norm_std
    
    def __call__(self, rgb: np.ndarray, gt: np.ndarray,
                 depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 归一化
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        depth = normalize(depth, self.depth_norm_mean, self.depth_norm_std)
        
        # HWC -> CHW
        rgb = rgb.transpose(2, 0, 1)
        depth = depth.transpose(2, 0, 1)
        
        return rgb, gt, depth


# ============== 数据集类 ==============
class NYUv2Dataset(Dataset):
    """NYUDepthv2 数据集"""
    
    def __init__(self,
                 dataset_root: str = None,
                 split: str = "val",
                 preprocess = None,
                 config: dict = None):
        """
        Args:
            dataset_root: 数据集根目录
            split: "train" 或 "val"
            preprocess: 预处理函数
            config: 数据集配置
        """
        super().__init__()
        
        self.dataset_root = dataset_root or DEFAULT_DATASET_DIR
        self.split = split
        self.preprocess = preprocess
        self.config = config or NYUV2_CONFIG
        
        # 路径配置
        self.rgb_root = os.path.join(self.dataset_root, "RGB")
        self.depth_root = os.path.join(self.dataset_root, "Depth")
        self.label_root = os.path.join(self.dataset_root, "Label")
        
        # 读取文件列表
        list_file = "train.txt" if split == "train" else "test.txt"
        list_path = os.path.join(self.dataset_root, list_file)
        
        with open(list_path, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        
        print(f"[NYUv2] Loaded {len(self.file_names)} {split} samples from {self.dataset_root}")
    
    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item_name = self.file_names[index]
        
        # 解析文件名 (格式: "NYUDepthv2/xxx" 或直接 "xxx")
        if "/" in item_name:
            item_name = item_name.split("/")[-1]
        item_name = item_name.replace(".jpg", "").replace(".png", "")
        
        # 构建路径
        rgb_path = os.path.join(self.rgb_root, item_name + self.config["rgb_format"])
        depth_path = os.path.join(self.depth_root, item_name + self.config["x_format"])
        label_path = os.path.join(self.label_root, item_name + self.config["gt_format"])
        
        # 读取图像
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # BGR
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if rgb is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if depth is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        if label is None:
            raise FileNotFoundError(f"Label image not found: {label_path}")
        
        # Depth 转为 3 通道 (复制成 HxWx3)
        depth = cv2.merge([depth, depth, depth])
        
        # Label 变换 (label - 1, 因为原始 label 从 1 开始)
        if self.config.get("gt_transform", True):
            label = label.astype(np.int32) - 1
            label[label < 0] = 255  # 无效区域设为 255
            label = label.astype(np.uint8)
        
        # 预处理
        if self.preprocess is not None:
            rgb, label, depth = self.preprocess(rgb, label, depth)
        
        # 转为 Tensor
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        depth = torch.from_numpy(np.ascontiguousarray(depth)).float()
        
        return {
            "data": rgb,           # [3, H, W]
            "label": label,        # [H, W]
            "modal_x": depth,      # [3, H, W]
            "fn": rgb_path,
            "name": item_name,
        }


# ============== 公开 API ==============
def get_NYUv2_train_loader(
    dataset_root: str = None,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    scale_array: List[float] = None,
    config_path: str = None,
    augment: bool = True,
) -> DataLoader:
    """
    获取 NYUv2 训练数据加载器
    
    Args:
        dataset_root: 数据集根目录，None 则使用默认路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        scale_array: 随机缩放比例列表
        config_path: YAML 配置文件路径
        augment: 是否启用数据增强（翻转、缩放、裁剪）。False 则只做归一化，与 val_loader 对齐
    
    Returns:
        DataLoader
    """
    config = NYUV2_CONFIG.copy()
    
    # 从 YAML 读取配置
    if config_path:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            batch_size = yaml_config.get("batch_size", batch_size)
    
    if augment:
        preprocess = TrainPreprocess(
            norm_mean=config["norm_mean"],
            norm_std=config["norm_std"],
            depth_norm_mean=config["depth_norm_mean"],
            depth_norm_std=config["depth_norm_std"],
            image_height=config["image_height"],
            image_width=config["image_width"],
            scale_array=scale_array,
        )
    else:
        # 不做数据增强，只做归一化，与 val_loader 对齐
        preprocess = ValPreprocess(
            norm_mean=config["norm_mean"],
            norm_std=config["norm_std"],
            depth_norm_mean=config["depth_norm_mean"],
            depth_norm_std=config["depth_norm_std"],
        )
    
    dataset = NYUv2Dataset(
        dataset_root=dataset_root,
        split="train",
        preprocess=preprocess,
        config=config,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return loader


def get_NYUv2_val_loader(
    dataset_root: str = None,
    batch_size: int = 1,
    num_workers: int = 4,
    config_path: str = None,
) -> DataLoader:
    """
    获取 NYUv2 验证数据加载器
    
    Args:
        dataset_root: 数据集根目录，None 则使用默认路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        config_path: YAML 配置文件路径
    
    Returns:
        DataLoader
    """
    config = NYUV2_CONFIG.copy()
    
    # 从 YAML 读取配置
    if config_path:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            batch_size = yaml_config.get("batch_size", batch_size)
    
    preprocess = ValPreprocess(
        norm_mean=config["norm_mean"],
        norm_std=config["norm_std"],
        depth_norm_mean=config["depth_norm_mean"],
        depth_norm_std=config["depth_norm_std"],
    )
    
    dataset = NYUv2Dataset(
        dataset_root=dataset_root,
        split="val",
        preprocess=preprocess,
        config=config,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return loader


def get_NYUv2_dataset(
    dataset_root: str = None,
    split: str = "val",
    preprocess = None,
) -> NYUv2Dataset:
    """
    获取 NYUv2 数据集对象（不使用 DataLoader）
    
    Args:
        dataset_root: 数据集根目录
        split: "train" 或 "val"
        preprocess: 自定义预处理函数
    
    Returns:
        NYUv2Dataset
    """
    if preprocess is None:
        config = NYUV2_CONFIG
        preprocess = ValPreprocess(
            norm_mean=config["norm_mean"],
            norm_std=config["norm_std"],
            depth_norm_mean=config["depth_norm_mean"],
            depth_norm_std=config["depth_norm_std"],
        )
    
    return NYUv2Dataset(
        dataset_root=dataset_root,
        split=split,
        preprocess=preprocess,
    )


# ============== 导出 ==============
__all__ = [
    "get_NYUv2_train_loader",
    "get_NYUv2_val_loader",
    "get_NYUv2_dataset",
    "NYUv2Dataset",
    "TrainPreprocess",
    "ValPreprocess",
    "NYUV2_CONFIG",
    "DEFAULT_DATASET_DIR",
]
