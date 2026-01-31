"""
DFormer Model Loader - API Wrapper
封装原作者 DFormer repo，通过 YAML 配置文件加载模型

Usage:
    from get_model.DFormer import get_dformer, get_dformerv2, load_from_config
    
    # 方式1: 快速加载
    model = get_dformer(model_type="base")
    model = get_dformerv2(model_type="base")
    
    # 方式2: 从配置文件加载
    model = load_from_config("config/DFormer.yaml")
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict

# ============== 路径配置 ==============
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

# 添加原版 DFormer 到 Python path
_DFORMER_ROOT = os.path.join(_PROJECT_ROOT, "models", "DFormer")
if _DFORMER_ROOT not in sys.path:
    sys.path.insert(0, _DFORMER_ROOT)

# 默认路径
DEFAULT_WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, "model_weights", "dformer")
DEFAULT_DATASET_DIR = os.path.join(_PROJECT_ROOT, "datasets", "NYUDepthv2")
DEFAULT_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")

# 导入原版 DFormer
from models.builder import EncoderDecoder


# ============== 配置工具 ==============
def load_yaml_config(yaml_path: str) -> edict:
    """加载 YAML 配置文件"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


def get_model_config(
    backbone: str = "DFormer-Base",
    decoder: str = "ham",
    decoder_embed_dim: int = 512,
    num_classes: int = 40,
    drop_path_rate: float = 0.1,
    **kwargs
) -> edict:
    """
    创建模型配置
    
    Args:
        backbone: 骨干网络 
            - DFormer v1: "DFormer-Tiny", "DFormer-Small", "DFormer-Base", "DFormer-Large"
            - DFormer v2: "DFormerv2_S", "DFormerv2_B", "DFormerv2_L"
        decoder: 解码器类型 ("ham", "MLPDecoder")
        decoder_embed_dim: 解码器嵌入维度
        num_classes: 类别数量
        drop_path_rate: DropPath 比率
    """
    C = edict()
    C.backbone = backbone
    C.decoder = decoder
    C.decoder_embed_dim = decoder_embed_dim
    C.num_classes = num_classes
    C.bn_eps = 1e-3
    C.bn_momentum = 0.1
    C.drop_path_rate = drop_path_rate
    C.aux_rate = 0
    C.background = 255
    C.pretrained_model = None
    return C


# ============== Backbone 映射表 ==============
DFORMER_V1_BACKBONES = {
    "tiny": "DFormer-Tiny",
    "small": "DFormer-Small",
    "base": "DFormer-Base",
    "large": "DFormer-Large",
}

DFORMER_V2_BACKBONES = {
    "small": "DFormerv2_S",
    "base": "DFormerv2_B",
    "large": "DFormerv2_L",
}

# 权重文件名映射
WEIGHT_FILES = {
    # DFormer v1
    "DFormer-Tiny": "NYUv2_DFormer_Tiny.pth",
    "DFormer-Small": "NYUv2_DFormer_Small.pth",
    "DFormer-Base": "NYUv2_DFormer_Base.pth",
    "DFormer-Large": "NYUv2_DFormer_Large.pth",
    # DFormer v2
    "DFormerv2_S": "DFormerv2_Small_NYU.pth",
    "DFormerv2_B": "DFormerv2_Base_NYU.pth",
    "DFormerv2_L": "DFormerv2_Large_NYU.pth",
}


# ============== 核心加载函数 ==============
def _load_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> nn.Module:
    """加载权重到模型"""
    if not os.path.exists(checkpoint_path):
        print(f"[DFormer] Warning: Checkpoint not found at {checkpoint_path}")
        return model
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 处理不同的 checkpoint 格式
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # 移除 'module.' 前缀 (如果有)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    print(f"[DFormer] Loaded checkpoint: {checkpoint_path}")
    
    return model


def _build_model(
    backbone: str,
    checkpoint_path: str = None,
    num_classes: int = 40,
    decoder: str = "ham",
    decoder_embed_dim: int = 512,
    device: str = None,
    freeze: bool = True,
) -> nn.Module:
    """
    构建并加载 DFormer 模型
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建配置
    config = get_model_config(
        backbone=backbone,
        decoder=decoder,
        decoder_embed_dim=decoder_embed_dim,
        num_classes=num_classes,
    )
    
    # 构建模型 (criterion=None 跳过自动权重初始化)
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    
    # 加载权重
    if checkpoint_path:
        # 处理相对路径
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(_PROJECT_ROOT, checkpoint_path)
        model = _load_checkpoint(model, checkpoint_path, device)
    
    model.to(device)
    model.eval()
    
    # 冻结参数
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    
    cudnn.benchmark = True
    
    return model


# ============== 公开 API ==============
def get_dformer(
    model_type: str = "base",
    checkpoint_path: str = None,
    num_classes: int = 40,
    device: str = None,
    freeze: bool = True,
) -> nn.Module:
    """
    加载 DFormer v1 模型
    
    Args:
        model_type: 模型类型 ("tiny", "small", "base", "large")
        checkpoint_path: 权重路径，None 则使用默认路径
        num_classes: 分类数量
        device: 设备，None 则自动检测
        freeze: 是否冻结参数
    
    Returns:
        加载好的模型
    
    Example:
        model = get_dformer("base")
        output = model(rgb, depth)  # [B, 40, H, W]
    """
    model_type = model_type.lower()
    if model_type not in DFORMER_V1_BACKBONES:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(DFORMER_V1_BACKBONES.keys())}")
    
    backbone = DFORMER_V1_BACKBONES[model_type]
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(DEFAULT_WEIGHTS_DIR, WEIGHT_FILES[backbone])
    
    return _build_model(
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
        freeze=freeze,
    )


def get_dformerv2(
    model_type: str = "base",
    checkpoint_path: str = None,
    num_classes: int = 40,
    device: str = None,
    freeze: bool = True,
) -> nn.Module:
    """
    加载 DFormer v2 模型
    
    Args:
        model_type: 模型类型 ("small", "base", "large")
        checkpoint_path: 权重路径，None 则使用默认路径
        num_classes: 分类数量
        device: 设备，None 则自动检测
        freeze: 是否冻结参数
    
    Returns:
        加载好的模型
    
    Example:
        model = get_dformerv2("base")
        output = model(rgb, depth)  # [B, 40, H, W]
    """
    model_type = model_type.lower()
    if model_type not in DFORMER_V2_BACKBONES:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(DFORMER_V2_BACKBONES.keys())}")
    
    backbone = DFORMER_V2_BACKBONES[model_type]
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(DEFAULT_WEIGHTS_DIR, WEIGHT_FILES[backbone])
    
    return _build_model(
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
        freeze=freeze,
    )


def load_from_config(config_path: str, device: str = None) -> nn.Module:
    """
    从 YAML 配置文件加载模型
    
    Args:
        config_path: YAML 配置文件路径
        device: 设备
    
    Returns:
        加载好的模型
    
    Example:
        model = load_from_config("config/DFormer.yaml")
    """
    # 处理相对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    
    cfg = load_yaml_config(config_path)
    
    # 确定 backbone
    model_type = cfg.get("model_type", "base")
    version = cfg.get("version", "v1")  # v1 或 v2
    
    if version == "v2":
        backbone = DFORMER_V2_BACKBONES.get(model_type, "DFormerv2_B")
    else:
        backbone = DFORMER_V1_BACKBONES.get(model_type, "DFormer-Base")
    
    # 也支持直接指定 backbone
    if "backbone" in cfg:
        backbone = cfg.backbone
    
    return _build_model(
        backbone=backbone,
        checkpoint_path=cfg.get("checkpoint_path"),
        num_classes=cfg.get("num_classes", 40),
        decoder=cfg.get("decoder", "ham"),
        decoder_embed_dim=cfg.get("decoder_embed_dim", 512),
        device=device or cfg.get("device", "cuda"),
        freeze=cfg.get("freeze", True),
    )


# ============== 数据集路径工具 ==============
def get_dataset_paths(dataset_root: str = None) -> dict:
    """
    获取 NYUDepthv2 数据集路径
    
    Returns:
        {
            "rgb_root": "datasets/NYUDepthv2/RGB",
            "depth_root": "datasets/NYUDepthv2/Depth",
            "label_root": "datasets/NYUDepthv2/Label",
            "train_list": "datasets/NYUDepthv2/train.txt",
            "test_list": "datasets/NYUDepthv2/test.txt",
        }
    """
    if dataset_root is None:
        dataset_root = DEFAULT_DATASET_DIR
    
    return {
        "dataset_root": dataset_root,
        "rgb_root": os.path.join(dataset_root, "RGB"),
        "depth_root": os.path.join(dataset_root, "Depth"),
        "label_root": os.path.join(dataset_root, "Label"),
        "train_list": os.path.join(dataset_root, "train.txt"),
        "test_list": os.path.join(dataset_root, "test.txt"),
    }


# ============== 导出 ==============
__all__ = [
    # 模型加载
    "get_dformer",
    "get_dformerv2",
    "load_from_config",
    # 配置工具
    "get_model_config",
    "load_yaml_config",
    # 路径工具
    "get_dataset_paths",
    "DEFAULT_WEIGHTS_DIR",
    "DEFAULT_DATASET_DIR",
    # Backbone 映射
    "DFORMER_V1_BACKBONES",
    "DFORMER_V2_BACKBONES",
]
