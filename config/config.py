from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Config:
    """attack configuration"""
    
    # ---------- 模型版本 ----------
    version: str = "v1"                    # v1 (DFormer) 或 v2 (DFormerv2)
    model_type: str = "base"               # tiny, small, base, large
    backbone: Optional[str] = None         # 直接指定 backbone (优先级高于 version + model_type)
    
    # ---------- 权重文件 ----------
    checkpoint_path: str = "model_weights/dformer/NYUv2_DFormer_Base.pth"
    
    # ---------- 模型参数 ----------
    num_classes: int = 40                  # NYUv2: 40 类
    decoder: str = "ham"                   # ham, MLPDecoder
    decoder_embed_dim: int = 512
    
    # ---------- 推理设置 ----------
    device: str = "cuda"
    freeze: bool = True                    # 是否冻结参数
    
    # ---------- 对抗攻击参数 ----------
    lambda_print: float = 0.0              # 可打印性损失权重
    lambda_smooth: float = 0.0             # 平滑损失权重
    learning_rate: float = 0.01            # 优化器学习率
    attack_epochs: int = 2                 # 攻击迭代次数
    
    # ---------- 数据集参数 ----------
    ignore_label: int = 255                # 忽略的标签值
    
    # ---------- 保存参数 ----------
    save_path: str = "output/test"         # 结果保存路径
    
    # ---------- 训练设置----------
    epochs: int = 100
    batch_size: int = 8


