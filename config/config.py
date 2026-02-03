from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Config:
    """attack configuration"""
    
    version: str = "v1"                    
    model_type: str = "base"               
    backbone: Optional[str] = None         
    
    checkpoint_path: str = "model_weights/dformer/NYUv2_DFormer_Base.pth"
    save_path: str = "output/test"         
    log_path : str = "./logs"
    
    num_classes: int = 40                  
    decoder: str = "ham"                   
    decoder_embed_dim: int = 512
    
    device: str = "cuda"
    freeze: bool = True                    
    
    learning_rate: float = 0.01            
    
    ignore_label: int = 255                
    
    
    epochs: int = 100
    batch_size: int = 8
    pad : int = 30

    momentum: float = 0.9
    weight_decay: float = 0.0
    warmup_steps: int = 1000


