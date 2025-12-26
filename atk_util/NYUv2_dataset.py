import argparse 
import yaml 
from importlib import import_module

from models.DFormer.utils.dataloader.dataloader import  get_val_loader
from models.DFormer.utils.dataloader.RGBXDataset import RGBXDataset
from models.DFormer.utils.engine.engine import Engine

import os
import sys

# ===== 模拟 torchrun 设置的环境变量 =====
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "29957"

def create_attack_parser():
    """创建包含 attack.sh 配置的 ArgumentParser"""
    parser = argparse.ArgumentParser()
    
    # attack.sh 中的参数值
    parser.add_argument("--save_path", default="output/")
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--config", default="local_configs.NYUDepthv2.DFormer_Base")
    parser.add_argument("--continue_fpath", default="models/DFormer/checkpoints/trained/NYUv2_DFormer_Base.pth")
    
    # 其他可能需要的参数
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--epochs", default=0)
    parser.add_argument("--show_image", "-s", default=False, action="store_true")
    parser.add_argument("--checkpoint_dir", default=None)
    
    return parser

def get_NYUv2_val_loader():

    parser = create_attack_parser()

    with Engine(custom_parser=parser) as engine:
        # with open("config/DFormer.yaml", "r", encoding="utf-8") as f:
        #     config_yaml = yaml.safe_load(f)
        # config = getattr(import_module(args.config), "C")

        args = parser.parse_args()
        config = getattr(import_module(args.config), "C")
        config.pad = False  # Do not pad when inference
        if "x_modal" not in config:
            config["x_modal"] = "d"
        
                # ===== 修复路径问题：将相对路径转换为绝对路径 =====
        # 获取 DFormer 模型所在目录的绝对路径
        dformer_dir = os.path.join(os.getcwd(), "models/DFormer")
        
        # 如果 root_dir 是相对路径，将其转换为基于 DFormer 目录的绝对路径
        if hasattr(config, 'root_dir') and not os.path.isabs(config.root_dir):
            config.root_dir = os.path.join(dformer_dir, config.root_dir)
        
        # 更新所有依赖 root_dir 的路径
        config.dataset_path = os.path.join(config.root_dir, os.path.basename(config.dataset_path))
        config.rgb_root_folder = os.path.join(config.dataset_path, "RGB")
        config.gt_root_folder = os.path.join(config.dataset_path, "Label")
        config.x_root_folder = os.path.join(config.dataset_path, "Depth")
        config.train_source = os.path.join(config.dataset_path, "train.txt")
        config.eval_source = os.path.join(config.dataset_path, "test.txt")
        
        print(f"Dataset path: {config.dataset_path}")
        print(f"Eval source: {config.eval_source}")


        val_loader, val_sampler = get_val_loader(engine, RGBXDataset, config, 1)

    return val_loader