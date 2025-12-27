import sys
sys.path.append("models/DFormer")

from importlib import import_module
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from models.DFormer.models.builder import EncoderDecoder as segmodel
import yaml


def get_dformer() :

    with open("config/DFormer.yaml", "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    
    config = getattr(import_module(config_yaml["config"]), "C")
    config.pad = False  # Do not pad when inference
    if "x_modal" not in config:
        config["x_modal"] = "d"
    cudnn.benchmark = True

    model = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)
    # weight = torch.load(args.continue_fpath)['state_dict']
    weight = torch.load(config_yaml["continue_fpath"])["model"]
    model.load_state_dict(weight, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def get_dformerv2() :

    with open("config/DFormerv2.yaml", "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    
    config = getattr(import_module(config_yaml["config"]), "C")
    config.pad = False  # Do not pad when inference
    if "x_modal" not in config:
        config["x_modal"] = "d"
    cudnn.benchmark = True

    model = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)
    weight = torch.load(config_yaml["continue_fpath"])['state_dict']
    # weight = torch.load(config_yaml["continue_fpath"])["model"]
    model.load_state_dict(weight, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model

