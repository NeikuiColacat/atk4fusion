import argparse
import importlib
import os
import random
import sys
import time
from importlib import import_module

from matplotlib import pyplot as plt
import pathlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import ValPre, get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import group_weight, init_weight
from utils.lr_policy import WarmUpPolyLR
from utils.metric import compute_score, hist_info
from utils.pyt_utils import (
    all_reduce_tensor,
    ensure_dir,
    link_file,
    load_model,
    parse_devices,
)
from utils.val_mm import evaluate, evaluate_msf
from utils.visualize import print_iou, show_img
from utils.metrics_new import Metrics



def de_norm(img : torch.Tensor, mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225]):

    mean = torch.tensor(mean, device=img.device, dtype=img.dtype).reshape(-1 , 3 , 1 , 1)
    std = torch.tensor(std, device=img.device, dtype=img.dtype).reshape(-1 , 3 , 1, 1)
    img = img * std + mean
    return img.clamp(0, 1)

def save_rgb(img_without_norm : torch.Tensor, file_name : str, save_path = "output/sample_img"):

    if img_without_norm.shape[0] == 1 :
        img_without_norm = img_without_norm.squeeze(dim=0)

    assert img_without_norm.dim() == 3

    save_path = save_path + "/" + file_name

    img = img_without_norm.permute(1,2,0).cpu().numpy()  # [H,W,3]
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(save_path, img)

def save_adv_results(model, img_adv, modal_xs, labels, palette, file_prefix="sample", save_path="output/adv_results"):
    """
    保存带对抗 patch 的图像和它的分割结果
    Args:
        model: 分割模型
        img_adv: 带 patch 的图像 (B,C,H,W)
        modal_xs: 深度或其他模态输入
        labels: 标签 (B,H,W)，这里只是为了对齐尺寸
        palette: 调色板 (numpy array)，形状 [num_classes, 3]
        file_prefix: 文件名前缀
        save_path: 保存路径
    """
    # 1. 保存带 patch 的输入图像
    img_adv_vis = de_norm(img_adv).detach().cpu()
    save_rgb(img_adv_vis, f"{file_prefix}_adv_input.png", save_path)

    # 2. 计算分割结果
    logits = model(img_adv, modal_xs).softmax(dim=1)
    preds = logits.argmax(dim=1).cpu().squeeze().numpy().astype(np.uint8)

    # 3. 调色板映射
    preds_color = palette[preds]

    # 4. 保存分割结果
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.imsave(f"{save_path}/{file_prefix}_adv_pred.png", preds_color)




         


    
def infer_single_card_mode():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--gpus", help="used gpu number")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--epochs", default=0)
    parser.add_argument("--show_image", "-s", default=False, action="store_true")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--continue_fpath")

    with Engine(custom_parser=parser) as engine:

        args = parser.parse_args()
        config = getattr(import_module(args.config), "C")
        config.pad = False  # Do not pad when inference
        if "x_modal" not in config:
            config["x_modal"] = "d"
        cudnn.benchmark = True

        val_loader, val_sampler = get_val_loader(engine, RGBXDataset, config, int(args.gpus))
        print(len(val_loader))

        model = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)
        # weight = torch.load(args.continue_fpath)['state_dict']

        weight = torch.load(args.continue_fpath)["model"]
        model.load_state_dict(weight, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        save_dir=args.save_path

        model.eval()
        device = torch.device("cuda")
        n_classes = config.num_classes
        metrics = Metrics(n_classes, config.background, device)
        palette = get_palette()

        ###############################modify
        for p in model.parameters():
            p.requires_grad = False

        clean_tt = 0
        adv_tt = 0
        adv_dep_none = 0
        clean_dep_none = 0
        for idx , minibatch in enumerate(val_loader):

            images = minibatch["data"].to(device)
            labels = minibatch["label"].to(device)
            modal_xs = minibatch["modal_x"].to(device)

            loss_mgr = Loss_Manager(images, modal_xs, labels, model)
            optimizer = torch.optim.Adam([loss_mgr.patch_gen.patch] , lr=0.001)

            for train_epoch in range(300) :
                optimizer.zero_grad()
                loss = loss_mgr()

                # losses = loss_mgr()
                # adv_loss = losses["adv_loss"]
                # cons_loss = losses["cons_loss"]
                # patch = loss_mgr.patch_gen.patch
                
                # # 分别算梯度
                # g_adv = torch.autograd.grad(adv_loss, patch, retain_graph=True)[0]
                # g_cons = torch.autograd.grad(cons_loss, patch, retain_graph=True)[0]
                
                # # 投影去除冲突分量
                # def dot(a, b):
                #     return (a * b).sum()
                
                # def proj_remove(a, b):
                #     bn2 = (b * b).sum().clamp_min(1e-12)
                #     scale = dot(a, b) / bn2
                #     return a - scale * b
                
                # if dot(g_adv, g_cons) < 0:
                #     g_cons = proj_remove(g_cons, g_adv)
                
                # # 合并梯度更新（200 是一致性权重）
                # g_total = g_adv + 200 * g_cons
                # patch.grad = g_total
                # optimizer.step()

                loss.backward()
                if train_epoch % 20 == 0:
                    print(loss.item())

            img_adv = loss_mgr.patch_gen()
            mask = loss_mgr.patch_gen.mask.detach().cpu()

            _, miou = compute_sample_iou(
                model(images, modal_xs), labels, n_classes, 255, mask 
            )
            print("clean_with_depth", miou)
            clean_tt += miou

            img_adv = loss_mgr.patch_gen()
            _, miou = compute_sample_iou(
                model(img_adv, modal_xs), labels, n_classes, 255, mask
            )
            print("adv_with_depth", miou)
            adv_tt += miou

            depth_none = torch.full_like(modal_xs, modal_xs.mean())
            _, miou = compute_sample_iou(
                model(images, depth_none), labels, n_classes, 255, mask
            )
            print("clean_withou_depth", miou)
            clean_dep_none += miou

            _, miou = compute_sample_iou(
                model(img_adv, depth_none), labels, n_classes, 255, mask
            )
            print("adv_without_depth", miou)
            adv_dep_none += miou


            # print("-------------------\n") 
            # print("mean clean : ", clean_tt / (idx + 1))
            # print("mean adv: ", adv_tt / (idx + 1))
            # print("mean clean dep none: ", clean_dep_none / (idx + 1))
            # print("mean adv dep none : ", adv_dep_none / (idx + 1))
            # print("-------------------\n") 
            save_adv_results(model, img_adv, modal_xs, labels, palette, file_prefix=f"Dformerv1{idx}" , save_path="output/adv_result")



if __name__ == "__main__":
    infer_single_card_mode()



