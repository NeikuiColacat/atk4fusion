from fontTools.misc.symfont import C
import torch
import triton
import argparse
import yaml

from config.config import Config
from atk_utils.metrics import get_mIoU_sklearn 
from atk_utils.DFormer import get_dformer
from atk_utils.DFormer import get_dformerv2
from atk_utils.NYUv2_dataset import get_NYUv2_val_loader 
from atk_utils.atk_loss_vanilla import Loss_Manager
from atk_utils.NYUv2_img_with_patch import PatchGenerator
from atk_utils.save_tool import save_all_results , save_mIoU_log , get_4_logits
from atk_utils.metrics import StreamingMIoU
from torch.cuda.amp import autocast

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader 

def print_mIoU_single_batch(
    idx: int,
    labels: torch.Tensor,
    mask: torch.Tensor,
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    logits_clean_no_depth: torch.Tensor,
    logits_adv_no_depth: torch.Tensor,
):
    """
    计算并打印单个batch的4种mIoU，返回mIoU值用于累积
    
    Returns:
        (mIoU_clean, mIoU_adv, mIoU_clean_no_depth, mIoU_adv_no_depth)
    """

    print(f"Batch_{idx}-----------------------------------------")

    _, mIoU_clean = get_mIoU_sklearn(logits_clean, labels, mask)
    print(f"Batch {idx} - clean mIoU: {mIoU_clean:.4f}")

    _, mIoU_adv = get_mIoU_sklearn(logits_adv, labels, mask)
    print(f"Batch {idx} - adv mIoU: {mIoU_adv:.4f}")

    _, mIoU_clean_no_depth = get_mIoU_sklearn(logits_clean_no_depth, labels, mask)
    print(f"Batch {idx} - clean mIoU no depth: {mIoU_clean_no_depth:.4f}")

    _, mIoU_adv_no_depth = get_mIoU_sklearn(logits_adv_no_depth, labels, mask)
    print(f"Batch {idx} - adv mIoU no depth: {mIoU_adv_no_depth:.4f}")

    print("-----------------------------------------------------")

    return mIoU_clean, mIoU_adv, mIoU_clean_no_depth, mIoU_adv_no_depth


def _mem(tag: str):
    """Print CUDA memory usage at a specific step."""


    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()

    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    max_reserv = torch.cuda.max_memory_reserved() / 1024**2
    print(f"[CUDA MEM] {tag:>18s} | alloc={alloc:8.1f}MB reserv={reserv:8.1f}MB "
          f"max_alloc={max_alloc:8.1f}MB max_reserv={max_reserv:8.1f}MB")

    torch.cuda.reset_peak_memory_stats()

def boot_args() -> Config:
    parser = argparse.ArgumentParser(description="boot config path")
    parser.add_argument(
        "--config",
        type=str,
        default="config/DFormer.yaml",
        help="path to config yaml file",
    )

    config_path:str= parser.parse_args().config
    with open(config_path , "r") as f:
        config: dict = yaml.safe_load(f)

    return Config(**config) 

def atk():
    config : Config = boot_args()
    torch.set_float32_matmul_precision('high')

    model: Module = get_dformer()
    # model = torch.compile(model)
    val_loader: DataLoader = get_NYUv2_val_loader(batch_size=config.batch_size) 

    for p in model.parameters():
        p.requires_grad = False

    device: str = config.device

    all_mIoU_clean: list = []
    all_mIoU_adv: list = []
    all_mIoU_clean_no_depth: list = []
    all_mIoU_adv_no_depth: list = []

    metrics: StreamingMIoU = StreamingMIoU(40)

    for idx , minibatch in enumerate(val_loader):
        images: Tensor = minibatch["data"].to(device)
        labels: Tensor = minibatch["label"].to(device)
        modal_xs: Tensor = minibatch["modal_x"].to(device)

        scales = [0.5, 0.75, 1.0, 1.25]
        import torch.nn.functional as F
        for s in scales:
            images_s = F.interpolate(images, scale_factor=s, mode="bilinear", align_corners=False)
            depth_s = F.interpolate(modal_xs, scale_factor=s, mode="bilinear", align_corners=False)
            labels_s = F.interpolate(labels.unsqueeze(1).float(), scale_factor=s, mode="nearest").squeeze(1).long()


            logits: Tensor = model(images_s, depth_s)
            metrics.update(logits,labels_s, torch.zeros_like(labels_s))

        print("done")
        continue
        patch_gen = PatchGenerator(images) 
        loss_mgr = Loss_Manager(images, modal_xs, labels, patch_gen , model , lambda_print=0, lambda_smooth=0)
        optimizer = torch.optim.Adam([loss_mgr.patch_gen.patch] , lr=0.01)

        for train_epoch in range(config.epochs) :

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                loss = loss_mgr()
            loss.backward()

            if train_epoch % 10 == 0:
                print(loss.detach().cpu().item())
            
            optimizer.step()
        
        # save adv result 
        with torch.no_grad():
            img_adv = loss_mgr.patch_gen()
            mask = loss_mgr.patch_gen.mask

        logits_clean, logits_adv, logits_clean_no_depth, logits_adv_no_depth = (
            get_4_logits(model, images, img_adv, modal_xs)
        )

        mIoU_clean, mIoU_adv, mIoU_clean_no_depth, mIoU_adv_no_depth = (
            print_mIoU_single_batch(
                idx,
                labels,
                mask,
                logits_clean,
                logits_adv,
                logits_clean_no_depth,
                logits_adv_no_depth,
            )
        )

        all_mIoU_clean.append(mIoU_clean)
        all_mIoU_adv.append(mIoU_adv)
        all_mIoU_clean_no_depth.append(mIoU_clean_no_depth)
        all_mIoU_adv_no_depth.append(mIoU_adv_no_depth)

        save_path = "output/test"

        save_all_results(
            images,
            img_adv,
            idx,
            save_path,
            logits_clean,
            logits_adv,
            logits_clean_no_depth,
            logits_adv_no_depth,
            labels,
        )

        
        if idx % 1 == 0 or idx == len(val_loader) - 1 :
            # 计算全局平均 mIoU
            avg_mIoU_clean = sum(all_mIoU_clean) / len(all_mIoU_clean)
            avg_mIoU_adv = sum(all_mIoU_adv) / len(all_mIoU_adv)
            avg_mIoU_clean_no_depth = sum(all_mIoU_clean_no_depth) / len(all_mIoU_clean_no_depth)
            avg_mIoU_adv_no_depth = sum(all_mIoU_adv_no_depth) / len(all_mIoU_adv_no_depth)

            # 使用 save_mIoU_log 保存全局平均 mIoU
            save_mIoU_log(
                avg_mIoU_clean,
                avg_mIoU_adv,
                avg_mIoU_clean_no_depth,
                avg_mIoU_adv_no_depth,
                log_path=save_path + "/mIoU_average.txt"
            )
        
    print(metrics.compute())

if __name__ == "__main__":
    atk()



