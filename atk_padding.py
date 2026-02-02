import torch
import triton
import argparse
import yaml

from config.config import Config
from atk_utils.metrics import get_mIoU_sklearn 
from atk_utils.DFormer import get_dformer
from atk_utils.DFormer import get_dformerv2
from atk_utils.NYUv2_dataset import get_NYUv2_val_loader 
from atk_utils.atk_loss_vanilla import Loss_Manager, LossManagerPadding
from atk_utils.NYUv2_img_with_patch import PatchGeneratorPadding
from atk_utils.save_tool import save_all_results , save_mIoU_log , get_4_logits
from atk_utils.metrics import StreamingMIoU
from torch.cuda.amp import autocast

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader 

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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

def train_utils_builder(config: Config) -> tuple[
    Module,
    DataLoader,
    PatchGeneratorPadding,
    LossManagerPadding,
    torch.optim.Optimizer,
    SequentialLR,
    StreamingMIoU,
    StreamingMIoU,
    StreamingMIoU,
    StreamingMIoU,
]:
    val_loader: DataLoader = get_NYUv2_val_loader(batch_size=config.batch_size)
    sample = next(iter(val_loader))
    device = config.device
    
    model : Module = get_dformer()

    images: Tensor = sample["data"].to(device)
    labels: Tensor = sample["label"].to(device)
    modal_xs: Tensor = sample["modal_x"].to(device)

    patch_gen: PatchGeneratorPadding = PatchGeneratorPadding(
        config.pad, images, labels, modal_xs
    )

    loss_mgr = LossManagerPadding(patch_gen , model)

    ############################### optimizer strategy
    optimizer = torch.optim.Adam(
        [loss_mgr.patch_gen.patch],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(val_loader) * config.epochs

    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps
    )

    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - config.warmup_steps,  # 剩下的 epoch 走余弦
        eta_min=1e-6,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[config.warmup_steps],  # 在第 10 个 epoch 切换调度器
    )
    ####################################

    clean_metrics: StreamingMIoU = StreamingMIoU(40)
    adv_metrics: StreamingMIoU = StreamingMIoU(40)
    clean_wo_depth_metrics: StreamingMIoU = StreamingMIoU(40)
    adv_wo_depth_metrics: StreamingMIoU = StreamingMIoU(40)


    return (
        model,
        val_loader,
        patch_gen,
        loss_mgr,
        optimizer,
        scheduler,
        clean_metrics,
        adv_metrics,
        clean_wo_depth_metrics,
        adv_wo_depth_metrics,
    )

def atk(config : Config):
    (
        model,
        val_loader ,
        patch_gen,
        loss_mgr,
        optimizer,
        scheduler,
        clean_metrics,
        adv_metrics,
        clean_wo_depth_metrics,
        adv_wo_depth_metrics,
    ) = train_utils_builder(config)

    # model = torch.compile(model)

    device: str = config.device
    for train_epoch in range(config.epochs) :

        for idx , minibatch in enumerate(val_loader):
            images: Tensor = minibatch["data"].to(device)
            labels: Tensor = minibatch["label"].to(device)
            modal_xs: Tensor = minibatch["modal_x"].to(device)

            optimizer.zero_grad()
            patch_gen.update_batch(images, labels, modal_xs)

            with autocast(dtype=torch.bfloat16):
                loss: Tensor = loss_mgr()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if train_epoch % 5 == 0:
            print(loss.item())

            
            # with torch.no_grad():
            #     img_adv = loss_mgr.patch_gen()
            #     mask = loss_mgr.patch_gen.mask

            # logits_clean, logits_adv, logits_clean_no_depth, logits_adv_no_depth = (
            #     get_4_logits(model, images, img_adv, modal_xs)
            # )

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    config : Config = boot_args()
    atk(config)



