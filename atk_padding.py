from distutils.command import clean
from triton.tools.compile import desc
import torch
import triton
import time
import argparse
import yaml
from pathlib import Path

from tqdm import tqdm
from config.config import Config
from atk_utils.metrics import get_mIoU_sklearn 
from atk_utils.DFormer import get_dformer
from atk_utils.DFormer import get_dformerv2
from atk_utils.NYUv2_dataset import get_NYUv2_val_loader , get_NYUv2_train_loader 
from atk_utils.atk_loss_vanilla import Loss_Manager, LossManagerPadding
from atk_utils.NYUv2_img_with_patch import PatchGeneratorPadding
from atk_utils.save_tool import save_all_results , save_mIoU_log , get_4_logits
from atk_utils.metrics import StreamingMIoU
from torch.cuda.amp import autocast

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader 

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter


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
    NYUv2_loader: DataLoader = get_NYUv2_train_loader(
        batch_size=config.batch_size, num_workers=8, augment=False
    )
    sample = next(iter(NYUv2_loader))
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

    total_steps = len(NYUv2_loader) * config.epochs

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
        NYUv2_loader,
        patch_gen,
        loss_mgr,
        optimizer,
        scheduler,
        clean_metrics,
        adv_metrics,
        clean_wo_depth_metrics,
        adv_wo_depth_metrics,
    )

def atk(config : Config , writer : SummaryWriter):
    (
        model,
        NYUv2_loader ,
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
    dataset_len: int = len(NYUv2_loader)

    for train_epoch in tqdm(range(config.epochs), desc="epochs", leave=True):
        for idx, minibatch in enumerate(tqdm(NYUv2_loader, desc="batch_id", leave=False)):

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

            step :int =  train_epoch * dataset_len + idx
            if step % 10 == 0:
                writer.add_scalar("train/loss" , loss.item(), step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                tqdm.write("tensorboard scalar updated")

        # 每 10 个 epoch 保存一次 checkpoint
        if train_epoch % 10 == 0 or train_epoch == config.epochs - 1:
            
            path : Path= Path(config.save_path)
            path.mkdir(parents=True, exist_ok=True)

            save_path: Path = path / f"patch_epoch{train_epoch}.pt"
            torch.save(patch_gen.patch.detach().cpu(), save_path)
            tqdm.write(f"Patch saved to {save_path}")


        # with torch.no_grad():
        #     img : Tensor = loss_mgr.patch_gen.img
        #     modal_xs : Tensor = loss_mgr.patch_gen.modal_xs
        #     label : Tensor = loss_mgr.patch_gen.label

        #     img_adv: Tensor = loss_mgr.patch_gen()
        #     modal_xs_padded: Tensor = loss_mgr.patch_gen.modal_xs_padded
        #     label_padded :Tensor = loss_mgr.patch_gen.label_padded

        #     logits_clean, logits_adv, logits_clean_no_depth, logits_adv_no_depth = (
        #         get_4_logits(model, img, img_adv, modal_xs, modal_xs_padded)
        #     )

        #     mask_no_pad :Tensor= torch.zeros_like(label)
        #     mask_padded : Tensor = loss_mgr.patch_gen.mask

        #     clean_metrics.update(logits_clean , label , mask_no_pad)
        #     clean_wo_depth_metrics.update(logits_clean_no_depth, label, mask_no_pad)

        #     adv_metrics.update(logits_adv, label_padded, mask_padded)
        #     adv_wo_depth_metrics.update(logits_adv_no_depth, label_padded, mask_padded)

    writer.close()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    config : Config = boot_args()

    ts : str = time.strftime("%Y%m%d-%H%M%S")
    writer : SummaryWriter = SummaryWriter(log_dir=f"{config.log_path}/{ts}" , flush_secs=20)
    atk(config, writer)



