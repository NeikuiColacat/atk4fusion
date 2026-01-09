import torch
import atk_util.atk_loss_vanilla as atk_util 

from get_model.DFormer import get_dformer
from get_model.DFormer import get_dformerv2
from atk_util.NYUv2_dataset import get_NYUv2_val_loader 
from atk_util.atk_loss_vanilla import Loss_Manager
from atk_util.NYUv2_img_with_patch import PatchGenerator
from atk_util.save_tool import save_all_results , save_mIoU_log



def print_mIoU_single_batch(model, images, img_adv, modal_xs, mean_depth, labels, mask, idx):
    """
    计算并打印单个batch的4种mIoU，返回mIoU值用于累积
    
    Returns:
        (mIoU_clean, mIoU_adv, mIoU_clean_no_depth, mIoU_adv_no_depth)
    """
    with torch.no_grad():
        logits_clean = model(images, modal_xs)
        logits_adv = model(img_adv, modal_xs)
        logits_clean_no_depth = model(images, mean_depth)
        logits_adv_no_depth = model(img_adv, mean_depth)

    print(f"Batch_{idx}-----------------------------------------")

    _, mIoU_clean = atk_util.get_mIoU_sklearn(logits_clean, labels, mask)
    print(f"Batch {idx} - clean mIoU: {mIoU_clean:.4f}")

    _, mIoU_adv = atk_util.get_mIoU_sklearn(logits_adv, labels, mask)
    print(f"Batch {idx} - adv mIoU: {mIoU_adv:.4f}")

    _, mIoU_clean_no_depth = atk_util.get_mIoU_sklearn(logits_clean_no_depth, labels, mask)
    print(f"Batch {idx} - clean mIoU no depth: {mIoU_clean_no_depth:.4f}")

    _, mIoU_adv_no_depth = atk_util.get_mIoU_sklearn(logits_adv_no_depth, labels, mask)
    print(f"Batch {idx} - adv mIoU no depth: {mIoU_adv_no_depth:.4f}")

    print("-----------------------------------------------------")

    return mIoU_clean, mIoU_adv, mIoU_clean_no_depth, mIoU_adv_no_depth

def atk():
    # model = get_dformerv2()
    model = get_dformer()
    val_loader = get_NYUv2_val_loader() 

    for p in model.parameters():
        p.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_mIoU_clean = []
    all_mIoU_adv = []
    all_mIoU_clean_no_depth = []
    all_mIoU_adv_no_depth = []

    for idx , minibatch in enumerate(val_loader):

        images = minibatch["data"].to(device)
        labels = minibatch["label"].to(device)
        modal_xs = minibatch["modal_x"].to(device)

        patch_gen = PatchGenerator(images) 
        loss_mgr = Loss_Manager(images, modal_xs, labels, patch_gen , model)
        optimizer = torch.optim.Adam([loss_mgr.patch_gen.patch] , lr=0.01)

        for train_epoch in range(50) :
            optimizer.zero_grad()
            loss = loss_mgr()
            loss.backward()

            if train_epoch % 10 == 0:
                print(loss.item())
            
            optimizer.step()
        

        img_adv = loss_mgr.patch_gen()
        mean_depth = torch.full_like(modal_xs, modal_xs.mean().item())
        mask = loss_mgr.patch_gen.mask

        mIoU_clean, mIoU_adv, mIoU_clean_no_depth, mIoU_adv_no_depth = (
            print_mIoU_single_batch(
                model, images, img_adv, modal_xs, mean_depth, labels, mask, idx
            )
        )

        all_mIoU_clean.append(mIoU_clean)
        all_mIoU_adv.append(mIoU_adv)
        all_mIoU_clean_no_depth.append(mIoU_clean_no_depth)
        all_mIoU_adv_no_depth.append(mIoU_adv_no_depth)

        save_all_results(
            model=model,
            images=images,
            img_adv=img_adv,
            modal_xs=modal_xs,
            mean_depth=mean_depth,
            idx=idx,
            save_path="output/adv_results"
        )

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
        log_path="output/adv_results/mIoU_average.txt"
    )

if __name__ == "__main__":
    atk()



