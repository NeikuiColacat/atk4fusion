import torch
import atk_util.atk_loss_vanilla as atk_util 
from get_model.DFormer import get_dformer
from get_model.DFormer import get_dformerv2
from atk_util.NYUv2_dataset import get_NYUv2_val_loader 
from atk_util.atk_loss_vanilla import Loss_Manager
from atk_util.NYUv2_img_with_patch import PatchGenerator
    
def atk():
    # model = get_dformerv2()
    model = get_dformer()
    val_loader = get_NYUv2_val_loader() 

    for p in model.parameters():
        p.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx , minibatch in enumerate(val_loader):

        images = minibatch["data"].to(device)
        labels = minibatch["label"].to(device)
        modal_xs = minibatch["modal_x"].to(device)


        patch_gen = PatchGenerator(images) 
        loss_mgr = Loss_Manager(images, modal_xs, labels, patch_gen , model)
        optimizer = torch.optim.Adam([loss_mgr.patch_gen.patch] , lr=0.01)

        for train_epoch in range(200) :
            optimizer.zero_grad()
            loss = loss_mgr()
            loss.backward()

            if train_epoch % 10 == 0:
                print(loss.item())
            
            optimizer.step()
        
        print("-----------------------------------------")

        img_adv = loss_mgr.patch_gen()
        mean_depth = torch.full_like(modal_xs, modal_xs.mean().item())
        mask = loss_mgr.patch_gen.mask

        logits_clean = model(images, modal_xs)
        logits_adv = model(img_adv, modal_xs)
        logits_clean_no_depth = model(images, mean_depth)
        logits_adv_no_depth = model(img_adv, mean_depth)

        _, mIoU = atk_util.get_mIoU_sklearn(logits_clean, labels, mask)
        print("clean mIoU" , mIoU)

        _, mIoU = atk_util.get_mIoU_sklearn(logits_adv, labels, mask)
        print("adv mIoU" , mIoU)

        _, mIoU = atk_util.get_mIoU_sklearn(logits_clean_no_depth, labels, mask)
        print("clean mIoU no depth" , mIoU)

        _, mIoU = atk_util.get_mIoU_sklearn(logits_adv_no_depth, labels, mask)
        print("adv mIoU no depth" , mIoU)

        print("-----------------------------------------")


if __name__ == "__main__":
    atk()



