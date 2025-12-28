import torch
import atk_util.atk_loss_vanilla as atk_util 
from get_model.DFormer import get_dformer
from get_model.DFormer import get_dformerv2
from atk_util.NYUv2_dataset import get_NYUv2_val_loader 
from atk_util.atk_loss_vanilla import Loss_Manager
from atk_util.NYUv2_img_with_patch import PatchGenerator
    
def atk():
    model = get_dformerv2()
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

        for train_epoch in range(250) :
            optimizer.zero_grad()
            loss = loss_mgr()
            loss.backward()

            if train_epoch % 10 == 0:
                print(loss.item())
            
            optimizer.step()
        
        print("-----------------------------------------")

        logits_clean = model(images , modal_xs)
        _ , mIoU = atk_util.get_mIoU(logits_clean , labels , loss_mgr.patch_gen.mask) 
        print("clean mIoU" , mIoU)

        logits_adv = model(loss_mgr.patch_gen(), modal_xs)
        _ , mIoU = atk_util.get_mIoU(logits_adv , labels , loss_mgr.patch_gen.mask) 
        print("adv mIoU" , mIoU)
        break

if __name__ == "__main__":
    atk()



