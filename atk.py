import torch
from get_model.DFormer import get_dformer
from atk_util.NYUv2_dataset import get_NYUv2_val_loader 
from atk_util.atk_loss_vanilla import Loss_Manager
from atk_util.NYUv2_img_with_patch import PatchGenerator
    
def atk():
    model = get_dformer()
    val_loader = get_NYUv2_val_loader() 

    for p in model.parameters():
        p.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clean_tt = 0
    adv_tt = 0
    adv_dep_none = 0
    clean_dep_none = 0
    for idx , minibatch in enumerate(val_loader):

        images = minibatch["data"].to(device)
        labels = minibatch["label"].to(device)
        modal_xs = minibatch["modal_x"].to(device)

        patch_gen = PatchGenerator(images) 
        loss_mgr = Loss_Manager(images, modal_xs, labels, patch_gen , model)
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




if __name__ == "__main__":
    atk()



