import torch 
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils


if __name__ == "__main__":

    ts = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/{ts}")

    cnt = 0
    while True:
        a = torch.randn(100,100).mean()

        writer.add_scalar("train/loss", a.item(), cnt)

        cnt += 1


    writer.flush()
    writer.close()
    # S:int = 1024
    # while True: 
    #     a : torch.Tensor = torch.randn((S,S),device="cuda")
    #     b : torch.Tensor = torch.randn((S,S),device="cuda")

    #     c = (a @ b).mean()

    #     print(c.item())

