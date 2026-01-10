import torch

a = torch.tensor([1,2,3] ,dtype=torch.float32)

b = torch.full_like(a , a.mean())

print(b)
