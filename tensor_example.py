import torch
a = torch.tensor([1,2,3])
print(a.tolist())

a = torch.tensor([[[1,2,3], [2,3,4]]])
print(a.tolist())