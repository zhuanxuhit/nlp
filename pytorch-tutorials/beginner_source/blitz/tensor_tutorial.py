import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)


x = torch.tensor([5.5, 3])
print(x)
print(x.shape)