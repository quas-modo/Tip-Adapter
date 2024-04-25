import torch

p = torch.randn(3, 5, dtype=float)

result = 1 - p

print(p)
print(result)