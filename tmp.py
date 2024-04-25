import torch

p = torch.tensor([0.0005, 0.0006, 0.0005, 0.0006, 0.0006, 0.0006, 0.0006])

ans = torch.mean(torch.sum(p * torch.log(p + 1e-5)), 0)

print(ans)