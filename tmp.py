import torch
import torch.nn.functional as F

tensor = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])

tensor_2 = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])

tensor_3 = torch.cat([tensor, tensor_2], dim=1)

print(tensor_3)