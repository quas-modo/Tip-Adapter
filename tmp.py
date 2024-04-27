import torch
import torch.nn.functional as F

tensor = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])

normalized_tensor = F.normalize(tensor, dim=0)
print(normalized_tensor)
