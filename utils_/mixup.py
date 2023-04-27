from typing import Tuple
import numpy as np
import torch
from config import CFG

# def partial_mixup(input: torch.Tensor, gamma: float, indices: torch.Tensor) -> torch.Tensor:
#     if input.size(0) != indices.size(0):
#         raise RuntimeError("Size mismatch!")
#     perm_input = input[indices]
#     return input.mul(gamma).add(perm_input, alpha=1 - gamma)

# def mixup(input: torch.Tensor, target: torch.Tensor, gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
#     indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
#     return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)

def MixUp(input, target):
    if CFG['ALPHA'] > 0:
        lambda_ = np.random.beta(CFG['ALPHA'], CFG['ALPHA'])
    else:
        lambda_ = 1

    batch_size = input.size(0)
    index = torch.randperm(batch_size)
    
    mixed_input = lambda_ * input + (1 - lambda_) * input[index, :]    
    labels_a, labels_b = target, target[index]

    return mixed_input, labels_a, labels_b, lambda_