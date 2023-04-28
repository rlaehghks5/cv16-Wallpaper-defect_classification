import torch
import numpy as np
from config import CFG

def MixUp(input, target):
    if CFG['M_ALPHA1'] > 0: lambda_ = np.random.beta(CFG['M_ALPHA1'], CFG['M_ALPHA2'])
    else: lambda_ = 1

    batch_size = input.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_input = lambda_ * input + (1 - lambda_) * input[index, :]    
    labels_a, labels_b = target, target[index]

    return mixed_input, labels_a, labels_b, lambda_