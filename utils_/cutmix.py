import torch
import numpy as np
from config import CFG

def rand_bbox(size, lambda_): # # 64, 3, H, W, lambda
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lambda_) # cut 비율
    cut_w = np.int(W * cut_rat) # 전체 넓이, 높이 중 비율만큼 선택
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def CutMix(input, target):
    if CFG['C_ALPHA1'] > 0: lambda_ = np.random.beta(CFG['C_ALPHA1'], CFG['C_ALPHA2'])
    else: lambda_ = 1

    batch_size = input.size()[0]
    index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lambda_)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[index, :, bbx1:bbx2, bby1:bby2]
    lambda_ = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    
    label_a, label_b = target, target[index]
    
    return input, label_a, label_b, lambda_