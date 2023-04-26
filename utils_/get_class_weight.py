import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calc_class_weight(train_df, num_classes, device):
    labels = [label for label in train_df['label']]
    labels.sort()

    labels_weight = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=labels)
    labels_weight = torch.FloatTensor(labels_weight).to(device)

    return labels_weight
    