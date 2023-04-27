import argparse
import multiprocessing
import os
from importlib import import_module
from tqdm.auto import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CustomDataset
from config import CFG
from utils_.set_path import *

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("runner.model"), 'TimmModel')
    model = model_cls(
                CFG['MODEL'], num_classes=num_classes, pretrained=True
                ).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(saved_model, map_location=device))
    return model

@torch.no_grad()
def inference(model_pt_dir, output_dir,saved_name):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 19
    model = load_model(model_pt_dir, num_classes, device).to(device)    
    model.eval()

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df['filename'] = train_df['filename'].apply(lambda x : os.path.join(TRAIN_IMG_FOLDER_PATH, x))
    
    le = preprocessing.LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])

    test = pd.read_csv(TEST_CSV_PATH)
    test['img_path'] = test['img_path'].apply(lambda x : os.path.join('./data', '/'.join(x.split('/')[1:])))
    
    test_dataset = CustomDataset(test['img_path'].values, None, transforms=False, CFG=CFG)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, drop_last= False)

    print("Calculating inference results..")
    
    preds = []
    with torch.no_grad():
        for images in tqdm(iter(test_loader), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
            
    preds = le.inverse_transform(preds)

    submit = pd.read_csv(SUBMIT_CSV_PATH)
    submit['label'] = preds
    
    save_path = os.path.join(output_dir, f'{saved_name}')
    submit.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

if __name__ == '__main__':    
    model_name = CFG['MODEL']

    weight_name = 'efficientnet_b3_MIXUP:True_cross_entropy_AdamW_lr[0.0005]_score[0.8322]_loss[0.6032].pt'
    model_pt_dir = f'{MODEL_SAVE_PATH}/{weight_name}'
    
    output_dir = SUBMIT_SAVE_PATH
    
    saved_name = f'{weight_name}.csv'

    os.makedirs(output_dir, exist_ok=True)

    inference(model_pt_dir, output_dir, saved_name)
