import warnings
warnings.filterwarnings(action='ignore')

import os
import gc
import cv2
import random
import argparse
import numpy as np
import pandas as pd
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm.autonotebook import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import wandb

from utils_.set_path import *
from utils_.set_seed import seed_everything
from utils_.loss import create_criterion
# from utils_.get_class_weight import calc_class_weight
from runner.model import TimmModel
from runner.train_runner import CustomTrainer
from dataset import CustomDataset
from config import CFG 
from importlib import import_module

def main(device, num_classes):
    gc.collect() # python 자원 관리 
    torch.cuda.empty_cache() # gpu 자원관리

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df['filename'] = train_df['filename'].apply(lambda x : os.path.join(TRAIN_IMG_FOLDER_PATH, x))

    le = preprocessing.LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])
    
    train, val, _, _ = train_test_split(train_df, train_df['label'], test_size=0.2, stratify=train_df['label'], random_state=CFG['SEED'])
  
    print('='*25, f' Model Train Start', '='*25)
    
    # -- dataset & dataloader
    train_dataset = CustomDataset(train['filename'].values, train['label'].values, transforms=True, CFG=CFG)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
    
    val_dataset = CustomDataset(val['filename'].values, val['label'].values, transforms = False, CFG=CFG)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = TimmModel(CFG['MODEL'], num_classes=num_classes, pretrained=True).to(device)
    model = nn.DataParallel(model)
    
    # class_weight = calc_class_weight(train_dataset, model_name, num_classes, device)
    criterion = create_criterion(CFG['CRITERION'])
    optimizer = getattr(import_module("torch.optim"), CFG['OPTIMIZER'])(
                                filter(lambda p: p.requires_grad, model.parameters()),
                                lr=CFG['LEARNING_RATE'],
                                weight_decay=5e-4
                            )    

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-9, verbose=True)
    trainer = CustomTrainer(CFG=CFG, model=model, train_dataloader=train_loader, valid_dataloader=val_loader, optimizer=optimizer, scheduler=scheduler, criterion=criterion, device=device)
    
    model, best_score, best_loss = trainer.train()
    
    print('='*25, f' Model Train End', '='*25)
    
    return model, best_score, best_loss

if __name__ == '__main__':
    
    seed_everything(CFG['SEED'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
         
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="mask-classification",

    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": args.lr,
    #     "architecture": args.model,
    #     "epochs": args.epochs,
    #     "batch_size" : args.batch_size,
    #     "loss" : args.criterion,
    #     "how": args.how,
    #     },
    #     name=f"{args.model}_{args.criterion}_resize{args.resize}_crop[]_lr[{args.lr}]"

    model_save_path = MODEL_SAVE_PATH
    model_name = CFG['MODEL']
    
    # project_idx = len(glob('/workspace/models/*')) + 1
    # os.makedirs(f'/workspace/models/Project{project_idx}', exist_ok=True)
    
    model, best_score, best_loss = main(device, num_classes=19)
    torch.save(model.state_dict(), os.path.join(model_save_path, f'[{model_name}]_[score{best_score:.4f}]_[loss{best_loss:.4f}].pt'))