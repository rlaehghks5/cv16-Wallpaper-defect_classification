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
from utils_.loss import create_criterion, FocalLoss, LabelSmoothingLoss, F1Loss
# from utils_.get_class_weight import calc_class_weight
from runner.model import TimmModel
from runner.train_runner import CustomTrainer
from dataset import CustomDataset
from config import CFG 
from importlib import import_module

def main(num_classes):
    gc.collect() # python 자원 관리 
    torch.cuda.empty_cache() # gpu 자원관리
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_df = pd.read_csv('./data/train.csv')
    train_df['filename'] = train_df['filename'].apply(lambda x : os.path.join('./Data/Training_whole/PNG',x))
    train, val, _, _ = train_test_split(train_df, train_df['label'], test_size=0.3, stratify=train_df['label'], random_state=CFG['SEED'])
    
    le = preprocessing.LabelEncoder()
    train['label'] = le.fit_transform(train['label'])
    val['label'] = le.transform(val['label'])
    
    print('='*25, f' Model Train Start', '='*25)
    
    # -- dataset & dataloader
    train_dataset = CustomDataset(train['filename'].values, train['label'].values, transforms=True, CFG = CFG)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
    
    val_dataset = CustomDataset(val['filename'].values, val['label'].values, transforms = False, CFG = CFG)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    

    model = TimmModel(CFG['MODEL'], num_classes=num_classes, pretrained=True).to(device)
    # model = nn.DataParallel(model)
    
    # class_weight = calc_class_weight(train_dataset, model_name, num_classes, device)
    # criterion = FocalLoss(weight=class_weight).to(device)
    criterion = create_criterion(CFG['CRITERION'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    opt_module = getattr(import_module("torch.optim"), CFG['OPTIMIZER'])  # default: SGD
    
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG['LEARNING_RATE'],
        weight_decay=5e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs',min_lr=1e-9, verbose=True)

    trainer = CustomTrainer(CFG = CFG, model=model, train_dataloader=train_loader, valid_dataloader=val_loader, optimizer=optimizer, scheduler=scheduler, criterion=criterion, device=device)
    
    model, best_score, best_loss = trainer.train()
    
    print('='*25, f' Model Train End', '='*25)
    
    return model, best_score, best_loss

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    
    # # Data and model checkpoints directories
    # parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    # parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    # parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    # parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    # parser.add_argument('--model', type=str, default='resnet50', help='model type (default: BaseModel)')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: SGD)')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    # parser.add_argument('--criterion', type=str, default='CrossEntropy', help='criterion type CrossEntropy|FocalLoss|F1Loss|LabelSmoothingLoss (default: cross_entropy)')
    # parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    # parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    # parser.add_argument('--patience', type = int, default = 5, help = 'patience for earlystopping')
    # parser.add_argument('--how', default='Split', help='Model split mask, gender, age')
    
    # # Container environment
    # parser.add_argument('--data_dir', type=str, default='/workspace/data/train/images')
    # parser.add_argument('--model_dir', type=str, default='/workspace/models')
    
    # args = parser.parse_args()
    
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
    
         
    # data_dir = TRAIN_IMG_FOLDER_PATH
    model_dir = MODEL_SAVE_PATH
    model_name = CFG['MODEL']
    
    # project_idx = len(glob('/workspace/models/*')) + 1
    # os.makedirs(f'/workspace/models/Project{project_idx}', exist_ok=True)
    
    model, best_score, best_loss = main(num_classes = 19)
    torch.save(model.state_dict(), f'/opt/ml/Wallpaper-defect_classification/models/[{model_name}]_[score{best_score:.4f}]_[loss{best_loss:.4f}].pt')

    
    # model_name = "Mask"
    # mask_model, mask_best_score, mask_best_loss = main(args, data_dir, model_dir, model_name, 3)    
    # torch.save(mask_model.module.state_dict(), f'/workspace/models/Project{project_idx}/{model_name}_[{args.model}]_[score{mask_best_score:.4f}]_[loss{mask_best_loss:.4f}].pt')
    
    # model_name = "Gender"
    # gender_model, gender_best_score, gender_best_loss = main(args, data_dir, model_dir, model_name, 2)
    # torch.save(gender_model.module.state_dict(), f'/workspace/models/Project{project_idx}/{model_name}_[{args.model}]_[score{gender_best_score:.4f}]_[loss{gender_best_loss:.4f}].pt')
    
    # model_name = "Age"
    # age_model, age_best_score, age_best_loss = main(args, data_dir, model_dir, model_name, 3)
    # torch.save(age_model.module.state_dict(), f'/workspace/models/Project{project_idx}/{model_name}_[{args.model}]_[score{age_best_score:.4f}]_[loss{age_best_loss:.4f}].pt')
    
    