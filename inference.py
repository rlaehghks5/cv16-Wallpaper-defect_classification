import argparse
import multiprocessing
import os
from importlib import import_module
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CustomDataset, TestDataset
from config import CFG
from utils_.set_path import *

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("runner.model"), 'TimmModel')
    model = model_cls(
                CFG['MODEL'], num_classes=num_classes, pretrained=True
                )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(saved_model, map_location=device))

    return model


@torch.no_grad()
def inference(model_pt_dir, output_dir,saved_name):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 19  # 19
    model = load_model(model_pt_dir, num_classes, device).to(device)
    model.eval()

    train_df = pd.read_csv('./data/train.csv')
    train_df['filename'] = train_df['filename'].apply(lambda x : os.path.join('./Data/Training_whole/PNG',x))
    # train, val, _, _ = train_test_split(train_df, train_df['label'], test_size=0.3, stratify=train_df['label'], random_state=CFG['SEED'])
    
    le = preprocessing.LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])
    # val['label'] = le.transform(val['label'])

    test = pd.read_csv('./data/test.csv')
    test['img_path'] = test['img_path'].apply(lambda x : os.path.join('./data', '/'.join(x.split('/')[1:])))
    
    # img_root = os.path.join(data_dir, 'images')
    # info_path = os.path.join(data_dir, 'info.csv')
    # info = pd.read_csv(info_path)

    # img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    
    test_dataset = TestDataset(test['img_path'].values, None, transforms=False, CFG = CFG)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, drop_last= False)

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(test_loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
    preds = le.inverse_transform(preds)
    # info['ans'] = preds
    submit = pd.read_csv('./data/sample_submission.csv')
    submit['label'] = preds
    
    save_path = os.path.join(output_dir, f'{saved_name}')
    submit.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    # parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 32)')
    # parser.add_argument('--resize', nargs="+", type=int, default=[128, 96], help='resize size for image when you trained (default: (96, 128))')
    # parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    # args = parser.parse_args()

    model_name = CFG['MODEL']
    # data_dir = 
    
    model_pt_dir = f'/opt/ml/Wallpaper-defect_classification/models/[vit_base_patch16_224]_[score0.8078]_[loss0.6596].pt'
    output_dir = SUBMIT_SAVE_PATH
    saved_name = 'vit_base_patch16_224_base.csv'
    os.makedirs(output_dir, exist_ok=True)

    inference(model_pt_dir, output_dir,saved_name)
