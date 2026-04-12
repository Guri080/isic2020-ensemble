import os
import torch
import random
import copy
from PIL import Image, ImageOps

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2
import torchvision.transforms.v2 as v2

from custom_transformation import PadSquare

def _get_transformation():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = v2.Compose([
        PadSquare(fill=0, padding_mode='constant'),
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=90),
        
        v2.ToImage(),                    
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    val_transform = v2.Compose([
        PadSquare(fill=0, padding_mode='constant'),   
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    return {"train": train_transform, "val": val_transform}

class ISICDataset2020(Dataset):

    def __init__(self, df, root, split='val'):

        self.df = df
        self.root = root
    
        self.transform = _get_transformation()[split]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row.image_name + '.jpg'
        img_path = os.path.join(self.root, sample_name)

        image = Image.open(img_path).convert("RGB")

        label = torch.tensor(row.target, dtype=torch.long)

        image = self.transform(image)   
        
        return image, label
