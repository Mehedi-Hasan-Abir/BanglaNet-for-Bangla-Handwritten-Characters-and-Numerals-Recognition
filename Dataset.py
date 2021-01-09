import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


l = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ','ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ', 'ং', 'ঃ', 'ঁ',
     '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯'     ]
print(len(l))


mapping = {}
for i in range(60):
  mapping[str(i)] = l[i]
print(mapping["0"])


##!cp drive/Shared\ drives/Bangla-Handwritten/images.zip ./


dataset = pd.read_csv('drive/Shared drives/Bangla-Handwritten/final.csv', sep = ',')
dataset.drop('Unnamed: 0', axis = 1, inplace = True)
dataset.head()


IMAGE_PATH = './images'

class Dataset(Dataset):
    def __init__(self, df, root, transform=None):
        self.data = df
        self.root = root
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        path = self.root + "/" + item[0]
        image = Image.open(path).convert('L')
        label = item[1]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


import torch
from torchvision import datasets, transforms, models
import albumentations as A

mean = [0.5,]
std = [0.5, ]

dataset_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])


full_dataset = Dataset(dataset, IMAGE_PATH, dataset_transform)
print(len(full_dataset))