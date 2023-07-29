import pandas as pd
import torch
from torch.utils.data import Dataset
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from Image_Transform import ImageTransformer
from torchvision import transforms
import numpy as np
import os

class UCRDataset(Dataset):
    def __init__(self, root_path, file_path, lenth, image_size):
        self.df = pd.read_csv(file_path, sep='\t', header=None)
        self.data = self.df.iloc[:, 1:].values
        self.labels = self.df.iloc[:, 0].values
        self.lenth =lenth
        self.image_size = image_size
        self.image_transformer = ImageTransformer(lenth= self.lenth, image_size=image_size)
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]-1

        # 数据转换为图像表示
        gasf_img, gadf_img, mtf_img, rp_img = self.image_transformer.fit_transform(data.reshape(1, -1))

        #reshape

        gasf_tensor = torch.from_numpy(gasf_img).float()
        gadf_tensor = torch.from_numpy(gadf_img).float()
        mtf_tensor = torch.from_numpy(mtf_img).float()
        rp_tensor = torch.from_numpy(rp_img).float()

        # print(data.shape,gasf_tensor.shape,gadf_tensor.shape,mtf_tensor.shape,rp_tensor.shape)
        img_input = torch.cat((gasf_tensor, gadf_tensor, mtf_tensor, rp_tensor), dim=0)
        
        return torch.tensor(data).float(), img_input.float(), torch.tensor(label).long()

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
  


    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

