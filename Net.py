import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN_en(torch.nn.Module):
    def __init__(self, lenth, image_size):
        super(FCN_en, self).__init__()
        
        self.lenth = lenth 
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=256,
                               kernel_size=7,padding=3)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5,padding=2)
        self.bn2 = nn.BatchNorm1d(num_features=512)

        self.conv3 = nn.Conv1d(in_channels=512,
                               out_channels=128,
                               kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.pool = nn.AdaptiveAvgPool1d((1))
        self.drop = nn.Dropout(0.5)

        
    def forward(self, x):

        x = x.unsqueeze(1) # 增加一个channel维度，训练
        # x = x.unsqueeze(0) # 画图
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.drop(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.drop(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        x = self.pool(x)

        x = torch.flatten(x,start_dim=1)
        x = self.drop(x)
        # x = self.fc(x)
        return x
    
class RES_en(torch.nn.Module):
    def __init__(self, lenth, image_size):
        super(RES_en, self).__init__()
        
        self.image_size = image_size
        self.lenth = lenth 
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=64,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        #P = (K - 1) / 2
        self.res64_conv = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)
        self.bn1 = nn.BatchNorm2d(64)


        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.res128_conv = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.res256_conv = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=128,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.res512_conv = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        #res64
        y = self.bn1(self.res64_conv(x))
        y = self.relu(y)
        x = y + x
        # x = self.drop(x)

        #res128
        x = self.conv2(x)
        x = self.maxpool(x)
        y = self.relu(self.bn2(self.res128_conv(x)))
        x = y + x

        x = self.conv3(x)

        y = self.relu(self.bn3(self.res256_conv(x)))
        x = y + x
        # x = self.drop(x)

        x = self.conv4(x)
        y = self.relu(self.bn4(self.res512_conv(x)))
        x = y + x
        x = self.drop(x)


        x = self.avgpool(x)
        x = torch.flatten(x,start_dim=1)
        # x = x.view(x.size(0), 1, -1)
        return x

class T2I(torch.nn.Module):
    def __init__(self, num, lenth, image_size) -> None:
        super(T2I, self).__init__()
        self.image_size = image_size
        self.lenth = lenth 
       
        self.fcn = FCN_en(lenth=self.lenth, image_size=self.image_size)
        self.res = RES_en(lenth=self.lenth, image_size=self.image_size)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=128,
                             out_features=128)
        self.fc2 = nn.Linear(in_features=128,
                             out_features=128)
        self.conv1 = nn.Linear(256,256)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(in_features=256,
                             out_features=num)
        self.initialize()

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, y):
        x = self.fcn(x)
        # x = self.dropout(x)
        y = self.res(y)
        # y - self.dropout(y)
        x = self.fc1(x)
        y = self.fc2(y)
        z = torch.cat([x, y],dim=1)
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.dropout(z)
        return self.fc3(z)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
 
    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

