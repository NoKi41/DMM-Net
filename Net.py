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
    
    #TODO：卷积核大小是(image_size*4,image_size)

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

    #

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


# model = T2I(num=2,lenth=100,image_size=48)
# # 定义总参数量、可训练参数量及非可训练参数量变量
# Total_params = 0
# Trainable_params = 0
# NonTrainable_params = 0

# # 遍历model.parameters()返回的全局参数列表
# for param in model.parameters():
#     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
#     Total_params += mulValue  # 总参数量
#     if param.requires_grad:
#         Trainable_params += mulValue  # 可训练参数量
#     else:
#         NonTrainable_params += mulValue  # 非可训练参数量

# print(f'Total params: {Total_params}')
# print(f'Trainable params: {Trainable_params}')
# print(f'Non-trainable params: {NonTrainable_params}')


###### 形状测试
# import numpy as np
# b = np.random.randint(2, 4)
# test = torch.zeros(b,176)
# t = FCN_en(lenth=176, image_size=np.random.randint(10,100))
# y = t(test)
# print(y.shape)

# import numpy as np
# b = np.random.randint(2, 4)
# i = np.random.randint(10,100)
# test = torch.zeros(b,3,4*i,i)
# t = RES_en(lenth=176, image_size=i)
# y = t(test)
# print(y.shape)


# import numpy as np
# b = 5
# i = np.random.randint(10,100)
# t = T2I(num=37,lenth=176,image_size=i)
# x = torch.zeros(b,176)
# y = torch.zeros(b,4,i,i)
# pred = t(x, y)
# print(pred.shape)
