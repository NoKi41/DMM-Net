import UCRDataset
import Net
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib as matplot
import numpy as np
import random
import time

matplot.use('agg')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    #固定一个随机因子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2023)

# 调用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 先验超参数
lr = 0.001
m = 0.9
epochs = 160
batch_size = 64
img = 48

data_name = "StarLightCurves"

data_infos = {
    "CBF":[128, 3],#
    "DiatomSizeReduction": [345, 4],#
    "ECG5000": [140, 5],#
    "FordB": [500, 2],#
    "GunPoint": [150, 2],#
    "HandOutlines": [2709, 2],
    "InsectWingbeatSound": [256, 11],
    "MedicalImages": [99, 10],
    'ShapeletSim': [500, 2],
    "StarLightCurves": [1024, 3],
    "Strawberry": [235, 2],
    "SyntheticControl": [60, 6],
    "Trace": [275, 4],
    "TwoPatterns": [128, 4],
    "WordSynonyms": [270, 25],
}

features = data_infos[data_name][0]
sel = data_infos[data_name][1]

print("----------------Go.{}---------------------".format(data_name))

# 加载数据集
train_dataset = UCRDataset.UCRDataset(root_path=f'./UCR_selected/{data_name}/',file_path=f'./UCR_selected/{data_name}/{data_name}_TRAIN.tsv',
                                      lenth=features,
                                      image_size=img)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

# 初始化模型、优化器和损失函数
model = Net.T2I(num=sel, lenth=features, image_size=img).to(device)

optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=m, weight_decay=5e-4)#带冲量的优化器

criterion = torch.nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)


# 训练模型
train_losses = []
train_accs = []
max_accu = 0.01

t = 0

for epoch in range(epochs):
    train_loss = 0.0
    correct1 = 0
    total1 = 0
    model.train()

    time_start = time.time()
    
    for data, img_input, target in train_loader:
        data, img_input, target = data.to(device), img_input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data,img_input)

        loss = criterion(output, target)
        _, predicted = torch.max(output.data, 1)
        correct1 += (predicted == target).sum().item()
        total1 += target.size(0)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accs.append(correct1 / total1)
    scheduler.step()
    
    # 打印训练过程中的损失和准确率
    print('[{}]\t& {:.4f}    & {:.2f}\\%   \\\\ \\hline'.format(epoch+1, train_loss, 100*correct1/total1))

    time_end = time.time()
    t = t + time_end - time_start

print('Time cost: ' + str(t) + 's')

print("----------------End.{}---------------------".format(data_name))

# 绘制损失曲线和准确率曲线
plt.plot(train_losses, label='train loss')
plt.title('{}'.format(data_name))
plt.legend()
plt.show()
plt.savefig('./train/{}_loss_{}epoch.jpg'.format(data_name, epochs))
plt.close()
