import UCRDataset
import Net
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib as matplot
import numpy as np
import random

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

# 超参数
lr = 0.001
m = 0.9
epochs = 400
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
test_dataset = UCRDataset.UCRDataset(root_path=f'./UCR_selected/{data_name}/',file_path=f'./UCR_selected/{data_name}/{data_name}_TEST.tsv',
                                     lenth=features,
                                     image_size=img)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# 初始化模型、优化器和损失函数
model = Net.T2I(num=sel, lenth=features, image_size=img).to(device)

optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=m, weight_decay=5e-4)#带冲量的优化器

criterion = torch.nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)

# 训练模型
train_losses = []
test_losses = []
test_accs = []
train_accs = []
max_accu = 0.01

for epoch in range(epochs):
    train_loss = 0.0
    correct1 = 0
    total1 = 0
    model.train()
    
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
    # 在测试集上计算损失和准确率
    test_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, img_input, target in test_loader:
            data, img_input, target = data.to(device), img_input.to(device), target.to(device)
            output = model(data,img_input)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accs.append(correct / total)

    # 打印训练过程中的损失和准确率
    print('[{}]\t& {:.4f}    & {:.4f}    & {:.2f}\\%   & {:.2f}\\%    \\\\ \\hline'.format(epoch+1, train_loss, test_loss, 100*correct1/total1, 100*correct/total))
    
    if (correct/total) > max_accu:
        max_accu = correct/total
        print("Save Model based on Accu")
        torch.save(model.state_dict(),'./Model/model_{}.pth'.format(data_name))

print("----------------End.{}---------------------".format(data_name))


# 绘制损失曲线和准确率曲线
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()
plt.savefig('./Train_Log/{}_loss00.jpg'.format(data_name))

plt.close()
plt.plot(train_accs, label='train acc')
plt.plot(test_accs, label='test acc')
plt.legend()
plt.show()
plt.savefig('./Train_Log/{}_acc00.jpg'.format(data_name))