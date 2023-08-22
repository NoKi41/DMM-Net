# 导入所需库
import UCRDataset
import Net
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time

# 使用'agg'后端以便在没有显示设备的环境中运行
plt.switch_backend('agg')

# 设置环境变量，以便CUDA在同步模式下运行
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 调用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 先验超参数
lr = 0.001
m = 0.9
epochs = 200
batch_size = 64
img = 48

data_name = "CBF"

# 数据集信息
data_infos = {
    "CBF":[128, 3],#
    "DiatomSizeReduction": [345, 4],#
    "ECG5000": [140, 5],
    "FordB": [500, 2],
    "GunPoint": [150, 2],#
    "HandOutlines": [2709, 2],
    "InsectWingbeatSound": [256, 11],#
    "MedicalImages": [99, 10],#
    'ShapeletSim': [500, 2],#
    "StarLightCurves": [1024, 3],
    "Strawberry": [235, 2],
    "SyntheticControl": [60, 6],#
    "Trace": [275, 4],#
    "TwoPatterns": [128, 4],#
    "WordSynonyms": [270, 25],#
}


features = data_infos[data_name][0]
sel = data_infos[data_name][1]

print("----------------Go.{}---------------------".format(data_name))

# 创建保存模型参数的主文件夹
save_path = f'./saved_models/{data_name}/'
os.makedirs(save_path, exist_ok=True)

# 20次训练的循环
for run in range(20):
    # 加载数据集
    train_dataset = UCRDataset.UCRDataset(root_path=f'./UCR_selected/{data_name}/',
                                          file_path=f'./UCR_selected/{data_name}/{data_name}_TRAIN.tsv',
                                          lenth=features,
                                          image_size=img)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、优化器和损失函数
    model = Net.T2I(num=sel, lenth=features, image_size=img).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)

    # 训练模型的代码，与你的原始代码相同
    
    for epoch in range(epochs):
        train_loss = 0.0
        correct1 = 0
        total1 = 0
        model.train()

        time_start = time.time()
        
        for data, img_input, target in train_loader:
            data, img_input, target = data.to(device), img_input.to(device), target.to(device)
            # print(data.shape,img_input.shape,target.shape)
            optimizer.zero_grad()
            # print('Zero Gradient for one time')
            output = model(data,img_input)

            # print('Get output for one time')
            # print('Output:', output)
            # print('Target', target)
            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            correct1 += (predicted == target).sum().item()
            total1 += target.size(0)
            # print('Get loss for one time')
            loss.backward()
            # print('Backward for one time')
            optimizer.step()
            # print('Optimize for one time')
            train_loss += loss.item() * data.size(0)
            # a = a + 1
            # print(a)
        
        scheduler.step()

    # 保存模型参数
    torch.save(model.state_dict(), './saved_models/{}/{}.pth'.format(data_name, run))

    print("----------------End.{}[{}]---------------------".format(data_name, run))
