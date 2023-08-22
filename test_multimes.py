import torch
from torch.utils.data import DataLoader
import UCRDataset
import Net
from sklearn.metrics import recall_score, f1_score
import numpy as np

# 调用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 先验超参数
batch_size = 64
img = 48
data_name = "MedicalImages"

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

# 加载数据集
test_dataset = UCRDataset.UCRDataset(root_path=f'./UCR_selected/{data_name}/',file_path=f'./UCR_selected/{data_name}/{data_name}_TEST.tsv',
                                     lenth=features,
                                     image_size=img)

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# 加载模型
model = Net.T2I(num=sel, lenth=features, image_size=img).to(device)
model.eval()

print("----------------Go.{}---------------------".format(data_name))

accuracy_list = []
f1_list = []

# 计算训练集和测试集的准确率
for i in range(20):
    model_path = f'./saved_models/{data_name}/{i}.pth'
    # 加载模型
    model = Net.T2I(num=sel, lenth=features, image_size=img).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 计算测试集的准确率
    with torch.no_grad():
        correct = 0
        total = 0
        targets = []
        predictions = []
        for data, img_input, target in test_loader:
            data, img_input, target = data.to(device), img_input.to(device), target.to(device)

            output = model(data, img_input)
            _, predicted = torch.max(output.data, 1)
            targets.extend(target.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            correct += (predicted == target).sum().item()
            total += target.size(0)

        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)

        f1 = 100 * f1_score(targets, predictions, average='macro')
        f1_list.append(f1)

        print(f'Model {i}: Accuracy = {accuracy:.2f}%, F1 = {f1:.2f}%')

# 计算Accuracy和F1的均值和方差
accuracy_mean = np.mean(accuracy_list)
accuracy_variance = np.var(accuracy_list)

f1_mean = np.mean(f1_list)
f1_variance = np.var(f1_list)

print(f'Accuracy - Mean: {accuracy_mean:.2f}%, Variance: {accuracy_variance:.2f}')
print(f'F1 - Mean: {f1_mean:.2f}%, Variance: {f1_variance:.2f}')

print("----------------End.{}---------------------".format(data_name))

    