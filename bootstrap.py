import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import UCRDataset
import Net
from sklearn.metrics import recall_score, f1_score
from sklearn.utils import resample

# 调用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 先验超参数
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
test_dataset = UCRDataset.UCRDataset(root_path=f'./UCR_selected/{data_name}/',file_path=f'./UCR_selected/{data_name}/{data_name}_TEST.tsv',
                                     lenth=features,
                                     image_size=img)

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# 加载模型
model = Net.T2I(num=sel, lenth=features, image_size=img).to(device)
model.load_state_dict(torch.load(f'./Model/model_{data_name}.pth'))
model.eval()


# 重复采样次数
n_iterations = 1000

# 置信区间
confidence_level = 0.95

# 保存模型得分
accuracies = []
recalls = []
f1s = []

# 计算训练集和测试集的准确率
with torch.no_grad():
    # 获取全部测试数据
    test_data = list(test_loader)
    for i in range(n_iterations):
        # Bootstrap采样
        bootstrap_sample = resample(test_data, replace=True)
        correct = 0
        total = 0
        targets = []
        predictions = []
        for data, img_input, target in bootstrap_sample:
            data, img_input, target = data.to(device), img_input.to(device), target.to(device)

            output = model(data, img_input)
            _, predicted = torch.max(output.data, 1)
            targets.extend(target.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            correct += (predicted == target).sum().item()
            total += target.size(0)

        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        recall = 100 * recall_score(targets, predictions, average='macro', zero_division=0)
        recalls.append(recall)

        f1 = 100 * f1_score(targets, predictions, average='macro')
        f1s.append(f1)


# 计算统计信息
accuracy_mean = np.mean(accuracies)
accuracy_lower = np.percentile(accuracies, (1 - confidence_level) / 2 * 100)
accuracy_upper = np.percentile(accuracies, (1 + confidence_level) / 2 * 100)

recall_mean = np.mean(recalls)
recall_lower = np.percentile(recalls, (1 - confidence_level) / 2 * 100)
recall_upper = np.percentile(recalls, (1 + confidence_level) / 2 * 100)

f1_mean = np.mean(f1s)
f1_lower = np.percentile(f1s, (1 - confidence_level) / 2 * 100)
f1_upper = np.percentile(f1s, (1 + confidence_level) / 2 * 100)

print(f'Accuracy: {accuracy_mean:.2f}% ({accuracy_lower:.2f} - {accuracy_upper:.2f})')
print(f'Recall: {recall_mean:.2f}% ({recall_lower:.2f} - {recall_upper:.2f})')
print(f'F1 Score: {f1_mean:.2f}% ({f1_lower:.2f} - {f1_upper:.2f})')

# 绘制得分的分布
fig, ax = plt.subplots(3, 1, figsize=(12, 18))

ax[0].hist(accuracies, bins=20, alpha=0.5, color='g')
ax[0].axvline(accuracy_mean, color='g', linestyle='dashed', linewidth=2)
ax[0].set_title('Accuracy Distribution')

ax[1].hist(recalls, bins=20, alpha=0.5, color='b')
ax[1].axvline(recall_mean, color='b', linestyle='dashed', linewidth=2)
ax[1].set_title('Recall Distribution')

ax[2].hist(f1s, bins=20, alpha=0.5, color='r')
ax[2].axvline(f1_mean, color='r', linestyle='dashed', linewidth=2)
ax[2].set_title('F1 Score Distribution')


# plt.title(f'{data_name}', loc="center")
plt.show()