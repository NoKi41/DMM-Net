import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket
from sktime.datasets import load_UCR_UEA_dataset

def classify_with_rocket(dataset_name):
    # 加载数据集
    X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

    # 创建Rocket转换器实例
    rocket = Rocket()

    # 在训练集上拟合Rocket转换器
    rocket.fit(X_train)

    # 将Rocket转换器应用于训练和测试集
    X_train_transform = rocket.transform(X_train)
    X_test_transform = rocket.transform(X_test)

    # 创建并拟合分类器
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    classifier.fit(X_train_transform, y_train)

    # 在测试集上评估分类器
    accuracy = classifier.score(X_test_transform, y_test)

    return accuracy


dataset_name = "ECG5000"
accuracy = classify_with_rocket(dataset_name)
print(f"Accuracy on {dataset_name}: {accuracy}")
