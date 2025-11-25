#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Exercise2 的实现
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# 1. 数据加载
print("=" * 60)
print("加载数据...")
print("=" * 60)

names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
         'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
         'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

data_path = "data/breast-cancer-wisconsin.data"
data = pd.read_csv(data_path, names=names)

# 2 数据预处理
# 2.1 数据清洗：去除含缺失值的样本
data = data.replace(to_replace="?", value=np.nan)
data = data.dropna()

# 2.2 将特征列和标签列转换为数值类型
data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
data = data.dropna()

x = data.iloc[:, 1:10].values.astype(np.float64)
y = data["Class"].values.astype(int)
y = np.where(y == 4, 1, 0)

print(f"数据集形状: X={x.shape}, y={y.shape}")
print(f"正类样本数: {np.sum(y==1)}, 负类样本数: {np.sum(y==0)}")


# 2.3 数据集划分
def train_test_split_manual(X, y, test_size=0.25, random_state=2025):
    """手动实现训练集和测试集的划分"""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    # 生成随机索引
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# 2.4 特征标准化
class StandardScaler:
    # 实现特征标准化
    # 标准化公式: z = (x - mean) / std

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """计算均值和标准差"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """使用均值和标准差进行标准化"""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("需要先调用 fit 方法计算均值和标准差")

        # 避免除以0的情况
        std = np.where(self.std_ == 0, 1, self.std_)
        return (X - self.mean_) / std

    def fit_transform(self, X):
        """拟合并转换"""
        self.fit(X)
        return self.transform(X)


# 3. 实现逻辑回归
class LogisticRegression:
    """
    参数:
        learning_rate: 学习率
        n_iterations: 迭代次数
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """实现sigmoid函数"""
        # sigmoid(z) = 1 / (1 + e^(-z))
        # 为了数值稳定性，对z进行裁剪
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练模型 - 使用批量梯度下降法（BGD）优化权重和偏置"""

        # 1. 初始化权重和偏置
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. 进行n_iterations次迭代，添加进度条
        print(f"\n开始训练模型 (学习率={self.learning_rate}, 迭代次数={self.n_iterations})...")
        for i in tqdm(range(self.n_iterations), desc="训练进度"):
            # step1. 计算预测值
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)

            # step2. 计算梯度
            # 损失函数对权重的偏导数: (1/m) * X^T * (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            # 损失函数对偏置的偏导数: (1/m) * sum(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # step3. 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 每100次迭代打印一次损失（可选，用于调试）
            if (i + 1) % 200 == 0:
                loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
                tqdm.write(f"  迭代 {i+1}/{self.n_iterations}, 损失: {loss:.4f}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """预测"""
        # 1. 进行线性运算
        linear_pred = np.dot(X, self.weights) + self.bias

        # 2. 进行sigmoid计算
        y_pred_prob = self.sigmoid(linear_pred)

        # 3. 将结果与阈值进行比较
        y_pred_class = (y_pred_prob >= threshold).astype(int)

        return y_pred_class


def get_metrics(y_true, y_pred):
    """
        获得评测指标
    """
    def recall_score(y_true, y_pred):
        """
        计算召回率
        召回率 = TP / (TP + FN)
        """
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        # 避免除以0
        if (TP + FN) == 0:
            return 0.0

        return TP / (TP + FN)

    def precision_score(y_true, y_pred):
        """精确率 = TP / (TP + FP)"""
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))

        # 避免除以0
        if (TP + FP) == 0:
            return 0.0

        return TP / (TP + FP)

    def accuracy_score(y_true, y_pred):
        """准确率 = (TP + TN) / 总数"""
        return np.mean(y_true == y_pred)


    def confusion_matrix(y_true, y_pred):
        """混淆矩阵"""
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TP = np.sum((y_true == 1) & (y_pred == 1))

        return np.array([[TN, FP], [FN, TP]])

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return recall, precision, accuracy, cm


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("乳腺癌诊断预测 - 逻辑回归实现")
    print("=" * 60)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split_manual(x, y, test_size=0.25, random_state=22)

    print(f"\n数据集信息:")
    print(f"训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    print(f"特征数: {X_train.shape[1]}")

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练逻辑回归模型
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)

    recall, precision, accuracy, cm = get_metrics(y_test, y_pred)

    print(f"\n召回率 (Recall):    {recall:.4f} ({recall*100:.2f}%)")
    print(f"精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
    print(f"准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"\n混淆矩阵:")
    print(f"              预测负类  预测正类")
    print(f"实际负类        {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"实际正类        {cm[1,0]:4d}     {cm[1,1]:4d}")

    print("\n" + "=" * 60)
