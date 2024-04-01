from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class ElasticNetModel:
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        """
        初始化弹性网模型。
        :param alpha: 正则化强度。
        :param l1_ratio: 弹性网混合参数（0 <= l1_ratio <= 1）。
                         l1_ratio=0 表示 L2 正则化，l1_ratio=1 表示 L1 正则化。
        """
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    def train(self, X_train, y_train):
        """
        训练弹性网模型。
        :param X_train: 训练数据特征。
        :param y_train: 训练数据标签。
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        使用弹性网模型进行预测。
        :param X_test: 测试数据特征。
        :return: 预测结果。
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        评估模型性能。
        :param X_test: 测试数据特征。
        :param y_test: 测试数据标签。
        :return: 均方误差和 R^2 分数。
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def get_model(self):
        """
        获取模型实例。
        :return: 弹性网模型。
        """
        return self.model
