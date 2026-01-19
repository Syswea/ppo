from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd

class CategoricalFeatureSelector:
    def __init__(self, target_threshold=200):
        self.target_threshold = target_threshold
        self.feature_scores = {}

    def mutual_info_selection(self, X, y):
        """
        互信息特征选择 —— 正确处理连续特征
        注意：假设 X 是数值型（连续或已编码的分类），y 是二分类
        """
        # 确保 X 是数值型且无缺失（mutual_info_classif 要求）
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 关键：discrete_features=False 表示所有特征都是连续的
        # 或者你可以传一个布尔数组指定哪些是离散的
        mi_scores = mutual_info_classif(
            X_clean, y,
            discrete_features=False,  # ✅ 所有特征视为连续
            random_state=42
        )

        # 创建特征分数字典
        self.feature_scores = dict(zip(X.columns, mi_scores))

        # 选择 top-k 特征
        selected_features = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.target_threshold]

        return [feat for feat, _ in selected_features], mi_scores