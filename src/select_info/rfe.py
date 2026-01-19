# rfe.py (修复版)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# 注意：不再需要 LabelEncoder

class RecursiveFeatureElimination:
    def __init__(self, n_features=200):
        self.n_features = n_features
        # 移除了 self.label_encoders，因为我们不再需要它

    def rfe_selection(self, X, y, estimator='rf'):
        """
        递归特征消除
        注意：假设 X 是数值型DataFrame（可以包含NaN，但最好提前处理）
        """
        # ✅ 关键修复：不再对X进行任何编码！直接使用原始数值。
        # 如果你的数据中有缺失值，应该在这里或之前处理，例如用均值/中位数填充。
        # X_clean = X.fillna(X.median()) # 示例

        # 选择基础模型
        if estimator == 'rf':
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            # LogisticRegression 需要确保输入是有限的数值
            base_model = LogisticRegression(random_state=42, max_iter=1000)

        # RFE选择
        rfe = RFE(estimator=base_model, n_features_to_select=self.n_features)
        rfe.fit(X, y)  # ✅ 直接传入原始 X

        selected_features = X.columns[rfe.support_]
        feature_ranking = rfe.ranking_
        return selected_features, feature_ranking

if __name__ == "__main__":
    x = RecursiveFeatureElimination()