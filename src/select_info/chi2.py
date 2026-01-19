from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder

class ChiSquareSelector:
    def __init__(self, n_features=200):
        self.n_features = n_features
        self.selector = SelectKBest(score_func=chi2, k=n_features)
        self.label_encoders = {}
    
    def chi2_selection(self, X, y):
        """
        卡方检验特征选择
        """
        X_encoded = X.copy()
        
        # 编码分类特征
        for col in X.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # 确保所有特征都是非负整数
        X_encoded = X_encoded - X_encoded.min() + 1
        
        # 拟合 selector
        self.selector.fit(X_encoded, y)

        # 获取布尔掩码（长度 = 原始特征数）
        selected_mask = self.selector.get_support()  # ✅ 这才是正确的 mask！

        # 提取选中的特征名
        selected_features = X.columns[selected_mask]
        
        # 获取卡方分数
        chi2_scores = self.selector.scores_
        
        return selected_features, chi2_scores

if __name__ == "__main__":
    x = ChiSquareSelector()