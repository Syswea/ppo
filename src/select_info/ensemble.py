from chi2 import ChiSquareSelector
from rfe import RecursiveFeatureElimination
from mutual_info import CategoricalFeatureSelector


class EnsembleFeatureSelector:
    def __init__(self, target_features=200):
        self.target_features = target_features
        self.mi_selector = CategoricalFeatureSelector(target_threshold=target_features)
        self.chi_selector = ChiSquareSelector(n_features=target_features)
        self.rfe_selector = RecursiveFeatureElimination(n_features=target_features)
    
    def ensemble_selection(self, X, y, voting_threshold=0.6):
        """
        多方法集成投票选择
        """
        # 方法1: 互信息
        mi_features, mi_scores = self.mi_selector.mutual_info_selection(X, y)
        
        # 方法2: 卡方检验
        chi2_features, chi2_scores = self.chi_selector.chi2_selection(X, y)
        
        # 方法3: 递归特征消除
        rfe_features, rfe_rankings = self.rfe_selector.rfe_selection(X, y)
        
        # 统计每个特征被选中的次数
        feature_votes = {}
        all_features = set(X.columns)
        
        # 投票计数
        for feature in mi_features:
            feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
        for feature in chi2_features:
            feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
        for feature in rfe_features:
            feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # 根据投票数选择特征
        min_votes = int(3 * voting_threshold)  # 3个方法中至少2个选中
        
        final_features = [
            feature for feature, votes in feature_votes.items() 
            if votes >= min_votes
        ]
        
        # 如果选中特征不足目标数量，从剩余特征中补充
        if len(final_features) < self.target_features:
            remaining_features = [f for f in all_features if f not in final_features]
            # 按互信息分数排序补充
            remaining_scores = [(f, self.mi_selector.feature_scores.get(f, 0)) 
                              for f in remaining_features]
            remaining_scores.sort(key=lambda x: x[1], reverse=True)
            
            need_more = self.target_features - len(final_features)
            additional_features = [f[0] for f in remaining_scores[:need_more]]
            final_features.extend(additional_features)
        
        return final_features, {
            'mutual_info': dict(zip(mi_features, mi_scores)),
            'chi2_scores': dict(zip(chi2_features, chi2_scores)),
            'rfe_ranking': dict(zip(X.columns, rfe_rankings)),
            'votes': feature_votes
        }

if __name__ == "__main__":
    x = EnsembleFeatureSelector()