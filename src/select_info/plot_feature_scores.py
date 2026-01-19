import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



class FeatureImportanceVisualizer:
    def __init__(self):
        self.fig_size = (15, 10)
    
    def plot_feature_scores(self, feature_scores, top_n=50):
        """
        绘制特征重要性分数
        """
        # 创建分数数据框
        scores_df = pd.DataFrame([
            {'feature': feature, 'score': score} 
            for feature, score in feature_scores.items()
        ]).sort_values('score', ascending=False).head(top_n)
        
        plt.figure(figsize=self.fig_size)
        sns.barplot(data=scores_df, x='score', y='feature')
        plt.title(f'Top {top_n} Feature Importance Scores')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return scores_df
    
    def plot_method_comparison(self, method_results):
        """
        比较不同方法的特征选择结果
        """
        methods = list(method_results.keys())
        n_selected = [len(method_results[method]) for method in methods]
        
        plt.figure(figsize=(10, 6))
        plt.bar(methods, n_selected, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Number of Features Selected by Each Method')
        plt.ylabel('Number of Features')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()