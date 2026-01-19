import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import optuna
from optuna.samplers import TPESampler
import warnings
import os

warnings.filterwarnings('ignore')

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_df = pd.read_csv(os.path.join(workspace, 'data', 'cluster_train_selected.csv'))

print(f"数据集形状: {train_df.shape}")
print(f"目标变量 'fpd1' 分布:\n{train_df['fpd1'].value_counts()}")
print(f"正样本比例: {train_df['fpd1'].mean():.4f}")

X = train_df.drop('fpd1', axis=1)
y = train_df['fpd1']

N_FOLDS = 5
kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'num_leaves': trial.suggest_int('num_leaves', 20, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
        cv_scores.append(fold_auc)

    return np.mean(cv_scores)

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

print(f"\n开始Optuna超参数优化 ({N_FOLDS}-Fold CV)...")
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_params = study.best_params
best_cv_auc = study.best_value

print(f"\n最优参数:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"最优 {N_FOLDS}-Fold CV AUC: {best_cv_auc:.4f}")

print(f"\n使用最优参数进行全量数据 {N_FOLDS}-Fold 训练...")

oof_predictions = np.zeros(len(X))
feature_importance_list = []
cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        random_state=42,
        verbose=-1,
        n_jobs=-1,
        **best_params
    )

    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_idx] = y_pred_proba

    fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
    cv_scores.append(fold_auc)

    print(f"  Fold {fold_idx + 1}: AUC = {fold_auc:.4f}")

    feature_importance_list.append(model.feature_importances_)

print(f"\n{N_FOLDS}-Fold CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"各折 AUC: {[f'{s:.4f}' for s in cv_scores]}")

oof_auc = roc_auc_score(y, oof_predictions)
print(f"\nOOF AUC: {oof_auc:.4f}")

oof_pred_labels = (oof_predictions > 0.5).astype(int)

print("\n" + "="*50)
print("模型评估结果 (OOF 预测)")
print("="*50)
print(f"AUC-ROC: {oof_auc:.4f}")
print(f"准确率: {accuracy_score(y, oof_pred_labels):.4f}")
print(f"精确率: {precision_score(y, oof_pred_labels):.4f}")
print(f"召回率: {recall_score(y, oof_pred_labels):.4f}")
print(f"F1分数: {f1_score(y, oof_pred_labels):.4f}")

print("\n混淆矩阵:")
cm = confusion_matrix(y, oof_pred_labels)
print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")

print("\n分类报告:")
print(classification_report(y, oof_pred_labels, target_names=['负样本', '正样本']))

avg_feature_importance = np.mean(feature_importance_list, axis=0)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': avg_feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 20 重要特征:")
print(feature_importance.head(20).to_string(index=False))

model_save_path = os.path.join(workspace, 'models', 'lgb_model.txt')
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

final_model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    **best_params
)
final_model.fit(X, y)
final_model.booster_.save_model(model_save_path)
print(f"\n全量数据训练模型已保存至: {model_save_path}")

importance_save_path = os.path.join(workspace, 'data', 'feature_importance.csv')
feature_importance.to_csv(importance_save_path, index=False)
print(f"特征重要性已保存至: {importance_save_path}")

