import pandas as pd
import numpy as np
import os

# 当前文件的绝对位置
current_file_path = __file__

# workspace 为当前文件的绝对位置的父目录的父目录
workspace = os.path.dirname(os.path.dirname(current_file_path))

def clean_df(df):
    for col in df.columns:
        # 步骤1: 检查原始列（替换前）是否有混合列（既有数字又有非None非数字）
        non_missing = df[col].dropna()
        has_number = False
        has_non_number = False
        
        for value in non_missing:
            # 尝试将值转换为float（判断是否为数字）
            try:
                float_value = float(value)
                has_number = True
            except (TypeError, ValueError):
                has_non_number = True
        
        # 如果是混合列（既有数字又有非数字），打印列名
        if has_number and has_non_number:
            print(f"Column '{col}' contains both numbers and non-numeric values (non-None fields)")
        
        
        if has_number:
            col_series = pd.to_numeric(df[col], errors='coerce')
            col_series = col_series.apply(lambda x: None if x == -9999 else x)
            df[col] = col_series.astype('float64')
        elif has_non_number:
            df[col] = df[col].astype('object')

train_df_path = os.path.join(workspace, 'parquet_data', 'train_data.parquet')
test_df_path = os.path.join(workspace, 'parquet_data', 'test_data.parquet')

train_df = pd.read_parquet(train_df_path)
test_df = pd.read_parquet(test_df_path)

# 对train_df进行清洗
clean_df(train_df)

# 对test_df进行清洗
clean_df(test_df)

# 保存清洗后的DataFrame到csv文件
train_df.to_csv(os.path.join(workspace, 'data', 'train.csv'), index=False)
test_df.to_csv(os.path.join(workspace, 'data', 'test.csv'), index=False)

print("clean done")
