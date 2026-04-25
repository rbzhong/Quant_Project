#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject
@File    ：data_loader.py
@IDE     ：PyCharm
@Author  ：钟若冰
@Date    ：2026/4/20 19:16
@Describe：本文件用于加载数据、增加复权价以及预处理
"""
import numpy as np
import pandas as pd


# backward compatible helper kept from the original project style

def build_samples_by_group(df,code_col='code',date_col='date',window_size=5):
    x = []
    y = []

    df = df.sort_values([code_col, date_col]).reset_index(drop=True)

    for _, g in df.groupby(code_col):
        g = g.sort_values(date_col).reset_index(drop=True)
        if len(g) < window_size:
            continue

        x_code = g.iloc[:, 2:-1].values.astype(np.float32)
        y_code = g.iloc[:, -1].values.astype(np.float32)

        for i in range(window_size - 1, len(g)):
            x_code_i = x_code[i - window_size + 1: i + 1]
            y_code_i = y_code[i]
            x.append(x_code_i[np.newaxis, :, :])
            y.append(y_code_i[np.newaxis])

    x = np.concatenate(x, axis=0).astype('float32')
    y = np.concatenate(y, axis=0).astype('float32')
    return x, y


# data_loader.py
def load_raw_data(path):
    df = pd.read_parquet(path)
    #df = df.loc[(df['TradingDay'] >= '2023-01-01') & (df['StockID'] <= 12408)]
    return df


def preprocess_basic(df):
    df = df.copy()
    df['TradingDay'] = pd.to_datetime(df['TradingDay'])
    df = df.sort_values(['StockID', 'TradingDay']).reset_index(drop=True)

    # 类型转换
    num_cols = [
        'PrevClosePrice', 'OpenPrice', 'HighPrice',
        'LowPrice', 'ClosePrice', 'Volume', 'Amount'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def add_adj_price(df):
    df = df.copy()
    eps = 1e-12

    # 单期复权因子
    df['AdjFactor1d'] = df.groupby('StockID')['ClosePrice'].shift(1) / (df['PrevClosePrice'] + eps)

    df['AdjFactor1d'] = df['AdjFactor1d'].fillna(1.0)

    # 累乘
    df['AdjFactor'] = df.groupby('StockID')['AdjFactor1d'].cumprod()

    # 复权价
    for col in ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']:
        df[f'Adj{col}'] = df[col] * df['AdjFactor']

    return df


def add_label(df, label_horizon=5):
    df = df.copy()

    df[f'target_{label_horizon}d'] = df.groupby('StockID')['AdjClosePrice'].shift(-(label_horizon + 1))/ df.groupby('StockID')['AdjClosePrice'].shift(-1) - 1
    return df


def load_data(path, label_horizon=5):
    df = load_raw_data(path)
    df = preprocess_basic(df)
    df = add_adj_price(df)
    df = add_label(df, label_horizon)

    return df


def make_sequence_dataset(df, feature_cols, window_size = 20):
    sequences = []
    targets = []
    metas = []

    df = df.sort_values(['StockID', 'TradingDay']).reset_index(drop=True)
    for stock_id, group in df.groupby('StockID'):
        group = group.reset_index(drop=True)
        values = group[feature_cols].to_numpy(dtype=np.float32)
        #labels = group['ret_5'].to_numpy(dtype=np.float32)
        labels = group['target_5d'].to_numpy(dtype=np.float32)

        for idx in range(window_size - 1, len(group)):
            seq = values[idx - window_size + 1: idx + 1]
            if np.isnan(seq).any() or np.isnan(labels[idx]):
                continue
            sequences.append(seq)
            targets.append(labels[idx])
            metas.append({
                'StockID': stock_id,
                'TradingDay': group.loc[idx, 'TradingDay'],
                'ClosePrice': float(group.loc[idx, 'ClosePrice']),
                'IndustryName': group.loc[idx, 'IndustryName'],
            })

    x = np.stack(sequences).astype(np.float32)
    y = np.array(targets, dtype=np.float32)
    meta = pd.DataFrame(metas)
    return x, y, meta
