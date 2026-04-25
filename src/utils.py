#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject
@File    ：main1.py
@IDE     ：PyCharm
@Author  ：钟若冰
@Date    ：2026/4/20 19:07
@Describe：本文件为工具函数文件，包括创建路径，一些因子算子，已经图片拼接函数
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import math
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def rank_cross(df, col, date_col="TradingDay"):
    return df.groupby(date_col)[col].rank(pct=True, method="average")


def delta(series, n, by):
    return series.groupby(by).diff(n)


def ts_sum(series, n, by):
    return series.groupby(by).rolling(n, min_periods=n).sum().reset_index(level=0, drop=True)


def ts_mean(series, n, by):
    return series.groupby(by).rolling(n, min_periods=n).mean().reset_index(level=0, drop=True)


def ts_std(series, n, by):
    return series.groupby(by).rolling(n, min_periods=n).std().reset_index(level=0, drop=True)


def ts_min(series, n, by):
    return series.groupby(by).rolling(n, min_periods=n).min().reset_index(level=0, drop=True)


def ts_max(series, n, by):
    return series.groupby(by).rolling(n, min_periods=n).max().reset_index(level=0, drop=True)


def sma_cn(series, n, m, by):
    # 通达信/Alpha191 常见 SMA(x,n,m) = EMA(alpha=m/n)
    return series.groupby(by, group_keys=False).apply(lambda x: x.ewm(alpha=m / n, adjust=False).mean())


def decay_linear(series, n, by):
    weights = np.arange(1, n + 1, dtype=float)
    denom = weights.sum()
    return series.groupby(by).rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, weights) / denom), raw=True).reset_index(level=0, drop=True)


def rolling_corr(x, y, n, by):
    out = pd.Series(index=x.index, dtype=float)
    for sid, idx in by.groupby(by).groups.items():
        xg = x.loc[idx]
        yg = y.loc[idx]
        out.loc[idx] = xg.rolling(n, min_periods=n).corr(yg)
    return out


def rolling_cov(x, y, n, by):
    out = pd.Series(index=x.index, dtype=float)
    for sid, idx in by.groupby(by).groups.items():
        xg = x.loc[idx]
        yg = y.loc[idx]
        out.loc[idx] = xg.rolling(n, min_periods=n).cov(yg)
    return out

def make_image_grid(image_dir, output_path, n_cols=2, thumb_width=900, thumb_height=300, padding=20, add_filename=True):
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    image_paths = sorted(image_dir.glob('*.png'))

    if len(image_paths) == 0:
        raise ValueError(f'没有找到图片：{image_dir}')

    text_height = 35 if add_filename else 0
    n_rows = math.ceil(len(image_paths) / n_cols)

    cell_width = thumb_width + padding * 2
    cell_height = thumb_height + text_height + padding * 2

    canvas = Image.new('RGB', (n_cols * cell_width, n_rows * cell_height), 'white')
    draw = ImageDraw.Draw(canvas)

    for i, img_path in enumerate(image_paths):
        row = i // n_cols
        col = i % n_cols

        x0 = col * cell_width + padding
        y0 = row * cell_height + padding

        img = Image.open(img_path).convert('RGB')
        img.thumbnail((thumb_width, thumb_height))

        x = x0 + (thumb_width - img.width) // 2
        y = y0 + (thumb_height - img.height) // 2

        canvas.paste(img, (x, y))

        if add_filename:
            draw.text((x0, y0 + thumb_height + 8), img_path.stem[:80], fill='black')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(f'拼图完成：{output_path}')
    print(f'共拼接 {len(image_paths)} 张图，{n_rows} 行 × {n_cols} 列')


if __name__ == '__main__':
    make_image_grid(
        image_dir=r'F:\PythonProjects\text\results\nolimit',
        output_path=r'F:\PythonProjects\text\results\nolimit_grid.jpg',
        n_cols=3,
        thumb_width=600,
        thumb_height=200,
        padding=20,
        add_filename=True
    )