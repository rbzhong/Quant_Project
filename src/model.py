#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject
@File    ：model.py
@IDE     ：PyCharm
@Author  ：钟若冰
@Date    ：2026/4/20 19:07
@Describe：本文件定义模型结构，包括 GRU 和 BiAGRU
"""
import torch
import torch.nn as nn

class Gru(nn.Module):
    def __init__(
        self,
        *,
        input_size=2,
        hidden_size=32,
        num_layers=2,# 两层gru
        drpt_rate=0.1,
    ):
        super().__init__()
        gru_dropout = drpt_rate if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.fltn = nn.Flatten(0, -1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fltn(self.fc(x[:, -1, :]))
        return x


class BiAGru(nn.Module):
    def __init__(
        self,
        input_size = 2,
        hidden_size = 32,
        num_layers = 2,
        num_heads = 2,
        drpt_rate = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = drpt_rate,
            bidirectional = True,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim = hidden_size*2,
            kdim = None,
            vdim = None, # None使用embed_dim
            num_heads = num_heads,
            batch_first = True,
            dropout = drpt_rate,
            bias = True,
            add_bias_kv=False,
            add_zero_attn=False,
        )
        self.lin = nn.Sequential(
            nn.Linear(hidden_size*2, 1),
            nn.Flatten(start_dim=0, end_dim=-1),
        )
    def forward(self, x):
        x, _ = self.gru(x)
        x, _ = self.attn(x,x,x)
        x = x[:, -1, :]
        x = self.lin(x)
        return x

if __name__=='__main__':
    x = torch.randn(2,4,3)
    # model=BiAGru(input_size = 3,
    #     hidden_size = 8,
    #     num_layers = 2,
    #     num_heads = 2,
    #     drpt_rate = 0.,)
    # yp = model(x)
    model2=Gru(input_size=3,
        hidden_size=64,
        num_layers=2,
        drpt_rate = 0)
    yp=model2(x)