#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject
@File    ：model.py
@IDE     ：PyCharm
@Author  ：钟若冰
@Date    ：2026/4/20 19:08
@Describe：本文件用于训练模型
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# original name preserved for compatibility

def train_model(model,x_trn,y_trn,x_val,y_val,epcohs=20,batch_size=32,lr=2.5e-4,loss_type='MSE',patience=3,device='cpu',model_pth='results/model.pth'):
    model = model.to(device)
    train_ds = TensorDataset(torch.tensor(x_trn), torch.tensor(y_trn))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_type=='MSE':
        criterion = nn.MSELoss()
    elif loss_type=='L1':
        criterion = nn.L1Loss()

    history = []
    best_val_loss = float('inf')
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epcohs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_losses.append(loss.item())

        train_loss = float(sum(train_losses) / max(1, len(train_losses)))
        val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'Epoch {epoch}/{epcohs} - train_loss={train_loss:.6f} val_loss={val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_pth)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print('Early stopping triggered.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_model(model, x, device='cpu', batch_size=512):
    model = model.to(device)
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            x_tensor = torch.tensor(x_batch,dtype=torch.float32,device=device)
            pred = model(x_tensor)
            preds_list.append(pred.detach().cpu())
            del x_tensor, pred
    return torch.cat(preds_list, dim=0).numpy()
