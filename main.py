#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject
@File    ：main.py
@IDE     ：PyCharm
@Author  ：钟若冰
@Date    ：2026/4/22 17:01
@Describe：项目主程序，运行将完整计算因子，训练模型，回测的全过程
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import pandas as pd
import warnings
import torch
import gc

from src.backtest import prepare_backtest, topkdrop_holdings, weekly_backtest, calculate_metrics, plot_curve
from src.data_loader import make_sequence_dataset, load_data
from src.features import FEATURE_COLUMNS, FEATURE_COLUMNS_IND,FEATURE_COLUMNS_ALPHA191,build_daily_factors, normalize_features,build_daily_factors_ts
from src.model import BiAGru, Gru
from src.train import predict_model, train_model
from src.utils import ensure_dir
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results'
REPORTS_DIR = BASE_DIR / 'reports'
DATA_DIR=BASE_DIR / 'data' / 'Quote.parquet'
FEATURE_DIR=BASE_DIR / 'data' / 'feature.parquet'

def get_rolling_windows():
    return [
        {
            'window_id': 1,
            'train_start': '2019-01-01',
            'train_end': '2020-12-31',
            'val_start': '2021-01-01',
            'val_end': '2021-12-31',
            'test_start': '2022-01-01',
            'test_end': '2022-12-31',
        },
        {
            'window_id': 2,
            'train_start': '2020-01-01',
            'train_end': '2021-12-31',
            'val_start': '2022-01-01',
            'val_end': '2022-12-31',
            'test_start': '2023-01-01',
            'test_end': '2023-12-31',
        }
    ]


def run_single_window(raw_df,model_name,hidden_size_candidates,window_cfg,feature_type,label_type,features_group,device='cpu'):
    if FEATURE_DIR.exists():
        feature_df = pd.read_parquet(FEATURE_DIR)
    else:
        feature_df = build_daily_factors(raw_df)
        feature_df=build_daily_factors_ts(feature_df)
        feature_df.to_parquet(FEATURE_DIR)
    #feature_df = normalize_features(feature_df, train_end=window_cfg['train_end'])
    feature_df = normalize_features(feature_df,window_cfg['train_end'],feature_type=feature_type,label_type=label_type).fillna(0)
    if features_group=='gp1':
        final_columns=FEATURE_COLUMNS
    elif features_group=='gp2':
        final_columns = FEATURE_COLUMNS + FEATURE_COLUMNS_IND
    elif features_group=='gp3':
        final_columns=FEATURE_COLUMNS + FEATURE_COLUMNS_ALPHA191+FEATURE_COLUMNS_IND

    model_df = feature_df.dropna(subset=final_columns + ['target_5d']).copy()
    x, y, meta = make_sequence_dataset(model_df, final_columns, window_size=20)

    meta = meta.copy()
    meta['TradingDay'] = pd.to_datetime(meta['TradingDay'])

    train_mask = (meta['TradingDay'] >= pd.Timestamp(window_cfg['train_start'])) & (meta['TradingDay'] <= pd.Timestamp(window_cfg['train_end']))
    val_mask = (meta['TradingDay'] >= pd.Timestamp(window_cfg['val_start'])) & (meta['TradingDay'] <= pd.Timestamp(window_cfg['val_end']))
    test_mask = (meta['TradingDay'] >= pd.Timestamp(window_cfg['test_start'])) & (meta['TradingDay'] <= pd.Timestamp(window_cfg['test_end']))

    x_train, y_train = x[train_mask.values], y[train_mask.values]
    x_val, y_val = x[val_mask.values], y[val_mask.values]
    x_test, y_test = x[test_mask.values], y[test_mask.values]

    test_meta = meta.loc[test_mask].reset_index(drop=True)

    best_val_loss = None
    best_hidden_size = None
    best_model = None
    all_histories = []

    # hidden_size=32/64
    for hidden_size in hidden_size_candidates:
        name = model_name.split('_')[0].lower()
        if name == 'gru':
            model = Gru(input_size=len(final_columns), hidden_size=hidden_size, num_layers=2, drpt_rate=0.1)
        if name in {'bigru', 'biagru'}:
            model = BiAGru(input_size=len(final_columns), hidden_size=hidden_size, num_layers=2, drpt_rate=0.1)
        # raise ValueError(f'Unsupported model: {model_name}')
        # model = build_model(model_name, input_size=len(final_columns),hidden_size=hidden_size)

        model_path = RESULTS_DIR / f"{model_name.lower()}_win{window_cfg['window_id']}_hs{hidden_size}.pth"

        history = train_model(
            model=model,
            x_trn=x_train,
            y_trn=y_train,
            x_val=x_val,
            y_val=y_val,
            epcohs=20,
            batch_size=1024,
            lr=2.5e-4,
            patience=5,
            device=device,
            model_pth=str(model_path),
        )

        hist_df = pd.DataFrame(history)
        hist_df['model_name'] = model_name
        hist_df['hidden_size'] = hidden_size
        hist_df['window_id'] = window_cfg['window_id']
        all_histories.append(hist_df)

        cur_val_loss = hist_df['val_loss'].min()

        if (best_val_loss is None) or (cur_val_loss < best_val_loss):
            best_val_loss = cur_val_loss
            best_hidden_size = hidden_size
            best_model = model

    # 最优超参预测当前窗口测试集
    predictions = predict_model(best_model, x_test, device=device)
    del best_model
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    pred_df = test_meta.copy()
    pred_df['prediction'] = predictions
    pred_df['target_5d'] = y_test
    pred_df['model_name'] = model_name
    pred_df['window_id'] = window_cfg['window_id']
    pred_df['best_hidden_size'] = best_hidden_size
    pred_df['best_val_loss'] = best_val_loss

    history_df = pd.concat(all_histories, ignore_index=True)

    return pred_df, history_df




def main():
    ensure_dir(str(RESULTS_DIR))
    ensure_dir(str(REPORTS_DIR))

    #raw_df = load_data(r'E:\zrb\pythonProject\ai\zrb\zrb\Quote.parquet')
    # raw_df = load_data(r'/opt/zhongruobing/jupyter/zrb/Quote.parquet')
    raw_df=load_data(DATA_DIR)
    #raw_df.to_csv(RESULTS_DIR / 'raw_quote_data.csv', index=False, encoding='utf-8-sig')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size_candidates = [32, 64]

    summary_list = []

    for model_name in ['gru']:#,'biagru']:
        for features_group in ['gp3']:#,'gp3']:#,'gp2','gp3']:
            for feature_type in ['cross']:  # 'train','cross',
                for label_type in ['ind']:#,'normalize''raw','ind']:
                    model_run_name = f'{model_name}_{features_group}_{feature_type}_{label_type}'
                    rolling_windows = get_rolling_windows()
                    all_preds = []
                    all_histories = []
                    for window_cfg in rolling_windows:
                        pred_df, history_df = run_single_window(raw_df=raw_df,model_name=model_run_name,hidden_size_candidates=hidden_size_candidates,window_cfg=window_cfg,feature_type=feature_type,label_type=label_type,features_group=features_group,device=device)
                        all_preds.append(pred_df)
                        all_histories.append(history_df)
                    gc.collect()
                    if str(device).startswith('cuda'):
                        torch.cuda.empty_cache()
                    pred_all = pd.concat(all_preds, ignore_index=True)
                    history_all = pd.concat(all_histories, ignore_index=True)
                    history_all.to_csv(RESULTS_DIR / f'{model_run_name.lower()}_rolling_training_history.csv',index=False,encoding='utf-8-sig')
                    pred_all.to_csv(RESULTS_DIR / f'{model_run_name.lower()}_rolling_predictions.csv',index=False,encoding='utf-8-sig')
                    backtest_input = raw_df.merge(pred_all[['TradingDay', 'StockID', 'prediction']],on=['TradingDay', 'StockID'],how='inner')
                    backtest_input = prepare_backtest(backtest_input)
                    weights = topkdrop_holdings(backtest_input, pred_col='prediction', topk=200, dropk=50)
                    backtest_df = weekly_backtest(weights, backtest_input, cost_rate=0.003)
                    backtest_df.to_csv(RESULTS_DIR / f'{model_run_name.lower()}_rolling_backtest_timeseries.csv',index=False,encoding='utf-8-sig')
                    metrics = calculate_metrics(backtest_df)
                    metrics_df = pd.DataFrame({
                        'metric': metrics.keys(),
                        'value': metrics.values(),
                    })
                    metrics_df.to_csv(RESULTS_DIR / f'{model_run_name.lower()}_rolling_backtest_results.csv',index=False,encoding='utf-8-sig')
                    equity_curve_path = RESULTS_DIR / f'{model_run_name.lower()}_rolling_equity_curve.png'
                    plot_curve(backtest_df, str(equity_curve_path), title=f'{model_run_name} Rolling Strategy')

                    metric_map = dict(zip(metrics_df['metric'], metrics_df['value']))
                    metric_map['model_name'] = model_name+features_group+label_type
                    summary_list.append(metric_map)

                    print('=' * 80)
                    print(f'Model: {model_name}')
                    print(metrics_df.to_string(index=False))

    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(RESULTS_DIR / 'model_compare_summary.csv',index=False,encoding='utf-8-sig')

    print('=' * 80)
    print('Project pipeline finished successfully.')
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()