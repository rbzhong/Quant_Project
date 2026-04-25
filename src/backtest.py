#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pythonProject
@File    ：backtest.py
@IDE     ：PyCharm
@Author  ：钟若冰
@Date    ：2026/4/21 18:16
@Describe：本文件用于模型回测并绘制图像
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import make_sequence_dataset,load_data

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False


# 预处理
def prepare_backtest(df, date_col='TradingDay', stock_col='StockID'):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(['StockID', 'TradingDay']).reset_index(drop=True)

    # 下个交易日原始 OHLC
    data['next_open'] = data.groupby(stock_col)['OpenPrice'].shift(-1)
    data['next_high'] = data.groupby(stock_col)['HighPrice'].shift(-1)
    data['next_low'] = data.groupby(stock_col)['LowPrice'].shift(-1)

    # 下个交易日复权收盘价
    data['next_close_adj'] = data.groupby(stock_col)['AdjClosePrice'].shift(-1)

    # 剔除下个交易日停牌股票：开高低都为0
    c1 = (data['next_open'].fillna(-1) == 0)
    c2 = (data['next_high'].fillna(-1) == 0)
    c3 = (data['next_low'].fillna(-1) == 0)
    data['is_trade'] = ~(c1 | c2 | c3)

    # 要使用每周最后一个交易日的信号
    data['week'] = data[date_col].dt.to_period('W-FRI')
    week_last = data.groupby('week')[date_col].max().rename('rebalance_day').reset_index()
    data = data.merge(week_last, on='week', how='left')
    data = data[data[date_col] == data['rebalance_day']].copy()
    return data


# 生成持仓
def topkdrop_holdings(df, pred_col='prediction', date_col='TradingDay', stock_col='StockID', topk=200, dropk=50):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([date_col, stock_col]).reset_index(drop=True)
    c1 = data[pred_col].notna()
    c = c1 & (data['is_trade'])
    data = data.loc[c, [date_col, stock_col, pred_col, 'next_close_adj']].copy()
    rebalance_dates = sorted(data[date_col].unique())
    all_results = []
    prev_weights = set()

    for i, dt in enumerate(rebalance_dates):
        tmp = data[data[date_col] == dt].copy()
        tmp = tmp.sort_values(pred_col, ascending=False).reset_index(drop=True)
        if i == 0:
            selected = tmp.head(topk).copy()
        else:
            old_pf = tmp[tmp[stock_col].isin(prev_weights)].copy()
            new_pf = tmp[~tmp[stock_col].isin(prev_weights)].copy()
            new_pf = new_pf.head(dropk)  # 实现换手约束
            candidate_pool = pd.concat([old_pf, new_pf], axis=0, ignore_index=True)
            candidate_pool = candidate_pool.sort_values([pred_col, stock_col], ascending=[False, True]).reset_index(
                drop=True)
            selected = candidate_pool.head(topk).copy()

        n = len(selected)
        selected['weight'] = 1.0 / n if n > 0 else 0.0
        all_results.append(selected[[date_col, stock_col, pred_col, 'weight', 'next_close_adj']])
        prev_weights = set(list(selected[stock_col]))

    weights = pd.concat(all_results, axis=0, ignore_index=True)
    return weights


# 计算收益
def weekly_backtest(weights, price_df, cost_rate=0.003):
    w = weights.copy()
    w = w.pivot(index='TradingDay', columns='StockID', values='weight').fillna(0)
    price = price_df.copy()
    price = price.pivot(index='TradingDay', columns='StockID', values='next_close_adj')
    common_dates = w.index.intersection(price.index)
    common_cols = w.columns.union(price.columns)
    w = w.loc[common_dates].reindex(columns=common_cols).copy()
    price = price.loc[common_dates].reindex(columns=common_cols).copy()
    stock_ret = price.shift(-1) / price - 1
    stock_ret = stock_ret.iloc[:-1]  # 去掉nan
    w = w.iloc[:-1]

    strategy_ret = (w * stock_ret).sum(axis=1)
    dw = w.diff().abs().sum(axis=1)
    dw.iloc[0] = w.iloc[0].abs().sum()
    cost = dw * cost_rate
    strategy_ret = strategy_ret - cost
    benchmark_ret = stock_ret.mean(axis=1)
    excess_ret = strategy_ret - benchmark_ret

    bkt = pd.DataFrame({
        'cost': cost,
        'turnover': dw/2,
        'strategy_ret': strategy_ret,
        'benchmark_ret': benchmark_ret,
        'excess_ret': excess_ret,
    })
    bkt.index.name = 'TradingDay'
    bkt['strategy_nav'] = (1 + bkt['strategy_ret'].fillna(0)).cumprod()
    bkt['benchmark_nav'] = (1 + bkt['benchmark_ret'].fillna(0)).cumprod()
    bkt['excess_nav'] = (1 + bkt['excess_ret'].fillna(0)).cumprod()
    return bkt


# 计算指标
def calculate_metrics(backtest_df):
    bkt = backtest_df.copy()
    strat = bkt['strategy_ret'].dropna()
    bench = bkt['benchmark_ret'].dropna()
    excess = bkt['excess_ret'].dropna()

    yield_ = (1 + strat).prod() ** (52 / len(strat)) - 1
    base_yield = (1 + bench).prod() ** (52 / len(bench)) - 1
    excess_yield = (1 + excess).prod() ** (52 / len(excess)) - 1
    vol = strat.std(ddof=1) * np.sqrt(52)
    sharpe = yield_ / vol
    te = excess.std(ddof=1) * np.sqrt(52)
    ir = excess_yield / te
    nav = bkt['excess_nav'].dropna()
    dd = nav / nav.cummax() - 1
    maxdd = dd.min()
    end = dd.idxmin()
    start = nav.loc[:end].idxmax()
    before = nav.loc[:start]
    after = nav.loc[end:]
    dd2_before = (before / before.cummax() - 1).min() if not before.empty else 0
    dd2_after = (after / after.cummax() - 1).min() if not after.empty else 0
    maxdd2 = min(dd2_before, dd2_after)

    # 次大回撤的起止时间
    second_dd_start = None
    second_dd_end = None
    if dd2_before <= dd2_after and not before.empty:
        second_dd_end = (before / before.cummax() - 1).idxmin()
        second_dd_start = before.loc[:second_dd_end].idxmax()
    elif not after.empty:
        second_dd_end = (after / after.cummax() - 1).idxmin()
        second_dd_start = after.loc[:second_dd_end].idxmax()

    monthly_excess = (1 + bkt['excess_ret']).groupby(pd.Grouper(freq='M')).prod() - 1
    monthly_winrate = (monthly_excess > 0).mean()
    turnover = bkt.iloc[1:,:]['turnover'].mean() * 52
    cost = bkt['cost'].mean()

    return {
        'yield': yield_,
        'base_yield': base_yield,
        'excess_yield': excess_yield,
        'vol': vol,
        'sharpe': sharpe,
        'te': te,
        'ir': ir,
        'maxdd': -maxdd,
        'Max Drawdown Start': start,
        'Max Drawdown End': end,
        'maxdd2': -maxdd2,
        'Second Drawdown Start': second_dd_start,
        'Second Drawdown End': second_dd_end,
        'monthly_winrate': monthly_winrate,
        'turnover': turnover,
        'cost': cost,
    }

# 绘制净值曲线
def plot_curve(backtest_df, save_path,title='Weekly Backtest'):
    bkt = backtest_df.copy()
    metrics = calculate_metrics(bkt)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(bkt.index, bkt['strategy_nav'], label='Strategy')
    ax.plot(bkt.index, bkt['benchmark_nav'], label='Benchmark')
    ax.plot(bkt.index, bkt['excess_nav'], label='Excess')

    # 最大回撤区间
    if pd.notna(metrics['Max Drawdown Start']) and pd.notna(metrics['Max Drawdown End']):
        ax.axvspan(
            metrics['Max Drawdown Start'],
            metrics['Max Drawdown End'],
            facecolor='red',
            alpha=0.3,
            label='Max Drawdown'
        )

    # 次大回撤区间
    if pd.notna(metrics['Second Drawdown Start']) and pd.notna(metrics['Second Drawdown End']):
        ax.axvspan(
            metrics['Second Drawdown Start'],
            metrics['Second Drawdown End'],
            facecolor='red',
            alpha=0.1,
            label='Second Drawdown'
        )

    title_text = (
        f'{title}\n'
        f"Excess Return {metrics['excess_yield']:.2%}  "
        f"IR {metrics['ir']:.2f}  "
        f"Sharpe {metrics['sharpe']:.2f}  "
        f"Max Drawdown {metrics['maxdd']:.2%}  "
        f"2nd Max Drawdown {metrics['maxdd2']:.2%}  "
        f"Annual Turnover {metrics['turnover']:.2%}  "
        f"Monthly Win Rate {metrics['monthly_winrate']:.2%}"
    )

    ax.set_title(title_text)
    ax.set_xlabel('TradingDay')
    ax.set_ylabel('NAV')
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    #return fig, ax


if __name__ == '__main__':
    dropk_dir={'nolimit':200,'limit25':50,'limit10':20}
    for limit in ['nolimit','limit25','limit10']:
        summary_list = []
        for model_name in ['gru','biagru']:
            for features_group in ['gp1','gp2','gp3']:#,'gp2','gp3']:
                for feature_type in ['train','cross']:  # 'cross',
                    for label_type in ['raw','normalize','ind']:
                        pred_all = pd.read_csv(f'F:\PythonProjects\\text\\results\\{model_name}_{features_group}_{feature_type}_{label_type}_rolling_predictions.csv')
                        pred_all['TradingDay']=pd.to_datetime(pred_all['TradingDay'])
                        #pre = pd.read_csv('F:\PythonProjects\\text\\results\gru_gp1_train_ind_rolling_predictions.csv')
                        raw_df=load_data('F:\PythonProjects\\text\\data\\Quote.parquet')
                        backtest_input = raw_df.merge(pred_all[['TradingDay', 'StockID', 'prediction']], on=['TradingDay', 'StockID'],how='inner')
                        backtest_input = prepare_backtest(backtest_input)
                        weights = topkdrop_holdings(backtest_input, pred_col='prediction', topk=200, dropk=dropk_dir[limit])
                        backtest_df = weekly_backtest(weights, backtest_input, cost_rate=0.003)
                        backtest_df.to_csv(f'F:\PythonProjects\\text\\results\\{limit}\\{model_name}_{features_group}_{feature_type}_{label_type}_drop{dropk_dir[limit]}_rolling_backtest_timeseries.csv', index=False)
                        metrics = calculate_metrics(backtest_df)
                        metrics_df = pd.DataFrame({
                            'metric': metrics.keys(),
                            'value': metrics.values(),
                        })
                        metrics_df.to_csv(f'F:\PythonProjects\\text\\results\\{limit}\\{model_name.lower()}_{features_group}_{feature_type}_{label_type}_drop{dropk_dir[limit]}_rolling_backtest_results.csv', index=False)
                        equity_curve_path = f'F:\PythonProjects\\text\\results\\{limit}\\{model_name.lower()}_{features_group}_{feature_type}_{label_type}_drop{dropk_dir[limit]}_rolling_equity_curve.png'
                        plot_curve(backtest_df, str(equity_curve_path), title=f'{model_name} drop{dropk_dir[limit]} Rolling Strategy')
                        metric_map = dict(zip(metrics_df['metric'], metrics_df['value']))
                        metric_map['model_name'] = model_name + features_group + label_type
                        summary_list.append(metric_map)
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv(f'F:\PythonProjects\\text\\results\\{limit}\\model_drop{dropk_dir[limit]}_compare_summary.csv', index=False)


    # data=load_data('E:\\zrb\\pythonProject\\ai\\zrb\\zrb\\Quote.parquet')
    # data['prediction']=-data.groupby('StockID')['AdjClosePrice'].pct_change()
    # data=prepare_backtest(data)
    # weights=topkdrop_holdings(data,dropk=200)
    # bkt = weekly_backtest(weights, data, cost_rate=0.003)
    # fig, ax = plot_curve(bkt, 'E:\\zrb\\pythonProject\\ai\\zrb\\zrb\\results\\test_plot.png',title='TopKDrop Weekly Strategy')
    # plt.show()