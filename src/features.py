#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject
@File    ：features.py
@IDE     ：PyCharm
@Author  ：钟若冰
@Date    ：2026/4/20 14:34
@Describe：本文件用于计算特征集
"""
from src.utils import *
FEATURE_COLUMNS = [
    'f_intraday_ret',
    'f_gap_ret',
    'f_day_ret',
    'f_amplitude',
    'f_upper_shadow',
    #'f_lower_shadow',#abs(rankic)<0.003
    'f_body_ratio',
    'f_close_pos',
    'f_open_pos',
    'f_close_vs_avgprice',
    'f_avgprice_pos',
    'f_log_volume',
    'f_log_amount',
    'f_vol_x_intraday_ret',
    'f_amt_x_intraday_ret',
    'f_strong_close_vol',
    'f_signed_body',
    'f_gap_intraday_interact',
    'f_shadow_imbalance',
    'f_intraday_ret_rank',
    'f_amplitude_rank',
    'f_close_vs_avgprice_rank',
    'f_intraday_ret_ind_neutral',
    'f_amplitude_ind_neutral',
    'f_close_vs_avgprice_ind_neutral',
    'f_log_amount_ind_neutral',
    #'f_ind_excess_ret_x_amt',#abs(rankic)<0.003
]
FEATURE_COLUMNS_ALPHA191 = ['alpha003', 'alpha009', 'alpha014', 'alpha018', 'alpha019', 'alpha020',
                            'alpha021', 'alpha022', 'alpha024', 'alpha027', 'alpha029', 'alpha031',
                            'alpha032', 'alpha036',  'alpha043', 'alpha046', 'alpha049', 'alpha050',
                            'alpha051', 'alpha052', 'alpha053', 'alpha058', 'alpha059', 'alpha063',
                            'alpha065', 'alpha066', 'alpha067', 'alpha068', 'alpha069', 'alpha071',
                            'alpha078', 'alpha079', 'alpha080', 'alpha081', 'alpha084', 'alpha088',
                            'alpha089', 'alpha093', 'alpha097', 'alpha098', 'alpha099', 'alpha100',
                            'alpha102', 'alpha103', 'alpha106', 'alpha109', 'alpha110', 'alpha112',
                            'alpha116', 'alpha118', 'alpha128', 'alpha129', 'alpha133', 'alpha134',
                            'alpha136', 'alpha144', 'alpha153', 'alpha158', 'alpha159', 'alpha161',
                            'alpha167', 'alpha168', 'alpha172', 'alpha187', 'alpha189']
FEATURE_COLUMNS_IND=['ind_银行', 'ind_房地产', 'ind_机械设备', 'ind_综合', 'ind_电力设备', 'ind_建筑材料',
                     'ind_家用电器', 'ind_计算机', 'ind_电子', 'ind_交通运输', 'ind_汽车', 'ind_公用事业',
                     'ind_医药生物', 'ind_环保', 'ind_农林牧渔', 'ind_基础化工', 'ind_石油石化', 'ind_有色金属',
                     'ind_商贸零售', 'ind_通信', 'ind_传媒', 'ind_非银金融', 'ind_轻工制造', 'ind_国防军工',
                     'ind_食品饮料', 'ind_钢铁', 'ind_纺织服饰', 'ind_社会服务', 'ind_煤炭', 'ind_建筑装饰',
                     'ind_美容护理']
EPS = 1e-12


def build_daily_factors(df):
    data = df.copy()
    data['AdjFactor1d'] = data.groupby('StockID')['ClosePrice'].shift(1) / (data['PrevClosePrice'] + EPS)
    data['AdjFactor1d'] = data['AdjFactor1d'].fillna(1.0)
    data['AdjFactor'] = data.groupby('StockID')['AdjFactor1d'].cumprod()

    for col in ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']:
        data[f'Adj{col}'] = data[col] * data['AdjFactor']

    data['TradingDay'] = pd.to_datetime(data['TradingDay'])
    data = data.drop(columns=['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice'])
    data = data.rename(columns={'AdjOpenPrice': 'OpenPrice', 'AdjHighPrice': 'HighPrice', 'AdjLowPrice': 'LowPrice', 'AdjClosePrice': 'ClosePrice'})
    data = data.sort_values(['TradingDay', 'StockID']).reset_index(drop=True)

    num_cols = ['PrevClosePrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'Volume', 'Amount']
    for c in num_cols:
        data[c] = pd.to_numeric(data[c], errors='coerce')

    valid_price = data['OpenPrice'].notna() & data['HighPrice'].notna() & data['LowPrice'].notna() & data['ClosePrice'].notna()
    c1=data['OpenPrice'].abs() > EPS
    c2=data['PrevClosePrice'].abs() > EPS
    c3=(data['HighPrice'] - data['LowPrice']).abs() > EPS
    c4=data['avg_price'].notna()
    c5=data['avg_price'].abs() > EPS
    data['avg_price'] = np.where(data['Volume'].abs() > EPS, data['Amount'] / (data['Volume'] + EPS), np.nan)

    data['f_intraday_ret'] = np.where(valid_price & c1, (data['ClosePrice'] - data['OpenPrice']) / (data['OpenPrice'] + EPS), np.nan)
    data['f_gap_ret'] = np.where(valid_price & c2, (data['OpenPrice'] - data['PrevClosePrice']) / (data['PrevClosePrice'] + EPS), np.nan)
    data['f_day_ret'] = np.where(valid_price & c2, (data['ClosePrice'] - data['PrevClosePrice']) / (data['PrevClosePrice'] + EPS), np.nan)
    data['f_amplitude'] = np.where(valid_price & c1, (data['HighPrice'] - data['LowPrice']) / (data['OpenPrice'] + EPS), np.nan)
    data['f_upper_shadow'] = np.where(valid_price & c3, (data['HighPrice'] - np.maximum(data['OpenPrice'], data['ClosePrice'])) / (data['HighPrice'] - data['LowPrice'] + EPS), np.nan)
    data['f_lower_shadow'] = np.where(valid_price & c3, (np.minimum(data['OpenPrice'], data['ClosePrice']) - data['LowPrice']) / (data['HighPrice'] - data['LowPrice'] + EPS), np.nan)
    data['f_body_ratio'] = np.where(valid_price & c3, np.abs(data['ClosePrice'] - data['OpenPrice']) / (data['HighPrice'] - data['LowPrice'] + EPS), np.nan)
    data['f_close_pos'] = np.where(valid_price & c3, (data['ClosePrice'] - data['LowPrice']) / (data['HighPrice'] - data['LowPrice'] + EPS), np.nan)
    data['f_open_pos'] = np.where(valid_price & c3, (data['OpenPrice'] - data['LowPrice']) / (data['HighPrice'] - data['LowPrice'] + EPS), np.nan)
    data['f_close_vs_avgprice'] = np.where(c4 & c5, (data['ClosePrice'] - data['avg_price']) / (data['avg_price'] + EPS), np.nan)
    data['f_avgprice_pos'] = np.where(c4 & c3, (data['avg_price'] - data['LowPrice']) / (data['HighPrice'] - data['LowPrice'] + EPS), np.nan)

    data['f_vol_x_intraday_ret'] = data['Volume'] * data['f_intraday_ret']
    data['f_amt_x_intraday_ret'] = data['Amount'] * data['f_intraday_ret']
    data['f_log_volume'] = np.log1p(data['Volume'].clip(lower=0))
    data['f_log_amount'] = np.log1p(data['Amount'].clip(lower=0))

    cs_rank_cols = ['f_intraday_ret', 'f_amplitude', 'f_close_vs_avgprice']
    for col in cs_rank_cols:
        data[f'{col}_rank'] = data.groupby('TradingDay')[col].rank(method='average', pct=True)

    ind_group = data.groupby(['TradingDay', 'IndustryName'], dropna=False)
    ind_mean_cols = ['f_intraday_ret', 'f_amplitude', 'f_close_vs_avgprice', 'f_log_amount']
    for col in ind_mean_cols:
        data[f'{col}_ind_mean'] = ind_group[col].transform('mean')
        data[f'{col}_ind_neutral'] = data[col] - data[f'{col}_ind_mean']

    data['f_strong_close_vol'] = data['f_close_pos'] * data['f_log_volume']
    data['f_signed_body'] = data['f_intraday_ret'] * data['f_body_ratio']
    data['f_gap_intraday_interact'] = data['f_gap_ret'] * data['f_intraday_ret']
    data['f_shadow_imbalance'] = data['f_upper_shadow'] - data['f_lower_shadow']

    industry_dummies = pd.get_dummies(data['IndustryName'], prefix='ind')
    data = pd.concat([data, industry_dummies], axis=1)
    return data


def build_daily_factors_ts(raw_df: pd.DataFrame) -> pd.DataFrame:
    data = raw_df.copy()
    data['TradingDay'] = pd.to_datetime(data['TradingDay'])
    data = data.sort_values(['StockID', 'TradingDay']).reset_index(drop=True)


    sid = data['StockID']

    # 基础字段
    data['open_'] = data['OpenPrice']
    data['high'] = data['HighPrice']
    data['low'] = data['LowPrice']
    data['close'] = data['ClosePrice']
    data['volume'] = data['Volume']
    data['amount'] = data['Amount']
    data['vwap'] = data['Amount'] / (data['Volume'].replace(0, np.nan) + EPS)
    data['vwap'] = data['vwap'].fillna(data['close'])

    data['close_l1'] = data.groupby('StockID')['close'].shift(1)
    data['open_l1'] = data.groupby('StockID')['open_'].shift(1)
    data['high_l1'] = data.groupby('StockID')['high'].shift(1)
    data['low_l1'] = data.groupby('StockID')['low'].shift(1)
    data['vwap_l1'] = data.groupby('StockID')['vwap'].shift(1)
    data['ret'] = data['close'] / (data['close_l1'] + EPS) - 1

    # # 标签：未来5日收益
    # data['target_5d'] = data.groupby('StockID')['close'].shift(-5) / (data['close'] + EPS) - 1

    # 中间变量
    hl_range = (data['high'] - data['low']).replace(0, np.nan)
    vol = data['volume'].replace(0, np.nan)
    close = data['close']
    open_ = data['open_']
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    ret = data['ret']

    # Alpha191 子集
    alpha_dict = {}

    cond1 = close == data['close_l1']
    cond2 = close > data['close_l1']
    part3 = pd.Series(np.nan, index=data.index)
    part3[cond1] = 0
    part3[cond2] = close[cond2] - np.minimum(low[cond2], data['close_l1'][cond2])
    part3[~(cond1 | cond2)] = close[~(cond1 | cond2)] - np.maximum(high[~(cond1 | cond2)], data['close_l1'][~(cond1 | cond2)])
    alpha_dict['alpha003'] = ts_sum(part3, 6, sid)

    mean8 = ts_mean(close, 8, sid)
    std8 = ts_std(close, 8, sid)
    mean2 = ts_mean(close, 2, sid)
    vol_ratio20 = vol / (ts_mean(vol, 20, sid) + EPS)
    alpha_dict['alpha004'] = np.select(
        [mean8 + std8 < mean2, mean2 < mean8 - std8, vol_ratio20 >= 1],
        [-1, 1, 1],
        default=-1,
    )

    alpha_dict['alpha006'] = -rank_cross(data.assign(tmp=np.sign(delta(open_ * 0.85 + high * 0.15, 4, sid))), 'tmp')

    alpha_dict['alpha009'] = sma_cn(((high + low) / 2 - (data['high_l1'] + data['low_l1']) / 2) * (high - low) / (vol + EPS), 7, 2, sid)

    alpha_dict['alpha011'] = ts_sum(((close - low) - (high - close)) / (hl_range + EPS) * vol, 6, sid)
    alpha_dict['alpha014'] = close - data.groupby('StockID')['close'].shift(5)
    alpha_dict['alpha018'] = close / (data.groupby('StockID')['close'].shift(5) + EPS)

    close_l5 = data.groupby('StockID')['close'].shift(5)
    alpha_dict['alpha019'] = np.select(
        [close < close_l5, close == close_l5, close > close_l5],
        [(close - close_l5) / (close_l5 + EPS), 0, (close - close_l5) / (close + EPS)],
        default=np.nan,
    )
    alpha_dict['alpha020'] = (close - data.groupby('StockID')['close'].shift(6)) / (data.groupby('StockID')['close'].shift(6) + EPS) * 100

    beta_x = np.arange(1, 7, dtype=float)
    alpha_dict['alpha021'] = close.groupby(sid).rolling(6, min_periods=6).mean().reset_index(level=0, drop=True)
    alpha_dict['alpha021'] = (
        alpha_dict['alpha021'].groupby(sid)
        .rolling(6, min_periods=6)
        .apply(lambda y: np.polyfit(beta_x, y, deg=1)[0], raw=True)
        .reset_index(level=0, drop=True)
    )

    mean6 = ts_mean(close, 6, sid)
    norm_dev = (close - mean6) / (mean6 + EPS)
    alpha_dict['alpha022'] = sma_cn(norm_dev - data.groupby('StockID')[norm_dev.name if norm_dev.name else 'close'].shift(3), 12, 1, sid) if False else sma_cn(norm_dev - norm_dev.groupby(sid).shift(3), 12, 1, sid)

    std20 = ts_std(close, 20, sid)
    up_std = std20.where(close > data['close_l1'], 0)
    down_std = std20.where(close <= data['close_l1'], 0)
    alpha_dict['alpha023'] = 100 * sma_cn(up_std, 20, 1, sid) / (sma_cn(up_std, 20, 1, sid) + sma_cn(down_std, 20, 1, sid) + EPS)

    alpha_dict['alpha024'] = sma_cn(close - close.groupby(sid).shift(5), 5, 1, sid)

    alpha_dict['alpha027'] = decay_linear(
        (close - close.groupby(sid).shift(3)) / (close.groupby(sid).shift(3) + EPS) * 100 +
        (close - close.groupby(sid).shift(6)) / (close.groupby(sid).shift(6) + EPS) * 100,
        12,
        sid,
    )

    alpha_dict['alpha028'] = (
        3 * sma_cn((close - ts_min(low, 9, sid)) / (ts_max(high, 9, sid) - ts_min(low, 9, sid) + EPS) * 100, 3, 1, sid)
        - 2 * sma_cn(sma_cn((close - ts_min(low, 9, sid)) / (ts_max(high, 9, sid) - ts_min(low, 9, sid) + EPS) * 100, 3, 1, sid), 3, 1, sid)
    )
    alpha_dict['alpha029'] = (close - close.groupby(sid).shift(6)) / (close.groupby(sid).shift(6) + EPS) * vol
    alpha_dict['alpha031'] = (close - ts_mean(close, 12, sid)) / (ts_mean(close, 12, sid) + EPS) * 100

    alpha_dict['alpha032'] = -ts_sum(
        rank_cross(data.assign(tmp=rolling_corr(rank_cross(data.assign(tmp=high), 'tmp'), rank_cross(data.assign(tmp=vol), 'tmp'), 3, sid)), 'tmp'),
        3,
        sid,
    )

    alpha_dict['alpha036'] = rank_cross(data.assign(tmp=ts_sum(rolling_corr(rank_cross(data.assign(tmp=vol), 'tmp'), rank_cross(data.assign(tmp=vwap), 'tmp'), 6, sid), 2, sid)), 'tmp')

    cond38 = ts_mean(high, 20, sid) < high
    alpha_dict['alpha038'] = np.where(cond38, -delta(high, 2, sid), 0)

    up_vol = vol.where(close > data['close_l1'], 0)
    down_vol = vol.where(close <= data['close_l1'], 0)

    signed_vol = np.select([close > data['close_l1'], close < data['close_l1']], [vol, -vol], default=0)
    alpha_dict['alpha043'] = ts_sum(pd.Series(signed_vol, index=data.index), 6, sid)
    alpha_dict['alpha046'] = (ts_mean(close, 3, sid) + ts_mean(close, 6, sid) + ts_mean(close, 12, sid) + ts_mean(close, 24, sid)) / (4 * close + EPS)
    alpha_dict['alpha047'] = sma_cn((ts_max(high, 6, sid) - close) / (ts_max(high, 6, sid) - ts_min(low, 6, sid) + EPS) * 100, 9, 1, sid)

    cond49 = (high + low) > (data['high_l1'] + data['low_l1'])
    tr_part = np.maximum((high - data['high_l1']).abs(), (low - data['low_l1']).abs())
    p1 = pd.Series(np.where(cond49, 0, tr_part), index=data.index)
    p2 = pd.Series(np.where(cond49, tr_part, 0), index=data.index)
    alpha_dict['alpha049'] = ts_sum(p1, 12, sid) / (ts_sum(p1, 12, sid) + ts_sum(p2, 12, sid) + EPS)
    alpha_dict['alpha050'] = (ts_sum(p1, 12, sid) - ts_sum(p2, 12, sid)) / (ts_sum(p1, 12, sid) + ts_sum(p2, 12, sid) + EPS)
    alpha_dict['alpha051'] = ts_sum(p2, 12, sid) / (ts_sum(p1, 12, sid) + ts_sum(p2, 12, sid) + EPS)
    alpha_dict['alpha052'] = ts_sum(np.maximum(high - data['close_l1'], 0), 26, sid) / (ts_sum(np.maximum(data['close_l1'] - low, 0), 26, sid) + EPS) * 100
    alpha_dict['alpha053'] = close.gt(data['close_l1']).astype(float).groupby(sid).rolling(12, min_periods=12).mean().reset_index(level=0, drop=True) * 100
    alpha_dict['alpha054'] = -rank_cross(data.assign(tmp=ts_std((close - open_).abs(), 10, sid) + (close - open_) + rolling_corr(close, open_, 10, sid)), 'tmp')
    alpha_dict['alpha057'] = sma_cn((close - ts_min(low, 9, sid)) / (ts_max(high, 9, sid) - ts_min(low, 9, sid) + EPS) * 100, 3, 1, sid)
    alpha_dict['alpha058'] = close.gt(data['close_l1']).astype(float).groupby(sid).rolling(20, min_periods=20).mean().reset_index(level=0, drop=True) * 100

    part59 = pd.Series(np.nan, index=data.index)
    cond59a = close == data['close_l1']
    cond59b = close > data['close_l1']
    part59[cond59a] = 0
    part59[cond59b] = close[cond59b] - np.minimum(low[cond59b], data['close_l1'][cond59b])
    part59[~(cond59a | cond59b)] = close[~(cond59a | cond59b)] - np.maximum(low[~(cond59a | cond59b)], data['close_l1'][~(cond59a | cond59b)])
    alpha_dict['alpha059'] = ts_sum(part59, 20, sid)
    alpha_dict['alpha063'] = sma_cn(np.maximum(close - data['close_l1'], 0), 6, 1, sid) / (sma_cn((close - data['close_l1']).abs(), 6, 1, sid) + EPS) * 100
    alpha_dict['alpha065'] = ts_mean(close, 6, sid) / (close + EPS)
    alpha_dict['alpha066'] = (close - ts_mean(close, 6, sid)) / (ts_mean(close, 6, sid) + EPS) * 100
    alpha_dict['alpha067'] = sma_cn(np.maximum(close - data['close_l1'], 0), 24, 1, sid) / (sma_cn((close - data['close_l1']).abs(), 24, 1, sid) + EPS) * 100
    alpha_dict['alpha068'] = sma_cn(((high + low) / 2 - (data['high_l1'] + data['low_l1']) / 2) * (high - low) / (vol + EPS), 15, 2, sid)

    dtm = pd.Series(np.where(open_ <= data['open_l1'], 0, np.maximum(high - open_, open_ - data['open_l1'])), index=data.index)
    dbm = pd.Series(np.where(open_ >= data['open_l1'], 0, np.maximum(open_ - low, open_ - data['open_l1'])), index=data.index)
    sum_dtm20 = ts_sum(dtm, 20, sid)
    sum_dbm20 = ts_sum(dbm, 20, sid)
    alpha_dict['alpha069'] = np.select(
        [sum_dtm20 > sum_dbm20, sum_dtm20 == sum_dbm20, sum_dtm20 < sum_dbm20],
        [(sum_dtm20 - sum_dbm20) / (sum_dtm20 + EPS), 0, (sum_dtm20 - sum_dbm20) / (sum_dbm20 + EPS)],
        default=np.nan,
    )

    alpha_dict['alpha070'] = ts_std(data['amount'], 6, sid)
    alpha_dict['alpha071'] = (close - ts_mean(close, 24, sid)) / (ts_mean(close, 24, sid) + EPS) * 100
    alpha_dict['alpha072'] = sma_cn((ts_max(high, 6, sid) - close) / (ts_max(high, 6, sid) - ts_min(low, 6, sid) + EPS) * 100, 15, 1, sid)
    alpha_dict['alpha076'] = ts_std((close / (data['close_l1'] + EPS) - 1).abs() / (vol + EPS), 20, sid) / (ts_mean((close / (data['close_l1'] + EPS) - 1).abs() / (vol + EPS), 20, sid) + EPS)
    alpha_dict['alpha078'] = ((high + low + close) / 3 - ts_mean((high + low + close) / 3, 12, sid)) / (0.015 * ts_mean((close - ts_mean((high + low + close) / 3, 12, sid)).abs(), 12, sid) + EPS)
    alpha_dict['alpha079'] = sma_cn(np.maximum(close - data['close_l1'], 0), 12, 1, sid) / (sma_cn((close - data['close_l1']).abs(), 12, 1, sid) + EPS) * 100
    alpha_dict['alpha080'] = (vol - vol.groupby(sid).shift(5)) / (vol.groupby(sid).shift(5) + EPS) * 100
    alpha_dict['alpha081'] = sma_cn(vol, 21, 2, sid)
    alpha_dict['alpha082'] = sma_cn((ts_max(high, 6, sid) - close) / (ts_max(high, 6, sid) - ts_min(low, 6, sid) + EPS) * 100, 20, 1, sid)
    alpha_dict['alpha084'] = ts_sum(pd.Series(np.select([close > data['close_l1'], close < data['close_l1']], [vol, 0], default=-vol), index=data.index), 20, sid)

    a86 = ((close.groupby(sid).shift(20) - close.groupby(sid).shift(10)) / 10) - ((close.groupby(sid).shift(10) - close) / 10)
    alpha_dict['alpha086'] = np.select([a86 > 0.25, a86 < 0], [-1, 1], default=-(close - data['close_l1']))
    alpha_dict['alpha088'] = (close - close.groupby(sid).shift(20)) / (close.groupby(sid).shift(20) + EPS) * 100
    alpha_dict['alpha089'] = 2 * (sma_cn(close, 13, 2, sid) - sma_cn(close, 27, 2, sid) - sma_cn(sma_cn(close, 13, 2, sid) - sma_cn(close, 27, 2, sid), 10, 2, sid))

    alpha_dict['alpha093'] = ts_sum(pd.Series(np.where(open_ >= data['open_l1'], 0, np.maximum(open_ - low, open_ - data['open_l1'])), index=data.index), 20, sid)
    alpha_dict['alpha095'] = ts_std(data['amount'], 20, sid)
    alpha_dict['alpha096'] = sma_cn(sma_cn((close - ts_min(low, 9, sid)) / (ts_max(high, 9, sid) - ts_min(low, 9, sid) + EPS) * 100, 3, 1, sid), 3, 1, sid)
    alpha_dict['alpha097'] = ts_std(vol, 10, sid)

    cond98 = delta(ts_mean(close, 100, sid), 100, sid) / (close.groupby(sid).shift(100) + EPS) <= 0.05
    alpha_dict['alpha098'] = np.where(cond98, -(close - ts_min(close, 100, sid)), -delta(close, 3, sid))

    alpha_dict['alpha099'] = -rank_cross(data.assign(tmp=rolling_cov(rank_cross(data.assign(tmp=close), 'tmp'), rank_cross(data.assign(tmp=vol), 'tmp'), 5, sid)), 'tmp')
    alpha_dict['alpha100'] = ts_std(vol, 20, sid)
    alpha_dict['alpha102'] = sma_cn(np.maximum(vol - vol.groupby(sid).shift(1), 0), 6, 1, sid) / (sma_cn((vol - vol.groupby(sid).shift(1)).abs(), 6, 1, sid) + EPS) * 100

    lowday20 = (
        low.groupby(sid).rolling(20, min_periods=20)
        .apply(lambda x: len(x) - np.argmin(x), raw=True)
        .reset_index(level=0, drop=True)
    )
    alpha_dict['alpha103'] = ((20 - lowday20) / 20) * 100
    alpha_dict['alpha104'] = -(delta(rolling_corr(high, vol, 5, sid), 5, sid) * rank_cross(data.assign(tmp=ts_std(close, 20, sid)), 'tmp'))
    alpha_dict['alpha106'] = close - close.groupby(sid).shift(20)
    alpha_dict['alpha109'] = sma_cn(high - low, 10, 2, sid) / (sma_cn(sma_cn(high - low, 10, 2, sid), 10, 2, sid) + EPS)
    alpha_dict['alpha110'] = ts_sum(np.maximum(high - data['close_l1'], 0), 20, sid) / (ts_sum(np.maximum(data['close_l1'] - low, 0), 20, sid) + EPS) * 100
    alpha_dict['alpha111'] = sma_cn(vol * ((close - low) - (high - close)) / (hl_range + EPS), 11, 2, sid) - sma_cn(vol * ((close - low) - (high - close)) / (hl_range + EPS), 4, 2, sid)

    pos = np.maximum(close - data['close_l1'], 0)
    neg = np.maximum(data['close_l1'] - close, 0)
    alpha_dict['alpha112'] = (ts_sum(pos, 12, sid) - ts_sum(neg, 12, sid)) / (ts_sum(pos, 12, sid) + ts_sum(neg, 12, sid) + EPS) * 100

    beta20_x = np.arange(1, 21, dtype=float)
    alpha_dict['alpha116'] = (
        close.groupby(sid)
        .rolling(20, min_periods=20)
        .apply(lambda y: np.polyfit(beta20_x, y, deg=1)[0], raw=True)
        .reset_index(level=0, drop=True)
    )

    alpha_dict['alpha118'] = ts_sum(high - open_, 20, sid) / (ts_sum(open_ - low, 20, sid) + EPS) * 100
    alpha_dict['alpha127'] = np.sqrt(ts_mean((100 * (close - ts_max(close, 12, sid)) / (ts_max(close, 12, sid) + EPS)) ** 2, 12, sid))

    typ = (high + low + close) / 3
    up_typ_amt = (typ * vol).where(typ > typ.groupby(sid).shift(1), 0)
    down_typ_amt = (typ * vol).where(typ < typ.groupby(sid).shift(1), 0)
    alpha_dict['alpha128'] = 100 - 100 / (1 + ts_sum(up_typ_amt, 14, sid) / (ts_sum(down_typ_amt, 14, sid) + EPS))
    alpha_dict['alpha129'] = ts_sum(np.maximum(data['close_l1'] - close, 0), 12, sid)
    alpha_dict['alpha132'] = ts_mean(data['amount'], 20, sid)

    highday20 = (
        high.groupby(sid).rolling(20, min_periods=20)
        .apply(lambda x: len(x) - np.argmax(x), raw=True)
        .reset_index(level=0, drop=True)
    )
    alpha_dict['alpha133'] = ((20 - highday20) / 20) * 100 - ((20 - lowday20) / 20) * 100
    alpha_dict['alpha134'] = (close - close.groupby(sid).shift(12)) / (close.groupby(sid).shift(12) + EPS) * vol
    alpha_dict['alpha135'] = sma_cn((close.groupby(sid).shift(1) / (close.groupby(sid).shift(20) + EPS)), 20, 1, sid)
    alpha_dict['alpha136'] = -rank_cross(data.assign(tmp=delta(ret, 3, sid)), 'tmp') * rolling_corr(open_, vol, 10, sid)

    alpha_dict['alpha144'] = ts_sum((close / (data['close_l1'] + EPS) - 1).abs() / (data['amount'] + EPS), 20, sid)
    alpha_dict['alpha153'] = (ts_mean(close, 3, sid) + ts_mean(close, 6, sid) + ts_mean(close, 12, sid) + ts_mean(close, 24, sid)) / 4
    alpha_dict['alpha158'] = (high - sma_cn(close, 15, 2, sid)) - (low - sma_cn(close, 15, 2, sid))
    alpha_dict['alpha159'] = (
        (close - ts_sum(np.minimum(low, data['close_l1']), 6, sid)) / (ts_sum(np.maximum(high, data['close_l1']), 6, sid) - ts_sum(np.minimum(low, data['close_l1']), 6, sid) + EPS) * 12 * 24
        + (close - ts_sum(np.minimum(low, data['close_l1']), 12, sid)) / (ts_sum(np.maximum(high, data['close_l1']), 12, sid) - ts_sum(np.minimum(low, data['close_l1']), 12, sid) + EPS) * 6 * 24
        + (close - ts_sum(np.minimum(low, data['close_l1']), 24, sid)) / (ts_sum(np.maximum(high, data['close_l1']), 24, sid) - ts_sum(np.minimum(low, data['close_l1']), 24, sid) + EPS) * 6 * 24
    ) / (6 * 12 + 6 * 24 + 12 * 24)
    alpha_dict['alpha161'] = ts_mean(np.maximum(high - low, np.maximum((data['close_l1'] - high).abs(), (data['close_l1'] - low).abs())), 12, sid)
    alpha_dict['alpha162'] = (sma_cn(np.maximum(close - data['close_l1'], 0), 12, 1, sid) / (sma_cn((close - data['close_l1']).abs(), 12, 1, sid) + EPS) * 100 - ts_min(sma_cn(np.maximum(close - data['close_l1'], 0), 12, 1, sid) / (sma_cn((close - data['close_l1']).abs(), 12, 1, sid) + EPS) * 100, 12, sid)) / (ts_max(sma_cn(np.maximum(close - data['close_l1'], 0), 12, 1, sid) / (sma_cn((close - data['close_l1']).abs(), 12, 1, sid) + EPS) * 100, 12, sid) - ts_min(sma_cn(np.maximum(close - data['close_l1'], 0), 12, 1, sid) / (sma_cn((close - data['close_l1']).abs(), 12, 1, sid) + EPS) * 100, 12, sid) + EPS)
    alpha_dict['alpha164'] = sma_cn(1 / (close - data['close_l1'] + EPS), 13, 2, sid) * np.where(close > data['close_l1'], 1, np.nan) - sma_cn(1 / (close - data['close_l1'] + EPS), 13, 2, sid) * np.where(close <= data['close_l1'], 1, np.nan)
    alpha_dict['alpha167'] = ts_sum(np.maximum(close - data['close_l1'], 0), 12, sid)
    alpha_dict['alpha168'] = -vol / (ts_mean(vol, 20, sid) + EPS)

    tr = np.maximum(high - low, np.maximum((high - data['close_l1']).abs(), (low - data['close_l1']).abs()))
    hd = high - data['high_l1']
    ld = data['low_l1'] - low
    p1_172 = pd.Series(np.where((ld > 0) & (ld > hd), ld, 0), index=data.index)
    p2_172 = pd.Series(np.where((hd > 0) & (hd > ld), hd, 0), index=data.index)
    alpha_dict['alpha172'] = ts_mean((ts_sum(p1_172, 14, sid) * 100 / (ts_sum(pd.Series(tr, index=data.index), 14, sid) + EPS) - ts_sum(p2_172, 14, sid) * 100 / (ts_sum(pd.Series(tr, index=data.index), 14, sid) + EPS)).abs() / (ts_sum(p1_172, 14, sid) * 100 / (ts_sum(pd.Series(tr, index=data.index), 14, sid) + EPS) + ts_sum(p2_172, 14, sid) * 100 / (ts_sum(pd.Series(tr, index=data.index), 14, sid) + EPS) + EPS) * 100, 6, sid)

    alpha_dict['alpha187'] = ts_sum(pd.Series(np.where(open_ <= data['open_l1'], 0, np.maximum(high - open_, open_ - data['open_l1'])), index=data.index), 20, sid)
    alpha_dict['alpha188'] = (high - low - sma_cn(high - low, 11, 2, sid)) / (sma_cn(high - low, 11, 2, sid) + EPS) * 100
    alpha_dict['alpha189'] = ts_mean((close - ts_mean(close, 6, sid)).abs(), 6, sid)

    alpha_df = pd.DataFrame(alpha_dict, index=data.index)
    data = pd.concat([data, alpha_df], axis=1)

    # 清理极值
    for col in FEATURE_COLUMNS:
        if col in data.columns:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    # FEATURE_COLUMNS_IND = [f'ind_{x}' for x in data['IndustryName'].astype(str).unique()]
    keep_cols = [
        'StockID', 'TradingDay', 'PrevClosePrice', 'OpenPrice', 'HighPrice', 'LowPrice',
        'ClosePrice', 'Volume', 'Amount','IndustryName', 'target_5d'
    ] + [c for c in FEATURE_COLUMNS if c in data.columns]+[c for c in FEATURE_COLUMNS_ALPHA191 if c in data.columns]+[c for c in FEATURE_COLUMNS_IND if c in data.columns]

    return data[keep_cols]


def normalize_features(df,train_end,feature_type='train',label_type='raw'):
    data = df.copy()
    data['TradingDay'] = pd.to_datetime(data['TradingDay'])
    if feature_type=='train':
        train_mask = data['TradingDay'] <= pd.Timestamp(train_end)
        train_df = data.loc[train_mask, FEATURE_COLUMNS + FEATURE_COLUMNS_ALPHA191]
        lower = train_df.quantile(0.01)
        upper = train_df.quantile(0.99)
        clipped_train = train_df.clip(lower, upper, axis=1)
        mean = clipped_train.mean()
        std = clipped_train.std().replace(0, 1.0)
        data[FEATURE_COLUMNS + FEATURE_COLUMNS_ALPHA191] = data[FEATURE_COLUMNS + FEATURE_COLUMNS_ALPHA191].clip(lower, upper, axis=1)
        data[FEATURE_COLUMNS + FEATURE_COLUMNS_ALPHA191] = (data[FEATURE_COLUMNS + FEATURE_COLUMNS_ALPHA191] - mean) / std
    elif feature_type=='cross':
        fac_group = data.groupby(['TradingDay', 'IndustryName'], dropna=False)
        fac_mean = fac_group[FEATURE_COLUMNS+ FEATURE_COLUMNS_ALPHA191].transform('mean')
        fac_std = fac_group[FEATURE_COLUMNS+ FEATURE_COLUMNS_ALPHA191].transform('std')
        fac_std = fac_std.replace(0, 1.0).fillna(1.0)
        data[FEATURE_COLUMNS+ FEATURE_COLUMNS_ALPHA191] = (data[FEATURE_COLUMNS+ FEATURE_COLUMNS_ALPHA191] - fac_mean) / fac_std

    if label_type=='raw':
        pass
    elif label_type=='normalize':
        y_group = data.groupby('TradingDay', dropna=False)['target_5d']
        y_mean = y_group.transform('mean')
        y_std = y_group.transform('std')
        y_std = y_std.replace(0, 1.0).fillna(1.0)
        data['target_5d'] = (data['target_5d'] - y_mean) / y_std
    elif label_type=='ind':
        y_group = data.groupby(['TradingDay', 'IndustryName'],dropna=False)['target_5d']
        y_mean = y_group.transform('mean')
        y_std = y_group.transform('std')
        y_std = y_std.replace(0, 1.0).fillna(1.0)
        data['target_5d'] = (data['target_5d'] - y_mean) / y_std

    return data