# 量化项目

## 运行方式
- 进入目录：`..\zrb\text`
- 运行：`python main.py`

## 说明
- 按项目说明在 `text/src` 内完成数据生成、特征工程、模型训练、预测、回测与报告输出，为训练方便也储存了因子文件；`text/reports`里保存了策略报告和因子说明报告。 

## 结果目录
- `results/nolimit`不限换手结果，包括回测指标，净值序列和回测图
- `results/nolimit`单边换手25%结果，包括回测指标，净值序列和回测图
- `results/nolimit`单边换手10%结果，包括回测指标，净值序列和回测图，以上三个文件夹结果由 backtest.py 运行得到，由参数 dropk 控制换手。
- `results/feature_dataset.csv`
- `results/*.pth` 保存了每个窗口的最优模型
- `results/*_rolling_training_history.csv` 保存了模型每个 epoch 的训练 loss 与验证 loss。 
- `results/*_rolling_predictions.csv` 保存了模型的预测结果
- `results/*_rolling_backtest_timeseries.csv` 保存了净值序列
- `results/*_backtest_results.csv` 保存了回测指标
- `results/*_equity_curve.png` 保存了回测图

