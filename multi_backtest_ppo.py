"""
多资产 DRL 独立回测 (v5 PPO)

用法:
  python multi_backtest_ppo.py --config HKUST_Full --seed 42

总是加载 "_best.zip" (validation Sharpe 最高的 checkpoint).
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from multi_train_ppo import CrossAssetAttentionExtractor, DOW_30_TICKERS, build_config
from multi_asset_data import MultiAssetDataEngine
from portfolio_env import PortfolioTradingEnv

import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    sentiment_source, use_sentiment, use_attention, use_confidence = build_config(args.config)
    SUFFIX = f"_{args.config}_v5_ppo_s{args.seed}"

    print("\n" + "=" * 70)
    print(f"PPO 回测 | config={args.config} | seed={args.seed}")
    print(f"suffix={SUFFIX}")
    print("=" * 70)

    # ---- 数据 ----
    engine = MultiAssetDataEngine(
        tickers=DOW_30_TICKERS, start_date="2022-01-01", end_date="2025-12-31"
    )
    engine.download_and_clean_price()

    if use_sentiment:
        sent_path = ("stock_news/finbert_hkust_sentiment.csv" if sentiment_source == "HKUST"
                     else "dow30_news_with_sentiment.csv")
        engine.load_sentiment_data(sent_path, confidence_threshold=0.3 if use_confidence else 0.0)

    panel_data, trade_dates = engine.build_3d_tensor(
        use_sentiment=use_sentiment, use_confidence=use_confidence, train_end_date="2024-12-31"
    )

    split_idx = 0
    for i, date in enumerate(trade_dates):
        if date.year == 2025:
            split_idx = i
            break
    test_data = panel_data[split_idx:]
    print(f"测试集 (2025 全年): {len(test_data)} 天")

    def make_test_env():
        return PortfolioTradingEnv(test_data, DOW_30_TICKERS, transaction_cost=0.0001)
    env = DummyVecEnv([make_test_env])

    MODEL_PATH = f"models/portfolio_ppo{SUFFIX}_best.zip"
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到 {MODEL_PATH}")
        return

    print(f"加载最佳 checkpoint: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env, device='cpu')

    base_env = env.envs[0]
    obs = env.reset()
    done = np.array([False])

    daily_returns = []
    net_worths = [base_env.initial_balance]
    trade_logs = []
    daily_drawdowns = []
    daily_turnovers = []

    # 大盘基准
    start_idx = base_env.window_size
    end_idx = len(test_data) - 1
    asset_returns_matrix = test_data[start_idx:end_idx, :, 0]
    num_assets = asset_returns_matrix.shape[1]
    equal_weights = np.ones(num_assets) / num_assets
    bh_daily = np.dot(asset_returns_matrix, equal_weights)
    bh_cum = np.concatenate([[1.0], np.cumprod(1 + bh_daily)])
    bh_net_worths = bh_cum * base_env.initial_balance

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info_arr = env.step(action)
        info = info_arr[0]

        daily_ret = info.get('portfolio_return', 0.0)
        daily_returns.append(daily_ret)
        net_worths.append(info['net_worth'])
        daily_drawdowns.append(info['drawdown'])
        daily_turnovers.append(info['turnover'])

        w = base_env.current_weights
        log_entry = {
            'Day_Step': len(trade_logs) + 1,
            'Net_Worth': round(info['net_worth'], 2),
            'Daily_Return(%)': round(daily_ret * 100, 4),
            'Cost_Friction(%)': round(info.get('cost_rate', 0) * 100, 4),
            'Drawdown(%)': round(info['drawdown'] * 100, 3),
            'Turnover(%)': round(info['turnover'] * 100, 3),
            'Cash_Weight(%)': round(w[0] * 100, 2),
        }
        for i, tk in enumerate(DOW_30_TICKERS):
            log_entry[f'{tk}_Weight(%)'] = round(w[i + 1] * 100, 2)
        trade_logs.append(log_entry)

    final_nw = net_worths[-1]
    bh_final = bh_net_worths[-1]
    ret_pct = (final_nw / base_env.initial_balance - 1) * 100
    bh_ret_pct = (bh_final / base_env.initial_balance - 1) * 100
    win_rate = (sum(1 for r in daily_returns if r > 0) / len(daily_returns)) * 100

    nw_arr = np.array(net_worths)
    cummax = np.maximum.accumulate(nw_arr)
    mdd = np.max((cummax - nw_arr) / cummax) * 100

    mean_r = np.mean(daily_returns); std_r = np.std(daily_returns)
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
    excess = ret_pct - bh_ret_pct
    calmar = (ret_pct / mdd) if mdd > 0 else 0.0
    avg_turnover = np.mean(daily_turnovers) * 100
    avg_dd = np.mean(daily_drawdowns) * 100

    print("=" * 70)
    print(f"Return: {ret_pct:.2f}% | Sharpe: {sharpe:.3f} | MDD: {mdd:.2f}% | Alpha: {excess:+.2f}%")
    print(f"Calmar: {calmar:.3f} | Win: {win_rate:.2f}% | AvgTurn: {avg_turnover:.2f}%")
    print("=" * 70)

    os.makedirs('results', exist_ok=True)

    report_text = f"""{'='*70}
PPO 独立回测 | config={args.config} | seed={args.seed}
{'='*70}
Return Rate   : {ret_pct:.2f}%
Sharpe Ratio  : {sharpe:.3f}
Max Drawdown  : {mdd:.2f}%
Calmar Ratio  : {calmar:.3f}
Win Rate      : {win_rate:.2f}%
Avg Turnover  : {avg_turnover:.2f}%
Benchmark Ret : {bh_ret_pct:.2f}%
Alpha         : {excess:+.2f}%
{'='*70}
"""
    with open(f"results/backtest_academic_report{SUFFIX}.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    metrics = {
        "Metric": [
            "Initial Capital", "Final Value", "Return Rate (%)",
            "Win Rate (%)", "Max Drawdown (%)", "Sharpe Ratio",
            "Calmar Ratio", "Avg Daily Turnover (%)", "Avg Daily Drawdown (%)",
            "Benchmark Final", "Benchmark Return (%)", "Alpha (%)"
        ],
        "Value": [
            round(base_env.initial_balance, 2),
            round(final_nw, 2), round(ret_pct, 2),
            round(win_rate, 2), round(mdd, 2), round(sharpe, 3),
            round(calmar, 3), round(avg_turnover, 2), round(avg_dd, 2),
            round(bh_final, 2), round(bh_ret_pct, 2), round(excess, 2),
        ]
    }
    pd.DataFrame(metrics).to_csv(f"results/backtest_metrics_summary{SUFFIX}.csv", index=False)
    pd.DataFrame(trade_logs).to_csv(f'results/independent_backtest_logs{SUFFIX}.csv', index=False)

    # 图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2 = axes
    ax1.plot(net_worths, color='darkorange', linewidth=2.5, label=f'PPO {args.config} s={args.seed}')
    ax1.plot(range(len(bh_net_worths)), bh_net_worths, color='b', linestyle='--',
             alpha=0.7, label='Dow30 Equal Weight')
    ax1.fill_between(range(len(net_worths)), net_worths, bh_net_worths,
                     where=(np.array(net_worths) >= np.array(bh_net_worths)),
                     interpolate=True, color='orange', alpha=0.12)
    ax1.axhline(y=base_env.initial_balance, color='r', linestyle=':', alpha=0.5)
    ax1.set_title(f'OOS 2025 | {args.config} seed={args.seed} | '
                  f'Ret={ret_pct:.2f}% Shp={sharpe:.3f} MDD={mdd:.2f}%', fontsize=12)
    ax1.set_ylabel('Net Worth ($)')
    ax1.legend(loc='upper left'); ax1.grid(alpha=0.3)

    ax2.fill_between(range(1, len(net_worths)),
                     [-d*100 for d in daily_drawdowns], 0,
                     color='crimson', alpha=0.3, label='Drawdown (%)')
    ax2.set_xlabel('Trading Days'); ax2.set_ylabel('DD (%)')
    ax2.grid(alpha=0.3); ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(f'results/independent_backtest_curve{SUFFIX}.png', dpi=130)
    plt.close()

    print(f"CSV / 图表保存完毕 (suffix: {SUFFIX})")


if __name__ == "__main__":
    main()
