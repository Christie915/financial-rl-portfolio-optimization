"""
Walk-Forward Validation for PPO Portfolio Optimization
=======================================================

Purpose:
  Adds a second out-of-sample fold (OOS-2024) alongside the existing OOS-2025,
  addressing the "single evaluation period" limitation noted in Section 5.3.

Walk-Forward Folds:
  Fold 1 (existing): Train 2022-2023, Val 2023-Q4, Test 2024
  Fold 2 (original): Train 2022-2024, Val 2024-Q4, Test 2025

Usage:
  # Run a single config + seed on the new 2024-OOS fold
  python walk_forward.py --config HKUST_NoAttn --seed 42

  # Run all main configs × all seeds (full walk-forward grid)
  python walk_forward.py --all

  # Generate the combined walk-forward summary table
  python walk_forward.py --summarize

Output:
  results/wf_fold1_backtest_{config}_s{seed}.csv     (per-seed metrics)
  results/wf_fold1_logs_{config}_s{seed}.csv          (daily trade logs)
  results/wf_fold1_curve_{config}_s{seed}.png         (equity curve)
  results/wf_summary.csv                              (cross-fold comparison)
"""
import os
import sys
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

# Import from existing codebase
from multi_train_ppo import CrossAssetAttentionExtractor, ValidationCheckpointCallback
from config import DOW_30_TICKERS, build_config, CONFIGS_MAIN, SEEDS
from multi_asset_data import MultiAssetDataEngine
from portfolio_env import PortfolioTradingEnv

import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# Walk-Forward Fold Configuration
# ==============================================================================
FOLD_1 = {
    'name': 'Fold1_OOS2024',
    'data_start': '2022-01-01',
    'data_end': '2025-12-31',      # need full data for date alignment
    'train_end_year': 2023,         # train on 2022-2023
    'test_year': 2024,              # OOS on 2024
    'val_days': 75,                 # last 75 days of training slice
    'description': 'Train 2022-2023, Val late-2023, Test 2024',
}

# Fold 2 = the original experiment (already done); we just load existing results
FOLD_2 = {
    'name': 'Fold2_OOS2025',
    'train_end_year': 2024,
    'test_year': 2025,
    'description': 'Train 2022-2024, Val late-2024, Test 2025 (original)',
}


def set_all_seeds(seed):
    """Lock all RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data(config_name, data_start, data_end, train_end_date):
    """Build 3D tensor using the existing MultiAssetDataEngine."""
    sentiment_source, use_sentiment, use_attention, use_confidence = build_config(config_name)

    engine = MultiAssetDataEngine(
        tickers=DOW_30_TICKERS,
        start_date=data_start,
        end_date=data_end,
    )
    engine.download_and_clean_price()

    if use_sentiment:
        sent_path = ("stock_news/finbert_hkust_sentiment.csv"
                     if sentiment_source == "HKUST"
                     else "dow30_news_with_sentiment.csv")
        # Handle both old and new API signatures
        try:
            engine.load_sentiment_data(sent_path,
                                       confidence_threshold=0.3 if use_confidence else 0.0)
        except TypeError:
            engine.load_sentiment_data(sent_path)

    # Handle both old and new API signatures
    try:
        panel_data, trade_dates = engine.build_3d_tensor(
            use_sentiment=use_sentiment,
            use_confidence=use_confidence,
            train_end_date=train_end_date,
        )
    except TypeError:
        panel_data, trade_dates = engine.build_3d_tensor(
            use_sentiment=use_sentiment,
        )

    return panel_data, trade_dates, use_attention


def split_data(panel_data, trade_dates, train_end_year, test_year, val_days=75):
    """Split panel data into train / val / test by year boundary."""
    # Find test start index
    test_start = None
    for i, d in enumerate(trade_dates):
        if d.year == test_year:
            test_start = i
            break

    if test_start is None:
        raise ValueError(f"No data found for test year {test_year}")

    # Find train end (everything before test_year that is <= train_end_year)
    train_end = test_start  # everything before test start

    full_train = panel_data[:train_end]
    val_start = max(0, len(full_train) - val_days)
    train_data = full_train[:val_start]
    val_data = full_train[val_start:]
    test_data = panel_data[test_start:]

    # Also get test dates for reporting
    test_dates = trade_dates[test_start:]

    return train_data, val_data, test_data, test_dates


def train_fold(config_name, seed, train_data, val_data, use_attention,
               suffix, timesteps=400_000, eval_freq=50_000):
    """Train PPO on the given fold and return path to best model."""
    _, _, _, use_confidence = build_config(config_name)

    def make_train_env():
        return PortfolioTradingEnv(train_data, DOW_30_TICKERS, transaction_cost=0.0001)

    env = DummyVecEnv([make_train_env])
    base_env = env.envs[0]

    policy_kwargs = dict(
        features_extractor_class=CrossAssetAttentionExtractor,
        features_extractor_kwargs=dict(
            num_tokens=base_env.num_tokens if hasattr(base_env, 'num_tokens') else base_env.num_assets + 1,
            per_token_dim=base_env.per_asset_feat_dim if hasattr(base_env, 'per_asset_feat_dim')
                          else base_env.window_size * base_env.num_features + 1,
            embed_dim=128,
            num_heads=4,
            features_dim=256,
            use_attention=use_attention,
            dropout=0.1,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    os.makedirs('models', exist_ok=True)
    best_path = f"models/portfolio_ppo_wf{suffix}_best.zip"

    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5, max_grad_norm=0.5,
        verbose=0,
        seed=seed,
        device='auto',
    )

    val_cb = ValidationCheckpointCallback(
        val_data=val_data,
        val_tickers=DOW_30_TICKERS,
        eval_freq=eval_freq,
        model_save_path=best_path,
        verbose=1,
    )

    print(f"  Training {timesteps:,} steps (train={len(train_data)}d, val={len(val_data)}d)...")
    model.learn(total_timesteps=timesteps, callback=val_cb, progress_bar=False)

    # Fallback: if best was never saved
    if not os.path.exists(best_path):
        model.save(best_path)

    print(f"  Best checkpoint: step {val_cb.best_step}, ValSharpe={val_cb.best_val_sharpe:.3f}")
    return best_path


def backtest_fold(model_path, test_data, config_name, seed, suffix, cost=0.0001):
    """Run deterministic OOS backtest and return metrics dict + daily returns."""
    _, _, use_attention, _ = build_config(config_name)

    def make_env():
        return PortfolioTradingEnv(test_data, DOW_30_TICKERS, transaction_cost=cost)

    env = DummyVecEnv([make_env])

    # Register custom extractor
    sys.modules['__main__'].CrossAssetAttentionExtractor = CrossAssetAttentionExtractor
    model = PPO.load(model_path, env=env, device='cpu')

    base_env = env.envs[0]
    obs = env.reset()
    done = np.array([False])

    daily_returns = []
    net_worths = [base_env.initial_balance]
    trade_logs = []

    # Benchmark
    start_idx = base_env.window_size
    end_idx = len(test_data) - 1
    asset_returns_matrix = test_data[start_idx:end_idx, :, 0]
    num_assets = asset_returns_matrix.shape[1]
    equal_w = np.ones(num_assets) / num_assets
    bh_daily = np.dot(asset_returns_matrix, equal_w)
    bh_cum = np.concatenate([[1.0], np.cumprod(1 + bh_daily)])
    bh_net = bh_cum * base_env.initial_balance

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info_arr = env.step(action)
        info = info_arr[0]

        dr = info.get('portfolio_return', 0.0)
        daily_returns.append(dr)
        net_worths.append(info['net_worth'])

        w = base_env.current_weights
        log = {
            'Day_Step': len(trade_logs) + 1,
            'Net_Worth': round(info['net_worth'], 2),
            'Daily_Return(%)': round(dr * 100, 4),
            'Cash_Weight(%)': round(w[0] * 100, 2),
        }
        for i, tk in enumerate(DOW_30_TICKERS):
            log[f'{tk}_Weight(%)'] = round(w[i + 1] * 100, 2)
        trade_logs.append(log)

    # Metrics
    final_nw = net_worths[-1]
    bh_final = bh_net[-1] if len(bh_net) > 0 else base_env.initial_balance
    ret = (final_nw / base_env.initial_balance - 1) * 100
    bh_ret = (bh_final / base_env.initial_balance - 1) * 100
    win_rate = sum(1 for r in daily_returns if r > 0) / max(len(daily_returns), 1) * 100

    nw_arr = np.array(net_worths)
    cummax = np.maximum.accumulate(nw_arr)
    mdd = np.max((cummax - nw_arr) / cummax) * 100

    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns)
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
    alpha = ret - bh_ret
    calmar = (ret / mdd) if mdd > 0 else 0.0

    metrics = {
        'Return Rate (%)': round(ret, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Max Drawdown (%)': round(mdd, 2),
        'Calmar Ratio': round(calmar, 3),
        'Win Rate (%)': round(win_rate, 2),
        'Benchmark Return (%)': round(bh_ret, 2),
        'Alpha (%)': round(alpha, 2),
    }

    # Save outputs
    os.makedirs('results', exist_ok=True)

    pd.DataFrame([{'Metric': k, 'Value': v} for k, v in metrics.items()]).to_csv(
        f'results/wf_fold1_backtest_{config_name}_s{seed}.csv', index=False)
    pd.DataFrame(trade_logs).to_csv(
        f'results/wf_fold1_logs_{config_name}_s{seed}.csv', index=False)

    # Plot equity curve
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2 = axes

    ax1.plot(net_worths, color='darkorange', linewidth=2,
             label=f'PPO {config_name} (seed {seed})')
    ax1.plot(range(len(bh_net)), bh_net, color='steelblue', linestyle='--',
             alpha=0.7, label='Dow30 Equal Weight')
    ax1.set_title(f'Walk-Forward Fold 1 (OOS 2024) | {config_name} s={seed}\n'
                  f'Ret={ret:.2f}% Sharpe={sharpe:.3f} MDD={mdd:.2f}% Alpha={alpha:+.2f}%',
                  fontsize=11)
    ax1.set_ylabel('Net Worth ($)')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # Drawdown panel
    dd_series = (cummax - nw_arr) / cummax * 100
    ax2.fill_between(range(len(dd_series)), -dd_series, 0, color='crimson', alpha=0.3)
    ax2.set_xlabel('Trading Days (2024)')
    ax2.set_ylabel('DD (%)')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/wf_fold1_curve_{config_name}_s{seed}.png', dpi=130)
    plt.close()

    print(f"  → Ret={ret:.2f}% | Sharpe={sharpe:.3f} | MDD={mdd:.2f}% | Alpha={alpha:+.2f}%")
    return metrics, daily_returns


def run_one(config_name, seed, timesteps=400_000):
    """Full pipeline: data → train → backtest for Fold 1."""
    print(f"\n{'='*70}")
    print(f"Walk-Forward Fold 1 | {config_name} | seed={seed}")
    print(f"Train: 2022-2023 | Val: late-2023 | Test: 2024")
    print(f"{'='*70}")

    set_all_seeds(seed)
    fold = FOLD_1
    suffix = f"_{config_name}_fold1_s{seed}"

    # Build data with train_end_date = "2023-12-31"
    panel_data, trade_dates, use_attention = build_data(
        config_name,
        data_start=fold['data_start'],
        data_end=fold['data_end'],
        train_end_date="2023-12-31",
    )

    train_data, val_data, test_data, test_dates = split_data(
        panel_data, trade_dates,
        train_end_year=fold['train_end_year'],
        test_year=fold['test_year'],
        val_days=fold['val_days'],
    )

    print(f"  Data splits: train={len(train_data)}d, val={len(val_data)}d, test={len(test_data)}d")

    # Train
    model_path = train_fold(
        config_name, seed, train_data, val_data,
        use_attention, suffix, timesteps=timesteps,
    )

    # Backtest
    metrics, daily_rets = backtest_fold(
        model_path, test_data, config_name, seed, suffix,
    )

    return metrics


def run_all(timesteps=400_000):
    """Run Fold 1 for all main configs × all seeds."""
    results = []
    for cfg in CONFIGS_MAIN:
        for seed in SEEDS:
            try:
                m = run_one(cfg, seed, timesteps=timesteps)
                m['Config'] = cfg
                m['Seed'] = seed
                m['Fold'] = 'Fold1_OOS2024'
                results.append(m)
            except Exception as e:
                print(f"  [ERROR] {cfg} seed={seed}: {e}")
                continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv('results/wf_fold1_all_results.csv', index=False)
        print(f"\n→ results/wf_fold1_all_results.csv ({len(results)} runs)")
    return results


def summarize():
    """
    Combine Fold 1 (OOS-2024) and Fold 2 (OOS-2025, from existing results)
    into a single walk-forward summary table.
    """
    rows = []

    # --- Fold 1 (OOS 2024) ---
    for cfg in CONFIGS_MAIN:
        fold1_metrics = []
        for seed in SEEDS:
            path = f'results/wf_fold1_backtest_{cfg}_s{seed}.csv'
            if os.path.exists(path):
                df = pd.read_csv(path)
                d = dict(zip(df['Metric'], df['Value']))
                fold1_metrics.append(d)

        if fold1_metrics:
            df1 = pd.DataFrame(fold1_metrics)
            rows.append({
                'Config': cfg,
                'Fold': 'OOS-2024',
                'N_Seeds': len(fold1_metrics),
                'Return_median': round(df1['Return Rate (%)'].median(), 2),
                'Return_std': round(df1['Return Rate (%)'].std(), 2),
                'Sharpe_median': round(df1['Sharpe Ratio'].median(), 3),
                'Sharpe_std': round(df1['Sharpe Ratio'].std(), 3),
                'MDD_median': round(df1['Max Drawdown (%)'].median(), 2),
                'Alpha_median': round(df1['Alpha (%)'].median(), 2),
            })

    # --- Fold 2 (OOS 2025) — from existing aggregated results ---
    for cfg in CONFIGS_MAIN:
        path = f'results/backtest_metrics_aggregated_{cfg}_v5_ppo.csv'
        if os.path.exists(path):
            df2 = pd.read_csv(path)
            d2 = dict(zip(df2['Metric'], df2['Median']))
            d2_std = dict(zip(df2['Metric'], df2['Std']))
            rows.append({
                'Config': cfg,
                'Fold': 'OOS-2025',
                'N_Seeds': int(df2['N_Seeds'].iloc[0]) if 'N_Seeds' in df2.columns else 3,
                'Return_median': round(d2.get('Return Rate (%)', 0), 2),
                'Return_std': round(d2_std.get('Return Rate (%)', 0), 2),
                'Sharpe_median': round(d2.get('Sharpe Ratio', 0), 3),
                'Sharpe_std': round(d2_std.get('Sharpe Ratio', 0), 3),
                'MDD_median': round(d2.get('Max Drawdown (%)', 0), 2),
                'Alpha_median': round(d2.get('Alpha (%)', 0), 2),
            })
        else:
            # Try per-seed files
            fold2_metrics = []
            for seed in SEEDS:
                p = f'results/backtest_metrics_summary_{cfg}_v5_ppo_s{seed}.csv'
                if os.path.exists(p):
                    df_s = pd.read_csv(p)
                    fold2_metrics.append(dict(zip(df_s['Metric'], df_s['Value'])))
            if fold2_metrics:
                df2 = pd.DataFrame(fold2_metrics)
                rows.append({
                    'Config': cfg,
                    'Fold': 'OOS-2025',
                    'N_Seeds': len(fold2_metrics),
                    'Return_median': round(df2['Return Rate (%)'].median(), 2),
                    'Return_std': round(df2['Return Rate (%)'].std(), 2),
                    'Sharpe_median': round(df2['Sharpe Ratio'].median(), 3),
                    'Sharpe_std': round(df2['Sharpe Ratio'].std(), 3),
                    'MDD_median': round(df2['Max Drawdown (%)'].median(), 2),
                    'Alpha_median': round(df2['Alpha (%)'].median(), 2),
                })

    if not rows:
        print("No results found. Run --all first.")
        return

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(['Config', 'Fold']).reset_index(drop=True)
    summary.to_csv('results/wf_summary.csv', index=False)

    print("\n" + "=" * 90)
    print("Walk-Forward Summary: OOS-2024 vs OOS-2025")
    print("=" * 90)
    print(summary.to_string(index=False))
    print(f"\n→ results/wf_summary.csv")

    # Cross-fold consistency check
    print("\n--- Cross-Fold Consistency ---")
    for cfg in CONFIGS_MAIN:
        sub = summary[summary['Config'] == cfg]
        if len(sub) == 2:
            s1 = sub[sub['Fold'] == 'OOS-2024'].iloc[0]
            s2 = sub[sub['Fold'] == 'OOS-2025'].iloc[0]
            delta_sharpe = s2['Sharpe_median'] - s1['Sharpe_median']
            delta_alpha = s2['Alpha_median'] - s1['Alpha_median']
            consistent = "✓" if (s1['Alpha_median'] > 0 and s2['Alpha_median'] > 0) else "✗"
            print(f"  {cfg:20s} | ΔSharpe={delta_sharpe:+.3f} | ΔAlpha={delta_alpha:+.2f}% | "
                  f"Alpha>0 both folds: {consistent}")


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Validation')
    parser.add_argument('--config', help='Single config name')
    parser.add_argument('--seed', type=int, help='Single seed')
    parser.add_argument('--all', action='store_true',
                        help='Run Fold 1 for all configs × all seeds')
    parser.add_argument('--summarize', action='store_true',
                        help='Generate cross-fold summary from existing results')
    parser.add_argument('--timesteps', type=int, default=400_000)
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    if args.summarize:
        summarize()
    elif args.all:
        run_all(timesteps=args.timesteps)
        summarize()
    elif args.config and args.seed is not None:
        run_one(args.config, args.seed, timesteps=args.timesteps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
