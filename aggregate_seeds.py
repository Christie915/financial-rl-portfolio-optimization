"""
聚合 3-seed 结果

用法:
  python aggregate_seeds.py --config HKUST_Full
  python aggregate_seeds.py --all      (聚合全部 6 个 config)

产出:
  results/backtest_metrics_aggregated_{config}_v5_ppo.csv
  - 每个指标给 median / mean / std / min / max
  - 额外存 seed_42, seed_123, seed_2024 的原始值
"""
import os
import argparse
import pandas as pd
import numpy as np

CONFIGS = ['NONE_Vanilla', 'NONE_NoSent', 'EODHD_NoAttn', 'EODHD_Full',
           'HKUST_NoAttn', 'HKUST_Full']
SEEDS = [42, 123, 2024]


def aggregate_one(config):
    """聚合某个 config 的 3 seed 结果"""
    rows = []
    for seed in SEEDS:
        path = f"results/backtest_metrics_summary_{config}_v5_ppo_s{seed}.csv"
        if not os.path.exists(path):
            print(f"  [miss] {path}")
            continue
        df = pd.read_csv(path)
        d = dict(zip(df['Metric'], df['Value']))
        d['seed'] = seed
        rows.append(d)

    if not rows:
        print(f"[skip] {config}: 没有任何 seed 的结果")
        return None

    df_all = pd.DataFrame(rows)
    metrics_to_agg = [
        "Return Rate (%)", "Win Rate (%)", "Max Drawdown (%)",
        "Sharpe Ratio", "Calmar Ratio", "Avg Daily Turnover (%)",
        "Alpha (%)",
    ]

    summary = {}
    for m in metrics_to_agg:
        if m not in df_all.columns:
            continue
        vals = df_all[m].values
        summary[m] = {
            'median': np.median(vals),
            'mean':   np.mean(vals),
            'std':    np.std(vals),
            'min':    np.min(vals),
            'max':    np.max(vals),
            'seeds':  list(vals),
            'n':      len(vals),
        }

    # 打印简表
    print(f"\n===== {config} (n={len(rows)} seeds) =====")
    header = f"{'Metric':<28} {'median':>9} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}  seeds"
    print(header)
    print("-" * len(header))
    for m, s in summary.items():
        print(f"{m:<28} {s['median']:>9.3f} {s['mean']:>9.3f} {s['std']:>9.3f} "
              f"{s['min']:>9.3f} {s['max']:>9.3f}  {[f'{v:.2f}' for v in s['seeds']]}")

    # 存 CSV
    out_rows = []
    for m, s in summary.items():
        row = {'Metric': m,
               'Median': round(s['median'], 4),
               'Mean':   round(s['mean'], 4),
               'Std':    round(s['std'], 4),
               'Min':    round(s['min'], 4),
               'Max':    round(s['max'], 4),
               'N_Seeds': s['n']}
        for seed, v in zip(SEEDS[:len(s['seeds'])], s['seeds']):
            row[f'seed_{seed}'] = round(v, 4)
        out_rows.append(row)

    out_path = f"results/backtest_metrics_aggregated_{config}_v5_ppo.csv"
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"→ {out_path}")
    return summary


def aggregate_all():
    """生成所有 config 的汇总 + 一个大总表"""
    all_summaries = {}
    for c in CONFIGS:
        s = aggregate_one(c)
        if s:
            all_summaries[c] = s

    # 大总表: 每个 config 一行, 显示主要指标的 median ± std
    big_rows = []
    for c, s in all_summaries.items():
        row = {'Config': c, 'N_Seeds': s['Return Rate (%)']['n']}
        for m in ['Return Rate (%)', 'Sharpe Ratio', 'Max Drawdown (%)',
                  'Calmar Ratio', 'Alpha (%)']:
            if m in s:
                row[f'{m}_median'] = round(s[m]['median'], 3)
                row[f'{m}_std']    = round(s[m]['std'], 3)
        big_rows.append(row)

    if big_rows:
        big_df = pd.DataFrame(big_rows)
        big_path = "results/backtest_metrics_summary_ALL_v5_ppo.csv"
        big_df.to_csv(big_path, index=False)
        print(f"\n====== 全配置总表 ======")
        print(big_df.to_string(index=False))
        print(f"→ {big_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='单个 config')
    parser.add_argument('--all', action='store_true', help='聚合全部')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    if args.all:
        aggregate_all()
    elif args.config:
        aggregate_one(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
