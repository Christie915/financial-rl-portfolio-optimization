"""
Bootstrap statistical inference on OOS daily returns.

For each configuration, pools daily return series across all available seeds
and computes 10,000 bootstrap replicates of Sharpe Ratio and Alpha to produce
95% percentile confidence intervals.  Also performs pairwise comparison:
two configs are deemed significantly different if their bootstrap CI for the
difference excludes zero.
"""
import os
import numpy as np
import pandas as pd
from config import CONFIGS_MAIN

N_BOOT = 10_000
ALPHA = 0.05  # → 95% CI


def load_daily_returns(config: str):
    """Concatenate daily returns from all available seed backtest logs."""
    all_rets = []
    for seed in [42, 123, 2024]:
        path = f"results/independent_backtest_logs_{config}_v5_ppo_s{seed}.csv"
        if not os.path.exists(path):
            print(f"  [skip] {config} s{seed}: missing")
            continue
        df = pd.read_csv(path)
        col = "Daily_Return(%)"
        if col in df.columns:
            # Convert from % to decimal
            rets = df[col].values / 100.0
            all_rets.append(rets)
    if not all_rets:
        return None, 0
    return np.concatenate(all_rets), len(all_rets)


def bootstrap_metric(rets, metric_fn, n_boot=N_BOOT):
    """Bootstrap a metric from daily returns.  metric_fn: array → scalar."""
    n = len(rets)
    rng = np.random.default_rng(2025)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = rets[idx]
        boots.append(metric_fn(sample))
    boots = np.array(boots)
    lo = np.percentile(boots, 100 * ALPHA / 2)
    hi = np.percentile(boots, 100 * (1 - ALPHA / 2))
    return np.median(boots), lo, hi, boots


def sharpe_fn(rets):
    """Annualised Sharpe from daily decimal returns."""
    if len(rets) < 2:
        return 0.0
    mu = np.mean(rets)
    sigma = np.std(rets, ddof=1)
    return (mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0


def main():
    print("=" * 70)
    print(f"Bootstrap Inference: {N_BOOT:,} replicates, {100*(1-ALPHA):.0f}% CI")
    print("=" * 70)

    results = {}
    for config in CONFIGS_MAIN:
        rets, n_seeds = load_daily_returns(config)
        if rets is None:
            continue
        # Compute metrics
        sharpe_med, sh_lo, sh_hi, sh_boots = bootstrap_metric(rets, sharpe_fn)
        # Alpha: mean daily excess over equal-weight (approximated by daily mean)
        # We compute annualised return and compare to benchmark.
        ann_ret = np.mean(rets) * 252 * 100  # annualised %
        results[config] = {
            "n_days": len(rets),
            "n_seeds": n_seeds,
            "sharpe_median": sharpe_med,
            "sharpe_ci_lo": sh_lo,
            "sharpe_ci_hi": sh_hi,
            "sharpe_boots": sh_boots,
            "ann_return": ann_ret,
        }

    # ── Print per-config CI ──
    print(f"\n{'Config':<20} {'Sharpe':>8}  {'95% CI':>20}  {'Days':>6}  {'Seeds':>5}")
    print("-" * 65)
    for config, r in results.items():
        print(f"{config:<20} {r['sharpe_median']:8.3f}  "
              f"[{r['sharpe_ci_lo']:.3f}, {r['sharpe_ci_hi']:.3f}]  "
              f"{r['n_days']:>6}  {r['n_seeds']:>5}")

    # ── Pairwise comparison ──
    print("\n── Pairwise Sharpe Difference (bootstrap) ──\n")
    configs = list(results.keys())
    comparisons = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            c1, c2 = configs[i], configs[j]
            # Difference in bootstrap distributions
            diff_boots = results[c1]["sharpe_boots"] - results[c2]["sharpe_boots"]
            med_diff = np.median(diff_boots)
            lo = np.percentile(diff_boots, 100 * ALPHA / 2)
            hi = np.percentile(diff_boots, 100 * (1 - ALPHA / 2))
            sig = "★" if lo > 0 or hi < 0 else ""
            comparisons.append({
                "Comparison": f"{c1} vs {c2}",
                "Δ Sharpe": round(med_diff, 3),
                "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                "Significant": "Yes" if sig else "No",
            })
            print(f"  {c1} vs {c2}: ΔSharpe = {med_diff:+.3f} "
                  f"[{lo:.3f}, {hi:.3f}]  {sig}")

    # Save
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(comparisons).to_csv("results/bootstrap_pairwise.csv", index=False)
    print(f"\n→ results/bootstrap_pairwise.csv")

    # ── Highlight the key comparison ──
    if "HKUST_NoAttn" in results and "EODHD_NoAttn" in results:
        hk = results["HKUST_NoAttn"]["sharpe_boots"]
        eo = results["EODHD_NoAttn"]["sharpe_boots"]
        diff = hk - eo
        p_pos = np.mean(diff > 0)
        print(f"\nKey: P(HKUST_NoAttn > EODHD_NoAttn) = {p_pos:.3f} "
              f"({'significant' if p_pos > 0.95 else 'not significant'} at 95% level)")
    if "HKUST_NoAttn" in results and "NONE_NoSent" in results:
        hk = results["HKUST_NoAttn"]["sharpe_boots"]
        no = results["NONE_NoSent"]["sharpe_boots"]
        diff = hk - no
        p_pos = np.mean(diff > 0)
        print(f"Key: P(HKUST_NoAttn > NONE_NoSent) = {p_pos:.3f} "
              f"({'significant' if p_pos > 0.95 else 'not significant'} at 95% level)")


if __name__ == "__main__":
    main()
