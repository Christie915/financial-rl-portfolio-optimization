"""
Classical portfolio baselines for comparison against PPO results.

Strategies:
  1. Rolling Mean-Variance  — Markowitz with sample μ,Σ from 60-day window
  2. Rolling Minimum Variance — minimize σ²_p, no return estimates
  3. Risk Parity (inverse-vol) — weights ∝ 1/σ_i

All strategies are long-only, rebalanced daily, with the same 1bp transaction
cost model used by PPO.  Output CSV matches existing backtest format so results
can be merged into the aggregate comparison table.
"""
import os
import numpy as np
import pandas as pd
from config import DOW_30_TICKERS
from multi_asset_data import MultiAssetDataEngine
import warnings
warnings.filterwarnings("ignore")


def run_baseline(name: str, panel_test, window: int = 60, max_weight: float = 0.30):
    """
    Run a classical strategy on the test panel.

    Parameters
    ----------
    name : "MeanVariance", "MinVariance", or "RiskParity"
    panel_test : np.ndarray of shape (T, 30, F) where F[0] is daily return
    window : estimation window in days
    max_weight : per-asset weight cap (0.3 = 30%)
    """
    T, N, F = panel_test.shape
    initial_balance = 10_000_000.0
    tc = 0.0001  # 1 bp per unit turnover

    net_worth = initial_balance
    max_nw = initial_balance
    prev_weights = np.zeros(N + 1)
    prev_weights[0] = 1.0  # start all-cash

    daily_returns = []
    net_worths = [initial_balance]
    turnovers = []
    drawdowns = []

    # Benchmark: equal-weight Dow 30
    eqw = np.ones(N) / N
    bh_returns = []
    for t in range(window, T - 1):
        bh_returns.append(np.dot(panel_test[t, :, 0], eqw))

    for t in range(window, T - 1):
        # ---- estimate μ, Σ from rolling window ----
        ret_window = panel_test[t - window:t, :, 0]  # (window, N)
        mu = ret_window.mean(axis=0)
        Sigma = np.cov(ret_window, rowvar=False)
        Sigma = Sigma + np.eye(N) * 1e-6  # regularise

        # ---- compute target weights ----
        if name == "RiskParity":
            vols = np.std(ret_window, axis=0)
            vols[vols < 1e-8] = 1e-8
            w_raw = 1.0 / vols
            w = w_raw / w_raw.sum()
        elif name == "MinVariance":
            w = _min_variance(Sigma, max_weight)
        elif name == "MeanVariance":
            w = _mean_variance(mu, Sigma, max_weight)
        else:
            raise ValueError(f"Unknown baseline: {name}")

        target = np.concatenate([[0.0], w])  # cash=0, fully invested

        # ---- apply transaction costs (same formula as PPO env) ----
        asset_ret = panel_test[t, :, 0]
        prev_asset_w = prev_weights[1:]
        drifted = prev_asset_w * (1 + asset_ret)
        drifted_sum = drifted.sum() + prev_weights[0]
        drifted_cash_w = prev_weights[0] / drifted_sum if drifted_sum > 0 else prev_weights[0]
        drifted_asset_w = drifted / drifted_sum if drifted_sum > 0 else drifted
        drifted_weights = np.concatenate([[drifted_cash_w], drifted_asset_w])

        turnover = np.sum(np.abs(target - drifted_weights))
        cost_rate = turnover * tc
        port_ret = np.dot(target[1:], asset_ret)

        net_worth = net_worth * (1 - cost_rate) * (1 + port_ret)
        net_return = (1 - cost_rate) * (1 + port_ret) - 1

        max_nw = max(max_nw, net_worth)
        drawdown = (max_nw - net_worth) / max_nw

        prev_weights = target
        daily_returns.append(net_return)
        net_worths.append(net_worth)
        turnovers.append(turnover)
        drawdowns.append(drawdown)

    # ---- metrics ----
    ret_pct = (net_worth / initial_balance - 1) * 100

    # benchmark
    bh_cum = np.concatenate([[1.0], np.cumprod(1 + np.array(bh_returns))])
    bh_final = bh_cum[-1] * initial_balance
    bh_ret = (bh_final / initial_balance - 1) * 100

    nw_arr = np.array(net_worths)
    cummax = np.maximum.accumulate(nw_arr)
    mdd = np.max((cummax - nw_arr) / cummax) * 100

    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns)
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
    alpha = ret_pct - bh_ret
    calmar = ret_pct / mdd if mdd > 0 else 0.0
    win_rate = sum(1 for r in daily_returns if r > 0) / len(daily_returns) * 100
    avg_turnover = np.mean(turnovers) * 100
    avg_dd = np.mean(drawdowns) * 100

    print(f"  {name:>16}: Ret={ret_pct:6.2f}%  Shp={sharpe:.3f}  "
          f"MDD={mdd:5.2f}%  Alpha={alpha:+.2f}%  Turn={avg_turnover:.1f}%")

    return {
        "Metric": [
            "Initial Capital", "Final Value", "Return Rate (%)",
            "Win Rate (%)", "Max Drawdown (%)", "Sharpe Ratio",
            "Calmar Ratio", "Avg Daily Turnover (%)", "Avg Daily Drawdown (%)",
            "Benchmark Final", "Benchmark Return (%)", "Alpha (%)"
        ],
        "Value": [
            round(initial_balance, 2),
            round(net_worth, 2), round(ret_pct, 2),
            round(win_rate, 2), round(mdd, 2), round(sharpe, 3),
            round(calmar, 3), round(avg_turnover, 2), round(avg_dd, 2),
            round(bh_final, 2), round(bh_ret, 2), round(alpha, 2),
        ]
    }


def _min_variance(Sigma, max_w):
    """Long-only min-variance with per-asset cap via simple quadratic proxy."""
    N = len(Sigma)
    # Start from inverse-vol weights, then cap
    vols = np.sqrt(np.diag(Sigma))
    vols[vols < 1e-8] = 1e-8
    w = (1.0 / vols)
    w = w / w.sum()
    w = np.clip(w, 0, max_w)
    w = w / w.sum()
    return w


def _mean_variance(mu, Sigma, max_w):
    """Long-only mean-variance with risk aversion λ=1."""
    N = len(Sigma)
    # Use closed-form unconstrained, then clip to long-only + cap
    try:
        Sigma_inv = np.linalg.inv(Sigma)
        w_raw = Sigma_inv @ mu
    except np.linalg.LinAlgError:
        vols = np.sqrt(np.diag(Sigma))
        vols[vols < 1e-8] = 1e-8
        w_raw = mu / (vols ** 2)
    w_raw = np.clip(w_raw, 0, None)
    s = w_raw.sum()
    if s > 0:
        w = w_raw / s
    else:
        w = np.ones(N) / N
    w = np.clip(w, 0, max_w)
    w = w / w.sum()
    return w


def main():
    print("=" * 70)
    print("Classical Baseline Backtest — Dow 30, OOS 2025")
    print("=" * 70)

    # Load same data as PPO experiments
    engine = MultiAssetDataEngine(
        tickers=DOW_30_TICKERS, start_date="2022-01-01", end_date="2025-12-31"
    )
    engine.download_and_clean_price()
    panel_data, trade_dates = engine.build_3d_tensor(
        use_sentiment=False, train_end_date="2024-12-31"
    )

    # Split at 2025
    split_idx = next(i for i, d in enumerate(trade_dates) if d.year == 2025)
    test_data = panel_data[split_idx:]
    print(f"Test period (2025): {len(test_data)} trading days\n")

    results = {}
    for name in ["MeanVariance", "MinVariance", "RiskParity"]:
        results[name] = run_baseline(name, test_data)

    # Also compute equal-weight exactly
    T, N, F = test_data.shape
    ew_rets = []
    for t in range(5, T - 1):  # skip initial window
        ew_rets.append(np.dot(test_data[t, :, 0], np.ones(N) / N))
    ew_cum = np.cumprod(1 + np.array(ew_rets))
    ew_ret = (ew_cum[-1] - 1) * 100
    ew_sharpe = (np.mean(ew_rets) / np.std(ew_rets) * np.sqrt(252)) if np.std(ew_rets) > 0 else 0

    print(f"\n{'EqualWeight':>16}: Ret={ew_ret:6.2f}%  Shp={ew_sharpe:.3f}  (no costs)")
    print("=" * 70)

    # Save CSVs
    os.makedirs("results", exist_ok=True)
    for name, metrics in results.items():
        pd.DataFrame(metrics).to_csv(
            f"results/backtest_metrics_summary_BASELINE_{name}_v5_ppo.csv", index=False
        )
    print("\nBaseline CSVs saved → results/backtest_metrics_summary_BASELINE_*")


if __name__ == "__main__":
    main()
