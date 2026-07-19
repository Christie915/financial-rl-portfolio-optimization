# Research Summary

A condensed account of the methodology and findings behind this repository. It is written to be read in a few minutes; the full dissertation is not included here.

---

## 1. Research question

Multi-asset portfolio systems built on deep reinforcement learning (DRL) frequently report strong backtest performance, but two problems make results difficult to compare across studies:

1. **Single-seed reporting.** DRL is known to be highly sensitive to random seed choice ([Henderson et al., 2017](https://arxiv.org/abs/1709.06560); [Colas et al., 2018](https://arxiv.org/abs/1806.08295)), yet applied financial DRL work still commonly reports a single run.
2. **Under-audited preprocessing.** Sentiment coverage over the training window and the temporal scope of standardisation statistics are rarely stated, leaving room for look-ahead leakage that is difficult to detect after the fact.

This project asks: **does domain-specific financial sentiment improve out-of-sample portfolio allocation, when the claim is required to survive multi-seed evaluation, ablation, and a second walk-forward fold?**

---

## 2. Data and features

| Component | Detail |
|---|---|
| Universe | 30 large-cap US equities (project Dow 30 list) |
| Period | 2022-01-01 to 2025-12-31 |
| Price data | Daily OHLCV via `yfinance` |
| Features per asset | Daily return, 20-day rolling volatility, volume change, MACD histogram, RSI, sentiment |
| News corpus | 310,648 articles, scored locally with a domain-specific FinBERT-tone model |
| Sentiment score | `P(positive) − P(negative)` ∈ [−1, 1], mean-aggregated per ticker-day and aligned to trading dates |
| Generic baseline | Non-domain-adapted sentiment scores from a commercial data API |

When daily sentiment is aligned to trading dates, the latest available score is
forward-filled; zero is used only where no earlier score exists. The trading-date
index is the intersection across all thirty tickers, which prevents stale prices
entering the state tensor.

The state is a `(T, N, F)` tensor — trading day × asset × feature — with the raw daily return deliberately left unstandardised so that the environment's profit-and-loss accounting stays in physical units.

---

## 3. Agent and environment

- **Algorithm:** PPO (`stable-baselines3`), chosen for its stability under a clipped surrogate objective.
- **Observation:** 5-day sliding window over all features for 31 tokens (30 assets + cash), each token carrying its own current portfolio weight.
- **Action:** 31-dimensional continuous vector in [−3, 3], softmax-transformed into long-only weights summing to one. The bounded range avoids the near-one-hot allocations that unbounded logits produce.
- **Reward:** `100 × (R_net − 0.1 × σ_recent)`, where `R_net` is the daily portfolio return net of transaction costs and `σ_recent` is the volatility of the most recent return window.
- **Transaction costs:** 1 bp per unit turnover, measured as the L1 norm of the weight change against *drift-adjusted* previous weights, so passive price drift is not charged as turnover.
- **Optional cross-asset attention:** 4-head self-attention over the 31 tokens, embedding dimension 128, with residual connections and layer normalisation.
- **Training:** 400,000 environment steps per configuration per seed; lr 3e-4, rollout 2048, minibatch 256, 10 epochs, γ 0.99, GAE λ 0.95, clip 0.2, entropy coefficient 0.005.

---

## 4. Evaluation protocol

This is the part of the project that matters most, and the part most likely to differ from comparable work.

**Time isolation.** Z-score parameters for every non-return feature are estimated *only* on the 2022–2024 training slice and applied unchanged to the 2025 test window. Fitting these statistics over the full sample is mathematically valid in-sample but injects a mild look-ahead bias, since 2025 feature distributions would not be available at the point of a live 2025 decision.

**Checkpoint selection.** Every 50,000 steps the policy is evaluated deterministically on a held-out 75-day validation tail of the training window; the checkpoint with the highest validation Sharpe is retained. No gradient updates are computed on validation data, and the test set plays no role in model selection.

**Multi-seed reporting.** Six configurations (three sentiment sources × attention on/off) × three fixed seeds (42, 123, 2024) = 18 independent runs. Seeds were fixed in advance and all results are reported regardless of outcome. Metrics are given as median ± standard deviation, with best-seed values noted separately rather than presented as the headline.

**Statistical inference.** Daily out-of-sample returns are pooled across seeds and resampled 10,000 times to produce percentile bootstrap confidence intervals and pairwise comparison probabilities.

**Walk-forward.** A second fold (train 2022–2023, test 2024) repeats all six configurations under the same three seeds, giving 18 further runs.

---

## 5. Results

### 5.1 Main ablation — out-of-sample 2025, three-seed median

| Configuration | Return % | Sharpe | MDD % | Alpha % |
|---|---:|---:|---:|---:|
| No sentiment | 18.48 | 1.153 | 14.79 | +0.12 |
| No sentiment + attention | 20.83 | 1.288 | 14.40 | +2.47 |
| Generic sentiment | 18.87 | 1.208 | 16.05 | +0.51 |
| Generic sentiment + attention | 18.25 | 1.158 | 15.12 | −0.11 |
| **Domain sentiment** | **26.07** | **1.411** | **14.22** | **+7.71** |
| Domain sentiment + attention | 17.95 | 1.147 | 14.47 | −0.42 |
| Equal-weight benchmark | 18.36 | 1.133 | — | 0.00 |

The alpha ladder — domain +7.71% > generic +0.51% > none +0.12% — indicates that **sentiment source quality, not architectural complexity, is the dominant lever** in this setting.

### 5.2 Classical baselines, same period and cost model

| Strategy | Return % | Sharpe | MDD % |
|---|---:|---:|---:|
| Mean-variance (constrained QP) | −1.40 | 0.025 | 18.10 |
| Minimum-variance (constrained QP) | 2.74 | 0.329 | 10.61 |
| Risk parity (inverse volatility) | 13.03 | 1.091 | 11.22 |
| Equal weight | 18.36 | 1.133 | — |
| PPO, domain sentiment (median) | 26.07 | 1.411 | 14.22 |

Rolling strategies use a 60-day estimation window and are therefore evaluated over ~183 rather than 243 trading days. Mean-variance optimisation underperforms badly, consistent with the well-documented instability of sample-covariance Markowitz allocation when asset count approaches window length.

### 5.3 Walk-forward consistency

| Configuration | Fold | Sharpe | Alpha % |
|---|---|---:|---:|
| Domain sentiment | 2024 | 1.382 | +16.73 |
| Domain sentiment | 2025 | 1.411 | +7.71 |
| Domain + attention | 2024 | 1.148 | +4.59 |
| Domain + attention | 2025 | 1.147 | −0.42 |

The domain-sentiment configuration without attention is the only sentiment-bearing configuration producing positive alpha above +5% in both folds. Benchmark returns differ substantially across folds (34.32% in 2024, 18.36% in 2025), so alpha rather than absolute return is the appropriate cross-fold comparator.

---

## 6. Seed variance

Within the single best configuration, across three seeds:

| Seed | Return % | Sharpe | MDD % | Alpha % |
|---|---:|---:|---:|---:|
| 42 | 26.07 | 1.411 | 14.22 | +7.71 |
| 123 | 17.57 | 1.128 | 15.41 | −0.80 |
| 2024 | 30.05 | 1.825 | 10.45 | +11.68 |

A Sharpe spread of **0.69** and a return spread of **12.5 percentage points** arise from nothing but the random seed. A single-seed report of this same configuration could honestly have claimed any Sharpe between 1.128 and 1.825. That spread exceeds the performance improvement typically attributed to architectural changes in comparable applied work, which is the central argument for treating multi-seed reporting as a minimum standard rather than a robustness footnote.

---

## 7. Negative and non-significant results

These are retained deliberately.

**The advantage over a no-sentiment baseline is not statistically significant at three seeds.** The bootstrap probability that the domain-sentiment configuration outperforms the generic-sentiment baseline is 0.961, whereas the corresponding probability against the no-sentiment baseline is **0.537**. The economic difference in medians is large; the formal statistical confirmation is not there at this seed count, and the results should be read accordingly.

**Cross-asset attention hurt performance here.** Adding attention to the domain-sentiment configuration reduced median Sharpe from 1.411 to 1.147 and alpha from +7.71% to −0.42%. A capacity argument explains why this is plausible rather than anomalous: the attention block adds 66,304 parameters, while the Dow 30 over three training years supplies only 31 × 750 = 23,250 token-days. The resulting data-to-parameter ratio is **0.35**, roughly 30 times below a simple 10-observations-per-parameter heuristic. This is not a formal threshold, but it motivates a testable hypothesis for a larger-universe, longer-horizon study rather than a general claim that attention does not work.

**Confidence-gated sentiment failed.** Hard-filtering articles with |score| < 0.3 removed an estimated 60–70% of the corpus from daily aggregation. Median Sharpe fell from 1.411 to 1.116, while seed variance collapsed from 0.286 to 0.012 — a roughly 24-fold reduction. The interpretation is information loss, not a flaw in the confidence feature itself: a sparser signal admits a narrower set of plausible policies, so seeds converge to more similar and more mediocre outcomes.

---

## 8. Known limitations

- **Three seeds is a floor, not a target.** Additional seeds would be required to estimate small effects with useful precision. Three is sufficient to expose the instability of a single-seed claim, not to establish significance.
- **One universe, short horizon.** Dow 30 over three training years is small by current standards. The attention finding should be read as specific to this regime.
- **Two out-of-sample years.** Both 2024 and 2025 were broadly rising markets; neither fold tests a sustained drawdown regime.
- **Simple cost model.** Linear 1 bp per unit turnover ignores market impact, spread variation, and liquidity-dependent slippage. Median daily turnover for the best configuration is 57.81%, so a realistic retail-level friction assumption would materially reduce reported returns.
- **CGS is not decoupled.** The confidence-gating variant changes two things at once (threshold filtering and an added feature); a separated ablation would identify which drives the effect.
- **One sentiment model.** Alternative domain-adapted variants were not compared, despite sentiment source quality being the dominant factor identified here.

---

## 9. Reproducing the results

See the repository [README](../README.md) for setup and run commands. The raw news corpus and trained checkpoints are not distributed; [`data/README.md`](../data/README.md) documents the expected schema so the pipeline can be rebuilt from an equivalent source.

Seeds are fixed at 42, 123, and 2024. Preprocessing statistics are fitted on the training window only. Test folds remain chronologically separated from training throughout.

---

*This summary condenses an undergraduate final-year research project in Data Science and Big Data Technology. Third-party components — the FinBERT-tone sentiment model, `stable-baselines3`, Hugging Face `transformers`, and market data providers — remain under their respective licences.*
