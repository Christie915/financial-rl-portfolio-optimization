"""
Generate three thesis figures:
  Figure 1: System architecture flowchart  (Section 3.1)
  Figure 2: Seed variance bar chart         (Section 4.4)
  Figure 3: CGS comparison chart            (Section 4.5.2)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrowPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / "results"
RES = PROJECT_ROOT / "results"

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: System Architecture Flowchart
# ─────────────────────────────────────────────────────────────────────────────
def make_arch_figure():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def box(ax, x, y, w, h, text, color, fontsize=9.5, subtext=None):
        fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.12",
                               fc=color, ec='#444444', lw=1.2, zorder=3)
        ax.add_patch(fancy)
        if subtext:
            ax.text(x, y + 0.18, text, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', zorder=4)
            ax.text(x, y - 0.28, subtext, ha='center', va='center',
                    fontsize=7.8, color='#333333', zorder=4, style='italic')
        else:
            ax.text(x, y, text, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', zorder=4)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
                    zorder=2)

    # Stage colours
    C1 = '#D6EAF8'   # data
    C2 = '#D5F5E3'   # tensor
    C3 = '#FDEBD0'   # PPO
    C4 = '#F9EBEA'   # eval
    CHEAD = '#EBF5FB'

    # ── Stage labels ──────────────────────────────────────────────────────────
    for x, lbl, col in [(2.0, 'Stage 1\nData Engine', C1),
                         (5.2, 'Stage 2\nTensor Builder', C2),
                         (8.4, 'Stage 3\nPPO Agent', C3),
                         (11.4, 'Stage 4\nEvaluator', C4)]:
        ax.text(x, 6.55, lbl, ha='center', va='center', fontsize=8.5,
                fontweight='bold', color='#1a1a1a',
                bbox=dict(boxstyle='round,pad=0.3', fc=col, ec='#aaaaaa', lw=0.8))

    # ── Stage 1 boxes ─────────────────────────────────────────────────────────
    box(ax, 2.0, 5.4, 3.2, 0.75, 'Price & OHLCV Data', C1,
        subtext='yfinance · Dow 30 · 2022–2025')
    box(ax, 2.0, 4.3, 3.2, 0.75, 'Technical Indicators', C1,
        subtext='Return · Vol · MACD · RSI · VolChg')
    box(ax, 2.0, 3.2, 3.2, 0.75, 'News Corpus', C1,
        subtext='310,648 articles · 2022–2025')
    box(ax, 2.0, 2.1, 3.2, 0.75, 'FinBERT Inference', C1,
        subtext='HKUST yiyanghkust/finbert-tone')
    box(ax, 2.0, 1.0, 3.2, 0.75, 'EODHD Sentiment', C1,
        subtext='Generic news baseline')

    arrow(ax, 2.0, 5.02, 2.0, 4.67)
    arrow(ax, 2.0, 3.92, 2.0, 3.57)
    arrow(ax, 2.0, 2.82, 2.0, 2.47)

    # ── Stage 2 boxes ─────────────────────────────────────────────────────────
    box(ax, 5.2, 4.3, 2.8, 0.75, '3-D State Tensor', C2,
        subtext='(T × 30 × F)  strict Z-score isolation')
    box(ax, 5.2, 3.0, 2.8, 0.75, 'Train / Val / Test Split', C2,
        subtext='Train 2022–2024 · Val 75 days\nTest 2025 (243 days)')

    arrow(ax, 3.6, 4.3, 4.8, 4.3)
    arrow(ax, 5.2, 3.92, 5.2, 3.37)

    # connector from Stage1 price indicators to Stage2
    ax.annotate('', xy=(4.8, 4.0), xytext=(3.6, 4.0),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0, linestyle='dashed'))

    # ── Stage 3 boxes ─────────────────────────────────────────────────────────
    box(ax, 8.4, 5.1, 2.8, 0.75, 'Cross-Asset Attention\n(optional)', C3,
        subtext='embed=128 · 4 heads · pool → 256-d')
    box(ax, 8.4, 3.8, 2.8, 0.75, 'PPO Policy', C3,
        subtext='Actor-Critic · softmax action\nclip=0.2 · ent=0.005')
    box(ax, 8.4, 2.55, 2.8, 0.75, 'Checkpoint Selection', C3,
        subtext='Best val-Sharpe @ every 50k steps')
    box(ax, 8.4, 1.35, 2.8, 0.75, 'Trading Environment', C3,
        subtext='TC = 1 bp/unit turnover\nreward = 100 × net return − vol_pen')

    arrow(ax, 6.6, 3.0, 7.0, 3.0)
    ax.annotate('', xy=(8.4, 4.65), xytext=(7.1, 3.0),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2,
                                connectionstyle='arc3,rad=-0.3'))
    arrow(ax, 8.4, 4.72, 8.4, 4.17)
    arrow(ax, 8.4, 3.42, 8.4, 2.92)
    arrow(ax, 8.4, 2.17, 8.4, 1.72)

    # ── Stage 4 boxes ─────────────────────────────────────────────────────────
    box(ax, 11.4, 4.3, 2.0, 0.75, 'Out-of-Sample\nBacktest (2025)', C4)
    box(ax, 11.4, 3.0, 2.0, 0.75, 'Multi-Seed\nAggregation', C4,
        subtext='Seeds 42 · 123 · 2024')
    box(ax, 11.4, 1.7, 2.0, 0.75, 'Metrics Report', C4,
        subtext='Sharpe · MDD · Alpha · Calmar')

    arrow(ax, 9.8, 2.55, 11.05, 4.0)
    arrow(ax, 11.4, 3.92, 11.4, 3.37)
    arrow(ax, 11.4, 2.62, 11.4, 2.07)

    ax.set_title('Figure 1 — System Architecture Overview', fontsize=12,
                 fontweight='bold', pad=10)
    plt.tight_layout()
    path = OUT / 'fig1_system_architecture.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Seed Variance Bar Chart (Section 4.4)
# ─────────────────────────────────────────────────────────────────────────────
def make_seed_variance_figure():
    configs = ['NONE_Vanilla', 'NONE_NoSent', 'EODHD_NoAttn',
               'EODHD_Full', 'HKUST_NoAttn', 'HKUST_Full']
    labels  = ['NONE\nVanilla', 'NONE\nNoSent', 'EODHD\nNoAttn',
               'EODHD\nFull', 'HKUST\nNoAttn\n(flagship)', 'HKUST\nFull']

    medians, stds, seed42, seed123, seed2024 = [], [], [], [], []
    for cfg in configs:
        path = RES / f'backtest_metrics_aggregated_{cfg}_v5_ppo.csv'
        df = pd.read_csv(path)
        sr = df[df['Metric'] == 'Sharpe Ratio'].iloc[0]
        medians.append(float(sr['Median']))
        stds.append(float(sr['Std']))
        seed42.append(float(sr['seed_42']))
        seed123.append(float(sr['seed_123']))
        seed2024.append(float(sr['seed_2024']))

    x = np.arange(len(configs))
    colors = ['#5B9BD5', '#70AD47', '#ED7D31', '#FFC000', '#FF0000', '#7030A0']
    flagship_idx = 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: median Sharpe with individual seed dots
    bars = ax1.bar(x, medians, color=colors, alpha=0.75, edgecolor='#333333',
                   linewidth=0.8, zorder=2)
    ax1.errorbar(x, medians, yerr=stds, fmt='none', ecolor='#333333',
                 elinewidth=1.5, capsize=5, zorder=3)
    for i, (s42, s123, s24) in enumerate(zip(seed42, seed123, seed2024)):
        ax1.scatter([i]*3, [s42, s123, s24], color='black', s=28, zorder=4,
                    marker='D')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel('Sharpe Ratio', fontsize=10)
    ax1.set_title('(a) Median Sharpe Ratio ± Std\n(◆ = individual seeds)', fontsize=10)
    ax1.set_ylim(0.8, 2.1)
    ax1.axhline(medians[flagship_idx], color='red', lw=1.0, ls='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3, zorder=1)
    bars[flagship_idx].set_edgecolor('red')
    bars[flagship_idx].set_linewidth(2.0)

    # Right: Sharpe std (seed variance)
    ax2.bar(x, stds, color=colors, alpha=0.75, edgecolor='#333333',
            linewidth=0.8, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel('Sharpe Std (across 3 seeds)', fontsize=10)
    ax2.set_title('(b) Seed-Level Variance\n(higher = more unstable)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, zorder=1)
    for i, v in enumerate(stds):
        ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Figure 2 — Median Sharpe Ratio and Seed Variance Across Six Configurations\n'
                 'Out-of-Sample 2025, Three Seeds (42 · 123 · 2024)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = OUT / 'fig2_seed_variance.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: CGS Comparison Chart (Section 4.5.2)
# ─────────────────────────────────────────────────────────────────────────────
def make_cgs_figure():
    # Data from Table 4.3
    data = {
        'HKUST_NoAttn\n(baseline)':    {'median': 1.411, 'std': 0.286, 'color': '#5B9BD5'},
        'HKUST_NoAttn\n_CGS':          {'median': 1.116, 'std': 0.012, 'color': '#9DC3E6'},
        'HKUST_Full\n(baseline)':      {'median': 1.147, 'std': 0.102, 'color': '#FF0000'},
        'HKUST_Full\n_CGS':            {'median': 1.208, 'std': 0.081, 'color': '#FF9999'},
    }
    labels  = list(data.keys())
    medians = [v['median'] for v in data.values()]
    stds    = [v['std']    for v in data.values()]
    colors  = [v['color']  for v in data.values()]

    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: median Sharpe
    bars = ax1.bar(x, medians, color=colors, edgecolor='#333333',
                   linewidth=0.9, alpha=0.85, zorder=2)
    ax1.errorbar(x, medians, yerr=stds, fmt='none', ecolor='#333333',
                 elinewidth=1.5, capsize=6, zorder=3)
    for i, (m, s) in enumerate(zip(medians, stds)):
        ax1.text(i, m + s + 0.015, f'{m:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Median Sharpe Ratio', fontsize=10)
    ax1.set_title('(a) Median Sharpe Ratio ± Std', fontsize=10)
    ax1.set_ylim(0.8, 1.85)
    ax1.grid(axis='y', alpha=0.3, zorder=1)

    # Annotate deltas
    ax1.annotate('', xy=(1, 1.116), xytext=(0, 1.411),
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5,
                                 connectionstyle='arc3,rad=0.25'))
    ax1.text(0.5, 1.30, '−20.9%', ha='center', color='#c0392b', fontsize=8.5)

    ax1.annotate('', xy=(3, 1.208), xytext=(2, 1.147),
                 arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5,
                                 connectionstyle='arc3,rad=-0.25'))
    ax1.text(2.5, 1.19, '+5.3%', ha='center', color='#27ae60', fontsize=8.5)

    # Right: seed std (variance reduction)
    ax2.bar(x, stds, color=colors, edgecolor='#333333',
            linewidth=0.9, alpha=0.85, zorder=2)
    for i, v in enumerate(stds):
        ax2.text(i, v + 0.003, f'{v:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Sharpe Std (3 seeds)', fontsize=10)
    ax2.set_title('(b) Seed Variance\n(lower = more stable)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, zorder=1)

    ax2.annotate('', xy=(1, 0.012), xytext=(0, 0.286),
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5,
                                 connectionstyle='arc3,rad=0.25'))
    ax2.text(0.5, 0.17, '×24 reduction', ha='center', color='#c0392b', fontsize=8.5)

    # Legend patches
    p1 = mpatches.Patch(color='#5B9BD5', label='HKUST_NoAttn (baseline)')
    p2 = mpatches.Patch(color='#9DC3E6', label='HKUST_NoAttn_CGS')
    p3 = mpatches.Patch(color='#FF0000', label='HKUST_Full (baseline)')
    p4 = mpatches.Patch(color='#FF9999', label='HKUST_Full_CGS')
    fig.legend(handles=[p1, p2, p3, p4], loc='lower center', ncol=4,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Figure 3 — Confidence-Gated Sentiment (CGS) vs Baseline\n'
                 'Median Sharpe and Seed Variance, Out-of-Sample 2025',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = OUT / 'fig3_cgs_comparison.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    make_arch_figure()
    make_seed_variance_figure()
    make_cgs_figure()
    print('All figures generated.')
