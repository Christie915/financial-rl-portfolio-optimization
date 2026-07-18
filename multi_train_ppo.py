"""
多资产 DRL 训练 (v5 - PPO + Cross-Asset Attention + Validation Checkpoint 挑优)

用法 (命令行):
  python multi_train_ppo.py --config HKUST_Full --seed 42
  python multi_train_ppo.py --config HKUST_NoAttn --seed 123
  ...

核心设计:
  1. 用 stable-baselines3 的 PPO + 自定义 Cross-Asset Attention feature extractor
  2. Validation 集 = 训练集的末 75 天 (约 2024 Q4), 不在这段数据上做参数更新
     每 50k steps eval 一次, 挑 Sharpe Ratio 最高的 checkpoint 保留
  3. 最终保存的是 val Sharpe 最高的那次权重, 不是训练末端
  4. 日志、模型文件、回测 CSV 都按 (config, seed) 独立命名, 不相互覆盖
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from multi_asset_data import MultiAssetDataEngine
from portfolio_env import PortfolioTradingEnv

import warnings
warnings.filterwarnings('ignore')


DOW_30_TICKERS = [
    "AAPL", "MSFT", "JPM", "V", "UNH", "JNJ", "WMT", "PG", "HD", "CVX",
    "MRK", "KO", "CSCO", "MCD", "DIS", "ADBE", "CRM", "VZ", "AMGN", "NKE",
    "IBM", "BA", "HON", "CAT", "GS", "TRV", "AXP", "MMM", "INTC", "AMZN"
]


# ==============================================================================
# Cross-Asset Attention Feature Extractor
# ==============================================================================
class CrossAssetAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space,
                 num_tokens, per_token_dim,
                 embed_dim=128, num_heads=4,
                 features_dim=256,
                 use_attention=True,
                 dropout=0.1):
        super().__init__(observation_space, features_dim=features_dim)
        self.num_tokens = num_tokens
        self.per_token_dim = per_token_dim
        self.use_attention = use_attention

        self.token_embed = nn.Linear(per_token_dim, embed_dim)

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True
            )
            self.layer_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        B = observations.size(0)
        tokens = observations.view(B, self.num_tokens, self.per_token_dim)

        h = self.token_embed(tokens)

        if self.use_attention:
            attn_out, _ = self.attention(h, h, h)
            h = self.layer_norm(h + attn_out)

        h = self.ffn(h) + h

        h_mean = h.mean(dim=1)
        q = self.pool_query.expand(B, -1, -1)
        scores = torch.matmul(q, h.transpose(1, 2)) / (h.size(-1) ** 0.5)
        w = F.softmax(scores, dim=-1)
        h_attn_pool = torch.matmul(w, h).squeeze(1)

        pooled = torch.cat([h_mean, h_attn_pool], dim=-1)
        return self.final_proj(pooled)


# ==============================================================================
# Validation Checkpoint Callback
# ==============================================================================
class ValidationCheckpointCallback(BaseCallback):
    """
    每 eval_freq 步, 在 validation 集上跑一次 deterministic 回测,
    算出 Sharpe Ratio. 保留 val Sharpe 最高的模型权重.
    """
    def __init__(self, val_data, val_tickers, eval_freq=50000,
                 model_save_path=None, verbose=1):
        super().__init__(verbose)
        self.val_data = val_data
        self.val_tickers = val_tickers
        self.eval_freq = eval_freq
        self.model_save_path = model_save_path

        self.best_val_sharpe = -np.inf
        self.best_val_return = 0.0
        self.best_step = 0

        self.history = []  # [(step, val_sharpe, val_return, val_mdd)]

    def _evaluate(self):
        """跑一次 validation"""
        val_env = PortfolioTradingEnv(self.val_data, self.val_tickers,
                                       transaction_cost=0.0001)
        obs, _ = val_env.reset()
        done = False

        daily_returns = []
        net_worths = [val_env.initial_balance]

        while not done:
            # 用训练中的策略, deterministic
            obs_batch = np.array([obs], dtype=np.float32)
            action, _ = self.model.predict(obs_batch, deterministic=True)
            action = action[0]  # unbatch
            obs, reward, done, _, info = val_env.step(action)
            daily_returns.append(info.get('portfolio_return', 0.0))
            net_worths.append(info['net_worth'])

        # 指标
        final_nw = net_worths[-1]
        ret_pct = (final_nw / val_env.initial_balance - 1) * 100
        nw_arr = np.array(net_worths)
        cummax = np.maximum.accumulate(nw_arr)
        mdd_pct = np.max((cummax - nw_arr) / cummax) * 100

        mean_r = np.mean(daily_returns)
        std_r = np.std(daily_returns)
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

        return sharpe, ret_pct, mdd_pct

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            val_sharpe, val_ret, val_mdd = self._evaluate()
            self.history.append({
                'step': self.n_calls,
                'val_sharpe': val_sharpe,
                'val_return': val_ret,
                'val_mdd': val_mdd,
            })

            star = ""
            if val_sharpe > self.best_val_sharpe:
                self.best_val_sharpe = val_sharpe
                self.best_val_return = val_ret
                self.best_step = self.n_calls
                if self.model_save_path:
                    self.model.save(self.model_save_path)
                star = " ★ NEW BEST"

            if self.verbose > 0:
                print(f"    [step {self.n_calls:>7d}] "
                      f"ValSharpe={val_sharpe:6.3f} | "
                      f"ValRet={val_ret:6.2f}% | "
                      f"ValMDD={val_mdd:5.2f}%{star}")
        return True


def build_config(config_name):
    """解析 config 名称 → 四元组 (source, use_sent, use_attn, use_confidence)"""
    mapping = {
        # (sentiment_source, use_sentiment, use_attention, use_confidence)
        "NONE_Vanilla":  ("HKUST", False, False, False),
        "NONE_NoSent":   ("HKUST", False, True,  False),
        "EODHD_NoAttn":  ("EODHD", True,  False, False),
        "EODHD_Full":    ("EODHD", True,  True,  False),
        "HKUST_NoAttn":  ("HKUST", True,  False, False),
        "HKUST_Full":    ("HKUST", True,  True,  False),
        # CGS variants: same as Full/NoAttn but with confidence gating
        "HKUST_Full_CGS":   ("HKUST", True, True,  True),
        "HKUST_NoAttn_CGS": ("HKUST", True, False, True),
    }
    return mapping[config_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='e.g. HKUST_Full')
    parser.add_argument('--seed', type=int, required=True, help='e.g. 42')
    parser.add_argument('--timesteps', type=int, default=400_000)
    parser.add_argument('--val_days', type=int, default=75,
                        help='验证集 = 训练集末 N 天')
    parser.add_argument('--eval_freq', type=int, default=50_000)
    args = parser.parse_args()

    # 设种子
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sentiment_source, use_sentiment, use_attention, use_confidence = build_config(args.config)
    SUFFIX = f"_{args.config}_v5_ppo_s{args.seed}"

    print("\n" + "=" * 70)
    print(f"PPO 训练 | config={args.config} | seed={args.seed}")
    print(f"suffix={SUFFIX} | timesteps={args.timesteps:,}")
    print("=" * 70)

    # ---- 数据 ----
    engine = MultiAssetDataEngine(
        tickers=DOW_30_TICKERS, start_date="2022-01-01", end_date="2025-12-31"
    )
    engine.download_and_clean_price()

    if use_sentiment:
        if sentiment_source == "HKUST":
            sent_path = "stock_news/finbert_hkust_sentiment.csv"
        else:
            sent_path = "dow30_news_with_sentiment.csv"
        engine.load_sentiment_data(sent_path, confidence_threshold=0.3 if use_confidence else 0.0)

    panel_data, trade_dates = engine.build_3d_tensor(
        use_sentiment=use_sentiment, use_confidence=use_confidence, train_end_date="2024-12-31"
    )

    # 切分: 训练集 2022 - 2024 早期 / 验证集 2024 末 / 测试集 2025
    split_idx = 0
    for i, date in enumerate(trade_dates):
        if date.year == 2025:
            split_idx = i
            break

    full_train = panel_data[:split_idx]  # 2022-2024 全段
    val_start = max(0, len(full_train) - args.val_days)
    train_data = full_train[:val_start]       # 真正用于参数更新
    val_data = full_train[val_start:]         # validation
    print(f"训练集: {len(train_data)} 天 | Val 集: {len(val_data)} 天")

    # ---- Train env ----
    def make_train_env():
        return PortfolioTradingEnv(train_data, DOW_30_TICKERS, transaction_cost=0.0001)
    env = DummyVecEnv([make_train_env])

    base_env = env.envs[0]
    num_tokens = base_env.num_tokens
    per_token_dim = base_env.per_asset_feat_dim

    # ---- PPO ----
    policy_kwargs = dict(
        features_extractor_class=CrossAssetAttentionExtractor,
        features_extractor_kwargs=dict(
            num_tokens=num_tokens,
            per_token_dim=per_token_dim,
            embed_dim=128,
            num_heads=4,
            features_dim=256,
            use_attention=use_attention,
            dropout=0.1,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    os.makedirs('models', exist_ok=True)
    best_model_path = f"models/portfolio_ppo{SUFFIX}_best.zip"

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
        verbose=0,     # 用自己的 callback, SB3 内置 log 关掉
        seed=args.seed,
        device='auto',
    )
    print(f"模型参数量: {sum(p.numel() for p in model.policy.parameters()):,}")

    val_callback = ValidationCheckpointCallback(
        val_data=val_data,
        val_tickers=DOW_30_TICKERS,
        eval_freq=args.eval_freq,
        model_save_path=best_model_path,
        verbose=1,
    )

    print(f"开始训练, 每 {args.eval_freq:,} 步做一次 val eval...")
    model.learn(total_timesteps=args.timesteps, callback=val_callback, progress_bar=False)

    # 总是保留一个 final 副本 (防万一 val eval 从没 triggered)
    final_model_path = f"models/portfolio_ppo{SUFFIX}_final.zip"
    model.save(final_model_path)

    # 如果 best_model_path 从来没被写过 (比如 eval_freq > timesteps), 用 final 代替
    if not os.path.exists(best_model_path):
        print(f"[warn] best 模型未触发 eval, 用 final 替代")
        import shutil
        shutil.copy(final_model_path, best_model_path)

    print(f"\n最优 checkpoint: step {val_callback.best_step}, ValSharpe={val_callback.best_val_sharpe:.3f}")
    print(f"1. 最佳模型 → {best_model_path}")
    print(f"2. 最终模型 → {final_model_path}")

    # 保存 val 历史
    os.makedirs('results', exist_ok=True)
    val_log_path = f"results/val_history{SUFFIX}.csv"
    pd.DataFrame(val_callback.history).to_csv(val_log_path, index=False)
    print(f"3. Val 日志 → {val_log_path}")

    # 画 val 曲线
    if val_callback.history:
        df = pd.DataFrame(val_callback.history)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df['step'], df['val_sharpe'], 'o-', color='darkblue', label='Val Sharpe')
        ax.axvline(val_callback.best_step, color='red', linestyle='--', alpha=0.5,
                   label=f'Best @ step {val_callback.best_step}')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Val Sharpe')
        ax.set_title(f'Validation Checkpoint Selection | {args.config} seed={args.seed}')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"results/val_curve{SUFFIX}.png", dpi=130)
        plt.close()

    print("=" * 70)


if __name__ == "__main__":
    main()
