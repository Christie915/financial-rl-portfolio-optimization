"""
Portfolio Trading Env (v5 PPO 版)

相较 v4:
  - 移除 risk reward 选项 (统一用纯收益 reward, 让 PPO 自己学)
  - 保留资产维度 state (N+1 tokens × per_token_dim 特征)
  - step info 多返回 Sharpe 估计需要的收益序列 (其实是通过外部累积)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, panel_data, tickers,
                 initial_balance=10000000.0,
                 window_size=5,
                 transaction_cost=0.0001):
        super().__init__()

        self.panel_data = panel_data
        self.tickers = tickers
        self.time_steps = panel_data.shape[0]
        self.num_assets = panel_data.shape[1]
        self.num_features = panel_data.shape[2]

        self.initial_balance = initial_balance
        self.window_size = window_size
        self.transaction_cost = transaction_cost

        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.num_assets + 1,), dtype=np.float32
        )

        self.per_asset_feat_dim = self.window_size * self.num_features + 1
        self.num_tokens = self.num_assets + 1
        self.state_dim = self.num_tokens * self.per_asset_feat_dim
        self.asset_state_shape = (self.num_tokens, self.per_asset_feat_dim)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance

        self.current_weights = np.zeros(self.num_assets + 1)
        self.current_weights[0] = 1.0

        self.history_net_worth = [self.net_worth]
        return self._get_state(), {}

    def _get_state(self):
        window_data = self.panel_data[self.current_step - self.window_size:self.current_step]
        per_asset = window_data.transpose(1, 0, 2).reshape(self.num_assets, -1)
        asset_weights = self.current_weights[1:].reshape(-1, 1)
        per_asset_full = np.concatenate([per_asset, asset_weights], axis=1)

        cash_feat = np.zeros(self.window_size * self.num_features, dtype=np.float32)
        cash_full = np.concatenate([cash_feat, [self.current_weights[0]]])

        full_state = np.vstack([cash_full[None, :], per_asset_full])
        full_state = np.nan_to_num(full_state, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

        return full_state.flatten()

    def step(self, action):
        exp_preds = np.exp(action - np.max(action))
        target_weights = exp_preds / np.sum(exp_preds)

        # 用上一步价格漂移后的权重作基准，而非原始目标权重
        asset_returns = self.panel_data[self.current_step, :, 0]
        prev_asset_w = self.current_weights[1:]
        drifted_asset_w = prev_asset_w * (1 + asset_returns)
        drifted_sum = drifted_asset_w.sum() + self.current_weights[0]
        if drifted_sum > 0:
            drifted_cash_w = self.current_weights[0] / drifted_sum
            drifted_asset_w = drifted_asset_w / drifted_sum
        else:
            drifted_cash_w = self.current_weights[0]
        drifted_weights = np.concatenate([[drifted_cash_w], drifted_asset_w])

        weight_changes = np.abs(target_weights - drifted_weights)
        turnover = np.sum(weight_changes)
        cost_rate = turnover * self.transaction_cost

        portfolio_return = np.sum(target_weights[1:] * asset_returns)
        # 先扣交易成本，再享受当期收益
        self.net_worth = self.net_worth * (1 - cost_rate) * (1 + portfolio_return)
        net_return = (1 - cost_rate) * (1 + portfolio_return) - 1

        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.history_net_worth.append(self.net_worth)
        self.current_weights = target_weights

        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth

        self.current_step += 1
        done = self.current_step >= self.time_steps - 1

        # 风险调整收益：鼓励高收益同时惩罚高波动
        recent_returns = np.diff(self.history_net_worth[-21:]) / np.array(self.history_net_worth[-21:-1]) \
            if len(self.history_net_worth) > 2 else np.array([net_return])
        vol = float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0
        reward = (net_return - 0.1 * vol) * 100.0

        info = {
            'net_worth': self.net_worth,
            'portfolio_return': net_return,
            'cost_rate': cost_rate,
            'drawdown': drawdown,
            'turnover': turnover,
        }
        return self._get_state(), reward, done, False, info
