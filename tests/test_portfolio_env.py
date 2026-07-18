"""
Unit tests for PortfolioTradingEnv — verifies invariants that must hold
regardless of market data.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_env import PortfolioTradingEnv


def _make_synthetic_panel(seed: int = 42, n_days: int = 60, n_assets: int = 3):
    """Build a minimal valid 3-D panel (T, N, F) with controlled returns."""
    rng = np.random.default_rng(seed)
    F = 5  # return, vol, vol_chg, macd, rsi
    data = np.zeros((n_days, n_assets, F), dtype=np.float32)
    # Small random returns around zero
    data[:, :, 0] = rng.normal(0.0005, 0.01, (n_days, n_assets)).astype(np.float32)
    # Fill other features with small noise so z-score standardization works
    for f in range(1, F):
        data[:, :, f] = rng.normal(0, 0.5, (n_days, n_assets)).astype(np.float32)
    return data


class TestPortfolioTradingEnv:
    """Core invariants that must hold under any panel data."""

    def test_initial_weights_cash_is_one(self):
        panel = _make_synthetic_panel()
        env = PortfolioTradingEnv(panel, ["A", "B", "C"])
        obs, _ = env.reset()
        np.testing.assert_almost_equal(env.current_weights[0], 1.0)
        np.testing.assert_array_almost_equal(env.current_weights[1:], np.zeros(3))

    def test_step_weights_are_valid_distribution(self):
        """After any step, weights must be non-negative and sum to ~1."""
        panel = _make_synthetic_panel(n_days=100)
        env = PortfolioTradingEnv(panel, ["A", "B", "C"])
        env.reset()
        rng = np.random.default_rng(42)
        for _ in range(20):
            action = rng.uniform(-3, 3, 4).astype(np.float32)
            env.step(action)
        w = env.current_weights
        assert np.all(w >= 0), f"negative weights: {w}"
        np.testing.assert_almost_equal(np.sum(w), 1.0, decimal=5)

    def test_net_worth_decreases_with_cost_no_return(self):
        """When returns are zero but cost > 0, net worth must strictly decrease."""
        F = 5
        panel = np.zeros((60, 2, F), dtype=np.float32)  # all zeros → zero returns
        env = PortfolioTradingEnv(panel, ["A", "B"], transaction_cost=0.001)
        env.reset()
        nw_before = env.net_worth
        # Action: shift from [1,0,0] to [0.5, 0.25, 0.25] → positive turnover
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        env.step(action)
        assert env.net_worth < nw_before, f"net_worth did not decrease: {nw_before} → {env.net_worth}"

    def test_state_shape_consistent(self):
        """State returned by reset() and step() must have identical shape."""
        panel = _make_synthetic_panel(n_days=100, n_assets=3)
        env = PortfolioTradingEnv(panel, ["A", "B", "C"])
        obs_reset, _ = env.reset()
        obs_step, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        assert obs_reset.shape == obs_step.shape
        # Shape = (num_tokens * per_token_dim,) = (4 * (5*5 + 1),) = (4 * 26,) = (104,)
        assert obs_reset.shape == (env.state_dim,)

    def test_deterministic_reset_with_seed(self):
        """Same seed must produce identical initial state."""
        panel = _make_synthetic_panel()
        e1 = PortfolioTradingEnv(panel, ["A", "B", "C"])
        e2 = PortfolioTradingEnv(panel, ["A", "B", "C"])
        obs1, _ = e1.reset(seed=42)
        obs2, _ = e2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_softmax_action_invariant_to_constant_shift(self):
        """Softmax is invariant to adding a constant to all logits."""
        panel = _make_synthetic_panel(n_days=100)
        env1 = PortfolioTradingEnv(panel, ["A", "B", "C"])
        env2 = PortfolioTradingEnv(panel, ["A", "B", "C"])
        env1.reset(seed=1)
        env2.reset(seed=1)

        action1 = np.array([1.0, -0.5, 0.5, -1.0], dtype=np.float32)
        action2 = action1 + 2.0  # constant shift

        _, _, _, _, info1 = env1.step(action1)
        _, _, _, _, info2 = env2.step(action2)

        np.testing.assert_almost_equal(info1['net_worth'], info2['net_worth'], decimal=3)

    def test_done_flag_fires_before_end(self):
        """done should become True before we run out of data."""
        panel = _make_synthetic_panel(n_days=30)
        env = PortfolioTradingEnv(panel, ["A", "B", "C"])
        env.reset()
        done = False
        steps = 0
        while not done:
            action = np.zeros(4, dtype=np.float32)
            _, _, done, _, _ = env.step(action)
            steps += 1
            if steps > 200:
                pytest.fail("env never terminated after 200 steps")
        # We should terminate at time_steps - 1 - window_size ≈ 30 - 1 - 5 = 24
        assert steps < 30

    def test_net_worth_tracks_in_history(self):
        """net_worth history must exactly match step return values."""
        panel = _make_synthetic_panel(n_days=100)
        env = PortfolioTradingEnv(panel, ["A", "B", "C"])
        env.reset()
        for _ in range(10):
            env.step(np.zeros(4, dtype=np.float32))
        assert len(env.history_net_worth) == 11  # initial + 10 steps
        assert env.history_net_worth[-1] == env.net_worth
