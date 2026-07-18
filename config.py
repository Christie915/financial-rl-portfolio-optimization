"""
Shared configuration for the PPO portfolio optimization framework.
All modules import constants and config mapping from here.
"""

# ── Universe ──────────────────────────────────────────────────────────────────
DOW_30_TICKERS = [
    "AAPL", "MSFT", "JPM", "V", "UNH", "JNJ", "WMT", "PG", "HD", "CVX",
    "MRK", "KO", "CSCO", "MCD", "DIS", "ADBE", "CRM", "VZ", "AMGN", "NKE",
    "IBM", "BA", "HON", "CAT", "GS", "TRV", "AXP", "MMM", "INTC", "AMZN"
]

# ── Experiment grid ───────────────────────────────────────────────────────────
CONFIGS_MAIN = [
    'HKUST_Full', 'HKUST_NoAttn',
    'EODHD_Full', 'EODHD_NoAttn',
    'NONE_NoSent', 'NONE_Vanilla',
]

CONFIGS_CGS = [
    'HKUST_Full_CGS', 'HKUST_NoAttn_CGS',
]

CONFIGS_ALL = CONFIGS_MAIN + CONFIGS_CGS

SEEDS = [42, 123, 2024]

# ── Training defaults ─────────────────────────────────────────────────────────
TIMESTEPS = 400_000
EVAL_FREQ = 50_000
VAL_DAYS = 75

# ── Config → (sentiment_source, use_sentiment, use_attention, use_confidence) ─
CONFIG_MAPPING = {
    "NONE_Vanilla":      ("HKUST", False, False, False),
    "NONE_NoSent":       ("HKUST", False, True,  False),
    "EODHD_NoAttn":      ("EODHD", True,  False, False),
    "EODHD_Full":        ("EODHD", True,  True,  False),
    "HKUST_NoAttn":      ("HKUST", True,  False, False),
    "HKUST_Full":        ("HKUST", True,  True,  False),
    "HKUST_Full_CGS":    ("HKUST", True,  True,  True),
    "HKUST_NoAttn_CGS":  ("HKUST", True,  False, True),
}


def build_config(config_name: str) -> tuple:
    """Parse config name → (sentiment_source, use_sentiment, use_attention, use_confidence)."""
    return CONFIG_MAPPING[config_name]
