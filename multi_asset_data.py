"""Multi-asset data engine (English rewrite + CGS novelty)."""
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
import warnings
warnings.filterwarnings("ignore")


class MultiAssetDataEngine:
    def __init__(self, tickers, start_date="2020-01-01", end_date="2025-12-31"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.sentiment_data = None
        self._sentiment_threshold_used = None

    def download_and_clean_price(self):
        print(f"\nDownloading price data for {len(self.tickers)} tickers...")
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            if df.empty:
                print(f"Warning: no data for {ticker}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = df.dropna()
            df["return"] = df["Close"].pct_change()
            df["volatility"] = df["return"].rolling(window=20).std()
            df["volume_change"] = df["Volume"].pct_change()
            macd = ta.trend.MACD(close=df["Close"])
            df["macd"] = macd.macd_diff()
            df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
            df = df.dropna()
            self.data[ticker] = df
        print(f"Price features ready for {len(self.data)} tickers.")

    def load_sentiment_data(self, csv_path, confidence_threshold=0.0):
        """
        Load FinBERT sentiment scores.
        confidence_threshold: drop entries where |sentiment_score| < threshold. 0.0 = keep all.
        """
        if not os.path.exists(csv_path):
            print(f"Sentiment file not found: {csv_path}")
            return
        self._sentiment_threshold_used = confidence_threshold
        print(f"Loading sentiment: {csv_path}")
        df_sent = pd.read_csv(csv_path)
        n_raw = len(df_sent)
        if confidence_threshold > 0.0:
            df_sent = df_sent[df_sent["sentiment_score"].abs() >= confidence_threshold].copy()
            n_after = len(df_sent)
            pct = 100 * n_after / n_raw if n_raw > 0 else 0
            print(f"  [Denoising] |score| >= {confidence_threshold:.2f}: kept {n_after}/{n_raw} ({pct:.1f}%)")
        else:
            print(f"  [Denoising] disabled -- using all {n_raw} entries")
        df_sent["publish_time"] = pd.to_datetime(df_sent["publish_time"])
        self.sentiment_data = {}
        for ticker in self.tickers:
            ticker_sent = df_sent[df_sent["ticker"] == ticker].copy()
            if not ticker_sent.empty:
                ticker_sent.set_index("publish_time", inplace=True)
                daily = ticker_sent.groupby(
                    ticker_sent.index.normalize()
                )["sentiment_score"].mean()
                self.sentiment_data[ticker] = daily
        print(f"Sentiment mapped for {len(self.sentiment_data)} tickers.")

    def build_3d_tensor(self, use_sentiment=True, use_confidence=False, train_end_date=None):
        """
        Build 3D state tensor (Time x Asset x Feature).

        Args:
            use_sentiment:   include sentiment_score as a feature.
            use_confidence:  (CGS novelty) also append |sentiment_score| as a confidence
                             proxy. High |score| = FinBERT is certain; low = uncertain/noisy.
                             PPO learns to gate on confidence implicitly.
                             Only active when use_sentiment=True.
            train_end_date:  "YYYY-MM-DD". Z-score stats computed only on the training
                             window to prevent look-ahead bias. None = full-data fit.
        """
        print(f"\nBuilding 3D tensor (use_sentiment={use_sentiment}, use_confidence={use_confidence})...")
        if train_end_date is not None:
            print(f"  Z-score fit: training period only (<= {train_end_date})")
        else:
            print("  Warning: Z-score fitted on full data -- minor look-ahead present")

        common_dates = None
        for ticker, df in self.data.items():
            common_dates = df.index if common_dates is None else common_dates.intersection(df.index)
        common_dates = common_dates.sort_values()

        train_idx = None
        if train_end_date is not None:
            train_cutoff = pd.Timestamp(train_end_date)
            train_idx = int((common_dates <= train_cutoff).sum())
            if train_idx == 0 or train_idx == len(common_dates):
                print("  Warning: train_end_date out of range, falling back to full fit")
                train_idx = None

        tensor_list = []
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            if use_sentiment:
                if self.sentiment_data is not None and ticker in self.sentiment_data:
                    sent_series = self.sentiment_data[ticker]
                    sent_aligned = sent_series.reindex(df.index.normalize(), method="ffill")
                    df["sentiment"] = sent_aligned.values
                else:
                    df["sentiment"] = 0.0
                df["sentiment"] = df["sentiment"].fillna(0.0)

                if use_confidence:
                    # CGS: |sentiment_score| as FinBERT confidence proxy.
                    # Both dimensions fed to PPO; agent learns to gate on confidence.
                    df["confidence"] = df["sentiment"].abs()
                    features = ["return", "volatility", "volume_change", "macd", "rsi",
                                "sentiment", "confidence"]
                else:
                    features = ["return", "volatility", "volume_change", "macd", "rsi", "sentiment"]
            else:
                features = ["return", "volatility", "volume_change", "macd", "rsi"]

            df = df.loc[df.index.intersection(common_dates)]
            feat_mat = df[features].values

            if train_idx is not None:
                mean = feat_mat[:train_idx].mean(axis=0)
                std  = feat_mat[:train_idx].std(axis=0)
            else:
                mean = feat_mat.mean(axis=0)
                std  = feat_mat.std(axis=0)
            std[std == 0] = 1e-8

            feat_norm = feat_mat.copy()
            feat_norm[:, 1:] = (feat_mat[:, 1:] - mean[1:]) / std[1:]  # col 0 = raw return
            tensor_list.append(feat_norm)

        panel_data = np.stack(tensor_list, axis=1)
        print(f"3D tensor: {panel_data.shape}  features={features}")
        if train_idx is not None:
            print(f"  Train/test split: {common_dates[train_idx-1].date()} | {common_dates[train_idx].date()}")
        return panel_data, common_dates