# Data

The original financial-news corpus and derived sentiment CSV files are not
distributed in this repository because of source licensing and redistribution
constraints.

The project code expects sentiment data with at least these columns:

| Column | Description |
| --- | --- |
| `ticker` | Equity ticker matching the configured universe |
| `publish_time` | Publication timestamp parseable by pandas |
| `sentiment_score` | Continuous FinBERT-derived sentiment score |

The experiment scripts expect the following project-local paths:

- HKUST FinBERT: `stock_news/finbert_hkust_sentiment.csv`
- EODHD baseline: `dow30_news_with_sentiment.csv`

Both paths are excluded from Git. Price data are downloaded separately through
`yfinance`.

To reproduce the full sentiment experiments, create the expected CSV files
locally using the schema above. Keep them at the documented paths so the
experiment scripts can discover them without code changes.
