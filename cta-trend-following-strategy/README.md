# CTA Trend-Following Strategy

A research pipeline that evaluates a transparent, rule-based CTA (Commodity Trading Advisor style) trend strategy against a machine-learning direction classifier — and a hybrid of the two — across a universe of liquid crypto assets.

The guiding question: **can a simple, auditable EMA/RSI trend rule, tuned with Bayesian optimisation, hold up against (or complement) a lightweight ML classifier, out-of-sample?**

## What's inside

`cta_trend_following_strategy.ipynb` runs an end-to-end pipeline:

1. **Data & risk diagnostics** — since-inception OHLCV pull from Yahoo Finance, return distributions, rolling realised volatility, and seasonality heatmaps per asset.
2. **CTA signal generation** — a dual-EMA + RSI trend rule with the RSI entry threshold tuned via Bayesian optimisation on a 70% training window.
3. **Backtest evaluation** — out-of-sample equity curves, drawdowns, and a full risk/return metrics table (CAGR, Sharpe, Sortino, Calmar, max drawdown) vs. buy-and-hold.
4. **ML direction predictor** — a logistic regression classifier on engineered price features (momentum, RSI, EMA ratio, realised vol) predicting next-day direction.
5. **Hybrid ensemble** — a combined signal that only takes a position when the CTA rule and the ML classifier agree.
6. **Model comparison** — Sharpe ratios across all four approaches (CTA, ML, hybrid, buy-and-hold), visualised side by side.

Every stage is built with a strict time-based train/test split and no look-ahead — the emphasis throughout is on **clarity and reproducibility over complexity**.

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn yfinance bayesian-optimization
jupyter notebook cta_trend_following_strategy.ipynb
```

Run all cells top to bottom — Part 0 downloads data and defines the asset universe and global config, and every later part depends on the outputs saved by the ones before it.

## Disclaimer

This is a research/educational project, not investment advice. Backtests use simplifying assumptions (no slippage or borrow costs, a fixed asset list) that a live strategy would need to address.

## License

MIT — see [LICENSE](LICENSE).
