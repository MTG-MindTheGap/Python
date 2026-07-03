# Dynamic Portfolio Optimizer: ML Ensemble + Black-Litterman

A quarterly-rebalanced, long-only S&P 500 equity strategy that combines a walk-forward machine-learning ensemble with Black-Litterman portfolio construction, backtested out-of-sample with full performance attribution against SPY.

```
Raw market data → Feature Engineering → ML Ensemble (Ridge + RF + XGBoost)
                → Black-Litterman → Max Sharpe Optimizer → Backtest
```

| Stage | Job | Output |
|---|---|---|
| 1 — Data | Download price + fundamental history | `daily_returns`, `daily_prices`, `fund_rank` |
| 2 — Features | Build technical + fundamental feature matrix | `feature_store` |
| 3 — ML Ensemble | Walk-forward Ridge + RF + XGBoost scoring | `ml_scores` |
| 4–5 — BL + Optimizer | Black-Litterman posterior → Max Sharpe weights | `weights_store` |
| 6 — Backtest | Out-of-sample evaluation with attribution | Tearsheet + metrics |

**Rebalancing:** Quarterly · **OOS period:** 2005 Q1 → 2024 Q4 · **Warm-up:** 2000 Q1 → 2004 Q4

## Method

- **Features:** technical (12M/6M momentum, 3M volatility, MA crossover, 52-week high ratio, beta) and fundamental (E/P, B/M, ROE, revenue growth), all cross-sectionally rank-transformed.
- **ML ensemble:** Ridge, Random Forest, and XGBoost trained walk-forward with a strict 1-quarter gap between training labels and the test quarter (no label leakage). Model weights are confidence-weighted by each model's most recent out-of-sample R².
- **Black-Litterman:** the ensemble's predicted excess returns become "views," blended with market-equilibrium returns implied by SPY market-cap weights.
- **Optimization:** Max Sharpe on BL-adjusted expected returns + Ledoit-Wolf shrinkage covariance, constrained to 8% max per stock and 30% max per sector, with a turnover penalty.
- **Attribution:** Brinson-Hood-Beebower decomposition (allocation / selection / interaction) plus tracking error, information ratio, and hit rate vs. SPY.

Known limitations (also flagged inline in the notebook): fundamental features use current yfinance snapshots rather than point-in-time historical data (look-ahead bias pre-2010), and the universe is today's S&P 500 constituent list (survivorship bias over 2000-2024).

## What's inside

- **`black_litterman_portfolio_optimizer.ipynb`** — the full pipeline described above.
- **`baseline_portfolios.py`** — a standalone companion script benchmarking four classical long-only portfolios (Equal Weight, Max Return, Global Min Variance, Max Sharpe) with the same walk-forward discipline, for comparison against the ML + BL approach.
- **`sp500_companies.csv`** — S&P 500 constituent metadata (sector, sub-industry, headquarters, date added).
- **`build_price_history.py`** — regenerates `sp500_stocks.csv`, the daily OHLCV history the notebook reads (not checked into the repo — see below).

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn xgboost yfinance

# 1. Regenerate the price history CSV (not included — see note below)
python build_price_history.py               # full 2000-2024 history, ~500 tickers
# or: python build_price_history.py --start 2015-01-01   for a faster, shorter run

# 2. Run the main notebook
jupyter notebook black_litterman_portfolio_optimizer.ipynb

# 3. (Optional) run the classical-baseline comparison
python baseline_portfolios.py
```

**Why `sp500_stocks.csv` isn't in the repo:** the full daily history for 500+ tickers back to 2000 is ~300MB, well past what's practical to check into git. `build_price_history.py` regenerates it via yfinance, or you can substitute the similarly-shaped [Kaggle "S&P 500 Stocks" dataset](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks).

## Disclaimer

This is a research/educational project, not investment advice. See the notebook's own disclaimer section and the limitations noted above before drawing any conclusions from the backtest results.

## License

MIT — see [LICENSE](LICENSE).
