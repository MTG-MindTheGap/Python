# Technical Signal Research

Two related, standalone tools for turning price-chart structure into trading signals:

1. **A technical indicator dashboard** — moving averages, rolling volatility, and a rule-based trend/vol-regime classifier across a small multi-sector ticker universe.
2. **A named chart-pattern ML framework** — detects classical technical patterns (double top/bottom, head & shoulders, triangle breakouts) from confirmed price pivots, quantifies them as features, and trains an SVM to trade only when both the model and the pattern evidence agree.

## What's inside

- **`technical_indicator_dashboard.py`** — pulls daily prices for SPY, NVDA, JPM, D, WM, KO, WMT directly from Yahoo Finance's chart API, then produces: per-ticker daily-return charts, 20/50/200-day moving average overlays with golden/death-cross shading, rolling annualised volatility with high-vol zone highlighting, a normalised price comparison, a Sharpe/volatility/action summary dashboard, a return correlation matrix, and a text report tying it all into simple trend + volatility-regime trading signals.

- **`chart_pattern_svm.py`** — detects named chart patterns (double top/bottom, head & shoulders, inverse head & shoulders, triangle breakout/breakdown) from confirmed pivot points, encodes pattern strength as ML features alongside standard technicals, and trains an SVM (RBF kernel) on a strict 70/30 time-based train/test split. Trades model probabilities only when there's corroborating pattern evidence — outputs positions, equity curves, drawdowns, and pattern-level diagnostics.

- **`chart_pattern_svm_tradingview_style.py`** — the same pipeline and model, but renders detected pattern examples as dark, single-panel, TradingView-style candlestick charts instead of the base version's line charts with pivot/EMA overlays. Useful if you want cleaner, presentation-ready pattern visualizations.

## Running it

```bash
pip install requests pandas numpy matplotlib scikit-learn yfinance

python technical_indicator_dashboard.py
python chart_pattern_svm.py
python chart_pattern_svm_tradingview_style.py
```

Each script is self-contained and saves its charts/reports to an `output/` folder next to it. Edit the `TICKERS` / `START_DATE` / `END_DATE` constants near the top of each file to point at a different universe or window.

## Disclaimer

Shared for research and educational purposes only — not investment advice. These are signal-generation experiments, not production trading systems; none of the backtests account for realistic slippage, market impact, or execution constraints beyond a flat transaction-cost assumption.

## License

MIT — see [LICENSE](LICENSE).
