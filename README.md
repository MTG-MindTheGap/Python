# Python

A collection of Python quant research projects and tools — portfolio
construction, risk classification, trend/direction strategies, technical
signal research, and a standalone insider-trading dashboard app. Each
project is self-contained in its own folder with its own README, license,
and run instructions.

## Portfolio Construction
| Project | What it does |
|---|---|
| [`black-litterman-portfolio-optimizer`](black-litterman-portfolio-optimizer/) | Quarterly-rebalanced, long-only S&P 500 strategy combining a walk-forward ML ensemble (Ridge + Random Forest + XGBoost) with Black-Litterman portfolio construction, backtested 2005–2024 with full performance attribution against SPY. Includes a companion script benchmarking four classical portfolios (Equal Weight, Max Return, Global Min Variance, Max Sharpe) for comparison. |

## Risk & Classification
| Project | What it does |
|---|---|
| [`credit-default-risk-classifier`](credit-default-risk-classifier/) | Regularised logistic regression pipeline predicting credit-card default on the UCI Credit Card Default dataset (30,000 customers), comparing an unregularised baseline against L1 (Lasso) and L2 (Ridge) penalties, with leakage-safe preprocessing and a full diagnostic report (ROC, confusion matrices, coefficient comparison). |

## Trend & Direction Strategies
| Project | What it does |
|---|---|
| [`cta-trend-following-strategy`](cta-trend-following-strategy/) | Compares a rule-based EMA/RSI trend-following strategy (tuned via Bayesian optimisation) against an ML direction classifier and a hybrid of the two, across a universe of liquid crypto assets, with full out-of-sample risk/return metrics. |
| [`ml-market-direction-models`](ml-market-direction-models/) | Two notebooks moving from ML fundamentals to a real prediction problem: linear regression built from first principles, then next-day SPY direction prediction via Logistic Regression, a hand-built Decision Tree, and a Random Forest — all retrained monthly on an expanding, leakage-audited walk-forward window. |

## Technical Signal Research
| Project | What it does |
|---|---|
| [`technical-signal-research`](technical-signal-research/) | A technical indicator dashboard (moving averages, rolling volatility, trend/vol-regime classification across a multi-sector ticker universe) plus a named chart-pattern ML framework that detects classical patterns (double top/bottom, head & shoulders, triangle breakouts) and trades an SVM only when the model and pattern evidence agree. |

## Apps & Tools
| Project | What it does |
|---|---|
| [`InsiderFlow App`](InsiderFlow%20App/) | A local dashboard tracking corporate insider buy/sell activity from SEC EDGAR Form 4 filings, overlaid on a price chart via Yahoo Finance, with a local chat assistant (via Ollama) for querying loaded insider transactions. No API keys or cloud services required. |

## Running any project
Each folder is independent — `cd` into it and follow its own README for
setup and run instructions. Dependencies, datasets, and disclaimers are
documented per project rather than globally, since each has different
requirements.

## Disclaimer
Everything here is shared for research and educational purposes only —
none of it is financial advice. See each project's own README for
project-specific caveats and known limitations.
