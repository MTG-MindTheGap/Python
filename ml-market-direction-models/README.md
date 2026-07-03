# ML Market-Direction Models

Two notebooks moving from ML fundamentals to a real market-prediction problem, sharing the same emphasis on rigorous train/test evaluation.

## What's inside

- **`01_linear_regression_from_scratch.ipynb`** — implements linear regression from first principles: synthetic data generation with a known ground-truth slope/intercept, a closed-form fit via the normal equation, a cross-check against `scikit-learn`, and evaluation with MSE and R². A foundations notebook for the evaluation discipline used in notebook 2.

- **`02_spy_direction_tree_ensemble.ipynb`** — predicts next-day SPY direction over a 2-year test window using Logistic Regression, a single Decision Tree, and a Random Forest, retrained monthly on an expanding window. Covers:
  - A hand-built tree split using Gini impurity, before reaching for `sklearn`
  - An explicit leakage audit (feature/label timing, walk-forward training boundaries)
  - Overfitting control via `max_depth`, `min_samples_leaf`, and cost-complexity pruning (`ccp_alpha`)
  - Random Forest with `oob_score` and `feature_importances_`
  - Full train/test comparison: confusion matrices, equity curves, hyperparameter sensitivity

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn yfinance
jupyter notebook 01_linear_regression_from_scratch.ipynb
jupyter notebook 02_spy_direction_tree_ensemble.ipynb
```

## Disclaimer

Shared for research and educational purposes only — not investment advice. Next-day direction prediction on a single, heavily-arbitraged index is an intentionally hard, low-signal problem; the value here is the leakage-safe walk-forward methodology, not a claimed trading edge.

## License

MIT — see [LICENSE](LICENSE).
