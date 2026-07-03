# Credit Default Risk Classifier

A regularised logistic regression pipeline predicting customer credit-card default on the [UCI Credit Card Default dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (30,000 customers, Taiwan, 2005), comparing an unregularised baseline against L1 (Lasso) and L2 (Ridge) penalties.

## Method

- **Robust CSV loading** — auto-detects single- vs. double-header-row CSV variants and locates the target column under any common naming alias.
- **Data cleaning** — coerces to numeric, drops invalid/duplicate rows, clips ordinal features (`PAY_*`, `EDUCATION`, `MARRIAGE`) to their documented valid ranges.
- **Leakage-safe preprocessing** — imputation and scaling are fit on the training split only, then applied to test.
- **Modelling** — baseline (no penalty), L2 Ridge, and L1 Lasso logistic regression, each with regularisation strength `C` selected via 5-fold cross-validated AUC.
- **Evaluation** — accuracy, precision, recall, F1, AUC, ROC curves, confusion matrices, predicted-probability distributions, and a coefficient-magnitude comparison across all three models.
- **Business framing** — a closing reflection on why recall (not accuracy) is the metric that matters most for default prediction, given the cost asymmetry between missed defaults and declined-but-creditworthy applicants.

Output: five diagnostic PNGs plus a multi-page PDF report (cover page with summary metrics table, one page per model, and a closing reflection page) saved to `output/`.

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn
python credit_default_classifier.py
```

`UCI_Credit_Card.csv` is included in this repo (2.9MB, the standard UCI release).

## Disclaimer

Shared for research and educational purposes only — not financial or credit-risk advice for production use.

## License

MIT — see [LICENSE](LICENSE).
