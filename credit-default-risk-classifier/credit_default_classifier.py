# -*- coding: utf-8 -*-
"""
Credit Default Risk Classifier

Predicts customer credit card default using Logistic Regression (baseline,
L1 Lasso, L2 Ridge), with 5-fold cross-validated regularisation strength
selection and full ROC/confusion-matrix/coefficient diagnostics.

Dataset: UCI Credit Card Default — reads from local CSV file.
All plots and PDF report saved to output/ folder next to this script.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports & paths
# ─────────────────────────────────────────────────────────────────────────────
import os
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "UCI_Credit_Card.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Reading CSV  : {CSV_PATH}")
print(f"Saving to    : {OUTPUT_DIR}\n")

def savefig(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close()

for _style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot", "default"):
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load CSV — auto-detect single vs double header row
# ─────────────────────────────────────────────────────────────────────────────
print("Loading dataset...")

_peek     = pd.read_csv(CSV_PATH, nrows=2, header=None)
_REAL     = {"LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2"}
_row0     = {str(v).strip().upper() for v in _peek.iloc[0]}
_row1     = {str(v).strip().upper() for v in _peek.iloc[1]}

if len(_REAL & _row0) >= 3:
    df = pd.read_csv(CSV_PATH, header=0)
    print("  Single-header CSV detected — using header=0")
elif len(_REAL & _row1) >= 3:
    df = pd.read_csv(CSV_PATH, header=1)
    print("  Two-row header CSV detected — using header=1")
else:
    df = pd.read_csv(CSV_PATH, header=0)
    print("  Defaulting to header=0")

# Normalise column names: strip, uppercase, spaces+dots -> underscore
df.columns = (df.columns
               .str.strip()
               .str.upper()
               .str.replace(" ", "_", regex=False)
               .str.replace(".", "_", regex=False))

print(f"Raw shape    : {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Locate target column (handles any naming variant in the wild)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_ALIASES = [
    "DEFAULT_PAYMENT_NEXT_MONTH",
    "DEFAULT",
    "Y",
]
target_col = next(
    (a.replace(" ", "_").replace(".", "_").upper()
     for a in TARGET_ALIASES
     if a.replace(" ", "_").replace(".", "_").upper() in df.columns),
    None
)
if target_col is None:
    raise KeyError(
        f"Cannot find target column. Available columns:\n{df.columns.tolist()}"
    )

df = df.rename(columns={target_col: "DEFAULT"})
df = df.drop(columns=["ID"], errors="ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Data cleaning
# ─────────────────────────────────────────────────────────────────────────────

# 4a. Force all columns to numeric (coerces stray strings to NaN)
df = df.apply(pd.to_numeric, errors="coerce")

# 4b. Keep only rows where target is exactly 0 or 1
before = len(df)
df = df[df["DEFAULT"].isin([0.0, 1.0])].copy()
print(f"Dropped {before - len(df):,} rows with invalid/missing target")

# 4c. Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df):,} duplicate rows")

# 4d. Clip ordinal features to documented valid ranges
pay_status_cols = [c for c in df.columns if c.startswith("PAY_") and "AMT" not in c]
df[pay_status_cols] = df[pay_status_cols].clip(-2, 9)   # PAY_0…PAY_6: -2 to 9
if "EDUCATION" in df.columns:
    df["EDUCATION"] = df["EDUCATION"].clip(1, 6)        # 1-4 documented; 5-6 kept
if "MARRIAGE" in df.columns:
    df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)       # 0 undocumented -> 3 (other)

print(f"\nClean shape  : {df.shape}")
print(df.describe().T[["mean", "min", "max"]].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature / target split, train-test split, impute, scale
# ─────────────────────────────────────────────────────────────────────────────
X = df.drop(columns=["DEFAULT"])
y = df["DEFAULT"].astype(int)

print(f"\nClass balance:\n{y.value_counts().sort_index().to_string()}")
print(f"Default rate : {y.mean():.2%}")
assert y.sum() > 0,     "No default=1 rows — check CSV"
assert y.mean() < 0.99, "Almost all rows are default=1 — target may be inverted"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute THEN scale — fit only on train to prevent data leakage
imputer    = SimpleImputer(strategy="mean")
X_train_im = imputer.fit_transform(X_train)
X_test_im  = imputer.transform(X_test)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_im)
X_test_sc  = scaler.transform(X_test_im)

print(f"\nTrain size   : {X_train_sc.shape[0]:,}  |  Test size: {X_test_sc.shape[0]:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Modelling
# ─────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dense enough to actually resolve where regularisation stops costing AUC —
# the previous grid [0.001, 0.01, 0.1, 1, 10, 100] was too coarse: CV AUC
# rises monotonically across it with no interior peak, so picking argmax
# always lands on C=100 (the weakest regularisation tested) for both L1 and
# L2. That made "L1 Lasso" converge to the same dense, near-unregularised
# solution as the baseline — zero coefficients were ever actually zeroed
# out, contradicting the whole point of comparing L1 vs L2 vs baseline.
C_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# sklearn <1.2 requires penalty="none" (string); >=1.2 accepts penalty=None
_sk_ver       = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
_penalty_none = None if _sk_ver >= (1, 2) else "none"

# 6a. Baseline — no regularisation
baseline = LogisticRegression(penalty=_penalty_none, solver="lbfgs",
                               max_iter=2000, random_state=42)
baseline.fit(X_train_sc, y_train)


def select_C_one_se_rule(C_grid, fold_scores):
    """
    Pick the smallest C (strongest regularisation) whose mean CV AUC is
    within one standard error of the best mean CV AUC across the grid.

    Plain argmax over a monotonically-improving AUC curve always selects
    the least-regularised model in the grid, which defeats the purpose of
    regularisation (and, for L1, feature selection): if AUC is still
    climbing at the edge of the grid there's no interior optimum to find,
    and looser regularisation always looks "best" even when the difference
    is noise. The one-standard-error rule (standard practice for Lasso
    path selection, e.g. glmnet's lambda.1se) instead prefers the simplest
    model that is statistically indistinguishable from the best one.
    """
    means = np.array([s.mean() for s in fold_scores])
    ses   = np.array([s.std(ddof=1) / np.sqrt(len(s)) for s in fold_scores])
    best_idx = np.argmax(means)
    threshold = means[best_idx] - ses[best_idx]
    candidates = [i for i, m in enumerate(means) if m >= threshold]
    return C_grid[min(candidates)]  # smallest C among statistically-tied candidates


# 6b. L2 Ridge — grid search over C, select via one-SE rule
l2_fold_scores = [
    cross_val_score(LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                        max_iter=2000, random_state=42),
                     X_train_sc, y_train, cv=cv, scoring="roc_auc")
    for C in C_grid
]
l2_cv_scores = [s.mean() for s in l2_fold_scores]
best_C_l2 = select_C_one_se_rule(C_grid, l2_fold_scores)
l2_model  = LogisticRegression(penalty="l2", C=best_C_l2, solver="lbfgs",
                                 max_iter=2000, random_state=42)
l2_model.fit(X_train_sc, y_train)

# 6c. L1 Lasso — grid search over C, select via one-SE rule
l1_fold_scores = [
    cross_val_score(LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                        max_iter=2000, random_state=42),
                     X_train_sc, y_train, cv=cv, scoring="roc_auc")
    for C in C_grid
]
l1_cv_scores = [s.mean() for s in l1_fold_scores]
best_C_l1 = select_C_one_se_rule(C_grid, l1_fold_scores)
l1_model  = LogisticRegression(penalty="l1", C=best_C_l1, solver="liblinear",
                                 max_iter=2000, random_state=42)
l1_model.fit(X_train_sc, y_train)

n_zeroed = int(np.sum(l1_model.coef_[0] == 0))
print(f"\nBest C (one-SE rule) — L2 (Ridge): {best_C_l2}  |  L1 (Lasso): {best_C_l1}")
print(f"L1 Lasso zeroed out {n_zeroed}/{X.shape[1]} features")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "Baseline (no reg)": baseline,
    "L2 Ridge":          l2_model,
    "L1 Lasso":          l1_model,
}
COLORS = {
    "Baseline (no reg)": "#2196F3",
    "L2 Ridge":          "#4CAF50",
    "L1 Lasso":          "#FF5722",
}

def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    return {
        "Accuracy":  accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall":    recall_score(y, pred, zero_division=0),
        "F1":        f1_score(y, pred, zero_division=0),
        "AUC":       roc_auc_score(y, proba),
        "proba":     proba,
        "pred":      pred,
    }

results = {name: evaluate(m, X_test_sc, y_test) for name, m in MODELS.items()}

print("\n=== Test Set Performance ===")
summary_df = pd.DataFrame(
    {n: {k: v for k, v in r.items() if k not in ("proba", "pred")}
     for n, r in results.items()}
).T
print(summary_df.to_string(float_format="{:.4f}".format))

# ─────────────────────────────────────────────────────────────────────────────
# 8. Plots
# ─────────────────────────────────────────────────────────────────────────────

# Plot 1 — Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
y.value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4CAF50", "#FF5722"])
ax.set_xticklabels(["No Default (0)", "Default (1)"], rotation=0)
ax.set_title("Class Distribution"); ax.set_ylabel("Count"); ax.grid(True, axis="y")
plt.tight_layout(); savefig("01_class_distribution.png")

# Plot 2 — CV regularisation curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(C_grid, l2_cv_scores, marker="o", color="#4CAF50")
axes[0].axvline(best_C_l2, color="#4CAF50", linestyle="--", alpha=0.6)
axes[0].set_xscale("log"); axes[0].set_title("L2 Ridge — CV AUC vs C")
axes[0].set_xlabel("C"); axes[0].set_ylabel("CV AUC"); axes[0].grid(True)
axes[1].plot(C_grid, l1_cv_scores, marker="o", color="#FF5722")
axes[1].axvline(best_C_l1, color="#FF5722", linestyle="--", alpha=0.6)
axes[1].set_xscale("log"); axes[1].set_title("L1 Lasso — CV AUC vs C")
axes[1].set_xlabel("C"); axes[1].set_ylabel("CV AUC"); axes[1].grid(True)
plt.suptitle("Cross-Validation Regularisation Strength Selection",
             fontsize=12, fontweight="bold")
plt.tight_layout(); savefig("02_cv_regularisation.png")

# Plot 3 — ROC curves
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    ax.plot(fpr, tpr, label=f"{name}  (AUC={res['AUC']:.3f})",
            color=COLORS[name], linewidth=1.8)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
ax.set_title("ROC Curves — All Models"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout(); savefig("03_roc_curves.png")

# Plot 4 — Metric bar chart
fig, axes = plt.subplots(1, 4, figsize=(13, 4))
for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
    vals = [results[n][metric] for n in MODELS]
    axes[i].bar(list(MODELS.keys()), vals, color=list(COLORS.values()))
    axes[i].set_title(metric); axes[i].set_ylim(0, 1)
    axes[i].set_xticklabels(list(MODELS.keys()), rotation=20, ha="right")
    axes[i].grid(True, axis="y")
plt.suptitle("Evaluation Metrics at Threshold = 0.5", fontsize=12, fontweight="bold")
plt.tight_layout(); savefig("04_metric_bars.png")

# Plot 5 — Coefficient heatmap
top_n    = 15
coefs_l1 = pd.Series(l1_model.coef_[0], index=X.columns)
top_feat = coefs_l1.abs().nlargest(top_n).index
coef_df  = pd.DataFrame({
    "Baseline": baseline.coef_[0],
    "L2 Ridge": l2_model.coef_[0],
    "L1 Lasso": l1_model.coef_[0],
}, index=X.columns).loc[top_feat]

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(coef_df.T.values, aspect="auto", cmap="RdBu_r",
               vmin=-coef_df.abs().values.max(), vmax=coef_df.abs().values.max())
ax.set_xticks(range(top_n)); ax.set_xticklabels(top_feat, rotation=45, ha="right")
ax.set_yticks(range(3));     ax.set_yticklabels(coef_df.columns)
plt.colorbar(im, ax=ax, label="Coefficient value")
ax.set_title(f"Top {top_n} Features by |L1 Coefficient|",
             fontsize=12, fontweight="bold")
plt.tight_layout(); savefig("05_coefficient_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. PDF Report
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DESCRIPTIONS = {
    "Baseline (no reg)": "Standard logistic regression with no regularisation penalty.\n"
                          "Provides the reference benchmark; prone to overfitting on noisy features.",
    "L2 Ridge":          f"Logistic regression with L2 (Ridge) penalty; best C={best_C_l2} chosen by 5-fold CV.\n"
                          "Shrinks all coefficients toward zero but retains all features.",
    "L1 Lasso":          f"Logistic regression with L1 (Lasso) penalty; best C={best_C_l1} chosen by 5-fold CV.\n"
                          "Performs automatic feature selection by zeroing out irrelevant coefficients.",
}

pdf_path = os.path.join(OUTPUT_DIR, "credit_default_report.pdf")

with PdfPages(pdf_path) as pdf:

    # Cover page
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax_c = fig.add_axes([0, 0, 1, 1]); ax_c.set_axis_off()
    ax_c.text(0.5, 0.74, "Credit Default Prediction Report",
              ha="center", fontsize=26, fontweight="bold", color="white",
              transform=ax_c.transAxes)
    ax_c.text(0.5, 0.64, "Regularised Logistic Regression — Baseline vs. L1 vs. L2",
              ha="center", fontsize=15, color="#aaaacc", transform=ax_c.transAxes)
    ax_c.text(0.5, 0.57,
              f"Dataset: UCI Credit Card Default  |  n={len(df):,} observations  |  "
              f"Default rate: {y.mean():.1%}",
              ha="center", fontsize=11, color="#cccccc", transform=ax_c.transAxes)
    ax_c.text(0.5, 0.51,
              "Models: Baseline LR  |  L2 Ridge  |  L1 Lasso  |  Threshold: 0.5",
              ha="center", fontsize=11, color="#cccccc", transform=ax_c.transAxes)
    col_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    cell_text  = [[f"{results[n]['Accuracy']:.3f}", f"{results[n]['Precision']:.3f}",
                   f"{results[n]['Recall']:.3f}",   f"{results[n]['F1']:.3f}",
                   f"{results[n]['AUC']:.3f}"] for n in MODELS]
    tbl = ax_c.table(cellText=cell_text, rowLabels=list(MODELS.keys()),
                     colLabels=col_labels, loc="center",
                     bbox=[0.10, 0.08, 0.80, 0.32])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#16213e" if r % 2 == 0 else "#0f3460")
        cell.set_text_props(color="white"); cell.set_edgecolor("#333355")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # One page per model
    for name, res in results.items():
        color = COLORS[name]; proba = res["proba"]; pred = res["pred"]
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#f7f9fc")
        gs = gridspec.GridSpec(2, 3, figure=fig, top=0.78, bottom=0.10,
                               left=0.08, right=0.96, hspace=0.50, wspace=0.40)
        fig.text(0.5, 0.92, f"Model: {name}",
                 ha="center", fontsize=20, fontweight="bold", color="#1a1a2e")
        fig.text(0.5, 0.875, MODEL_DESCRIPTIONS[name],
                 ha="center", fontsize=9, color="#555555", style="italic")

        ax_roc = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, color=color, linewidth=1.8)
        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax_roc.fill_between(fpr, tpr, alpha=0.10, color=color)
        ax_roc.set_title("ROC Curve", fontsize=10, fontweight="bold")
        ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.grid(True, alpha=0.4)
        ax_roc.text(0.60, 0.10, f"AUC = {res['AUC']:.3f}",
                    fontsize=10, fontweight="bold", color=color,
                    transform=ax_roc.transAxes)

        ax_cm = fig.add_subplot(gs[0, 1])
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(
            ax=ax_cm, colorbar=False, cmap="Blues")
        ax_cm.set_title("Confusion Matrix", fontsize=10, fontweight="bold")

        ax_hist = fig.add_subplot(gs[0, 2])
        ax_hist.hist(proba[y_test == 0], bins=40, alpha=0.6,
                     color="#4CAF50", label="No Default", density=True)
        ax_hist.hist(proba[y_test == 1], bins=40, alpha=0.6,
                     color="#FF5722", label="Default", density=True)
        ax_hist.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax_hist.set_title("Predicted Probabilities", fontsize=10, fontweight="bold")
        ax_hist.set_xlabel("P(Default)"); ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.4)

        ax_m = fig.add_subplot(gs[1, 0]); ax_m.set_axis_off()
        metrics_list = [("Accuracy",  f"{res['Accuracy']:.3f}"),
                        ("Precision", f"{res['Precision']:.3f}"),
                        ("Recall",    f"{res['Recall']:.3f}"),
                        ("F1 Score",  f"{res['F1']:.3f}"),
                        ("AUC",       f"{res['AUC']:.3f}")]
        y_pos = 0.95
        ax_m.text(0.5, y_pos, "Metrics @ t=0.5", ha="center", va="top",
                  fontsize=11, fontweight="bold", color="#1a1a2e",
                  transform=ax_m.transAxes); y_pos -= 0.08
        for lbl, val in metrics_list:
            ax_m.text(0.08, y_pos, lbl, ha="left", va="top", fontsize=9,
                      color="#777777", transform=ax_m.transAxes); y_pos -= 0.10
            ax_m.text(0.08, y_pos, val, ha="left", va="top", fontsize=15,
                      fontweight="bold", color=color,
                      transform=ax_m.transAxes); y_pos -= 0.10
            ax_m.plot([0.05, 0.95], [y_pos + 0.04, y_pos + 0.04], color="#dddddd",
                      linewidth=0.6, transform=ax_m.transAxes, clip_on=False)

        ax_coef = fig.add_subplot(gs[1, 1:])
        coefs = pd.Series(MODELS[name].coef_[0], index=X.columns)
        top   = coefs.abs().nlargest(10)
        vals  = coefs[top.index]
        bar_colors = [color if v > 0 else "#aaaaaa" for v in vals]
        ax_coef.barh(top.index[::-1], vals[top.index[::-1]], color=bar_colors[::-1])
        ax_coef.axvline(0, color="black", linewidth=0.8)
        ax_coef.set_title("Top 10 Feature Coefficients", fontsize=10, fontweight="bold")
        ax_coef.set_xlabel("Coefficient value"); ax_coef.grid(True, axis="x", alpha=0.4)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Reflection page
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax_r = fig.add_axes([0.08, 0.10, 0.84, 0.80]); ax_r.set_axis_off()
    ax_r.text(0.5, 0.95, "Which Metric Matters Most — and Why?",
              ha="center", fontsize=18, fontweight="bold", color="white",
              transform=ax_r.transAxes)
    reflection = (
        "In credit default prediction, Recall (sensitivity) is the most critical metric.\n\n"
        "A missed default (false negative) means the lender extends credit to a customer who\n"
        "subsequently defaults — resulting in a direct financial loss. In contrast, a false positive\n"
        "(flagging a good customer as risky) leads to a declined application — an opportunity cost,\n"
        "but not a capital loss.\n\n"
        "The asymmetry of consequences makes Recall the primary optimisation target:\n"
        "  * False Negative cost  ->  full loan loss (e.g. $10,000-$50,000+)\n"
        "  * False Positive cost  ->  lost interest income on one customer\n\n"
        "However, optimising Recall in isolation risks flagging almost everyone as a defaulter,\n"
        "collapsing Precision and making the model commercially useless.\n\n"
        "Recommendation: use F1 Score as the balancing metric, and consider lowering the\n"
        "classification threshold below 0.5 to improve Recall at an acceptable Precision cost.\n"
        "AUC-ROC provides a threshold-independent view of overall discriminative power."
    )
    ax_r.text(0.05, 0.82, reflection, ha="left", va="top", fontsize=11,
              color="#dddddd", transform=ax_r.transAxes, linespacing=1.7)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

print(f"\nAll done! Files saved to: {OUTPUT_DIR}")
print("  01_class_distribution.png")
print("  02_cv_regularisation.png")
print("  03_roc_curves.png")
print("  04_metric_bars.png")
print("  05_coefficient_heatmap.png")
print("  credit_default_report.pdf")