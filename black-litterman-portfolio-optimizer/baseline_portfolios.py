"""
Classical baseline portfolios — Equal Weight, Max Return, Global Min Variance,
and Max Sharpe (all long-only) — walk-forward backtested on a fixed multi-asset
universe. Serves as the simple-baseline comparison point for the ML + Black-
Litterman optimizer in the main notebook: same walk-forward discipline and
performance metrics, but with classical closed-form weights instead of ML views.

Generates a one-page-per-strategy PDF report. All plots and the PDF are saved
to the output/ folder next to this script.
"""

# ---------------------------------------
# 1. Imports, configuration, data download
# ---------------------------------------

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.covariance import LedoitWolf

# ── Output directory ────────────────────────────────────────────────────────
# All plots (PNG) and the PDF report are saved here alongside this script.
# Set OUTPUT_DIR to any other path if you'd prefer to save elsewhere.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs will be saved to: {OUTPUT_DIR}\n")

def savefig(filename):
    """Save the current figure to OUTPUT_DIR only — no pop-up."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()

# ── Style ────────────────────────────────────────────────────────────────────
for _style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot", "default"):
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

# ── Universe ─────────────────────────────────────────────────────────────────
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
           "SPY", "IWM", "TLT", "GLD"]
start_date = "2015-01-01"
end_date   = None                    # None = up to latest

# Download closing prices
px   = yf.download(tickers, start=start_date, end=end_date)["Close"].dropna()
rets = px.pct_change().dropna()

print("Data shape (days x assets):", rets.shape)
print(rets.head())

# -----------------------
# Plot 1 — Normalised price history
# -----------------------
fig, ax = plt.subplots(figsize=(10, 5))
(px / px.iloc[0]).plot(ax=ax)
ax.set_title("Normalised Price History")
ax.set_ylabel("Normalised Price (start = 1.0)")
ax.grid(True)
plt.legend(title="Ticker")
plt.tight_layout()
savefig("01_price_history.png")

# ---------------------------------------
# 2. Metrics helpers
# ---------------------------------------

def annualised_sharpe(returns, rf=0.0, periods=252):
    """Annualised Sharpe ratio."""
    excess = returns - rf / periods
    return excess.mean() / excess.std() * np.sqrt(periods)

def cagr(returns, periods_per_year=252):
    """Compound Annual Growth Rate."""
    cum_growth = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return cum_growth ** (1 / years) - 1

def annualised_vol(returns, periods=252):
    """Annualised volatility."""
    return returns.std() * np.sqrt(periods)

def max_drawdown(returns):
    """Maximum drawdown."""
    equity_curve = (1 + returns).cumprod()
    peak = equity_curve.cummax()
    return ((equity_curve - peak) / peak).min()

# ---------------------------------------
# 3. Long-only weight helper
# ---------------------------------------

def long_only(w):
    """Clip negatives and renormalise; fallback to EW if all non-positive."""
    w = np.clip(w, 0, None)
    total = w.sum()
    return np.ones(len(w)) / len(w) if total <= 0 else w / total

# ---------------------------------------
# 4. Strategy functions  (all long-only)
# ---------------------------------------

def strategy_ew(train_rets):
    """Equal-weight (1/n) — the naive long-only baseline."""
    n = train_rets.shape[1]
    return np.ones(n) / n

def strategy_mrp(train_rets):
    """Maximum Return Portfolio — 100 % in highest-mean asset."""
    mu_hat = train_rets.mean()
    w = np.zeros(train_rets.shape[1])
    w[list(train_rets.columns).index(mu_hat.idxmax())] = 1.0
    return w

def strategy_gmvp(train_rets, use_shrinkage=True):
    """Global Minimum Variance Portfolio with Ledoit-Wolf shrinkage."""
    Sigma = LedoitWolf().fit(train_rets.values).covariance_ if use_shrinkage \
            else np.cov(train_rets.values, rowvar=False)
    ones = np.ones(Sigma.shape[0])
    w = np.linalg.solve(Sigma, ones)
    return long_only(w / w.sum())

def strategy_msrp(train_rets, rf=0.02, use_shrinkage=True):
    """Maximum Sharpe Ratio Portfolio — closed-form, long-only."""
    mu_hat = train_rets.mean() * 252
    Sigma = LedoitWolf().fit(train_rets.values).covariance_ if use_shrinkage \
            else np.cov(train_rets.values, rowvar=False)
    excess = mu_hat.values - rf * np.ones(Sigma.shape[0])
    w = np.linalg.solve(Sigma, excess)
    return long_only(w / w.sum())

# -----------------------
# Plot 2 — Sample weights from the last training window
# -----------------------

def plot_sample_weights(train_rets, title_prefix="Sample Weights", save_name=None):
    _strategies = {
        "EW":   strategy_ew,
        "MRP":  strategy_mrp,
        "GMVP": lambda r: strategy_gmvp(r, use_shrinkage=True),
        "MSRP": lambda r: strategy_msrp(r, rf=0.02, use_shrinkage=True),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    axes = axes.flatten()
    for ax, (name, fn) in zip(axes, _strategies.items()):
        w = fn(train_rets)
        ax.bar(train_rets.columns, w)
        ax.set_title(f"{title_prefix} – {name}")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y")
    plt.tight_layout()
    savefig(save_name or "02_sample_weights.png")

train_window_example = rets.iloc[-252:]
plot_sample_weights(train_window_example, title_prefix="Last Window Weights",
                    save_name="02_sample_weights.png")

# ---------------------------------------
# 5. Walk-forward backtest
# ---------------------------------------

def walk_forward(rets, strategy_fn, window=252, rebal=21, rf=0.0, tcost_bps=10):
    """Walk-forward backtest; returns a Series of daily portfolio returns."""
    tcost  = tcost_bps / 10_000
    pnl    = []
    w_prev = np.zeros(rets.shape[1])
    dates  = rets.index

    for t in range(window, len(rets), rebal):
        train = rets.iloc[t - window: t]
        w = strategy_fn(train)

        cost = tcost * np.abs(w - w_prev).sum()
        period_pnl = rets.iloc[t: t + rebal] @ w

        if len(period_pnl) > 0:
            period_pnl.iloc[0] -= cost

        pnl.extend(period_pnl.values)
        w_prev = w

    return pd.Series(pnl, index=dates[window: window + len(pnl)])

# -----------------------
# Run all four strategies
# -----------------------

strategies = {
    "EW":   lambda r: strategy_ew(r),
    "MRP":  lambda r: strategy_mrp(r),
    "GMVP": lambda r: strategy_gmvp(r, use_shrinkage=True),
    "MSRP": lambda r: strategy_msrp(r, rf=0.02, use_shrinkage=True),
}

WINDOW    = 252
REBAL     = 21
TCOST_BPS = 10
RF        = 0.02

pnl_dict = {}
for name, fn in strategies.items():
    pnl_dict[name] = walk_forward(rets, fn, window=WINDOW, rebal=REBAL,
                                  tcost_bps=TCOST_BPS)

# -----------------------
# Plot 3 — Overview equity curves (all strategies)
# -----------------------
fig, ax = plt.subplots(figsize=(10, 5))
for name, pnl in pnl_dict.items():
    (1 + pnl).cumprod().plot(ax=ax, label=name)
ax.set_title("Walk-Forward Equity Curves — All Strategies")
ax.set_ylabel("Portfolio Value (start = 1.0)")
ax.legend()
ax.grid(True)
plt.tight_layout()
savefig("03_equity_curves_all.png")

# -----------------------
# Plot 4 — Metric bar charts
# -----------------------

summary_rows = []
for name, pnl in pnl_dict.items():
    summary_rows.append({
        "Strategy": name,
        "CAGR":    cagr(pnl),
        "Vol_Ann": annualised_vol(pnl),
        "Sharpe":  annualised_sharpe(pnl, rf=RF),
        "MaxDD":   max_drawdown(pnl),
    })
summary_df = pd.DataFrame(summary_rows).set_index("Strategy")

print("\n=== Performance Summary ===")
print(summary_df.to_string(float_format="{:.4f}".format))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
summary_df["CAGR"].plot(kind="bar", ax=axes[0])
axes[0].set_title("CAGR by Strategy"); axes[0].set_ylabel("CAGR"); axes[0].grid(True, axis="y")
summary_df["Vol_Ann"].plot(kind="bar", ax=axes[1])
axes[1].set_title("Annualised Volatility"); axes[1].set_ylabel("Volatility"); axes[1].grid(True, axis="y")
summary_df["Sharpe"].plot(kind="bar", ax=axes[2])
axes[2].set_title("Sharpe Ratio"); axes[2].set_ylabel("Sharpe"); axes[2].grid(True, axis="y")
plt.tight_layout()
savefig("04_metric_bars.png")

# ---------------------------------------
# 6. PDF Report  —  one page per strategy
# ---------------------------------------

STRATEGY_DESCRIPTIONS = {
    "EW":   "Equal Weight (1/N): allocates the same weight to every asset regardless of\n"
            "expected returns or covariances. A robust, cost-efficient baseline.",
    "MRP":  "Maximum Return Portfolio: concentrates 100 % of capital in the single asset\n"
            "with the highest sample mean return over the training window.",
    "GMVP": "Global Minimum Variance Portfolio: minimises portfolio variance using a\n"
            "Ledoit-Wolf shrunk covariance matrix. Long-only constrained.",
    "MSRP": "Maximum Sharpe Ratio Portfolio: closed-form, targeting the highest\n"
            "risk-adjusted return. Long-only constrained (rf = 2 %).",
}

COLORS = {
    "EW":   "#2196F3",
    "MRP":  "#FF5722",
    "GMVP": "#4CAF50",
    "MSRP": "#9C27B0",
}

pdf_path = os.path.join(OUTPUT_DIR, "strategy_report.pdf")

with PdfPages(pdf_path) as pdf:

    # ── Cover page ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax_cover = fig.add_axes([0, 0, 1, 1])
    ax_cover.set_axis_off()
    ax_cover.text(0.5, 0.70, "Portfolio Strategy Report",
                  ha="center", va="center", fontsize=28, fontweight="bold",
                  color="white", transform=ax_cover.transAxes)
    ax_cover.text(0.5, 0.60, "Classical Long-Only Baseline Strategies",
                  ha="center", va="center", fontsize=16, color="#aaaacc",
                  transform=ax_cover.transAxes)
    ax_cover.text(0.5, 0.53,
                  f"Universe: {', '.join(tickers)}  |  "
                  f"Period: {str(rets.index[0].date())} → {str(rets.index[-1].date())}",
                  ha="center", va="center", fontsize=12, color="#cccccc",
                  transform=ax_cover.transAxes)
    ax_cover.text(0.5, 0.46,
                  f"Train window: {WINDOW} days  |  Rebal: {REBAL} days  |  "
                  f"T-cost: {TCOST_BPS} bps  |  Long-only",
                  ha="center", va="center", fontsize=11, color="#cccccc",
                  transform=ax_cover.transAxes)
    col_labels = ["CAGR", "Vol (Ann)", "Sharpe", "Max DD"]
    cell_text  = [[f"{v['CAGR']:.2%}", f"{v['Vol_Ann']:.2%}",
                   f"{v['Sharpe']:.2f}", f"{v['MaxDD']:.2%}"]
                  for v in summary_df.to_dict("index").values()]
    tbl = ax_cover.table(cellText=cell_text, rowLabels=list(summary_df.index),
                         colLabels=col_labels, loc="center",
                         bbox=[0.15, 0.10, 0.70, 0.28])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#16213e" if r % 2 == 0 else "#0f3460")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333355")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── One page per strategy ────────────────────────────────────────────────
    for name, pnl in pnl_dict.items():
        color  = COLORS[name]
        equity = (1 + pnl).cumprod()
        dd_series = (equity - equity.cummax()) / equity.cummax()

        m_cagr   = cagr(pnl)
        m_vol    = annualised_vol(pnl)
        m_sharpe = annualised_sharpe(pnl, rf=RF)
        m_mdd    = max_drawdown(pnl)

        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#f7f9fc")
        gs = gridspec.GridSpec(3, 3, figure=fig,
                               top=0.80, bottom=0.08,
                               left=0.08, right=0.96,
                               hspace=0.55, wspace=0.35)

        fig.text(0.5, 0.94, f"Strategy: {name}",
                 ha="center", va="center", fontsize=20,
                 fontweight="bold", color="#1a1a2e")
        fig.text(0.5, 0.905, STRATEGY_DESCRIPTIONS[name],
                 ha="center", va="center", fontsize=9,
                 color="#555555", style="italic")

        # Equity curve
        ax_eq = fig.add_subplot(gs[0:2, 0:2])
        ax_eq.plot(equity.index, equity.values, color=color, linewidth=1.6)
        ax_eq.fill_between(equity.index, 1, equity.values, alpha=0.15, color=color)
        ax_eq.axhline(1, color="#999999", linewidth=0.8, linestyle="--")
        ax_eq.set_title("Equity Curve", fontsize=11, fontweight="bold")
        ax_eq.set_ylabel("Portfolio Value (start = 1.0)")
        ax_eq.grid(True, alpha=0.4)

        # Drawdown
        ax_dd = fig.add_subplot(gs[2, 0:2])
        ax_dd.fill_between(dd_series.index, dd_series.values, 0,
                           color="#e74c3c", alpha=0.6)
        ax_dd.plot(dd_series.index, dd_series.values, color="#c0392b", linewidth=0.8)
        ax_dd.set_title("Drawdown", fontsize=11, fontweight="bold")
        ax_dd.set_ylabel("Drawdown")
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax_dd.grid(True, alpha=0.4)

        # Metrics panel
        ax_m = fig.add_subplot(gs[0:3, 2])
        ax_m.set_axis_off()
        metrics = [
            ("CAGR",     f"{m_cagr:.2%}"),
            ("Ann. Vol", f"{m_vol:.2%}"),
            ("Sharpe",   f"{m_sharpe:.2f}"),
            ("Max DD",   f"{m_mdd:.2%}"),
            ("Period",   f"{str(pnl.index[0].date())}\n→ {str(pnl.index[-1].date())}"),
            ("# Days",   f"{len(pnl):,}"),
            ("Universe", ", ".join(tickers)),
        ]
        y_pos = 0.95
        ax_m.text(0.5, y_pos, "Key Metrics", ha="center", va="top",
                  fontsize=12, fontweight="bold", color="#1a1a2e",
                  transform=ax_m.transAxes)
        y_pos -= 0.07
        for label, value in metrics:
            ax_m.text(0.08, y_pos, label, ha="left", va="top",
                      fontsize=9, color="#777777", transform=ax_m.transAxes)
            y_pos -= 0.055
            ax_m.text(0.08, y_pos, value, ha="left", va="top",
                      fontsize=14 if "\n" not in value else 10,
                      fontweight="bold", color=color,
                      transform=ax_m.transAxes)
            y_pos -= 0.09
            ax_m.plot([0.05, 0.95], [y_pos + 0.02, y_pos + 0.02],
                      color="#dddddd", linewidth=0.6,
                      transform=ax_m.transAxes, clip_on=False)
            y_pos -= 0.01

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"\nAll done! Files saved to: {OUTPUT_DIR}")
print("  01_price_history.png")
print("  02_sample_weights.png")
print("  03_equity_curves_all.png")
print("  04_metric_bars.png")
print("  strategy_report.pdf")