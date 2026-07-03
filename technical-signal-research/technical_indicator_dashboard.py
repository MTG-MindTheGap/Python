"""
Technical Indicator Dashboard — moving averages, rolling volatility, and a
rule-based trend/vol-regime trading signal, computed per ticker across a
small multi-sector universe.

No yfinance needed — pulls directly from Yahoo Finance's public chart API
using requests.
pip install requests pandas matplotlib numpy
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# OUTPUT FOLDER SETUP
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
OUTPUT_PREFIX = "signal"
OUTPUT_DIR    = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
TICKERS      = ["SPY", "NVDA", "JPM", "D", "WM", "KO", "WMT",]
START_DATE   = "2021-04-23"
END_DATE     = "2026-04-23"
SHORT_WINDOW = 20    # ~1 month
MID_WINDOW   = 50    # ~2.5 months
LONG_WINDOW  = 200   # ~10 months
VOL_WINDOW   = 20    # rolling volatility window
COLORS       = {"SPY": "#1f77b4", "NVDA": "#ff7f0e", "JPM": "#2ca02c",
                "D": "#d62728", "WM": "#9467bd", "KO": "#8c564b", "WMT": "#e377c2"}
YAHOO_URL    = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# ─────────────────────────────────────────────────────────────────────
# TASK 1 — DOWNLOAD DATA
# ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 1: Downloading data from Yahoo Finance...")
print("=" * 60)

def fetch_yahoo(ticker):
    url  = YAHOO_URL.format(ticker=ticker)
    resp = requests.get(
        url,
        params={"interval": "1d", "range": "5y"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    resp.raise_for_status()
    data   = resp.json()
    result = data["chart"]["result"][0]
    ts     = result["timestamp"]
    ohlcv  = result["indicators"]["quote"][0]

    dates  = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts]
    closes = ohlcv.get("close", [])

    # Remove rows where close is None
    cleaned = [(d, c) for d, c in zip(dates, closes) if c is not None]
    df = pd.DataFrame(cleaned, columns=["date", ticker])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

close_prices = pd.DataFrame()
for ticker in TICKERS:
    print(f"  Fetching {ticker}...", end=" ")
    df = fetch_yahoo(ticker)
    close_prices = df if close_prices.empty else close_prices.join(df, how="outer")
    print(f"✓  {len(df)} rows")

print(f"\nDate range : {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
print(f"Shape      : {close_prices.shape}")

# ─────────────────────────────────────────────────────────────────────
# TASK 2 — HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 2: Handling missing values")
print("=" * 60)

print("Missing values per ticker BEFORE cleaning:")
print(close_prices.isnull().sum().to_string())

# Forward-fill: carry the last known price forward (e.g. market holidays)
close_prices.ffill(inplace=True)
# Back-fill: only needed if there are NaNs at the very start of the series
close_prices.bfill(inplace=True)

print("\nMissing values per ticker AFTER cleaning:")
print(close_prices.isnull().sum().to_string())

# ─────────────────────────────────────────────────────────────────────
# COMPUTE DAILY RETURNS
# ─────────────────────────────────────────────────────────────────────
# pct_change() gives (today - yesterday) / yesterday
# dropna() removes the first row which has no prior day
returns = close_prices.pct_change().dropna()

# ─────────────────────────────────────────────────────────────────────
# TASK 3 — DAILY RETURNS  (one chart per ticker)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 3: Plotting daily returns (individual charts)")
print("=" * 60)

for ticker in TICKERS:
    fig, ax = plt.subplots(figsize=(14, 4))
    r = returns[ticker] * 100

    # Color bars: green for positive days, red for negative
    bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in r]
    ax.bar(r.index, r, color=bar_colors, alpha=0.7, width=1.2)
    ax.axhline(0, color="black", linewidth=0.8)

    # Annotate mean return
    mean_r = r.mean()
    ax.axhline(mean_r, color=COLORS[ticker], linewidth=1.2,
               linestyle="--", label=f"Mean daily return: {mean_r:.3f}%")

    ax.set_title(f"{ticker} — Daily Returns (2021–2026)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Daily Return (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{OUTPUT_PREFIX}_task3_daily_returns_{ticker}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────
# TASK 4a — MOVING AVERAGES ON PRICE  (one chart per ticker)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 4a: Moving averages — individual charts")
print("=" * 60)

for ticker in TICKERS:
    price = close_prices[ticker]
    ma20  = price.rolling(SHORT_WINDOW).mean()
    ma50  = price.rolling(MID_WINDOW).mean()
    ma200 = price.rolling(LONG_WINDOW).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(price.index, price, color=COLORS[ticker], alpha=0.3,
            linewidth=0.8, label="Close Price")
    ax.plot(price.index, ma20,  color=COLORS[ticker],
            linewidth=1.5, label=f"MA{SHORT_WINDOW}")
    ax.plot(price.index, ma50,  color="orange",
            linewidth=1.5, linestyle="--", label=f"MA{MID_WINDOW}")
    ax.plot(price.index, ma200, color="black",
            linewidth=1.5, linestyle=":", label=f"MA{LONG_WINDOW}")

    # Shade golden cross (MA20 > MA200) zones
    ax.fill_between(price.index, price.min()*0.9, price.max()*1.1,
                    where=(ma20 > ma200), alpha=0.04, color="green",
                    label="MA20 > MA200 (bullish)")
    ax.fill_between(price.index, price.min()*0.9, price.max()*1.1,
                    where=(ma20 < ma200), alpha=0.04, color="red",
                    label="MA20 < MA200 (bearish)")

    ax.set_ylim(price.min() * 0.9, price.max() * 1.1)
    ax.set_title(f"{ticker} — Price & Moving Averages (20 / 50 / 200 day)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{OUTPUT_PREFIX}_task4a_moving_averages_{ticker}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────
# TASK 4b — ROLLING VOLATILITY  (one chart per ticker)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 4b: Rolling volatility — individual charts")
print("=" * 60)

for ticker in TICKERS:
    # Rolling std dev of daily returns, annualised by multiplying by sqrt(252)
    # 252 = trading days in a year
    vol = returns[ticker].rolling(VOL_WINDOW).std() * np.sqrt(252) * 100
    avg = vol.mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(vol.index, vol, alpha=0.3, color=COLORS[ticker])
    ax.plot(vol.index, vol, color=COLORS[ticker], linewidth=1.2,
            label=f"{VOL_WINDOW}-day rolling vol")
    ax.axhline(avg, color="black", linewidth=1.0, linestyle="--",
               label=f"5-year avg: {avg:.1f}%")

    # Shade high-vol periods (> avg * 1.5)
    high_vol_threshold = avg * 1.5
    ax.axhline(high_vol_threshold, color="red", linewidth=0.8,
               linestyle=":", alpha=0.6, label=f"High vol threshold: {high_vol_threshold:.1f}%")
    ax.fill_between(vol.index, vol, high_vol_threshold,
                    where=(vol > high_vol_threshold),
                    alpha=0.2, color="red", label="High vol zone")

    ax.set_title(f"{ticker} — {VOL_WINDOW}-day Rolling Volatility (Annualised)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Annualised Volatility (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{OUTPUT_PREFIX}_task4b_rolling_volatility_{ticker}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────
# TASK 4c — NORMALISED PRICE COMPARISON  (all tickers, one chart)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 4c: Normalised price comparison")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 6))
for ticker in TICKERS:
    norm = close_prices[ticker] / close_prices[ticker].iloc[0] * 100
    final_val = norm.iloc[-1]
    ax.plot(norm.index, norm, color=COLORS[ticker], linewidth=1.8,
            label=f"{ticker}  ({final_val:.0f})")

ax.axhline(100, color="gray", linewidth=0.8, linestyle="--", alpha=0.6,
           label="Base = 100")
ax.set_title("Normalised Price Performance — Base 100 at Apr 2021\n"
             "(Legend shows final indexed value)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Indexed Price")
ax.set_xlabel("Date")
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(labelsize=9)
plt.tight_layout()
fname = OUTPUT_DIR / f"{OUTPUT_PREFIX}_task4c_normalised_prices.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────
# TASK 5 — INSIGHTS & TRADING SIGNALS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 5: Insights and trading signals")
print("=" * 60)

rows = []
for ticker in TICKERS:
    r     = returns[ticker]
    price = close_prices[ticker]

    # Annualised return: mean daily return × 252 trading days
    ann_ret = r.mean() * 252

    # Annualised volatility: std dev of daily returns × sqrt(252)
    ann_vol = r.std() * np.sqrt(252)

    # Sharpe ratio (no risk-free rate — simplification for academic use)
    sharpe = ann_ret / ann_vol

    # Max drawdown: worst peak-to-trough decline over the period
    # cummax() tracks the running high; dividing gives the drawdown at each point
    max_dd = ((price / price.cummax()) - 1).min()

    # Latest MA values for signal
    ma20  = price.rolling(SHORT_WINDOW).mean().iloc[-1]
    ma50  = price.rolling(MID_WINDOW).mean().iloc[-1]
    ma200 = price.rolling(LONG_WINDOW).mean().iloc[-1]
    last  = price.iloc[-1]

    # Trend signal: classic MA stack logic
    if last > ma20 > ma50 > ma200:
        trend = "STRONG UPTREND"
    elif last > ma50 > ma200:
        trend = "UPTREND"
    elif last < ma20 < ma50 < ma200:
        trend = "STRONG DOWNTREND"
    elif last < ma50 < ma200:
        trend = "DOWNTREND"
    else:
        trend = "MIXED"

    # Volatility regime: compare recent 20-day vol vs full-period vol
    rec_vol = r.iloc[-VOL_WINDOW:].std() * np.sqrt(252)
    regime  = ("HIGH VOL" if rec_vol > ann_vol * 1.2 else
               "LOW VOL"  if rec_vol < ann_vol * 0.8 else "NORMAL")

    # Action
    if "UPTREND" in trend and regime != "HIGH VOL":
        action = "BUY / HOLD"
    elif "DOWNTREND" in trend:
        action = "SELL / AVOID"
    elif regime == "HIGH VOL":
        action = "REDUCE SIZE"
    else:
        action = "NEUTRAL / WATCH"

    rows.append({
        "Ticker":       ticker,
        "Last Price":   f"${last:.2f}",
        "Ann. Return":  f"{ann_ret*100:+.1f}%",
        "Ann. Vol":     f"{ann_vol*100:.1f}%",
        "Sharpe":       round(sharpe, 2),
        "Max Drawdown": f"{max_dd*100:.1f}%",
        "MA Trend":     trend,
        "Vol Regime":   regime,
        "Action":       action,
    })

df = pd.DataFrame(rows).set_index("Ticker")
print("\n", df.to_string())

# ─── Sharpe + Volatility summary chart ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tickers_list = [r["Ticker"] for r in rows]
sharpe_vals  = [r["Sharpe"] for r in rows]
vol_vals     = [float(r["Ann. Vol"].replace("%", "")) for r in rows]
actions      = [r["Action"] for r in rows]
acolors      = {"BUY / HOLD": "#2ca02c", "NEUTRAL / WATCH": "#ff7f0e",
                "REDUCE SIZE": "#1f77b4", "SELL / AVOID": "#d62728"}
bcolors      = [acolors.get(a, "gray") for a in actions]

# Sharpe chart
ax1 = axes[0]
bars = ax1.bar(tickers_list, sharpe_vals, color=bcolors, edgecolor="white", linewidth=1.2)
ax1.axhline(0,   color="black", linewidth=0.8)
ax1.axhline(1.0, color="green", linewidth=1.0, linestyle="--",
            alpha=0.6, label="Sharpe = 1.0 (benchmark)")
for bar, val, act in zip(bars, sharpe_vals, actions):
    ypos = val + 0.03 if val >= 0 else val - 0.08
    va   = "bottom" if val >= 0 else "top"
    ax1.text(bar.get_x() + bar.get_width()/2, ypos,
             act, ha="center", va=va, fontsize=7, fontweight="bold")
ax1.set_title("Sharpe Ratio by Ticker", fontweight="bold", fontsize=12)
ax1.set_ylabel("Sharpe Ratio (annualised, no Rf)")
ax1.legend(fontsize=8)

# Volatility chart
ax2 = axes[1]
bars2 = ax2.bar(tickers_list, vol_vals, color=bcolors, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars2, vol_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
ax2.set_title("Annualised Volatility (Std Dev × √252)", fontweight="bold", fontsize=12)
ax2.set_ylabel("Volatility (%)")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=c, label=a) for a, c in acolors.items()],
           loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.suptitle("Task 5 — Risk/Return Summary & Trading Signals",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fname = OUTPUT_DIR / f"{OUTPUT_PREFIX}_task5_signals_dashboard.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {fname}")

# ─── Correlation matrix ───
corr = returns.corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(TICKERS))); ax.set_xticklabels(TICKERS, fontsize=10)
ax.set_yticks(range(len(TICKERS))); ax.set_yticklabels(TICKERS, fontsize=10)
for i in range(len(TICKERS)):
    for j in range(len(TICKERS)):
        val = corr.iloc[i, j]
        # White text on dark cells, black on light
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Return Correlation Matrix (5-year daily returns)",
             fontweight="bold", fontsize=11)
plt.tight_layout()
fname = OUTPUT_DIR / f"{OUTPUT_PREFIX}_task5_correlation_matrix.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fname}")

# ─── Text report ───
report_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_task5_insights_report.txt"
with open(report_path, "w") as f:
    f.write("TECHNICAL INDICATOR DASHBOARD\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Tickers     : {', '.join(TICKERS)}\n")
    f.write(f"Period      : {START_DATE} to {END_DATE}\n")
    f.write(f"Data source : Yahoo Finance (direct API, adjusted close)\n")
    f.write(f"MA windows  : {SHORT_WINDOW}d / {MID_WINDOW}d / {LONG_WINDOW}d\n")
    f.write(f"Vol window  : {VOL_WINDOW}-day rolling std dev × sqrt(252)\n\n")
    f.write("-" * 65 + "\n")
    f.write("PERFORMANCE SUMMARY\n")
    f.write("-" * 65 + "\n")
    f.write(df.to_string() + "\n\n")
    f.write("-" * 65 + "\n")
    f.write("SIGNAL LOGIC\n")
    f.write("-" * 65 + "\n")
    f.write("  Trend (MA Stack):\n")
    f.write("    STRONG UPTREND   : Last > MA20 > MA50 > MA200\n")
    f.write("    UPTREND          : Last > MA50 > MA200\n")
    f.write("    STRONG DOWNTREND : Last < MA20 < MA50 < MA200\n")
    f.write("    DOWNTREND        : Last < MA50 < MA200\n")
    f.write("    MIXED            : None of the above\n\n")
    f.write("  Vol Regime:\n")
    f.write("    HIGH VOL  : Recent 20d vol > 120% of 5-year avg\n")
    f.write("    LOW VOL   : Recent 20d vol < 80% of 5-year avg\n")
    f.write("    NORMAL    : Within 80–120% of 5-year avg\n\n")
    f.write("  Action:\n")
    f.write("    BUY / HOLD    : Uptrend + normal/low vol\n")
    f.write("    SELL / AVOID  : Downtrend (any vol regime)\n")
    f.write("    REDUCE SIZE   : High vol (mixed trend)\n")
    f.write("    NEUTRAL/WATCH : Mixed signals\n\n")
    f.write("-" * 65 + "\n")
    f.write("NOTE ON SHARPE RATIO\n")
    f.write("-" * 65 + "\n")
    f.write("  Sharpe = Ann. Return / Ann. Volatility\n")
    f.write("  No risk-free rate applied (simplification for academic use).\n\n")
    f.write("-" * 65 + "\n")
    f.write("Disclaimer: Educational purposes only. Not financial advice.\n")

print(f"  Saved: {report_path}")

print("\n" + "=" * 60)
print(f"ALL DONE. Output folder: {OUTPUT_DIR}")
print(f"Files prefixed with: {OUTPUT_PREFIX}_")
print("=" * 60)