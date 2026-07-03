# ============================================================
# NAMED CHART PATTERN ML TRADING FRAMEWORK (TradingView-style charts)
# SVM + RBF Kernel
# Yahoo Finance Daily Data
# 70% Train / 30% Out-of-Sample Test
#
# Same pipeline as chart_pattern_svm.py:
# 1. Detects named chart patterns using confirmed pivots
# 2. Quantifies pattern strength as ML features
# 3. Trains SVM-RBF on technical + named-pattern features
# 4. Trades model probabilities only when chart-pattern evidence exists
# 5. Outputs train/test positions, equity curves, drawdown, ML metrics,
#    and pattern diagnostics
#
# Difference: pattern-example plots are rendered as dark, single-panel,
# TradingView-style candlestick charts instead of the plain line charts
# with pivot/EMA overlays used in the base version.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. USER INPUT
# ============================================================

TICKERS = [
    "SOL-USD",   # Crypto
    # "SPY",       # ETF
    # "GLD",       # gold ETF
    # "TLT",       # bonds ETF
]

START_DATE = "2000-01-01"
END_DATE = "2026-05-30"

TRAIN_RATIO = 0.70

PREDICTION_HORIZON = 10       # Future return horizon used for ML label
LABEL_THRESHOLD = 0.03        # +3% long label, -3% short label, otherwise no-trade

TRANSACTION_COST_BPS = 5      # 5 bps = 0.05% per position change
RANDOM_STATE = 42
SHOW_PLOTS = True

# Named chart-pattern settings
PIVOT_CONFIRM_BARS = 3        # Pivot is confirmed only after this many bars; helps avoid lookahead
PATTERN_LOOKBACK_PIVOTS = 12  # Number of recent confirmed pivots used for pattern detection
REQUIRE_PATTERN_FOR_ENTRY = True
MIN_PATTERN_SCORE_FOR_ENTRY = 0.10
USE_EMA_TREND_FILTER = False  # False lets reversal patterns trade earlier; True makes entries stricter


# ============================================================
# 2. OUTPUT FOLDER
# ============================================================

def create_output_folder():
    output_dir = Path(__file__).parent / "output" / "chart_pattern_svm"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_annualization(ticker):
    if "-USD" in ticker.upper():
        return 365
    return 252

# ============================================================
# 3. DOWNLOAD YAHOO DAILY OHLCV DATA
# ============================================================

def download_daily_data(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError("No data downloaded. Check ticker/date range.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })

    if "adj_close" in df.columns:
        df["price"] = df["adj_close"]
    else:
        df["price"] = df["close"]

    df = df[["open", "high", "low", "close", "price", "volume"]].dropna()
    return df


# ============================================================
# 4. STANDARD TECHNICAL FEATURES
# ============================================================

def add_standard_technical_features(df):
    df = df.copy()

    price = df["price"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # Return / momentum
    df["ret_1d"] = price.pct_change()
    df["ret_5d"] = price.pct_change(5)
    df["ret_10d"] = price.pct_change(10)
    df["ret_20d"] = price.pct_change(20)

    # EMA trend
    df["ema_20"] = price.ewm(span=20, adjust=False).mean()
    df["ema_50"] = price.ewm(span=50, adjust=False).mean()
    df["ema_100"] = price.ewm(span=100, adjust=False).mean()
    df["price_above_ema100"] = (price > df["ema_100"]).astype(int)
    df["ema20_slope_5d"] = df["ema_20"].pct_change(5)
    df["ema50_slope_10d"] = df["ema_50"].pct_change(10)

    # RSI 14
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ATR 14
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / price

    # Bollinger width / z-score
    rolling_mean_20 = price.rolling(20).mean()
    rolling_std_20 = price.rolling(20).std()
    df["bb_width_20"] = (4 * rolling_std_20) / price
    df["zscore_30"] = (price - price.rolling(30).mean()) / price.rolling(30).std()

    # Support / resistance / breakout
    prev_high_20 = high.shift(1).rolling(20).max()
    prev_low_20 = low.shift(1).rolling(20).min()
    df["distance_to_20d_high"] = price / prev_high_20 - 1
    df["distance_to_20d_low"] = price / prev_low_20 - 1
    df["breakout_20d"] = (price > prev_high_20).astype(int)
    df["breakdown_20d"] = (price < prev_low_20).astype(int)
    df["drawdown_from_20d_high"] = price / price.rolling(20).max() - 1

    # Volume confirmation
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    df["volume_zscore_20"] = (volume - vol_mean_20) / vol_std_20.replace(0, np.nan)
    df["volume_ratio_20"] = volume / vol_mean_20.replace(0, np.nan)

    return df


# ============================================================
# 5. CONFIRMED PIVOT DETECTION
# Important: this uses delayed confirmation to reduce lookahead bias.
# A pivot at t-PIVOT_CONFIRM_BARS is only known at t.
# ============================================================

def add_confirmed_pivots(df, confirm_bars=3):
    df = df.copy()
    w = int(confirm_bars)
    window = 2 * w + 1

    centered_high = df["high"].shift(w)
    centered_low = df["low"].shift(w)

    rolling_high = df["high"].rolling(window).max()
    rolling_low = df["low"].rolling(window).min()

    df["pivot_high_confirmed"] = ((centered_high == rolling_high) & centered_high.notna()).astype(int)
    df["pivot_low_confirmed"] = ((centered_low == rolling_low) & centered_low.notna()).astype(int)

    df["confirmed_pivot_high_price"] = np.where(
        df["pivot_high_confirmed"] == 1,
        centered_high,
        np.nan,
    )
    df["confirmed_pivot_low_price"] = np.where(
        df["pivot_low_confirmed"] == 1,
        centered_low,
        np.nan,
    )

    return df


# ============================================================
# 6. NAMED CHART-PATTERN FEATURE ENGINEERING
# Patterns quantified:
# - Head and Shoulders bearish reversal
# - Inverse Head and Shoulders bullish reversal
# - Double Top bearish reversal
# - Double Bottom bullish reversal
# - Triangle / consolidation breakout and breakdown
# ============================================================

def _line_value(p1_idx, p1_price, p2_idx, p2_price, current_idx):
    if p2_idx == p1_idx:
        return 0.5 * (p1_price + p2_price)
    slope = (p2_price - p1_price) / (p2_idx - p1_idx)
    return p1_price + slope * (current_idx - p1_idx)


def _find_recent_sequence(pivots, sequence):
    if len(pivots) < len(sequence):
        return None
    for start in range(len(pivots) - len(sequence), -1, -1):
        chunk = pivots[start:start + len(sequence)]
        types = [p[0] for p in chunk]
        if types == sequence:
            return chunk
    return None


def _safe_ratio(a, b):
    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return a / b


def add_named_chart_pattern_features(
    df,
    confirm_bars=3,
    lookback_pivots=12,
):
    df = add_confirmed_pivots(df, confirm_bars=confirm_bars).copy()

    pattern_cols = [
        "head_shoulders_score",
        "inverse_head_shoulders_score",
        "double_top_score",
        "double_bottom_score",
        "triangle_contraction_score",
        "triangle_breakout_up_score",
        "triangle_breakdown_down_score",
        "bullish_pattern_score",
        "bearish_pattern_score",
    ]

    flag_cols = [
        "head_shoulders_flag",
        "inverse_head_shoulders_flag",
        "double_top_flag",
        "double_bottom_flag",
        "triangle_flag",
        "triangle_breakout_up_flag",
        "triangle_breakdown_down_flag",
    ]

    for col in pattern_cols + flag_cols:
        df[col] = 0.0

    df["active_chart_pattern"] = "none"

    confirmed_pivots = []
    prices = df["price"].values
    volumes = df["volume"].values
    vol_ma20 = df["volume"].rolling(20).mean().values

    for i in range(len(df)):
        # Add newly confirmed pivot highs/lows. Actual pivot occurred confirm_bars ago.
        pivot_actual_idx = i - confirm_bars

        if df["pivot_high_confirmed"].iloc[i] == 1 and pivot_actual_idx >= 0:
            pivot_price = float(df["confirmed_pivot_high_price"].iloc[i])
            confirmed_pivots.append(("H", pivot_actual_idx, pivot_price))

        if df["pivot_low_confirmed"].iloc[i] == 1 and pivot_actual_idx >= 0:
            pivot_price = float(df["confirmed_pivot_low_price"].iloc[i])
            confirmed_pivots.append(("L", pivot_actual_idx, pivot_price))

        recent = confirmed_pivots[-lookback_pivots:]
        current_price = float(prices[i])
        current_volume = float(volumes[i])
        volume_confirm = 0.0
        if i < len(vol_ma20) and not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0:
            volume_confirm = max(current_volume / vol_ma20[i] - 1.0, 0.0)

        # ----------------------------------------------------
        # Head and Shoulders: H-L-H-L-H
        # ----------------------------------------------------
        hs = _find_recent_sequence(recent, ["H", "L", "H", "L", "H"])
        if hs is not None:
            ls, low1, head, low2, rs = hs
            ls_p, head_p, rs_p = ls[2], head[2], rs[2]
            low1_p, low2_p = low1[2], low2[2]

            head_gap = _safe_ratio(head_p, max(ls_p, rs_p)) - 1
            shoulder_diff = abs(ls_p - rs_p) / max(0.5 * (ls_p + rs_p), 1e-12)
            shoulder_symmetry = max(1 - shoulder_diff / 0.15, 0)
            neckline_now = _line_value(low1[1], low1_p, low2[1], low2_p, i)
            breakdown_distance = max((neckline_now - current_price) / max(neckline_now, 1e-12), 0)

            if head_gap > 0.015 and shoulder_symmetry > 0:
                structure_score = min(head_gap * 10, 1.0) * 0.45 + shoulder_symmetry * 0.35
                confirm_score = min(breakdown_distance * 20, 1.0) * 0.15 + min(volume_confirm, 1.0) * 0.05
                score = structure_score + confirm_score
                df.iloc[i, df.columns.get_loc("head_shoulders_score")] = score
                if breakdown_distance > 0:
                    df.iloc[i, df.columns.get_loc("head_shoulders_flag")] = 1.0

        # ----------------------------------------------------
        # Inverse Head and Shoulders: L-H-L-H-L
        # ----------------------------------------------------
        inv_hs = _find_recent_sequence(recent, ["L", "H", "L", "H", "L"])
        if inv_hs is not None:
            ls, high1, head, high2, rs = inv_hs
            ls_p, head_p, rs_p = ls[2], head[2], rs[2]
            high1_p, high2_p = high1[2], high2[2]

            head_depth = _safe_ratio(min(ls_p, rs_p), head_p) - 1
            shoulder_diff = abs(ls_p - rs_p) / max(0.5 * (ls_p + rs_p), 1e-12)
            shoulder_symmetry = max(1 - shoulder_diff / 0.15, 0)
            neckline_now = _line_value(high1[1], high1_p, high2[1], high2_p, i)
            breakout_distance = max((current_price - neckline_now) / max(neckline_now, 1e-12), 0)

            if head_depth > 0.015 and shoulder_symmetry > 0:
                structure_score = min(head_depth * 10, 1.0) * 0.45 + shoulder_symmetry * 0.35
                confirm_score = min(breakout_distance * 20, 1.0) * 0.15 + min(volume_confirm, 1.0) * 0.05
                score = structure_score + confirm_score
                df.iloc[i, df.columns.get_loc("inverse_head_shoulders_score")] = score
                if breakout_distance > 0:
                    df.iloc[i, df.columns.get_loc("inverse_head_shoulders_flag")] = 1.0

        # ----------------------------------------------------
        # Double Top: H-L-H
        # ----------------------------------------------------
        dt = _find_recent_sequence(recent, ["H", "L", "H"])
        if dt is not None:
            h1, valley, h2 = dt
            h1_p, valley_p, h2_p = h1[2], valley[2], h2[2]
            top_diff = abs(h1_p - h2_p) / max(0.5 * (h1_p + h2_p), 1e-12)
            top_similarity = max(1 - top_diff / 0.08, 0)
            valley_depth = max(min(h1_p, h2_p) / max(valley_p, 1e-12) - 1, 0)
            breakdown_distance = max((valley_p - current_price) / max(valley_p, 1e-12), 0)

            if top_similarity > 0 and valley_depth > 0.03:
                score = top_similarity * 0.45 + min(valley_depth * 8, 1.0) * 0.35
                score += min(breakdown_distance * 20, 1.0) * 0.15 + min(volume_confirm, 1.0) * 0.05
                df.iloc[i, df.columns.get_loc("double_top_score")] = score
                if breakdown_distance > 0:
                    df.iloc[i, df.columns.get_loc("double_top_flag")] = 1.0

        # ----------------------------------------------------
        # Double Bottom: L-H-L
        # ----------------------------------------------------
        db = _find_recent_sequence(recent, ["L", "H", "L"])
        if db is not None:
            l1, peak, l2 = db
            l1_p, peak_p, l2_p = l1[2], peak[2], l2[2]
            low_diff = abs(l1_p - l2_p) / max(0.5 * (l1_p + l2_p), 1e-12)
            low_similarity = max(1 - low_diff / 0.08, 0)
            bounce_depth = max(peak_p / max(max(l1_p, l2_p), 1e-12) - 1, 0)
            breakout_distance = max((current_price - peak_p) / max(peak_p, 1e-12), 0)

            if low_similarity > 0 and bounce_depth > 0.03:
                score = low_similarity * 0.45 + min(bounce_depth * 8, 1.0) * 0.35
                score += min(breakout_distance * 20, 1.0) * 0.15 + min(volume_confirm, 1.0) * 0.05
                df.iloc[i, df.columns.get_loc("double_bottom_score")] = score
                if breakout_distance > 0:
                    df.iloc[i, df.columns.get_loc("double_bottom_flag")] = 1.0

        # ----------------------------------------------------
        # Triangle / consolidation contraction
        # Uses last 3 pivot highs and last 3 pivot lows
        # ----------------------------------------------------
        highs = [p for p in recent if p[0] == "H"][-3:]
        lows = [p for p in recent if p[0] == "L"][-3:]

        if len(highs) == 3 and len(lows) == 3:
            h_idx = np.array([p[1] for p in highs], dtype=float)
            h_price = np.array([p[2] for p in highs], dtype=float)
            l_idx = np.array([p[1] for p in lows], dtype=float)
            l_price = np.array([p[2] for p in lows], dtype=float)

            if len(set(h_idx)) == 3 and len(set(l_idx)) == 3:
                h_slope, h_intercept = np.polyfit(h_idx, h_price, 1)
                l_slope, l_intercept = np.polyfit(l_idx, l_price, 1)
                upper_now = h_slope * i + h_intercept
                lower_now = l_slope * i + l_intercept

                initial_range = max(h_price[0] - l_price[0], 1e-12)
                current_range = max(upper_now - lower_now, 1e-12)
                contraction = max(1 - current_range / initial_range, 0)

                lower_highs = h_price[-1] < h_price[0]
                higher_lows = l_price[-1] > l_price[0]

                if lower_highs and higher_lows and contraction > 0.10:
                    triangle_score = min(contraction, 1.0)
                    df.iloc[i, df.columns.get_loc("triangle_contraction_score")] = triangle_score
                    df.iloc[i, df.columns.get_loc("triangle_flag")] = 1.0

                    breakout_up = max((current_price - upper_now) / max(upper_now, 1e-12), 0)
                    breakdown_down = max((lower_now - current_price) / max(lower_now, 1e-12), 0)

                    if breakout_up > 0:
                        score = triangle_score * 0.65 + min(breakout_up * 20, 1.0) * 0.30 + min(volume_confirm, 1.0) * 0.05
                        df.iloc[i, df.columns.get_loc("triangle_breakout_up_score")] = score
                        df.iloc[i, df.columns.get_loc("triangle_breakout_up_flag")] = 1.0

                    if breakdown_down > 0:
                        score = triangle_score * 0.65 + min(breakdown_down * 20, 1.0) * 0.30 + min(volume_confirm, 1.0) * 0.05
                        df.iloc[i, df.columns.get_loc("triangle_breakdown_down_score")] = score
                        df.iloc[i, df.columns.get_loc("triangle_breakdown_down_flag")] = 1.0

        # ----------------------------------------------------
        # Aggregate bullish/bearish named-pattern score
        # ----------------------------------------------------
        bullish_scores = {
            "inverse_head_shoulders": df["inverse_head_shoulders_score"].iloc[i],
            "double_bottom": df["double_bottom_score"].iloc[i],
            "triangle_breakout_up": df["triangle_breakout_up_score"].iloc[i],
        }
        bearish_scores = {
            "head_shoulders": df["head_shoulders_score"].iloc[i],
            "double_top": df["double_top_score"].iloc[i],
            "triangle_breakdown_down": df["triangle_breakdown_down_score"].iloc[i],
        }

        bullish_best_name = max(bullish_scores, key=bullish_scores.get)
        bearish_best_name = max(bearish_scores, key=bearish_scores.get)
        bullish_best_score = float(bullish_scores[bullish_best_name])
        bearish_best_score = float(bearish_scores[bearish_best_name])

        df.iloc[i, df.columns.get_loc("bullish_pattern_score")] = bullish_best_score
        df.iloc[i, df.columns.get_loc("bearish_pattern_score")] = bearish_best_score

        if bullish_best_score >= bearish_best_score and bullish_best_score > 0:
            df.iloc[i, df.columns.get_loc("active_chart_pattern")] = bullish_best_name
        elif bearish_best_score > bullish_best_score and bearish_best_score > 0:
            df.iloc[i, df.columns.get_loc("active_chart_pattern")] = bearish_best_name

    return df


# ============================================================
# 7. LABEL ENGINEERING
# ============================================================

def add_labels(df, horizon=10, threshold=0.03):
    df = df.copy()
    df["future_return"] = df["price"].shift(-horizon) / df["price"] - 1
    df["label"] = 0
    df.loc[df["future_return"] > threshold, "label"] = 1
    df.loc[df["future_return"] < -threshold, "label"] = -1
    return df


# ============================================================
# 8. MODEL PROBABILITY HELPER
# ============================================================

def get_class_probabilities(model, X):
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    p_long = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    p_short = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(X))

    return p_long, p_short


# ============================================================
# 9. POSITION ENGINE
# ============================================================

def build_positions(
    df,
    p_long,
    p_short,
    entry_threshold=0.60,
    exit_threshold=0.45,
    max_holding_days=10,
    atr_stop_mult=2.0,
    atr_take_profit_mult=3.0,
    require_pattern=True,
    min_pattern_score=0.10,
    use_ema_filter=False,
):
    position = []
    current_pos = 0
    entry_price = np.nan
    entry_atr = np.nan
    holding_days = 0

    prices = df["price"].values
    ema100 = df["ema_100"].values
    atr = df["atr_14"].values
    bullish_score = df["bullish_pattern_score"].values
    bearish_score = df["bearish_pattern_score"].values

    for i in range(len(df)):
        price_i = prices[i]
        ema_i = ema100[i]
        atr_i = atr[i]
        long_prob = p_long[i]
        short_prob = p_short[i]
        bull_i = bullish_score[i]
        bear_i = bearish_score[i]

        long_pattern_ok = (not require_pattern) or (bull_i >= min_pattern_score)
        short_pattern_ok = (not require_pattern) or (bear_i >= min_pattern_score)

        long_trend_ok = (not use_ema_filter) or (price_i > ema_i)
        short_trend_ok = (not use_ema_filter) or (price_i < ema_i)

        if current_pos == 0:
            holding_days = 0

            if long_prob >= entry_threshold and long_pattern_ok and long_trend_ok:
                current_pos = 1
                entry_price = price_i
                entry_atr = atr_i

            elif short_prob >= entry_threshold and short_pattern_ok and short_trend_ok:
                current_pos = -1
                entry_price = price_i
                entry_atr = atr_i

        elif current_pos == 1:
            holding_days += 1
            stop_price = entry_price - atr_stop_mult * entry_atr
            take_profit_price = entry_price + atr_take_profit_mult * entry_atr

            exit_long = (
                long_prob < exit_threshold
                or bear_i > bull_i + 0.10
                or price_i <= stop_price
                or price_i >= take_profit_price
                or holding_days >= max_holding_days
            )

            if use_ema_filter:
                exit_long = exit_long or price_i < ema_i

            if exit_long:
                current_pos = 0
                entry_price = np.nan
                entry_atr = np.nan
                holding_days = 0

        elif current_pos == -1:
            holding_days += 1
            stop_price = entry_price + atr_stop_mult * entry_atr
            take_profit_price = entry_price - atr_take_profit_mult * entry_atr

            exit_short = (
                short_prob < exit_threshold
                or bull_i > bear_i + 0.10
                or price_i >= stop_price
                or price_i <= take_profit_price
                or holding_days >= max_holding_days
            )

            if use_ema_filter:
                exit_short = exit_short or price_i > ema_i

            if exit_short:
                current_pos = 0
                entry_price = np.nan
                entry_atr = np.nan
                holding_days = 0

        position.append(current_pos)

    return pd.Series(position, index=df.index, name="position")


# ============================================================
# 10. BACKTEST ENGINE
# ============================================================

def backtest_from_position(df, position, p_long, p_short, transaction_cost_bps=5):
    out = df.copy()
    out["p_long"] = p_long
    out["p_short"] = p_short
    out["position"] = position

    # Anti-lookahead: signal today is traded tomorrow
    out["position_lagged"] = out["position"].shift(1).fillna(0)

    out["asset_return"] = out["price"].pct_change().fillna(0)
    turnover = out["position_lagged"].diff().abs().fillna(out["position_lagged"].abs())
    cost = turnover * (transaction_cost_bps / 10000)

    out["strategy_return"] = out["position_lagged"] * out["asset_return"] - cost
    out["equity_curve"] = (1 + out["strategy_return"]).cumprod()
    out["buy_hold_curve"] = (1 + out["asset_return"]).cumprod()

    return out


# ============================================================
# 11. PERFORMANCE METRICS
# ============================================================

def performance_metrics(bt, annualization=252):
    r = bt["strategy_return"].dropna()
    equity = bt["equity_curve"].dropna()

    if len(r) == 0 or len(equity) == 0:
        return {}

    total_return = equity.iloc[-1] - 1
    years = len(r) / annualization
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else np.nan
    annualized_vol = r.std() * np.sqrt(annualization)
    sharpe = r.mean() / r.std() * np.sqrt(annualization) if r.std() != 0 else np.nan
    downside = r[r < 0]
    sortino = r.mean() / downside.std() * np.sqrt(annualization) if downside.std() != 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    trade_entries = ((bt["position_lagged"] != 0) & (bt["position_lagged"].shift(1).fillna(0) == 0)).sum()
    exposure = (bt["position_lagged"] != 0).mean()
    active_returns = r[r != 0]
    win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else np.nan

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Vol": annualized_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar": calmar,
        "Win Rate": win_rate,
        "Trade Entries": int(trade_entries),
        "Exposure": exposure,
    }


# ============================================================
# 12. TRAIN-ONLY TRADING PARAMETER OPTIMIZER
# ============================================================

def optimize_trading_thresholds(
    train_df,
    model,
    X_train,
    annualization,
    transaction_cost_bps=5,
):
    p_long, p_short = get_class_probabilities(model, X_train)

    entry_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
    exit_thresholds = [0.35, 0.40, 0.45, 0.50]
    max_holding_days_list = [5, 10, 20]
    atr_stop_list = [1.5, 2.0, 2.5]
    atr_tp_list = [2.0, 3.0, 4.0]
    min_pattern_scores = [0.05, 0.10, 0.20]

    results = []

    for entry in entry_thresholds:
        for exit_ in exit_thresholds:
            for max_hold in max_holding_days_list:
                for stop_mult in atr_stop_list:
                    for tp_mult in atr_tp_list:
                        for min_score in min_pattern_scores:
                            pos = build_positions(
                                train_df,
                                p_long,
                                p_short,
                                entry_threshold=entry,
                                exit_threshold=exit_,
                                max_holding_days=max_hold,
                                atr_stop_mult=stop_mult,
                                atr_take_profit_mult=tp_mult,
                                require_pattern=REQUIRE_PATTERN_FOR_ENTRY,
                                min_pattern_score=min_score,
                                use_ema_filter=USE_EMA_TREND_FILTER,
                            )

                            bt = backtest_from_position(
                                train_df,
                                pos,
                                p_long,
                                p_short,
                                transaction_cost_bps=transaction_cost_bps,
                            )

                            metrics = performance_metrics(bt, annualization=annualization)
                            sharpe = metrics["Sharpe"]
                            mdd = metrics["Max Drawdown"]
                            trades = metrics["Trade Entries"]
                            exposure = metrics["Exposure"]
                            total_return = metrics["Total Return"]

                            # Penalize tiny sample / fake high Sharpe
                            if trades < 5 or exposure < 0.02:
                                selection_score = -999
                            else:
                                selection_score = sharpe - 0.50 * abs(mdd) + 0.10 * np.sign(total_return)

                            results.append({
                                "entry_threshold": entry,
                                "exit_threshold": exit_,
                                "max_holding_days": max_hold,
                                "atr_stop_mult": stop_mult,
                                "atr_take_profit_mult": tp_mult,
                                "min_pattern_score": min_score,
                                "train_sharpe": sharpe,
                                "train_mdd": mdd,
                                "train_total_return": total_return,
                                "train_trades": trades,
                                "train_exposure": exposure,
                                "selection_score": selection_score,
                            })

    results_df = pd.DataFrame(results).sort_values("selection_score", ascending=False)
    best_params = results_df.iloc[0].to_dict()
    return best_params, results_df


# ============================================================
# 13. PATTERN DIAGNOSTICS
# ============================================================

def pattern_diagnostics(df, pattern_flag_cols):
    rows = []
    for col in pattern_flag_cols:
        subset = df[df[col] > 0]
        if len(subset) == 0:
            rows.append({
                "pattern": col,
                "count": 0,
                "avg_future_return": np.nan,
                "median_future_return": np.nan,
                "positive_rate": np.nan,
                "long_label_rate": np.nan,
                "short_label_rate": np.nan,
            })
            continue

        rows.append({
            "pattern": col,
            "count": len(subset),
            "avg_future_return": subset["future_return"].mean(),
            "median_future_return": subset["future_return"].median(),
            "positive_rate": (subset["future_return"] > 0).mean(),
            "long_label_rate": (subset["label"] == 1).mean(),
            "short_label_rate": (subset["label"] == -1).mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# 14. PLOTTING
# ============================================================

def save_or_show_plot(filepath, show_plots=False):
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_equity_curves(train_bt, test_bt, ticker, output_prefix, show_plots=False):
    split_date = test_bt.index[0]

    plt.figure(figsize=(14, 6))
    plt.plot(train_bt.index, train_bt["equity_curve"], label="ML Named-Pattern Strategy - Train 70%")
    plt.plot(train_bt.index, train_bt["buy_hold_curve"], label="Buy & Hold - Train 70%")
    plt.title(f"{ticker} Train Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    save_or_show_plot(f"{output_prefix}_train_equity_curve.png", show_plots)

    plt.figure(figsize=(14, 6))
    plt.plot(test_bt.index, test_bt["equity_curve"], label="ML Named-Pattern Strategy - OOS Test 30%")
    plt.plot(test_bt.index, test_bt["buy_hold_curve"], label="Buy & Hold - OOS Test 30%")
    plt.title(f"{ticker} OOS Test Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    save_or_show_plot(f"{output_prefix}_test_equity_curve.png", show_plots)

    full_bt = pd.concat([train_bt, test_bt]).copy()
    full_bt["full_strategy_equity"] = (1 + full_bt["strategy_return"]).cumprod()
    full_bt["full_buy_hold_equity"] = (1 + full_bt["asset_return"]).cumprod()

    plt.figure(figsize=(15, 6))
    plt.plot(full_bt.index, full_bt["full_strategy_equity"], label="ML Named-Pattern Strategy Full")
    plt.plot(full_bt.index, full_bt["full_buy_hold_equity"], label="Buy & Hold Full")
    plt.axvline(split_date, linestyle="--", label="70/30 Train-Test Split")
    plt.title(f"{ticker} Full Equity Curve: Train + OOS Test")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    save_or_show_plot(f"{output_prefix}_full_train_test_equity_curve.png", show_plots)

    equity = test_bt["equity_curve"]
    drawdown = equity / equity.cummax() - 1
    plt.figure(figsize=(14, 5))
    plt.plot(drawdown.index, drawdown)
    plt.title(f"{ticker} OOS Test Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    save_or_show_plot(f"{output_prefix}_test_drawdown.png", show_plots)

def plot_pattern_events(test_bt, ticker, output_prefix, show_plots=False):
    plt.figure(figsize=(15, 6))
    plt.plot(test_bt.index, test_bt["price"], label="Price")

    long_events = test_bt[test_bt["bullish_pattern_score"] >= MIN_PATTERN_SCORE_FOR_ENTRY]
    short_events = test_bt[test_bt["bearish_pattern_score"] >= MIN_PATTERN_SCORE_FOR_ENTRY]

    plt.scatter(long_events.index, long_events["price"], marker="^", label="Bullish Pattern", s=35)
    plt.scatter(short_events.index, short_events["price"], marker="v", label="Bearish Pattern", s=35)

    plt.title(f"{ticker} OOS Named Chart-Pattern Detections")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    save_or_show_plot(f"{output_prefix}_oos_named_pattern_events.png", show_plots)

def plot_individual_pattern_examples(data, ticker, output_prefix, max_examples=5):
    """
    Saves one full-size TradingView-style daily candlestick chart per pattern example.
    Log scale. No EMAs, no pivots, no volume. Saves directly to output folder.
    """

    patterns = {
        "Double_Bottom":             ("double_bottom_flag",            "bullish", "^", "#26a69a"),
        "Double_Top":                ("double_top_flag",               "bearish", "v", "#ef5350"),
        "Head_and_Shoulders":        ("head_shoulders_flag",           "bearish", "v", "#ef5350"),
        "Inverse_Head_and_Shoulders":("inverse_head_shoulders_flag",   "bullish", "^", "#26a69a"),
        "Triangle_Breakout_Up":      ("triangle_breakout_up_flag",     "bullish", "^", "#26a69a"),
        "Triangle_Breakdown_Down":   ("triangle_breakdown_down_flag",  "bearish", "v", "#ef5350"),
    }

    BG_COLOR      = "#131722"
    PANEL_COLOR   = "#1e222d"
    GRID_COLOR    = "#2a2e39"
    TEXT_COLOR    = "#d1d4dc"
    BULL_COLOR    = "#26a69a"
    BEAR_COLOR    = "#ef5350"
    VLINE_COLOR   = "#ffffff"

    WINDOW = 80

    for pattern_name, (flag_col, direction, marker, detect_color) in patterns.items():

        if flag_col not in data.columns:
            continue

        detections = data.index[data[flag_col] > 0].tolist()
        if len(detections) == 0:
            print(f"  No detections for: {pattern_name}")
            continue

        selected = []
        last_pos = -9999
        for dt in detections:
            pos = data.index.get_loc(dt)
            if pos - last_pos > WINDOW:
                selected.append(dt)
                last_pos = pos
            if len(selected) >= max_examples:
                break

        print(f"\n  {pattern_name}: {len(selected)} example(s)")

        for n, detection_date in enumerate(selected, start=1):

            center = data.index.get_loc(detection_date)
            start  = max(0, center - WINDOW)
            end    = min(len(data) - 1, center + WINDOW)
            w      = data.iloc[start:end + 1].copy().reset_index()
            date_col = w.columns[0]

            detect_x = int(w.index[w[date_col] == detection_date][0])

            # ------------------------------------------------
            # Single price panel only
            # ------------------------------------------------
            fig, ax = plt.subplots(figsize=(18, 9), facecolor=BG_COLOR)
            ax.set_facecolor(PANEL_COLOR)
            ax.tick_params(colors=TEXT_COLOR, labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COLOR)
            ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle="-", alpha=0.6)

            # ------------------------------------------------
            # Log scale
            # ------------------------------------------------
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:,.2f}")
            )
            ax.yaxis.set_minor_formatter(
                plt.FuncFormatter(lambda x, _: "")
            )

            # ------------------------------------------------
            # Candlesticks
            # ------------------------------------------------
            candle_width = 0.6

            for xi, row in w.iterrows():
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                is_bull    = c >= o
                body_color = BULL_COLOR if is_bull else BEAR_COLOR
                body_bottom = min(o, c)
                body_height = max(abs(c - o), l * 0.001)

                # Wick
                ax.plot(
                    [xi, xi], [l, h],
                    color=body_color, linewidth=0.8, zorder=2
                )
                # Body
                rect = Rectangle(
                    (xi - candle_width / 2, body_bottom),
                    candle_width, body_height,
                    facecolor=body_color,
                    edgecolor=body_color,
                    linewidth=0.4,
                    zorder=3,
                )
                ax.add_patch(rect)

            # ------------------------------------------------
            # Detection vertical line + marker
            # ------------------------------------------------
            ax.axvline(detect_x, color=VLINE_COLOR, linewidth=1.2,
                       linestyle="--", alpha=0.9, zorder=7)

            detect_price = float(data.loc[detection_date, "price"])
            ax.scatter(
                [detect_x], [detect_price],
                marker=marker, color=detect_color,
                s=250, zorder=8,
            )

            # ------------------------------------------------
            # X-axis date labels
            # ------------------------------------------------
            tick_spacing   = max(1, len(w) // 10)
            tick_positions = list(range(0, len(w), tick_spacing))
            tick_labels    = [str(w[date_col].iloc[i])[:10] for i in tick_positions]
            ax.set_xlim(-1, len(w))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=30, ha="right", color=TEXT_COLOR)

            # ------------------------------------------------
            # Annotation box
            # ------------------------------------------------
            score_col  = flag_col.replace("_flag", "_score")
            score_val  = float(data.loc[detection_date, score_col]) if score_col in data.columns else np.nan
            future_ret = float(data.loc[detection_date, "future_return"]) if "future_return" in data.columns else np.nan
            label_val  = int(data.loc[detection_date, "label"]) if "label" in data.columns else None
            label_str  = {1: "LONG ▲", -1: "SHORT ▼", 0: "NO TRADE —"}.get(label_val, "?")
            label_color = {1: BULL_COLOR, -1: BEAR_COLOR, 0: "#aaaaaa"}.get(label_val, TEXT_COLOR)
            horizon    = data.attrs.get("horizon", 10)

            ann_text = (
                f"Pattern : {pattern_name.replace('_', ' ')}\n"
                f"Score   : {score_val:.2f}\n"
                f"Fwd {horizon}d  : {future_ret * 100:+.1f}%\n"
                f"ML Label: {label_str}"
            )
            ax.annotate(
                ann_text,
                xy=(0.01, 0.97),
                xycoords="axes fraction",
                fontsize=10,
                va="top", ha="left",
                color=TEXT_COLOR,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR,
                          edgecolor=label_color, alpha=0.92),
                zorder=10,
            )

            # ------------------------------------------------
            # Title
            # ------------------------------------------------
            title_color = BULL_COLOR if direction == "bullish" else BEAR_COLOR
            title_dir   = "BULLISH ▲" if direction == "bullish" else "BEARISH ▼"
            fig.suptitle(
                f"{ticker}  |  1D  |  {pattern_name.replace('_', ' ')}  "
                f"({title_dir})  |  {str(detection_date)[:10]}  "
                f"|  Example {n}/{len(selected)}",
                fontsize=12, fontweight="bold",
                color=title_color, y=0.998,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.995])

            # ------------------------------------------------
            # Save directly — never show
            # ------------------------------------------------
            filepath = f"{output_prefix}_pattern_{pattern_name}_ex{n:02d}.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
            plt.close(fig)
            print(f"    Saved: {filepath}")

# ============================================================
# 15. MAIN FRAMEWORK
# ============================================================
def run_named_chart_pattern_ml_framework(
    ticker,
    start_date=START_DATE,
    end_date=END_DATE,
    train_ratio=TRAIN_RATIO,
    prediction_horizon=PREDICTION_HORIZON,
    label_threshold=LABEL_THRESHOLD,
    transaction_cost_bps=TRANSACTION_COST_BPS,
    show_plots=SHOW_PLOTS,
):

    output_dir = create_output_folder()
    annualization = get_annualization(ticker)

    print("=" * 110)
    print("NAMED CHART PATTERN ML TRADING FRAMEWORK")
    print("=" * 110)
    print(f"Ticker: {ticker}")
    print(f"Date range: {start_date} to {end_date}")
    print("Data source: Yahoo Finance daily OHLCV")
    print("ML model: SVM with RBF kernel")
    print("Named patterns: Head & Shoulders, Inverse H&S, Double Top, Double Bottom, Triangle Breakout/Breakdown")
    print(f"Train/Test: {int(train_ratio * 100)}% train / {int((1 - train_ratio) * 100)}% OOS test")
    print(f"Annualization: {annualization}")
    print(f"Output folder: {output_dir}")
    print("=" * 110)

    raw = download_daily_data(ticker, start_date, end_date)
    data = add_standard_technical_features(raw)
    data = add_named_chart_pattern_features(
        data,
        confirm_bars=PIVOT_CONFIRM_BARS,
        lookback_pivots=PATTERN_LOOKBACK_PIVOTS,
    )
    data = add_labels(data, horizon=prediction_horizon, threshold=label_threshold)
    data.attrs["horizon"] = prediction_horizon      # <-- add this line right after add_labels

    standard_feature_cols = [
        "ret_5d", "ret_10d", "ret_20d",
        "price_above_ema100", "ema20_slope_5d", "ema50_slope_10d",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "atr_pct", "bb_width_20", "zscore_30",
        "distance_to_20d_high", "distance_to_20d_low",
        "breakout_20d", "breakdown_20d", "drawdown_from_20d_high",
        "volume_zscore_20", "volume_ratio_20",
    ]

    named_pattern_feature_cols = [
        "head_shoulders_score",
        "inverse_head_shoulders_score",
        "double_top_score",
        "double_bottom_score",
        "triangle_contraction_score",
        "triangle_breakout_up_score",
        "triangle_breakdown_down_score",
        "bullish_pattern_score",
        "bearish_pattern_score",
    ]

    pattern_flag_cols = [
        "head_shoulders_flag",
        "inverse_head_shoulders_flag",
        "double_top_flag",
        "double_bottom_flag",
        "triangle_flag",
        "triangle_breakout_up_flag",
        "triangle_breakdown_down_flag",
    ]

    feature_cols = standard_feature_cols + named_pattern_feature_cols

    data = data.dropna(
        subset=feature_cols + ["label", "future_return", "atr_14", "ema_100"]
    ).copy()

    if len(data) < 300:
        raise ValueError("Not enough usable rows after feature engineering. Use a longer date range.")

    split_idx = int(len(data) * train_ratio)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    print("\nTrain period:")
    print(train_df.index[0], "to", train_df.index[-1])
    print("\nOOS test period:")
    print(test_df.index[0], "to", test_df.index[-1])

    print("\nLabel distribution - Train:")
    print(y_train.value_counts(normalize=True).sort_index())
    print("\nLabel distribution - Test:")
    print(y_test.value_counts(normalize=True).sort_index())

    print("\nNamed chart-pattern detection counts - Train:")
    print(train_df[pattern_flag_cols].sum().astype(int))
    print("\nNamed chart-pattern detection counts - Test:")
    print(test_df[pattern_flag_cols].sum().astype(int))

    train_pattern_diag = pattern_diagnostics(train_df, pattern_flag_cols)
    test_pattern_diag = pattern_diagnostics(test_df, pattern_flag_cols)

    print("\nTrain pattern diagnostics:")
    print(train_pattern_diag)
    print("\nOOS test pattern diagnostics:")
    print(test_pattern_diag)

    # --------------------------------------------------------
    # SVM-RBF model
    # --------------------------------------------------------
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "model__C": [0.5, 1, 5, 10],
        "model__gamma": ["scale", 0.01, 0.05, 0.10],
    }

    cv_splits = 4 if len(train_df) >= 800 else 3

    grid = GridSearchCV(
        estimator=svm_pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=TimeSeriesSplit(n_splits=cv_splits),
        n_jobs=-1,
    )

    print("\nTraining SVM-RBF on technical + named-pattern features...")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("\nBest SVM parameters:")
    print(grid.best_params_)
    print(f"Best train CV F1 Macro: {grid.best_score_:.4f}")

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    print("\nTraining classification report:")
    print(classification_report(
        y_train,
        train_pred,
        labels=[-1, 0, 1],
        target_names=["Short", "No Trade", "Long"],
        zero_division=0,
    ))

    print("\nOOS test classification report:")
    print(classification_report(
        y_test,
        test_pred,
        labels=[-1, 0, 1],
        target_names=["Short", "No Trade", "Long"],
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, test_pred, labels=[-1, 0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Short", "Actual No Trade", "Actual Long"],
        columns=["Pred Short", "Pred No Trade", "Pred Long"],
    )
    print("\nOOS confusion matrix:")
    print(cm_df)

    # --------------------------------------------------------
    # Optimize trading parameters on train only
    # --------------------------------------------------------
    print("\nOptimizing trading rules on training sample only...")
    best_trade_params, threshold_results = optimize_trading_thresholds(
        train_df=train_df,
        model=best_model,
        X_train=X_train,
        annualization=annualization,
        transaction_cost_bps=transaction_cost_bps,
    )

    print("\nBest frozen trading parameters:")
    for key in [
        "entry_threshold", "exit_threshold", "max_holding_days",
        "atr_stop_mult", "atr_take_profit_mult", "min_pattern_score",
    ]:
        print(f"{key}: {best_trade_params[key]}")

    # --------------------------------------------------------
    # Generate train/test probabilities and positions
    # --------------------------------------------------------
    p_train_long, p_train_short = get_class_probabilities(best_model, X_train)
    p_test_long, p_test_short = get_class_probabilities(best_model, X_test)

    train_position = build_positions(
        train_df,
        p_train_long,
        p_train_short,
        entry_threshold=best_trade_params["entry_threshold"],
        exit_threshold=best_trade_params["exit_threshold"],
        max_holding_days=int(best_trade_params["max_holding_days"]),
        atr_stop_mult=best_trade_params["atr_stop_mult"],
        atr_take_profit_mult=best_trade_params["atr_take_profit_mult"],
        require_pattern=REQUIRE_PATTERN_FOR_ENTRY,
        min_pattern_score=best_trade_params["min_pattern_score"],
        use_ema_filter=USE_EMA_TREND_FILTER,
    )

    test_position = build_positions(
        test_df,
        p_test_long,
        p_test_short,
        entry_threshold=best_trade_params["entry_threshold"],
        exit_threshold=best_trade_params["exit_threshold"],
        max_holding_days=int(best_trade_params["max_holding_days"]),
        atr_stop_mult=best_trade_params["atr_stop_mult"],
        atr_take_profit_mult=best_trade_params["atr_take_profit_mult"],
        require_pattern=REQUIRE_PATTERN_FOR_ENTRY,
        min_pattern_score=best_trade_params["min_pattern_score"],
        use_ema_filter=USE_EMA_TREND_FILTER,
    )

    train_bt = backtest_from_position(
        train_df,
        train_position,
        p_train_long,
        p_train_short,
        transaction_cost_bps=transaction_cost_bps,
    )

    test_bt = backtest_from_position(
        test_df,
        test_position,
        p_test_long,
        p_test_short,
        transaction_cost_bps=transaction_cost_bps,
    )

    train_metrics = performance_metrics(train_bt, annualization=annualization)
    test_metrics = performance_metrics(test_bt, annualization=annualization)
    metrics_table = pd.DataFrame([train_metrics, test_metrics], index=["Train 70%", "OOS Test 30%"])

    print("\nPerformance Summary:")
    print(metrics_table)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    output_prefix = output_dir / f"{ticker.replace('-', '_')}_Named_Chart_Pattern_ML"

    data.to_csv(f"{output_prefix}_full_feature_dataset.csv")
    train_bt.to_csv(f"{output_prefix}_train_positions_backtest.csv")
    test_bt.to_csv(f"{output_prefix}_oos_test_positions_backtest.csv")
    threshold_results.to_csv(f"{output_prefix}_train_threshold_search.csv", index=False)
    metrics_table.to_csv(f"{output_prefix}_performance_summary.csv")
    cm_df.to_csv(f"{output_prefix}_oos_confusion_matrix.csv")
    train_pattern_diag.to_csv(f"{output_prefix}_train_pattern_diagnostics.csv", index=False)
    test_pattern_diag.to_csv(f"{output_prefix}_oos_pattern_diagnostics.csv", index=False)

    plot_equity_curves(train_bt, test_bt, ticker, output_prefix)
    plot_pattern_events(test_bt, ticker, output_prefix)

    print("\nGenerating individual pattern example charts...")
    plot_individual_pattern_examples(data, ticker, output_prefix, max_examples=5)

    print("\nFiles saved successfully to:")
    print(output_dir)

    print("\nKey saved files:")
    print(f"{output_prefix}_full_feature_dataset.csv")
    print(f"{output_prefix}_train_positions_backtest.csv")
    print(f"{output_prefix}_oos_test_positions_backtest.csv")
    print(f"{output_prefix}_performance_summary.csv")
    print(f"{output_prefix}_train_pattern_diagnostics.csv")
    print(f"{output_prefix}_oos_pattern_diagnostics.csv")
    print(f"{output_prefix}_train_equity_curve.png")
    print(f"{output_prefix}_test_equity_curve.png")
    print(f"{output_prefix}_full_train_test_equity_curve.png")
    print(f"{output_prefix}_oos_named_pattern_events.png")

    return {
        "data": data,
        "train_backtest": train_bt,
        "test_backtest": test_bt,
        "best_model": best_model,
        "best_trade_params": best_trade_params,
        "threshold_results": threshold_results,
        "metrics_table": metrics_table,
        "confusion_matrix": cm_df,
        "train_pattern_diagnostics": train_pattern_diag,
        "test_pattern_diagnostics": test_pattern_diag,
        "output_dir": output_dir,
    }

# ============================================================
# 16. RUN
# ============================================================

if __name__ == "__main__":
    all_results = {}
    all_metrics = []

    for ticker in TICKERS:
        print(f"\n{'#' * 110}")
        print(f"# RUNNING: {ticker}")
        print(f"{'#' * 110}")

        try:
            results = run_named_chart_pattern_ml_framework(
                ticker=ticker,
                start_date=START_DATE,
                end_date=END_DATE,
                train_ratio=TRAIN_RATIO,
                prediction_horizon=PREDICTION_HORIZON,
                label_threshold=LABEL_THRESHOLD,
                transaction_cost_bps=TRANSACTION_COST_BPS,
                show_plots=SHOW_PLOTS,
            )
            all_results[ticker] = results

            metrics_tagged = results["metrics_table"].copy()
            metrics_tagged.insert(0, "Ticker", ticker)
            all_metrics.append(metrics_tagged)

        except Exception as e:
            print(f"\nSKIPPED {ticker}: {e}")
            continue

    if all_metrics:
        summary = pd.concat(all_metrics)
        print("\n" + "=" * 110)
        print("CROSS-TICKER PERFORMANCE SUMMARY")
        print("=" * 110)
        print(summary.to_string())

        output_dir = create_output_folder()
        summary.to_csv(output_dir / "ALL_TICKERS_performance_summary.csv")
        print(f"\nCross-ticker summary saved to: {output_dir / 'ALL_TICKERS_performance_summary.csv'}")