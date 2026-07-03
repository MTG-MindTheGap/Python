"""
insiderflow_app.py — InsiderFlow Dashboard (Full Version)
==============================================
- Insider buys/sells from SEC EDGAR Form 4 filings
- Full history (up to IPO) via paginated EDGAR requests
- Price chart from Yahoo Finance (free, no API key)
- Insider transactions plotted as markers on the price chart

Run:
    python3 insiderflow_app.py

Dependencies are installed automatically on first run.
Open: http://127.0.0.1:8090
"""

import os
import sys
import subprocess

# ── Run relative to this file's own location, regardless of caller's cwd ──────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Auto-install missing dependencies so this never breaks on a fresh machine ─
def _ensure_deps():
    required = {"flask": "flask", "requests": "requests"}
    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[*] Installing missing dependencies: {', '.join(missing)}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

_ensure_deps()

from flask import Flask, jsonify, render_template_string, request
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import threading
import re
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

app = Flask(__name__)

# ── Logging ───────────────────────────────────────────────────────────────────
# Rotates at ~2MB, keeps the last 3 files, and mirrors everything to the
# console too (see README "LOGGING AND TROUBLESHOOTING").

logger = logging.getLogger("insiderflow")
logger.setLevel(logging.INFO)
_file_handler = RotatingFileHandler("finance_tools.log", maxBytes=2 * 1024 * 1024, backupCount=3)
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)
logger.propagate = False

# ── Config ────────────────────────────────────────────────────────────────────

SEC_HEADERS = {
    "User-Agent": "InsiderFlow research@insiderflow.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html, */*",
}

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_ATOM_URL      = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=100&output=atom"
EDGAR_ATOM_PAGE_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb={dateb}&owner=include&count=100&output=atom&start={start}"
YAHOO_CHART_URL     = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# Local Ollama, used for the chat assistant (optional — dashboard works without it)
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

TX_LABELS = {
    "P": ("Buy",         "buy"),
    "S": ("Sell",        "sell"),
    "A": ("Award",       "award"),
    "M": ("Option Exec", "option"),
    "G": ("Gift",        "gift"),
    "F": ("Tax",         "tax"),
    "D": ("Dispose",     "sell"),
    "I": ("Indirect",    "neutral"),
    "J": ("Other",       "neutral"),
}

_cache      = {}
_cache_lock = threading.Lock()
_ticker_map = {}


# ── Ticker map ────────────────────────────────────────────────────────────────

def load_ticker_map():
    global _ticker_map
    if _ticker_map:
        return _ticker_map
    logger.info("Loading SEC ticker map...")
    resp = requests.get(COMPANY_TICKERS_URL, headers=SEC_HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    _ticker_map = {
        v["ticker"].upper(): {
            "cik":   str(v["cik_str"]).zfill(10),
            "title": v["title"],
        }
        for v in data.values() if v.get("ticker")
    }
    logger.info(f"Loaded {len(_ticker_map)} tickers")
    return _ticker_map


# ── Fetch ALL Form 4 accessions (paginated) ───────────────────────────────────

def get_all_form4_accessions(cik: str, days: Optional[int] = None) -> list:
    """
    Fetch Form 4 filing list. If days=None or days=0, fetches ALL filings
    since the company's first filing (full history). Otherwise limits by date.
    """
    cutoff = None
    if days and days > 0:
        cutoff = datetime.today() - timedelta(days=days)

    all_accessions = []
    start = 0

    while True:
        url = EDGAR_ATOM_PAGE_URL.format(cik=cik, dateb="", start=start)
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed fetching page at start={start}: {e}")
            break

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            break

        entries = root.findall("atom:entry", ns)
        if not entries:
            break  # No more pages

        page_accessions = []
        oldest_date = None

        for entry in entries:
            updated = entry.find("atom:updated", ns)

            if updated is None or not updated.text:
                continue
            date_str = updated.text[:10]
            try:
                filing_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Bad date format: {date_str!r}")
                continue

            oldest_date = filing_date  # entries are newest-first

            # If cutoff set and we've gone past it, stop
            if cutoff and filing_date < cutoff:
                break

            for link in entry.findall("atom:link", ns):
                href = link.get("href", "")
                m    = re.search(r'(\d{10}-\d{2}-\d{6})', href)
                if m:
                    page_accessions.append({
                        "accession": m.group(1),
                        "date":      date_str,
                        "cik":       cik,
                    })
                    break

        all_accessions.extend(page_accessions)

        # Stop conditions
        if cutoff and oldest_date and oldest_date < cutoff:
            break
        if len(entries) < 100:
            break  # Last page

        start += 100
        time.sleep(0.1)

    logger.info(f"Total Form 4 accessions found: {len(all_accessions)}")
    return all_accessions


# ── Parse Form 4 XML ──────────────────────────────────────────────────────────

def parse_form4(accession: str, cik: str, company_name: str) -> list:
    acc_clean = accession.replace("-", "")
    cik_int   = int(cik)

    # Fetch index to find actual XML filename. SEC occasionally rate-limits
    # this request with a non-200 response whose body has no XML links, which
    # would silently fall through to a guessed filename that often 404s — so
    # retry once after a brief backoff before giving up and guessing.
    idx_url  = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{accession}-index.htm"
    fallback = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{accession}.xml"
    xml_url  = fallback

    for attempt in range(2):
        try:
            idx_resp = requests.get(idx_url, headers=SEC_HEADERS, timeout=10)
            if idx_resp.status_code != 200:
                raise requests.exceptions.HTTPError(f"status {idx_resp.status_code}")
            xml_files = re.findall(r'href="(/Archives/[^"]+\.xml)"', idx_resp.text)
            # Exclude stylesheet wrapper, keep raw XML
            xml_files = [f for f in xml_files if "FilingSummary" not in f and "xslF345X05" not in f]
            if xml_files:
                xml_url = "https://www.sec.gov" + xml_files[0]
            break
        except Exception:
            if attempt == 0:
                time.sleep(0.5)
            # second failure: keep the guessed fallback URL

    try:
        resp = requests.get(xml_url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Could not fetch {accession}: {e}")
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    # Issuer
    ticker = ""
    issuer_name = company_name
    el = root.find(".//issuerTradingSymbol")
    if el is not None: ticker = (el.text or "").strip().upper()
    el = root.find(".//issuerName")
    if el is not None: issuer_name = (el.text or company_name).strip()

    # Owner
    owner_name = owner_title = ""
    is_director = is_officer = is_ten_pct = False
    for owner in root.findall(".//reportingOwner"):
        n = owner.find(".//rptOwnerName")
        t = owner.find(".//officerTitle")
        d = owner.find(".//isDirector")
        o = owner.find(".//isOfficer")
        p = owner.find(".//isTenPercentOwner")
        if n is not None: owner_name  = (n.text or "").strip()
        if t is not None: owner_title = (t.text or "").strip()
        if d is not None and d.text == "1": is_director = True
        if o is not None and o.text == "1": is_officer  = True
        if p is not None and p.text == "1": is_ten_pct  = True

    if not owner_title:
        if is_officer:    owner_title = "Officer"
        elif is_director: owner_title = "Director"
        elif is_ten_pct:  owner_title = "10% Owner"

    transactions = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code_el   = txn.find(".//transactionCode")
        date_el   = txn.find(".//transactionDate/value")
        shares_el = txn.find(".//transactionShares/value")
        price_el  = txn.find(".//transactionPricePerShare/value")
        owned_el  = txn.find(".//sharesOwnedFollowingTransaction/value")
        disp_el   = txn.find(".//transactionAcquiredDisposedCode/value")

        if code_el is None or date_el is None:
            continue

        code     = (code_el.text or "").strip()
        disposed = (disp_el.text or "").strip() if disp_el is not None else ""

        try:
            shares = float(shares_el.text) if shares_el is not None and shares_el.text else 0
            price  = float(price_el.text)  if price_el  is not None and price_el.text  else 0
            owned  = float(owned_el.text)  if owned_el  is not None and owned_el.text  else 0
        except ValueError:
            shares = price = owned = 0

        label, tx_type = TX_LABELS.get(code, ("Unknown", "neutral"))
        if code == "S" or (disposed == "D" and code not in ("A", "M", "G", "J", "F")):
            tx_type, label = "sell", "Sell"
        elif code == "P":
            tx_type, label = "buy", "Buy"

        transactions.append({
            "date":        date_el.text or "",
            "ticker":      ticker,
            "company":     issuer_name,
            "insider":     owner_name,
            "title":       owner_title,
            "tx_type":     tx_type,
            "tx_label":    label,
            "tx_code":     code,
            "shares":      shares,
            "price":       price,
            "value":       shares * price,
            "owned_after": owned,
            "accession":   accession,
            "filing_url":  f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{accession}-index.htm",
        })

    return transactions


# ── Fetch all transactions for a ticker ───────────────────────────────────────

def fetch_for_ticker(ticker: str, days: int = 90) -> list:
    m     = load_ticker_map()
    entry = m.get(ticker.upper())
    if not entry:
        return []

    cik, company = entry["cik"], entry["title"]
    logger.info(f"Fetching Form 4s for {ticker} (CIK {cik}, days={days})")

    accessions = get_all_form4_accessions(cik, days=days if days > 0 else None)
    logger.info(f"Parsing {len(accessions)} filings...")

    all_txns = []
    for i, item in enumerate(accessions):
        txns = parse_form4(item["accession"], cik, company)
        for t in txns:
            if not t["ticker"]: t["ticker"] = ticker.upper()
            if not t["date"]:   t["date"]   = item["date"]
        all_txns.extend(txns)
        time.sleep(0.11)
        if (i + 1) % 20 == 0:
            logger.info(f"  ... {i+1}/{len(accessions)} parsed, {len(all_txns)} txns so far")

    all_txns.sort(key=lambda x: x.get("date", ""), reverse=True)
    logger.info(f"{ticker}: {len(all_txns)} total transactions")
    return all_txns


# ── Yahoo Finance price data ───────────────────────────────────────────────────

def fetch_price_history(ticker: str, days: int = 365) -> dict:
    """
    Fetch OHLC price history from Yahoo Finance.
    Returns {timestamps: [...], closes: [...], highs: [...], lows: [...]}
    """
    # Map days to Yahoo interval/range
    if days <= 30:
        interval, range_ = "1d", "1mo"
    elif days <= 90:
        interval, range_ = "1d", "3mo"
    elif days <= 180:
        interval, range_ = "1d", "6mo"
    elif days <= 365:
        interval, range_ = "1d", "1y"
    elif days <= 730:
        interval, range_ = "1wk", "2y"
    elif days <= 1825:
        interval, range_ = "1wk", "5y"
    else:
        interval, range_ = "1mo", "max"  # Full history

    url = YAHOO_CHART_URL.format(ticker=ticker)
    try:
        resp = requests.get(
            url,
            params={"interval": interval, "range": range_},
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
        highs  = ohlcv.get("high",  [])
        lows   = ohlcv.get("low",   [])

        # Clean None values
        cleaned = [(d, c, h, l) for d, c, h, l in zip(dates, closes, highs, lows)
                   if c is not None]

        return {
            "dates":  [r[0] for r in cleaned],
            "closes": [round(r[1], 2) for r in cleaned],
            "highs":  [round(r[2], 2) for r in cleaned],
            "lows":   [round(r[3], 2) for r in cleaned],
        }
    except Exception as e:
        logger.error(f"Price fetch failed for {ticker}: {e}")
        return {"dates": [], "closes": [], "highs": [], "lows": []}


# ── Cache ─────────────────────────────────────────────────────────────────────

def get_cached(key, ttl, fetch_fn):
    with _cache_lock:
        e = _cache.get(key)
        if e and (time.time() - e["ts"]) < ttl:
            return e["data"]
    data = fetch_fn()
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}
    return data


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/ticker/<ticker>")
def api_ticker(ticker):
    days      = int(request.args.get("days", 90))
    tx_filter = request.args.get("type", "all")
    min_value = float(request.args.get("min_value", 0))

    # days=0 means "all history"
    cache_key = f"ticker_{ticker.upper()}_{days}"
    data      = get_cached(cache_key, 600, lambda: fetch_for_ticker(ticker.upper(), days))

    result = [r for r in data
              if (tx_filter == "all" or r["tx_type"] == tx_filter)
              and r["value"] >= min_value]
    return jsonify({"ticker": ticker.upper(), "count": len(result), "rows": result})


@app.route("/api/price/<ticker>")
def api_price(ticker):
    days  = int(request.args.get("days", 365))
    data  = get_cached(
        f"price_{ticker.upper()}_{days}",
        300,
        lambda: fetch_price_history(ticker.upper(), days)
    )
    return jsonify(data)


def build_chat_context(ticker: str, days: int) -> str:
    if not ticker:
        return "No ticker is currently selected in the dashboard."

    txns = get_cached(f"ticker_{ticker}_{days}", 600, lambda: fetch_for_ticker(ticker, days))
    if not txns:
        return f"No Form 4 insider transactions were found for {ticker} in the selected time window."

    lines = [f"Recent SEC Form 4 insider transactions for {ticker} (most recent first):"]
    for t in txns[:40]:
        lines.append(
            f"- {t['date']}: {t['insider'] or 'Unknown insider'} ({t['title'] or 'insider'}) "
            f"{t['tx_label']} {t['shares']:,.0f} shares @ ${t['price']:.2f} "
            f"(${t['value']:,.0f}), owned after: {t['owned_after']:,.0f}"
        )
    return "\n".join(lines)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body     = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    ticker   = (body.get("ticker") or "").strip().upper()
    days     = int(body.get("days") or 90)

    if not question:
        return jsonify({"error": "Question is required."}), 400

    context = build_chat_context(ticker, days)
    system_prompt = (
        "You are InsiderFlow's assistant. Answer questions about corporate "
        "insider stock trading using ONLY the SEC Form 4 transaction data "
        "below. Be concise and specific (name insiders, dates, amounts). If "
        "the data doesn't answer the question, say so instead of guessing. "
        "This is not financial advice.\n\n" + context
    )

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        answer = (resp.json().get("message") or {}).get("content", "").strip()
        return jsonify({"answer": answer or "The assistant didn't return a response. Try rephrasing."})
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not reach Ollama at {OLLAMA_HOST}")
        return jsonify({
            "error": f"Couldn't reach Ollama at {OLLAMA_HOST}. Install it from "
                     f"https://ollama.com, run `ollama pull {OLLAMA_MODEL}`, and make "
                     f"sure Ollama is running, then try again."
        }), 503
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        return jsonify({"error": "Chat request failed. Check finance_tools.log for details."}), 500


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").upper().strip()
    if not q:
        return jsonify([])
    m       = load_ticker_map()
    results = [
        {"ticker": k, "name": v["title"]}
        for k, v in m.items()
        if q in k or q.lower() in v["title"].lower()
    ][:20]
    return jsonify(results)


# ── HTML Template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>InsiderFlow</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --bg:#080a0f;--surface:#0e1118;--card:#12161f;--border:#1c2130;--border2:#252d3d;
  --buy:#00d964;--buy-dim:rgba(0,217,100,0.12);--sell:#ff3b5c;--sell-dim:rgba(255,59,92,0.12);
  --text:#dde3f0;--dim:#5b6b8a;--accent:#4d7cff;--yellow:#f59e0b;
  --mono:'DM Mono',monospace;--display:'Syne',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;min-height:100vh}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(77,124,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(77,124,255,0.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}

/* Header */
header{position:sticky;top:0;z-index:100;background:rgba(8,10,15,0.95);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);padding:0 32px;height:56px;display:flex;align-items:center;justify-content:space-between}
.logo{font-family:var(--display);font-weight:800;font-size:18px;letter-spacing:-0.02em;color:#fff;display:flex;align-items:center;gap:10px}
.logo-badge{width:28px;height:28px;background:linear-gradient(135deg,#4d7cff,#7c5cff);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;font-family:var(--mono)}
.live-dot{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--dim);letter-spacing:0.06em}
.live-dot::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--buy);box-shadow:0 0 8px var(--buy);animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}

/* Main */
main{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:24px 32px}

/* Stats */
.stats-bar{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 18px;position:relative;overflow:hidden;transition:border-color 0.2s}
.stat-card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.stat-card.all-card::after{background:#a855f7}.stat-card.buy-card::after{background:var(--buy)}.stat-card.sell-card::after{background:var(--sell)}.stat-card.vol-card::after{background:var(--accent)}
.stat-label{font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--dim);margin-bottom:5px}
.stat-value{font-family:var(--display);font-size:22px;font-weight:700;color:#fff;line-height:1}
.stat-sub{font-size:10px;color:var(--dim);margin-top:3px}

/* Controls */
.controls{display:flex;align-items:center;gap:10px;margin-bottom:16px;flex-wrap:wrap}
.search-wrap{position:relative;flex:1;min-width:220px;max-width:340px}
.search-wrap svg{position:absolute;left:11px;top:50%;transform:translateY(-50%);color:var(--dim);pointer-events:none}
.search-input{width:100%;padding:9px 12px 9px 34px;background:var(--card);border:1px solid var(--border);border-radius:7px;color:var(--text);font-family:var(--mono);font-size:12px;outline:none;transition:border-color 0.2s}
.search-input:focus{border-color:var(--accent)}.search-input::placeholder{color:var(--dim)}
.filter-group{display:flex;gap:4px}
.filter-btn{padding:7px 12px;background:var(--card);border:1px solid var(--border);border-radius:6px;color:var(--dim);font-family:var(--mono);font-size:11px;cursor:pointer;transition:all 0.15s}
.filter-btn.active{background:var(--border2);color:#fff;border-color:var(--border2)}
.filter-btn.buy-btn.active{background:var(--buy-dim);color:var(--buy);border-color:var(--buy)}
.filter-btn.sell-btn.active{background:var(--sell-dim);color:var(--sell);border-color:var(--sell)}
.ctrl-select{padding:7px 10px;background:var(--card);border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:var(--mono);font-size:11px;outline:none;cursor:pointer}
.min-val-wrap{display:flex;align-items:center;gap:6px}
.min-val-label{font-size:10px;color:var(--dim);white-space:nowrap}
.refresh-btn{padding:7px 14px;background:var(--accent);border:none;border-radius:6px;color:#fff;font-family:var(--mono);font-size:11px;cursor:pointer;display:flex;align-items:center;gap:5px;transition:opacity 0.15s}
.refresh-btn:hover{opacity:0.85}.refresh-btn.loading{opacity:0.5;pointer-events:none}

/* Chart section */
.chart-section{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px;display:none}
.chart-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.chart-title{font-family:var(--display);font-size:15px;font-weight:700;color:#fff}
.chart-meta{display:flex;align-items:center;gap:16px;font-size:11px;color:var(--dim)}
.chart-price{font-family:var(--display);font-size:22px;font-weight:700;color:#fff}
.chart-change{font-size:12px;padding:2px 7px;border-radius:4px}
.chart-change.pos{background:var(--buy-dim);color:var(--buy)}.chart-change.neg{background:var(--sell-dim);color:var(--sell)}
.chart-legend{display:flex;gap:14px;font-size:11px;color:var(--dim)}
.legend-item{display:flex;align-items:center;gap:5px}
.legend-dot{width:8px;height:8px;border-radius:50%}
.chart-wrap{position:relative;height:360px}
.chart-loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--dim);font-size:12px;gap:8px}

/* Table */
.table-wrap{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.table-header{padding:14px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.table-title{font-family:var(--display);font-size:14px;font-weight:600;color:#fff}
.table-count{font-size:11px;color:var(--dim)}
.data-table{width:100%;border-collapse:collapse}
.data-table thead th{padding:9px 14px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--dim);border-bottom:1px solid var(--border);white-space:nowrap;font-weight:500}
.data-table thead th:first-child{padding-left:20px}
.data-table tbody tr{border-bottom:1px solid var(--border);transition:background 0.1s;animation:fadeIn 0.25s ease both}
.data-table tbody tr:last-child{border-bottom:none}
.data-table tbody tr:hover{background:rgba(255,255,255,0.025)}
@keyframes fadeIn{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:translateY(0)}}
.data-table tbody td{padding:10px 14px;vertical-align:middle;white-space:nowrap}
.data-table tbody td:first-child{padding-left:20px}
tr.row-buy{border-left:2px solid var(--buy)}tr.row-sell{border-left:2px solid var(--sell)}
.cell-date{color:var(--dim);font-size:11px}
.cell-ticker{font-family:var(--display);font-weight:700;font-size:13px;color:#fff}
.cell-company{color:var(--dim);font-size:11px;max-width:160px;overflow:hidden;text-overflow:ellipsis}
.cell-insider{font-size:12px}.cell-role{font-size:10px;color:var(--dim);margin-top:1px}
.tx-badge{display:inline-flex;align-items:center;padding:3px 8px;border-radius:4px;font-size:10px;font-weight:500;letter-spacing:0.04em}
.tx-buy{background:var(--buy-dim);color:var(--buy)}.tx-sell{background:var(--sell-dim);color:var(--sell)}.tx-neutral{background:rgba(91,107,138,0.15);color:var(--dim)}
.cell-shares,.cell-price,.cell-value,.cell-owned{text-align:right}
.cell-value{font-weight:500}.value-buy{color:var(--buy)}.value-sell{color:var(--sell)}
.cell-owned{font-size:11px;color:var(--dim)}
.cell-filing a{color:var(--accent);text-decoration:none;font-size:11px;opacity:0.6}
.cell-filing a:hover{opacity:1}
.loading-row td{padding:50px 20px;text-align:center;color:var(--dim)}
.spinner{width:22px;height:22px;border:2px solid var(--border2);border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 10px}
@keyframes spin{to{transform:rotate(360deg)}}
.empty-state{padding:60px 20px;text-align:center;color:var(--dim);line-height:2}
.empty-icon{font-size:36px;margin-bottom:10px}
.error-banner{background:rgba(255,59,92,0.1);border:1px solid rgba(255,59,92,0.3);border-radius:8px;padding:11px 16px;margin-bottom:14px;font-size:12px;color:var(--sell);display:none}
.search-dropdown{position:absolute;top:calc(100% + 4px);left:0;right:0;background:var(--card);border:1px solid var(--border2);border-radius:8px;overflow:hidden;z-index:200;box-shadow:0 8px 32px rgba(0,0,0,0.5);display:none}
.search-dropdown.show{display:block}
.dropdown-item{padding:9px 14px;cursor:pointer;display:flex;align-items:center;gap:10px;transition:background 0.1s}
.dropdown-item:hover{background:var(--border)}
.dropdown-ticker{font-weight:500;color:#fff;font-size:12px;min-width:52px}
.dropdown-name{font-size:11px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.history-note{font-size:10px;color:var(--dim);padding:8px 20px;border-top:1px solid var(--border)}
@media(max-width:900px){.stats-bar{grid-template-columns:repeat(2,1fr)}main{padding:14px}header{padding:0 14px}.cell-company,.cell-owned,.cell-filing{display:none}.chart-wrap{height:240px}}

/* Chat widget */
.chat-toggle{position:fixed;bottom:24px;right:24px;z-index:300;width:48px;height:48px;border-radius:50%;background:var(--accent);border:none;color:#fff;font-size:20px;cursor:pointer;box-shadow:0 4px 20px rgba(77,124,255,0.4);display:flex;align-items:center;justify-content:center}
.chat-toggle:hover{opacity:0.9}
.chat-panel{position:fixed;bottom:84px;right:24px;z-index:300;width:340px;max-height:460px;background:var(--card);border:1px solid var(--border2);border-radius:12px;display:none;flex-direction:column;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,0.5)}
.chat-panel.show{display:flex}
.chat-header{padding:12px 14px;border-bottom:1px solid var(--border);font-family:var(--display);font-weight:700;font-size:13px;color:#fff;display:flex;align-items:center;justify-content:space-between}
.chat-header span.sub{font-family:var(--mono);font-weight:400;font-size:10px;color:var(--dim)}
.chat-messages{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:10px;min-height:180px}
.chat-msg{font-size:12px;line-height:1.5;padding:8px 10px;border-radius:8px;max-width:88%;white-space:pre-wrap}
.chat-msg.user{align-self:flex-end;background:var(--accent);color:#fff}
.chat-msg.bot{align-self:flex-start;background:var(--border);color:var(--text)}
.chat-msg.error{align-self:flex-start;background:var(--sell-dim);color:var(--sell)}
.chat-empty{color:var(--dim);font-size:11px;text-align:center;margin:auto}
.chat-input-row{display:flex;gap:6px;padding:10px;border-top:1px solid var(--border)}
.chat-input{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:8px 10px;color:var(--text);font-family:var(--mono);font-size:12px;outline:none}
.chat-input:focus{border-color:var(--accent)}
.chat-send{background:var(--accent);border:none;border-radius:6px;color:#fff;padding:0 12px;cursor:pointer;font-size:12px}
.chat-send:disabled{opacity:0.5;cursor:default}
</style>
</head>
<body>
<header>
  <div class="logo"><div class="logo-badge">IF</div>InsiderFlow</div>
  <div class="live-dot">LIVE · SEC EDGAR</div>
</header>
<main>

  <!-- Stats -->
  <div class="stats-bar">
    <div class="stat-card all-card"><div class="stat-label">Transactions</div><div class="stat-value" id="statTotal">—</div><div class="stat-sub" id="statSub">Search a ticker to begin</div></div>
    <div class="stat-card buy-card"><div class="stat-label">Buys</div><div class="stat-value" id="statBuys">—</div><div class="stat-sub" id="statBuyVal">—</div></div>
    <div class="stat-card sell-card"><div class="stat-label">Sells</div><div class="stat-value" id="statSells">—</div><div class="stat-sub" id="statSellVal">—</div></div>
    <div class="stat-card vol-card"><div class="stat-label">Total Volume</div><div class="stat-value" id="statVolume">—</div><div class="stat-sub">USD estimated</div></div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <div class="search-wrap">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
      <input class="search-input" id="searchInput" placeholder="Search ticker — AAPL, MU, NVDA, TSLA…" autocomplete="off">
      <div class="search-dropdown" id="searchDropdown"></div>
    </div>
    <div class="filter-group">
      <button class="filter-btn active" data-filter="all">All</button>
      <button class="filter-btn buy-btn" data-filter="buy">Buys</button>
      <button class="filter-btn sell-btn" data-filter="sell">Sells</button>
    </div>
    <select class="ctrl-select" id="daysSelect">
      <option value="30">30 days</option>
      <option value="60">60 days</option>
      <option value="90" selected>90 days</option>
      <option value="180">180 days</option>
      <option value="365">1 year</option>
      <option value="730">2 years</option>
      <option value="1825">5 years</option>
      <option value="0">Since IPO (all)</option>
    </select>
    <div class="min-val-wrap">
      <span class="min-val-label">MIN $</span>
      <select class="ctrl-select" id="minValSelect">
        <option value="0">Any</option>
        <option value="50000">$50K+</option>
        <option value="100000">$100K+</option>
        <option value="500000">$500K+</option>
        <option value="1000000">$1M+</option>
        <option value="10000000">$10M+</option>
      </select>
    </div>
    <button class="refresh-btn" id="refreshBtn">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/></svg>
      Refresh
    </button>
  </div>

  <!-- Error -->
  <div class="error-banner" id="errorBanner"></div>

  <!-- Price Chart -->
  <div class="chart-section" id="chartSection">
    <div class="chart-header">
      <div>
        <div class="chart-title" id="chartTitle">—</div>
        <div style="display:flex;align-items:center;gap:10px;margin-top:4px">
          <span class="chart-price" id="chartPrice">—</span>
          <span class="chart-change" id="chartChange"></span>
        </div>
      </div>
      <div>
        <div class="chart-legend">
          <div class="legend-item"><div class="legend-dot" style="background:#4d7cff"></div>Price</div>
          <div class="legend-item"><div class="legend-dot" style="background:#00d964"></div>Buy</div>
          <div class="legend-item"><div class="legend-dot" style="background:#ff3b5c"></div>Sell</div>
          <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Other</div>
        </div>
      </div>
    </div>
    <div class="chart-wrap">
      <div class="chart-loading" id="chartLoading">
        <div class="spinner" style="margin:0"></div> Loading price data…
      </div>
      <canvas id="priceChart"></canvas>
    </div>
  </div>

  <!-- Table -->
  <div class="table-wrap">
    <div class="table-header">
      <div class="table-title" id="tableTitle">Insider Transactions</div>
      <div class="table-count" id="tableCount">—</div>
    </div>
    <div style="overflow-x:auto">
      <table class="data-table">
        <thead><tr>
          <th>Date</th><th>Ticker</th><th>Company</th><th>Insider</th><th>Title</th>
          <th>Type</th><th style="text-align:right">Shares</th><th style="text-align:right">Price</th>
          <th style="text-align:right">Value</th><th style="text-align:right">Owned After</th><th>Filing</th>
        </tr></thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
    <div class="history-note" id="historyNote" style="display:none"></div>
  </div>

</main>

<!-- Chat widget -->
<button class="chat-toggle" id="chatToggle" title="Ask about insider activity">💬</button>
<div class="chat-panel" id="chatPanel">
  <div class="chat-header">
    <span>InsiderFlow Assistant</span>
    <span class="sub" id="chatSub">local · Ollama</span>
  </div>
  <div class="chat-messages" id="chatMessages">
    <div class="chat-empty">Search a ticker, then ask things like "who sold the most this month?"</div>
  </div>
  <div class="chat-input-row">
    <input class="chat-input" id="chatInput" placeholder="Ask about insider activity…" autocomplete="off">
    <button class="chat-send" id="chatSend">Send</button>
  </div>
</div>

<script>
let allData=[], activeFilter='all', currentTicker=null, searchDebounce=null;
let priceChart=null;

const fmtNum=n=>{if(!n&&n!==0)return'—';if(n>=1e9)return'$'+(n/1e9).toFixed(2)+'B';if(n>=1e6)return'$'+(n/1e6).toFixed(2)+'M';if(n>=1e3)return'$'+(n/1e3).toFixed(0)+'K';return'$'+n.toFixed(0)};
const fmtShares=n=>{if(!n)return'—';if(n>=1e6)return(n/1e6).toFixed(2)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return n.toLocaleString()};
const fmtPrice=p=>p?'$'+p.toFixed(2):'—';
const fmtDate=d=>d?d.slice(0,10):'—';

// Helper: build a DOM element without ever touching innerHTML with dynamic data
function el(tag, opts={}){
  const node = document.createElement(tag);
  if(opts.className) node.className = opts.className;
  if(opts.text !== undefined) node.textContent = opts.text;
  if(opts.title !== undefined) node.title = opts.title;
  if(opts.attrs) for(const [k,v] of Object.entries(opts.attrs)) node.setAttribute(k, v);
  if(opts.children) opts.children.forEach(c => c && node.appendChild(c));
  return node;
}
function clear(node){ while(node.firstChild) node.removeChild(node.firstChild); }

// ── Init ─────────────────────────────────────────────────────────────────────
showPrompt();

function showPrompt(){
  const tbody = document.getElementById('tableBody');
  clear(tbody);
  const row = el('tr');
  const td  = el('td', {attrs:{colspan:'11'}});
  const empty = el('div', {className:'empty-state'});
  empty.appendChild(el('div', {className:'empty-icon', text:'🔍'}));
  empty.appendChild(document.createTextNode('Search for a ticker to load insider transactions and price chart.'));
  empty.appendChild(document.createElement('br'));
  empty.appendChild(el('small', {attrs:{style:'font-size:11px'}, text:'Try MU · AAPL · NVDA · TSLA · MSFT'}));
  td.appendChild(empty);
  row.appendChild(td);
  tbody.appendChild(row);

  document.getElementById('tableCount').textContent='—';
  document.getElementById('tableTitle').textContent='Insider Transactions';
  document.getElementById('chartSection').style.display='none';
  document.getElementById('historyNote').style.display='none';
}

// ── Load insider data ─────────────────────────────────────────────────────────
async function loadData(ticker){
  if(!ticker) return;
  setLoading(true); hideError();
  const days=document.getElementById('daysSelect').value;
  try{
    const resp=await fetch(`/api/ticker/${encodeURIComponent(ticker)}?days=${encodeURIComponent(days)}`);
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    const json=await resp.json();
    allData=json.rows||[];
    document.getElementById('tableTitle').textContent=ticker+' — Insider Transactions';

    const daysLabel = days==0 ? 'all time' : `last ${days} days`;
    if(!allData.length){
      showError(`No Form 4 transactions found for ${ticker} (${daysLabel}). Try a longer time range.`);
    } else {
      const note = document.getElementById('historyNote');
      note.textContent = `Showing ${allData.length} transactions — ${daysLabel} · Data source: SEC EDGAR Form 4`;
      note.style.display='block';
    }
    applyFilters();
    updateStats();
  }catch(e){
    showError('Failed to load insider data: '+e.message);
  }finally{
    setLoading(false);
  }
}

// ── Load price chart ──────────────────────────────────────────────────────────
async function loadChart(ticker){
  const days = parseInt(document.getElementById('daysSelect').value) || 365;
  const chartDays = days === 0 ? 99999 : Math.max(days, 365); // show at least 1yr of price

  document.getElementById('chartSection').style.display='block';
  document.getElementById('chartLoading').style.display='flex';
  document.getElementById('chartTitle').textContent = ticker + ' — Price & Insider Activity';

  try{
    const resp = await fetch(`/api/price/${encodeURIComponent(ticker)}?days=${chartDays}`);
    const price = await resp.json();

    document.getElementById('chartLoading').style.display='none';

    if(!price.dates || !price.dates.length){
      const cl = document.getElementById('chartLoading');
      clear(cl);
      cl.appendChild(el('span', {text:'Price data unavailable'}));
      cl.style.display='flex';
      return;
    }

    // Latest price
    const latest = price.closes[price.closes.length-1];
    const prev   = price.closes[price.closes.length-2] || latest;
    const chg    = latest - prev;
    const chgPct = (chg/prev*100).toFixed(2);
    document.getElementById('chartPrice').textContent = '$'+latest.toFixed(2);
    const chgEl = document.getElementById('chartChange');
    chgEl.textContent = (chg>=0?'+':'')+chg.toFixed(2)+' ('+chgPct+'%)';
    chgEl.className = 'chart-change '+(chg>=0?'pos':'neg');

    renderChart(price, allData);
  }catch(e){
    console.error('Chart error:', e);
    const cl = document.getElementById('chartLoading');
    clear(cl);
    cl.appendChild(el('span', {text:'Could not load price data'}));
    cl.style.display='flex';
  }
}

// ── Render Chart.js chart ─────────────────────────────────────────────────────
function renderChart(price, transactions){
  if(priceChart){ priceChart.destroy(); priceChart=null; }

  const ctx = document.getElementById('priceChart').getContext('2d');

  const buyPoints  = [];
  const sellPoints = [];
  const otherPoints= [];

  transactions.forEach(t=>{
    const idx = price.dates.indexOf(t.date);
    if(idx === -1) return;
    const pt = {x: t.date, y: price.closes[idx],
                insider: t.insider, title: t.title,
                value: t.value, shares: t.shares, label: t.tx_label};
    if(t.tx_type==='buy')       buyPoints.push(pt);
    else if(t.tx_type==='sell') sellPoints.push(pt);
    else                        otherPoints.push(pt);
  });

  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: price.dates,
      datasets: [
        {
          label: 'Price',
          data: price.closes,
          borderColor: '#4d7cff',
          backgroundColor: 'rgba(77,124,255,0.06)',
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
          tension: 0.1,
          order: 4,
        },
        {
          label: 'Buy',
          data: buyPoints,
          type: 'scatter',
          borderColor: '#00d964',
          backgroundColor: '#00d964',
          pointRadius: 7,
          pointHoverRadius: 10,
          pointStyle: 'triangle',
          order: 1,
          parsing: {xAxisKey:'x', yAxisKey:'y'},
        },
        {
          label: 'Sell',
          data: sellPoints,
          type: 'scatter',
          borderColor: '#ff3b5c',
          backgroundColor: '#ff3b5c',
          pointRadius: 7,
          pointHoverRadius: 10,
          pointStyle: 'triangle',
          rotation: 180,
          order: 2,
          parsing: {xAxisKey:'x', yAxisKey:'y'},
        },
        {
          label: 'Other',
          data: otherPoints,
          type: 'scatter',
          borderColor: '#f59e0b',
          backgroundColor: '#f59e0b',
          pointRadius: 5,
          pointHoverRadius: 8,
          order: 3,
          parsing: {xAxisKey:'x', yAxisKey:'y'},
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#12161f',
          borderColor: '#252d3d',
          borderWidth: 1,
          titleColor: '#dde3f0',
          bodyColor: '#5b6b8a',
          padding: 10,
          callbacks: {
            label(ctx){
              if(ctx.dataset.label==='Price')
                return ' $'+ctx.parsed.y.toFixed(2);
              const p = ctx.raw;
              return [
                ` ${ctx.dataset.label}: ${p.insider||''}`,
                ` ${p.title||''}`,
                ` Shares: ${p.shares ? p.shares.toLocaleString() : '—'}`,
                ` Value: ${p.value ? '$'+(p.value/1e6).toFixed(2)+'M' : '—'}`,
              ].filter(Boolean);
            }
          }
        }
      },
      scales: {
        x: {
          type: 'category',
          ticks: {
            color: '#5b6b8a', font:{family:"'DM Mono'",size:10},
            maxTicksLimit: 10, maxRotation: 0,
          },
          grid: { color: 'rgba(255,255,255,0.03)' },
        },
        y: {
          position: 'right',
          ticks: { color:'#5b6b8a', font:{family:"'DM Mono'",size:10},
                   callback: v=>'$'+v.toFixed(0) },
          grid: { color:'rgba(255,255,255,0.04)' },
        }
      }
    }
  });
}

// ── Filters / stats ───────────────────────────────────────────────────────────
function reload(){
  if(currentTicker){ allData=[]; loadData(currentTicker); loadChart(currentTicker); }
}

function setFilter(f, btn){
  activeFilter=f;
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
  if(priceChart && allData.length) renderChart(
    {dates: priceChart.data.labels, closes: priceChart.data.datasets[0].data},
    allData.filter(r=>f==='all'||r.tx_type===f)
  );
}

function applyFilters(){
  const minVal=parseFloat(document.getElementById('minValSelect').value)||0;
  let rows=allData;
  if(activeFilter!=='all') rows=rows.filter(r=>r.tx_type===activeFilter);
  if(minVal>0) rows=rows.filter(r=>r.value>=minVal);
  renderTable(rows);
  document.getElementById('tableCount').textContent=rows.length+' transactions';
}

function updateStats(){
  const buys=allData.filter(r=>r.tx_type==='buy');
  const sells=allData.filter(r=>r.tx_type==='sell');
  document.getElementById('statTotal').textContent=allData.length;
  document.getElementById('statSub').textContent=currentTicker||'';
  document.getElementById('statBuys').textContent=buys.length;
  document.getElementById('statBuyVal').textContent=fmtNum(buys.reduce((s,r)=>s+r.value,0));
  document.getElementById('statSells').textContent=sells.length;
  document.getElementById('statSellVal').textContent=fmtNum(sells.reduce((s,r)=>s+r.value,0));
  document.getElementById('statVolume').textContent=fmtNum(allData.reduce((s,r)=>s+r.value,0));
}

// SAFE table renderer — builds DOM nodes directly, never touches innerHTML
// with server/external data. All text is set via textContent, so nothing
// can ever be interpreted as HTML/script, no matter what a filing contains.
function renderTable(rows){
  const tbody=document.getElementById('tableBody');
  clear(tbody);

  if(!rows.length){
    const row = el('tr');
    const td  = el('td', {attrs:{colspan:'11'}});
    const empty = el('div', {className:'empty-state'});
    empty.appendChild(el('div', {className:'empty-icon', text:'📭'}));
    empty.appendChild(document.createTextNode('No transactions match this filter.'));
    td.appendChild(empty);
    row.appendChild(td);
    tbody.appendChild(row);
    return;
  }

  const frag = document.createDocumentFragment();

  rows.forEach((r,i)=>{
    const tc=r.tx_type==='buy'?'tx-buy':r.tx_type==='sell'?'tx-sell':'tx-neutral';
    const rc=r.tx_type==='buy'?'row-buy':r.tx_type==='sell'?'row-sell':'';
    const vc=r.tx_type==='buy'?'value-buy':r.tx_type==='sell'?'value-sell':'';

    const tr = el('tr', {className:rc, attrs:{style:`animation-delay:${Math.min(i*15,300)}ms`}});

    tr.appendChild(el('td', {className:'cell-date', text: fmtDate(r.date)}));
    tr.appendChild(el('td', {className:'cell-ticker', text: r.ticker||'—'}));
    tr.appendChild(el('td', {className:'cell-company', title: r.company||'', text: r.company||'—'}));

    const insiderTd = el('td');
    insiderTd.appendChild(el('div', {className:'cell-insider', text: r.insider||'—'}));
    tr.appendChild(insiderTd);

    const roleTd = el('td');
    roleTd.appendChild(el('div', {className:'cell-role', text: r.title||'—'}));
    tr.appendChild(roleTd);

    const typeTd = el('td');
    typeTd.appendChild(el('span', {className:`tx-badge ${tc}`, text: r.tx_label}));
    tr.appendChild(typeTd);

    tr.appendChild(el('td', {className:'cell-shares', text: fmtShares(r.shares)}));
    tr.appendChild(el('td', {className:'cell-price', text: fmtPrice(r.price)}));
    tr.appendChild(el('td', {className:`cell-value ${vc}`, text: fmtNum(r.value)}));
    tr.appendChild(el('td', {className:'cell-owned', text: fmtShares(r.owned_after)}));

    const filingTd = el('td', {className:'cell-filing'});
    const a = el('a', {text:'SEC ↗', attrs:{href: r.filing_url || '#', target:'_blank', rel:'noopener noreferrer'}});
    filingTd.appendChild(a);
    tr.appendChild(filingTd);

    frag.appendChild(tr);
  });

  tbody.appendChild(frag);
}

// ── Search ────────────────────────────────────────────────────────────────────
const searchInput=document.getElementById('searchInput');
const searchDropdown=document.getElementById('searchDropdown');

searchInput.addEventListener('input',()=>{
  clearTimeout(searchDebounce);
  const q=searchInput.value.trim();
  if(!q){ hideDropdown(); currentTicker=null; showPrompt(); return; }
  searchDebounce=setTimeout(()=>fetchSuggestions(q),250);
});

// SAFE dropdown renderer — builds DOM nodes and binds click listeners via
// addEventListener instead of building inline onclick="..." strings from
// external data (the pattern that triggered the browser's malware scanner).
async function fetchSuggestions(q){
  const resp=await fetch(`/api/search?q=${encodeURIComponent(q)}`);
  const data=await resp.json();
  clear(searchDropdown);
  if(!data.length){ hideDropdown(); return; }

  const frag = document.createDocumentFragment();
  data.forEach(d=>{
    const item = el('div', {className:'dropdown-item'});
    item.appendChild(el('span', {className:'dropdown-ticker', text: d.ticker}));
    item.appendChild(el('span', {className:'dropdown-name', text: d.name}));
    item.addEventListener('click', () => selectTicker(d.ticker));
    frag.appendChild(item);
  });
  searchDropdown.appendChild(frag);
  searchDropdown.classList.add('show');
}

function selectTicker(t){
  searchInput.value=t; currentTicker=t;
  hideDropdown(); allData=[];
  loadData(t);
  loadChart(t);
}

function hideDropdown(){ searchDropdown.classList.remove('show'); }
document.addEventListener('click',e=>{ if(!e.target.closest('.search-wrap')) hideDropdown(); });

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoading(v){
  document.getElementById('refreshBtn').classList.toggle('loading',v);
  if(v){
    const tbody = document.getElementById('tableBody');
    clear(tbody);
    const row = el('tr', {className:'loading-row'});
    const td  = el('td', {attrs:{colspan:'11'}});
    td.appendChild(el('div', {className:'spinner'}));
    td.appendChild(document.createTextNode('Fetching Form 4 filings from SEC EDGAR…'));
    td.appendChild(document.createElement('br'));
    td.appendChild(el('small', {attrs:{style:'font-size:10px;color:var(--dim)'}, text:'"Since IPO" may take 1–2 minutes for older companies'}));
    row.appendChild(td);
    tbody.appendChild(row);
  }
}
function showError(msg){ const b=document.getElementById('errorBanner'); b.textContent=msg; b.style.display='block'; }
function hideError(){ document.getElementById('errorBanner').style.display='none'; }

// ── Wire up controls (no inline onclick/onchange attributes anywhere) ─────────
document.querySelectorAll('.filter-btn').forEach(btn=>{
  btn.addEventListener('click', () => setFilter(btn.dataset.filter, btn));
});
document.getElementById('daysSelect').addEventListener('change', reload);
document.getElementById('minValSelect').addEventListener('change', applyFilters);
document.getElementById('refreshBtn').addEventListener('click', reload);

// ── Chat widget ───────────────────────────────────────────────────────────────
const chatToggle=document.getElementById('chatToggle');
const chatPanel=document.getElementById('chatPanel');
const chatMessages=document.getElementById('chatMessages');
const chatInput=document.getElementById('chatInput');
const chatSend=document.getElementById('chatSend');

chatToggle.addEventListener('click', ()=>{
  chatPanel.classList.toggle('show');
  if(chatPanel.classList.contains('show')) chatInput.focus();
});

function addChatMsg(text, cls){
  const empty = chatMessages.querySelector('.chat-empty');
  if(empty) empty.remove();
  chatMessages.appendChild(el('div', {className:`chat-msg ${cls}`, text}));
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendChat(){
  const question = chatInput.value.trim();
  if(!question) return;
  addChatMsg(question, 'user');
  chatInput.value=''; chatInput.disabled=true; chatSend.disabled=true;

  try{
    const resp = await fetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        question,
        ticker: currentTicker || '',
        days: document.getElementById('daysSelect').value,
      }),
    });
    const json = await resp.json();
    if(!resp.ok) throw new Error(json.error || ('HTTP '+resp.status));
    addChatMsg(json.answer, 'bot');
  }catch(e){
    addChatMsg(e.message, 'error');
  }finally{
    chatInput.disabled=false; chatSend.disabled=false; chatInput.focus();
  }
}

chatSend.addEventListener('click', sendChat);
chatInput.addEventListener('keydown', e=>{ if(e.key==='Enter') sendChat(); });
</script>
</body>
</html>"""


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════╗")
    print("║   InsiderFlow Dashboard              ║")
    print("║   http://127.0.0.1:8090              ║")
    print("╚══════════════════════════════════════╝")
    print()
    app.run(host='0.0.0.0', debug=False, port=8090, threaded=True, use_reloader=False)