# -*- coding: utf-8 -*-
"""
美股選股（Streamlit 網頁版）

三個分頁：
  1. 趨勢突破掃描：大盤環境 → 流動性 → Minervini 趨勢模板 → RS 相對強度 → ATR 停損與建議股數（只用價格，速度快）
  2. 多因子選股：動能 + 品質 + 價值 + 成長綜合排名 → 換手緩衝 → 產業/細產業上限 → 反波動加權 → 大盤曝險
  3. 預估上修選股：分析師 EPS 預估上修（財報預期動能）+ 價格動能 + 品質 → 產業上限 → ATR 停損與建議股數 → 大盤曝險

功能：
  - 自行增加個股：併入股票池，並附「診斷」說明每檔為何入選或被排除
  - 細產業上限（例如煉油、半導體）、下次財報日標示、可選擇不新買進財報在即的股票
  - 上月持股與自選股可存進網址（加入書籤即可帶走），也可匯入/匯出 CSV

使用方式：
  - 放在專案根目錄當主程式（Streamlit Cloud 的 Main file path 填此檔名），或放進現有專案的 pages/ 資料夾
  - 單獨執行：streamlit run 1_US_Stock_Screener.py
  - requirements.txt 需要：streamlit yfinance pandas numpy openpyxl lxml requests

注意：資料來自 Yahoo Finance（免費、延遲、偶有缺漏、雲端環境可能被限流）；本程式是篩選工具，不是投資建議。
"""

import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="美股選股", page_icon="📈", layout="wide")

BENCH = "SPY"
WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


# ============================================================
# 股票池
# ============================================================
def _wiki_tickers(url):
    import requests

    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    for tbl in pd.read_html(StringIO(resp.text)):
        cols = [str(c) for c in tbl.columns]
        for key in ("Symbol", "Ticker symbol", "Ticker"):
            if key in cols:
                syms = tbl[key].astype(str).str.strip().str.upper()
                return [s.replace(".", "-") for s in syms if s and s != "NAN"]
    raise RuntimeError(f"找不到代號欄位：{url}")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_index_universe(name):
    parts = ["sp500"] if name == "sp500" else ["sp500", "sp400", "sp600"]
    out = []
    for p in parts:
        out += _wiki_tickers(WIKI[p])
    return sorted(set(out))


def parse_tickers(text):
    raw = str(text or "").replace(",", " ").replace(";", " ").split()
    return sorted({t.strip().upper().replace(".", "-") for t in raw if t.strip()})


# ============================================================
# 價格資料
# ============================================================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _download_chunk(chunk, period):
    raw = yf.download(
        list(chunk), period=period, interval="1d", auto_adjust=True,
        group_by="ticker", threads=True, progress=False,
    )
    out = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        for t in chunk:
            if t in level0:
                d = raw[t].dropna(subset=["Close"])
                if len(d):
                    out[t] = d
    else:
        d = raw.dropna(subset=["Close"])
        if len(d):
            out[chunk[0]] = d
    return out


def load_prices(tickers, period="2y", progress=None, batch=80):
    out = {}
    chunks = [tuple(tickers[i : i + batch]) for i in range(0, len(tickers), batch)]
    for k, ch in enumerate(chunks, 1):
        try:
            out.update(_download_chunk(ch, period))
        except Exception as e:  # noqa: BLE001
            st.warning(f"第 {k} 批價格下載失敗：{e}")
        if progress is not None:
            progress.progress(k / len(chunks), text=f"下載價格 {k}/{len(chunks)} 批")
    return out


def analyze_price(df):
    d = df.dropna(subset=["Close"])
    if len(d) < 253:
        return None
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    last = float(c.iloc[-1])
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    hi52, lo52 = float(h.iloc[-252:].max()), float(l.iloc[-252:].min())
    sma200 = c.rolling(200).mean()
    vol50 = v.rolling(50).mean().iloc[-1]
    return {
        "last_date": d.index[-1],
        "close": last,
        "sma50": float(c.rolling(50).mean().iloc[-1]),
        "sma150": float(c.rolling(150).mean().iloc[-1]),
        "sma200": float(sma200.iloc[-1]),
        "sma200_prev": float(sma200.iloc[-22]),
        "hi52": hi52,
        "lo52": lo52,
        "near_high": last / hi52,
        "atr": float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]),
        "dollar_vol": float((c * v).rolling(50).mean().iloc[-1]),
        "vol_ratio": float(v.iloc[-5:].mean() / vol50) if vol50 else np.nan,
        "vol60": float(c.pct_change().iloc[-60:].std() * math.sqrt(252)),
        "r12_1": float(c.iloc[-22]) / float(c.iloc[-253]) - 1,
        "r6": last / float(c.iloc[-127]) - 1,
        "r3": last / float(c.iloc[-64]) - 1,
    }


def momentum_score(r12_1, r6, near_high):
    return (
        0.50 * r12_1.rank(pct=True) + 0.25 * r6.rank(pct=True) + 0.25 * near_high.rank(pct=True)
    ) * 100


def regime(spy_df):
    c = spy_df["Close"].dropna()
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    last = float(c.iloc[-1])
    above200 = last > float(s200.iloc[-1])
    golden = float(s50.iloc[-1]) > float(s200.iloc[-1])
    if above200 and golden:
        label, scale = "進攻（多頭）", 1.0
    elif above200:
        label, scale = "謹慎（僅在200日線上）", 0.6
    else:
        label, scale = "防守（SPY 在 200 日線下）", 0.3
    return {"label": label, "scale": scale, "spy_close": last,
            "spy_sma50": float(s50.iloc[-1]), "spy_sma200": float(s200.iloc[-1])}


def build_tech(prices):
    rows = {}
    for t, d in prices.items():
        if t == BENCH:
            continue
        r = analyze_price(d)
        if r:
            rows[t] = r
    return pd.DataFrame.from_dict(rows, orient="index")


# ============================================================
# 財報日
# ============================================================
def _to_date(x):
    try:
        ts = pd.Timestamp(x)
        if ts is pd.NaT:
            return None
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.normalize()
    except Exception:  # noqa: BLE001
        return None


def pick_next_earnings(cal, info=None):
    # 從 yfinance 的 calendar / info 找「今天以後最近的一個財報日」，找不到回傳 None
    today = pd.Timestamp.today().normalize()
    cands = []
    try:
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed is not None:
                cands += list(ed) if isinstance(ed, (list, tuple)) else [ed]
        elif cal is not None and not getattr(cal, "empty", True) and "Earnings Date" in cal.index:
            cands += [v for v in cal.loc["Earnings Date"].tolist() if pd.notna(v)]
    except Exception:  # noqa: BLE001
        pass
    if info:
        for k in ("earningsTimestampStart", "earningsTimestamp"):
            v = info.get(k)
            if v:
                try:
                    cands.append(pd.to_datetime(v, unit="s"))
                except Exception:  # noqa: BLE001
                    pass
    dates = sorted({d for d in (_to_date(c) for c in cands) if d is not None and d >= today})
    return dates[0].strftime("%Y-%m-%d") if dates else None


def add_earnings_cols(df, warn_days):
    x = df.copy()
    today = pd.Timestamp.today().normalize()
    if "next_earnings" in x.columns:
        d = pd.to_datetime(x["next_earnings"], errors="coerce")
    else:
        d = pd.Series(pd.NaT, index=x.index)
        x["next_earnings"] = None
    x["earn_days"] = (d - today).dt.days
    soon = x["earn_days"].between(0, warn_days)
    label = "⚠ " + x["earn_days"].astype("Int64").astype(str) + " 天內財報"
    x["earn_flag"] = np.where(soon, label, "")
    x["earn_soon"] = soon.fillna(False).astype(bool)
    return x


@st.cache_resource
def _earn_store():
    return {}


def fetch_earnings_only(ticker):
    try:
        return pick_next_earnings(yf.Ticker(ticker).calendar)
    except Exception:  # noqa: BLE001
        return None


def get_earnings(tickers, progress=None, max_age_h=24, workers=4):
    store = _earn_store()
    now = time.time()
    todo = [t for t in tickers if t not in store or now - store[t][0] > max_age_h * 3600]
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_earnings_only, t): t for t in todo}
            for f in as_completed(futs):
                store[futs[f]] = (time.time(), f.result())
                done += 1
                if progress is not None:
                    progress.progress(done / len(todo), text=f"查詢財報日 {done}/{len(todo)}")
    return pd.Series({t: store[t][1] for t in tickers if t in store}, dtype=object).reindex(tickers)


# ============================================================
# 基本面（含 Piotroski F-Score）
# ============================================================
FUND_COLS = ["name", "sector", "industry", "market_cap", "roe", "gross_prof", "debt_eq", "fscore",
             "earn_yield", "fwd_ey", "fcf_yield", "rev_g", "eps_g", "eps_ttm", "next_earnings"]
TEXT_COLS = ("name", "sector", "industry", "next_earnings")


def _row(df, names, col):
    if df is None or getattr(df, "empty", True) or col >= df.shape[1]:
        return None
    for n in names:
        if n in df.index:
            try:
                val = df.loc[n].iloc[col]
            except Exception:  # noqa: BLE001
                continue
            if pd.notna(val):
                return float(val)
    return None


def _ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def piotroski(fin, bs, cf):
    NI = ["Net Income", "Net Income Common Stockholders"]
    TA = ["Total Assets"]
    CFO = ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
    LTD = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
    CA = ["Current Assets", "Total Current Assets"]
    CL = ["Current Liabilities", "Total Current Liabilities"]
    SH = ["Ordinary Shares Number", "Share Issued"]
    GP = ["Gross Profit"]
    REV = ["Total Revenue", "Operating Revenue"]

    ni0 = _row(fin, NI, 0)
    ta0, ta1 = _row(bs, TA, 0), _row(bs, TA, 1)
    ni1 = _row(fin, NI, 1)
    cfo0 = _row(cf, CFO, 0)
    ltd0 = _row(bs, LTD, 0) or 0.0
    ltd1 = _row(bs, LTD, 1) or 0.0
    cr0 = _ratio(_row(bs, CA, 0), _row(bs, CL, 0))
    cr1 = _ratio(_row(bs, CA, 1), _row(bs, CL, 1))
    sh0, sh1 = _row(bs, SH, 0), _row(bs, SH, 1)
    rev0, rev1 = _row(fin, REV, 0), _row(fin, REV, 1)
    gm0 = _ratio(_row(fin, GP, 0), rev0)
    gm1 = _ratio(_row(fin, GP, 1), rev1)
    roa0, roa1 = _ratio(ni0, ta0), _ratio(ni1, ta1)
    lev0, lev1 = _ratio(ltd0, ta0), _ratio(ltd1, ta1)
    at0, at1 = _ratio(rev0, ta0), _ratio(rev1, ta1)

    def gt(x, y):
        return None if x is None or y is None else x > y

    return [
        None if roa0 is None else roa0 > 0,
        None if cfo0 is None else cfo0 > 0,
        gt(roa0, roa1),
        gt(cfo0, ni0),
        None if lev0 is None or lev1 is None else (lev0 == 0 or lev0 < lev1),
        gt(cr0, cr1),
        None if sh0 is None or sh1 is None else sh0 <= sh1 * 1.005,
        gt(gm0, gm1),
        gt(at0, at1),
    ]


def fetch_fundamentals(ticker):
    out = {}
    t = yf.Ticker(ticker)
    info = t.info or {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    mcap, eps = info.get("marketCap"), info.get("trailingEps")
    pe, fpe, fcf = info.get("trailingPE"), info.get("forwardPE"), info.get("freeCashflow")
    if pe and pe > 0:
        earn_yield = 1 / pe
    elif price and eps is not None:
        earn_yield = eps / price
    else:
        earn_yield = None
    out.update(
        name=info.get("shortName"), sector=info.get("sector"), industry=info.get("industry"),
        market_cap=mcap, roe=info.get("returnOnEquity"), debt_eq=info.get("debtToEquity"), eps_ttm=eps,
        earn_yield=earn_yield, fwd_ey=(1 / fpe) if fpe and fpe > 0 else None,
        fcf_yield=(fcf / mcap) if fcf is not None and mcap else None,
        rev_g=info.get("revenueGrowth"), eps_g=info.get("earningsGrowth"),
    )
    try:
        out["next_earnings"] = pick_next_earnings(t.calendar, info)
    except Exception:  # noqa: BLE001
        out["next_earnings"] = pick_next_earnings(None, info)
    try:
        fin, bs, cf = t.financials, t.balance_sheet, t.cashflow
        out["gross_prof"] = _ratio(_row(fin, ["Gross Profit"], 0), _row(bs, ["Total Assets"], 0))
        checks = [c for c in piotroski(fin, bs, cf) if c is not None]
        out["fscore"] = float(sum(checks)) if len(checks) >= 7 else None
    except Exception:  # noqa: BLE001
        pass
    return out


@st.cache_resource
def _fund_store():
    # 跨使用者共用的基本面快取：{ticker: (時間戳, 資料)}；Yahoo 限流時不會重複狂抓
    return {}


def get_fundamentals(tickers, progress=None, max_age_h=24, workers=4):
    store = _fund_store()
    now = time.time()
    todo = [t for t in tickers
            if t not in store or now - store[t][0] > max_age_h * 3600 or "industry" not in store[t][1]]

    def work(t):
        try:
            d = fetch_fundamentals(t)
            return t, (d if d and any(v is not None for v in d.values()) else None)
        except Exception:  # noqa: BLE001
            return t, None

    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for f in as_completed(futs):
                t, d = f.result()
                if d is not None:
                    store[t] = (time.time(), d)
                done += 1
                if progress is not None:
                    progress.progress(done / len(todo), text=f"抓取基本面 {done}/{len(todo)}")
    rows = {t: store[t][1] for t in tickers if t in store}
    out = pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()
    out = out.reindex(index=tickers, columns=FUND_COLS)
    for c in FUND_COLS:
        if c not in TEXT_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ============================================================
# 分析師 EPS 預估上修（Yahoo 的 eps_trend / eps_revisions）
# ============================================================
REV_COLS = ["cy_est", "ny_est", "cy_rev30", "cy_rev90", "ny_rev30", "cy_up30", "cy_down30",
            "rev_breadth", "n_analysts", "surprise_avg"]


def _pct_change(cur, prev):
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev)


def _cell(df, row, col):
    try:
        v = df.loc[row, col]
        return float(v) if pd.notna(v) else None
    except Exception:  # noqa: BLE001
        return None


def _find_col(df, *keys):
    # Yahoo 的欄位大小寫不一致（例如 upLast30days / downLast7Days），用關鍵字找
    if df is None or getattr(df, "empty", True):
        return None
    for c in df.columns:
        name = str(c).lower().replace(" ", "")
        if all(k in name for k in keys):
            return c
    return None


def fetch_revisions(ticker):
    t = yf.Ticker(ticker)

    def safe(attr):
        try:
            return getattr(t, attr)
        except Exception:  # noqa: BLE001
            return None

    tr, er, est = safe("eps_trend"), safe("eps_revisions"), safe("earnings_estimate")
    out = {}
    for row, tag in (("0y", "cy"), ("+1y", "ny")):
        cur = _cell(tr, row, "current")
        out[f"{tag}_est"] = cur
        out[f"{tag}_rev30"] = _pct_change(cur, _cell(tr, row, "30daysAgo"))
        if tag == "cy":
            out["cy_rev90"] = _pct_change(cur, _cell(tr, row, "90daysAgo"))
    up = _cell(er, "0y", _find_col(er, "up", "30"))
    down = _cell(er, "0y", _find_col(er, "down", "30"))
    out["cy_up30"], out["cy_down30"] = up, down
    out["rev_breadth"] = (up - down) / (up + down) if up is not None and down is not None and up + down > 0 else None
    out["n_analysts"] = _cell(est, "0y", "numberOfAnalysts")

    # 近 4 季 EPS 驚喜（實際 vs 預估）：第二個獨立訊號，預估上修資料缺漏時也能提供佐證
    out["surprise_avg"] = None
    try:
        hist = safe("earnings_history")
        col = _find_col(hist, "surprise")
        if col is not None:
            s = pd.to_numeric(hist[col], errors="coerce").dropna()
            if len(s):
                out["surprise_avg"] = float(s.tail(4).mean())
    except Exception:  # noqa: BLE001
        pass
    return out


def diagnose_revisions(tickers=("AAPL", "MSFT", "NVDA")):
    # 資料自檢：直接問 Yahoo 幾檔大型股，列出各資料表的欄位與解析結果，
    # 用來確認 Yahoo 格式沒變、也沒有被限流（大型股通常資料最完整）
    rows = []
    for tk in tickers:
        row = {"代號": tk}
        try:
            t = yf.Ticker(tk)
        except Exception as e:  # noqa: BLE001
            row["狀態"] = f"❌ 無法連線：{type(e).__name__}"
            rows.append(row)
            continue
        for attr in ("eps_trend", "eps_revisions", "earnings_estimate", "earnings_history"):
            try:
                df = getattr(t, attr)
                row[attr] = ("✅ " + ", ".join(map(str, df.columns))) if df is not None and not df.empty else "⚠ 空"
            except Exception as e:  # noqa: BLE001
                row[attr] = f"❌ {type(e).__name__}"
        try:
            parsed = fetch_revisions(tk)
            ok = sum(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in parsed.values())
            row["解析成功欄位數"] = f"{ok}/{len(REV_COLS)}"
            row["狀態"] = "✅ 正常" if ok >= 6 else "⚠ 欄位偏少，Yahoo 格式可能改了或被限流"
        except Exception as e:  # noqa: BLE001
            row["狀態"] = f"❌ 解析失敗：{type(e).__name__}"
        rows.append(row)
    return rows


@st.cache_resource
def _rev_store():
    return {}


def get_revisions(tickers, progress=None, max_age_h=24, workers=4):
    store = _rev_store()
    now = time.time()
    todo = [t for t in tickers if t not in store or now - store[t][0] > max_age_h * 3600]

    def work(t):
        try:
            d = fetch_revisions(t)
            if not d or all(v is None for v in d.values()):
                time.sleep(1.0)  # Yahoo 偶爾暫時限流，隔一秒再試一次
                d = fetch_revisions(t)
            return t, (d if d and any(v is not None for v in d.values()) else None)
        except Exception:  # noqa: BLE001
            return t, None

    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for f in as_completed(futs):
                t, d = f.result()
                if d is not None:
                    store[t] = (time.time(), d)
                done += 1
                if progress is not None:
                    progress.progress(done / len(todo), text=f"抓取分析師預估 {done}/{len(todo)}")
    rows = {t: store[t][1] for t in tickers if t in store}
    out = pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()
    out = out.reindex(index=tickers, columns=REV_COLS)
    for c in REV_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ============================================================
# 因子評分與投組
# ============================================================
QUALITY ={"roe": 1, "gross_prof": 1, "fscore": 1, "debt_eq": -1}
VALUE = {"earn_yield": 1, "fwd_ey": 1, "fcf_yield": 1}
GROWTH = {"rev_g": 1, "eps_g": 1}


def pct_rank(s, groups=None, min_group=8):
    overall = s.rank(pct=True)
    if groups is None:
        return overall
    g = groups.fillna("Unknown")
    within = s.groupby(g).rank(pct=True)
    cnt = s.groupby(g).transform("count")
    return within.where(cnt >= min_group, overall)


def group_score(df, metrics, groups):
    ranks = []
    for col, sign in metrics.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if col == "debt_eq":
            s = s.mask(s < 0, 1e6)
        if sign < 0:
            s = -s
        ranks.append(pct_rank(s, groups))
    if not ranks:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(ranks, axis=1).mean(axis=1, skipna=True) * 100


def score_factors(data, p):
    groups = data["sector"] if p.sector_neutral else None
    S = pd.DataFrame({
        "mom": data["mom"],
        "quality": group_score(data, QUALITY, groups),
        "value": group_score(data, VALUE, groups),
        "growth": group_score(data, GROWTH, groups),
    }, index=data.index)
    W = pd.Series({"mom": p.w_mom, "quality": p.w_quality, "value": p.w_value, "growth": p.w_growth})
    avail = S.notna()
    wsum = (avail * W).sum(axis=1)
    comp = (S.fillna(0) * W).sum(axis=1) / wsum.replace(0, np.nan)
    ok = avail["mom"] & (avail.sum(axis=1) >= 3)
    S["composite"] = comp.where(ok)
    return data.join(S.drop(columns="mom"))


def inverse_vol_weights(vols, max_w):
    w = 1.0 / vols.clip(lower=1e-4)
    w = w / w.sum()
    for _ in range(20):
        over = w > max_w + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        under = ~over
        if not under.any() or w[under].sum() == 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


def build_portfolio(ranked, prev, p):
    # 回傳 (入選清單, 賣出清單, 未入選原因 dict)
    rank_pos = {t: i + 1 for i, t in enumerate(ranked.index)}
    chosen, sector_cnt, industry_cnt, notes = [], {}, {}, {}
    prev_set = set(prev)

    def blocked(t):
        sec, ind = ranked.at[t, "sector"], ranked.at[t, "industry"]
        if sec != "Unknown" and sector_cnt.get(sec, 0) >= p.max_per_sector:
            return f"同產業（{sec}）已滿 {p.max_per_sector} 檔"
        if ind != "Unknown" and industry_cnt.get(ind, 0) >= p.max_per_industry:
            return f"同細產業（{ind}）已滿 {p.max_per_industry} 檔"
        return None

    def add(t):
        chosen.append(t)
        sec, ind = ranked.at[t, "sector"], ranked.at[t, "industry"]
        sector_cnt[sec] = sector_cnt.get(sec, 0) + 1
        industry_cnt[ind] = industry_cnt.get(ind, 0) + 1

    keep = sorted([t for t in prev if t in rank_pos and rank_pos[t] <= p.keep_rank], key=rank_pos.get)
    for t in keep:                                           # 1) 續抱
        if len(chosen) >= p.n_hold:
            break
        why = blocked(t)
        if why:
            notes[t] = why
        else:
            add(t)
    for t in ranked.index:                                   # 2) 依排名補進
        if len(chosen) >= p.n_hold:
            break
        if t in chosen:
            continue
        if p.skip_earnings and t not in prev_set and bool(ranked.at[t, "earn_soon"]):
            notes[t] = f"財報在 {int(ranked.at[t, 'earn_days'])} 天內，暫不新買進"
            continue
        why = blocked(t)
        if why:
            notes[t] = why
        else:
            add(t)
    sells = []
    for t in prev:
        if t in chosen:
            continue
        if t not in rank_pos:
            why = "不符篩選條件（流動性/趨勢/資料不足/虧損）"
        elif rank_pos[t] > p.keep_rank:
            why = f"排名第 {rank_pos[t]}，跌出前 {p.keep_rank} 名"
        else:
            why = notes.get(t, "名額限制")
        sells.append({"代號": t, "原因": why})
    return chosen, pd.DataFrame(sells, columns=["代號", "原因"]), notes


# ============================================================
# 兩套選股流程
# ============================================================
TREND_RULES = {
    "收盤>150與200日線": lambda t: (t.close > t.sma150) & (t.close > t.sma200),
    "150日線>200日線": lambda t: t.sma150 > t.sma200,
    "200日線走升": lambda t: t.sma200 > t.sma200_prev,
    "50日線>150與200日線": lambda t: (t.sma50 > t.sma150) & (t.sma50 > t.sma200),
    "收盤>50日線": lambda t: t.close > t.sma50,
    "高於52週低點30%以上": lambda t: t.close >= 1.30 * t.lo52,
    "距52週高點25%內": lambda t: t.close >= 0.75 * t.hi52,
}


def run_trend(prices, p, progress=None):
    tech = build_tech(prices)
    if tech.empty:
        raise RuntimeError("沒有任何股票有足夠的價格資料。")
    funnel = {"股票池（成功下載）": len([t for t in prices if t != BENCH]), "資料足夠(>=253日)": len(tech)}
    liquid = tech[(tech["close"] >= p.min_price) & (tech["dollar_vol"] >= p.min_dv)].copy()
    funnel["通過股價與流動性"] = len(liquid)
    if liquid.empty:
        raise RuntimeError("流動性篩選後沒有任何股票。")
    breadth = float((liquid["close"] > liquid["sma200"]).mean() * 100)

    raw = (0.4 * liquid["r12_1"].rank(pct=True) + 0.3 * liquid["r6"].rank(pct=True)
           + 0.3 * liquid["r3"].rank(pct=True))
    liquid["rs"] = (raw.rank(pct=True) * 98 + 1).round(0)
    for name, rule in TREND_RULES.items():
        liquid[name] = rule(liquid)
    liquid["trend_ok"] = liquid[list(TREND_RULES)].all(axis=1)
    funnel["通過趨勢模板（7 條件）"] = int(liquid["trend_ok"].sum())
    cand_all = liquid[liquid["trend_ok"] & (liquid["rs"] >= p.min_rs)].sort_values("rs", ascending=False)
    funnel[f"且 RS>={p.min_rs}"] = len(cand_all)
    cand = cand_all.head(p.top).copy()

    reg = regime(prices[BENCH])
    risk_pct = p.risk * reg["scale"]
    cand["stop"] = cand["close"] - p.atr_mult * cand["atr"]
    per_share = (cand["close"] - cand["stop"]).clip(lower=0.01)
    by_risk = (p.equity * risk_pct / per_share).apply(math.floor)
    by_cap = (p.equity * p.max_position / cand["close"]).apply(math.floor)
    cand["shares"] = np.minimum(by_risk, by_cap).astype(int)
    cand["position_value"] = cand["shares"] * cand["close"]
    cand["dist_high"] = (1 - cand["near_high"]) * 100
    cand["ext50"] = (cand["close"] / cand["sma50"] - 1) * 100

    def status(r):
        if r["ext50"] > 20:
            return "過度延伸，等回檔"
        if r["dist_high"] <= 5:
            return "接近新高，可關注突破"
        return "趨勢中，等回檔至50日線附近"

    cand["status"] = cand.apply(status, axis=1) if len(cand) else []

    # 財報日：候選股 + 自選股（通過流動性者）
    extra_liq = [t for t in p.extra if t in liquid.index]
    if p.check_earnings and (len(cand) or extra_liq):
        ed = get_earnings(list(dict.fromkeys(list(cand.index) + extra_liq)), progress=progress)
        cand["next_earnings"] = ed.reindex(cand.index)
    else:
        extra_liq_ed = None
        ed = pd.Series(dtype=object)
        cand["next_earnings"] = None
    cand = add_earnings_cols(cand, p.warn_days)

    # 自選股診斷
    diag = []
    for t in p.extra:
        if t not in prices:
            diag.append({"代號": t, "結果": "❌ 查無價格資料（代號可能有誤，或 Yahoo 沒有這檔）"})
        elif t not in tech.index:
            diag.append({"代號": t, "結果": "❌ 價格資料不足（上市未滿約 1 年）"})
        elif t not in liquid.index:
            diag.append({"代號": t, "結果": f"❌ 股價或成交額不足（收盤 {tech.at[t, 'close']:.2f}、日均成交額 {tech.at[t, 'dollar_vol'] / 1e6:.1f}M）"})
        else:
            row = liquid.loc[t]
            failed = [n for n in TREND_RULES if not bool(row[n])]
            if failed:
                res = "❌ 未通過趨勢模板：" + "、".join(failed)
            elif row["rs"] < p.min_rs:
                res = f"➖ 通過趨勢模板，但 RS {row['rs']:.0f} 低於門檻 {p.min_rs}"
            elif t in cand.index:
                res = "✅ 入選候選清單"
            else:
                res = "✅ 符合條件，但超出「最多顯示檔數」"
            ed_t = ed.get(t) if len(ed) else None
            diag.append({"代號": t, "結果": res, "收盤價": round(float(row["close"]), 2),
                         "RS評分": float(row["rs"]), "下次財報日": ed_t})
    diag_df = pd.DataFrame(diag) if diag else pd.DataFrame(columns=["代號", "結果"])
    return {"cand": cand, "regime": reg, "funnel": funnel, "breadth": breadth, "diag": diag_df,
            "charts": {t: prices[t]["Close"].iloc[-260:] for t in cand.index}}


def run_multifactor(prices, p, progress=None):
    tech = build_tech(prices)
    if tech.empty:
        raise RuntimeError("沒有任何股票有足夠的價格資料。")
    funnel = {"股票池（成功下載）": len([t for t in prices if t != BENCH]), "資料足夠(>=253日)": len(tech)}
    liq1 = tech[(tech["close"] >= p.min_price) & (tech["dollar_vol"] >= p.min_dv)].copy()
    funnel["通過股價與流動性"] = len(liq1)
    liquid = liq1
    if p.trend_filter:
        liquid = liq1[liq1["close"] > liq1["sma200"]].copy()
        funnel["收盤>200日線"] = len(liquid)
    if liquid.empty:
        raise RuntimeError("流動性與趨勢篩選後沒有任何股票（大盤可能很弱）。")

    liquid["mom"] = momentum_score(liquid["r12_1"], liquid["r6"], liquid["near_high"])
    cut = liquid["mom"].quantile(1 - p.prefilter_top)
    pre = liquid[liquid["mom"] >= cut].sort_values("mom", ascending=False).head(p.max_fundamental)
    funnel[f"動能前{int(p.prefilter_top * 100)}%（抓基本面）"] = len(pre)
    extra_in = [t for t in p.extra if t in liquid.index and t not in pre.index]
    if extra_in:                                    # 自選股只要通過前面的濾網就一定評分
        pre = pd.concat([pre, liquid.loc[extra_in]])
        funnel["自選股另外加入評分"] = len(extra_in)

    fund = get_fundamentals(list(pre.index), progress=progress)
    funnel["基本面取得成功"] = int(fund[["sector", "roe", "earn_yield"]].notna().any(axis=1).sum())
    data = pre.join(fund)
    data["sector"] = data["sector"].fillna("Unknown")
    data["industry"] = data["industry"].fillna("Unknown")
    if not p.allow_unprofitable:
        data = data[~(data["eps_ttm"].notna() & (data["eps_ttm"] <= 0))]
        funnel["排除近四季虧損"] = len(data)

    scored = score_factors(data, p)
    ranked = scored.dropna(subset=["composite"]).sort_values("composite", ascending=False).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    funnel["可評分（資料足夠）"] = len(ranked)
    if ranked.empty:
        raise RuntimeError("沒有任何股票可評分（基本面資料可能抓取失敗或被 Yahoo 限流，請稍後再試）。")
    ranked = add_earnings_cols(ranked, p.warn_days)

    chosen, sells, notes = build_portfolio(ranked, p.prev, p)
    pf = ranked.loc[chosen].copy()
    reg = regime(prices[BENCH])
    w = inverse_vol_weights(pf["vol60"].fillna(pf["vol60"].median()), p.max_weight)
    pf["weight_pct"] = w * reg["scale"] * 100
    pf["target_value"] = p.equity * w * reg["scale"]
    pf["shares"] = (pf["target_value"] / pf["close"]).apply(math.floor).astype(int)
    pf["stop"] = pf["close"] - p.atr_mult * pf["atr"]
    pf["stop_risk_pct"] = pf["shares"] * (pf["close"] - pf["stop"]) / p.equity * 100
    pf["action"] = ["續抱" if t in p.prev else "買進" for t in pf.index]

    # 自選股診斷
    rank_pos = {t: i + 1 for i, t in enumerate(ranked.index)}
    diag = []
    for t in p.extra:
        if t not in prices:
            r = "❌ 查無價格資料（代號可能有誤，或 Yahoo 沒有這檔）"
        elif t not in tech.index:
            r = "❌ 價格資料不足（上市未滿約 1 年）"
        elif t not in liq1.index:
            r = f"❌ 股價或成交額不足（收盤 {tech.at[t, 'close']:.2f}、日均成交額 {tech.at[t, 'dollar_vol'] / 1e6:.1f}M）"
        elif t not in liquid.index:
            r = "❌ 收盤在 200 日線之下（可取消「只考慮收盤 > 200 日線」）"
        elif t not in data.index:
            r = "❌ 近四季虧損，已排除（可勾選「允許近四季虧損」）"
        elif t not in rank_pos:
            r = "❌ 基本面資料不足，無法評分（可能被 Yahoo 限流，稍後再試）"
        elif t in chosen:
            r = f"✅ 入選持股（排名 {rank_pos[t]}）"
        else:
            r = f"➖ 排名第 {rank_pos[t]}，未入選（{notes.get(t, '名額已滿')}）"
        row = {"代號": t, "結果": r}
        if t in rank_pos:
            rr = ranked.loc[t]
            row.update({"綜合分": round(float(rr["composite"]), 1), "動能分": round(float(rr["mom"]), 1),
                        "品質分": None if pd.isna(rr["quality"]) else round(float(rr["quality"]), 1),
                        "價值分": None if pd.isna(rr["value"]) else round(float(rr["value"]), 1),
                        "成長分": None if pd.isna(rr["growth"]) else round(float(rr["growth"]), 1),
                        "下次財報日": rr["next_earnings"], "財報提醒": rr["earn_flag"]})
        diag.append(row)
    diag_df = pd.DataFrame(diag) if diag else pd.DataFrame(columns=["代號", "結果"])
    return {"pf": pf, "sells": sells, "ranked": ranked, "regime": reg, "funnel": funnel, "diag": diag_df,
            "charts": {t: prices[t]["Close"].iloc[-260:] for t in pf.index}}


def run_revisions(prices, p, progress=None):
    # 分析師預估上修 + 價格動能 + 品質：找「基本面預期正在變好、股價也確認」的股票
    tech = build_tech(prices)
    if tech.empty:
        raise RuntimeError("沒有任何股票有足夠的價格資料。")
    funnel = {"股票池（成功下載）": len([t for t in prices if t != BENCH]), "資料足夠(>=253日)": len(tech)}
    liq1 = tech[(tech["close"] >= p.min_price) & (tech["dollar_vol"] >= p.min_dv)].copy()
    funnel["通過股價與流動性"] = len(liq1)
    liquid = liq1
    if p.trend_filter:
        liquid = liq1[liq1["close"] > liq1["sma200"]].copy()
        funnel["收盤>200日線"] = len(liquid)
    if liquid.empty:
        raise RuntimeError("流動性與趨勢篩選後沒有任何股票（大盤可能很弱）。")

    liquid["mom"] = momentum_score(liquid["r12_1"], liquid["r6"], liquid["near_high"])
    cut = liquid["mom"].quantile(1 - p.prefilter_top)
    pre = liquid[liquid["mom"] >= cut].sort_values("mom", ascending=False).head(p.max_fundamental)
    funnel[f"動能前{int(p.prefilter_top * 100)}%（抓預估與基本面）"] = len(pre)
    extra_in = [t for t in p.extra if t in liquid.index and t not in pre.index]
    if extra_in:
        pre = pd.concat([pre, liquid.loc[extra_in]])
        funnel["自選股另外加入評分"] = len(extra_in)

    rev = get_revisions(list(pre.index), progress=progress)
    funnel["分析師預估取得成功"] = int(rev[["cy_rev30", "rev_breadth"]].notna().any(axis=1).sum())
    fund = get_fundamentals(list(pre.index), progress=progress)
    data = pre.join(rev).join(fund)
    data["sector"] = data["sector"].fillna("Unknown")
    data["industry"] = data["industry"].fillna("Unknown")

    # 資料品質關卡：Yahoo 的預估資料對小型股常常缺漏，缺資料不能當成「中性」排進名單
    sig_cols = [c for c in ("cy_rev30", "cy_rev90", "ny_rev30", "rev_breadth", "surprise_avg") if c in data.columns]
    data["rev_signals"] = data[sig_cols].notna().sum(axis=1)
    data["data_quality"] = np.where(data["rev_signals"] >= 4, "完整", np.where(data["rev_signals"] >= 2, "部分", "不足"))
    if p.min_analysts > 0:
        data = data[data["n_analysts"].fillna(0) >= p.min_analysts]
        funnel[f"分析師家數>={p.min_analysts}"] = len(data)
    data = data[data["rev_signals"] >= 2]
    funnel["預估訊號>=2項（資料品質足夠）"] = len(data)

    if not p.allow_unprofitable:
        data = data[~(data["eps_ttm"].notna() & (data["eps_ttm"] <= 0))]
        funnel["排除近四季虧損"] = len(data)
    if p.require_up:
        data = data[data["cy_rev30"].fillna(-1) > 0]
        funnel["本年度 EPS 預估 30 日內上修"] = len(data)

    rev_ranks = [data[c].rank(pct=True) for c in ("cy_rev30", "cy_rev90", "ny_rev30", "rev_breadth", "surprise_avg")
                 if c in data.columns and data[c].notna().any()]
    rev_score = (pd.concat(rev_ranks, axis=1).mean(axis=1, skipna=True) * 100) if rev_ranks \
        else pd.Series(np.nan, index=data.index)
    groups = data["sector"] if p.sector_neutral else None
    S = pd.DataFrame({"rev_score": rev_score, "mom": data["mom"],
                      "quality": group_score(data, QUALITY, groups)}, index=data.index)
    W = pd.Series({"rev_score": p.w_rev, "mom": p.w_mom, "quality": p.w_quality})
    avail = S.notna()
    wsum = (avail * W).sum(axis=1)
    comp = (S.fillna(0) * W).sum(axis=1) / wsum.replace(0, np.nan)
    S["composite"] = comp.where(avail["rev_score"] & avail["mom"])
    scored = data.drop(columns=["mom"]).join(S)

    ranked = scored.dropna(subset=["composite"]).sort_values("composite", ascending=False).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    funnel["可評分（資料足夠）"] = len(ranked)
    if ranked.empty:
        raise RuntimeError("沒有任何股票可評分（預估資料可能抓取失敗或被 Yahoo 限流，請稍後再試）。")
    ranked = add_earnings_cols(ranked, p.warn_days)

    chosen, _, notes = build_portfolio(ranked, [], p)
    pf = ranked.loc[chosen].copy()
    reg = regime(prices[BENCH])
    risk_pct = p.risk * reg["scale"]
    pf["stop"] = pf["close"] - p.atr_mult * pf["atr"]
    per_share = (pf["close"] - pf["stop"]).clip(lower=0.01)
    by_risk = (p.equity * risk_pct / per_share).apply(math.floor)
    by_cap = (p.equity * p.max_position / pf["close"]).apply(math.floor)
    pf["shares"] = np.minimum(by_risk, by_cap).astype(int)
    pf["position_value"] = pf["shares"] * pf["close"]

    rank_pos = {t: i + 1 for i, t in enumerate(ranked.index)}
    diag = []
    for t in p.extra:
        if t not in prices:
            r = "❌ 查無價格資料（代號可能有誤，或 Yahoo 沒有這檔）"
        elif t not in tech.index:
            r = "❌ 價格資料不足（上市未滿約 1 年）"
        elif t not in liq1.index:
            r = f"❌ 股價或成交額不足（收盤 {tech.at[t, 'close']:.2f}、日均成交額 {tech.at[t, 'dollar_vol'] / 1e6:.1f}M）"
        elif t not in liquid.index:
            r = "❌ 收盤在 200 日線之下（可取消「只考慮收盤 > 200 日線」）"
        elif t not in data.index:
            r = "❌ 已排除：分析師家數不足、預估資料訊號少於 2 項、近四季虧損，或本年度 EPS 預估近 30 日沒有上修"
        elif t not in rank_pos:
            r = "❌ 預估資料不足，無法評分（Yahoo 可能沒有這檔的分析師預估）"
        elif t in chosen:
            r = f"✅ 入選（排名 {rank_pos[t]}）"
        else:
            r = f"➖ 排名第 {rank_pos[t]}，未入選（{notes.get(t, '名額已滿')}）"
        row = {"代號": t, "結果": r}
        if t in rank_pos:
            rr = ranked.loc[t]
            row.update({"綜合分": round(float(rr["composite"]), 1), "預估上修分": round(float(rr["rev_score"]), 1),
                        "下次財報日": rr["next_earnings"], "財報提醒": rr["earn_flag"]})
        diag.append(row)
    diag_df = pd.DataFrame(diag) if diag else pd.DataFrame(columns=["代號", "結果"])
    return {"pf": pf, "ranked": ranked, "regime": reg, "funnel": funnel, "diag": diag_df,
            "charts": {t: prices[t]["Close"].iloc[-260:] for t in pf.index}}


# ============================================================
# 顯示與匯出
# ============================================================
def fmt(df, mapping, pct_cols=()):
    x = df.copy()
    for c in pct_cols:
        if c in x:
            x[c] = x[c] * 100
    if "near_high" in x:
        x["dist_high"] = (1 - x["near_high"]) * 100
    if "last_date" in x:
        x["last_date"] = pd.to_datetime(x["last_date"]).dt.strftime("%Y-%m-%d")
    keep = [c for c in mapping if c in x.columns]
    x = x[keep].rename(columns=mapping)
    num = x.select_dtypes("number").columns
    x[num] = x[num].round(2)
    x.index.name = "代號"
    return x


TREND_MAP = {
    "close": "收盤價", "rs": "RS評分", "r12_1": "12-1月報酬%", "r6": "6月報酬%", "dist_high": "距52週高%",
    "ext50": "高於50日線%", "vol_ratio": "近5日量/50日均量", "atr": "ATR14", "stop": "建議停損價",
    "shares": "建議股數", "position_value": "部位金額", "status": "狀態",
    "next_earnings": "下次財報日", "earn_flag": "財報提醒", "last_date": "資料日期",
}
MF_MAP = {
    "name": "公司", "sector": "產業", "industry": "細產業", "action": "動作", "earn_flag": "財報提醒",
    "next_earnings": "下次財報日", "rank": "排名", "composite": "綜合分",
    "mom": "動能分", "quality": "品質分", "value": "價值分", "growth": "成長分", "close": "收盤價",
    "weight_pct": "建議權重%", "shares": "建議股數", "target_value": "建議金額", "stop": "建議停損價",
    "stop_risk_pct": "停損風險占資金%", "r12_1": "12-1月報酬%", "r6": "6月報酬%", "dist_high": "距52週高%",
    "vol60": "年化波動%", "fscore": "F-Score(0-9)", "earn_yield": "盈餘殖利率%", "fcf_yield": "FCF殖利率%",
    "roe": "ROE%", "last_date": "資料日期",
}


REV_MAP = {
    "name": "公司", "sector": "產業", "industry": "細產業", "earn_flag": "財報提醒", "next_earnings": "下次財報日",
    "rank": "排名", "composite": "綜合分", "rev_score": "預估上修分", "mom": "動能分", "quality": "品質分",
    "close": "收盤價", "cy_rev30": "本年度EPS預估30日變化%", "cy_rev90": "本年度EPS預估90日變化%",
    "ny_rev30": "明年度EPS預估30日變化%", "rev_breadth": "上修−下修占比%(30日)",
    "cy_up30": "30日上修家數", "cy_down30": "30日下修家數", "n_analysts": "分析師家數",
    "data_quality": "預估資料品質", "surprise_avg": "近4季EPS驚喜(Yahoo原值)",
    "stop": "建議停損價", "shares": "建議股數", "position_value": "部位金額", "r12_1": "12-1月報酬%",
    "r6": "6月報酬%", "dist_high": "距52週高%", "fscore": "F-Score(0-9)", "roe": "ROE%", "last_date": "資料日期",
}


def to_excel_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=df.index.name is not None)
        for ws in w.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 48)
    return buf.getvalue()


def show_regime(reg, extra=None):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("大盤環境", reg["label"].split("（")[0])
    c2.metric("建議曝險", f"{reg['scale']:.0%}")
    c3.metric("SPY 收盤 / 200日線", f"{reg['spy_close']:.2f}", f"{(reg['spy_close'] / reg['spy_sma200'] - 1) * 100:+.1f}%")
    if extra is not None:
        c4.metric("流動股站上200日線比例", f"{extra:.0f}%")


def show_funnel(funnel):
    with st.expander("篩選漏斗（每一關剩下幾檔）"):
        st.dataframe(pd.Series(funnel, name="檔數").to_frame())


def show_chart(charts, key):
    if not charts:
        return
    t = st.selectbox("查看價格走勢", list(charts), key=key)
    c = charts[t]
    st.line_chart(pd.DataFrame({"收盤價": c, "50日線": c.rolling(50, min_periods=1).mean()}))


def show_diag(diag):
    if diag is None or diag.empty:
        return
    st.subheader("你加入的個股：診斷")
    st.dataframe(diag.set_index("代號"))


# ============================================================
# 狀態保存（網址 / CSV）
# ============================================================
def _init_state():
    # 第一次載入時，從網址參數還原「自選股」與「上月持股」
    if "init_done" not in st.session_state:
        st.session_state["init_done"] = True
        qp = st.query_params
        st.session_state["extra"] = qp.get("extra", "")
        st.session_state["m_prev"] = qp.get("prev", "")


def _set_param(key, tickers):
    if tickers:
        st.query_params[key] = ",".join(tickers)
    elif key in st.query_params:
        del st.query_params[key]


def save_to_url():
    _set_param("extra", parse_tickers(st.session_state.get("extra", "")))
    _set_param("prev", parse_tickers(st.session_state.get("m_prev", "")))
    st.session_state["flash"] = "已存到網址：請把這個網頁加入書籤，下次從書籤開啟就會自動帶回自選股與上月持股。"


def adopt_holdings(tickers):
    st.session_state["m_prev"] = ", ".join(tickers)
    _set_param("prev", list(tickers))
    st.session_state["flash"] = "已把本次建議持股設為「上月持股」（也已寫入網址，記得加入書籤）。"


def import_holdings():
    f = st.session_state.get("m_upload")
    if f is None:
        return
    try:
        df = pd.read_csv(f)
        col = next((c for c in df.columns if str(c).strip().lower() in ("ticker", "代號", "symbol")), df.columns[0])
        tks = parse_tickers(" ".join(map(str, df[col].dropna())))
        st.session_state["m_prev"] = ", ".join(tks)
        st.session_state["flash"] = f"已匯入 {len(tks)} 檔持股。"
    except Exception as e:  # noqa: BLE001
        st.session_state["flash"] = f"匯入失敗：{e}"


def get_universe(uni, custom, extra):
    if uni == "自訂代號":
        base = parse_tickers(custom)
    else:
        try:
            base = load_index_universe("sp500" if uni == "S&P 500" else "sp1500")
        except Exception as e:  # noqa: BLE001
            st.error(f"無法取得指數成分股名單（{e}）。可改用「自訂代號」。")
            st.stop()
    tickers = sorted(set(base) | set(extra))
    if not tickers:
        st.error("請在左側輸入至少一個代號。")
        st.stop()
    return tickers


def prepare_prices(uni, custom, extra):
    tickers = get_universe(uni, custom, extra)
    bar = st.progress(0.0, text="準備下載價格…")
    prices = load_prices(sorted(set(tickers + [BENCH])), progress=bar)
    bar.empty()
    if BENCH not in prices:
        st.error("下載不到 SPY（Yahoo Finance 可能暫時限流或無法連線），請稍後再試。")
        st.stop()
    return prices


# ============================================================
# 介面
# ============================================================
_init_state()
st.title("📈 美股選股")
st.caption("資料來源：Yahoo Finance（免費、有延遲、偶有缺漏）。本工具為篩選與研究用途，不是投資建議。")

with st.sidebar:
    st.header("股票池與共同設定")
    uni = st.selectbox("股票池", ["S&P 500", "S&P 1500", "自訂代號"], key="uni")
    custom = st.text_area("自訂代號（逗號或換行分隔）", "NVDA, AAPL, MSFT, AMZN, META, GOOGL, AVGO, LLY, COST, NFLX",
                          disabled=(uni != "自訂代號"), key="custom")
    st.text_area("➕ 額外加入個股（自選股）", key="extra", placeholder="例如：TSLA, PLTR, SMCI",
                 help="這些代號會併入股票池。通過股價/成交額與趨勢濾網的一定會被評分，結果下方的「診斷」會說明每檔為何入選或被排除。")
    st.button("🔖 把自選股與上月持股存到網址", on_click=save_to_url, key="save_url",
              help="存好後把網頁加入書籤，下次從書籤開啟就會自動帶回。")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    st.divider()
    min_price = st.number_input("最低股價（美元）", min_value=1.0, value=10.0, step=1.0, key="min_price")
    min_dv_m = st.number_input("最低日均成交額（百萬美元）", min_value=0.0, value=20.0, step=5.0, key="min_dv")
    equity = st.number_input("總資金（美元）", min_value=1000, value=100000, step=10000, key="equity")
    warn_days = st.number_input("財報警示天數", min_value=1, max_value=60, value=14, step=1, key="warn_days",
                                help="下次財報日在這麼多天以內，表中會標示「⚠ N 天內財報」。")
    st.caption("首次執行會下載約 2 年日線，之後 6 小時內重跑會用快取。S&P 1500 較慢。")

extra_list = parse_tickers(st.session_state.get("extra", ""))
tab_trend, tab_mf, tab_rev = st.tabs(["🚀 趨勢突破掃描", "🧮 多因子選股（月度）", "📊 預估上修選股"])

# ---------------- 分頁 1：趨勢突破 ----------------
with tab_trend:
    st.markdown("找出**已經處於強勢上升趨勢**的股票：Minervini 趨勢模板 + RS 相對強度，再用 ATR 算停損與建議股數。只用價格，速度快。")
    c1, c2, c3, c4 = st.columns(4)
    min_rs = c1.slider("最低 RS 評分", 50, 99, 80, key="t_rs")
    atr_mult_t = c2.number_input("停損 = 收盤 − N × ATR", 1.0, 6.0, 2.5, 0.5, key="t_atr")
    risk_t = c3.number_input("單筆風險占總資金 %", 0.1, 5.0, 1.0, 0.1, key="t_risk") / 100
    max_pos_t = c4.number_input("單檔部位上限 %", 5.0, 100.0, 20.0, 5.0, key="t_maxpos") / 100
    c1, c2 = st.columns(2)
    top_t = c1.slider("最多顯示幾檔", 5, 100, 30, key="t_top")
    check_earn = c2.checkbox("查詢候選股的下次財報日", value=True, key="t_earn")

    if st.button("開始趨勢掃描", type="primary", key="run_trend"):
        prices = prepare_prices(uni, custom, extra_list)
        p = SimpleNamespace(min_price=min_price, min_dv=min_dv_m * 1e6, equity=equity, min_rs=min_rs,
                            atr_mult=atr_mult_t, risk=risk_t, max_position=max_pos_t, top=top_t,
                            extra=extra_list, check_earnings=check_earn, warn_days=int(warn_days))
        bar = st.progress(0.0, text="分析中…")
        try:
            st.session_state["res_trend"] = run_trend(prices, p, progress=bar)
        except RuntimeError as e:
            st.session_state.pop("res_trend", None)
            st.error(str(e))
        bar.empty()

    r = st.session_state.get("res_trend")
    if r:
        show_regime(r["regime"], r["breadth"])
        show_funnel(r["funnel"])
        if r["cand"].empty:
            st.info("今天沒有符合條件的股票，這本身也是訊號：寧可空手。")
        else:
            table = fmt(r["cand"], TREND_MAP, pct_cols=("r12_1", "r6"))
            st.dataframe(table)
            st.download_button("下載 Excel", to_excel_bytes({"趨勢候選": table}),
                               file_name="us_trend_candidates.xlsx", key="dl_trend",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            show_chart(r["charts"], "chart_trend")
        show_diag(r.get("diag"))

# ---------------- 分頁 2：多因子 ----------------
with tab_mf:
    st.markdown("每月用 **動能、品質、價值、成長** 四個因子綜合排名，挑一籃子股票；有換手緩衝、產業與細產業上限、反波動加權與大盤曝險。")
    c1, c2, c3, c4 = st.columns(4)
    w_mom = c1.slider("動能權重 %", 0, 100, 40, key="m_wm")
    w_q = c2.slider("品質權重 %", 0, 100, 25, key="m_wq")
    w_v = c3.slider("價值權重 %", 0, 100, 20, key="m_wv")
    w_g = c4.slider("成長權重 %", 0, 100, 15, key="m_wg")
    c1, c2, c3, c4 = st.columns(4)
    n_hold = c1.slider("持股檔數", 5, 30, 15, key="m_n")
    keep_rank = c2.slider("續抱排名門檻（前 N 名）", 10, 100, 30, key="m_keep")
    max_sec = c3.slider("同產業最多幾檔", 1, 10, 3, key="m_sec")
    max_ind = c4.slider("同細產業最多幾檔", 1, 5, 2, key="m_ind",
                        help="細產業例如「Oil & Gas Refining & Marketing」「Semiconductors」，可避免押在同一個題材。")
    c1, c2, c3, c4 = st.columns(4)
    max_fund = c1.slider("最多抓幾檔基本面", 20, 250, 80, key="m_fund")
    prefilter = c2.slider("動能前幾 % 才抓基本面", 20, 100, 50, key="m_pre")
    atr_mult_m = c3.number_input("停損 = 收盤 − N × ATR", 1.0, 6.0, 3.0, 0.5, key="m_atr")
    max_w = c4.number_input("單檔權重上限 %", 5.0, 50.0, 12.0, 1.0, key="m_maxw") / 100
    c1, c2, c3, c4 = st.columns(4)
    sector_neutral = c1.checkbox("品質/價值/成長在同產業內比較", value=True, key="m_sn")
    trend_filter = c2.checkbox("只考慮收盤 > 200 日線", value=True, key="m_tf")
    allow_unprof = c3.checkbox("允許近四季虧損的公司", value=False, key="m_up")
    skip_earn = c4.checkbox("財報在警示天數內者不新買進", value=False, key="m_skipearn",
                            help="已持有的仍會續抱並標示；只是不再新買進財報在即的股票。")
    st.text_area("上月持股（逗號或換行分隔，用來計算續抱／賣出）", key="m_prev")
    with st.expander("持股備份（CSV）"):
        st.file_uploader("匯入持股 CSV（欄位名稱 ticker / 代號 / symbol，或第一欄）", type="csv",
                         key="m_upload", on_change=import_holdings)
        st.download_button("匯出目前「上月持股」CSV",
                           pd.DataFrame({"ticker": parse_tickers(st.session_state.get("m_prev", ""))}).to_csv(index=False).encode("utf-8-sig"),
                           file_name="us_holdings.csv", key="dl_prev", mime="text/csv")
    st.caption("基本面抓取較慢（每檔約 1～2 秒，已用 4 條平行下載並快取 24 小時）。若出現抓取失敗，通常是 Yahoo 限流，稍後再試即可。")

    if st.button("開始多因子選股", type="primary", key="run_mf"):
        if w_mom + w_q + w_v + w_g == 0:
            st.error("四個權重不能全部為 0。")
        else:
            prices = prepare_prices(uni, custom, extra_list)
            p = SimpleNamespace(
                min_price=min_price, min_dv=min_dv_m * 1e6, equity=equity, trend_filter=trend_filter,
                allow_unprofitable=allow_unprof, w_mom=w_mom, w_quality=w_q, w_value=w_v, w_growth=w_g,
                sector_neutral=sector_neutral, prefilter_top=prefilter / 100, max_fundamental=max_fund,
                n_hold=n_hold, keep_rank=keep_rank, max_per_sector=max_sec, max_per_industry=max_ind,
                max_weight=max_w, atr_mult=atr_mult_m, prev=parse_tickers(st.session_state.get("m_prev", "")),
                extra=extra_list, warn_days=int(warn_days), skip_earnings=skip_earn,
            )
            bar = st.progress(0.0, text="準備抓取基本面…")
            try:
                st.session_state["res_mf"] = run_multifactor(prices, p, progress=bar)
            except RuntimeError as e:
                st.session_state.pop("res_mf", None)
                st.error(str(e))
            bar.empty()

    r = st.session_state.get("res_mf")
    if r:
        show_regime(r["regime"])
        show_funnel(r["funnel"])
        pf_cols = ["name", "sector", "industry", "action", "earn_flag", "next_earnings", "rank", "composite",
                   "mom", "quality", "value", "growth", "close", "weight_pct", "shares", "target_value", "stop",
                   "stop_risk_pct", "r12_1", "r6", "near_high", "vol60", "fscore", "earn_yield", "fcf_yield",
                   "roe", "last_date"]
        pct = ("r12_1", "r6", "vol60", "earn_yield", "fcf_yield", "roe")
        t_pf = fmt(r["pf"][[c for c in pf_cols if c in r["pf"].columns]], MF_MAP, pct_cols=pct)
        rk_cols = [c for c in pf_cols if c not in ("action", "weight_pct", "shares", "target_value", "stop", "stop_risk_pct")]
        t_rank = fmt(r["ranked"].head(100)[[c for c in rk_cols if c in r["ranked"].columns]], MF_MAP, pct_cols=pct)
        st.subheader("建議持股")
        st.dataframe(t_pf)
        st.button("✅ 把本次建議持股設為「上月持股」", on_click=adopt_holdings, args=(list(r["pf"].index),),
                  key="adopt_pf", help="下次執行時就會有續抱緩衝；同時寫入網址，加入書籤即可保存。")
        if not r["sells"].empty:
            st.subheader("賣出清單")
            st.dataframe(r["sells"])
        show_diag(r.get("diag"))
        with st.expander("排名前 100"):
            st.dataframe(t_rank)
        sheets = {"建議持股": t_pf, "排名前100": t_rank}
        if not r["sells"].empty:
            sheets["賣出清單"] = r["sells"].set_index("代號")
        if not r["diag"].empty:
            sheets["自選股診斷"] = r["diag"].set_index("代號")
        st.download_button("下載 Excel", to_excel_bytes(sheets), file_name="us_multifactor_candidates.xlsx",
                           key="dl_mf", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        show_chart(r["charts"], "chart_mf")

# ---------------- 分頁 3：預估上修 ----------------
with tab_rev:
    st.markdown("找出**分析師盈餘預估正在上調、股價動能也確認**的股票（財報預期動能），再用品質分數擋掉體質差的公司。"
                "預估資料來自 Yahoo，小型股常常沒有或家數很少，結果請當作研究線索。")
    c1, c2, c3 = st.columns(3)
    w_rev3 = c1.slider("預估上修權重 %", 0, 100, 50, key="r_wr")
    w_mom3 = c2.slider("價格動能權重 %", 0, 100, 25, key="r_wm")
    w_q3 = c3.slider("品質權重 %", 0, 100, 25, key="r_wq")
    c1, c2, c3, c4 = st.columns(4)
    n_hold3 = c1.slider("持股檔數", 5, 30, 12, key="r_n")
    max_sec3 = c2.slider("同產業最多幾檔", 1, 10, 3, key="r_sec")
    max_ind3 = c3.slider("同細產業最多幾檔", 1, 5, 2, key="r_ind")
    max_fund3 = c4.slider("最多抓幾檔預估與基本面", 20, 250, 80, key="r_fund")
    c1, c2, c3, c4 = st.columns(4)
    prefilter3 = c1.slider("動能前幾 % 才抓預估", 20, 100, 50, key="r_pre")
    atr_mult3 = c2.number_input("停損 = 收盤 − N × ATR", 1.0, 6.0, 2.5, 0.5, key="r_atr")
    risk3 = c3.number_input("單筆風險占總資金 %", 0.1, 5.0, 1.0, 0.1, key="r_risk") / 100
    max_pos3 = c4.number_input("單檔部位上限 %", 5.0, 100.0, 15.0, 5.0, key="r_maxpos") / 100
    c1, c2, c3, c4 = st.columns(4)
    sn3 = c1.checkbox("品質在同產業內比較", value=True, key="r_sn")
    tf3 = c2.checkbox("只考慮收盤 > 200 日線", value=True, key="r_tf")
    up3 = c3.checkbox("允許近四季虧損的公司", value=False, key="r_up")
    req3 = c4.checkbox("只留本年度 EPS 預估 30 日內上修者", value=True, key="r_req",
                       help="預估沒有上調的股票，即使動能強也不會入選。")
    c1, c2 = st.columns(2)
    min_analysts3 = c1.slider("最少分析師家數", 0, 15, 3, key="r_minan",
                              help="Yahoo 對小型股的預估常只有 1～2 位分析師，變動很大；低於這個家數的股票不評分。設 0 代表不限制。")
    skip3 = c2.checkbox("財報在警示天數內者不買進", value=False, key="r_skip")
    st.caption("每檔需要額外查詢分析師預估與基本面，約 1～2 秒，已平行下載並快取 24 小時。Yahoo 限流時請稍後再試。")
    with st.expander("🔍 資料自檢（確認 Yahoo 預估資料格式正常）"):
        st.caption("對 AAPL、MSFT、NVDA 各查一次，列出 Yahoo 回傳的資料表欄位與解析結果。"
                   "如果這裡顯示異常，下面的選股結果就不可靠。")
        if st.button("執行自檢", key="diag_rev"):
            with st.spinner("檢查中…"):
                st.dataframe(pd.DataFrame(diagnose_revisions()).set_index("代號"))

    if st.button("開始預估上修選股", type="primary", key="run_rev"):
        if w_rev3 + w_mom3 + w_q3 == 0:
            st.error("三個權重不能全部為 0。")
        else:
            prices = prepare_prices(uni, custom, extra_list)
            p = SimpleNamespace(
                min_price=min_price, min_dv=min_dv_m * 1e6, equity=equity, trend_filter=tf3,
                allow_unprofitable=up3, require_up=req3, min_analysts=min_analysts3,
                w_rev=w_rev3, w_mom=w_mom3, w_quality=w_q3,
                sector_neutral=sn3, prefilter_top=prefilter3 / 100, max_fundamental=max_fund3,
                n_hold=n_hold3, keep_rank=10_000, max_per_sector=max_sec3, max_per_industry=max_ind3,
                atr_mult=atr_mult3, risk=risk3, max_position=max_pos3, prev=[], extra=extra_list,
                warn_days=int(warn_days), skip_earnings=skip3,
            )
            bar = st.progress(0.0, text="準備抓取分析師預估…")
            try:
                st.session_state["res_rev"] = run_revisions(prices, p, progress=bar)
            except RuntimeError as e:
                st.session_state.pop("res_rev", None)
                st.error(str(e))
            bar.empty()

    r = st.session_state.get("res_rev")
    if r:
        show_regime(r["regime"])
        show_funnel(r["funnel"])
        rev_cols = ["name", "sector", "industry", "earn_flag", "next_earnings", "rank", "composite", "rev_score",
                    "mom", "quality", "close", "cy_rev30", "cy_rev90", "ny_rev30", "rev_breadth", "cy_up30",
                    "cy_down30", "n_analysts", "data_quality", "surprise_avg", "stop", "shares", "position_value",
                    "r12_1", "r6", "near_high", "fscore", "roe", "last_date"]
        pct3 = ("cy_rev30", "cy_rev90", "ny_rev30", "rev_breadth", "r12_1", "r6", "roe")
        t_pf3 = fmt(r["pf"][[c for c in rev_cols if c in r["pf"].columns]], REV_MAP, pct_cols=pct3)
        rk3 = [c for c in rev_cols if c not in ("stop", "shares", "position_value")]
        t_rank3 = fmt(r["ranked"].head(100)[[c for c in rk3 if c in r["ranked"].columns]], REV_MAP, pct_cols=pct3)
        if "data_quality" in r["ranked"].columns:
            dq = r["ranked"]["data_quality"].value_counts()
            st.caption("可評分股票的預估資料品質：" + "、".join(f"{k} {int(v)} 檔" for k, v in dq.items())
                       + "。「部分」代表只有 2～3 項預估訊號，排名參考性較低。")
        st.subheader("入選名單")
        if t_pf3.empty:
            st.info("目前沒有符合條件的股票。可以放寬條件（例如取消「只留預估上修者」）。")
        else:
            st.dataframe(t_pf3)
        show_diag(r.get("diag"))
        with st.expander("排名前 100"):
            st.dataframe(t_rank3)
        sheets3 = {"入選名單": t_pf3, "排名前100": t_rank3}
        if not r["diag"].empty:
            sheets3["自選股診斷"] = r["diag"].set_index("代號")
        st.download_button("下載 Excel", to_excel_bytes(sheets3), file_name="us_revisions_candidates.xlsx",
                           key="dl_rev", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        show_chart(r["charts"], "chart_rev")

with st.expander("策略說明與限制"):
    st.markdown(
        """
- **趨勢突破**：大盤環境 → 股價與流動性 → Minervini 趨勢模板（股價在 150/200 日線上、均線多頭排列、200 日線走升、距 52 週高點 25% 內、高於 52 週低點 30%）→ RS 評分 → ATR 停損與依風險計算的股數。
- **多因子**：動能（12-1 個月、6 個月、距 52 週高）、品質（ROE、毛利/總資產、F-Score、低負債）、價值（盈餘殖利率、預估盈餘殖利率、FCF 殖利率）、成長（營收與盈餘成長），以百分位排名加權；品質、價值、成長預設在同產業內比較。
- **產業與細產業上限**：同產業最多 N 檔、同細產業最多 M 檔，避免押在同一個題材（例如三檔煉油股）。
- **財報日**：來自 Yahoo，可能缺漏或只是預估日期；進場前請以公司公告為準。
- **自選股**：左側「額外加入個股」會併入股票池；結果下方的「診斷」會說明每一檔為何入選或被排除。
- **保存**：Streamlit 免費主機不會保存資料。請用「存到網址」加入書籤，或匯入/匯出 CSV。
- **預估上修**：用分析師對本年度、明年度 EPS 預估近 30/90 天的變化，以及 30 天內上修與下修的家數差，加上價格動能與品質分數綜合排名；入選後依 ATR 停損與單筆風險算股數，並套用大盤曝險。預估資料取自 Yahoo，小型股常缺漏或家數很少，所以有三道把關：分析師家數下限、至少 2 項預估訊號才評分、結果表標示「預估資料品質」；另加入近 4 季 EPS 驚喜當第二個佐證訊號。
- **大盤曝險**：SPY 在 200 日線上且 50 日線 > 200 日線 → 100%；只在 200 日線上 → 60%；否則 30%。
- **限制**：Yahoo 基本面不是歷史時點資料且偶有缺漏；本頁面不含回測（回測請用 Notebook 版本）；結果僅供研究，進場前請自行確認消息面與產業集中度。這不是投資建議。
"""
    )
