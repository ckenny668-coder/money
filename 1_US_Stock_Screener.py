# -*- coding: utf-8 -*-
"""
美股選股（Streamlit 網頁版）

兩個分頁：
  1. 趨勢突破掃描：大盤環境 → 流動性 → Minervini 趨勢模板 → RS 相對強度 → ATR 停損與建議股數（只用價格，速度快）
  2. 多因子選股：動能 + 品質 + 價值 + 成長綜合排名 → 換手緩衝 → 產業上限 → 反波動加權 → 大盤曝險

使用方式：
  - 放進現有 Streamlit 專案的 pages/ 資料夾（要和主程式同一層），側邊欄會多出「US Stock Screener」頁面
  - 或單獨執行：streamlit run 1_US_Stock_Screener.py
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
    raw = text.replace(",", " ").replace(";", " ").split()
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
# 基本面（含 Piotroski F-Score）
# ============================================================
FUND_COLS = ["name", "sector", "market_cap", "roe", "gross_prof", "debt_eq", "fscore",
             "earn_yield", "fwd_ey", "fcf_yield", "rev_g", "eps_g", "eps_ttm"]


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
        name=info.get("shortName"), sector=info.get("sector"), market_cap=mcap,
        roe=info.get("returnOnEquity"), debt_eq=info.get("debtToEquity"), eps_ttm=eps,
        earn_yield=earn_yield, fwd_ey=(1 / fpe) if fpe and fpe > 0 else None,
        fcf_yield=(fcf / mcap) if fcf is not None and mcap else None,
        rev_g=info.get("revenueGrowth"), eps_g=info.get("earningsGrowth"),
    )
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
    todo = [t for t in tickers if t not in store or now - store[t][0] > max_age_h * 3600]

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
        if c not in ("name", "sector"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ============================================================
# 因子評分與投組
# ============================================================
QUALITY = {"roe": 1, "gross_prof": 1, "fscore": 1, "debt_eq": -1}
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
    rank_pos = {t: i + 1 for i, t in enumerate(ranked.index)}
    chosen, sector_cnt = [], {}

    def can_add(t):
        sec = ranked.at[t, "sector"]
        return sec == "Unknown" or sector_cnt.get(sec, 0) < p.max_per_sector

    def add(t):
        chosen.append(t)
        sec = ranked.at[t, "sector"]
        sector_cnt[sec] = sector_cnt.get(sec, 0) + 1

    keep = sorted([t for t in prev if t in rank_pos and rank_pos[t] <= p.keep_rank], key=rank_pos.get)
    for t in keep:
        if len(chosen) < p.n_hold and can_add(t):
            add(t)
    for t in ranked.index:
        if len(chosen) >= p.n_hold:
            break
        if t not in chosen and can_add(t):
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
            why = "產業上限或名額限制"
        sells.append({"代號": t, "原因": why})
    return chosen, pd.DataFrame(sells, columns=["代號", "原因"])


# ============================================================
# 兩套選股流程
# ============================================================
def run_trend(prices, p):
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
    t = liquid
    liquid["trend_ok"] = (
        (t.close > t.sma150) & (t.close > t.sma200) & (t.sma150 > t.sma200)
        & (t.sma200 > t.sma200_prev) & (t.sma50 > t.sma150) & (t.sma50 > t.sma200)
        & (t.close > t.sma50) & (t.close >= 1.30 * t.lo52) & (t.close >= 0.75 * t.hi52)
    )
    funnel["通過趨勢模板（7 條件）"] = int(liquid["trend_ok"].sum())
    cand = liquid[liquid["trend_ok"] & (liquid["rs"] >= p.min_rs)].sort_values("rs", ascending=False)
    funnel[f"且 RS>={p.min_rs}"] = len(cand)
    cand = cand.head(p.top).copy()

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
    return {"cand": cand, "regime": reg, "funnel": funnel, "breadth": breadth,
            "charts": {t: prices[t]["Close"].iloc[-260:] for t in cand.index}}


def run_multifactor(prices, p, progress=None):
    tech = build_tech(prices)
    if tech.empty:
        raise RuntimeError("沒有任何股票有足夠的價格資料。")
    funnel = {"股票池（成功下載）": len([t for t in prices if t != BENCH]), "資料足夠(>=253日)": len(tech)}
    liquid = tech[(tech["close"] >= p.min_price) & (tech["dollar_vol"] >= p.min_dv)].copy()
    funnel["通過股價與流動性"] = len(liquid)
    if p.trend_filter:
        liquid = liquid[liquid["close"] > liquid["sma200"]].copy()
        funnel["收盤>200日線"] = len(liquid)
    if liquid.empty:
        raise RuntimeError("流動性與趨勢篩選後沒有任何股票（大盤可能很弱）。")

    liquid["mom"] = momentum_score(liquid["r12_1"], liquid["r6"], liquid["near_high"])
    cut = liquid["mom"].quantile(1 - p.prefilter_top)
    pre = liquid[liquid["mom"] >= cut].sort_values("mom", ascending=False).head(p.max_fundamental)
    funnel[f"動能前{int(p.prefilter_top * 100)}%（抓基本面）"] = len(pre)

    fund = get_fundamentals(list(pre.index), progress=progress)
    funnel["基本面取得成功"] = int(fund[["sector", "roe", "earn_yield"]].notna().any(axis=1).sum())
    data = pre.join(fund)
    data["sector"] = data["sector"].fillna("Unknown")
    if not p.allow_unprofitable:
        data = data[~(data["eps_ttm"].notna() & (data["eps_ttm"] <= 0))]
        funnel["排除近四季虧損"] = len(data)

    scored = score_factors(data, p)
    ranked = scored.dropna(subset=["composite"]).sort_values("composite", ascending=False).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    funnel["可評分（資料足夠）"] = len(ranked)
    if ranked.empty:
        raise RuntimeError("沒有任何股票可評分（基本面資料可能抓取失敗或被 Yahoo 限流，請稍後再試）。")

    chosen, sells = build_portfolio(ranked, p.prev, p)
    pf = ranked.loc[chosen].copy()
    reg = regime(prices[BENCH])
    w = inverse_vol_weights(pf["vol60"].fillna(pf["vol60"].median()), p.max_weight)
    pf["weight_pct"] = w * reg["scale"] * 100
    pf["target_value"] = p.equity * w * reg["scale"]
    pf["shares"] = (pf["target_value"] / pf["close"]).apply(math.floor).astype(int)
    pf["stop"] = pf["close"] - p.atr_mult * pf["atr"]
    pf["stop_risk_pct"] = pf["shares"] * (pf["close"] - pf["stop"]) / p.equity * 100
    pf["action"] = ["續抱" if t in p.prev else "買進" for t in pf.index]
    return {"pf": pf, "sells": sells, "ranked": ranked, "regime": reg, "funnel": funnel,
            "charts": {t: prices[t]["Close"].iloc[-260:] for t in pf.index}}


# ============================================================
# 顯示與匯出
# ============================================================
def fmt(df, mapping, pct_cols=(), ratio_cols=()):
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
    "shares": "建議股數", "position_value": "部位金額", "status": "狀態", "last_date": "資料日期",
}
MF_MAP = {
    "name": "公司", "sector": "產業", "action": "動作", "rank": "排名", "composite": "綜合分",
    "mom": "動能分", "quality": "品質分", "value": "價值分", "growth": "成長分", "close": "收盤價",
    "weight_pct": "建議權重%", "shares": "建議股數", "target_value": "建議金額", "stop": "建議停損價",
    "stop_risk_pct": "停損風險占資金%", "r12_1": "12-1月報酬%", "r6": "6月報酬%", "dist_high": "距52週高%",
    "vol60": "年化波動%", "fscore": "F-Score(0-9)", "earn_yield": "盈餘殖利率%", "fcf_yield": "FCF殖利率%",
    "roe": "ROE%", "last_date": "資料日期",
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


def get_universe(uni, custom):
    if uni == "自訂代號":
        tks = parse_tickers(custom)
        if not tks:
            st.error("請在左側輸入至少一個代號。")
            st.stop()
        return tks
    try:
        return load_index_universe("sp500" if uni == "S&P 500" else "sp1500")
    except Exception as e:  # noqa: BLE001
        st.error(f"無法取得指數成分股名單（{e}）。可改用「自訂代號」。")
        st.stop()


def prepare_prices(uni, custom):
    tickers = get_universe(uni, custom)
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
st.title("📈 美股選股")
st.caption("資料來源：Yahoo Finance（免費、有延遲、偶有缺漏）。本工具為篩選與研究用途，不是投資建議。")

with st.sidebar:
    st.header("股票池與共同設定")
    uni = st.selectbox("股票池", ["S&P 500", "S&P 1500", "自訂代號"], key="uni")
    custom = st.text_area("自訂代號（逗號或換行分隔）", "NVDA, AAPL, MSFT, AMZN, META, GOOGL, AVGO, LLY, COST, NFLX",
                          disabled=(uni != "自訂代號"), key="custom")
    min_price = st.number_input("最低股價（美元）", min_value=1.0, value=10.0, step=1.0, key="min_price")
    min_dv_m = st.number_input("最低日均成交額（百萬美元）", min_value=0.0, value=20.0, step=5.0, key="min_dv")
    equity = st.number_input("總資金（美元）", min_value=1000, value=100000, step=10000, key="equity")
    st.caption("首次執行會下載約 2 年日線，之後 6 小時內重跑會用快取。S&P 1500 較慢。")

tab_trend, tab_mf = st.tabs(["🚀 趨勢突破掃描", "🧮 多因子選股（月度）"])

# ---------------- 分頁 1：趨勢突破 ----------------
with tab_trend:
    st.markdown("找出**已經處於強勢上升趨勢**的股票：Minervini 趨勢模板 + RS 相對強度，再用 ATR 算停損與建議股數。只用價格，速度快。")
    c1, c2, c3, c4 = st.columns(4)
    min_rs = c1.slider("最低 RS 評分", 50, 99, 80, key="t_rs")
    atr_mult_t = c2.number_input("停損 = 收盤 − N × ATR", 1.0, 6.0, 2.5, 0.5, key="t_atr")
    risk_t = c3.number_input("單筆風險占總資金 %", 0.1, 5.0, 1.0, 0.1, key="t_risk") / 100
    max_pos_t = c4.number_input("單檔部位上限 %", 5.0, 100.0, 20.0, 5.0, key="t_maxpos") / 100
    top_t = st.slider("最多顯示幾檔", 5, 100, 30, key="t_top")

    if st.button("開始趨勢掃描", type="primary", key="run_trend"):
        prices = prepare_prices(uni, custom)
        p = SimpleNamespace(min_price=min_price, min_dv=min_dv_m * 1e6, equity=equity, min_rs=min_rs,
                            atr_mult=atr_mult_t, risk=risk_t, max_position=max_pos_t, top=top_t)
        try:
            st.session_state["res_trend"] = run_trend(prices, p)
        except RuntimeError as e:
            st.session_state.pop("res_trend", None)
            st.error(str(e))

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

# ---------------- 分頁 2：多因子 ----------------
with tab_mf:
    st.markdown("每月用 **動能、品質、價值、成長** 四個因子綜合排名，挑一籃子股票；有換手緩衝、產業上限、反波動加權與大盤曝險。")
    c1, c2, c3, c4 = st.columns(4)
    w_mom = c1.slider("動能權重 %", 0, 100, 40, key="m_wm")
    w_q = c2.slider("品質權重 %", 0, 100, 25, key="m_wq")
    w_v = c3.slider("價值權重 %", 0, 100, 20, key="m_wv")
    w_g = c4.slider("成長權重 %", 0, 100, 15, key="m_wg")
    c1, c2, c3, c4 = st.columns(4)
    n_hold = c1.slider("持股檔數", 5, 30, 15, key="m_n")
    keep_rank = c2.slider("續抱排名門檻（前 N 名）", 10, 100, 30, key="m_keep")
    max_sec = c3.slider("同產業最多幾檔", 1, 10, 3, key="m_sec")
    max_fund = c4.slider("最多抓幾檔基本面", 20, 250, 80, key="m_fund")
    c1, c2, c3, c4 = st.columns(4)
    prefilter = c1.slider("動能前幾 % 才抓基本面", 20, 100, 50, key="m_pre")
    atr_mult_m = c2.number_input("停損 = 收盤 − N × ATR", 1.0, 6.0, 3.0, 0.5, key="m_atr")
    max_w = c3.number_input("單檔權重上限 %", 5.0, 50.0, 12.0, 1.0, key="m_maxw") / 100
    sector_neutral = c4.checkbox("品質/價值/成長在同產業內比較", value=True, key="m_sn")
    c1, c2 = st.columns(2)
    trend_filter = c1.checkbox("只考慮收盤 > 200 日線", value=True, key="m_tf")
    allow_unprof = c2.checkbox("允許近四季虧損的公司", value=False, key="m_up")
    prev_text = st.text_area("上月持股（可選，逗號或換行分隔，用來計算續抱／賣出）", "", key="m_prev")
    st.caption("基本面抓取較慢（每檔約 1～2 秒，已用 4 條平行下載並快取 24 小時）。若出現抓取失敗，通常是 Yahoo 限流，稍後再試即可。")

    if st.button("開始多因子選股", type="primary", key="run_mf"):
        if w_mom + w_q + w_v + w_g == 0:
            st.error("四個權重不能全部為 0。")
        else:
            prices = prepare_prices(uni, custom)
            p = SimpleNamespace(
                min_price=min_price, min_dv=min_dv_m * 1e6, equity=equity, trend_filter=trend_filter,
                allow_unprofitable=allow_unprof, w_mom=w_mom, w_quality=w_q, w_value=w_v, w_growth=w_g,
                sector_neutral=sector_neutral, prefilter_top=prefilter / 100, max_fundamental=max_fund,
                n_hold=n_hold, keep_rank=keep_rank, max_per_sector=max_sec, max_weight=max_w,
                atr_mult=atr_mult_m, prev=parse_tickers(prev_text),
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
        pf_cols = ["name", "sector", "action", "rank", "composite", "mom", "quality", "value", "growth", "close",
                   "weight_pct", "shares", "target_value", "stop", "stop_risk_pct", "r12_1", "r6", "near_high",
                   "vol60", "fscore", "earn_yield", "fcf_yield", "roe", "last_date"]
        pct = ("r12_1", "r6", "vol60", "earn_yield", "fcf_yield", "roe")
        t_pf = fmt(r["pf"][[c for c in pf_cols if c in r["pf"].columns]], MF_MAP, pct_cols=pct)
        rk_cols = [c for c in pf_cols if c not in ("action", "weight_pct", "shares", "target_value", "stop", "stop_risk_pct")]
        t_rank = fmt(r["ranked"].head(100)[[c for c in rk_cols if c in r["ranked"].columns]], MF_MAP, pct_cols=pct)
        st.subheader("建議持股")
        st.dataframe(t_pf)
        if not r["sells"].empty:
            st.subheader("賣出清單")
            st.dataframe(r["sells"])
        with st.expander("排名前 100"):
            st.dataframe(t_rank)
        sheets = {"建議持股": t_pf, "排名前100": t_rank}
        if not r["sells"].empty:
            sheets["賣出清單"] = r["sells"].set_index("代號")
        st.download_button("下載 Excel", to_excel_bytes(sheets), file_name="us_multifactor_candidates.xlsx",
                           key="dl_mf", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption("提示：下單後，把「建議持股」的代號貼到「上月持股」欄，下個月才會有換手緩衝。")
        show_chart(r["charts"], "chart_mf")

with st.expander("策略說明與限制"):
    st.markdown(
        """
- **趨勢突破**：大盤環境 → 股價與流動性 → Minervini 趨勢模板（股價在 150/200 日線上、均線多頭排列、200 日線走升、距 52 週高點 25% 內、高於 52 週低點 30%）→ RS 評分 → ATR 停損與依風險計算的股數。
- **多因子**：動能（12-1 個月、6 個月、距 52 週高）、品質（ROE、毛利/總資產、F-Score、低負債）、價值（盈餘殖利率、預估盈餘殖利率、FCF 殖利率）、成長（營收與盈餘成長），以百分位排名加權；品質、價值、成長預設在同產業內比較。
- **大盤曝險**：SPY 在 200 日線上且 50 日線 > 200 日線 → 100%；只在 200 日線上 → 60%；否則 30%。
- **限制**：Yahoo 基本面不是歷史時點資料且偶有缺漏；本頁面不含回測（回測請用 Notebook 版本）；結果僅供研究，進場前請自行確認財報日、消息面與產業集中度。這不是投資建議。
"""
    )
