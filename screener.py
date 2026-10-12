
# -*- coding: utf-8 -*-
"""
GitHub Actions 版 - 每天跑，輸出 JSON 給 GitHub Pages 手機版
用法: python screener.py --universe sp1500 --out data/
"""
import argparse, json, math, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yfinance as yf
import requests

BENCH = "SPY"
WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

def _wiki_tickers(url):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    for tbl in pd.read_html(StringIO(resp.text)):
        cols = [str(c) for c in tbl.columns]
        for key in ("Symbol", "Ticker symbol", "Ticker"):
            if key in cols:
                syms = tbl[key].astype(str).str.strip().str.upper()
                return [s.replace(".", "-") for s in syms if s and s != "NAN"]
    raise RuntimeError(url)

def load_index_universe(name):
    parts = ["sp500"] if name == "sp500" else ["sp500", "sp400", "sp600"]
    out = []
    for p in parts:
        out += _wiki_tickers(WIKI[p])
    return sorted(set(out))

def _download_chunk(chunk, period):
    raw = yf.download(list(chunk), period=period, interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    out = {}
    if raw is None or raw.empty:
        return out
    # 相容兩種 MultiIndex 順序
    if isinstance(raw.columns, pd.MultiIndex):
        for t in chunk:
            try:
                d = raw.xs(t, level=1, axis=1) if t in raw.columns.get_level_values(1) else raw[t]
                d = d.dropna(subset=["Close"])
                if len(d): out[t] = d
            except Exception:
                continue
    else:
        d = raw.dropna(subset=["Close"])
        if len(d): out[chunk[0]] = d
    return out

def load_prices(tickers, period="2y", batch=80):
    out = {}
    chunks = [tuple(tickers[i:i+batch]) for i in range(0, len(tickers), batch)]
    for k,ch in enumerate(chunks,1):
        try:
            out.update(_download_chunk(ch, period))
            time.sleep(1) # 避免 429
        except Exception as e:
            print(f"batch {k} fail {e}")
    return out

def analyze_price(df):
    d = df.dropna(subset=["Close"])
    if len(d) < 253: return None
    c,h,l,v = d["Close"], d["High"], d["Low"], d["Volume"]
    last = float(c.iloc[-1])
    prev_c = c.shift(1)
    tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    hi52, lo52 = float(h.iloc[-252:].max()), float(l.iloc[-252:].min())
    sma200 = c.rolling(200).mean()
    return {
        "last_date": str(d.index[-1].date()),
        "close": last,
        "sma50": float(c.rolling(50).mean().iloc[-1]),
        "sma150": float(c.rolling(150).mean().iloc[-1]),
        "sma200": float(sma200.iloc[-1]),
        "sma200_prev": float(sma200.iloc[-22]),
        "hi52": hi52, "lo52": lo52,
        "near_high": last/hi52,
        "atr": float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]),
        "dollar_vol": float((c*v).rolling(50).mean().iloc[-1]),
        "vol_ratio": float(v.iloc[-5:].mean() / v.rolling(50).mean().iloc[-1]),
        "vol60": float(c.pct_change().iloc[-60:].std()*math.sqrt(252)),
        "r12_1": float(c.iloc[-22])/float(c.iloc[-253])-1,
        "r6": last/float(c.iloc[-127])-1,
        "r3": last/float(c.iloc[-64])-1,
    }

def build_tech(prices):
    rows={}
    for t,d in prices.items():
        if t==BENCH: continue
        r=analyze_price(d)
        if r: rows[t]=r
    return pd.DataFrame.from_dict(rows, orient="index")

def regime(spy_df):
    c=spy_df["Close"].dropna()
    s50,s200=c.rolling(50).mean(), c.rolling(200).mean()
    last=float(c.iloc[-1])
    above=last>float(s200.iloc[-1])
    golden=float(s50.iloc[-1])>float(s200.iloc[-1])
    if above and golden: label,scale="進攻（多頭）",1.0
    elif above: label,scale="謹慎（僅在200日線上）",0.6
    else: label,scale="防守（SPY 在 200 日線下）",0.3
    return {"label":label,"scale":scale,"spy_close":last,"spy_sma50":float(s50.iloc[-1]),"spy_sma200":float(s200.iloc[-1])}

def run_trend(prices, p):
    tech=build_tech(prices)
    liquid=tech[(tech["close"]>=p.min_price)&(tech["dollar_vol"]>=p.min_dv)].copy()
    breadth=float((liquid["close"]>liquid["sma200"]).mean()*100) if len(liquid) else 0
    raw=0.4*liquid["r12_1"].rank(pct=True)+0.3*liquid["r6"].rank(pct=True)+0.3*liquid["r3"].rank(pct=True)
    liquid["rs"]=(raw.rank(pct=True)*98+1).round(0)
    t=liquid
    liquid["trend_ok"]= (t.close>t.sma150)&(t.close>t.sma200)&(t.sma150>t.sma200)&(t.sma200>t.sma200_prev)&(t.sma50>t.sma150)&(t.sma50>t.sma200)&(t.close>t.sma50)&(t.close>=1.30*t.lo52)&(t.close>=0.75*t.hi52)
    cand=liquid[liquid["trend_ok"]&(liquid["rs"]>=p.min_rs)].sort_values("rs",ascending=False).head(p.top).copy()
    reg=regime(prices[BENCH])
    risk_pct=p.risk*reg["scale"]
    cand["stop"]=cand["close"]-p.atr_mult*cand["atr"]
    per_share=(cand["close"]-cand["stop"]).clip(lower=0.01)
    by_risk=(p.equity*risk_pct/per_share).apply(math.floor)
    by_cap=(p.equity*p.max_position/cand["close"]).apply(math.floor)
    cand["shares"]=np.minimum(by_risk,by_cap).astype(int)
    cand["dist_high"]=(1-cand["near_high"])*100
    cand["ext50"]=(cand["close"]/cand["sma50"]-1)*100
    cand["status"]=cand.apply(lambda r: "過度延伸，等回檔" if r["ext50"]>20 else ("接近新高，可關注突破" if r["dist_high"]<=5 else "趨勢中，等回檔至50日線附近"), axis=1)
    cand["ticker"]=cand.index
    cand["close_history"]=[prices[t]["Close"].iloc[-60:].tolist()[-20:] for t in cand.index] # 只存20天給手機畫 sparkline
    highs=int((liquid["near_high"]>=0.98).sum()); lows=int((liquid["close"]<=liquid["lo52"]*1.02).sum())
    return {"cand":cand, "regime":reg, "breadth":breadth, "highs":highs, "lows":lows, "funnel":{"下載":len(prices)-1,"足夠資料":len(tech),"流動性":len(liquid),"趨勢":int(liquid["trend_ok"].sum()),"RS":len(cand)}}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--out", default="data")
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tickers=load_index_universe(args.universe)
    print(f"Universe {args.universe}: {len(tickers)}")
    prices=load_prices(sorted(set(tickers+[BENCH])), period="2y")
    if BENCH not in prices: raise SystemExit("SPY download fail")
    p=SimpleNamespace(min_price=10, min_dv=20e6, equity=100000, min_rs=80, atr_mult=2.5, risk=0.01, max_position=0.2, top=30)
    res=run_trend(prices,p)
    # 輸出 JSON
    out_cand=res["cand"].reset_index(drop=True).to_dict(orient="records")
    with open(f"{args.out}/trend.json","w",encoding="utf-8") as f:
        json.dump({"regime":res["regime"],"breadth":res["breadth"],"highs":res["highs"],"lows":res["lows"],"funnel":res["funnel"],"stocks":out_cand,"updated":time.strftime("%Y-%m-%d %H:%M")}, f, ensure_ascii=False, indent=2)
    print(f"Saved {args.out}/trend.json {len(out_cand)} stocks")
