"""
強勢股 + 整理轉強 掃描器（階段 1）
==================================
對應貼文的邏輯：
  1. 近 1 / 3 / 6 個月漲幅排名，漲幅 > 30%，排名前 20~30%，剔除成交稀少者
  2. 上漲後進入整理：回調低點墊高、波動收斂、股價站上並走平/向上的 9、21 EMA
  3. 週線背景：收盤在 10 週均線之上

設計原則（避免回測常見的錯）：
  * 所有計算只用 as_of 當天(含)以前的資料，沒有 look-ahead。
  * 百分位排名是「同一天、整個股票池」的橫斷面排名，不是用整段歷史一起排。
  * 波段低點(pivot low)需要右側 k 根 K 線確認，最後 k 根不會被當成低點。
  * 每個條件都輸出成獨立的布林欄位，方便做消融測試(拿掉某條件看績效變化)。

用法：
  python momentum_scanner.py --period 2y --out watchlist.csv
  python momentum_scanner.py --as-of 2026-10-12 --out watchlist.csv
"""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 參數：除了貼文明確寫的(30%、前20~30%、9/21 EMA)，其餘是我的預設值，可調
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    # 流動性
    min_price: float = 10.0
    min_dollar_vol: float = 20e6          # 20 日平均成交金額(美元)
    # 強勢篩選
    lookbacks: tuple = (21, 63, 126)      # 約 1 / 3 / 6 個月(交易日)
    top_pct: float = 0.70                 # 百分位 >= 0.70 即前 30%
    min_top_hits: int = 2                 # 三個週期中至少幾個進前 30%
    min_gain: float = 0.30                # 漲幅門檻
    gain_windows: tuple = (63, 126)       # 3 或 6 個月任一個 > min_gain
    # 整理型態
    high_lookback: int = 126              # 找近期高點的回看天數
    min_depth: float = 0.10               # 高點後最大回檔下限
    max_depth: float = 0.25               # 高點後最大回檔上限
    max_off_high: float = 0.15            # 現價距高點不得超過(已在轉強)
    min_days_since_high: int = 5          # 高點後至少整理幾天
    pivot_k: int = 5                      # 波段低點左右各 k 根
    min_higher_low: float = 0.01          # 新低點至少比前低點高 1%
    ema_fast: int = 9
    ema_slow: int = 21
    ema_slope_days: int = 3
    ema_flat_tol: float = 0.005           # EMA 較 3 日前跌幅不超過 0.5% 視為走平
    atr_short: int = 10
    atr_long: int = 30
    atr_ratio_max: float = 0.8            # ATR(10)/ATR(30) 小於此值視為收斂
    weekly_sma: int = 10                  # 週線 10 週均線
    # Minervini 趨勢模板（8 條）
    tt_ma_slope_days: int = 21            # 200 日均線至少上升一個月
    tt_min_above_low: float = 0.30        # 距 52 週低點至少 +30%
    tt_max_below_high: float = 0.25       # 距 52 週高點不超過 25%
    tt_rs_min: float = 0.70               # RS 百分位至少 70
    require_trend_template: bool = False  # True：通過/差一項都必須同時符合趨勢模板


@dataclass
class Prices:
    """寬表：index=日期, columns=代號。價格請用還原價(adjusted)。"""
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame

    def upto(self, as_of) -> "Prices":
        if as_of is None:
            return self
        t = pd.Timestamp(as_of)
        return Prices(self.close.loc[:t], self.high.loc[:t],
                      self.low.loc[:t], self.volume.loc[:t])


# --------------------------------------------------------------------------
# 基礎元件
# --------------------------------------------------------------------------
def pivot_lows(low: np.ndarray, k: int) -> list[int]:
    """回傳波段低點的位置：該根最低價是左右各 k 根內的最小值。
    位置 i 需要 i+k 存在才能確認，所以最後 k 根不會出現(無未來資料)。
    距離 <= k 的兩個低點必然等值(平底)，只保留第一個。"""
    n = len(low)
    if n < 2 * k + 1:
        return []
    win = np.lib.stride_tricks.sliding_window_view(low, 2 * k + 1)
    centers = np.arange(k, n - k)
    idx = centers[win[:, k] <= win.min(axis=1)]
    out: list[int] = []
    for i in idx:
        if out and i - out[-1] <= k:
            continue
        out.append(int(i))
    return out


def consolidation_features(close: pd.Series, high: pd.Series, low: pd.Series,
                           volume: pd.Series, p: Params) -> dict:
    """單一股票的整理型態特徵。輸入需已截到 as_of，且按日期排序。"""
    keep = close.notna() & high.notna() & low.notna()
    close, high, low = close[keep], high[keep], low[keep]
    volume = volume[keep].fillna(0.0)

    res = dict(depth=np.nan, off_high=np.nan, days_since_high=np.nan,
               pivot_prev=np.nan, pivot_last=np.nan, atr_ratio=np.nan,
               vol_ratio=np.nan, depth_ok=False, higher_low=False,
               contraction=False, above_emas=False, ema_rising=False,
               weekly_ok=False, consolidating=False)

    need = max(p.high_lookback, p.atr_long, p.ema_slow * 3) + 1
    if len(close) < need:
        return res

    c = close.to_numpy(float)
    h = high.to_numpy(float)
    l = low.to_numpy(float)
    n = p.high_lookback

    # EMA 位置與斜率
    ema_f = close.ewm(span=p.ema_fast, adjust=False).mean().to_numpy()
    ema_s = close.ewm(span=p.ema_slow, adjust=False).mean().to_numpy()
    s, tol = p.ema_slope_days, p.ema_flat_tol
    res["above_emas"] = bool(c[-1] > ema_f[-1] and c[-1] > ema_s[-1])
    res["ema_rising"] = bool(ema_f[-1] >= ema_f[-1 - s] * (1 - tol)
                             and ema_s[-1] >= ema_s[-1 - s] * (1 - tol))

    # 近期高點與回檔幅度(高點用最高價、回檔低點用最低價)
    hw, lw = h[-n:], l[-n:]
    pk = int(np.argmax(hw))
    peak = hw[pk]
    trough = lw[pk:].min()
    res["depth"] = (peak - trough) / peak
    res["off_high"] = 1 - c[-1] / peak
    res["days_since_high"] = len(hw) - 1 - pk
    res["depth_ok"] = bool(p.min_depth <= res["depth"] <= p.max_depth
                           and res["off_high"] <= p.max_off_high
                           and res["days_since_high"] >= p.min_days_since_high)

    # 回調低點是否墊高：最後兩個已確認的波段低點，最後一個須在高點之後
    piv = pivot_lows(lw, p.pivot_k)
    if len(piv) >= 2 and piv[-1] > pk:
        prev_low, last_low = lw[piv[-2]], lw[piv[-1]]
        res["pivot_prev"], res["pivot_last"] = prev_low, last_low
        res["higher_low"] = bool(last_low >= prev_low * (1 + p.min_higher_low)
                                 and c[-1] > last_low)

    # 波動收斂：ATR(短)/ATR(長)
    pc = c[:-1]
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - pc), np.abs(l[1:] - pc)])
    atr_l = tr[-p.atr_long:].mean()
    if atr_l > 0:
        res["atr_ratio"] = tr[-p.atr_short:].mean() / atr_l
        res["contraction"] = bool(res["atr_ratio"] < p.atr_ratio_max)

    # 量能(僅供參考，不參與篩選)
    v = volume.to_numpy(float)
    if len(v) >= 50 and v[-50:].mean() > 0:
        res["vol_ratio"] = v[-10:].mean() / v[-50:].mean()

    # 週線：收盤在 N 週均線之上(最後一週可能是未完成的週，用到 as_of 為止的資料)
    if isinstance(close.index, pd.DatetimeIndex):
        wk = close.resample("W-FRI").last().dropna()
        if len(wk) >= p.weekly_sma:
            res["weekly_ok"] = bool(wk.iloc[-1] > wk.iloc[-p.weekly_sma:].mean())

    res["consolidating"] = bool(res["depth_ok"] and res["higher_low"]
                                and res["contraction"] and res["above_emas"]
                                and res["ema_rising"] and res["weekly_ok"])
    return res


TT_LABELS = {
    "tt_1": "價>150/200MA", "tt_2": "150MA>200MA", "tt_3": "200MA上升",
    "tt_4": "50MA>150/200MA", "tt_5": "價>50MA", "tt_6": "距52週低未達+30%",
    "tt_7": "距52週高超過25%", "tt_8": "RS不足",
}


def trend_template_features(close: pd.Series, high: pd.Series, low: pd.Series,
                            p: Params) -> dict:
    """Minervini 趨勢模板前 7 條(價量均線條件)；第 8 條 RS 是橫斷面排名，在 scan() 算。
    資料不足 252 根日線時全部為 False。"""
    res = {f"tt_{i}": False for i in range(1, 8)}
    keep = close.notna() & high.notna() & low.notna()
    close, high, low = close[keep], high[keep], low[keep]
    if len(close) < 252:
        return res
    c = close.to_numpy(float)
    ma50, ma150, ma200 = (close.rolling(n).mean().to_numpy() for n in (50, 150, 200))
    s = p.tt_ma_slope_days
    lo52 = low.to_numpy(float)[-252:].min()
    hi52 = high.to_numpy(float)[-252:].max()
    res["tt_1"] = bool(c[-1] > ma150[-1] and c[-1] > ma200[-1])
    res["tt_2"] = bool(ma150[-1] > ma200[-1])
    res["tt_3"] = bool(ma200[-1] > ma200[-1 - s])
    res["tt_4"] = bool(ma50[-1] > ma150[-1] and ma50[-1] > ma200[-1])
    res["tt_5"] = bool(c[-1] > ma50[-1])
    res["tt_6"] = bool(c[-1] >= lo52 * (1 + p.tt_min_above_low))
    res["tt_7"] = bool(c[-1] >= hi52 * (1 - p.tt_max_below_high))
    return res


# --------------------------------------------------------------------------
# 掃描主流程
# --------------------------------------------------------------------------
def scan(prices: Prices, as_of=None, params: Params | None = None,
         info: pd.DataFrame | None = None) -> pd.DataFrame:
    """回傳每檔股票一列，含所有條件欄位與 passed。
    info: 可選，index=代號，欄位如 sector / industry，會併入結果。"""
    p = params or Params()
    px = prices.upto(as_of)
    close = px.close
    max_lb = max(p.lookbacks)
    if len(close) < max_lb + 1:
        raise ValueError(f"資料不足：至少需要 {max_lb + 1} 根日線")

    last = close.iloc[-1]                    # 最後一天沒有價格的(停牌/下市)會是 NaN
    out = pd.DataFrame(index=close.columns)
    out["close"] = last

    # 橫斷面報酬與百分位(同一天、整個股票池)
    for lb in p.lookbacks:
        out[f"ret_{lb}"] = last / close.iloc[-1 - lb] - 1
        out[f"pct_{lb}"] = out[f"ret_{lb}"].rank(pct=True)
    pct_cols = [f"pct_{lb}" for lb in p.lookbacks]
    out["top_hits"] = (out[pct_cols] >= p.top_pct).sum(axis=1)
    out["avg_pct"] = out[pct_cols].mean(axis=1)

    gain_ok = pd.concat([out[f"ret_{w}"] > p.min_gain for w in p.gain_windows],
                        axis=1).any(axis=1)
    out["strong"] = gain_ok & (out["top_hits"] >= p.min_top_hits) & last.notna()

    # 流動性
    out["dollar_vol"] = (close * px.volume).iloc[-20:].mean()
    out["liquid"] = (last >= p.min_price) & (out["dollar_vol"] >= p.min_dollar_vol)

    # 整理型態(逐檔)
    feats = {
        t: consolidation_features(close[t], px.high[t], px.low[t], px.volume[t], p)
        for t in close.columns
    }
    out = out.join(pd.DataFrame.from_dict(feats, orient="index"))

    # Minervini 趨勢模板：前 7 條逐檔算，第 8 條 RS 為同一天全股票池的百分位
    # RS 分數採 IBD 式近似：近 3 個月 40%，6 / 9 / 12 個月各 20%（皆為累積報酬）
    tt = {t: trend_template_features(close[t], px.high[t], px.low[t], p) for t in close.columns}
    out = out.join(pd.DataFrame.from_dict(tt, orient="index"))
    if len(close) >= 253:
        r = {n: last / close.iloc[-1 - n] - 1 for n in (63, 126, 189, 252)}
        out["rs_rating"] = (0.4 * r[63] + 0.2 * r[126] + 0.2 * r[189] + 0.2 * r[252]).rank(pct=True)
    else:
        out["rs_rating"] = np.nan
    out["tt_8"] = out["rs_rating"] >= p.tt_rs_min
    tt_keys = [f"tt_{i}" for i in range(1, 9)]
    tflags = out[tt_keys].astype(bool)
    out["tt_count"] = tflags.sum(axis=1)
    out["tt_missing"] = tflags.apply(lambda r: ",".join(TT_LABELS[c] for c in tt_keys if not r[c]), axis=1)
    out["trend_template"] = out["tt_count"] == 8
    gate = out["trend_template"] if p.require_trend_template else True

    out["passed"] = out["liquid"] & out["strong"] & out["consolidating"] & gate
    # 差一項：流動性、強勢都過，整理型態六項只差一項(列出差哪一項，供人工看圖)
    cons = ["depth_ok", "higher_low", "contraction", "above_emas", "ema_rising", "weekly_ok"]
    flags = out[cons].astype(bool)
    out["miss_n"] = len(cons) - flags.sum(axis=1)
    out["missing"] = flags.apply(lambda r: ",".join(c for c in cons if not r[c]), axis=1)
    out["near_miss"] = out["liquid"] & out["strong"] & (out["miss_n"] == 1) & gate
    # 排名模式：流動性、強勢(與趨勢模板，若要求)為硬條件，整理型態六項改成分數(0~6)
    out["cons_score"] = len(cons) - out["miss_n"]
    out["candidate"] = out["liquid"] & out["strong"] & gate
    if info is not None:
        out = out.join(info, how="left")
    out.index.name = "ticker"
    return out.sort_values(["passed", "avg_pct"], ascending=[False, False])


# --------------------------------------------------------------------------
# 資料載入(需要網路；yfinance 與 Wikipedia)
# --------------------------------------------------------------------------
_UA = {"User-Agent": "Mozilla/5.0 (momentum-scanner)"}
_WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def load_sp1500() -> pd.DataFrame:
    """從 Wikipedia 取 S&P 500/400/600 成分股與 GICS 產業。
    注意：只代表「現在」的成分股，用來回測會有存活者偏差。"""
    import requests
    frames = []
    for name, url in _WIKI.items():
        html = requests.get(url, headers=_UA, timeout=30).text
        for t in pd.read_html(io.StringIO(html)):
            cols = {c: str(c) for c in t.columns}
            t = t.rename(columns=cols)
            sym = next((c for c in t.columns if c in ("Symbol", "Ticker symbol", "Ticker")), None)
            if sym and "GICS Sector" in t.columns:
                d = pd.DataFrame({
                    "ticker": t[sym].astype(str).str.replace(".", "-", regex=False),
                    "sector": t["GICS Sector"],
                    "industry": t.get("GICS Sub-Industry"),
                    "index": name,
                })
                frames.append(d)
                break
        else:
            raise RuntimeError(f"找不到 {name} 的成分股表格，Wikipedia 版面可能改了")
    uni = pd.concat(frames).drop_duplicates("ticker").set_index("ticker")
    return uni


def download_prices(tickers: list[str], period: str = "2y",
                    batch_size: int = 100) -> Prices:
    import yfinance as yf
    parts = {k: [] for k in ("Close", "High", "Low", "Volume")}
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i:i + batch_size]
        df = yf.download(chunk, period=period, auto_adjust=True, progress=False,
                         threads=True, group_by="column")
        if df is None or df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):       # 只有一檔時
            df.columns = pd.MultiIndex.from_product([df.columns, chunk[:1]])
        for k in parts:
            if k in df.columns.get_level_values(0):
                parts[k].append(df[k])
    if not parts["Close"]:
        raise RuntimeError("沒有下載到任何資料，請檢查網路或代號")

    def merge(k):
        w = pd.concat(parts[k], axis=1)
        w = w.loc[:, ~w.columns.duplicated()]
        if getattr(w.index, "tz", None) is not None:
            w.index = w.index.tz_localize(None)
        return w.dropna(axis=1, how="all").sort_index()

    close = merge("Close")
    cols = close.columns
    return Prices(close, merge("High").reindex(columns=cols),
                  merge("Low").reindex(columns=cols),
                  merge("Volume").reindex(columns=cols))


def funnel(res: pd.DataFrame, with_tt: bool = False) -> pd.DataFrame:
    """逐層過濾的剩餘檔數，用來看是哪個條件把名單擋空。"""
    steps = [("股票池", None), ("流動性", "liquid"), ("強勢(漲幅+排名)", "strong"),
             ("回檔10~25%且已整理", "depth_ok"), ("回調低點墊高", "higher_low"),
             ("波動收斂", "contraction"), ("站上9/21 EMA", "above_emas"),
             ("EMA走平或向上", "ema_rising"), ("週線在10週線上", "weekly_ok")]
    if with_tt:
        steps.append(("Minervini趨勢模板(8條)", "trend_template"))
    mask = pd.Series(True, index=res.index)
    rows = []
    for name, col in steps:
        if col:
            mask &= res[col].astype(bool)
        rows.append((name, int(mask.sum())))
    return pd.DataFrame(rows, columns=["條件(累積)", "剩餘檔數"])


def run_cli(load_universe, base_params: Params, desc: str = "強勢股 + 整理轉強掃描"):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--period", default="2y")
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD，預設最新一天")
    ap.add_argument("--out", default="watchlist.csv")
    ap.add_argument("--atr-max", type=float, default=None,
                    help=f"波動收斂門檻，預設 {base_params.atr_ratio_max}")
    ap.add_argument("--max-depth", type=float, default=None,
                    help=f"回檔上限，預設 {base_params.max_depth}")
    ap.add_argument("--all", action="store_true", help="輸出全部欄位與全部股票，不只通過者")
    ap.add_argument("--rank", type=int, default=None,
                    help="排名模式：輸出流動性＋強勢者中，整理條件(共6條)至少過 N 條的股票，依分數排序")
    ap.add_argument("--tt", action="store_true", help="通過/差一項都必須同時符合 Minervini 趨勢模板")
    a = ap.parse_args()

    uni = load_universe()
    print(f"股票池 {len(uni)} 檔，下載價格中…")
    prices = download_prices(list(uni.index), a.period)
    p = base_params
    if a.atr_max is not None:
        p = replace(p, atr_ratio_max=a.atr_max)
    if a.max_depth is not None:
        p = replace(p, max_depth=a.max_depth)
    if a.tt:
        p = replace(p, require_trend_template=True)
    info_cols = [c for c in ("name", "sector", "industry") if c in uni.columns]
    res = scan(prices, as_of=a.as_of, params=p, info=uni[info_cols])
    print(funnel(res, with_tt=p.require_trend_template).to_string(index=False))
    if a.all:
        show = res
    elif a.rank is not None:
        show = res[res["candidate"] & (res["cons_score"] >= a.rank)].sort_values(
            ["cons_score", "avg_pct"], ascending=False)
    else:
        show = res[res["passed"] | res["near_miss"]]
    show.to_csv(a.out, encoding="utf-8-sig")
    asof = prices.upto(a.as_of).close.index[-1].date()
    print(f"資料截至 {asof}，全部通過 {int(res['passed'].sum())} 檔、"
          f"差一項 {int(res['near_miss'].sum())} 檔 → {a.out}")
    print(f"趨勢模板 8 條全過：{int((res['trend_template'] & res['liquid']).sum())} 檔（流動性過關者）")
    cols = (["name"] if "name" in res.columns else []) + [
        "close", "ret_63", "ret_126", "avg_pct", "depth", "atr_ratio",
        "missing", "tt_count", "tt_missing", "sector", "industry"]
    print("\n[全部通過]")
    print(res[res["passed"]][cols].round(3).to_string())
    print("\n[差一項，missing 欄是沒過的那項，請人工看圖判斷]")
    print(res[res["near_miss"]].head(40)[cols].round(3).to_string())


def main():
    run_cli(load_sp1500, Params())


if __name__ == "__main__":
    main()
