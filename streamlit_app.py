"""網頁版：強勢股 + 整理轉強掃描（Streamlit）
部署時 Main file path 請設為 streamlit_app.py，並與 momentum_scanner.py、requirements.txt 放在同一層。"""
import traceback
from dataclasses import replace

import pandas as pd
import streamlit as st

import momentum_scanner as ms
import tw_scanner as tw

st.set_page_config(page_title="強勢整理掃描", layout="wide")
st.title("強勢股 + 整理轉強掃描")
st.caption("美股 S&P 1500／台股上市櫃 · 流程：強勢排名 → 流動性 → 上漲後整理(低點墊高、波動收斂、站上 9/21 EMA、週線在 10 週線上)。"
           "只用價量資料，不推測誰在買。結果是縮小範圍的觀察名單，請再人工看圖。")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_universe(market: str) -> pd.DataFrame:
    return tw.load_tw_universe() if market == "tw" else ms.load_sp1500()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_prices(tickers: tuple, period: str) -> ms.Prices:
    return ms.download_prices(list(tickers), period)


# ---------------- 側邊欄參數 ----------------
with st.sidebar:
    market_label = st.radio("市場", ["美股 S&P 1500", "台股 上市櫃"])
    market = "tw" if market_label.startswith("台") else "us"
    cur = "新台幣" if market == "tw" else "美元"
    d = tw.TW_PARAMS if market == "tw" else ms.Params()
    st.header("參數")
    atr_max = st.slider("波動收斂：ATR(10)/ATR(30) 小於", 0.5, 1.3, d.atr_ratio_max, 0.05)
    depth_lo, depth_hi = st.slider("高點後回檔幅度", 0.0, 0.5, (d.min_depth, d.max_depth), 0.01)
    max_off = st.slider("現價距高點上限", 0.05, 0.4, d.max_off_high, 0.01)
    pivot_k = st.slider("波段低點：左右各幾根", 2, 8, d.pivot_k)
    min_hl = st.slider("新低點至少比前低點高", 0.0, 0.05, d.min_higher_low, 0.005, format="%.3f")
    min_gain = st.slider("3 或 6 個月漲幅大於", 0.0, 1.0, d.min_gain, 0.05)
    use_tt = st.checkbox("要求通過 Minervini 趨勢模板（8 條）", value=False,
                         help="股價在 150/200 日線之上、200 日線上升、50>150>200、距 52 週低 +30% 以上、"
                              "距 52 週高 25% 內、RS 前 30%。至少需要一年的價格資料。")
    min_dv = st.number_input(f"20 日均成交金額下限（百萬{cur}）", 1, 5000, int(d.min_dollar_vol / 1e6))
    st.caption("更動參數不用重新下載價格，會立即重算。")

params = replace(d, atr_ratio_max=atr_max, min_depth=depth_lo, max_depth=depth_hi,
                 max_off_high=max_off, pivot_k=pivot_k, min_higher_low=min_hl,
                 min_gain=min_gain, min_dollar_vol=min_dv * 1e6,
                 require_trend_template=use_tt)

# ---------------- 載入資料 ----------------
if st.button("載入／更新價格資料", type="primary"):
    st.session_state["loaded"] = True
if st.button("清除快取並重抓"):
    st.cache_data.clear()
    st.session_state["loaded"] = False
    st.rerun()

if not st.session_state.get("loaded"):
    st.info("按「載入／更新價格資料」開始。第一次要下載約 1,500 檔，約需 1～3 分鐘。")
    st.stop()

try:
    with st.spinner("取得股票池與價格中…（約 1～3 分鐘）"):
        uni = get_universe(market)
        prices = get_prices(tuple(uni.index), "2y")
    info_cols = [c for c in ("name", "sector", "industry") if c in uni.columns]
    res = ms.scan(prices, params=params, info=uni[info_cols])
except Exception as e:                      # 出錯時顯示原因，不要空白
    st.error(f"執行失敗：{e}")
    st.code(traceback.format_exc())
    st.stop()

asof = prices.close.index[-1].date()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("資料截至", str(asof))
c2.metric("股票數", f"{len(res):,}")
c3.metric("全部通過", int(res["passed"].sum()))
c4.metric("差一項", int(res["near_miss"].sum()))
c5.metric("趨勢模板8條全過", int((res["trend_template"] & res["liquid"]).sum()))

st.subheader("條件漏斗")
st.dataframe(ms.funnel(res, with_tt=use_tt), hide_index=True, width=360)

cols = (["name"] if "name" in res.columns else []) + [
    "close", "ret_63", "ret_126", "avg_pct", "depth", "off_high",
    "atr_ratio", "missing", "tt_count", "tt_missing", "sector", "industry"]
fmt = {c: "{:.3f}" for c in ["ret_63", "ret_126", "avg_pct", "depth", "off_high", "atr_ratio"]}
fmt["close"] = "{:.2f}"

passed = res[res["passed"]]
near = res[res["near_miss"]]
cand = res[res["passed"] | res["near_miss"]]

t1, t2, t3, t4 = st.tabs(["全部通過", "差一項", "板塊共振", "看圖"])
with t1:
    if passed.empty:
        st.warning("目前沒有股票通過全部條件，可看「差一項」或放寬左側參數。")
    else:
        st.dataframe(passed[cols].style.format(fmt), width='stretch')
with t2:
    st.caption("missing 欄是唯一沒過的條件。")
    st.dataframe(near[cols].style.format(fmt), width='stretch')
with t3:
    st.caption("在『全部通過＋差一項』中，同一板塊有 2 檔以上者（階段 2 的雛形）。"
               + ("台股的板塊是證交所產業別；放 tw_themes.csv 可改成自訂題材。" if market == "tw" else ""))
    if cand.empty:
        st.write("沒有候選股。")
    else:
        g = (cand.dropna(subset=["industry"]).groupby("industry")
             .agg(檔數=("close", "size"), 全過=("passed", "sum"),
                  代號=("close", lambda s: ", ".join(s.index)))
             .query("檔數 >= 2").sort_values(["檔數", "全過"], ascending=False))
        st.dataframe(g, width='stretch') if len(g) else st.write("沒有同細產業 2 檔以上。")
with t4:
    if cand.empty:
        st.write("沒有候選股可看。")
    else:
        label = (lambda t: f"{t} {cand.loc[t, 'name']}") if "name" in cand.columns else (lambda t: t)
        tk = st.selectbox("選一檔", list(cand.index), format_func=label)
        n = st.slider("顯示天數", 60, 400, 180)
        s = prices.close[tk].dropna()
        df = pd.DataFrame({"收盤": s, "EMA9": s.ewm(span=9, adjust=False).mean(),
                           "EMA21": s.ewm(span=21, adjust=False).mean()}).tail(n)
        st.line_chart(df)
        r = res.loc[tk]
        st.write(f"{tk} · {r.get('industry', '')} · 回檔 {r['depth']:.1%} · "
                 f"ATR 比 {r['atr_ratio']:.2f} · 前低 {r['pivot_prev']:.2f} → 最近低 {r['pivot_last']:.2f}"
                 + (f" · 未過：{r['missing']}" if r["missing"] else ""))
