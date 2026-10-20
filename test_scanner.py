import numpy as np
import pandas as pd
import pytest

from momentum_scanner import Params, Prices, consolidation_features, pivot_lows, scan

N = 280
DATES = pd.bdate_range(end="2026-10-12", periods=N)
P = Params()


def ohlcv(knots_x, knots_y, range_x, range_y, shares=3e6):
    """用折線決定收盤價，用折線決定每日振幅比例(high/low = close*(1±r/2))。"""
    x = np.arange(N)
    c = np.interp(x, knots_x, knots_y)
    r = np.interp(x, range_x, range_y)
    return pd.DataFrame({"c": c, "h": c * (1 + r / 2), "l": c * (1 - r / 2),
                         "v": np.full(N, shares)}, index=DATES)


def ideal(pivot2=34.5, shares=3e6):
    # 平盤 -> 上漲 20→40 -> 回檔到 33 -> 反彈 36.5 -> 回檔 pivot2 -> 緩步走高，振幅逐步收斂
    return ohlcv([0, 150, 200, 220, 235, 250, 279], [20, 20, 40, 33, 36.5, pivot2, 37.5],
                 [0, 255, 270, 279], [0.04, 0.04, 0.012, 0.012], shares)


def quiet(seed):
    rng = np.random.default_rng(seed)
    c = 50 * np.cumprod(1 + 0.003 * rng.standard_normal(N))
    return pd.DataFrame({"c": c, "h": c * 1.003, "l": c * 0.997,
                         "v": np.full(N, 2e6)}, index=DATES)


def build(stocks: dict) -> Prices:
    def w(col):
        return pd.DataFrame({t: d[col] for t, d in stocks.items()})
    return Prices(w("c"), w("h"), w("l"), w("v"))


def universe(**special):
    s = {f"Q{i}": quiet(i) for i in range(60)}
    s.update(special)
    return build(s)


# ---------------- pivot low ----------------
def test_pivot_basic():
    low = np.array([5, 4, 3, 2, 3, 4, 5, 4, 3, 2.5, 3, 4, 5], float)
    assert pivot_lows(low, 3) == [3, 9]


def test_pivot_needs_k_bars_on_right():
    # 最後一根是最低，但右側沒有 k 根確認，不能算低點
    low = np.array([5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2], float)
    assert pivot_lows(low, 3) == []
    # 補上右側 3 根較高的 K 線後，才會被確認
    assert pivot_lows(np.append(low, [3, 4, 5]), 3) == [11]


def test_pivot_flat_bottom_counted_once():
    low = np.array([5, 5, 5, 2, 2, 5, 5, 5, 5], float)
    assert pivot_lows(low, 3) == [3]


# ---------------- 單項特徵 ----------------
def test_depth_and_days_since_high():
    f = consolidation_features(*(lambda d: (d.c, d.h, d.l, d.v))(ideal()), P)
    peak = 40 * (1 + 0.04 / 2)
    trough = 33 * (1 - 0.04 / 2)
    assert f["depth"] == pytest.approx((peak - trough) / peak, rel=1e-6)
    assert f["days_since_high"] == 279 - 200
    assert f["pivot_last"] > f["pivot_prev"]


def test_atr_ratio_known_value():
    # 收盤價固定，前 70 根振幅 2、後 10 根振幅 1 -> ATR10=1, ATR30=(20*2+10*1)/30
    n = 200
    idx = pd.bdate_range(end="2026-10-12", periods=n)
    c = pd.Series(100.0, index=idx)
    rng = np.where(np.arange(n) < n - 10, 2.0, 1.0)
    f = consolidation_features(c, c + rng / 2, c - rng / 2, pd.Series(1e6, index=idx), P)
    assert f["atr_ratio"] == pytest.approx(1.0 / (50 / 30), rel=1e-9)
    assert f["contraction"] is True


# ---------------- 整體篩選 ----------------
def test_ideal_pattern_passes_all_conditions():
    res = scan(universe(IDEAL=ideal()))
    r = res.loc["IDEAL"]
    for col in ["liquid", "strong", "depth_ok", "higher_low", "contraction",
                "above_emas", "ema_rising", "weekly_ok", "consolidating", "passed"]:
        assert r[col], f"{col} 應為 True：{r.to_dict()}"
    assert res["passed"].sum() == 1          # 安靜的對照組一檔都不該通過


def test_lower_low_fails_higher_low():
    res = scan(universe(BAD=ideal(pivot2=31.0)))   # 第二個低點低於第一個
    r = res.loc["BAD"]
    assert not r["higher_low"] and not r["passed"]
    assert r["strong"] and r["contraction"]        # 其他條件仍成立，證明是低點條件擋下


def test_downtrend_like_backtest_table_is_not_strong():
    c = np.linspace(158, 100, N)
    d = pd.DataFrame({"c": c, "h": c * 1.01, "l": c * 0.99, "v": 3e6}, index=DATES)
    r = scan(universe(FALLING=d)).loc["FALLING"]
    assert not r["strong"] and not r["passed"]


def test_illiquid_is_filtered_but_other_flags_unchanged():
    res = scan(universe(THIN=ideal(shares=1000)))
    r = res.loc["THIN"]
    assert not r["liquid"] and not r["passed"]
    assert r["strong"] and r["consolidating"]


def test_percentile_is_cross_sectional_and_ordered():
    stocks = {}
    for i in range(10):
        c = 50 * (1 + 0.001 * (i + 1)) ** np.arange(N)      # 斜率遞增
        stocks[f"S{i}"] = pd.DataFrame({"c": c, "h": c, "l": c, "v": 2e6}, index=DATES)
    res = scan(build(stocks))
    assert res.loc["S9", "pct_126"] == pytest.approx(1.0)
    assert res.loc["S0", "pct_126"] == pytest.approx(0.1)
    assert (res.sort_index()["pct_63"].diff().dropna() > 0).all()


# ---------------- 無未來資料 ----------------
def test_no_lookahead_as_of_equals_truncated():
    full = universe(IDEAL=ideal())
    as_of = DATES[-30]
    a = scan(full, as_of=as_of)
    cut = Prices(full.close.loc[:as_of], full.high.loc[:as_of],
                 full.low.loc[:as_of], full.volume.loc[:as_of])
    b = scan(cut)
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_future_data_does_not_change_past_result():
    base = universe(IDEAL=ideal())
    as_of = DATES[-30]
    a = scan(base, as_of=as_of)
    # 把 as_of 之後的資料全部改成暴漲，結果不能變
    c2, h2, l2 = base.close.copy(), base.high.copy(), base.low.copy()
    c2.loc[c2.index > as_of] *= 5
    h2.loc[h2.index > as_of] *= 5
    l2.loc[l2.index > as_of] *= 5
    b = scan(Prices(c2, h2, l2, base.volume), as_of=as_of)
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_stale_ticker_without_last_bar_is_excluded():
    d = ideal()
    d.iloc[-1] = np.nan                                  # 最後一天沒資料(停牌/下市)
    res = scan(universe(STALE=d))
    assert not res.loc["STALE", "passed"]


def test_near_miss_names_the_single_failed_condition():
    res = scan(universe(BAD=ideal(pivot2=32.6), IDEAL=ideal()))
    assert res.loc["IDEAL", "passed"] and not res.loc["IDEAL", "near_miss"]
    r = res.loc["BAD"]
    assert r["near_miss"] and r["missing"] == "higher_low" and r["miss_n"] == 1
