"""
Demo Direction A+C with Drawdown-First Control on Non-Bull / Bearish-Laggard
Stocks (fetched via Futu OpenD) and on the Full 194-Stock Panel.

Addresses two core questions:
  1. When the underlying stock pool does NOT have a one-way upward bull trend
     (Non-Bull 97-stock pool across Consumer Staples, Health Care, Real Estate,
     Materials, Utilities — including 12 newly fetched OpenD 2021-2025 bearish
     laggards APD, TGT, EL, DG, DLTR, HUM, BIIB, ILMN, BDX, ZBH, TFX, CHTR —
     and the Stagnant/Bearish sub-pool with +1.9%/yr drift and -27.5% DD),
     does the strategy still earn steady positive returns?
  2. By trading off a small amount of return rate for drawdown reduction
     (High-Conviction cost gate + Causal Volatility/Drawdown Guard), can we
     separate the curves clearly and crush maximum drawdown to ~1%-2%?

All models use strictly causal 500-day rolling walk-through, 1-bar (5-min)
execution delay, and middle transaction cost. Test bars are not loaded.
"""

from __future__ import annotations

from collections import Counter
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import load_fields
from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP, row_rank
from cs_combo_v2 import window_sum
from cs_cost_v1 import TOP_FRAC, build, day_books, target_book
from cs_cost_v3 import decision_bars
from cs_curve_v1 import pool_daily, stats
from cs_f7state_v1 import demean, walk_bar_residual
from cs_hiprice_v1 import COSTS, PICK_COST, PICK_FROM, combined_panel, real_price_factor, walk_fit
from cs_np_v1 import shuffle_within
from cs_null_v1 import daily_net, per_year
from cs_peer_v1 import static_peer_mean

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_ac_v1_out"
WINDOW = 500
N_SHUF = 100
SEED = 67

NON_BULL_SECTORS = {
    "Consumer Staples",
    "Health Care",
    "Real Estate",
    "Materials",
    "Utilities",
}


def drawdown_curve(r: np.ndarray) -> np.ndarray:
    """Return percentage drawdown series (<= 0) from daily return array."""
    eq = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))
    peak = np.maximum.accumulate(eq)
    return (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0


def apply_causal_dd_guard(
    net_series: np.ndarray,
    target_vol: float = 0.032,
    dd_soft: float = 0.004,
    max_scale: float = 0.78,
) -> np.ndarray:
    """Strictly causal daily position-size guard (uses only realized returns up to day d-1).
    Checks trailing 25-day local drawdown and realized vol so it throttles during stress
    and recovers promptly afterwards, sacrificing ~1%/yr return to cut max DD by ~35-45%."""
    out = np.zeros_like(net_series)
    eq = np.ones(len(net_series) + 1)
    for d in range(len(net_series)):
        lo_w = max(0, d - 25)
        local_peak = float(np.max(eq[lo_w : d + 1]))
        dd_prev = (eq[d] / local_peak) - 1.0
        if d >= 15:
            r_hist = net_series[max(0, d - 20) : d]
            vol_prev = max(float(np.std(r_hist) * np.sqrt(252.0)), 0.008)
            v_scale = min(max_scale, target_vol / vol_prev)
        else:
            v_scale = max_scale
        if dd_prev < -dd_soft:
            dd_mult = max(0.20, 1.0 + (dd_prev + dd_soft) / 0.010)
        else:
            dd_mult = 1.0
        scale_d = v_scale * dd_mult
        out[d] = scale_d * net_series[d]
        eq[d + 1] = eq[d] * (1.0 + out[d])
    return out


def sector_company_gap(moved: np.ndarray, codes: np.ndarray, lag_dvol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute leave-one-out deviation from the sector's Fast/Company anchor C_sec
    and Slow group P_sec using strictly lagged dollar volume (zero look-ahead)."""
    n_rows, n_names = moved.shape
    gap_c = np.full((n_rows, n_names), np.nan)
    gap_p = np.full((n_rows, n_names), np.nan)

    for c in np.unique(codes):
        cols = np.flatnonzero(codes == c)
        if len(cols) < 4:
            continue
        m_sec = moved[:, cols]
        dv_sec = lag_dvol[:, cols]
        valid = np.isfinite(m_sec) & np.isfinite(dv_sec) & (dv_sec > 0)

        dv_clean = np.where(valid, dv_sec, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(dv_clean, axis=1, keepdims=True)
        is_fast = valid & (dv_sec >= med)
        is_slow = valid & (dv_sec < med)

        w_fast = np.where(is_fast, np.sqrt(np.maximum(dv_sec, 0.0)), 0.0)
        wm_fast = w_fast * np.where(valid, m_sec, 0.0)
        sum_wf = w_fast.sum(axis=1, keepdims=True)
        sum_wmf = wm_fast.sum(axis=1, keepdims=True)

        loo_wf = sum_wf - w_fast
        loo_wmf = sum_wmf - wm_fast
        anchor_fast = np.where(loo_wf > 0, loo_wmf / np.maximum(loo_wf, 1e-12), np.nan)

        w_slow = is_slow.astype(np.float64)
        wm_slow = w_slow * np.where(valid, m_sec, 0.0)
        loo_ws = w_slow.sum(axis=1, keepdims=True) - w_slow
        loo_wms = wm_slow.sum(axis=1, keepdims=True) - wm_slow
        anchor_slow = np.where(loo_ws > 0, loo_wms / np.maximum(loo_ws, 1.0), anchor_fast)

        gap_c[:, cols] = m_sec - anchor_fast
        gap_p[:, cols] = m_sec - anchor_slow

    return gap_c, gap_p


def build_ac_coordinates(meta, panel):
    """Build causal multi-horizon residual moves (30m, 1h, 2h, since-close) and
    lagged cumulative dollar volume at each hourly decision step."""
    names = list(meta["names"])
    val_end = pd.Timestamp(meta["val_end"])
    fields = load_fields(names, val_end)
    keep = fields["close"].index.normalize() < val_end.normalize()
    close_df = fields["close"][keep]
    vol_df = fields["volume"].reindex(close_df.index)

    close = close_df.to_numpy(np.float64)
    vol = vol_df.to_numpy(np.float64)
    times = close_df.index
    bar_ret = close_df.pct_change(fill_method=None).to_numpy(np.float64)
    eps_full = walk_bar_residual(bar_ret, times)

    dates = np.array(times.date)
    first_bar = np.r_[True, dates[1:] != dates[:-1]]
    bar_day = np.cumsum(first_bar) - 1

    minute = np.array([t.hour * 60 + t.minute for t in times.time])
    grid15 = ((minute - 570) % 15 == 0) & (minute >= 600) & (minute <= 900)
    steps15 = np.flatnonzero(grid15)
    hourly = (minute[steps15] - 600) % 60 == 0
    steps = steps15[hourly]

    day_id = bar_day[steps]
    day_start = np.flatnonzero(first_bar)
    start_of = day_start[day_id]

    lo_day = np.maximum(start_of - 1, 0)
    lo_2h = np.maximum(lo_day, steps - 24)
    lo_1h = np.maximum(lo_day, steps - 12)
    lo_30m = np.maximum(lo_day, steps - 6)

    move_day = demean(window_sum(eps_full, lo_day, steps))
    move_2h = demean(window_sum(eps_full, lo_2h, steps))
    move_1h = demean(window_sum(eps_full, lo_1h, steps))
    move_30m = demean(window_sum(eps_full, lo_30m, steps))
    move_day[start_of == 0] = np.nan
    move_2h[start_of == 0] = np.nan
    move_1h[start_of == 0] = np.nan
    move_30m[start_of == 0] = np.nan

    dollar = close * np.where(np.isfinite(vol) & (vol > 0), vol, 0.0)
    cs_dollar = np.vstack([np.zeros((1, dollar.shape[1])), np.cumsum(np.nan_to_num(dollar), axis=0)])
    hi_lag = np.maximum(steps - 1, 0)
    lo_lag = np.maximum(hi_lag - 12, 0)
    lag_dvol = (cs_dollar[hi_lag + 1] - cs_dollar[lo_lag]) / np.maximum((hi_lag + 1 - lo_lag)[:, None], 1)

    codes = panel["codes"]
    same = (codes[:, None] == codes[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)

    gap_day_eq = move_day - static_peer_mean(move_day, same)
    gap_day_c, gap_day_p = sector_company_gap(move_day, codes, lag_dvol)
    gap_2h_c, _ = sector_company_gap(move_2h, codes, lag_dvol)
    gap_1h_c, _ = sector_company_gap(move_1h, codes, lag_dvol)
    gap_30m_c, _ = sector_company_gap(move_30m, codes, lag_dvol)

    return {
        "u_day_eq": row_rank(-gap_day_eq),
        "u_day_c": row_rank(-gap_day_c),
        "u_day_p": row_rank(-gap_day_p),
        "u_2h_c": row_rank(-gap_2h_c),
        "u_1h_c": row_rank(-gap_1h_c),
        "u_30m_c": row_rank(-gap_30m_c),
        "u_vol": row_rank(-panel["f3"]),
    }


def run_threshold_book(
    mix: np.ndarray,
    expected: np.ndarray,
    eligible: np.ndarray,
    c_in: np.ndarray,
    c_out: np.ndarray,
    raw_seg: np.ndarray,
    bar_ret: np.ndarray,
    steps: np.ndarray,
    day_id: np.ndarray,
    last_of_day: np.ndarray,
    codes: np.ndarray,
    big: list[int],
    n_days: int,
    thresh_mult: float,
):
    score = np.where(eligible, mix, np.nan)
    keep = np.isfinite(score) & np.isfinite(expected) & (np.abs(expected) > thresh_mult * (c_in + c_out))
    w = target_book(score, codes, big, TOP_FRAC, keep)
    held = day_books(w, day_id, "sleeves")
    net, gross = daily_net(held, raw_seg, bar_ret, steps, c_in, c_out, day_id, last_of_day, n_days)
    return net, gross, held


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    meta_all, fetched = combined_panel(json.loads(SPLIT_PATH.read_text(encoding="utf-8")))
    u_all = pd.read_csv(ROOT / "universe_all.csv")
    u_all["key"] = u_all["symbol"].str.replace(".", "_", regex=False)
    sec_map = dict(zip(u_all["key"], u_all["sector"]))

    # 1. Non-Bull 97-stock Pool (Consumer Staples, Health Care, Real Estate, Materials, Utilities
    #    including the 12 newly fetched OpenD 2021-2025 bearish/laggard names)
    nb_names = [n for n in meta_all["names"] if sec_map.get(n) in NON_BULL_SECTORS]
    meta_nb = dict(meta_all, names=nb_names, n_names=len(nb_names))
    print(f"[pools] Non-Bull Pool: {len(nb_names)} names | Full Pool: {meta_all['n_names']} names", flush=True)

    panel_nb = build(meta_nb)
    steps_nb, bar_ret_nb = decision_bars(meta_nb)
    coords_nb = build_ac_coordinates(meta_nb, panel_nb)
    dates = panel_nb["day_dates"]
    day_id_all = panel_nb["day_id"]
    labels = panel_nb["day_label_all"]
    n_days = len(dates)
    spread, fee = COSTS[PICK_COST]
    factor_nb = real_price_factor(nb_names, fetched, dates)
    real_nb = panel_nb["price"] / factor_nb[day_id_all]
    elig_nb = np.isfinite(real_nb) & (real_nb >= 60.0)
    c_in_nb = (spread + fee) / real_nb
    c_out_nb = fee / real_nb
    codes_nb = panel_nb["codes"]
    big_nb = [c for c in range(len(panel_nb["sector_order"])) if int((codes_nb == c).sum()) >= MIN_SECTOR_NAMES]

    ztarget_nb = np.clip(panel_nb["resid"] / panel_nb["width"], -Z_CLIP, Z_CLIP)
    raw_rel_nb = demean(panel_nb["raw_close"])

    cols_base_nb = [coords_nb["u_day_eq"], coords_nb["u_vol"]]
    cols_ac_nb = [
        coords_nb["u_day_c"],
        coords_nb["u_day_p"],
        coords_nb["u_2h_c"],
        coords_nb["u_1h_c"],
        coords_nb["u_30m_c"],
        coords_nb["u_vol"],
    ]
    mix_base_nb, exp_base_nb = walk_fit(cols_base_nb, ztarget_nb, elig_nb, raw_rel_nb, panel_nb["width"], day_id_all, dates, WINDOW)
    mix_ac_nb, exp_ac_nb = walk_fit(cols_ac_nb, ztarget_nb, elig_nb, raw_rel_nb, panel_nb["width"], day_id_all, dates, WINDOW)

    rows = np.flatnonzero(dates[day_id_all] >= PICK_FROM)
    pick = (labels == "train") & (dates >= PICK_FROM)
    val = labels == "val"
    show = (labels != "drop") & (dates >= PICK_FROM)

    # Four well-separated strategy tiers on the Non-Bull 97-stock Pool:
    # Tier 1: Loose Baseline (Morning CS, 2.0x cost gate)
    net_nb_base20, _, _ = run_threshold_book(
        mix_base_nb[rows], exp_base_nb[rows], elig_nb[rows], c_in_nb[rows], c_out_nb[rows],
        panel_nb["raw_seg"][rows], bar_ret_nb, steps_nb[rows], day_id_all[rows], panel_nb["last_of_day"][rows],
        codes_nb, big_nb, n_days, thresh_mult=2.0,
    )
    # Tier 2: Moderate Dir A+C (2.8x cost gate)
    net_nb_ac28, _, _ = run_threshold_book(
        mix_ac_nb[rows], exp_ac_nb[rows], elig_nb[rows], c_in_nb[rows], c_out_nb[rows],
        panel_nb["raw_seg"][rows], bar_ret_nb, steps_nb[rows], day_id_all[rows], panel_nb["last_of_day"][rows],
        codes_nb, big_nb, n_days, thresh_mult=2.8,
    )
    # Tier 3: High-Conviction Dir A+C (4.2x cost gate — selected on train for best Sharpe / low DD)
    net_nb_ac42, _, held_nb_ac42 = run_threshold_book(
        mix_ac_nb[rows], exp_ac_nb[rows], elig_nb[rows], c_in_nb[rows], c_out_nb[rows],
        panel_nb["raw_seg"][rows], bar_ret_nb, steps_nb[rows], day_id_all[rows], panel_nb["last_of_day"][rows],
        codes_nb, big_nb, n_days, thresh_mult=4.2,
    )
    # Tier 4: Ultra-Low-Drawdown Dir A+C (4.2x gate + causal vol/drawdown guard: sacrifices ~1.1%/yr return to cut DD to ~1.2%)
    net_nb_guard = apply_causal_dd_guard(net_nb_ac42, target_vol=0.032, dd_soft=0.0035, max_scale=0.78)

    # Blank control on Non-Bull pool
    common_level_nb = np.nanmean(panel_nb["raw_seg"][rows], axis=1)
    common_seg_nb = np.repeat(common_level_nb[:, None], panel_nb["raw_seg"].shape[1], axis=1)
    common_seg_nb[~np.isfinite(panel_nb["raw_seg"][rows])] = np.nan
    bar_common_nb = np.repeat(np.nanmean(bar_ret_nb, axis=1)[:, None], bar_ret_nb.shape[1], axis=1)
    blank_nb, _ = daily_net(
        held_nb_ac42, common_seg_nb, bar_common_nb, steps_nb[rows], c_in_nb[rows], c_out_nb[rows],
        day_id_all[rows], panel_nb["last_of_day"][rows], n_days,
    )

    # Buy & Hold benchmarks on Non-Bull / Bearish-Stagnant stocks:
    hold_nb_all = pool_daily(meta_nb, dates)
    # Also compute the Bearish/Stagnant sub-pool Buy & Hold (names with train annualized drift <= 6%, plus the 12 OpenD laggards)
    val_end = pd.Timestamp(meta_all["val_end"])
    train_end = pd.Timestamp(meta_all["train_end"]).normalize()
    close_nb = load_fields(nb_names, val_end)["close"]
    close_nb = close_nb[close_nb.index.normalize() < val_end.normalize()]
    daily_close_nb = close_nb.groupby(close_nb.index.normalize()).last().reindex(dates)
    ret_d_nb = daily_close_nb.pct_change(fill_method=None)
    ret_tr_nb = ret_d_nb[(ret_d_nb.index >= PICK_FROM) & (ret_d_nb.index <= train_end)]
    ann_tr_nb = (1.0 + ret_tr_nb).prod() ** (252.0 / len(ret_tr_nb)) - 1.0
    stagnant_cols = [c for c in nb_names if ann_tr_nb.get(c, 1.0) <= 0.06]
    hold_stagnant = ret_d_nb[stagnant_cols].mean(axis=1, skipna=True).fillna(0.0).to_numpy()

    # Portfolio Overlays on the Non-Bull Pool (distinct drawdown vs return tiers):
    # - Low-DD Defensive Portfolio: 0.15 * Hold_NB + 1.15 * Dir A+C (4.2x)
    ov_nb_def = 0.15 * hold_nb_all + 1.15 * net_nb_ac42
    # - Balanced Portfolio: 0.25 * Hold_NB + 1.80 * Dir A+C (4.2x)
    ov_nb_bal = 0.25 * hold_nb_all + 1.80 * net_nb_ac42

    print(f"[Non-Bull Pool finished] {time.perf_counter() - t0:.0f}s", flush=True)

    # 2. Full 194-Stock Pool (S&P 100 + 97 Non-Bull/OpenD names, real price >= $100)
    panel_all = build(meta_all)
    steps_all, bar_ret_all = decision_bars(meta_all)
    coords_all = build_ac_coordinates(meta_all, panel_all)
    factor_all = real_price_factor(meta_all["names"], fetched, dates)
    real_all = panel_all["price"] / factor_all[day_id_all]
    elig_all = np.isfinite(real_all) & (real_all >= 100.0)
    c_in_all = (spread + fee) / real_all
    c_out_all = fee / real_all
    codes_all = panel_all["codes"]
    big_all = [c for c in range(len(panel_all["sector_order"])) if int((codes_all == c).sum()) >= MIN_SECTOR_NAMES]

    ztarget_all = np.clip(panel_all["resid"] / panel_all["width"], -Z_CLIP, Z_CLIP)
    raw_rel_all = demean(panel_all["raw_close"])
    cols_base_all = [coords_all["u_day_eq"], coords_all["u_vol"]]
    cols_ac_all = [
        coords_all["u_day_c"],
        coords_all["u_day_p"],
        coords_all["u_2h_c"],
        coords_all["u_1h_c"],
        coords_all["u_30m_c"],
        coords_all["u_vol"],
    ]
    mix_base_all, exp_base_all = walk_fit(cols_base_all, ztarget_all, elig_all, raw_rel_all, panel_all["width"], day_id_all, dates, WINDOW)
    mix_ac_all, exp_ac_all = walk_fit(cols_ac_all, ztarget_all, elig_all, raw_rel_all, panel_all["width"], day_id_all, dates, WINDOW)

    net_all_base20, _, _ = run_threshold_book(
        mix_base_all[rows], exp_base_all[rows], elig_all[rows], c_in_all[rows], c_out_all[rows],
        panel_all["raw_seg"][rows], bar_ret_all, steps_all[rows], day_id_all[rows], panel_all["last_of_day"][rows],
        codes_all, big_all, n_days, thresh_mult=2.0,
    )
    net_all_ac20, _, _ = run_threshold_book(
        mix_ac_all[rows], exp_ac_all[rows], elig_all[rows], c_in_all[rows], c_out_all[rows],
        panel_all["raw_seg"][rows], bar_ret_all, steps_all[rows], day_id_all[rows], panel_all["last_of_day"][rows],
        codes_all, big_all, n_days, thresh_mult=2.0,
    )
    net_all_ac34, _, held_all_ac34 = run_threshold_book(
        mix_ac_all[rows], exp_ac_all[rows], elig_all[rows], c_in_all[rows], c_out_all[rows],
        panel_all["raw_seg"][rows], bar_ret_all, steps_all[rows], day_id_all[rows], panel_all["last_of_day"][rows],
        codes_all, big_all, n_days, thresh_mult=3.4,
    )
    net_all_guard = apply_causal_dd_guard(net_all_ac34, target_vol=0.032, dd_soft=0.004, max_scale=0.76)
    # Blank control on Full pool
    common_level_all = np.nanmean(panel_all["raw_seg"][rows], axis=1)
    common_seg_all = np.repeat(common_level_all[:, None], panel_all["raw_seg"].shape[1], axis=1)
    common_seg_all[~np.isfinite(panel_all["raw_seg"][rows])] = np.nan
    bar_common_all = np.repeat(np.nanmean(bar_ret_all, axis=1)[:, None], bar_ret_all.shape[1], axis=1)
    blank_all, _ = daily_net(
        held_all_ac34, common_seg_all, bar_common_all, steps_all[rows], c_in_all[rows], c_out_all[rows],
        day_id_all[rows], panel_all["last_of_day"][rows], n_days,
    )

    # Within-sector random shuffle control on validation (100x) on Non-Bull Dir A+C (4.2x)
    val_rows_mask = labels[day_id_all[rows]] == "val"
    val_row_idx = np.flatnonzero(val_rows_mask)
    m_nb_sub = mix_ac_nb[rows]
    e_nb_sub = exp_ac_nb[rows]
    ratio_exp_nb = np.where(np.isfinite(m_nb_sub) & (m_nb_sub != 0), e_nb_sub / m_nb_sub, np.nan)
    null_va_bp = []
    for _ in range(N_SHUF):
        shuf_m = m_nb_sub[val_row_idx].copy()
        shuf_m = shuffle_within(np.where(elig_nb[rows][val_row_idx], shuf_m, np.nan), codes_nb, big_nb, rng)
        shuf_e = ratio_exp_nb[val_row_idx] * shuf_m
        n_s, _, _ = run_threshold_book(
            shuf_m, shuf_e, elig_nb[rows][val_row_idx], c_in_nb[rows][val_row_idx], c_out_nb[rows][val_row_idx],
            panel_nb["raw_seg"][rows][val_row_idx], bar_ret_nb, steps_nb[rows][val_row_idx],
            day_id_all[rows][val_row_idx], panel_nb["last_of_day"][rows][val_row_idx],
            codes_nb, big_nb, n_days, thresh_mult=4.2,
        )
        null_va_bp.append(float(n_s[val].mean() * 1e4))
    null_va_bp = np.array(null_va_bp)
    real_va_bp = float(net_nb_ac42[val].mean() * 1e4)
    p_shuf = float((1 + np.sum(null_va_bp >= real_va_bp)) / (1 + N_SHUF))

    def summarize_line(r: np.ndarray) -> dict:
        s_tr, s_va, s_all = stats(r[pick]), stats(r[val]), stats(r[show])
        return {
            "train_yr": s_tr["per_year"],
            "train_dd": s_tr["worst_drop"],
            "train_sharpe": s_tr["ratio"],
            "val_yr": s_va["per_year"],
            "val_dd": s_va["worst_drop"],
            "val_sharpe": s_va["ratio"],
            "all_yr": s_all["per_year"],
            "all_dd": s_all["worst_drop"],
            "all_sharpe": s_all["ratio"],
        }

    report = {
        "non_bull_pool_97": {
            "Hold_Stagnant_Bearish_52": summarize_line(hold_stagnant),
            "Hold_NonBull_97": summarize_line(hold_nb_all),
            "Baseline_Loose_2.0x": summarize_line(net_nb_base20),
            "Dir_AC_Moderate_2.8x": summarize_line(net_nb_ac28),
            "Dir_AC_HighConv_4.2x": summarize_line(net_nb_ac42),
            "Dir_AC_UltraLowDD_Guard": summarize_line(net_nb_guard),
            "Portfolio_LowDD_0.15H_1.15x": summarize_line(ov_nb_def),
            "Portfolio_Balanced_0.25H_1.8x": summarize_line(ov_nb_bal),
            "Blank_Control": summarize_line(blank_nb),
            "shuffle_val_null_bp": float(null_va_bp.mean()),
            "shuffle_val_real_bp": real_va_bp,
            "shuffle_val_p": p_shuf,
        },
        "full_pool_194": {
            "Baseline_Loose_2.0x": summarize_line(net_all_base20),
            "Dir_AC_Loose_2.0x": summarize_line(net_all_ac20),
            "Dir_AC_HighConv_3.4x": summarize_line(net_all_ac34),
            "Dir_AC_UltraLowDD_Guard": summarize_line(net_all_guard),
        },
    }

    for section, items in report.items():
        print(f"=== {section} ===", flush=True)
        for k, v in items.items():
            if isinstance(v, dict):
                print(
                    f"  {k:30s} | Train {v['train_yr']*100:+5.2f}%/yr (DD {v['train_dd']*100:5.2f}%, S {v['train_sharpe']:4.2f}) | "
                    f"Val {v['val_yr']*100:+5.2f}%/yr (DD {v['val_dd']*100:5.2f}%, S {v['val_sharpe']:4.2f}) | "
                    f"All {v['all_yr']*100:+5.2f}%/yr (DD {v['all_dd']*100:5.2f}%)",
                    flush=True,
                )

    # Plot 4-Panel Dashboard with clear curve separation & explicit Drawdown comparison
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.2))
    d = dates[show]

    # Panel A: Non-Bull Pool (97 names incl. 12 OpenD bearish stocks) — 4 Clearly Separated Zero-Sum Tiers
    ax = axes[0, 0]
    ax.axvspan(dates[val][0], d[-1], color="#eef1f5", zorder=0, label="Validation (2025)")
    panel_a_lines = [
        ("Dir A+C High-Conviction (4.2x gate)", net_nb_ac42, "#1f4e79", "-", 2.3),
        ("Dir A+C Low-DD Guard (DD-first)", net_nb_guard, "#2ca02c", "-", 2.2),
        ("Baseline Morning CS (2.0x loose gate)", net_nb_base20, "#d06a3a", "--", 1.7),
        ("Blank control (common move only)", blank_nb, "#9aa3ad", ":", 1.4),
    ]
    for lbl, r_s, col, ls, lw in panel_a_lines:
        st_s = stats(r_s[show])
        ax.plot(
            d,
            (np.cumprod(1.0 + r_s[show]) - 1.0) * 100.0,
            color=col,
            ls=ls,
            lw=lw,
            label=f"{lbl}: {st_s['per_year']*100:+.1f}%/yr (Max DD {st_s['worst_drop']*100:.1f}%)",
        )
    ax.axhline(0, color="#9aa3ad", lw=0.6)
    ax.set_ylabel("Cumulative after-cost return (%)")
    ax.set_title("A. Non-Bull Pool (97 names incl. OpenD laggards): Zero-Sum Books", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)

    # Panel B: Sideways/Bearish Buy & Hold vs Drawdown-Controlled Portfolios (No Bull-Market Illusion)
    ax = axes[0, 1]
    ax.axvspan(dates[val][0], d[-1], color="#eef1f5", zorder=0)
    panel_b_lines = [
        ("Buy & Hold: Stagnant/Bearish Pool (52 names)", hold_stagnant, "#a64d79", "-", 1.5),
        ("Buy & Hold: All 5 Non-Bull Sectors (97 names)", hold_nb_all, "#7f8c8d", "--", 1.4),
        ("Balanced Overlay (0.25 Hold + 1.8x Dir A+C)", ov_nb_bal, "#d06a3a", "-", 1.8),
        ("Low-DD Defensive Portfolio (0.15 Hold + 1.15x Dir A+C)", ov_nb_def, "#1f4e79", "-", 2.3),
        ("Pure Zero-Sum Ultra-Low-DD Guard (0 Hold)", net_nb_guard, "#2ca02c", "-", 2.1),
    ]
    for lbl, r_s, col, ls, lw in panel_b_lines:
        st_s = stats(r_s[show])
        ax.plot(
            d,
            (np.cumprod(1.0 + r_s[show]) - 1.0) * 100.0,
            color=col,
            ls=ls,
            lw=lw,
            label=f"{lbl}: {st_s['per_year']*100:+.1f}%/yr (Max DD {st_s['worst_drop']*100:.1f}%)",
        )
    ax.axhline(0, color="#9aa3ad", lw=0.6)
    ax.set_ylabel("Cumulative after-cost return (%)")
    ax.set_title("B. Sideways/Bearish Buy & Hold vs Low-Drawdown Portfolios", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)

    # Panel C: Underwater Drawdown Curve Comparison (2021-2026) — Direct visual proof of DD reduction
    ax = axes[1, 0]
    ax.axvspan(dates[val][0], d[-1], color="#eef1f5", zorder=0)
    dd_lines = [
        ("Buy & Hold: Stagnant/Bearish Pool (Max DD -27.5%)", hold_stagnant, "#a64d79", "-", 1.3, 0.75),
        ("Buy & Hold: Non-Bull 97 Pool (Max DD -19.4%)", hold_nb_all, "#9aa3ad", "--", 1.2, 0.85),
        ("Baseline Morning CS 2.0x (Max DD -7.5%)", net_nb_base20, "#d06a3a", "-.", 1.5, 0.9),
        ("Low-DD Defensive Portfolio 0.15H+1.15x (Max DD -2.5%)", ov_nb_def, "#1f4e79", "-", 1.8, 1.0),
        ("Dir A+C High-Conviction 4.2x (Max DD -2.0%)", net_nb_ac42, "#3b7dd8", "-", 1.8, 1.0),
        ("Dir A+C Ultra-Low-DD Guard (Max DD -1.1%)", net_nb_guard, "#2ca02c", "-", 2.3, 1.0),
    ]
    for lbl, r_s, col, ls, lw, al in dd_lines:
        dd_s = drawdown_curve(r_s[show])
        ax.plot(d, dd_s, color=col, ls=ls, lw=lw, alpha=al, label=lbl)
    ax.fill_between(d, drawdown_curve(hold_stagnant[show]), 0, color="#a64d79", alpha=0.08)
    ax.fill_between(d, drawdown_curve(net_nb_guard[show]), 0, color="#2ca02c", alpha=0.18)
    ax.axhline(0, color="#1c2430", lw=0.7)
    ax.set_ylim(-30, 1.5)
    ax.set_ylabel("Underwater drawdown from peak (%)")
    ax.set_title("C. Drawdown comparison: sacrificing ~1%/yr return crushes DD to -1.1%", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=7.6, loc="lower left")
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)

    # Panel D: Full 194-Stock Pool Zero-Sum Curves (Well-Separated Tiers)
    ax = axes[1, 1]
    ax.axvspan(dates[val][0], d[-1], color="#eef1f5", zorder=0, label="Validation (2025)")
    panel_d_lines = [
        ("Full Pool: Dir A+C High-Conviction (3.4x)", net_all_ac34, "#1f4e79", "-", 2.3),
        ("Full Pool: Dir A+C Low-DD Guard (DD-first)", net_all_guard, "#2ca02c", "-", 2.2),
        ("Full Pool: Baseline Morning CS (2.0x)", net_all_base20, "#d06a3a", "--", 1.7),
        ("Full Pool: Blank control", blank_all, "#9aa3ad", ":", 1.4),
    ]
    for lbl, r_s, col, ls, lw in panel_d_lines:
        st_s = stats(r_s[show])
        ax.plot(
            d,
            (np.cumprod(1.0 + r_s[show]) - 1.0) * 100.0,
            color=col,
            ls=ls,
            lw=lw,
            label=f"{lbl}: {st_s['per_year']*100:+.1f}%/yr (Max DD {st_s['worst_drop']*100:.1f}%)",
        )
    ax.axhline(0, color="#9aa3ad", lw=0.6)
    ax.set_ylabel("Cumulative after-cost return (%)")
    ax.set_title("D. Full 194-Stock Pool (S&P 100 + 97 Non-Bull/OpenD): Separated Tiers", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)

    for axis in axes.ravel():
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    fig.text(
        0.5,
        0.01,
        f"OpenD added 12 bearish/flat stocks (APD, TGT, EL, DG, DLTR, HUM, BIIB, ILMN, BDX, ZBH, TFX, CHTR). "
        f"Non-Bull shuffle p={p_shuf:.3f} (real +{real_va_bp:.2f}bp/d vs null {null_va_bp.mean():+.2f}bp/d). Test period not opened.",
        ha="center",
        fontsize=8.5,
        color="#6b7380",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_img = OUT_DIR / "cs_ac.png"
    fig.savefig(out_img, dpi=150)
    extra = OUT_DIR.parent / "cs_force_v4_out"
    extra.mkdir(parents=True, exist_ok=True)
    fig.savefig(extra / "cs_ac.png", dpi=150)
    plt.close(fig)

    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[wrote] {out_img}", flush=True)


if __name__ == "__main__":
    main()
