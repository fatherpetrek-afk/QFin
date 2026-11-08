"""
Does the current framework earn its keep, with the cost gate held fixed?

Pool: the 97-name non-bull set (Staples, Health Care, Real Estate, Materials,
Utilities). Real price at least $60. Gate is 4.2 times the round trip, the
same number already chosen for that pool. Nothing here retunes it.
Test bars are not loaded.

The checks, all after cost:
  1. Drop pieces of the six coordinates.
  2. Replace the fast-half anchor with the equal-weight sector mean.
  3. Rebuild the anchor on a random half of the sector, and refit.
  4. Rebuild the anchor inside randomly relabeled sectors, and refit.
  5. Trade yesterday's score, and trade the opposite sign.
  6. Rank correlation of each coordinate with the residual to the close.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import load_fields
from cs_ac_v1 import NON_BULL_SECTORS, run_threshold_book, sector_company_gap
from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP, rank_corr, row_rank
from cs_combo_v2 import window_sum
from cs_cost_v1 import build
from cs_cost_v3 import decision_bars
from cs_curve_v1 import stats
from cs_f7state_v1 import demean, walk_bar_residual
from cs_hiprice_v1 import COSTS, PICK_COST, PICK_FROM, combined_panel, real_price_factor, walk_fit
from cs_peer_v1 import static_peer_mean
from cs_resid_v1 import newey_west

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_fw_test_v1_out"
WINDOW = 500
THRESH = 4.2
N_ANCHOR = 12
N_SECTOR = 8
SEED = 67
MIN_PX = 60.0


def load_moves(meta):
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
    move_day = demean(window_sum(eps_full, lo_day, steps))
    move_2h = demean(window_sum(eps_full, np.maximum(lo_day, steps - 24), steps))
    move_1h = demean(window_sum(eps_full, np.maximum(lo_day, steps - 12), steps))
    move_30m = demean(window_sum(eps_full, np.maximum(lo_day, steps - 6), steps))
    move_day[start_of == 0] = np.nan
    move_2h[start_of == 0] = np.nan
    move_1h[start_of == 0] = np.nan
    move_30m[start_of == 0] = np.nan
    dollar = close * np.where(np.isfinite(vol) & (vol > 0), vol, 0.0)
    cs_dollar = np.vstack([np.zeros((1, dollar.shape[1])), np.cumsum(np.nan_to_num(dollar), axis=0)])
    hi_lag = np.maximum(steps - 1, 0)
    lo_lag = np.maximum(hi_lag - 12, 0)
    lag_dvol = (cs_dollar[hi_lag + 1] - cs_dollar[lo_lag]) / np.maximum((hi_lag + 1 - lo_lag)[:, None], 1)
    return {
        "day": move_day,
        "h2": move_2h,
        "h1": move_1h,
        "m30": move_30m,
        "lag_dvol": lag_dvol,
    }


def peer_matrix(codes):
    same = (codes[:, None] == codes[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    return same


def coordinates(moves, lag_dvol, anchor_codes, peer_codes, volume_state):
    day = moves["day"]
    gap_eq = day - static_peer_mean(day, peer_matrix(peer_codes))
    gap_c, gap_p = sector_company_gap(day, anchor_codes, lag_dvol)
    gap_2h, _ = sector_company_gap(moves["h2"], anchor_codes, lag_dvol)
    gap_1h, _ = sector_company_gap(moves["h1"], anchor_codes, lag_dvol)
    gap_30, _ = sector_company_gap(moves["m30"], anchor_codes, lag_dvol)
    return {
        "u_day_eq": row_rank(-gap_eq),
        "u_day_c": row_rank(-gap_c),
        "u_day_p": row_rank(-gap_p),
        "u_2h_c": row_rank(-gap_2h),
        "u_1h_c": row_rank(-gap_1h),
        "u_30m_c": row_rank(-gap_30),
        "u_vol": row_rank(-volume_state),
    }


def permute_within(values, codes, rng):
    out = values.copy()
    for c in np.unique(codes):
        cols = np.flatnonzero(codes == c)
        keys = rng.random((len(out), len(cols)))
        order = np.argsort(keys, axis=1)
        out[:, cols] = np.take_along_axis(out[:, cols], order, axis=1)
    return out


def shift_prev_day(arr, day_id):
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(day_id[1:] != day_id[:-1]) + 1]
    ends = np.r_[starts[1:], len(day_id)]
    loc = {int(day_id[a]): (int(a), int(b)) for a, b in zip(starts, ends)}
    for a, b in zip(starts, ends):
        prev = loc.get(int(day_id[a]) - 1)
        if prev is None:
            continue
        pa, pb = prev
        n = min(int(b - a), int(pb - pa))
        out[a : a + n] = arr[pa : pa + n]
    return out


def walk_fit_coef(cols, ztarget, eligible, raw_rel, width, day_id, day_dates, window):
    months = pd.DatetimeIndex(day_dates[day_id]).to_period("M")
    mix = np.full(ztarget.shape, np.nan)
    expected = np.full(ztarget.shape, np.nan)
    x_all = np.stack(cols, axis=-1)
    coefs = []
    for m in months.unique():
        rows = np.flatnonzero(months == m)
        first_day = day_id[rows[0]]
        lo = 0 if window is None else max(0, first_day - window)
        past = np.flatnonzero((day_id >= lo) & (day_id < first_day))
        if len(np.unique(day_id[past])) < 120:
            continue
        x = x_all[past][eligible[past]]
        y = ztarget[past][eligible[past]]
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        coef = np.linalg.lstsq(np.column_stack([x[ok], np.ones(int(ok.sum()))]), y[ok], rcond=None)[0][:-1]
        past_mix = x_all[past] @ coef
        ex = (past_mix * width[past])[eligible[past]]
        rr = raw_rel[past][eligible[past]]
        good = np.isfinite(ex) & np.isfinite(rr)
        scale = float((ex[good] * rr[good]).sum() / (ex[good] ** 2).sum())
        mix[rows] = x_all[rows] @ coef
        expected[rows] = scale * mix[rows] * width[rows]
        coefs.append(coef)
    return mix, expected, np.vstack(coefs)


def daily_ic(score, target, eligible, day_id, row_idx):
    buckets = {}
    for i in row_idx:
        c = rank_corr(
            np.where(eligible[i], score[i], np.nan),
            np.where(eligible[i], target[i], np.nan),
            20,
        )
        if np.isfinite(c):
            buckets.setdefault(int(day_id[i]), []).append(c)
    daily = np.array([float(np.mean(v)) for v in buckets.values()])
    mu = float(daily.mean()) if len(daily) else float("nan")
    se = float(daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 2 else float("nan")
    tstat = float(mu / se) if np.isfinite(se) and se > 0 else float("nan")
    return mu, tstat, int(len(daily))


def pack(net, gross, pick, val, show):
    out = {}
    for name, mask in (("train", pick), ("val", val), ("all", show)):
        s = stats(net[mask])
        out[name] = {
            "per_year": s["per_year"],
            "worst_drop": s["worst_drop"],
            "ratio": s["ratio"],
            "net_bp": float(net[mask].mean() * 1e4),
            "gross_bp": float(gross[mask].mean() * 1e4),
        }
    return out


def p_ge(null_values, real):
    null_values = np.asarray(null_values, dtype=np.float64)
    return float((1 + np.sum(null_values >= real)) / (1 + len(null_values)))


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    meta_all, fetched = combined_panel(json.loads(SPLIT_PATH.read_text(encoding="utf-8")))
    u_all = pd.read_csv(ROOT / "universe_all.csv")
    u_all["key"] = u_all["symbol"].str.replace(".", "_", regex=False)
    sec_map = dict(zip(u_all["key"], u_all["sector"]))
    nb_names = [n for n in meta_all["names"] if sec_map.get(n) in NON_BULL_SECTORS]
    meta = dict(meta_all, names=nb_names, n_names=len(nb_names))
    print(f"[pool] {meta['n_names']} names, gate {THRESH}, price >= ${MIN_PX:.0f}", flush=True)

    panel = build(meta)
    steps, bar_ret = decision_bars(meta)
    moves = load_moves(meta)
    dates = panel["day_dates"]
    day_id = panel["day_id"]
    labels = panel["day_label_all"]
    n_days = len(dates)
    assert len(day_id) == len(steps) == len(moves["day"])
    codes = panel["codes"]
    big = [c for c in range(len(panel["sector_order"])) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    spread, fee = COSTS[PICK_COST]
    factor = real_price_factor(nb_names, fetched, dates)
    real = panel["price"] / factor[day_id]
    eligible = np.isfinite(real) & (real >= MIN_PX)
    c_in = (spread + fee) / real
    c_out = fee / real
    ztarget = np.clip(panel["resid"] / panel["width"], -Z_CLIP, Z_CLIP)
    raw_rel = demean(panel["raw_close"])
    rows = np.flatnonzero(dates[day_id] >= PICK_FROM)
    pick = (labels == "train") & (dates >= PICK_FROM)
    val = labels == "val"
    show = (labels != "drop") & (dates >= PICK_FROM)
    train_rows = rows[labels[day_id[rows]] == "train"]
    val_rows = rows[labels[day_id[rows]] == "val"]
    print(f"[panel] {time.perf_counter() - t0:.0f}s", flush=True)

    base = coordinates(moves, moves["lag_dvol"], codes, codes, panel["f3"])
    specs = {
        "Full C/P": ["u_day_c", "u_day_p", "u_2h_c", "u_1h_c", "u_30m_c", "u_vol"],
        "Drop slow half": ["u_day_c", "u_2h_c", "u_1h_c", "u_30m_c", "u_vol"],
        "Drop volume": ["u_day_c", "u_day_p", "u_2h_c", "u_1h_c", "u_30m_c"],
        "Drop short windows": ["u_day_c", "u_day_p", "u_vol"],
        "Short windows only": ["u_2h_c", "u_1h_c", "u_30m_c"],
        "Close-gap vs C only": ["u_day_c"],
        "Equal-weight anchor": ["u_day_eq", "u_vol"],
    }
    books = {}
    mix_full = exp_full = coefs = None
    for name, keys in specs.items():
        cols = [base[k] for k in keys]
        if name == "Full C/P":
            mix, expected, coefs = walk_fit_coef(
                cols, ztarget, eligible, raw_rel, panel["width"], day_id, dates, WINDOW
            )
            mix_full, exp_full = mix, expected
        else:
            mix, expected = walk_fit(cols, ztarget, eligible, raw_rel, panel["width"], day_id, dates, WINDOW)
        net, gross, held = run_threshold_book(
            mix[rows], expected[rows], eligible[rows], c_in[rows], c_out[rows],
            panel["raw_seg"][rows], bar_ret, steps[rows], day_id[rows], panel["last_of_day"][rows],
            codes, big, n_days, THRESH,
        )
        active = float((np.abs(held).sum(axis=1) > 1e-8).mean())
        books[name] = {
            "net": net,
            "gross": gross,
            "pack": pack(net, gross, pick, val, show),
            "active": active,
            "coefs": coefs if name == "Full C/P" else None,
        }
        s = books[name]["pack"]
        print(
            f"[{name:22s}] train {s['train']['per_year']*100:+5.2f}%/yr DD {s['train']['worst_drop']*100:6.2f}% "
            f"val {s['val']['per_year']*100:+5.2f}%  active {active:.2f}  [{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )

    def from_score(mix, expected, label):
        net, gross, held = run_threshold_book(
            mix[rows], expected[rows], eligible[rows], c_in[rows], c_out[rows],
            panel["raw_seg"][rows], bar_ret, steps[rows], day_id[rows], panel["last_of_day"][rows],
            codes, big, n_days, THRESH,
        )
        books[label] = {
            "net": net,
            "gross": gross,
            "pack": pack(net, gross, pick, val, show),
            "active": float((np.abs(held).sum(axis=1) > 1e-8).mean()),
            "coefs": None,
        }
        s = books[label]["pack"]
        print(
            f"[{label:22s}] train {s['train']['per_year']*100:+5.2f}%/yr DD {s['train']['worst_drop']*100:6.2f}% "
            f"val {s['val']['per_year']*100:+5.2f}%  active {books[label]['active']:.2f}  [{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )
        return net

    mix_shift = shift_prev_day(mix_full, day_id)
    exp_shift = shift_prev_day(exp_full, day_id)
    from_score(mix_shift, exp_shift, "Previous-day score")
    from_score(-mix_full, -exp_full, "Sign flipped")

    anchor_nets = []
    for i in range(N_ANCHOR):
        lag = permute_within(moves["lag_dvol"], codes, rng)
        coord = coordinates(moves, lag, codes, codes, panel["f3"])
        cols = [coord[k] for k in specs["Full C/P"]]
        mix, expected = walk_fit(cols, ztarget, eligible, raw_rel, panel["width"], day_id, dates, WINDOW)
        net, gross, _held = run_threshold_book(
            mix[rows], expected[rows], eligible[rows], c_in[rows], c_out[rows],
            panel["raw_seg"][rows], bar_ret, steps[rows], day_id[rows], panel["last_of_day"][rows],
            codes, big, n_days, THRESH,
        )
        anchor_nets.append(net)
        print(f"[random half {i + 1:02d}/{N_ANCHOR}] train {stats(net[pick])['per_year']*100:+5.2f}%  [{time.perf_counter() - t0:.0f}s]", flush=True)
    anchor_nets = np.vstack(anchor_nets)
    anchor_mean = anchor_nets.mean(axis=0)
    books["Random fast/slow mean"] = {
        "net": anchor_mean,
        "gross": anchor_mean,
        "pack": pack(anchor_mean, anchor_mean, pick, val, show),
        "active": float("nan"),
        "coefs": None,
    }

    sector_nets = []
    for i in range(N_SECTOR):
        fake = rng.permutation(codes)
        coord = coordinates(moves, moves["lag_dvol"], fake, codes, panel["f3"])
        cols = [coord[k] for k in specs["Full C/P"]]
        mix, expected = walk_fit(cols, ztarget, eligible, raw_rel, panel["width"], day_id, dates, WINDOW)
        net, _gross, _held = run_threshold_book(
            mix[rows], expected[rows], eligible[rows], c_in[rows], c_out[rows],
            panel["raw_seg"][rows], bar_ret, steps[rows], day_id[rows], panel["last_of_day"][rows],
            codes, big, n_days, THRESH,
        )
        sector_nets.append(net)
        print(f"[wrong sector {i + 1:02d}/{N_SECTOR}] train {stats(net[pick])['per_year']*100:+5.2f}%  [{time.perf_counter() - t0:.0f}s]", flush=True)
    sector_nets = np.vstack(sector_nets)
    sector_mean = sector_nets.mean(axis=0)
    books["Wrong-sector mean"] = {
        "net": sector_mean,
        "gross": sector_mean,
        "pack": pack(sector_mean, sector_mean, pick, val, show),
        "active": float("nan"),
        "coefs": None,
    }

    real_tr_bp = books["Full C/P"]["pack"]["train"]["net_bp"]
    real_va_bp = books["Full C/P"]["pack"]["val"]["net_bp"]
    anchor_tr = anchor_nets[:, pick].mean(axis=1) * 1e4
    anchor_va = anchor_nets[:, val].mean(axis=1) * 1e4
    sector_tr = sector_nets[:, pick].mean(axis=1) * 1e4
    sector_va = sector_nets[:, val].mean(axis=1) * 1e4
    _, diff_eq_mu, diff_eq_t = newey_west((books["Full C/P"]["net"] - books["Equal-weight anchor"]["net"])[pick], 5)
    _, diff_eq_mu_v, diff_eq_t_v = newey_west((books["Full C/P"]["net"] - books["Equal-weight anchor"]["net"])[val], 5)

    ic_names = [
        ("Since close vs C", base["u_day_c"]),
        ("Since close vs slow", base["u_day_p"]),
        ("2h vs C", base["u_2h_c"]),
        ("1h vs C", base["u_1h_c"]),
        ("30m vs C", base["u_30m_c"]),
        ("Volume state", base["u_vol"]),
        ("Since close vs equal-weight", base["u_day_eq"]),
        ("Fitted mix", mix_full),
    ]
    ic = {}
    for name, score in ic_names:
        tr = daily_ic(score, panel["resid"], eligible, day_id, train_rows)
        va = daily_ic(score, panel["resid"], eligible, day_id, val_rows)
        ic[name] = {"train_ic": tr[0], "train_t": tr[1], "train_days": tr[2], "val_ic": va[0], "val_t": va[1], "val_days": va[2]}
        print(f"[IC {name:28s}] train {tr[0]:+.4f} (t {tr[1]:+5.2f})  val {va[0]:+.4f} (t {va[1]:+5.2f})", flush=True)

    coef_names = ["since close vs C", "since close vs slow", "2h vs C", "1h vs C", "30m vs C", "volume"]
    coef_mean = coefs.mean(axis=0)
    coef_sign = np.mean(np.sign(coefs) == np.sign(coef_mean), axis=0)
    print("[coefs] " + "  ".join(f"{n} {c:+.3f} same-sign {p:.0%}" for n, c, p in zip(coef_names, coef_mean, coef_sign)), flush=True)

    years = pd.DatetimeIndex(dates).year.to_numpy()
    year_rows = {}
    for y in sorted(set(years[show].tolist())):
        mask = show & (years == y)
        year_rows[str(y)] = {
            name: float(np.prod(1.0 + books[name]["net"][mask]) - 1.0)
            for name in ("Full C/P", "Equal-weight anchor", "Previous-day score", "Sign flipped")
        }

    report = {
        "pool": "non-bull 97",
        "threshold": THRESH,
        "min_price": MIN_PX,
        "n_random_anchor": N_ANCHOR,
        "n_wrong_sector": N_SECTOR,
        "books": {
            k: {"pack": v["pack"], "active_rows": v["active"]}
            for k, v in books.items()
        },
        "random_anchor": {
            "train_bp_mean": float(anchor_tr.mean()),
            "train_p": p_ge(anchor_tr, real_tr_bp),
            "val_bp_mean": float(anchor_va.mean()),
            "val_p": p_ge(anchor_va, real_va_bp),
        },
        "wrong_sector": {
            "train_bp_mean": float(sector_tr.mean()),
            "train_p": p_ge(sector_tr, real_tr_bp),
            "val_bp_mean": float(sector_va.mean()),
            "val_p": p_ge(sector_va, real_va_bp),
        },
        "full_minus_equal_weight": {
            "train_bp": float(diff_eq_mu * 1e4),
            "train_t": float(diff_eq_t),
            "val_bp": float(diff_eq_mu_v * 1e4),
            "val_t": float(diff_eq_t_v),
        },
        "ic": ic,
        "coef_mean": {n: float(c) for n, c in zip(coef_names, coef_mean)},
        "coef_same_sign": {n: float(p) for n, p in zip(coef_names, coef_sign)},
        "year_total": year_rows,
        "note": "Gate fixed at 4.2. Test bars not loaded. Random-anchor and wrong-sector books are refit.",
    }
    print(
        f"[null anchor] train real {real_tr_bp:+.2f}bp/d vs mean {anchor_tr.mean():+.2f}  p={report['random_anchor']['train_p']:.3f}  "
        f"| val p={report['random_anchor']['val_p']:.3f}",
        flush=True,
    )
    print(
        f"[null sector] train real {real_tr_bp:+.2f}bp/d vs mean {sector_tr.mean():+.2f}  p={report['wrong_sector']['train_p']:.3f}  "
        f"| val p={report['wrong_sector']['val_p']:.3f}",
        flush=True,
    )
    print(
        f"[full - equal-weight] train {diff_eq_mu*1e4:+.2f}bp/d t={diff_eq_t:.2f}  val {diff_eq_mu_v*1e4:+.2f}bp/d t={diff_eq_t_v:.2f}",
        flush=True,
    )

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.4))
    d = dates[show]
    ax = axes[0, 0]
    ax.axvspan(dates[val][0], d[-1], color="#eef1f5", zorder=0)
    for path in anchor_nets:
        ax.plot(d, (np.cumprod(1.0 + path[show]) - 1.0) * 100.0, color="#c5cad3", lw=0.8, alpha=0.85, zorder=1)
    lines = [
        ("Full C/P framework", books["Full C/P"]["net"], "#1f4e79", "-", 2.3),
        ("Equal-weight anchor, same gate", books["Equal-weight anchor"]["net"], "#d06a3a", "--", 1.8),
        ("Previous-day score", books["Previous-day score"]["net"], "#6b4c9a", "-.", 1.6),
        ("Sign flipped", books["Sign flipped"]["net"], "#a33b3b", ":", 1.7),
        ("Wrong-sector mean", sector_mean, "#2f6f4e", "--", 1.5),
    ]
    for lbl, series, col, ls, lw in lines:
        st = stats(series[show])
        ax.plot(
            d,
            (np.cumprod(1.0 + series[show]) - 1.0) * 100.0,
            color=col, ls=ls, lw=lw, zorder=2,
            label=f"{lbl}: {st['per_year']*100:+.1f}%/yr (DD {st['worst_drop']*100:.1f}%)",
        )
    ax.plot([], [], color="#c5cad3", lw=1.4, label=f"Random fast/slow half ({N_ANCHOR} refits)")
    ax.axhline(0, color="#9aa3ad", lw=0.6)
    ax.set_ylabel("Cumulative after-cost return (%)")
    ax.set_title("A. Same 4.2x gate. Gray lines rebuild the fast half at random", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)

    ax = axes[0, 1]
    order = [
        "Full C/P", "Drop slow half", "Drop volume", "Drop short windows",
        "Short windows only", "Close-gap vs C only", "Equal-weight anchor",
        "Previous-day score", "Sign flipped", "Random fast/slow mean", "Wrong-sector mean",
    ]
    y_pos = np.arange(len(order))
    train_pct = [books[k]["pack"]["train"]["per_year"] * 100 for k in order]
    val_pct = [books[k]["pack"]["val"]["per_year"] * 100 for k in order]
    ax.barh(y_pos + 0.16, train_pct, height=0.32, color="#1f4e79", label="Train, annualized")
    ax.barh(y_pos - 0.16, val_pct, height=0.32, color="#7ba3d0", label="Validation, annualized")
    ax.axvline(0, color="#1c2430", lw=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("After-cost return (%/yr)")
    ax.set_title("B. Which piece still makes money at the same gate", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color="#e6e9ee", lw=0.6)

    ax = axes[1, 0]
    ic_order = [n for n, _ in ic_names]
    y_pos = np.arange(len(ic_order))
    ax.barh(y_pos + 0.16, [ic[n]["train_ic"] for n in ic_order], height=0.32, color="#1f4e79", label="Train")
    ax.barh(y_pos - 0.16, [ic[n]["val_ic"] for n in ic_order], height=0.32, color="#7ba3d0", label="Validation")
    ax.axvline(0, color="#1c2430", lw=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ic_order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean daily rank correlation with residual to the close")
    ax.set_title("C. Does the score point at the later residual", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color="#e6e9ee", lw=0.6)

    ax = axes[1, 1]
    ys = sorted(year_rows)
    x = np.arange(len(ys))
    width = 0.2
    year_colors = {
        "Full C/P": "#1f4e79",
        "Equal-weight anchor": "#d06a3a",
        "Previous-day score": "#6b4c9a",
        "Sign flipped": "#a33b3b",
    }
    for i, name in enumerate(year_colors):
        ax.bar(x + (i - 1.5) * width, [year_rows[y][name] * 100 for y in ys], width=width, color=year_colors[name], label=name)
    ax.axhline(0, color="#1c2430", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ys)
    ax.set_ylabel("Total after-cost return in that year (%)")
    ax.set_title("D. Year totals. 2025 includes the validation months", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=7.6, loc="best")
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)

    for axis in axes.ravel():
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.text(
        0.5, 0.012,
        f"Non-bull 97 names, real price >= $60, gate fixed at 4.2x. "
        f"Random half train p={report['random_anchor']['train_p']:.2f}. "
        f"Wrong sector train p={report['wrong_sector']['train_p']:.2f}. "
        f"Full minus equal-weight train {diff_eq_mu*1e4:+.2f} bp/day (t={diff_eq_t:.1f}). Test period not opened.",
        ha="center", fontsize=8.2, color="#6b7380",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_img = OUT_DIR / "cs_fw_test.png"
    fig.savefig(out_img, dpi=150)
    extra = ROOT / "cs_force_v4_out"
    extra.mkdir(parents=True, exist_ok=True)
    fig.savefig(extra / "cs_fw_test.png", dpi=150)
    plt.close(fig)
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[wrote] {out_img}", flush=True)
    print(f"[done] {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
