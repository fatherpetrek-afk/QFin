"""
Second pass on the combined pullback, held to the close.

The first pass reached about the cost line, but its sector rule picked every
sector, so it did not filter anything. Every choice here is made on train:

  sectors   the two sectors with the highest train t on the within-sector rank
            correlation, against all seven large sectors
  depth     top and bottom fifth, against top and bottom tenth
  moved     only names whose gap from their sector is already larger than
            their own width since last close
  hour      the single decision hour with the best train edge, one trade a day

Each line is read twice: on the residual, and on the raw price move. A long
short pair inside one sector earns the raw difference, which still carries the
factor exposure the residual strips out. Test bars are not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import label_days, load_fields
from cs_combo_v1 import HOUR_BARS, MIN_SECTOR_NAMES, STEP, WIDTH_BARS, Z_CLIP, daily_mean, fit_mix, ic_rows, nw_summary, row_rank
from cs_f7state_v1 import demean, f7_features, kalman_states, roll_mean, walk_bar_residual
from cs_peer_v1 import sector_codes, static_peer_mean
from cs_xdist_v1 import style_axes

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_combo_v2_out"
COST_BPS = 2.0
N_TOP_SECTORS = 2


def window_sum(values, steps, ends):
    finite = np.isfinite(values)
    cum = np.cumsum(np.where(finite, values, 0.0), axis=0)
    cnt = np.cumsum(finite, axis=0)
    out = cum[ends] - cum[steps]
    out[(cnt[ends] - cnt[steps]) < (ends - steps)[:, None]] = np.nan
    return out


def build_panel(meta):
    names = list(meta["names"])
    train_end = pd.Timestamp(meta["train_end"])
    val_start = pd.Timestamp(meta["val_eval_start"])
    val_end = pd.Timestamp(meta["val_end"])
    fields = load_fields(names, val_end)
    assert fields["close"].index.max() <= val_end
    keep = fields["close"].index.normalize() < val_end.normalize()
    close = fields["close"][keep]
    high = fields["high"][keep].to_numpy(np.float64)
    low = fields["low"][keep].to_numpy(np.float64)
    volume = fields["volume"][keep].to_numpy(np.float64)
    volume = np.where(volume > 0, volume, np.nan)
    times = close.index
    bar_ret = close.pct_change(fill_method=None).to_numpy(np.float64)
    eps_full = walk_bar_residual(bar_ret, times)
    dates = np.array(times.date)
    first_bar = np.r_[True, dates[1:] != dates[:-1]]
    eps_feat = eps_full.copy()
    eps_feat[first_bar] = np.nan

    feats = f7_features(eps_feat, volume, high, low, close.to_numpy(np.float64))
    minute = np.array([t.hour * 60 + t.minute for t in times.time])
    grid15 = ((minute - 570) % (5 * STEP) == 0) & (minute >= 600) & (minute <= 900)
    steps15 = np.flatnonzero(grid15)
    day15 = pd.DatetimeIndex(times[steps15]).normalize()
    day_index = pd.DatetimeIndex(sorted(set(day15)))
    day_id15 = np.searchsorted(day_index, day15)
    z = feats[steps15]
    with np.errstate(all="ignore"):
        z = (z - np.nanmean(z, axis=1, keepdims=True)) / (np.nanstd(z, axis=1, keepdims=True) + 1e-12)
    states15, _p = kalman_states(z, day_id15)
    hourly = (minute[steps15] - 600) % 60 == 0
    steps = steps15[hourly]
    states = states15[hourly]
    day_id = day_id15[hourly]

    day_start = np.flatnonzero(first_bar)
    day_end = np.append(day_start[1:], len(times)) - 1
    bar_day = np.cumsum(first_bar) - 1
    start_of = day_start[bar_day[steps]]
    end_of = day_end[bar_day[steps]]
    since_close = window_sum(eps_full, np.maximum(start_of - 1, 0), steps)
    since_close[start_of == 0] = np.nan
    sigma_bar = np.sqrt(roll_mean(eps_feat**2, WIDTH_BARS, need_frac=0.5))[steps]
    codes, sector_order = sector_codes(names)
    same = (codes[:, None] == codes[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    moved = demean(since_close)
    gap = moved - static_peer_mean(moved, same)
    return {
        "hour_label": np.array([f"{m // 60}:{m % 60:02d}" for m in minute[steps]]),
        "day_id": day_id,
        "day_label": label_days(day_index, train_end, val_start)[day_id],
        "codes": codes,
        "sector_order": sector_order,
        "gap": gap,
        "gap_width": sigma_bar * np.sqrt((steps - start_of + 1).astype(np.float64))[:, None],
        "f4": states[..., 3],
        "f3": states[..., 2],
        "resid": demean(window_sum(eps_full, steps, end_of)),
        "raw": window_sum(bar_ret, steps, end_of),
        "width": sigma_bar * np.sqrt(np.maximum(end_of - steps, 1).astype(np.float64))[:, None],
    }


def pair_pnl(signal, target, codes, sectors, frac, eligible=None):
    """Per row, inside each sector: long the top `frac`, short the bottom `frac`,
    half the capital on each side, equal weight across sectors."""
    out = np.full(len(signal), np.nan)
    groups = [np.flatnonzero(codes == c) for c in sectors]
    for i in range(len(signal)):
        legs = []
        for cols in groups:
            x = signal[i, cols]
            y = target[i, cols]
            m = np.isfinite(x) & np.isfinite(y)
            n_all = int(m.sum())
            if n_all < MIN_SECTOR_NAMES:
                continue
            k = max(1, int(np.floor(frac * n_all)))
            if eligible is not None:
                m &= eligible[i, cols]
            xs, ys = x[m], y[m]
            order = np.argsort(xs)
            long_side = order[::-1][:k]
            long_side = long_side[xs[long_side] > 0]
            short_side = order[:k]
            short_side = short_side[xs[short_side] < 0]
            if len(long_side) == 0 or len(short_side) == 0:
                continue
            legs.append(0.5 * (ys[long_side].mean() - ys[short_side].mean()))
        if legs:
            out[i] = float(np.mean(legs))
    return out


def edge_lines(pnl, panel, hours):
    day_id = panel["day_id"]
    train = panel["day_label"] == "train"
    val = panel["day_label"] == "val"
    per = {}
    for h in hours + ["all hours"]:
        sel = np.ones(len(pnl), dtype=bool) if h == "all hours" else panel["hour_label"] == h
        per[h] = {
            "train": nw_summary(daily_mean(pnl, day_id, train & sel), 1e4),
            "val": nw_summary(daily_mean(pnl, day_id, val & sel), 1e4),
            "val_rows": int(np.isfinite(pnl[val & sel]).sum()),
        }
    best = max(hours, key=lambda h: per[h]["train"]["mean"])
    return per, best


def save_figure(path, rows, note):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    y = np.arange(len(rows))[::-1]
    for offset, key, color, label in ((0.18, "resid", "#1f4e79", "按残差"), (-0.18, "raw", "#d9a441", "按实际价格")):
        vals = [r[key]["mean"] for r in rows]
        errs = [abs(r[key]["mean"] / r[key]["t"]) if np.isfinite(r[key]["t"]) and abs(r[key]["t"]) > 1e-9 else 0.0 for r in rows]
        ax.barh(y + offset, vals, height=0.34, color=color, label=label, xerr=errs, error_kw={"elinewidth": 0.8, "ecolor": "#6b7380"})
    ax.axvline(COST_BPS, color="#d06a3a", lw=1.1, ls="--", label="来回成本约 2 个基点")
    ax.axvline(0.0, color="#9aa3ad", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("验证段，每投入 1 块钱的毛利（基点），横线是一倍标准误")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    fig.suptitle("持有到收盘：哪种做法能越过成本线", fontsize=15, color="#1c2430")
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    panel = build_panel(meta)
    print(f"[panel] {time.perf_counter() - t0:.0f}s", flush=True)
    day_id = panel["day_id"]
    train = panel["day_label"] == "train"
    codes = panel["codes"]
    order = panel["sector_order"]
    hours = sorted(set(panel["hour_label"]), key=lambda s: int(s.split(":")[0]))

    cols = [row_rank(-panel["gap"]), row_rank(panel["f4"]), row_rank(-panel["f3"])]
    ztarget = np.clip(panel["resid"] / panel["width"], -Z_CLIP, Z_CLIP)
    coef = fit_mix(cols, ztarget, train[:, None] & np.ones(ztarget.shape, dtype=bool))
    mix = sum(c * b for c, b in zip(cols, coef))
    print(f"[mix] coef gap/f4/f3 {np.round(coef, 4)}", flush=True)

    big = [c for c in range(len(order)) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    sector_t = {}
    for c in big:
        ic_c = ic_rows(mix, panel["resid"], np.flatnonzero(codes == c), MIN_SECTOR_NAMES)
        sector_t[c] = nw_summary(daily_mean(ic_c, day_id, train))["t"]
    top = sorted(big, key=lambda c: -sector_t[c])[:N_TOP_SECTORS]
    print(f"[sectors] train t: {[(order[c], round(sector_t[c], 1)) for c in big]}", flush=True)
    print(f"[sectors] top {N_TOP_SECTORS} on train: {[order[c] for c in top]}", flush=True)

    moved = np.abs(panel["gap"]) > panel["gap_width"]
    variants = [
        ("七个大行业，前后五分之一", big, 0.2, None),
        ("七个大行业，前后十分之一", big, 0.1, None),
        ("七个大行业，只做已经跑出去的", big, 0.2, moved),
        (f"训练段前两名行业，前后五分之一", top, 0.2, None),
        (f"训练段前两名行业，前后十分之一", top, 0.1, None),
        (f"训练段前两名行业，只做已经跑出去的", top, 0.2, moved),
    ]
    report = {"coef": [float(c) for c in coef], "sector_train_t": {order[c]: sector_t[c] for c in big},
              "top_sectors": [order[c] for c in top], "variants": {}}
    rows = []
    for label, secs, frac, elig in variants:
        entry = {}
        for key in ("resid", "raw"):
            pnl = pair_pnl(mix, panel[key], codes, secs, frac, elig)
            per, best = edge_lines(pnl, panel, hours)
            entry[key] = {"per_hour": per, "best_train_hour": best}
            print(
                f"[{key:5s}] {label:24s} all hours val {per['all hours']['val']['mean']:+.2f}bp "
                f"(t {per['all hours']['val']['t']:+.1f}, up days {per['all hours']['val']['positive_share']:.2f})  "
                f"train {per['all hours']['train']['mean']:+.2f}  | train-best hour {best}: val "
                f"{per[best]['val']['mean']:+.2f} (t {per[best]['val']['t']:+.1f}), train {per[best]['train']['mean']:+.2f}",
                flush=True,
            )
        report["variants"][label] = entry
        rows.append({"label": label, "resid": entry["resid"]["per_hour"]["all hours"]["val"],
                     "raw": entry["raw"]["per_hour"]["all hours"]["val"]})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(
        OUT_DIR / "cs_combo_v2_variants.png",
        rows,
        f"训练段前两名行业：{'、'.join(order[c] for c in top)}。六个整点决策平均，每次持有到收盘，未扣成本。不是净值。",
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
