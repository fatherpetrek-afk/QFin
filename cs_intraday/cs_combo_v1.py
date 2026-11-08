"""
Three small pullbacks combined, held longer, read against the trading cost.

The two earlier experiments left three signals that pointed the same way on
train and on validation, all of them pullbacks after the common factors are
removed:

  gap  how far the name has run ahead of its own GICS sector since last close
  f4   whether it closed near the low of the last 15 minutes
  f3   whether price rose together with volume over the last hour

Each is ranked inside the cross-section at the decision bar. A linear mix is
fit on train only, against the next residual divided by the name's own width.
Decisions are on the hour, 10:00 to 15:00. Holding is either one hour or until
the close.

Sectors are picked on train by a rule fixed in advance: the within-sector
rank correlation of the mix with the move to the close has t >= 3. Validation
reads those sectors without changing the pick.

The money line is per unit of gross capital, half long and half short inside
each sector, top fifth against bottom fifth, on the residual. A round trip
costs about 2 basis points per unit of capital. Test bars are not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import label_days, load_fields
from cs_f7state_v1 import demean, f7_features, kalman_states, roll_mean, walk_bar_residual
from cs_peer_v1 import sector_codes, static_peer_mean
from cs_resid_v1 import newey_west
from cs_xdist_v1 import style_axes

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_combo_v1_out"
STEP = 3
HOUR_BARS = 12
WIDTH_BARS = 390
MIN_SECTOR_NAMES = 5
PICK_T = 3.0
TOP_FRAC = 0.2
COST_BPS = 2.0
Z_CLIP = 5.0


def row_rank(x: np.ndarray) -> np.ndarray:
    """Rank inside each row, mapped to [-1, 1]. Rows with fewer than 20 names stay missing."""
    out = np.full(x.shape, np.nan)
    for i in range(len(x)):
        m = np.isfinite(x[i])
        n = int(m.sum())
        if n < 20:
            continue
        r = np.argsort(np.argsort(x[i, m])).astype(np.float64)
        out[i, m] = 2.0 * r / (n - 1) - 1.0
    return out


def rank_corr(x: np.ndarray, y: np.ndarray, min_n: int) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < min_n:
        return np.nan
    xr = np.argsort(np.argsort(x[m])).astype(np.float64)
    yr = np.argsort(np.argsort(y[m])).astype(np.float64)
    xr -= xr.mean()
    yr -= yr.mean()
    den = np.sqrt((xr**2).sum() * (yr**2).sum())
    return float((xr * yr).sum() / den) if den > 1e-12 else np.nan


def ic_rows(signal, target, cols=None, min_n=20) -> np.ndarray:
    out = np.full(len(signal), np.nan)
    for i in range(len(signal)):
        x = signal[i] if cols is None else signal[i, cols]
        y = target[i] if cols is None else target[i, cols]
        out[i] = rank_corr(x, y, min_n)
    return out


def sector_pnl(signal, target, codes, sectors) -> np.ndarray:
    """Per row: top fifth long, bottom fifth short inside each sector, equal weight
    across sectors. Return per unit of gross capital."""
    out = np.full(len(signal), np.nan)
    groups = [np.flatnonzero(codes == c) for c in sectors]
    for i in range(len(signal)):
        legs = []
        for cols in groups:
            x = signal[i, cols]
            y = target[i, cols]
            m = np.isfinite(x) & np.isfinite(y)
            n = int(m.sum())
            if n < MIN_SECTOR_NAMES:
                continue
            k = max(1, int(np.floor(TOP_FRAC * n)))
            order = np.argsort(x[m])
            yy = y[m]
            legs.append(0.5 * (yy[order[-k:]].mean() - yy[order[:k]].mean()))
        if legs:
            out[i] = float(np.mean(legs))
    return out


def daily_mean(values: np.ndarray, day_id: np.ndarray, rows: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"d": day_id[rows], "v": values[rows]})
    return frame.groupby("d")["v"].mean().to_numpy()


def nw_summary(daily: np.ndarray, scale: float = 1.0) -> dict:
    n, mu, t = newey_west(daily, 1)
    good = daily[np.isfinite(daily)]
    return {
        "days": n,
        "mean": mu * scale,
        "t": t,
        "positive_share": float((good > 0).mean()) if len(good) else float("nan"),
    }


def fit_mix(columns, z, rows) -> np.ndarray:
    x = np.stack([c[rows] for c in columns], axis=-1)
    y = z[rows]
    ok = np.isfinite(x).all(axis=-1) & np.isfinite(y)
    x = np.column_stack([x[ok], np.ones(int(ok.sum()))])
    coef = np.linalg.lstsq(x, y[ok], rcond=None)[0]
    return coef[:-1]


def save_figure(path, table, note: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    hours = list(table["hours"])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True)
    for ax, horizon, title in zip(axes, ("hour", "close"), ("持有 1 小时", "持有到收盘")):
        x = np.arange(len(hours))
        for offset, key, color, label in (
            (-0.2, "picked", "#1f4e79", "训练段挑出的行业"),
            (0.2, "all", "#b7cce4", "全部行业"),
        ):
            vals = [table[horizon][key][h]["val"]["mean"] for h in hours]
            errs = [abs(v / t) if t and np.isfinite(t) and abs(t) > 1e-9 else 0.0 for v, t in zip(vals, [table[horizon][key][h]["val"]["t"] for h in hours])]
            ax.bar(x + offset, vals, width=0.38, color=color, label=label, yerr=errs, error_kw={"elinewidth": 0.8, "ecolor": "#6b7380"})
        ax.axhline(COST_BPS, color="#d06a3a", lw=1.1, ls="--", label="来回成本约 2 个基点")
        ax.axhline(0.0, color="#9aa3ad", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(hours)
        ax.set_xlabel("决策时刻")
        ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
        style_axes(ax)
    axes[0].set_ylabel("验证段，每投入 1 块钱的毛利（基点），竖线是一倍标准误")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("三个回吐信号合在一起，够不够付成本", fontsize=15, color="#1c2430")
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
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
    print(f"[walk] {time.perf_counter() - t0:.0f}s", flush=True)

    feats = f7_features(eps_feat, volume, high, low, close.to_numpy(np.float64))
    minute = np.array([t.hour * 60 + t.minute for t in times.time])
    grid15 = ((minute - 570) % (5 * STEP) == 0) & (minute >= 600) & (minute <= 900)
    steps15 = np.flatnonzero(grid15)
    day15 = pd.DatetimeIndex(times[steps15]).normalize()
    day_index = pd.DatetimeIndex(sorted(set(day15)))
    day_id15 = np.searchsorted(day_index, day15)
    z = feats[steps15]
    z = (z - np.nanmean(z, axis=1, keepdims=True)) / (np.nanstd(z, axis=1, keepdims=True) + 1e-12)
    states15, _p = kalman_states(z, day_id15)
    hourly = (minute[steps15] - 600) % 60 == 0
    steps = steps15[hourly]
    states = states15[hourly]
    day_id = day_id15[hourly]
    hour_label = np.array([f"{m // 60}:{m % 60:02d}" for m in minute[steps]])
    day_label = label_days(day_index, train_end, val_start)[day_id]
    train = day_label == "train"
    val = day_label == "val"

    finite = np.isfinite(eps_full)
    cum = np.cumsum(np.where(finite, eps_full, 0.0), axis=0)
    cnt = np.cumsum(finite, axis=0)
    day_start = np.flatnonzero(first_bar)
    day_end = np.append(day_start[1:], len(times)) - 1
    bar_day = np.cumsum(first_bar) - 1
    start_of = day_start[bar_day[steps]]
    end_of = day_end[bar_day[steps]]
    before = np.where(start_of[:, None] > 0, cum[np.maximum(start_of - 1, 0)], 0.0)
    before_n = np.where(start_of[:, None] > 0, cnt[np.maximum(start_of - 1, 0)], 0)
    since_close = cum[steps] - before
    seen = cnt[steps] - before_n
    since_close[seen < (steps - start_of + 1)[:, None] * 0.8] = np.nan
    to_close = cum[end_of] - cum[steps]
    to_close_n = cnt[end_of] - cnt[steps]
    to_close[to_close_n < (end_of - steps)[:, None]] = np.nan
    hour_end = np.minimum(steps + HOUR_BARS, end_of)
    next_hour = cum[hour_end] - cum[steps]
    next_hour[(cnt[hour_end] - cnt[steps]) < (hour_end - steps)[:, None]] = np.nan
    targets = {"hour": demean(next_hour), "close": demean(to_close)}
    remaining = {"hour": (hour_end - steps).astype(np.float64), "close": (end_of - steps).astype(np.float64)}
    sigma_bar = np.sqrt(roll_mean(eps_feat**2, WIDTH_BARS, need_frac=0.5))[steps]

    codes, sector_order = sector_codes(names)
    same = (codes[:, None] == codes[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    moved = demean(since_close)
    gap = moved - static_peer_mean(moved, same)
    signals = {
        "gap": row_rank(-gap),
        "f4": row_rank(states[..., 3]),
        "f3": row_rank(-states[..., 2]),
    }
    labels_sig = {"gap": "早盘跑出同行业越多越看空", "f4": "收在近 15 分钟低点越看多", "f3": "放量上涨越看空"}
    big_sectors = [c for c in range(len(sector_order)) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    hours = sorted(set(hour_label), key=lambda s: int(s.split(":")[0]))

    report = {"signals": {}, "mix": {}, "sectors": {}, "money": {}, "hours": hours}
    table = {"hours": hours}
    for horizon, target in targets.items():
        width = sigma_bar * np.sqrt(np.maximum(remaining[horizon], 1.0))[:, None]
        ztarget = np.clip(target / width, -Z_CLIP, Z_CLIP)
        ztarget[~np.isfinite(width) | (width <= 0)] = np.nan
        for name, sig in signals.items():
            ic = ic_rows(sig, target)
            row = {
                "train": nw_summary(daily_mean(ic, day_id, train)),
                "val": nw_summary(daily_mean(ic, day_id, val)),
            }
            report["signals"].setdefault(name, {})[horizon] = row
            print(
                f"[{horizon:5s} {name:3s}] IC train {row['train']['mean']:+.4f} (t {row['train']['t']:+.1f})  "
                f"val {row['val']['mean']:+.4f} (t {row['val']['t']:+.1f})",
                flush=True,
            )
        cols = [signals["gap"], signals["f4"], signals["f3"]]
        train_rows = train[:, None] & np.ones(target.shape, dtype=bool)
        coef = fit_mix(cols, ztarget, train_rows)
        mix = sum(c * b for c, b in zip(cols, coef))
        ic = ic_rows(mix, target)
        mix_row = {
            "coef": [float(c) for c in coef],
            "train": nw_summary(daily_mean(ic, day_id, train)),
            "val": nw_summary(daily_mean(ic, day_id, val)),
        }
        report["mix"][horizon] = mix_row
        print(
            f"[{horizon:5s} mix] coef gap/f4/f3 {np.round(coef, 4)}  IC train {mix_row['train']['mean']:+.4f} "
            f"(t {mix_row['train']['t']:+.1f})  val {mix_row['val']['mean']:+.4f} (t {mix_row['val']['t']:+.1f})",
            flush=True,
        )
        if horizon == "close":
            picked = []
            for c in big_sectors:
                cols_c = np.flatnonzero(codes == c)
                ic_c = ic_rows(mix, target, cols_c, MIN_SECTOR_NAMES)
                row = {
                    "n_names": int(len(cols_c)),
                    "train": nw_summary(daily_mean(ic_c, day_id, train)),
                    "val": nw_summary(daily_mean(ic_c, day_id, val)),
                }
                row["picked"] = bool(row["train"]["t"] >= PICK_T)
                if row["picked"]:
                    picked.append(c)
                report["sectors"][sector_order[c]] = row
                print(
                    f"[sector {sector_order[c]:24s}] train {row['train']['mean']:+.4f} (t {row['train']['t']:+.1f})  "
                    f"val {row['val']['mean']:+.4f} (t {row['val']['t']:+.1f})  {'PICK' if row['picked'] else ''}",
                    flush=True,
                )
            mix_close = mix
        else:
            mix_hour = mix

    picked_names = [sector_order[c] for c in picked]
    print(f"[picked on train] {picked_names}", flush=True)
    for horizon, mix in (("hour", mix_hour), ("close", mix_close)):
        target = targets[horizon]
        table[horizon] = {"picked": {}, "all": {}}
        for key, secs in (("picked", picked), ("all", big_sectors)):
            pnl = sector_pnl(mix, target, codes, secs) if secs else np.full(len(mix), np.nan)
            per = {}
            for h in hours + ["all hours"]:
                sel = np.ones(len(mix), dtype=bool) if h == "all hours" else hour_label == h
                per[h] = {
                    "train": nw_summary(daily_mean(pnl, day_id, train & sel), 1e4),
                    "val": nw_summary(daily_mean(pnl, day_id, val & sel), 1e4),
                }
            table[horizon][key] = per
            line = "  ".join(f"{h} {per[h]['val']['mean']:+.2f}" for h in hours)
            print(
                f"[money {horizon:5s} {key:6s}] val bp per unit: {line}  | all {per['all hours']['val']['mean']:+.2f} "
                f"(t {per['all hours']['val']['t']:+.1f})  train all {per['all hours']['train']['mean']:+.2f}",
                flush=True,
            )
    report["money"] = table
    report["picked_sectors"] = picked_names

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(
        OUT_DIR / "cs_combo_money.png",
        table,
        f"行业按训练段规则挑出：{'、'.join(picked_names) if picked_names else '无'}。行业内上五分之一做多、下五分之一做空，"
        "按残差算，未扣成本。不是净值。",
    )
    report["labels"] = labels_sig
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
