"""
Within-pool order changes the width of what is still left after 10:00.

The morning loading is already known. The first 30 minutes, projected on that
loading, give the early factor. Each name's lag loading is a regression of
its later within-pool move on that factor over the previous quarter, using
only earlier days. Their product is known at the cut.

The late move is split into that product and the leftover. The forecast
recombines the two with the second moments of earlier standardized pieces.
Today's product enters as a magnitude, so the band can widen or narrow.
The center stays the center of the historical cloud. It is not moved onto
the product itself.

The baseline is the 5-day/63-day energy of the late move, with no order term.
Validation only. Test bars are not loaded. No profit is computed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import label_days, load_fields
from cs_dynproj_v1 import hit_rate
from cs_improve_v1 import (
    cloud_quantiles,
    combined_width,
    past_moments,
    save_groups,
    tercile_groups,
    trailing_mean,
)
from cs_proj_v1 import walk_projection
from cs_resid_v1 import daily_last_close, fit_scale
from cs_xdist_v1 import style_axes
from factor_v1 import K, MIN_DAY_BARS, MIN_NAMES_FRAC, project
from nested_v1 import FAST_WIN, SLOW_WIN

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_order_v1_out"
CUT = pd.Timestamp("10:00").time()


def prices_at_cut(close: pd.DataFrame) -> pd.DataFrame:
    """Last close at or before 10:00. Days without a later bar stay missing."""
    rows = []
    index = []
    blank = np.full(close.shape[1], np.nan)
    for day, group in close.groupby(close.index.normalize(), sort=True):
        index.append(pd.Timestamp(day))
        early = group[group.index.time <= CUT]
        if len(group) < MIN_DAY_BARS or early.empty or group.index[-1].time() <= CUT:
            rows.append(blank.copy())
            continue
        rows.append(early.iloc[::-1].bfill().iloc[0].to_numpy(np.float64))
    return pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(index), columns=close.columns)


def demean_rows(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    good = np.isfinite(values)
    count = good.sum(axis=1)
    need = max(10, int(MIN_NAMES_FRAC * values.shape[1]))
    ok = np.flatnonzero(count >= need)
    if len(ok) == 0:
        return out
    summed = np.where(good, values, 0.0).sum(axis=1)
    mean = summed[ok] / count[ok]
    picked = values[ok] - mean[:, None]
    picked[~good[ok]] = np.nan
    out[ok] = picked
    return out


def order_shock(late_rel: np.ndarray, f_early: np.ndarray, window: int) -> np.ndarray:
    """L from the previous `window` days, times today's early factor."""
    n_days, n_names = late_rel.shape
    k = f_early.shape[1]
    shock = np.full((n_days, n_names), np.nan)
    eye = 1e-10 * np.eye(k)
    need = max(k + 2, window // 2)
    for t in range(window, n_days):
        f_now = f_early[t]
        if not np.isfinite(f_now).all():
            continue
        f_past = f_early[t - window : t]
        y_past = late_rel[t - window : t]
        good = np.isfinite(f_past).all(axis=1)
        if int(good.sum()) < need:
            continue
        ff = f_past[good]
        yy = y_past[good]
        finite = np.isfinite(yy)
        ready = finite.sum(axis=0) >= need
        complete = ready & finite.all(axis=0)
        if np.any(complete):
            gram = ff.T @ ff + eye
            coef = np.linalg.solve(gram, ff.T @ yy[:, complete])
            shock[t, complete] = f_now @ coef
        for i in np.flatnonzero(ready & ~complete):
            m = finite[:, i]
            f_i = ff[m]
            gram = f_i.T @ f_i + eye
            coef = np.linalg.solve(gram, f_i.T @ yy[m, i])
            shock[t, i] = float(f_now @ coef)
    return shock


def row_mean(values: np.ndarray) -> np.ndarray:
    good = np.isfinite(values)
    count = good.sum(axis=1)
    out = np.full(values.shape[0], np.nan)
    ok = count > 0
    out[ok] = np.where(good, values, 0.0).sum(axis=1)[ok] / count[ok]
    return out


def mix_width(move: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, float, int]:
    fast = trailing_mean(move**2, FAST_WIN)
    slow = trailing_mean(move**2, SLOW_WIN)
    alpha, epochs = fit_scale(
        row_mean(move**2),
        row_mean(fast),
        row_mean(slow),
        train,
    )
    var = np.maximum(alpha * fast + (1.0 - alpha) * slow, 0.0)
    return np.sqrt(var), alpha, epochs


def standardize(move: np.ndarray, width: np.ndarray) -> np.ndarray:
    out = np.full(move.shape, np.nan)
    ok = np.isfinite(move) & np.isfinite(width) & (width > 1e-8)
    out[ok] = move[ok] / width[ok]
    return out


def level_dict(y, quant, mask) -> dict:
    pairs = {"50": (2, 3), "80": (1, 4), "95": (0, 5)}
    return {name: hit_rate(y, quant[:, :, a], quant[:, :, b], mask) for name, (a, b) in pairs.items()}


def save_fan(path, val, width, realized, quant, note: str) -> None:
    x = width[val] * 100
    yy = realized[val] * 100
    q = quant[val]
    m = np.isfinite(x) & np.isfinite(yy) & np.isfinite(q).all(axis=2) & (width[val] > 1e-8)
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    unit = np.full(q.shape[-1], np.nan)
    for a in range(q.shape[-1]):
        ratio = q[:, :, a] / width[val]
        unit[a] = np.nanmedian(ratio[m])
    grid = np.linspace(np.nanquantile(x[m], 0.01), np.nanquantile(x[m], 0.99), 200)
    colors = ("#d9e4f0", "#b7cce4", "#7ea4cb")
    labels_q = ("预测 95%", "预测 80%", "预测 50%")
    for (a, b), color, label in zip(((0, 5), (1, 4), (2, 3)), colors, labels_q):
        lo_g = np.minimum(grid * unit[a], grid * unit[b])
        hi_g = np.maximum(grid * unit[a], grid * unit[b])
        ax.fill_between(grid, lo_g, hi_g, color=color, lw=0, label=label, zorder=1)
    ax.plot(grid, grid * 0.5 * (unit[2] + unit[3]), color="#16324f", lw=1.3, label="预测中位", zorder=3)
    shown = m & (yy >= np.nanquantile(yy[m], 0.01)) & (yy <= np.nanquantile(yy[m], 0.99))
    ax.scatter(x[shown], yy[shown], s=6, c="#1f4e79", alpha=0.15, lw=0, zorder=2)
    style_axes(ax)
    ax.set_xlabel("10:00 时的预测宽度（%）")
    ax.set_ylabel("10:00 之后的真实相对变动（%）")
    ax.set_title("预测分布 vs 真实", loc="left", color="#1c2430")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pack(levels) -> dict:
    return {name: {"hit_rate": hit, "n": n} for name, (hit, n) in levels.items()}


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_end = pd.Timestamp(meta["train_end"])
    val_start = pd.Timestamp(meta["val_eval_start"])
    val_end = pd.Timestamp(meta["val_end"])
    fields = load_fields(meta["names"], val_end)
    assert fields["close"].index.max() <= val_end
    close = fields["close"]
    close = close[close.index.normalize() < val_end.normalize()]
    prices = daily_last_close(close)
    cut_px = prices_at_cut(close).reindex(prices.index)
    daily_ret = prices.pct_change(fill_method=None)
    index, loading, _factor, _returns = walk_projection(
        close.pct_change(fill_method=None).to_numpy(np.float64),
        close.index,
        daily_ret.iloc[1:],
    )
    labels = label_days(index, train_end, val_start)
    train = labels == "train"
    prev = prices.shift(1).reindex(index).to_numpy(np.float64)
    cut = cut_px.reindex(index).to_numpy(np.float64)
    last = prices.reindex(index).to_numpy(np.float64)
    usable_px = np.isfinite(prev) & np.isfinite(cut) & np.isfinite(last) & (prev > 0) & (cut > 0) & (last > 0)
    r_early = np.full(cut.shape, np.nan)
    r_late = np.full(cut.shape, np.nan)
    r_early[usable_px] = cut[usable_px] / prev[usable_px] - 1.0
    r_late[usable_px] = last[usable_px] / cut[usable_px] - 1.0
    early_rel = demean_rows(r_early)
    late_rel = demean_rows(r_late)

    n_days = len(index)
    f_early = np.full((n_days, K), np.nan)
    for t in range(n_days):
        f_t, _eps = project(r_early[t], loading[t])
        f_early[t] = f_t
    shock = order_shock(late_rel, f_early, SLOW_WIN)
    leftover = late_rel - shock

    width_base, alpha_base, epochs_base = mix_width(late_rel, train)
    width_left, alpha_left, epochs_left = mix_width(leftover, train)
    width_shock, alpha_shock, epochs_shock = mix_width(shock, train)
    width_early, alpha_early, epochs_early = mix_width(early_rel, train)

    z_left = standardize(leftover, width_left)
    z_shock = standardize(shock, width_shock)
    z_late = standardize(late_rel, width_base)
    z_zero = np.zeros_like(z_late)
    scale_shock = np.abs(shock)
    count, sum_l, sum_s, sum_ls = past_moments(z_left, z_shock)
    width_order = combined_width(width_left, scale_shock, count, sum_l, sum_s, sum_ls)
    val_days = np.flatnonzero(labels == "val")
    quant_order = cloud_quantiles(z_left, z_shock, width_left, scale_shock, val_days)
    quant_base = cloud_quantiles(z_late, z_zero, width_base, z_zero, val_days)

    val = (labels == "val") & np.isfinite(quant_order).any(axis=(1, 2))
    usable_order = val[:, None] & np.isfinite(late_rel) & np.isfinite(quant_order).all(axis=2)
    usable_base = val[:, None] & np.isfinite(late_rel) & np.isfinite(quant_base).all(axis=2)
    moved = np.isfinite(early_rel) & np.isfinite(width_early) & (np.abs(early_rel) > width_early)
    still = np.isfinite(early_rel) & np.isfinite(width_early) & (np.abs(early_rel) <= width_early)

    slices = {
        "all": np.ones(late_rel.shape, dtype=bool),
        "already_moved": moved,
        "not_yet": still,
    }
    order_levels = {name: level_dict(late_rel, quant_order, usable_order & mask) for name, mask in slices.items()}
    base_levels = {name: level_dict(late_rel, quant_base, usable_base & mask) for name, mask in slices.items()}

    mse_ok = (labels == "val")[:, None] & np.isfinite(late_rel) & np.isfinite(shock)
    sse_lag = float(np.mean((late_rel[mse_ok] - shock[mse_ok]) ** 2))
    sse_zero = float(np.mean(late_rel[mse_ok] ** 2))
    mse_ratio = sse_lag / sse_zero if sse_zero > 0 else float("nan")

    print(
        f"[scale] base {alpha_base:.3f} leftover {alpha_left:.3f} shock {alpha_shock:.3f} early {alpha_early:.3f}  "
        f"epochs {epochs_base}/{epochs_left}/{epochs_shock}/{epochs_early}",
        flush=True,
    )
    print(f"[mse] lag/zero {mse_ratio:.3f}  n {int(mse_ok.sum())}", flush=True)
    for name in ("50", "80", "95"):
        print(
            f"[{name}] base {base_levels['all'][name][0]:.3f}  order {order_levels['all'][name][0]:.3f}  "
            f"moved {base_levels['already_moved'][name][0]:.3f}->{order_levels['already_moved'][name][0]:.3f}  "
            f"still {base_levels['not_yet'][name][0]:.3f}->{order_levels['not_yet'][name][0]:.3f}",
            flush=True,
        )
    gap80 = abs(order_levels["all"]["80"][0] - base_levels["all"]["80"][0])
    print(f"[gap] 80% all {gap80:.3f}", flush=True)

    group, edges = tercile_groups(width_order, train)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    note = (
        f"10:00 之后的池内相对变动。顺序只改宽度，中心仍是历史云的中心。"
        f"验证集上 80% 盖住 {order_levels['all']['80'][0]:.0%}，基线是 {base_levels['all']['80'][0]:.0%}。不是收益。"
    )
    save_groups(
        OUT_DIR / "cs_order_val.png",
        index,
        val,
        group,
        quant_order,
        late_rel,
        width_order,
        train,
        note,
    )
    save_fan(
        OUT_DIR / "cs_order_fan.png",
        val,
        width_order,
        late_rel,
        quant_order,
        "验证集。点是 10:00 之后股票相对池子的涨跌。色带是顺序宽度下的预测分布。不是收益。",
    )

    payload = {
        "cut": "10:00",
        "alpha_baseline": alpha_base,
        "alpha_leftover": alpha_left,
        "alpha_shock": alpha_shock,
        "alpha_early": alpha_early,
        "epochs": {
            "baseline": epochs_base,
            "leftover": epochs_left,
            "shock": epochs_shock,
            "early": epochs_early,
        },
        "mse_ratio_lag_vs_zero": mse_ratio,
        "mse_lag": sse_lag,
        "mse_zero": sse_zero,
        "mse_n": int(mse_ok.sum()),
        "train_order_width_terciles": [float(edges[0]), float(edges[1])],
        "validation_coverage_baseline": {name: pack(levels) for name, levels in base_levels.items()},
        "validation_coverage_order": {name: pack(levels) for name, levels in order_levels.items()},
        "coverage_gap_80_all": gap80,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
