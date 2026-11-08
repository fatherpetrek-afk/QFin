"""
Full relative move = projected shock + leftover, both scaled before the open.

The projection width is the recent-quarter cloud of factor coordinates, stretched
by the train-fitted 5-day/63-day mix. The leftover has its own width from the
same two clocks. The shape of each piece is the longer cloud of past
standardized outcomes, so the far tail is not limited to one quarter.

After the open, that print rescales the width of the remaining open-to-close
move. It does not move the center. Test bars are not loaded. No profit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import label_days, load_fields
from cs_dynproj_v1 import component_energy, hit_rate
from cs_proj_v1 import systematic_relative, walk_projection
from cs_resid_v1 import daily_last_close, fit_scale
from cs_xdist_v1 import style_axes
from nested_v1 import FAST_WIN, SLOW_WIN

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_improve_v1_out"
SHAPE_Q = np.array([0.025, 0.10, 0.25, 0.75, 0.90, 0.975])
MIN_CLOUD = SLOW_WIN


def daily_first(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index = []
    for day, group in frame.groupby(frame.index.normalize(), sort=True):
        rows.append(group.iloc[0].to_numpy(np.float64))
        index.append(pd.Timestamp(day))
    return pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(index), columns=frame.columns)


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of values[t-window:t]. values may already be squared."""
    total = np.cumsum(np.where(np.isfinite(values), values, 0.0), axis=0)
    count = np.cumsum(np.isfinite(values).astype(np.int32), axis=0)
    out = np.full(values.shape, np.nan)
    need = max(3, window // 2)
    for t in range(window, len(values)):
        end = t - 1
        start = t - window - 1
        summed = total[end] - (total[start] if start >= 0 else 0.0)
        seen = count[end] - (count[start] if start >= 0 else 0)
        ok = seen >= need
        out[t, ok] = summed[ok] / seen[ok]
    return out


def past_moments(z_a: np.ndarray, z_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """At index t, moments of days strictly before t. Both series must be finite."""
    both = np.isfinite(z_a) & np.isfinite(z_b)
    zero = np.zeros(z_a.shape[1])
    count = np.vstack([zero, np.cumsum(both, axis=0)])
    sum_a = np.vstack([zero, np.cumsum(np.where(both, z_a**2, 0.0), axis=0)])
    sum_b = np.vstack([zero, np.cumsum(np.where(both, z_b**2, 0.0), axis=0)])
    sum_ab = np.vstack([zero, np.cumsum(np.where(both, z_a * z_b, 0.0), axis=0)])
    return count[:-1], sum_a[:-1], sum_b[:-1], sum_ab[:-1]


def combined_width(scale_a, scale_b, count, sum_a, sum_b, sum_ab) -> np.ndarray:
    out = np.full(scale_a.shape, np.nan)
    ok = (count >= MIN_CLOUD) & np.isfinite(scale_a) & np.isfinite(scale_b)
    n = np.maximum(count, 1)
    var = (
        scale_a**2 * (sum_a / n)
        + scale_b**2 * (sum_b / n)
        + 2.0 * scale_a * scale_b * (sum_ab / n)
    )
    out[ok] = np.sqrt(np.maximum(var[ok], 0.0))
    return out


def cloud_quantiles(z_a, z_b, scale_a, scale_b, days) -> np.ndarray:
    quant = np.full(scale_a.shape + (len(SHAPE_Q),), np.nan)
    for t in days:
        both = np.isfinite(z_a[:t]) & np.isfinite(z_b[:t])
        count = both.sum(axis=0)
        keep = (count >= MIN_CLOUD) & np.isfinite(scale_a[t]) & np.isfinite(scale_b[t])
        if not np.any(keep):
            continue
        draws = scale_a[t] * z_a[:t] + scale_b[t] * z_b[:t]
        draws[:, ~keep] = np.nan
        got = np.nanquantile(draws, SHAPE_Q, axis=0).T
        quant[t, keep] = got[keep]
    return quant


def save_groups(path, index, val, group, quant, realized, width, train, note: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    shades = (
        (0, 5, "#e4edf6", "预测 95%"),
        (1, 4, "#b9cfe6", "预测 80%"),
        (2, 3, "#7ea4cb", "预测 50%"),
    )
    when = index[val]
    fig, axes = plt.subplots(3, 1, figsize=(11.6, 7.8), sharex=True)
    val_pos = np.flatnonzero(val)
    usable = val[:, None] & np.isfinite(realized) & np.isfinite(quant).all(axis=2)
    for panel, j in enumerate((0, 1, 2)):
        ax = axes[panel]
        med = np.full(len(val_pos), np.nan)
        real_lo = np.full(len(val_pos), np.nan)
        real_hi = np.full(len(val_pos), np.nan)
        pred = {}
        for a, b, color, label in shades:
            lo_d = np.full(len(val_pos), np.nan)
            hi_d = np.full(len(val_pos), np.nan)
            for i, t in enumerate(val_pos):
                m = (group[t] == j) & np.isfinite(quant[t]).all(axis=1)
                if m.sum() < 5:
                    continue
                lo_d[i] = np.median(np.minimum(quant[t, m, a], quant[t, m, b]))
                hi_d[i] = np.median(np.maximum(quant[t, m, a], quant[t, m, b]))
                if a == 0:
                    taken = realized[t, m]
                    med[i] = np.median(taken)
                    real_lo[i] = np.quantile(taken, 0.10)
                    real_hi[i] = np.quantile(taken, 0.90)
            pred[(a, b)] = (lo_d, hi_d, color, label)
        for (a, b), (lo_d, hi_d, color, label) in pred.items():
            ax.fill_between(when, lo_d * 100, hi_d * 100, color=color, lw=0, label=label, zorder=1)
        ax.plot(when, real_lo * 100, color="#d06a3a", lw=0.9, ls="--", label="真实的 10% / 90%", zorder=3)
        ax.plot(when, real_hi * 100, color="#d06a3a", lw=0.9, ls="--", zorder=3)
        ax.plot(when, med * 100, color="#16324f", lw=1.2, label="真实中位", zorder=4)
        member = usable & (group == j)
        hit80, _n = hit_rate(realized, quant[:, :, 1], quant[:, :, 4], member)
        share = float(member.sum() / max(int(usable.sum()), 1))
        mean_w = float(np.nanmean(width[train[:, None] & (group == j)]))
        ax.set_title(
            f"第 {panel + 1} 组，预测宽度 {mean_w * 100:.2f}% ，验证集约占 {share:.0%}，80% 盖住 {hit80:.0%}",
            loc="left",
            color="#1c2430",
            fontsize=11,
        )
        ax.set_ylabel("相对涨跌（%）")
        style_axes(ax)
        wide_lo, wide_hi = pred[(0, 5)][0], pred[(0, 5)][1]
        shown_y = np.concatenate([med, real_lo, real_hi, wide_lo, wide_hi])
        shown_y = shown_y[np.isfinite(shown_y)] * 100
        if shown_y.size:
            y0, y1 = np.quantile(shown_y, [0.02, 0.98])
            pad = 0.12 * max(y1 - y0, 0.2)
            ax.set_ylim(y0 - pad, y1 + pad)
        if panel == 0:
            ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("验证集交易日。色带是预测分布，虚线是这组当天的真实相对变动")
    fig.suptitle("预测分布 vs 真实", fontsize=16, color="#1c2430")
    fig.text(0.5, 0.004, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
    ax.set_xlabel("开盘前的预测宽度（%）")
    ax.set_ylabel("当天真实相对变动（%）")
    ax.set_title("预测分布 vs 真实", loc="left", color="#1c2430")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tercile_groups(width, train) -> tuple[np.ndarray, np.ndarray]:
    edges = np.nanquantile(width[train], [1.0 / 3.0, 2.0 / 3.0])
    group = np.full(width.shape, -1, dtype=int)
    known = np.isfinite(width)
    group[known & (width <= edges[0])] = 0
    group[known & (width > edges[0]) & (width <= edges[1])] = 1
    group[known & (width > edges[1])] = 2
    return group, edges


def level_dict(y, quant, mask) -> dict:
    pairs = {"50": (2, 3), "80": (1, 4), "95": (0, 5)}
    return {name: hit_rate(y, quant[:, :, a], quant[:, :, b], mask) for name, (a, b) in pairs.items()}


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_end = pd.Timestamp(meta["train_end"])
    val_start = pd.Timestamp(meta["val_eval_start"])
    val_end = pd.Timestamp(meta["val_end"])
    fields = load_fields(meta["names"], val_end)
    assert fields["close"].index.max() <= val_end
    close = fields["close"]
    opened = fields["open"]
    keep = close.index.normalize() < val_end.normalize()
    close = close[keep]
    opened = opened[keep]
    prices = daily_last_close(close)
    first_open = daily_first(opened).reindex(prices.index)
    daily_ret = prices.pct_change(fill_method=None)
    index, loading, factor, returns = walk_projection(
        close.pct_change(fill_method=None).to_numpy(np.float64),
        close.index,
        daily_ret.iloc[1:],
    )
    labels = label_days(index, train_end, val_start)
    train = labels == "train"
    market = np.nanmean(returns, axis=1)
    projected = systematic_relative(loading, factor, returns)
    full_relative = returns - market[:, None]
    leftover = full_relative - projected
    delta = loading - np.nanmean(loading, axis=1)[:, None, :]

    fast = component_energy(factor, FAST_WIN)
    slow = component_energy(factor, SLOW_WIN)
    alpha_f, epochs_f = fit_scale(np.nansum(factor**2, axis=1), np.nansum(fast, axis=1), np.nansum(slow, axis=1), train)
    sigma = np.sqrt(np.maximum(alpha_f * fast + (1.0 - alpha_f) * slow, 0.0))
    base = np.nansum(slow, axis=1)
    pred = np.nansum(sigma**2, axis=1)
    stretch = np.full(len(factor), np.nan)
    ok_scale = np.isfinite(base) & np.isfinite(pred) & (base > 1e-12) & (pred > 0)
    stretch[ok_scale] = np.sqrt(pred[ok_scale] / base[ok_scale])

    n_days, n_names, _k = delta.shape
    width_p = np.full((n_days, n_names), np.nan)
    for t in range(SLOW_WIN, n_days):
        if not np.isfinite(stretch[t]):
            continue
        past = factor[t - SLOW_WIN : t]
        good = np.isfinite(past).all(axis=1)
        if int(good.sum()) < SLOW_WIN // 2:
            continue
        db = delta[t]
        good_name = np.isfinite(db).all(axis=1)
        db_fill = np.where(good_name[:, None], db, 0.0)
        scores = (past[good] @ db_fill.T) * stretch[t]
        scores[:, ~good_name] = np.nan
        width_p[t] = np.sqrt(np.nanmean(scores**2, axis=0))

    fast_e = trailing_mean(leftover**2, FAST_WIN)
    slow_e = trailing_mean(leftover**2, SLOW_WIN)
    alpha_e, epochs_e = fit_scale(
        np.nanmean(leftover**2, axis=1),
        np.nanmean(fast_e, axis=1),
        np.nanmean(slow_e, axis=1),
        train,
    )
    var_e = np.maximum(alpha_e * fast_e + (1.0 - alpha_e) * slow_e, 0.0)
    width_e = np.sqrt(var_e)

    def standardize(move, width):
        out = np.full(move.shape, np.nan)
        ok = np.isfinite(move) & np.isfinite(width) & (width > 1e-8)
        out[ok] = move[ok] / width[ok]
        return out

    z_p = standardize(projected, width_p)
    z_e = standardize(leftover, width_e)
    count, sum_p, sum_e, sum_pe = past_moments(z_p, z_e)
    width_full = combined_width(width_p, width_e, count, sum_p, sum_e, sum_pe)
    val_days = np.flatnonzero(labels == "val")
    quant_full = cloud_quantiles(z_p, z_e, width_p, width_e, val_days)
    quant_proj = cloud_quantiles(z_p, np.zeros_like(z_p), width_p, np.zeros_like(width_p), val_days)

    # Open rescales the width of what is still left today. Center stays put.
    r_open = (first_open / prices.shift(1) - 1.0).reindex(index).to_numpy(np.float64)
    r_rest = (prices / first_open - 1.0).reindex(index).to_numpy(np.float64)
    open_rel = r_open - np.nanmean(r_open, axis=1)[:, None]
    rest_rel = r_rest - np.nanmean(r_rest, axis=1)[:, None]
    fast_r = trailing_mean(rest_rel**2, FAST_WIN)
    slow_r = trailing_mean(rest_rel**2, SLOW_WIN)
    alpha_r, epochs_r = fit_scale(
        np.nanmean(rest_rel**2, axis=1),
        np.nanmean(fast_r, axis=1),
        np.nanmean(slow_r, axis=1),
        train,
    )
    var_r = np.maximum(alpha_r * fast_r + (1.0 - alpha_r) * slow_r, 0.0)
    width_r = np.sqrt(var_r)
    open_e = np.nanmean(open_rel**2, axis=1)
    rest_e = np.nanmean(rest_rel**2, axis=1)
    pre_e = np.nanmean(var_r, axis=1)
    u_slow = np.full(n_days, np.nan)
    need_u = max(3, SLOW_WIN // 2)
    for t in range(SLOW_WIN, n_days):
        past_u = open_e[t - SLOW_WIN : t]
        seen = np.isfinite(past_u)
        if int(seen.sum()) < need_u:
            continue
        u_slow[t] = np.mean(past_u[seen])
    ratio = np.full(n_days, np.nan)
    ok_u = np.isfinite(open_e) & np.isfinite(u_slow) & (u_slow > 1e-12) & np.isfinite(pre_e) & (pre_e > 0)
    ratio[ok_u] = open_e[ok_u] / u_slow[ok_u]
    alpha_o, epochs_o = fit_scale(rest_e, pre_e * ratio, pre_e, train)
    mult = np.sqrt(np.maximum(alpha_o * ratio + (1.0 - alpha_o), 0.0))
    width_rest = width_r * mult[:, None]
    z_r = standardize(rest_rel, width_r)
    z_zero = np.zeros_like(z_r)
    count_r, sum_r, sum_z, sum_rz = past_moments(z_r, z_zero)
    width_rest_rms = combined_width(width_rest, z_zero, count_r, sum_r, sum_z, sum_rz)
    quant_rest = cloud_quantiles(z_r, z_zero, width_rest, z_zero, val_days)
    quant_rest_pre = cloud_quantiles(z_r, z_zero, width_r, z_zero, val_days)

    val = (labels == "val") & np.isfinite(quant_full).any(axis=(1, 2))
    usable_full = val[:, None] & np.isfinite(full_relative) & np.isfinite(quant_full).all(axis=2)
    usable_proj = val[:, None] & np.isfinite(projected) & np.isfinite(quant_proj).all(axis=2)
    usable_rest = val[:, None] & np.isfinite(rest_rel) & np.isfinite(quant_rest).all(axis=2)
    usable_pre = val[:, None] & np.isfinite(rest_rel) & np.isfinite(quant_rest_pre).all(axis=2)
    full_levels = level_dict(full_relative, quant_full, usable_full)
    proj_levels = level_dict(projected, quant_proj, usable_proj)
    rest_levels = level_dict(rest_rel, quant_rest, usable_rest)
    pre_levels = level_dict(rest_rel, quant_rest_pre, usable_pre)
    print(
        f"[scale] factor {alpha_f:.3f} leftover {alpha_e:.3f} rest {alpha_r:.3f} open {alpha_o:.3f}  "
        f"epochs {epochs_f}/{epochs_e}/{epochs_r}/{epochs_o}",
        flush=True,
    )
    for name in ("50", "80", "95"):
        print(
            f"[{name}] full {full_levels[name][0]:.3f}  projection {proj_levels[name][0]:.3f}  "
            f"rest-before-open {pre_levels[name][0]:.3f}  rest-after-open {rest_levels[name][0]:.3f}  "
            f"n {full_levels[name][1]}",
            flush=True,
        )
    print(
        f"[width] full p50 {np.nanmedian(width_full[val]):.4%}  "
        f"|stock-pool| p50 {np.nanmedian(np.abs(full_relative[val])):.4%}  "
        f"rest width p50 {np.nanmedian(width_rest_rms[val]):.4%}",
        flush=True,
    )

    group_full, edges_full = tercile_groups(width_full, train)
    group_rest, edges_rest = tercile_groups(width_rest_rms, train)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_groups(
        OUT_DIR / "cs_improve_val.png",
        index,
        val,
        group_full,
        quant_full,
        full_relative,
        width_full,
        train,
        f"开盘前。投影宽度加残差宽度，形状用更长的历史。验证集上 80% 盖住 {full_levels['80'][0]:.0%}。不是收益。",
    )
    save_fan(
        OUT_DIR / "cs_improve_fan.png",
        val,
        width_full,
        full_relative,
        quant_full,
        "验证集。点是股票相对池子的当天涨跌。色带是投影和残差合在一起的预测分布。不是收益。",
    )
    save_groups(
        OUT_DIR / "cs_improve_open.png",
        index,
        val,
        group_rest,
        quant_rest,
        rest_rel,
        width_rest_rms,
        train,
        f"开盘之后，用开盘的散开程度更新剩余时段的宽度，中心不动。80% 盖住 {rest_levels['80'][0]:.0%}。不是收益。",
    )

    def pack(levels):
        return {name: {"hit_rate": hit, "n": n} for name, (hit, n) in levels.items()}

    payload = {
        "alpha_factor": alpha_f,
        "alpha_leftover": alpha_e,
        "alpha_rest": alpha_r,
        "alpha_open": alpha_o,
        "epochs": {"factor": epochs_f, "leftover": epochs_e, "rest": epochs_r, "open": epochs_o},
        "train_full_width_terciles": [float(edges_full[0]), float(edges_full[1])],
        "train_rest_width_terciles": [float(edges_rest[0]), float(edges_rest[1])],
        "validation_coverage_full_relative": pack(full_levels),
        "validation_coverage_projection": pack(proj_levels),
        "validation_coverage_rest_before_open": pack(pre_levels),
        "validation_coverage_rest_after_open": pack(rest_levels),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
