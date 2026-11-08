"""
Relative move = dynamic within-pool loading x dynamic external shock.

The loading is the one known at the open: last week's slow state plus the
daily deviation already updated after yesterday's close. The shock's
predictive distribution uses only its own past: a train-fitted mix of the
5-day and 63-day energy, times the cloud of earlier standardized shocks.
The center is whatever that cloud has, which is about zero. Test bars are
not loaded. No profit is computed.
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
from cs_proj_v1 import systematic_relative, walk_projection
from cs_resid_v1 import daily_last_close, fit_scale
from cs_xdist_v1 import style_axes
from nested_v1 import FAST_WIN, SLOW_WIN

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_dynproj_v1_out"
SHAPE_Q = np.array([0.025, 0.10, 0.25, 0.75, 0.90, 0.975])


def component_energy(factor: np.ndarray, window: int) -> np.ndarray:
    """Mean of f^2 over the previous window, one scale per factor."""
    out = np.full(factor.shape, np.nan)
    sq = factor**2
    need = max(3, window // 2)
    for t in range(window, len(factor)):
        past = sq[t - window : t]
        good = np.isfinite(past).all(axis=1)
        if int(good.sum()) < need:
            continue
        out[t] = np.mean(past[good], axis=0)
    return out


def hit_rate(y, lo, hi, mask) -> tuple[float, int]:
    ok = mask & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if int(ok.sum()) == 0:
        return float("nan"), 0
    inside = (y[ok] >= np.minimum(lo[ok], hi[ok])) & (y[ok] <= np.maximum(lo[ok], hi[ok]))
    return float(inside.mean()), int(ok.sum())


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
    daily_ret = prices.pct_change(fill_method=None)
    index, loading, factor, returns = walk_projection(
        close.pct_change(fill_method=None).to_numpy(np.float64),
        close.index,
        daily_ret.iloc[1:],
    )
    labels = label_days(index, train_end, val_start)
    market = np.nanmean(returns, axis=1)
    relative = systematic_relative(loading, factor, returns)
    pool_relative = returns - market[:, None]
    delta = loading - np.nanmean(loading, axis=1)[:, None, :]

    fast = component_energy(factor, FAST_WIN)
    slow = component_energy(factor, SLOW_WIN)
    train = labels == "train"
    alpha, epochs = fit_scale(
        np.nansum(factor**2, axis=1),
        np.nansum(fast, axis=1),
        np.nansum(slow, axis=1),
        train,
    )
    sigma = np.sqrt(np.maximum(alpha * fast + (1.0 - alpha) * slow, 0.0))
    print(f"[scale] alpha {alpha:.3f} epochs {epochs}", flush=True)

    # The factor's coordinate system drifts with the loadings, so the shape
    # cloud is only the recent quarter. Today's external scale stretches it.
    base = np.nansum(slow, axis=1)
    pred = np.nansum(sigma**2, axis=1)
    stretch = np.full(len(factor), np.nan)
    ok_scale = np.isfinite(base) & np.isfinite(pred) & (base > 1e-12) & (pred > 0)
    stretch[ok_scale] = np.sqrt(pred[ok_scale] / base[ok_scale])

    n_days, n_names, _n_fac = delta.shape
    quant = np.full((n_days, n_names, len(SHAPE_Q)), np.nan)
    width = np.full((n_days, n_names), np.nan)
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
        width[t] = np.sqrt(np.nanmean(scores**2, axis=0))
        if labels[t] == "val":
            quant[t] = np.nanquantile(scores, SHAPE_Q, axis=0).T

    val = (labels == "val") & np.isfinite(quant).any(axis=(1, 2))
    print(
        f"[stretch] val p50 {np.nanmedian(stretch[val]):.2f}  p90 {np.nanquantile(stretch[val], 0.9):.2f}",
        flush=True,
    )
    usable = val[:, None] & np.isfinite(relative) & np.isfinite(quant).all(axis=2)
    levels = {
        "50": hit_rate(relative, quant[:, :, 2], quant[:, :, 3], usable),
        "80": hit_rate(relative, quant[:, :, 1], quant[:, :, 4], usable),
        "95": hit_rate(relative, quant[:, :, 0], quant[:, :, 5], usable),
    }
    total_levels = {
        "50": hit_rate(pool_relative, quant[:, :, 2], quant[:, :, 3], usable),
        "80": hit_rate(pool_relative, quant[:, :, 1], quant[:, :, 4], usable),
        "95": hit_rate(pool_relative, quant[:, :, 0], quant[:, :, 5], usable),
    }
    for name, (hit, count) in levels.items():
        print(
            f"[projected {name}] {hit:.3f} n {count}  [full relative {name}] {total_levels[name][0]:.3f}",
            flush=True,
        )
    exp_val = width[val]
    s_val = relative[val]
    print(
        f"[width] val |relative| p50 {np.nanmedian(np.abs(s_val)):.4%}  "
        f"width p50 {np.nanmedian(exp_val):.4%}",
        flush=True,
    )

    train_w = width[train]
    edges = np.nanquantile(train_w, [1.0 / 3.0, 2.0 / 3.0])
    group = np.full(width.shape, -1, dtype=int)
    known = np.isfinite(width)
    group[known & (width <= edges[0])] = 0
    group[known & (width > edges[0]) & (width <= edges[1])] = 1
    group[known & (width > edges[1])] = 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
                    realized = relative[t, m]
                    med[i] = np.median(realized)
                    real_lo[i] = np.quantile(realized, 0.10)
                    real_hi[i] = np.quantile(realized, 0.90)
            pred[(a, b)] = (lo_d, hi_d, color, label)
        for (a, b), (lo_d, hi_d, color, label) in pred.items():
            ax.fill_between(when, lo_d * 100, hi_d * 100, color=color, lw=0, label=label, zorder=1)
        ax.plot(when, real_lo * 100, color="#d06a3a", lw=0.9, ls="--", label="真实的 10% / 90%", zorder=3)
        ax.plot(when, real_hi * 100, color="#d06a3a", lw=0.9, ls="--", zorder=3)
        ax.plot(when, med * 100, color="#16324f", lw=1.2, label="真实中位", zorder=4)
        member = usable & (group == j)
        hit80, _n80 = hit_rate(relative, quant[:, :, 1], quant[:, :, 4], member)
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
    axes[-1].set_xlabel("验证集交易日。色带是开盘前的预测分布，虚线是这组当天真实的投影相对变动")
    fig.suptitle("预测分布 vs 真实", fontsize=16, color="#1c2430")
    fig.text(
        0.5,
        0.004,
        f"池内动态载荷 × 池外冲击分布。验证集上这条投影的 80% 盖住 {levels['80'][0]:.0%}。不是收益。",
        ha="center",
        fontsize=9,
        color="#6b7380",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT_DIR / "cs_dynproj_val.png", dpi=160)
    plt.close(fig)

    x = width[val] * 100
    yy = relative[val] * 100
    q = quant[val]
    m = np.isfinite(x) & np.isfinite(yy) & np.isfinite(q).all(axis=2)
    fig2, ax = plt.subplots(figsize=(11.2, 6.2))
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
    ax.set_ylabel("当天真实的投影相对变动（%）")
    ax.set_title("预测分布 vs 真实", loc="left", color="#1c2430")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    fig2.text(
        0.5,
        0.01,
        "验证集。点是每只股票当天的投影相对变动。色带是冲击的动态分布乘上这只股票当天的相对载荷。不是收益。",
        ha="center",
        fontsize=9,
        color="#6b7380",
    )
    fig2.tight_layout(rect=(0, 0.04, 1, 1))
    fig2.savefig(OUT_DIR / "cs_dynproj_fan.png", dpi=160)
    plt.close(fig2)

    payload = {
        "alpha_fast": alpha,
        "scale_epochs": epochs,
        "train_width_terciles": [float(edges[0]), float(edges[1])],
        "validation_coverage_projected": {name: {"hit_rate": hit, "n": n} for name, (hit, n) in levels.items()},
        "validation_coverage_full_relative": {
            name: {"hit_rate": hit, "n": n} for name, (hit, n) in total_levels.items()
        },
        "val_abs_relative_p50": float(np.nanmedian(np.abs(s_val))),
        "val_width_p50": float(np.nanmedian(exp_val)),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
