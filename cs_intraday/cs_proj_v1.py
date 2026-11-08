"""
Relative move inside the pool = exposure x the external factor.

Exposure is the name's loading minus the cross-sectional mean, known before
the day. The factor's predictive distribution is the train-fitted fast/slow
scale times the cloud of earlier standardized factors. A GMM, fit on train
exposures only, groups names so each group's band can be checked separately.

Validation only. Test bars are not loaded. No profit is computed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from combined_v1 import label_days, load_fields
from cs_resid_v1 import daily_last_close, fit_scale
from cs_xdist_v1 import style_axes
from factor_v1 import (
    K,
    MIN_NAMES_FRAC,
    QUARTER_DAYS,
    RHO,
    align_sign,
    day_loadings,
    project,
    top_loadings,
)
from nested_v1 import FAST_WIN, SLOW_WIN, causal_window, hold_weekly

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_proj_v1_out"
SHAPE_Q = np.array([0.025, 0.10, 0.25, 0.75, 0.90, 0.975])
K_CANDIDATES = (2, 3, 4)


def walk_projection(bar_returns: np.ndarray, times: pd.DatetimeIndex, daily_ret: pd.DataFrame):
    """Morning loading, that day's factor, and the close-to-close cross-section."""
    dates = np.array(times.date)
    change = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    day_starts = change
    day_ends = np.append(change[1:], len(times))
    iso = pd.DatetimeIndex(times).isocalendar()
    week_key = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    ret_lookup = {ts.normalize(): daily_ret.loc[ts].to_numpy(np.float64) for ts in daily_ret.index}
    n_names = bar_returns.shape[1]
    b_slow = None
    delta = np.zeros((n_names, K))
    rows = []
    for d in range(len(day_starts)):
        a, b = int(day_starts[d]), int(day_ends[d])
        if b_slow is not None:
            loadings = b_slow + delta
            day = pd.Timestamp(times[a]).normalize()
            if day in ret_lookup:
                r_day = ret_lookup[day]
                f, _eps = project(r_day, loadings)
                rows.append({"day": day, "b": loadings.copy(), "f": np.asarray(f, np.float64).copy(), "r": r_day.copy()})
            r_bar = bar_returns[a:b]
            if np.isfinite(r_bar).all():
                gram = loadings.T @ loadings + 1e-10 * np.eye(K)
                f_bar = r_bar @ np.linalg.solve(gram, loadings.T).T
            else:
                f_bar = np.full((b - a, K), np.nan)
                for i, t_i in enumerate(range(a, b)):
                    f_i, _ = project(bar_returns[t_i], loadings)
                    f_bar[i] = f_i
            realized = day_loadings(f_bar, bar_returns[a:b])
            if realized is not None:
                upd = np.isfinite(realized).all(axis=1) & np.isfinite(b_slow).all(axis=1)
                delta[upd] = RHO * delta[upd] + (1.0 - RHO) * (realized[upd] - b_slow[upd])
        week_end = d + 1 == len(day_starts) or week_key[b - 1] != week_key[day_starts[d + 1]]
        if d + 1 >= QUARTER_DAYS and (d + 1 == QUARTER_DAYS or week_end):
            fresh_pair = top_loadings(bar_returns[int(day_starts[d + 1 - QUARTER_DAYS]) : b])
            if fresh_pair is not None:
                fresh, used = fresh_pair
                fresh = align_sign(fresh, b_slow)
                if b_slow is None:
                    delta[:] = 0.0
                else:
                    old_total = b_slow + delta
                    delta = old_total - fresh
                    first = used & (np.max(np.abs(old_total), axis=1) < 1e-8)
                    delta[first] = 0.0
                b_slow = fresh
    order = sorted(rows, key=lambda row: row["day"])
    index = pd.DatetimeIndex([row["day"] for row in order])
    loading = np.stack([row["b"] for row in order])
    factor = np.stack([row["f"] for row in order])
    returns = np.vstack([row["r"] for row in order])
    return index, loading, factor, returns


def systematic_relative(loading: np.ndarray, factor: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """(B_i - mean B) · f, in return units. This is the projected relative move."""
    fitted = np.einsum("tnk,tk->tn", loading, factor)
    ok = np.isfinite(returns) & np.isfinite(factor).all(axis=1)[:, None] & np.isfinite(fitted)
    fitted = np.where(ok, fitted, np.nan)
    center = np.nanmean(fitted, axis=1)
    return fitted - center[:, None]


def causal_beta(relative: np.ndarray, market: np.ndarray, window: int) -> np.ndarray:
    """Each name's exposure to the external move, using only the previous window."""
    out = np.full(relative.shape, np.nan)
    for t in range(window, len(market)):
        g = market[t - window : t]
        s = relative[t - window : t]
        gm = np.nanmean(g)
        gc = g - gm
        good_g = np.isfinite(gc)
        if good_g.sum() < window // 2:
            continue
        var = float(np.dot(gc[good_g], gc[good_g]) / max(good_g.sum() - 1, 1))
        if var <= 1e-12:
            continue
        sm = np.nanmean(s, axis=0)
        sc = s - sm
        use = good_g[:, None] & np.isfinite(sc)
        acc = np.where(use, sc * gc[:, None], 0.0).sum(axis=0)
        count = use.sum(axis=0)
        cov = acc / np.maximum(count - 1, 1)
        beta = cov / var
        beta[count < window // 2] = np.nan
        out[t] = beta
    return out


def projected_variance(factor: np.ndarray, delta: np.ndarray, window: int) -> np.ndarray:
    """Variance of (B_i - mean B) · f, using today's loading on earlier factors.

    The three factors cancel inside the dot product. A diagonal variance misses
    that cancellation and inflates the width, so this uses the sample variance
    of the projected scalar itself.
    """
    out = np.full(delta.shape[:2], np.nan)
    min_obs = max(3, window // 2)
    for t in range(window, len(factor)):
        past = factor[t - window : t]
        good = np.isfinite(past).all(axis=1)
        if int(good.sum()) < min_obs:
            continue
        db = np.where(np.isfinite(delta[t]), delta[t], 0.0)
        scores = past[good] @ db.T
        out[t] = np.var(scores, axis=0, ddof=1)
    return out


def _row_mean(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], np.nan)
    good = np.isfinite(values).any(axis=1)
    if good.any():
        out[good] = np.nanmean(values[good], axis=1)
    return out


def choose_gmm(x_train: np.ndarray):
    best = None
    table = []
    scaler = StandardScaler().fit(x_train)
    z = scaler.transform(x_train)
    for k in K_CANDIDATES:
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=0)
        model.fit(z)
        labels = model.predict(z)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < 0.05 * len(labels):
            table.append({"k": k, "bic": None, "kept": False})
            continue
        bic = float(model.bic(z))
        table.append({"k": k, "bic": bic, "kept": True})
        if best is None or bic < best[0]:
            best = (bic, model, scaler)
    if best is None:
        raise RuntimeError("no usable GMM on train exposures")
    return best[1], best[2], table


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
    systematic = systematic_relative(loading, factor, returns)
    delta = loading - np.nanmean(loading, axis=1)[:, None, :]
    var_fast = projected_variance(factor, delta, FAST_WIN)
    var_slow = projected_variance(factor, delta, SLOW_WIN)
    energy = _row_mean(systematic**2)
    fast_e = _row_mean(var_fast)
    slow_e = _row_mean(var_slow)
    train = labels == "train"
    alpha, epochs = fit_scale(energy, fast_e, slow_e, train)
    exposure = np.sqrt(np.maximum(alpha * var_fast + (1.0 - alpha) * var_slow, 0.0))
    relative = systematic
    pool_relative = returns - market[:, None]
    corr = 1.0
    val_now = labels == "val"
    exp_val = exposure[val_now]
    s_val = systematic[val_now]
    print(
        f"[scale] alpha {alpha:.3f} epochs {epochs}  "
        f"val |relative| p50 {np.nanmedian(np.abs(s_val)):.4%}  "
        f"width p50 {np.nanmedian(exp_val):.4%}",
        flush=True,
    )

    shape = np.full((len(index), len(SHAPE_Q)), np.nan)
    z_hist = []
    for t in range(len(index)):
        width = exposure[t]
        move = systematic[t]
        ok = np.isfinite(width) & (width > 1e-8) & np.isfinite(move)
        if len(z_hist) >= 500:
            shape[t] = np.quantile(z_hist, SHAPE_Q)
        if ok.any():
            z_hist.extend((move[ok] / width[ok]).tolist())
    unit = shape

    val = (labels == "val") & np.isfinite(unit).all(axis=1)
    usable = val[:, None] & np.isfinite(exposure) & np.isfinite(relative)
    lo = exposure * unit[:, 0][:, None]
    hi = exposure * unit[:, -1][:, None]
    lo80 = exposure * unit[:, 1][:, None]
    hi80 = exposure * unit[:, -2][:, None]
    lo50 = exposure * unit[:, 2][:, None]
    hi50 = exposure * unit[:, -3][:, None]
    levels = {
        "50": hit_rate(relative, lo50, hi50, usable),
        "80": hit_rate(relative, lo80, hi80, usable),
        "95": hit_rate(relative, lo, hi, usable),
    }
    total_levels = {
        "50": hit_rate(pool_relative, lo50, hi50, usable),
        "80": hit_rate(pool_relative, lo80, hi80, usable),
        "95": hit_rate(pool_relative, lo, hi, usable),
    }
    for name, (hit, count) in levels.items():
        print(f"[projected {name}] {hit:.3f} n {count}  [full relative {name}] {total_levels[name][0]:.3f}", flush=True)

    train_x = exposure[train[:, None] & np.isfinite(exposure)].reshape(-1, 1)
    gmm, scaler, bic_table = choose_gmm(train_x)
    k = int(gmm.n_components)
    group = np.full(exposure.shape, -1, dtype=int)
    known = np.isfinite(exposure)
    group[known] = gmm.predict(scaler.transform(exposure[known].reshape(-1, 1)))
    print(f"[gmm] K {k}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = np.argsort([np.nanmean(exposure[train[:, None] & (group == j)]) for j in range(k)])
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    shades = ((0, -1, "#e4edf6", "预测 95%"), (1, -2, "#b9cfe6", "预测 80%"), (2, -3, "#7ea4cb", "预测 50%"))
    when = index[val]
    fig, axes = plt.subplots(k, 1, figsize=(11.6, 2.6 * k), sharex=True)
    if k == 1:
        axes = [axes]
    val_pos = np.flatnonzero(val)
    for panel, j in enumerate(order):
        ax = axes[panel]
        med = np.full(len(val_pos), np.nan)
        real_lo = np.full(len(val_pos), np.nan)
        real_hi = np.full(len(val_pos), np.nan)
        pred = {}
        for a, b, color, label in shades:
            lo_d = np.full(len(val_pos), np.nan)
            hi_d = np.full(len(val_pos), np.nan)
            for i, t in enumerate(val_pos):
                m = (group[t] == j) & np.isfinite(exposure[t])
                if m.sum() < 5 or not np.isfinite(unit[t]).all():
                    continue
                band_lo = exposure[t, m] * unit[t, a]
                band_hi = exposure[t, m] * unit[t, b]
                lo_d[i] = np.median(np.minimum(band_lo, band_hi))
                hi_d[i] = np.median(np.maximum(band_lo, band_hi))
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
        hit80, _n80 = hit_rate(relative, lo80, hi80, member)
        share = float(member.sum() / max(int(usable.sum()), 1))
        mean_exp = float(np.nanmean(exposure[train[:, None] & (group == j)]))
        ax.set_title(
            f"第 {panel + 1} 组，预测宽度 {mean_exp * 100:.2f}% ，验证集约占 {share:.0%}，80% 盖住 {hit80:.0%}",
            loc="left",
            color="#1c2430",
            fontsize=11,
        )
        ax.set_ylabel("相对涨跌（%）")
        style_axes(ax)
        wide_lo, wide_hi = pred[(0, -1)][0], pred[(0, -1)][1]
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
    axes[-1].set_xlabel("验证集交易日。色带是开盘前的预测分布，虚线是这组当天真实相对变动")
    fig.suptitle("预测分布 vs 真实", fontsize=16, color="#1c2430")
    fig.text(
        0.5,
        0.004,
        f"相对变动 = 相对载荷 × 外部冲击。验证集上这条投影的 80% 盖住 {levels['80'][0]:.0%}。不是收益。",
        ha="center",
        fontsize=9,
        color="#6b7380",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT_DIR / "cs_proj_val.png", dpi=160)
    plt.close(fig)

    # The function itself: realized relative move against exposure, in units of that day's predicted scale.
    x = exposure[val] * 100
    yy = relative[val] * 100
    m = np.isfinite(x) & np.isfinite(yy)
    fig2, ax = plt.subplots(figsize=(11.2, 6.2))
    # One wedge from the latest shape is misleading because shape is stable.
    q = np.nanmedian(shape[val], axis=0)
    grid = np.linspace(np.nanquantile(x[m], 0.01), np.nanquantile(x[m], 0.99), 200)
    colors = ("#d9e4f0", "#b7cce4", "#7ea4cb")
    labels_q = ("预测 95%", "预测 80%", "预测 50%")
    for (a, b), color, label in zip(((0, -1), (1, -2), (2, -3)), colors, labels_q):
        lo_g = np.minimum(grid * q[a], grid * q[b])
        hi_g = np.maximum(grid * q[a], grid * q[b])
        ax.fill_between(grid, lo_g, hi_g, color=color, lw=0, label=label, zorder=1)
    ax.plot(grid, grid * 0.5 * (q[2] + q[3]), color="#16324f", lw=1.3, label="预测中位", zorder=3)
    shown = m & (yy >= np.nanquantile(yy[m], 0.01)) & (yy <= np.nanquantile(yy[m], 0.99))
    ax.scatter(x[shown], yy[shown], s=6, c="#1f4e79", alpha=0.15, lw=0, zorder=2)
    style_axes(ax)
    ax.set_xlabel("开盘前的预测宽度（%）")
    ax.set_ylabel("当天真实的投影相对变动（%）")
    ax.set_title("预测分布 vs 真实", loc="left", color="#1c2430")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    fig2.text(0.5, 0.01, "验证集。点是每只股票当天的真实相对变动。色带是外部冲击的分布乘上这只股票的相对载荷。不是收益。", ha="center", fontsize=9, color="#6b7380")
    fig2.tight_layout(rect=(0, 0.04, 1, 1))
    fig2.savefig(OUT_DIR / "cs_proj_fan.png", dpi=160)
    plt.close(fig2)

    payload = {
        "alpha_fast": alpha,
        "scale_epochs": epochs,
        "gmm_k": k,
        "bic": bic_table,
        "sign_aligned_corr": float(corr),
        "validation_coverage_projected": {name: {"hit_rate": hit, "n": n} for name, (hit, n) in levels.items()},
        "validation_coverage_full_relative": {name: {"hit_rate": hit, "n": n} for name, (hit, n) in total_levels.items()},
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
