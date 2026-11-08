"""
Within-pool predictive distribution: the existing residual model, plus a GMM.

The mixture is fit on train days only. Each name is described by where it sat
in yesterday's cross-section and by its own leftover scale. The external
scale is not a clustering feature, so the groups are not just calm days
versus violent days. K is chosen by BIC on the train sample.

The forecast is a mixture of the train clouds of standardized leftovers.
Validation is scored after the mixture is frozen. Test bars are not loaded.
No profit is computed.
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
from cs_resid_v1 import daily_last_close, walk_daily_residual
from cs_xdist_v1 import cross_section_position, style_axes
from factor_v1 import MIN_DAY_BARS, MIN_NAMES_FRAC
from nested_v1 import FAST_WIN, SLOW_WIN, causal_window, hold_weekly

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_gmm_v1_out"
SHAPE_Q = np.array([0.025, 0.10, 0.25, 0.75, 0.90, 0.975])
K_CANDIDATES = (2, 3, 4)


def flat_valid(features: np.ndarray, valid: np.ndarray, days: np.ndarray):
    block = features[days][valid[days]]
    return block


def choose_gmm(x_train: np.ndarray):
    best = None
    table = []
    for k in K_CANDIDATES:
        if len(x_train) < k * 50:
            continue
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=0)
        model.fit(x_train)
        labels = model.predict(x_train)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < 0.05 * len(x_train):
            table.append({"k": k, "bic": None, "kept": False})
            continue
        bic = float(model.bic(x_train))
        table.append({"k": k, "bic": bic, "kept": True, "train_share": [float(c / len(labels)) for c in counts]})
        if best is None or bic < best[0]:
            best = (bic, model, counts)
    if best is None:
        raise RuntimeError("no GMM component sizes were usable on the train sample")
    return best[1], table


def empirical_cdf(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ordered = np.sort(samples[np.isfinite(samples)])
    if len(ordered) < 30:
        return np.full(len(grid), np.nan)
    return np.searchsorted(ordered, grid, side="right") / len(ordered)


def mixture_quantiles(responsibilities: np.ndarray, cdfs: np.ndarray, grid: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """responsibilities: N x K. cdfs: K x G. Returns N x len(levels) grid values."""
    mix = responsibilities @ cdfs
    out = np.full((len(responsibilities), len(levels)), np.nan)
    for j, level in enumerate(levels):
        crossed = mix >= level
        idx = np.argmax(crossed, axis=1)
        ok = crossed.any(axis=1)
        out[ok, j] = grid[idx[ok]]
    return out


def hit_rate(y, lo, hi, mask) -> tuple[float, int]:
    ok = mask & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if ok.sum() == 0:
        return float("nan"), 0
    inside = (y[ok] >= lo[ok]) & (y[ok] <= hi[ok])
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
    index, eps, _energy = walk_daily_residual(
        close.pct_change(fill_method=None).to_numpy(np.float64),
        close.index,
        daily_ret.iloc[1:],
    )
    labels = label_days(index, train_end, val_start)
    min_names = max(MIN_DAY_BARS, int(MIN_NAMES_FRAC * eps.shape[1]))
    fast_var = causal_window(eps**2, FAST_WIN)
    slow_var = hold_weekly(
        causal_window(eps**2, SLOW_WIN),
        (index.isocalendar().year * 100 + index.isocalendar().week).to_numpy(),
    )
    position = np.vstack([cross_section_position(eps[t - 1], min_names) for t in range(1, len(index))])
    features = np.full((len(index) - 1, eps.shape[1], 3), np.nan)
    sigma_own = np.sqrt(np.maximum(fast_var[1:], 0.0))
    features[..., 0] = position
    features[..., 1] = np.log(fast_var[1:] + 1e-18)
    features[..., 2] = np.log(slow_var[1:] + 1e-18)
    target = eps[1:]
    day_index = index[1:]
    day_labels = labels[1:]
    valid = np.isfinite(features).all(axis=2) & np.isfinite(sigma_own) & (sigma_own > 0) & np.isfinite(target)
    enough = valid.sum(axis=1) >= min_names
    train_days = np.flatnonzero((day_labels == "train") & enough)
    val_days = np.flatnonzero((day_labels == "val") & enough)
    x_train = flat_valid(features, valid, train_days)
    scaler = StandardScaler().fit(x_train)
    gmm, bic_table = choose_gmm(scaler.transform(x_train))
    k = int(gmm.n_components)
    print(f"[gmm] K {k} bic {json.dumps(bic_table)}", flush=True)

    z = target / sigma_own
    train_mask = np.zeros(len(day_index), dtype=bool)
    train_mask[train_days] = True
    train_rows = train_mask[:, None] & valid
    x_all = np.where(valid[..., None], features, 0.0).reshape(-1, 3)
    scaled_all = scaler.transform(x_all).reshape(features.shape)
    hard = np.full(valid.shape, -1, dtype=int)
    proba = np.zeros(valid.shape + (k,))
    train_x = scaled_all[train_rows]
    hard[train_rows] = gmm.predict(train_x)
    proba[train_rows] = gmm.predict_proba(train_x)
    val_mask = np.zeros(len(day_index), dtype=bool)
    val_mask[val_days] = True
    val_rows = val_mask[:, None] & valid
    val_x = scaled_all[val_rows]
    hard[val_rows] = gmm.predict(val_x)
    proba[val_rows] = gmm.predict_proba(val_x)

    z_train = z[train_rows]
    hard_train = hard[train_rows]
    shapes = []
    centers = []
    for j in range(k):
        sample = z_train[hard_train == j]
        shapes.append(np.quantile(sample, SHAPE_Q))
        centers.append(float(np.mean(sample)))
    shapes = np.vstack(shapes)
    grid = np.linspace(np.quantile(z_train, 0.001), np.quantile(z_train, 0.999), 401)
    cdfs = np.vstack([empirical_cdf(z_train[hard_train == j], grid) for j in range(k)])
    single_shape = np.quantile(z_train, SHAPE_Q)

    val_proba = proba[val_rows]
    mix_z = mixture_quantiles(val_proba, cdfs, grid, SHAPE_Q)
    mix_band = sigma_own[val_rows][:, None] * mix_z
    single_band = sigma_own[val_rows][:, None] * single_shape
    y_val = target[val_rows]
    mix_hits = {}
    single_hits = {}
    level_ix = {"50": (2, 3), "80": (1, 4), "95": (0, 5)}
    for name, (a, b) in level_ix.items():
        mix_hits[name] = hit_rate(y_val, mix_band[:, a], mix_band[:, b], np.ones(len(y_val), dtype=bool))
        single_hits[name] = hit_rate(y_val, single_band[:, a], single_band[:, b], np.ones(len(y_val), dtype=bool))
        print(f"[cover {name}] mixture {mix_hits[name][0]:.3f} single {single_hits[name][0]:.3f}", flush=True)

    component_stats = []
    for j in range(k):
        m = val_rows & (hard == j)
        realized = target[m]
        component_stats.append(
            {
                "train_share": float(np.mean(hard_train == j)),
                "val_n": int(m.sum()),
                "val_mean": float(np.mean(realized)) if m.sum() else float("nan"),
                "shape": [float(v) for v in shapes[j]],
                "train_mean_z": centers[j],
            }
        )
        print(f"[component {j}] val_n {m.sum()} mean {component_stats[-1]['val_mean']:.5f}", flush=True)

    payload = {
        "k": k,
        "bic": bic_table,
        "validation_coverage": {
            "mixture": {name: {"hit_rate": hit, "n": n} for name, (hit, n) in mix_hits.items()},
            "single_cloud": {name: {"hit_rate": hit, "n": n} for name, (hit, n) in single_hits.items()},
        },
        "components": component_stats,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    shades = (
        (0, 5, "#e4edf6", "预测 95%"),
        (1, 4, "#b9cfe6", "预测 80%"),
        (2, 3, "#7ea4cb", "预测 50%"),
    )
    fig, axes = plt.subplots(k, 1, figsize=(11.6, 2.55 * k), sharex=True)
    if k == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        member = val_rows & (hard == j)
        lohi = {pair: (np.full(len(val_days), np.nan), np.full(len(val_days), np.nan)) for pair in ((0, 5), (1, 4), (2, 3))}
        med = np.full(len(val_days), np.nan)
        real_lo = np.full(len(val_days), np.nan)
        real_hi = np.full(len(val_days), np.nan)
        for i, t in enumerate(val_days):
            m = member[t]
            if m.sum() < 5:
                continue
            med[i] = np.median(target[t, m])
            real_lo[i] = np.quantile(target[t, m], 0.10)
            real_hi[i] = np.quantile(target[t, m], 0.90)
            for a, b in lohi:
                lohi[(a, b)][0][i] = np.median(sigma_own[t, m] * shapes[j, a])
                lohi[(a, b)][1][i] = np.median(sigma_own[t, m] * shapes[j, b])
        when = day_index[val_days]
        for a, b, color, label in shades:
            lo, hi = lohi[(a, b)]
            ax.fill_between(when, lo * 100, hi * 100, color=color, lw=0, label=label, zorder=1)
        ax.plot(when, real_lo * 100, color="#d06a3a", lw=0.9, ls="--", label="真实的 10% / 90%", zorder=3)
        ax.plot(when, real_hi * 100, color="#d06a3a", lw=0.9, ls="--", zorder=3)
        ax.plot(when, med * 100, color="#16324f", lw=1.2, label="真实中位", zorder=4)
        share = component_stats[j]["val_n"] / max(int(val_rows.sum()), 1)
        ax.set_title(f"第 {j + 1} 组，验证集里约占 {share:.0%}", loc="left", color="#1c2430")
        ax.set_ylabel("剩下的涨跌（%）")
        style_axes(ax)
        if j == 0:
            ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("验证集交易日。色带是这一组在训练集上的预测分布，虚线是这组当天真实落点")
    mix80 = mix_hits["80"][0]
    one80 = single_hits["80"][0]
    fig.suptitle("预测分布 vs 真实", fontsize=16, color="#1c2430")
    fig.text(
        0.5,
        0.005,
        f"模型加 GMM。验证集上混合分布的 80% 盖住 {mix80:.0%}，不分的那一朵云盖住 {one80:.0%}。不是收益。",
        ha="center",
        fontsize=9,
        color="#6b7380",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT_DIR / "cs_gmm_val.png", dpi=160)
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
