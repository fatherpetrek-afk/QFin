"""
Checks that the high-price result is not the market's upward drift.

The forecast is the one already chosen in cs_hiprice_v1: two-year refit, real
price at least $100, middle cost, fixed threshold, positions stacked to the close.
Nothing here is used to change that rule. Test bars are not loaded.

Blank: the same weights, but every name is given that bar's cross-sectional
average return, so only a move common to all names remains.
Random: at each decision the scores are reassigned inside each sector.
Mixture: each name is kept with probability alpha and blanked otherwise.
Rank tests compare the daily result with the equal-weight path of the same names.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP, row_rank
from cs_cost_v1 import TOP_FRAC, build, day_books, target_book
from cs_cost_v3 import THRESHOLD, decision_bars, delayed_days
from cs_curve_v1 import DELAY, pool_daily, stats
from cs_f7state_v1 import demean
from cs_hiprice_v1 import COSTS, MIN_PRICE, PICK_COST, PICK_FROM, combined_panel, real_price_factor, walk_fit
from cs_np_v1 import rank_checks, shuffle_within

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_null_v1_out"
WINDOW = 500
N_RANDOM = 200
N_MIX = 24
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 41


def rescale(base_mix, base_exp, new_mix):
    ratio = np.where(np.isfinite(base_mix) & (base_mix != 0), base_exp / base_mix, np.nan)
    return ratio * new_mix


def daily_net(held, seg, bar_ret, steps, c_in, c_out, day_id, last, n_days):
    out = delayed_days(
        held, seg, bar_ret, steps, DELAY, np.nan_to_num(c_in), np.nan_to_num(c_out), day_id, last
    )
    net = np.zeros(n_days)
    gross = np.zeros(n_days)
    net[out.index.to_numpy()] = (out["g"] - out["c"]).to_numpy()
    gross[out.index.to_numpy()] = out["g"].to_numpy()
    return net, gross


def sleeve_net(mix, expected, eligible, c_in, c_out, raw_seg, bar_ret, steps, day_id, last, codes, big, n_days):
    score = np.where(eligible, mix, np.nan)
    keep = np.isfinite(score) & np.isfinite(expected) & (np.abs(expected) > THRESHOLD * (c_in + c_out))
    w = target_book(score, codes, big, TOP_FRAC, keep)
    held = day_books(w, day_id, "sleeves")
    net, gross = daily_net(held, raw_seg, bar_ret, steps, c_in, c_out, day_id, last, n_days)
    return net, gross, held


def per_year(net, mask):
    return stats(net[mask])["per_year"]


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    meta, fetched = combined_panel(json.loads(SPLIT_PATH.read_text(encoding="utf-8")))
    panel = build(meta)
    steps_all, bar_ret = decision_bars(meta)
    dates = panel["day_dates"]
    day_id_all = panel["day_id"]
    labels = panel["day_label_all"]
    n_days = len(dates)
    spread, fee = COSTS[PICK_COST]
    factor = real_price_factor(meta["names"], fetched, dates)
    real = panel["price"] / factor[day_id_all]
    eligible_all = np.isfinite(real) & (real >= MIN_PRICE)
    c_in_all = (spread + fee) / real
    c_out_all = fee / real
    codes = panel["codes"]
    big = [c for c in range(len(panel["sector_order"])) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    print(f"[panel] {meta['n_names']} names  {time.perf_counter() - t0:.0f}s", flush=True)

    cols = [row_rank(-panel["gap"]), row_rank(-panel["f3"])]
    ztarget = np.clip(panel["resid"] / panel["width"], -Z_CLIP, Z_CLIP)
    mix_all, exp_all = walk_fit(
        cols, ztarget, eligible_all, demean(panel["raw_close"]), panel["width"], day_id_all, dates, WINDOW
    )
    rows = np.flatnonzero(dates[day_id_all] >= PICK_FROM)
    mix, expected = mix_all[rows], exp_all[rows]
    eligible = eligible_all[rows]
    c_in, c_out = c_in_all[rows], c_out_all[rows]
    day_id = day_id_all[rows]
    steps = steps_all[rows]
    raw_seg = panel["raw_seg"][rows]
    last = panel["last_of_day"][rows]
    print(f"[fit] {time.perf_counter() - t0:.0f}s  decisions {len(rows)}", flush=True)

    real_net, real_gross, held = sleeve_net(
        mix, expected, eligible, c_in, c_out, raw_seg, bar_ret, steps, day_id, last, codes, big, n_days
    )
    common_level = np.nanmean(raw_seg, axis=1)
    common_seg = np.repeat(common_level[:, None], raw_seg.shape[1], axis=1)
    common_seg[~np.isfinite(raw_seg)] = np.nan
    bar_common = np.repeat(np.nanmean(bar_ret, axis=1)[:, None], bar_ret.shape[1], axis=1)
    common_net, common_gross = daily_net(
        held, common_seg, bar_common, steps, c_in, c_out, day_id, last, n_days
    )
    weight_sum = float(np.nanmean(np.abs(held.sum(axis=1))))
    gross_exposure = float(np.abs(held).sum(axis=1).mean())
    print(f"[weights] mean |sum| {weight_sum:.4f}  mean gross {gross_exposure:.3f}", flush=True)

    blank = np.full(mix.shape, np.nan)
    blank_net, _, _ = sleeve_net(
        blank, blank, eligible, c_in, c_out, raw_seg, bar_ret, steps, day_id, last, codes, big, n_days
    )

    pick = (labels == "train") & (dates >= PICK_FROM)
    val = labels == "val"
    hold = pool_daily(meta, dates)
    print(
        f"[real] train {per_year(real_net, pick) * 100:+.1f}%/yr ({real_net[pick].mean() * 1e4:+.2f}bp)  "
        f"val {per_year(real_net, val) * 100:+.1f}%/yr ({real_net[val].mean() * 1e4:+.2f}bp)",
        flush=True,
    )
    print(
        f"[common only] train {common_net[pick].mean() * 1e4:+.2f}bp/d  val {common_net[val].mean() * 1e4:+.2f}bp/d  "
        f"residual part train {(real_net - common_net)[pick].mean() * 1e4:+.2f}bp/d  "
        f"val {(real_net - common_net)[val].mean() * 1e4:+.2f}bp/d",
        flush=True,
    )
    print(f"[blank signal] train {blank_net[pick].mean() * 1e4:+.2f}bp  val {blank_net[val].mean() * 1e4:+.2f}bp", flush=True)

    null = np.zeros((N_RANDOM, n_days))
    for i in range(N_RANDOM):
        shuf = shuffle_within(np.where(eligible, mix, np.nan), codes, big, rng)
        net_i, _, _ = sleeve_net(
            shuf, rescale(mix, expected, shuf), eligible, c_in, c_out, raw_seg, bar_ret, steps, day_id, last, codes, big, n_days
        )
        null[i] = net_i
        if (i + 1) % 50 == 0:
            print(f"[random] {i + 1}/{N_RANDOM}  {time.perf_counter() - t0:.0f}s", flush=True)
    random_p = {}
    for split, mask in (("train", pick), ("val", val)):
        real_mean = float(real_net[mask].mean())
        null_mean = null[:, mask].mean(axis=1)
        random_p[split] = float((1 + (null_mean >= real_mean).sum()) / (1 + N_RANDOM))
        print(
            f"[random {split}] p {random_p[split]:.3f}  null mean {null_mean.mean() * 1e4:+.2f}bp  "
            f"real {real_mean * 1e4:+.2f}bp",
            flush=True,
        )

    mix_year = {a: {"train": [], "val": []} for a in ALPHAS}
    for alpha in ALPHAS:
        reps = 1 if alpha in (0.0, 1.0) else N_MIX
        for _ in range(reps):
            if alpha == 0.0:
                net_a = blank_net
            elif alpha == 1.0:
                net_a = real_net
            else:
                keep_name = rng.random(mix.shape) < alpha
                mixed = np.where(keep_name, mix, np.nan)
                net_a, _, _ = sleeve_net(
                    mixed, rescale(mix, expected, mixed), eligible, c_in, c_out,
                    raw_seg, bar_ret, steps, day_id, last, codes, big, n_days,
                )
            mix_year[alpha]["train"].append(per_year(net_a, pick))
            mix_year[alpha]["val"].append(per_year(net_a, val))
        print(
            f"[mix {alpha:.2f}] train {np.mean(mix_year[alpha]['train']) * 100:+.1f}%/yr  "
            f"val {np.mean(mix_year[alpha]['val']) * 100:+.1f}%/yr  {time.perf_counter() - t0:.0f}s",
            flush=True,
        )
    dose_p = {}
    for split in ("train", "val"):
        xs = np.repeat(np.array(ALPHAS), [len(mix_year[a][split]) for a in ALPHAS])
        ys = np.concatenate([mix_year[a][split] for a in ALPHAS])
        rho, p = st.spearmanr(xs, ys)
        dose_p[split] = {"rho": float(rho), "p": float(p)}
        print(f"[dose spearman {split}] rho {rho:+.2f}  p {p:.4g}", flush=True)

    trend = pd.Series(hold, index=dates).rolling(20).sum().shift(1).to_numpy()
    checks = {}
    for split, mask in (("train", pick), ("val", val)):
        ok = mask & np.isfinite(trend)
        checks[split] = rank_checks(real_net[ok], hold[ok], trend[ok], rng)
        ts = st.theilslopes(real_net[ok], hold[ok])
        checks[split]["theil_intercept"] = float(ts.intercept)
        v = checks[split]
        print(
            f"[np {split}] mean {v['mean_bp']:+.2f}bp  boot95 {np.round(v['boot_mean_bp_95'], 2)}  "
            f"wilcoxon p {v['wilcoxon_p']:.4f}  sign p {v['sign_p']:.4f}  "
            f"spearman vs pool {v['spearman'][0]:+.2f} (p {v['spearman'][1]:.3f})  "
            f"kruskal p {v['kruskal_p']:.3f}  mannwhitney p {v['mannwhitney_p']:.3f}  "
            f"theil slope {v['theil_sen_slope']:+.2f} [{v['theil_sen_95'][0]:+.2f}, {v['theil_sen_95'][1]:+.2f}]  "
            f"intercept {v['theil_intercept'] * 1e4:+.2f}bp",
            flush=True,
        )
        print(f"         quintile medians bp {np.round(v['quintile_median_bp'], 2)}", flush=True)

    show = (labels != "drop") & (dates >= PICK_FROM)
    d = dates[show]
    real_curve = np.cumprod(1.0 + real_net[show]) - 1.0
    common_curve = np.cumprod(1.0 + common_net[show]) - 1.0
    hold_curve = np.cumprod(1.0 + hold[show]) - 1.0
    null_curve = np.cumprod(1.0 + null[:, show], axis=1) - 1.0
    band_lo, band_md, band_hi = np.quantile(null_curve, [0.05, 0.50, 0.95], axis=0)

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2))

    ax = axes[0, 0]
    ax.axvspan(dates[val][0], d[-1], color="#eef1f5", zorder=0)
    ax.fill_between(d, band_lo * 100, band_hi * 100, color="#d06a3a", alpha=0.18, linewidth=0, label="Random scores, 5–95%")
    ax.plot(d, band_md * 100, color="#d06a3a", lw=1.1, label="Random scores, pointwise median")
    ax.plot(d, common_curve * 100, color="#4f9a6a", lw=1.3, label=f"Common return only: {per_year(common_net, show) * 100:+.1f}%/yr")
    ax.plot(d, hold_curve * 100, color="#9aa3ad", lw=1.1, label=f"Equal-weight names: {per_year(hold, show) * 100:+.1f}%/yr")
    ax.plot(d, real_curve * 100, color="#1f4e79", lw=1.8, label=f"Real signal: {per_year(real_net, show) * 100:+.1f}%/yr")
    ax.axhline(0, color="#9aa3ad", lw=0.6)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Since 2021  |  shaded: validation", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    ax = axes[0, 1]
    bins = np.linspace(-30, 30, 31)
    ax.hist(np.array([per_year(null[i], pick) for i in range(N_RANDOM)]) * 100, bins=bins, color="#b7c3d0", alpha=0.9, label="Train, random")
    ax.hist(np.array([per_year(null[i], val) for i in range(N_RANDOM)]) * 100, bins=bins, color="#f0c9b0", alpha=0.75, label="Validation, random")
    ax.axvline(per_year(real_net, pick) * 100, color="#1f4e79", lw=1.6, label=f"Train, real ({per_year(real_net, pick) * 100:+.1f}%)")
    ax.axvline(per_year(real_net, val) * 100, color="#d06a3a", lw=1.6, label=f"Validation, real ({per_year(real_net, val) * 100:+.1f}%)")
    ax.set_xlabel("Return per year (%)")
    ax.set_ylabel("Random redraws")
    ax.set_title(f"Scores shuffled inside sector  |  train p={random_p['train']:.3f}, val p={random_p['val']:.3f}", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1, 0]
    for split, color in (("train", "#1f4e79"), ("val", "#d06a3a")):
        center, lo, hi = [], [], []
        for alpha in ALPHAS:
            sample = np.array(mix_year[alpha][split]) * 100
            center.append(float(np.mean(sample)))
            lo.append(float(np.quantile(sample, 0.1)))
            hi.append(float(np.quantile(sample, 0.9)))
        center = np.array(center)
        ax.fill_between(ALPHAS, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(ALPHAS, center, color=color, marker="o", lw=1.6, label="Train" if split == "train" else "Validation")
    ax.axhline(0, color="#9aa3ad", lw=0.6)
    ax.set_xlabel("Fraction of names whose scores are kept")
    ax.set_ylabel("Return per year (%)")
    ax.set_title(
        f"Blank mixed with real scores  |  Spearman train {dose_p['train']['rho']:+.2f}, val {dose_p['val']['rho']:+.2f}",
        loc="left", fontsize=11, color="#1c2430",
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    x = np.arange(5)
    width = 0.36
    ax.bar(x - width / 2, checks["train"]["quintile_median_bp"], width=width, color="#1f4e79", label="Train")
    ax.bar(x + width / 2, checks["val"]["quintile_median_bp"], width=width, color="#d06a3a", label="Validation")
    ax.axhline(0, color="#9aa3ad", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["worst", "2", "3", "4", "best"])
    ax.set_xlabel("Same-day equal-weight return, five groups")
    ax.set_ylabel("Median strategy return (bp)")
    ax.set_title(
        f"Does it only pay when the names rise?  |  Kruskal train p={checks['train']['kruskal_p']:.2f}, val p={checks['val']['kruskal_p']:.2f}",
        loc="left", fontsize=11, color="#1c2430",
    )
    ax.legend(frameon=False, fontsize=8)

    for a in axes.ravel():
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="y", color="#e6e9ee", lw=0.6)
    fig.suptitle("Checks against a shared upward drift", fontsize=14, color="#1c2430")
    fig.text(
        0.5, 0.012,
        "Same forecast as the high-price result. Blanking removes a name's score. Random redraws keep each sector's scores and reassign them.\n"
        "Common return gives every name that bar's cross-sectional average. Middle cost. Test period not opened.",
        ha="center", fontsize=8, color="#6b7380",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "cs_null.png", dpi=150)
    plt.close(fig)

    report = {
        "weight_abs_sum": weight_sum,
        "gross_exposure": gross_exposure,
        "random_p": random_p,
        "dose_spearman": dose_p,
        "mix_mean_per_year": {str(a): {s: float(np.mean(v)) for s, v in splits.items()} for a, splits in mix_year.items()},
        "checks": checks,
        "real_per_year": {"train": per_year(real_net, pick), "val": per_year(real_net, val), "since_2021": per_year(real_net, show)},
        "common_mean_bp": {"train": float(common_net[pick].mean() * 1e4), "val": float(common_net[val].mean() * 1e4)},
        "blank_mean_bp": {"train": float(blank_net[pick].mean() * 1e4), "val": float(blank_net[val].mean() * 1e4)},
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
