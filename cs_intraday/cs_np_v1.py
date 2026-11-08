"""
Distribution-free checks of the intraday pullback book, pool by pool.

Every pool on disk trends up, so a t-stat on a mean leans on normal tails and
on the market path we happened to get. These checks use ranks, signs and
resampling instead:

  shuffle     inside every sector, at every decision bar, the score is handed
              to a random member. Same sectors, same number of names, same
              costs, same book. The p-value is the share of shuffles whose mean
              daily net is at least the real one.
  bootstrap   stationary bootstrap of the daily net (mean block 10 days), a
              95% interval for the mean and for the annualized ratio
  ranks       Wilcoxon signed-rank and a sign test on the daily net
  market      Spearman and Kendall against the pool's daily long-hold return,
              Kruskal-Wallis across the five quintiles of that return,
              Mann-Whitney between days after a rising and after a falling
              20-day pool trend, and the Theil-Sen slope on the pool return

Validation days only for the shuffle; the other checks are read on train and
on validation. Test bars are not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP, fit_mix, row_rank
from cs_cost_v1 import FEE_PER_SHARE, TICK, TOP_FRAC, build, day_books, target_book
from cs_cost_v3 import THRESHOLD, decision_bars, delayed_days
from cs_curve_v1 import DELAY, pool_daily
from cs_f7state_v1 import demean
from cs_pools_v1 import pools

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_np_v1_out"
N_SHUFFLE = 300
N_BOOT = 5000
MEAN_BLOCK = 10
TREND_DAYS = 20
SEED = 11


def stationary_bootstrap(x, n_boot, mean_block, rng):
    """Index matrix n_boot x n: runs of consecutive days with random length."""
    n = len(x)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    jump = rng.random((n_boot, n)) < 1.0 / mean_block
    fresh = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return x[idx]


def ratio(x, axis=-1):
    return np.mean(x, axis=axis) / np.std(x, axis=axis) * np.sqrt(252.0)


def rank_checks(book, hold, trend, rng):
    b = np.asarray(book)
    h = np.asarray(hold)
    boot = stationary_bootstrap(b, N_BOOT, MEAN_BLOCK, rng)
    quint = pd.qcut(h, 5, labels=False)
    groups = [b[quint == q] for q in range(5)]
    up = trend > 0
    ts = st.theilslopes(b, h)
    return {
        "days": int(len(b)),
        "mean_bp": float(b.mean() * 1e4),
        "boot_mean_bp_95": [float(q) for q in np.percentile(boot.mean(axis=1) * 1e4, [2.5, 97.5])],
        "ratio": float(ratio(b)),
        "boot_ratio_95": [float(q) for q in np.percentile(ratio(boot), [2.5, 97.5])],
        "boot_share_mean_le_0": float((boot.mean(axis=1) <= 0).mean()),
        "wilcoxon_p": float(st.wilcoxon(b[b != 0]).pvalue),
        "sign_p": float(st.binomtest(int((b > 0).sum()), int((b != 0).sum()), 0.5, alternative="greater").pvalue),
        "up_share": float((b > 0).mean()),
        "spearman": [float(v) for v in st.spearmanr(b, h)],
        "kendall": [float(v) for v in st.kendalltau(b, h)],
        "kruskal_p": float(st.kruskal(*groups).pvalue),
        "quintile_median_bp": [float(np.median(g) * 1e4) for g in groups],
        "trend_up_median_bp": float(np.median(b[up]) * 1e4),
        "trend_down_median_bp": float(np.median(b[~up]) * 1e4),
        "mannwhitney_p": float(st.mannwhitneyu(b[up], b[~up]).pvalue),
        "theil_sen_slope": float(ts.slope),
        "theil_sen_95": [float(ts.low_slope), float(ts.high_slope)],
    }


def shuffle_within(mix, codes, sectors, rng):
    out = mix.copy()
    rows = np.arange(len(mix))[:, None]
    for c in sectors:
        cols = np.flatnonzero(codes == c)
        x = mix[:, cols]
        finite = np.isfinite(x)
        keys = np.where(finite, rng.random(x.shape), np.inf)
        source = np.argsort(keys, axis=1)
        target = np.argsort(~finite, axis=1, kind="stable")
        block = x.copy()
        block[rows, target] = x[rows, source]
        out[:, cols] = block
    return out


def pool_book(meta):
    p = build(meta)
    steps, bar_ret = decision_bars(meta)
    train = p["day_label"] == "train"
    codes = p["codes"]
    big = [c for c in range(len(p["sector_order"])) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    c_in = (TICK + FEE_PER_SHARE) / p["price"]
    c_out = FEE_PER_SHARE / p["price"]
    cols = [row_rank(-p["gap"]), row_rank(p["f4"]), row_rank(-p["f3"])]
    ztarget = np.clip(p["resid"] / p["width"], -Z_CLIP, Z_CLIP)
    coef = fit_mix(cols, ztarget, train[:, None] & np.ones(ztarget.shape, dtype=bool))
    mix = sum(c * b for c, b in zip(cols, coef))
    expected = mix * p["width"]
    raw_rel = demean(p["raw_close"])
    ok = train[:, None] & np.isfinite(expected) & np.isfinite(raw_rel)
    scale = float((expected[ok] * raw_rel[ok]).sum() / (expected[ok] ** 2).sum())

    def run(rows, score):
        exp_ = scale * score * p["width"][rows]
        w = target_book(score, codes, big, TOP_FRAC, np.abs(exp_) > THRESHOLD * (c_in[rows] + c_out[rows]))
        held = day_books(w, p["day_id"][rows], "sleeves")
        d = delayed_days(held, p["raw_seg"][rows], bar_ret, steps[rows], DELAY, c_in[rows], c_out[rows],
                         p["day_id"][rows], p["last_of_day"][rows])
        return d

    return p, codes, big, mix, run


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    report = {}
    for title, members in pools(list(meta["names"])).items():
        sub = dict(meta, names=members)
        p, codes, big, mix, run = pool_book(sub)
        dates = p["day_dates"]
        labels = p["day_label_all"]
        hold_all = pd.Series(pool_daily(sub, dates), index=dates)
        trend = hold_all.rolling(TREND_DAYS).sum().shift(1)

        all_rows = np.arange(len(mix))
        d = run(all_rows, mix)
        net = (d["g"] - d["c"])
        active = net.index[(d["g"] != 0).to_numpy()].min()
        entry = {}
        for split in ("train", "val"):
            sel = (labels[net.index] == split) & (net.index >= active) & np.isfinite(trend.iloc[net.index].to_numpy())
            idx = net.index[sel]
            entry[split] = rank_checks(net.loc[idx].to_numpy(), hold_all.iloc[idx].to_numpy(), trend.iloc[idx].to_numpy(), rng)

        val_rows = np.flatnonzero(p["day_label"] == "val")
        real = float(run(val_rows, mix[val_rows]).pipe(lambda x: (x["g"] - x["c"]).mean()))
        null = np.array([
            float(run(val_rows, shuffle_within(mix[val_rows], codes, big, rng)).pipe(lambda x: (x["g"] - x["c"]).mean()))
            for _ in range(N_SHUFFLE)
        ])
        entry["shuffle"] = {
            "real_mean_bp": real * 1e4,
            "null_mean_bp": float(null.mean() * 1e4),
            "null_95_bp": [float(q) for q in np.percentile(null * 1e4, [2.5, 97.5])],
            "p": float((1 + (null >= real).sum()) / (1 + len(null))),
            "null": (null * 1e4).tolist(),
        }
        report[title] = entry
        v, t = entry["val"], entry["train"]
        print(
            f"[{title:14s}] val mean {v['mean_bp']:+.2f}bp boot95 {np.round(v['boot_mean_bp_95'], 2)} "
            f"ratio {v['ratio']:.2f} boot95 {np.round(v['boot_ratio_95'], 2)} | wilcoxon p {v['wilcoxon_p']:.3f} "
            f"sign p {v['sign_p']:.3f} | shuffle p {entry['shuffle']['p']:.3f} (null mean {entry['shuffle']['null_mean_bp']:+.2f}) "
            f"[{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )
        print(
            f"     val vs market: spearman {v['spearman'][0]:+.2f} (p {v['spearman'][1]:.3f}) kruskal p {v['kruskal_p']:.3f} "
            f"quintile medians {np.round(v['quintile_median_bp'], 2)} | trend up/down medians "
            f"{v['trend_up_median_bp']:+.2f}/{v['trend_down_median_bp']:+.2f} MW p {v['mannwhitney_p']:.3f} | "
            f"theil-sen {v['theil_sen_slope']:+.4f}",
            flush=True,
        )
        print(
            f"     train: mean {t['mean_bp']:+.2f} boot95 {np.round(t['boot_mean_bp_95'], 2)} wilcoxon p {t['wilcoxon_p']:.4f} "
            f"sign p {t['sign_p']:.4f} spearman {t['spearman'][0]:+.2f} kruskal p {t['kruskal_p']:.3f} "
            f"MW p {t['mannwhitney_p']:.3f}",
            flush=True,
        )

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.4))
    for ax, (title, e) in zip(axes.flat, report.items()):
        null = np.array(e["shuffle"]["null"])
        ax.hist(null, bins=30, color="#c9ced6", edgecolor="white")
        ax.axvline(e["shuffle"]["real_mean_bp"], color="#1f4e79", lw=2.2)
        ax.axvline(0, color="#9aa3ad", lw=0.8, ls="--")
        v = e["val"]
        ax.set_title(title, loc="left", fontsize=11, color="#1c2430")
        ax.text(
            0.02, 0.97,
            f"real signal: {e['shuffle']['real_mean_bp']:+.2f} bp/day\n"
            f"shuffle p = {e['shuffle']['p']:.3f}\n"
            f"bootstrap 95%: [{v['boot_mean_bp_95'][0]:+.1f}, {v['boot_mean_bp_95'][1]:+.1f}]\n"
            f"Wilcoxon p = {v['wilcoxon_p']:.3f}\n"
            f"vs market: Spearman {v['spearman'][0]:+.2f}, Kruskal p {v['kruskal_p']:.2f}",
            transform=ax.transAxes, va="top", fontsize=8.2, color="#1c2430",
        )
        ax.set_xlabel("Mean daily net, validation (bp)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax = axes.flat[-1]
    names = list(report)
    x = np.arange(len(names))
    for offset, key, color, lab in ((-0.2, "trend_down_median_bp", "#d06a3a", "after 20-day market fall"),
                                    (0.2, "trend_up_median_bp", "#1f4e79", "after 20-day market rise")):
        ax.bar(x + offset, [report[n]["val"][key] for n in names], width=0.38, color=color, label=lab)
    ax.axhline(0, color="#9aa3ad", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("Median daily net, validation (bp)")
    ax.set_title("Does it depend on the market trend?", loc="left", fontsize=11, color="#1c2430")
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.suptitle("Intraday strategy: real signal (blue line) vs the same book with the signal shuffled inside each sector (grey)",
                 fontsize=12.5, color="#1c2430")
    fig.text(0.5, 0.01, f"{N_SHUFFLE} shuffles per pool. Same sectors, costs and trading rules. Validation days only. Test period not opened.",
             ha="center", fontsize=8.5, color="#6b7380")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "cs_np_intraday.png", dpi=150)
    plt.close(fig)
    for e in report.values():
        e["shuffle"].pop("null")
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
