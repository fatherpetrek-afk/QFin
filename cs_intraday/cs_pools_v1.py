"""
The same pullback book on other pools carved from the data on disk.

The S&P 100 pool rose almost every year, so the long-hold line flatters any
comparison. Each pool here still meets the assumptions the book rests on:
several sectors with at least five names, so a name has same-sector peers,
and at least 30 names in all, so the three common factors are estimated on a
real cross-section.

  full      all 97 names
  no tech   without Information Technology and Communication Services
  growth    Information Technology, Communication Services, Consumer Discretionary
  half A/B  every sector split at random into two halves, fixed seed

Every pool is rebuilt from scratch: its own three factors, its own residuals,
its own mix weights, expected-move scale and cost rule, all fit on its train
days. The book, the costs and the one-bar entry delay are those of
cs_curve_v1. Test bars are not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP, fit_mix, row_rank
from cs_cost_v1 import FEE_PER_SHARE, TICK, TOP_FRAC, build, day_books, target_book
from cs_cost_v3 import THRESHOLD, decision_bars, delayed_days
from cs_curve_v1 import DELAY, pool_daily, stats
from cs_f7state_v1 import demean
from cs_peer_v1 import sector_codes

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_pools_v1_out"
SEED = 7
GROWTH = ("Information Technology", "Communication Services", "Consumer Discretionary")
TECH = ("Information Technology", "Communication Services")


def pools(names):
    codes, order = sector_codes(names)
    sector = np.array([order[c] for c in codes])
    rng = np.random.default_rng(SEED)
    half_a = []
    for s in order:
        members = [n for n, x in zip(names, sector) if x == s]
        rng.shuffle(members)
        half_a += members[: (len(members) + 1) // 2]
    return {
        "Full S&P 100": list(names),
        "No tech": [n for n, s in zip(names, sector) if s not in TECH],
        "Tech & growth": [n for n, s in zip(names, sector) if s in GROWTH],
        "Random half A": [n for n in names if n in set(half_a)],
        "Random half B": [n for n in names if n not in set(half_a)],
    }


def run_pool(meta):
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
    expected = expected * float((expected[ok] * raw_rel[ok]).sum() / (expected[ok] ** 2).sum())
    w = target_book(mix, codes, big, TOP_FRAC, np.abs(expected) > THRESHOLD * (c_in + c_out))
    held = day_books(w, p["day_id"], "sleeves")
    d = delayed_days(held, p["raw_seg"], bar_ret, steps, DELAY, c_in, c_out, p["day_id"], p["last_of_day"])

    dates = p["day_dates"]
    labels = p["day_label_all"]
    book = pd.Series(0.0, index=dates)
    book.iloc[d.index.to_numpy()] = (d["g"] - d["c"]).to_numpy()
    hold = pd.Series(pool_daily(meta, dates), index=dates)
    first_trade = dates[d.index[(d["g"] != 0).to_numpy()].min()]
    keep = (labels != "drop") & (dates >= first_trade)
    info = {"coef": [float(c) for c in coef], "sectors_traded": [p["sector_order"][c] for c in big]}
    return book[keep], hold[keep], labels[keep], info


def draw(ax, book, hold, labels, title, s):
    val = labels == "val"
    ax.axvspan(book.index[val][0], book.index[-1], color="#eef1f5", zorder=0)
    ax.plot(hold.index, (np.cumprod(1 + hold.to_numpy()) - 1) * 100, color="#9aa3ad", lw=1.2,
            label=f"Buy & hold pool: {s['all']['pool']['per_year'] * 100:+.1f}%/yr, max DD {s['all']['pool']['worst_drop'] * 100:.0f}%")
    ax.plot(book.index, (np.cumprod(1 + book.to_numpy()) - 1) * 100, color="#1f4e79", lw=1.8,
            label=f"Strategy after costs: {s['all']['book']['per_year'] * 100:+.1f}%/yr, max DD {s['all']['book']['worst_drop'] * 100:.0f}%")
    ax.axhline(0, color="#9aa3ad", lw=0.7)
    ax.set_title(
        f"{title}  |  validation: hold {s['val']['pool']['total'] * 100:+.1f}%, strategy {s['val']['book']['total'] * 100:+.1f}%",
        loc="left", fontsize=10.5, color="#1c2430",
    )
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)


def draw_summary(ax, results):
    names = list(results)
    x = np.arange(len(names))
    for offset, split, color, lab in ((-0.2, "train", "#b7cce4", "Strategy, train years"), (0.2, "val", "#1f4e79", "Strategy, validation")):
        ax.bar(x + offset, [results[n]["stats"][split]["book"]["per_year"] * 100 for n in names], width=0.38, color=color, label=lab)
    ax.scatter(x + 0.2, [results[n]["stats"]["val"]["pool"]["per_year"] * 100 for n in names], color="#d06a3a", zorder=3,
               label="Buy & hold, validation")
    ax.axhline(0, color="#9aa3ad", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8.5)
    ax.set_ylabel("Annualized return (%)")
    ax.set_title("Annualized return by pool", loc="left", fontsize=10.5, color="#1c2430")
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    results = {}
    for title, members in pools(list(meta["names"])).items():
        sub = dict(meta, names=members)
        book, hold, labels, info = run_pool(sub)
        val = labels == "val"
        s = {}
        for split, sel in (("all", np.ones(len(book), dtype=bool)), ("train", ~val), ("val", val)):
            s[split] = {"book": stats(book[sel].to_numpy()), "pool": stats(hold[sel].to_numpy()),
                        "corr": float(np.corrcoef(book[sel], hold[sel])[0, 1])}
        results[title] = {"n_names": len(members), "info": info, "stats": s, "series": (book, hold, labels)}
        print(
            f"[{title:14s}] n {len(members):3d} | val book {s['val']['book']['per_year'] * 100:+.1f}%/yr "
            f"DD {s['val']['book']['worst_drop'] * 100:.1f}% ratio {s['val']['book']['ratio']:.2f} | "
            f"val hold {s['val']['pool']['per_year'] * 100:+.1f}%/yr | train book {s['train']['book']['per_year'] * 100:+.1f}%/yr "
            f"ratio {s['train']['book']['ratio']:.2f} | train hold {s['train']['pool']['per_year'] * 100:+.1f}%/yr | "
            f"corr val {s['val']['corr']:.2f}  [{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 12.4))
    for ax, (title, r) in zip(axes.flat, results.items()):
        draw(ax, *r["series"], f"{title} ({r['n_names']} names)", r["stats"])
    draw_summary(axes.flat[-1], results)
    fig.suptitle("Buy & hold vs strategy on different stock pools", fontsize=15, color="#1c2430")
    fig.text(
        0.5, 0.008,
        "Shaded: validation period. Everything before it is train, where all parameters were fit, so that part flatters the strategy. "
        "Strategy: long and short equal amounts inside each sector, flat at the close, no leverage, filled 5 min late, "
        "2-cent spread and 0.35 cent/share fee. Test period not opened.",
        ha="center", fontsize=8.2, color="#6b7380", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "cs_pools.png", dpi=150)
    plt.close(fig)
    for r in results.values():
        r.pop("series")
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
