"""
Does the same book beat a random assignment of its own scores, on other stocks too?

The rule is unchanged: six coordinates, a 500-day monthly refit, real price at
least $60, and the 4.2x cost gate. That gate was chosen earlier on the non-bull
validation, so the non-bull panel is not a fresh test. The other two pools
never entered that choice. Test bars are not loaded.

The null keeps every score and only reassigns it to another name in the same
sector, at that bar. Each random book is drawn as a line. Blue is the middle
of those lines. Green is the outer edge. A block of the same colors marks the
percentile ranges.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from cs_ac_v1 import NON_BULL_SECTORS, build_ac_coordinates, run_threshold_book
from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP
from cs_curve_v1 import stats
from cs_f7state_v1 import demean
from cs_hiprice_v1 import COSTS, PICK_COST, PICK_FROM, combined_panel, real_price_factor, walk_fit
from cs_np_v1 import shuffle_within
from cs_resid_v1 import newey_west

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_band_v1_out"
WINDOW = 500
THRESH = 4.2
N_SHUF = 80
SEED = 67
MIN_PX = 60.0
KEYS = ["u_day_c", "u_day_p", "u_2h_c", "u_1h_c", "u_30m_c", "u_vol"]
CMAP = LinearSegmentedColormap.from_list(
    "bg", ["#1b4f8a", "#2f86b5", "#3aaa8a", "#b6e36a"]
)

POOLS = [
    ("Non-bull", NON_BULL_SECTORS),
    (
        "Financials, energy, industrials",
        {"Financials", "Energy", "Industrials"},
    ),
    (
        "Tech, communication, discretionary",
        {"Information Technology", "Communication Services", "Consumer Discretionary"},
    ),
]


def p_ge(null_values, real):
    null_values = np.asarray(null_values, dtype=np.float64)
    return float((1 + np.sum(null_values >= real)) / (1 + len(null_values)))


def one_pool(meta_all, fetched, sec_map, sectors, rng, t0):
    names = [n for n in meta_all["names"] if sec_map.get(n) in sectors]
    meta = dict(meta_all, names=names, n_names=len(names))
    from cs_cost_v1 import build
    from cs_cost_v3 import decision_bars

    panel = build(meta)
    steps, bar_ret = decision_bars(meta)
    coords = build_ac_coordinates(meta, panel)
    dates = panel["day_dates"]
    day_id = panel["day_id"]
    labels = panel["day_label_all"]
    n_days = len(dates)
    codes = panel["codes"]
    big = [c for c in range(len(panel["sector_order"])) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    spread, fee = COSTS[PICK_COST]
    factor = real_price_factor(names, fetched, dates)
    real = panel["price"] / factor[day_id]
    eligible = np.isfinite(real) & (real >= MIN_PX)
    c_in = (spread + fee) / real
    c_out = fee / real
    ztarget = np.clip(panel["resid"] / panel["width"], -Z_CLIP, Z_CLIP)
    raw_rel = demean(panel["raw_close"])
    rows = np.flatnonzero(dates[day_id] >= PICK_FROM)
    pick = (labels == "train") & (dates >= PICK_FROM)
    val = labels == "val"
    show = (labels != "drop") & (dates >= PICK_FROM)
    mix, expected = walk_fit(
        [coords[k] for k in KEYS],
        ztarget, eligible, raw_rel, panel["width"], day_id, dates, WINDOW,
    )
    common = dict(
        raw_seg=panel["raw_seg"][rows],
        bar_ret=bar_ret,
        steps=steps[rows],
        day_id=day_id[rows],
        last=panel["last_of_day"][rows],
        codes=codes,
        big=big,
        n_days=n_days,
    )
    net, gross, held = run_threshold_book(
        mix[rows], expected[rows], eligible[rows], c_in[rows], c_out[rows],
        common["raw_seg"], common["bar_ret"], common["steps"], common["day_id"], common["last"],
        codes, big, n_days, THRESH,
    )
    ratio = np.where(np.isfinite(mix[rows]) & (np.abs(mix[rows]) > 1e-12), expected[rows] / mix[rows], np.nan)
    base_m = np.where(eligible[rows], mix[rows], np.nan)
    null = np.empty((N_SHUF, n_days), dtype=np.float64)
    for i in range(N_SHUF):
        shuf_m = shuffle_within(base_m, codes, big, rng)
        shuf_e = ratio * shuf_m
        null[i], _, _ = run_threshold_book(
            shuf_m, shuf_e, eligible[rows], c_in[rows], c_out[rows],
            common["raw_seg"], common["bar_ret"], common["steps"], common["day_id"], common["last"],
            codes, big, n_days, THRESH,
        )
        if (i + 1) % 20 == 0:
            print(f"    shuffle {i + 1}/{N_SHUF}  [{time.perf_counter() - t0:.0f}s]", flush=True)
    out = {
        "names": len(names),
        "sectors": panel["sector_order"],
        "eligible": float(eligible[rows].sum(axis=1).mean()),
        "active": float((np.abs(held).sum(axis=1) > 1e-8).mean()),
        "net": net,
        "gross": gross,
        "null": null,
        "dates": dates,
        "mask_val": val,
        "mask_show": show,
    }
    for name, mask in (("train", pick), ("val", val), ("all", show)):
        real_bp = float(net[mask].mean() * 1e4)
        null_bp = null[:, mask].mean(axis=1) * 1e4
        _n, mu, tstat = newey_west(net[mask], 5)
        st = stats(net[mask])
        out[name] = {
            "per_year": st["per_year"],
            "worst_drop": st["worst_drop"],
            "ratio": st["ratio"],
            "net_bp": real_bp,
            "gross_bp": float(gross[mask].mean() * 1e4),
            "null_bp_mean": float(null_bp.mean()),
            "p": p_ge(null_bp, real_bp),
            "t": float(tstat),
            "null_per_year": np.array([stats(path[mask])["per_year"] for path in null]),
        }
    return out


def paint_cloud(ax, pack, title):
    d = pack["dates"][pack["mask_show"]]
    curves = (np.cumprod(1.0 + pack["null"][:, pack["mask_show"]], axis=1) - 1.0) * 100.0
    final = curves[:, -1]
    order = np.argsort(final)
    rank = np.empty(len(final))
    rank[order] = np.arange(len(final)) / max(len(final) - 1, 1)
    tail = np.abs(rank - 0.5) * 2.0
    levels = [0, 5, 15, 30, 50]
    lo = [np.percentile(curves, q, axis=0) for q in levels]
    hi = [np.percentile(curves, 100 - q, axis=0) for q in levels]
    for i in range(len(levels) - 1):
        frac = 1.0 - i / (len(levels) - 2)
        color = CMAP(frac)
        ax.fill_between(d, lo[i], lo[i + 1], color=color, alpha=0.55, lw=0, zorder=1)
        ax.fill_between(d, hi[i + 1], hi[i], color=color, alpha=0.55, lw=0, zorder=1)
    for i in np.argsort(tail):
        ax.plot(d, curves[i], color=CMAP(tail[i]), lw=0.55, alpha=0.22, zorder=2)
    real = (np.cumprod(1.0 + pack["net"][pack["mask_show"]]) - 1.0) * 100.0
    st = pack["all"]
    ax.plot(
        d, real, color="#161616", lw=2.2, zorder=4,
        label=f"Strategy: {st['per_year']*100:+.1f}%/yr, max DD {st['worst_drop']*100:.1f}%",
    )
    v0 = pack["dates"][pack["mask_val"]][0]
    ax.axvline(v0, color="#6d7682", lw=0.8, ls="--", zorder=3)
    ax.axhline(0, color="#9aa3ad", lw=0.6, zorder=3)
    ax.set_title(title, loc="left", fontsize=11, color="#1c2430")
    ax.set_ylabel("Cumulative after-cost return (%)")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)
    ax.text(
        0.99, 0.04,
        f"train p={pack['train']['p']:.3f}   val p={pack['val']['p']:.3f}\n"
        f"train t={pack['train']['t']:+.1f}   val t={pack['val']['t']:+.1f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#1c2430",
    )


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    meta_all, fetched = combined_panel(json.loads(SPLIT_PATH.read_text(encoding="utf-8")))
    u_all = pd.read_csv(ROOT / "universe_all.csv")
    u_all["key"] = u_all["symbol"].str.replace(".", "_", regex=False)
    sec_map = dict(zip(u_all["key"], u_all["sector"]))
    packs = []
    for title, sectors in POOLS:
        print(f"[pool] {title}", flush=True)
        pack = one_pool(meta_all, fetched, sec_map, sectors, rng, t0)
        packs.append((title, pack))
        print(
            f"  names {pack['names']}  eligible/bar {pack['eligible']:.1f}  active {pack['active']:.2f}  "
            f"train {pack['train']['per_year']*100:+.2f}%/yr p={pack['train']['p']:.3f} t={pack['train']['t']:+.2f}  "
            f"val {pack['val']['per_year']*100:+.2f}%/yr p={pack['val']['p']:.3f} t={pack['val']['t']:+.2f}  "
            f"[{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.6))
    paint_cloud(axes[0, 0], packs[0][1], "A. Non-bull sectors. Gate was chosen on this validation")
    paint_cloud(axes[0, 1], packs[1][1], "B. Other stocks: financials, energy, industrials")
    paint_cloud(axes[1, 0], packs[2][1], "C. Other stocks: tech, communication, discretionary")

    ax = axes[1, 1]
    for i, (title, pack) in enumerate(packs):
        yrs = pack["all"]["null_per_year"] * 100.0
        order = np.argsort(yrs)
        rank = np.arange(len(yrs)) / max(len(yrs) - 1, 1)
        tail = np.abs(rank - 0.5) * 2.0
        ax.scatter(
            yrs[order], np.full(len(yrs), i),
            c=CMAP(tail), s=16, linewidths=0, zorder=2,
        )
        ax.scatter(
            [pack["all"]["per_year"] * 100.0], [i],
            c="#161616", s=54, marker="D", zorder=3,
        )
    ax.axvline(0, color="#1c2430", lw=0.7)
    ax.set_yticks(range(len(packs)))
    ax.set_yticklabels([p[0] for p in packs], fontsize=8)
    ax.set_xlabel("Annualized after-cost return, 2021 through validation (%)")
    ax.set_title("D. Diamond = strategy. Dots = random lines, blue middle to green edge", loc="left", fontsize=11, color="#1c2430")
    ax.grid(axis="x", color="#e6e9ee", lw=0.6)
    ax.set_ylim(-0.6, len(packs) - 0.2)

    for axis in axes.ravel():
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.text(
        0.5, 0.012,
        "Same rule on every panel: 500-day refit, real price >= $60, 4.2x cost gate. "
        f"{N_SHUF} reassignments inside the sector. Blue is the dense middle, green is the outer 5%. "
        "Dashed line starts the validation. Test period not opened.",
        ha="center", fontsize=8.0, color="#6b7380",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_img = OUT_DIR / "cs_band.png"
    fig.savefig(out_img, dpi=150)
    extra = ROOT / "cs_force_v4_out"
    extra.mkdir(parents=True, exist_ok=True)
    fig.savefig(extra / "cs_band.png", dpi=150)
    plt.close(fig)

    report = {}
    for title, pack in packs:
        report[title] = {
            "names": pack["names"],
            "mean_eligible": pack["eligible"],
            "active_rows": pack["active"],
            "train": {k: (float(v) if np.ndim(v) == 0 else None) for k, v in pack["train"].items() if k != "null_per_year"},
            "val": {k: (float(v) if np.ndim(v) == 0 else None) for k, v in pack["val"].items() if k != "null_per_year"},
            "all": {k: (float(v) if np.ndim(v) == 0 else None) for k, v in pack["all"].items() if k != "null_per_year"},
            "null_all_per_year_p05": float(np.percentile(pack["all"]["null_per_year"], 5)),
            "null_all_per_year_p50": float(np.percentile(pack["all"]["null_per_year"], 50)),
            "null_all_per_year_p95": float(np.percentile(pack["all"]["null_per_year"], 95)),
        }
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[wrote] {out_img}", flush=True)
    print(f"[done] {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
