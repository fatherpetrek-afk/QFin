"""
The pullback book on high-priced names only, refit on a short rolling window.

Pool: the S&P 100 panel and the flat-sector panel together, today's members,
survivors accepted. One factor model on all of them.

Only names whose real price is at least $100 at the decision bar are traded.
The bars are forward-adjusted, so the real price is the adjusted price divided
by the Futu factors (get_rehab, data/rehab) of every split and dividend after
that day and before the fetch date. A cent of spread is a cent of real price,
so costs use the real price too.

Scores: the gap to the sector and the volume direction. The close position is
left out; most of it was the bid-ask bounce.

The mix weights and the expected-move scale are refit every month on the last
W trading days only: one year, two years, or all earlier days. W is picked on
train months from 2021 on, at the middle cost. The book, the cost rule and
the one-bar delay are those of cs_curve_v1.

Costs per side, real price:
  higher   3-cent spread (1.5 cents on entry), 0.5 cent/share fee
  middle   2-cent spread (1 cent on entry), 0.35 cent/share fee
  lower    1-cent spread (0.5 cent on entry), 0.2 cent/share fee
Exit is the closing auction, fee only. Test bars are not loaded.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import combined_v1
import cs_peer_v1
from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP, row_rank
from cs_cost_v1 import TOP_FRAC, build, day_books, target_book
from cs_cost_v3 import THRESHOLD, decision_bars, delayed_days
from cs_curve_v1 import DELAY, pool_daily, stats
from cs_f7state_v1 import demean
from cs_np_v1 import rank_checks, shuffle_within

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
ALL_DIR = ROOT / "data" / "5m_all"
ALL_UNIVERSE = ROOT / "universe_all.csv"
REHAB_DIR = ROOT / "data" / "rehab"
OUT_DIR = ROOT / "cs_hiprice_v1_out"
PANELS = (
    (ROOT / "data" / "5m", ROOT / "universe_sp100.csv", ROOT / "data" / "manifest.csv"),
    (ROOT / "data" / "5m_flat", ROOT / "universe_flat.csv", ROOT / "data" / "5m_flat" / "manifest.csv"),
)
MIN_PRICE = 100.0
MIN_TRAIN_COVERAGE = 0.9
WINDOWS = {"1 year": 250, "2 years": 500, "all past": None}
PICK_FROM = pd.Timestamp("2021-01-01")
COSTS = {
    "higher": (0.015, 0.005),
    "middle": (0.010, 0.0035),
    "lower": (0.005, 0.002),
}
PICK_COST = "middle"
N_SHUFFLE = 300
SEED = 23


def combined_panel(meta):
    """Hard links of both panels in one folder, one sector table, fetch dates."""
    ALL_DIR.mkdir(parents=True, exist_ok=True)
    tables, fetched = [], {}
    for data_dir, universe, manifest in PANELS:
        table = pd.read_csv(universe)
        table["key"] = table["symbol"].str.replace(".", "_", regex=False)
        man = pd.read_csv(manifest)
        stamp = pd.to_datetime(man["fetched_at"]).max() if "fetched_at" in man else pd.Timestamp(
            os.path.getmtime(manifest), unit="s")
        for key in table["key"]:
            src = data_dir / f"{key}.parquet"
            dst = ALL_DIR / f"{key}.parquet"
            if src.exists():
                if not dst.exists():
                    os.link(src, dst)
                fetched[key] = stamp.normalize()
        tables.append(table)
    table = pd.concat(tables, ignore_index=True).drop_duplicates("key")
    table[["symbol", "name", "sector"]].to_csv(ALL_UNIVERSE, index=False)
    combined_v1.DATA_DIR = ALL_DIR
    cs_peer_v1.UNIVERSE_PATH = ALL_UNIVERSE

    train_end, first = pd.Timestamp(meta["train_end"]), pd.Timestamp(meta["first_active"])
    counts = {}
    for key in fetched:
        t = pd.to_datetime(pd.read_parquet(ALL_DIR / f"{key}.parquet", columns=["time"])["time"])
        counts[key] = int(((t >= first) & (t <= train_end)).sum())
    best = max(counts.values())
    names = sorted(k for k, c in counts.items() if c >= MIN_TRAIN_COVERAGE * best)
    return dict(meta, names=names, n_names=len(names)), fetched


def real_price_factor(names, fetched, days):
    """Days x names: adjusted price / real price, from the events after each day."""
    out = np.ones((len(days), len(names)))
    day_values = days.to_numpy()
    for j, key in enumerate(names):
        path = REHAB_DIR / f"{key}.csv"
        if not path.exists():
            continue
        ev = pd.read_csv(path)
        ev["ex"] = pd.to_datetime(ev["ex_div_date"])
        ev = ev[(ev["ex"] <= fetched[key]) & ev["forward_adj_factorA"].notna()]
        for ex, a in zip(ev["ex"], ev["forward_adj_factorA"]):
            out[day_values < ex.to_datetime64(), j] *= float(a)
    return out


def walk_fit(cols, ztarget, eligible, raw_rel, width, day_id, day_dates, window):
    """Monthly refit on the last `window` days. Returns the mix and expected move per row."""
    months = pd.DatetimeIndex(day_dates[day_id]).to_period("M")
    mix = np.full(ztarget.shape, np.nan)
    expected = np.full(ztarget.shape, np.nan)
    x_all = np.stack(cols, axis=-1)
    for m in months.unique():
        rows = np.flatnonzero(months == m)
        first_day = day_id[rows[0]]
        lo = 0 if window is None else max(0, first_day - window)
        past = np.flatnonzero((day_id >= lo) & (day_id < first_day))
        if len(np.unique(day_id[past])) < 120:
            continue
        x = x_all[past][eligible[past]]
        y = ztarget[past][eligible[past]]
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        coef = np.linalg.lstsq(np.column_stack([x[ok], np.ones(int(ok.sum()))]), y[ok], rcond=None)[0][:-1]
        past_mix = x_all[past] @ coef
        ex = (past_mix * width[past])[eligible[past]]
        rr = raw_rel[past][eligible[past]]
        good = np.isfinite(ex) & np.isfinite(rr)
        scale = float((ex[good] * rr[good]).sum() / (ex[good] ** 2).sum())
        mix[rows] = x_all[rows] @ coef
        expected[rows] = scale * mix[rows] * width[rows]
    return mix, expected


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    meta, fetched = combined_panel(json.loads(SPLIT_PATH.read_text(encoding="utf-8")))
    print(f"[names] {meta['n_names']}", flush=True)
    p = build(meta)
    steps, bar_ret = decision_bars(meta)
    dates = p["day_dates"]
    day_id = p["day_id"]
    labels_row = p["day_label"]
    labels = p["day_label_all"]
    codes = p["codes"]
    print(f"[panel] {time.perf_counter() - t0:.0f}s", flush=True)

    factor = real_price_factor(meta["names"], fetched, dates)
    real = p["price"] / factor[day_id]
    eligible = np.isfinite(real) & (real >= MIN_PRICE)
    nvda = meta["names"].index("NVDA") if "NVDA" in meta["names"] else None
    if nvda is not None:
        print(f"[check] NVDA adjusted {p['price'][0, nvda]:.2f} -> real {real[0, nvda]:.2f} on {dates[day_id[0]].date()}", flush=True)
    sector_names = [p["sector_order"][c] for c in codes]
    elig_share = {s: float(eligible[:, [i for i, x in enumerate(sector_names) if x == s]].mean()) for s in p["sector_order"]}
    print(f"[eligible] names >= ${MIN_PRICE:.0f} per row: {eligible.sum(axis=1).mean():.1f} | by sector "
          f"{ {k: round(v, 2) for k, v in elig_share.items()} }", flush=True)

    big = [c for c in range(len(p["sector_order"])) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    cols = [row_rank(-p["gap"]), row_rank(-p["f3"])]
    ztarget = np.clip(p["resid"] / p["width"], -Z_CLIP, Z_CLIP)
    raw_rel = demean(p["raw_close"])
    hold = pd.Series(pool_daily(meta, dates), index=dates)
    trend = hold.rolling(20).sum().shift(1)

    def run_book(mix, expected, cost, rows=None):
        rows = np.arange(len(mix)) if rows is None else rows
        spread, fee = COSTS[cost]
        c_in = (spread + fee) / real[rows]
        c_out = fee / real[rows]
        score = np.where(eligible[rows], mix[rows], np.nan)
        keep = np.abs(expected[rows]) > THRESHOLD * (c_in + c_out)
        w = target_book(score, codes, big, TOP_FRAC, keep)
        held = day_books(w, day_id[rows], "sleeves")
        d = delayed_days(held, p["raw_seg"][rows], bar_ret, steps[rows], DELAY, np.nan_to_num(c_in), np.nan_to_num(c_out),
                         day_id[rows], p["last_of_day"][rows])
        net = pd.Series(0.0, index=dates)
        net.iloc[d.index.to_numpy()] = (d["g"] - d["c"]).to_numpy()
        gross = pd.Series(0.0, index=dates)
        gross.iloc[d.index.to_numpy()] = d["g"].to_numpy()
        return net, gross, float(np.abs(w).astype(bool).sum(axis=1)[np.abs(w).sum(axis=1) > 0].mean())

    fits = {}
    results = {}
    pick_sel = (labels == "train") & (dates >= PICK_FROM)
    val_sel = labels == "val"
    for wname, window in WINDOWS.items():
        mix, expected = walk_fit(cols, ztarget, eligible, raw_rel, p["width"], day_id, dates, window)
        fits[wname] = (mix, expected)
        results[wname] = {}
        for cost in COSTS:
            net, gross, names_held = run_book(mix, expected, cost)
            entry = {"names_held_per_row": names_held}
            for split, sel in (("train 2021+", pick_sel), ("val", val_sel)):
                s = stats(net[sel].to_numpy())
                s["gross_bp"] = float(gross[sel].mean() * 1e4)
                s["net_bp"] = float(net[sel].mean() * 1e4)
                entry[split] = s
            results[wname][cost] = {"stats": entry, "net": net}
            print(
                f"[{wname:8s} {cost:6s}] train 2021+ net {entry['train 2021+']['net_bp']:+.2f}bp/d "
                f"({entry['train 2021+']['per_year'] * 100:+.1f}%/yr, ratio {entry['train 2021+']['ratio']:.2f}) gross "
                f"{entry['train 2021+']['gross_bp']:+.2f} | val net {entry['val']['net_bp']:+.2f}bp/d "
                f"({entry['val']['per_year'] * 100:+.1f}%/yr, ratio {entry['val']['ratio']:.2f}, DD {entry['val']['worst_drop'] * 100:.1f}%) "
                f"gross {entry['val']['gross_bp']:+.2f} | names held {names_held:.1f}  [{time.perf_counter() - t0:.0f}s]",
                flush=True,
            )

    pick = max(WINDOWS, key=lambda w: results[w][PICK_COST]["stats"]["train 2021+"]["net_bp"])
    print(f"[picked on train 2021+ at {PICK_COST} cost] window = {pick}", flush=True)
    mix, expected = fits[pick]
    tests = {}
    val_rows = np.flatnonzero(labels_row == "val")
    for cost in COSTS:
        net = results[pick][cost]["net"]
        active = (labels != "drop") & np.isfinite(trend.to_numpy()) & (dates >= PICK_FROM)
        checks = {}
        for split in ("train", "val"):
            sel = active & (labels == split)
            checks[split] = rank_checks(net[sel].to_numpy(), hold[sel].to_numpy(), trend[sel].to_numpy(), rng)
        real_mean = float(run_book(mix, expected, cost, val_rows)[0][val_sel].mean())
        null = []
        for _ in range(N_SHUFFLE if cost == PICK_COST else 0):
            shuffled = mix.copy()
            shuffled[val_rows] = shuffle_within(np.where(eligible[val_rows], mix[val_rows], np.nan), codes, big, rng)
            exp_s = expected.copy()
            exp_s[val_rows] = np.where(np.isfinite(mix[val_rows]) & (mix[val_rows] != 0),
                                       expected[val_rows] / mix[val_rows], np.nan) * shuffled[val_rows]
            null.append(float(run_book(shuffled, exp_s, cost, val_rows)[0][val_sel].mean()))
        tests[cost] = {"checks": checks, "val_mean_bp": real_mean * 1e4}
        if null:
            null = np.array(null)
            tests[cost]["shuffle_p"] = float((1 + (null >= real_mean).sum()) / (1 + len(null)))
            tests[cost]["shuffle_null_mean_bp"] = float(null.mean() * 1e4)
        v = checks["val"]
        print(f"[tests {cost:6s}] val boot95 {np.round(v['boot_mean_bp_95'], 2)}bp wilcoxon p {v['wilcoxon_p']:.3f} "
              f"sign p {v['sign_p']:.3f} spearman vs market {v['spearman'][0]:+.2f} | train 2021+ boot95 "
              f"{np.round(checks['train']['boot_mean_bp_95'], 2)} wilcoxon p {checks['train']['wilcoxon_p']:.4f}"
              + (f" | shuffle p {tests[cost]['shuffle_p']:.3f} (null {tests[cost]['shuffle_null_mean_bp']:+.2f}bp)" if "shuffle_p" in tests[cost] else ""),
              flush=True)

    years = {}
    for cost in COSTS:
        net = results[pick][cost]["net"]
        keep = (labels != "drop") & (dates >= PICK_FROM)
        frame = pd.DataFrame({"r": net[keep].to_numpy(), "y": pd.DatetimeIndex(dates[keep]).year,
                              "lab": labels[keep]})
        years[cost] = {f"{y}{' val' if lab == 'val' else ''}": float(np.prod(1 + g['r']) - 1)
                       for (y, lab), g in frame.groupby(["y", "lab"])}
    print(f"[years, window {pick}] " + " | ".join(f"{c}: " + ", ".join(f"{k} {v * 100:+.1f}%" for k, v in ys.items())
                                                for c, ys in years.items()), flush=True)

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    keep = (labels != "drop") & (dates >= PICK_FROM)
    d = dates[keep]
    ax.axvspan(dates[val_sel][0], d[-1], color="#eef1f5", zorder=0)
    h = hold[keep].to_numpy()
    ax.plot(d, (np.cumprod(1 + h) - 1) * 100, color="#9aa3ad", lw=1.2,
            label=f"Buy & hold pool: {stats(h)['per_year'] * 100:+.1f}%/yr, max DD {stats(h)['worst_drop'] * 100:.0f}%")
    for cost, color in zip(COSTS, ("#d06a3a", "#1f4e79", "#4f9a6a")):
        r = results[pick][cost]["net"][keep].to_numpy()
        s = stats(r)
        ax.plot(d, (np.cumprod(1 + r) - 1) * 100, color=color, lw=1.8,
                label=f"Strategy, {cost} cost: {s['per_year'] * 100:+.1f}%/yr, max DD {s['worst_drop'] * 100:.0f}%")
    ax.axhline(0, color="#9aa3ad", lw=0.7)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(f"Since 2021, weights refit monthly on the last {pick}  |  shaded: validation", loc="left", fontsize=10.5,
                 color="#1c2430")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    keys = list(years[PICK_COST])
    x = np.arange(len(keys))
    for i, (cost, color) in enumerate(zip(COSTS, ("#d06a3a", "#1f4e79", "#4f9a6a"))):
        ax.bar(x + (i - 1) * 0.26, [years[cost][k] * 100 for k in keys], width=0.26, color=color, label=f"{cost} cost")
    ax.axhline(0, color="#9aa3ad", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel("Strategy return in the year (%)")
    ax.set_title("Year by year", loc="left", fontsize=10.5, color="#1c2430")
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="y", color="#e6e9ee", lw=0.6)
    sp = tests[PICK_COST].get("shuffle_p", float("nan"))
    fig.suptitle(f"High-priced names only (real price >= $100): buy & hold vs strategy   |   validation shuffle p = {sp:.3f}",
                 fontsize=13.5, color="#1c2430")
    fig.text(0.5, 0.01,
             f"{meta['n_names']} names: S&P 100 plus S&P 500 flat-sector names, today's members. Long and short equal amounts inside each "
             "sector, flat at the close, no leverage, filled 5 min late.\nRound trip at $100: higher 2.5bp, middle 1.7bp, lower 0.9bp. "
             "Test period not opened.", ha="center", fontsize=8, color="#6b7380")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "cs_hiprice.png", dpi=150)
    plt.close(fig)
    out = {"names": meta["names"], "picked_window": pick, "eligible_share_by_sector": elig_share,
           "results": {w: {c: r["stats"] for c, r in rc.items()} for w, rc in results.items()},
           "tests": tests, "years": years}
    (OUT_DIR / "metrics.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
