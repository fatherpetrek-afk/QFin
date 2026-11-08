"""Where the book breaks, pool by pool and sector by sector. Test bars are not loaded."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cs_ac_v1 import build_ac_coordinates
from cs_band_v1 import KEYS, MIN_PX, POOLS, THRESH, WINDOW
from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP, rank_corr
from cs_cost_v1 import TOP_FRAC, build
from cs_cost_v3 import decision_bars
from cs_f7state_v1 import demean
from cs_hiprice_v1 import COSTS, PICK_COST, PICK_FROM, combined_panel, real_price_factor, walk_fit

SPLIT = Path(__file__).resolve().parent / "data" / "splits.json"


def mean_ic(score, target, mask, rows, min_n):
    vals = []
    for i in rows:
        m = mask[i]
        vals.append(rank_corr(np.where(m, score[i], np.nan), np.where(m, target[i], np.nan), min_n))
    v = np.array([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if len(v) < 5:
        return float("nan"), 0
    return float(v.mean()), int(len(v))


def sector_abs_corr(x, cols):
    z = x[:, cols]
    keep = np.isfinite(z).mean(axis=0) > 0.8
    z = z[:, keep]
    if z.shape[1] < 4:
        return float("nan")
    z = np.where(np.isfinite(z), z, 0.0)
    z = z - z.mean(axis=0)
    sd = z.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = z / sd
    c = (z.T @ z) / max(len(z) - 1, 1)
    iu = np.triu_indices(c.shape[0], 1)
    return float(np.mean(np.abs(c[iu])))


def main() -> None:
    meta_all, fetched = combined_panel(json.loads(SPLIT.read_text(encoding="utf-8")))
    u = pd.read_csv(Path(__file__).resolve().parent / "universe_all.csv")
    u["key"] = u["symbol"].str.replace(".", "_", regex=False)
    sec_map = dict(zip(u["key"], u["sector"]))
    spread, fee = COSTS[PICK_COST]
    rt_cash = spread + 2.0 * fee

    for title, sectors in POOLS:
        names = [n for n in meta_all["names"] if sec_map.get(n) in sectors]
        meta = dict(meta_all, names=names, n_names=len(names))
        panel = build(meta)
        decision_bars(meta)
        coords = build_ac_coordinates(meta, panel)
        dates = panel["day_dates"]
        day_id = panel["day_id"]
        codes = panel["codes"]
        labels = panel["day_label_all"]
        factor = real_price_factor(names, fetched, dates)
        real = panel["price"] / factor[day_id]
        eligible = np.isfinite(real) & (real >= MIN_PX)
        rt = rt_cash / real
        rows = np.flatnonzero((dates[day_id] >= PICK_FROM) & (labels[day_id] != "drop"))
        ztarget = np.clip(panel["resid"] / panel["width"], -Z_CLIP, Z_CLIP)
        raw_rel = demean(panel["raw_close"])
        mix, expected = walk_fit(
            [coords[k] for k in KEYS], ztarget, eligible, raw_rel, panel["width"], day_id, dates, WINDOW,
        )
        finite = np.isfinite(mix)
        gate = eligible & finite & np.isfinite(expected) & (np.abs(expected) > THRESH * rt)
        ratio = np.abs(expected) / rt

        ic_close, n_close = mean_ic(mix, panel["resid"], eligible, rows, 20)
        ic_hold, n_hold = mean_ic(mix, panel["raw_seg"], eligible, rows, 20)
        ic_gate, n_gate = mean_ic(mix, panel["raw_seg"], gate, rows, 8)
        med_ratio = float(np.nanmedian(ratio[rows][eligible[rows]]))
        p90_ratio = float(np.nanpercentile(ratio[rows][eligible[rows]], 90))

        print(
            f"\n== {title}  names {len(names)} =="
            f"\n  IC score vs residual-to-close, eligible: {ic_close:+.4f}  ({n_close} bars)"
            f"\n  IC score vs next-hour raw return, eligible: {ic_hold:+.4f}  ({n_hold} bars)"
            f"\n  IC score vs next-hour raw return, past the cost gate: {ic_gate:+.4f}  ({n_gate} bars)"
            f"\n  |mu| / round-trip among eligible: median {med_ratio:.2f}x   90th {p90_ratio:.2f}x   gate is {THRESH:.1f}x"
            f"\n  price ok {eligible[rows].mean():.3f}   gate {gate[rows].mean():.3f}",
            flush=True,
        )
        print(
            f"  {'sector':28s} {'n':>3} {'px>=60':>7} {'med px':>8} {'corr':>6} {'gate|px':>8} {'both':>6} {'leg n':>6} {'gross bp':>9}",
            flush=True,
        )
        for c, name in enumerate(panel["sector_order"]):
            cols = np.flatnonzero(codes == c)
            if len(cols) < MIN_SECTOR_NAMES:
                continue
            both = 0
            leg_n = []
            gross = []
            for i in rows:
                x = mix[i, cols]
                ok = np.isfinite(x)
                n_all = int(ok.sum())
                if n_all < MIN_SECTOR_NAMES:
                    continue
                k = max(1, int(np.floor(TOP_FRAC * n_all)))
                keep = gate[i, cols]
                idx = np.flatnonzero(ok & keep)
                if len(idx) < 2:
                    continue
                xs = x[idx]
                order = np.argsort(xs)
                long_side = [j for j in order[::-1][:k] if xs[j] > 0]
                short_side = [j for j in order[:k] if xs[j] < 0]
                if not long_side or not short_side:
                    continue
                both += 1
                leg_n.append(len(long_side) + len(short_side))
                r = panel["raw_seg"][i, cols]
                g = 0.5 * np.nanmean(r[idx[long_side]]) - 0.5 * np.nanmean(r[idx[short_side]])
                if np.isfinite(g):
                    gross.append(g)
            elig_share = float(eligible[rows][:, cols].mean())
            px = real[rows][:, cols]
            med_px = float(np.nanmedian(px))
            gate_given = float(gate[rows][:, cols][eligible[rows][:, cols]].mean()) if elig_share > 0 else float("nan")
            corr = sector_abs_corr(panel["raw_seg"][rows], cols)
            g_bp = float(np.mean(gross) * 1e4) if gross else float("nan")
            leg = float(np.mean(leg_n)) if leg_n else float("nan")
            print(
                f"  {name:28s} {len(cols):3d} {elig_share:7.2f} {med_px:8.0f} {corr:6.2f} {gate_given:8.3f} {both / len(rows):6.2f} {leg:6.1f} {g_bp:9.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
