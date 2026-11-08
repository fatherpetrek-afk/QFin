"""One sector the account says should work, and one it says should not.

The fit is the full panel: same residual, same six coordinates, same 4.2x gate,
real price at least $60. Only the named sector is allowed into the book.
Information Technology is the predicted pass. Financials is the predicted miss.
Nothing here retunes the gate. Test bars are not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cs_ac_v1 import build_ac_coordinates, run_threshold_book
from cs_band_v1 import KEYS, MIN_PX, THRESH, WINDOW, paint_cloud, p_ge
from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP
from cs_cost_v1 import build
from cs_cost_v3 import decision_bars
from cs_curve_v1 import stats
from cs_f7state_v1 import demean
from cs_hiprice_v1 import COSTS, PICK_COST, PICK_FROM, combined_panel, real_price_factor, walk_fit
from cs_np_v1 import shuffle_within
from cs_resid_v1 import newey_west

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_sector_v1_out"
N_SHUF = 80
SEED = 67
CHECKS = (
    ("Information Technology", "A. Information Technology. Predicted to clear the cloud"),
    ("Financials", "B. Financials. Predicted to stay inside the cloud"),
)


def one_sector(mix, expected, eligible, c_in, c_out, common, codes, sector_id, n_days, rng, t0, label):
    keep = codes == sector_id
    m = np.where(keep, mix, np.nan)
    e = np.where(keep, expected, np.nan)
    ratio = np.where(np.isfinite(m) & (np.abs(m) > 1e-12), e / m, np.nan)
    base = np.where(eligible & keep, m, np.nan)
    big = [sector_id]
    net, gross, held = run_threshold_book(
        base, e, eligible, c_in, c_out,
        common["raw_seg"], common["bar_ret"], common["steps"], common["day_id"], common["last"],
        codes, big, n_days, THRESH,
    )
    null = np.empty((N_SHUF, n_days), dtype=np.float64)
    for i in range(N_SHUF):
        shuf_m = shuffle_within(base, codes, big, rng)
        null[i], _, _ = run_threshold_book(
            shuf_m, ratio * shuf_m, eligible, c_in, c_out,
            common["raw_seg"], common["bar_ret"], common["steps"], common["day_id"], common["last"],
            codes, big, n_days, THRESH,
        )
        if (i + 1) % 20 == 0:
            print(f"    {label} shuffle {i + 1}/{N_SHUF}  [{time.perf_counter() - t0:.0f}s]", flush=True)
    return net, gross, held, null


def pack_stats(net, gross, null, pick, val, show):
    out = {}
    for name, mask in (("train", pick), ("val", val), ("all", show)):
        real_bp = float(net[mask].mean() * 1e4)
        null_bp = null[:, mask].mean(axis=1) * 1e4
        _n, _mu, tstat = newey_west(net[mask], 5)
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


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    print(
        "Prediction before the runs: Information Technology clears the reassignment cloud. "
        "Financials does not. Gate stays at 4.2.",
        flush=True,
    )
    meta, fetched = combined_panel(json.loads(SPLIT_PATH.read_text(encoding="utf-8")))
    panel = build(meta)
    steps, bar_ret = decision_bars(meta)
    coords = build_ac_coordinates(meta, panel)
    dates = panel["day_dates"]
    day_id = panel["day_id"]
    labels = panel["day_label_all"]
    n_days = len(dates)
    codes = panel["codes"]
    order = list(panel["sector_order"])
    spread, fee = COSTS[PICK_COST]
    factor = real_price_factor(list(meta["names"]), fetched, dates)
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
    print(f"[fit] {time.perf_counter() - t0:.0f}s  names {meta['n_names']}", flush=True)
    common = dict(
        raw_seg=panel["raw_seg"][rows],
        bar_ret=bar_ret,
        steps=steps[rows],
        day_id=day_id[rows],
        last=panel["last_of_day"][rows],
    )
    packs = []
    for sector_name, title in CHECKS:
        sector_id = order.index(sector_name)
        n_sec = int((codes == sector_id).sum())
        if n_sec < MIN_SECTOR_NAMES:
            raise SystemExit(f"{sector_name} has only {n_sec} names")
        net, gross, held, null = one_sector(
            mix[rows], expected[rows], eligible[rows], c_in[rows], c_out[rows],
            common, codes, sector_id, n_days, rng, t0, sector_name,
        )
        stats_pack = pack_stats(net, gross, null, pick, val, show)
        pack = {
            "net": net,
            "null": null,
            "dates": dates,
            "mask_val": val,
            "mask_show": show,
            "active": float((np.abs(held).sum(axis=1) > 1e-8).mean()),
            "names": n_sec,
            **stats_pack,
        }
        packs.append((sector_name, title, pack))
        tr, va = stats_pack["train"], stats_pack["val"]
        print(
            f"[{sector_name}] n={n_sec} active {pack['active']:.2f}  "
            f"train {tr['per_year']*100:+.2f}%/yr DD {tr['worst_drop']*100:.2f}% "
            f"p={tr['p']:.3f} t={tr['t']:+.2f}  "
            f"val {va['per_year']*100:+.2f}%/yr p={va['p']:.3f} t={va['t']:+.2f}  "
            f"[{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
    for ax, (_name, title, pack) in zip(axes, packs):
        paint_cloud(ax, pack, title)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.text(
        0.5, 0.012,
        "Full-panel fit, then the book may hold only that sector. Same 4.2x gate, real price >= $60. "
        f"{N_SHUF} reassignments inside the sector. Blue is the middle, green is the outer edge. "
        "Dashed line starts validation. Test period not opened.",
        ha="center", fontsize=8.0, color="#6b7380",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_img = OUT_DIR / "cs_sector.png"
    fig.savefig(out_img, dpi=150)
    extra = ROOT / "cs_force_v4_out"
    extra.mkdir(parents=True, exist_ok=True)
    fig.savefig(extra / "cs_sector.png", dpi=150)
    plt.close(fig)
    report = {
        name: {
            "names": pack["names"],
            "active_rows": pack["active"],
            "train": {k: float(v) for k, v in pack["train"].items() if k != "null_per_year"},
            "val": {k: float(v) for k, v in pack["val"].items() if k != "null_per_year"},
            "all": {k: float(v) for k, v in pack["all"].items() if k != "null_per_year"},
        }
        for name, _title, pack in packs
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[wrote] {out_img}", flush=True)
    print(f"[done] {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
