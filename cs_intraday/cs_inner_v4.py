"""
Name-level 63-day energy is still too spread out across the pool.

Fast energy made the narrow and wide groups worse. Slow energy helped, and
dropping the energy altogether made every band too tight. This pass shrinks
each name's 63-day energy toward that day's pool level, then keeps the
already-moved / not-yet cloud. The shrink is fixed at one half and at zero,
not chosen on the validation sample. The external cloud is unchanged.

Validation only. Test bars are not loaded. No profit is computed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import label_days, load_fields
from cs_improve_v1 import save_groups, tercile_groups, trailing_mean
from cs_inner_v1 import (
    external_cloud,
    factor_stretch,
    joint_state_quantiles,
    level_dict,
    pack,
    project_days,
    state_flags,
)
from cs_inner_v2 import independent_sum
from cs_order_v1 import demean_rows, mix_width, prices_at_cut, standardize
from cs_proj_v1 import systematic_relative, walk_projection
from cs_resid_v1 import daily_last_close
from nested_v1 import SLOW_WIN

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_inner_v4_out"
LAMBDAS = (0.0, 0.5)


def shrunk_width(move: np.ndarray, lam: float) -> np.ndarray:
    slow = np.maximum(trailing_mean(move**2, SLOW_WIN), 0.0)
    pool = np.full(slow.shape[0], np.nan)
    good = np.isfinite(slow).any(axis=1)
    pool[good] = np.nanmedian(slow[good], axis=1)
    var = (1.0 - lam) * pool[:, None] + lam * slow
    return np.sqrt(np.maximum(var, 0.0))


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
    cut_px = prices_at_cut(close).reindex(prices.index)
    daily_ret = prices.pct_change(fill_method=None)
    index, loading, _factor, _returns = walk_projection(
        close.pct_change(fill_method=None).to_numpy(np.float64),
        close.index,
        daily_ret.iloc[1:],
    )
    labels = label_days(index, train_end, val_start)
    train = labels == "train"
    prev = prices.shift(1).reindex(index).to_numpy(np.float64)
    cut = cut_px.reindex(index).to_numpy(np.float64)
    last = prices.reindex(index).to_numpy(np.float64)
    usable_px = np.isfinite(prev) & np.isfinite(cut) & np.isfinite(last) & (prev > 0) & (cut > 0) & (last > 0)
    r_early = np.full(cut.shape, np.nan)
    r_late = np.full(cut.shape, np.nan)
    r_early[usable_px] = cut[usable_px] / prev[usable_px] - 1.0
    r_late[usable_px] = last[usable_px] / cut[usable_px] - 1.0
    early_rel = demean_rows(r_early)
    late_rel = demean_rows(r_late)
    f_early = project_days(r_early, loading)
    f_late = project_days(r_late, loading)
    early_ext = systematic_relative(loading, f_early, r_early)
    late_ext = systematic_relative(loading, f_late, r_late)
    early_inner = early_rel - early_ext
    late_inner = late_rel - late_ext
    delta = loading - np.nanmean(loading, axis=1)[:, None, :]
    stretch, _alpha_ext, _epochs_ext = factor_stretch(f_late, train)
    _width_ext, quant_ext = external_cloud(f_late, delta, stretch, labels)
    width_early, _a, _e = mix_width(early_inner, train)
    state = state_flags(early_inner, width_early)
    z_zero = np.zeros_like(late_inner)
    val_days = np.flatnonzero(labels == "val")
    val = labels == "val"
    ext_hit = level_dict(
        late_ext,
        quant_ext,
        val[:, None] & np.isfinite(late_ext) & np.isfinite(quant_ext).all(axis=2),
    )

    payload = {"cut": "10:00", "external_80": ext_hit["80"][0], "lambdas": {}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    slices = {
        "all": np.ones(late_inner.shape, dtype=bool),
        "already_moved": state == 1,
        "not_yet": state == 0,
    }
    for lam in LAMBDAS:
        width = shrunk_width(late_inner, lam)
        z = standardize(late_inner, width)
        quant = joint_state_quantiles(z_zero, z, z_zero, width, state, val_days)
        usable = val[:, None] & np.isfinite(late_inner) & np.isfinite(quant).all(axis=2)
        levels = {name: pack(level_dict(late_inner, quant, usable & mask)) for name, mask in slices.items()}
        group, edges = tercile_groups(width, train)
        terciles = {}
        for j, name in enumerate(("narrow", "mid", "wide")):
            terciles[name] = pack(level_dict(late_inner, quant, usable & (group == j)))
        quant_sum = independent_sum(f_late, delta, stretch, z, width, state, val_days)
        usable_sum = val[:, None] & np.isfinite(late_rel) & np.isfinite(quant_sum).all(axis=2)
        full = {name: pack(level_dict(late_rel, quant_sum, usable_sum & mask)) for name, mask in slices.items()}
        tag = f"{lam:.2f}"
        print(
            f"[lam {tag}] inner80 {levels['all']['80']['hit_rate']:.3f} "
            f"moved {levels['already_moved']['80']['hit_rate']:.3f} "
            f"still {levels['not_yet']['80']['hit_rate']:.3f} "
            f"narrow {terciles['narrow']['80']['hit_rate']:.3f} "
            f"wide {terciles['wide']['80']['hit_rate']:.3f} "
            f"full50 {full['all']['50']['hit_rate']:.3f} full80 {full['all']['80']['hit_rate']:.3f}",
            flush=True,
        )
        payload["lambdas"][tag] = {
            "inner": levels,
            "terciles": terciles,
            "full": full,
            "train_width_edges": [float(edges[0]), float(edges[1])],
        }
        save_groups(
            OUT_DIR / f"cs_inner_lam{tag}.png",
            index,
            val & np.isfinite(quant).any(axis=(1, 2)),
            group,
            quant,
            late_inner,
            width,
            train,
            (
                f"63 日能量向当天池子收缩，权重 {tag}。"
                f"已动 {levels['already_moved']['80']['hit_rate']:.0%}，"
                f"未动 {levels['not_yet']['80']['hit_rate']:.0%}，"
                f"窄组 {terciles['narrow']['80']['hit_rate']:.0%}，"
                f"宽组 {terciles['wide']['80']['hit_rate']:.0%}。不是收益。"
            ),
        )
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[external 80] {ext_hit['80'][0]:.3f}", flush=True)
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
