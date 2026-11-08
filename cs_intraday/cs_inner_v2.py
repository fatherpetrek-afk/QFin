"""
The state cloud fixed who had already moved. The width terciles did not.

Names with a narrow 5/63 band were still under-covered, and wide bands were
over-covered. That blend was left at the fast/slow mix. This pass keeps the
external quarter cloud and the already-moved / not-yet split, and only changes
which past energy stretches the within-pool shape: the blend, the 5-day energy,
or the 63-day energy.

The late relative move is also the sum of that external cloud and the state
cloud, drawn independently. On the previous pass their covariance was small,
and pairing them made the center band wide. Validation only. No profit.
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
from cs_order_v1 import demean_rows, mix_width, prices_at_cut, standardize
from cs_proj_v1 import systematic_relative, walk_projection
from cs_resid_v1 import daily_last_close
from nested_v1 import FAST_WIN, SLOW_WIN

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_inner_v2_out"
SHAPE_Q = np.array([0.025, 0.10, 0.25, 0.75, 0.90, 0.975])


def sqrt_energy(move: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(np.maximum(trailing_mean(move**2, window), 0.0))


def tercile_levels(y, quant, width, train, usable) -> dict:
    group, edges = tercile_groups(width, train)
    named = {}
    for j, name in enumerate(("narrow", "mid", "wide")):
        named[name] = pack(level_dict(y, quant, usable & (group == j)))
    return named, [float(edges[0]), float(edges[1])], group


def independent_sum(factor, delta, stretch, z, width, state, days) -> np.ndarray:
    """External quarter atoms plus the state cloud, without pairing the days."""
    quant = np.full(width.shape + (len(SHAPE_Q),), np.nan)
    for t in days:
        if not np.isfinite(stretch[t]):
            continue
        past = factor[t - SLOW_WIN : t]
        good = np.isfinite(past).all(axis=1)
        if int(good.sum()) < SLOW_WIN // 2:
            continue
        db = np.where(np.isfinite(delta[t]), delta[t], 0.0)
        ext = (past[good] @ db.T) * stretch[t]
        hist_z = z[:t]
        hist_s = state[:t]
        for bit in (0, 1):
            use = (hist_s == bit) & np.isfinite(hist_z)
            count = use.sum(axis=0)
            keep = np.flatnonzero((state[t] == bit) & np.isfinite(width[t]) & (count >= SLOW_WIN))
            if keep.size == 0:
                continue
            hist = hist_z[:, keep]
            mask = use[:, keep]
            scaled = np.where(mask, hist * width[t, keep], np.nan)
            rows = np.isfinite(scaled).any(axis=1)
            block = scaled[rows]
            ext_k = ext[:, keep]
            samples = ext_k[:, None, :] + block[None, :, :]
            quant[t, keep] = np.nanquantile(samples.reshape(-1, keep.size), SHAPE_Q, axis=0).T
    return quant


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
    stretch, alpha_ext, epochs_ext = factor_stretch(f_late, train)
    width_ext, quant_ext = external_cloud(f_late, delta, stretch, labels)

    width_blend, alpha_blend, epochs_blend = mix_width(late_inner, train)
    width_early, _alpha_early, _epochs_early = mix_width(early_inner, train)
    widths = {
        "blend": width_blend,
        "fast": sqrt_energy(late_inner, FAST_WIN),
        "slow": sqrt_energy(late_inner, SLOW_WIN),
    }
    state = state_flags(early_inner, width_early)
    z_zero = np.zeros_like(late_inner)
    val_days = np.flatnonzero(labels == "val")
    val = labels == "val"
    ext_levels = level_dict(late_ext, quant_ext, val[:, None] & np.isfinite(late_ext) & np.isfinite(quant_ext).all(axis=2))

    both = val[:, None] & np.isfinite(late_ext) & np.isfinite(late_inner)
    corr = float(np.corrcoef(late_ext[both], late_inner[both])[0, 1])
    inner_levels = {}
    full_levels = {}
    terciles = {}
    quant_inner = {}
    for name, width in widths.items():
        z = standardize(late_inner, width)
        quant = joint_state_quantiles(z_zero, z, z_zero, width, state, val_days)
        quant_sum = independent_sum(f_late, delta, stretch, z, width, state, val_days)
        quant_inner[name] = quant
        usable = val[:, None] & np.isfinite(late_inner) & np.isfinite(quant).all(axis=2)
        usable_sum = val[:, None] & np.isfinite(late_rel) & np.isfinite(quant_sum).all(axis=2)
        slices = {
            "all": np.ones(late_inner.shape, dtype=bool),
            "already_moved": state == 1,
            "not_yet": state == 0,
        }
        inner_levels[name] = {key: pack(level_dict(late_inner, quant, usable & mask)) for key, mask in slices.items()}
        full_levels[name] = {key: pack(level_dict(late_rel, quant_sum, usable_sum & mask)) for key, mask in slices.items()}
        terciles[name], edges, group = tercile_levels(late_inner, quant, width, train, usable)
        terciles[name]["edges"] = edges
        hit = inner_levels[name]["all"]["80"]["hit_rate"]
        moved = inner_levels[name]["already_moved"]["80"]["hit_rate"]
        still = inner_levels[name]["not_yet"]["80"]["hit_rate"]
        narrow = terciles[name]["narrow"]["80"]["hit_rate"]
        wide = terciles[name]["wide"]["80"]["hit_rate"]
        full80 = full_levels[name]["all"]["80"]["hit_rate"]
        full50 = full_levels[name]["all"]["50"]["hit_rate"]
        print(
            f"[{name}] inner80 {hit:.3f} moved {moved:.3f} still {still:.3f} "
            f"narrow {narrow:.3f} wide {wide:.3f}  full50 {full50:.3f} full80 {full80:.3f}",
            flush=True,
        )
        if name in ("fast", "slow"):
            val_plot = val & np.isfinite(quant).any(axis=(1, 2))
            save_ready = (name, val_plot, group, quant, width)
            if name == "fast":
                fast_plot = save_ready
            else:
                slow_plot = save_ready

    print(
        f"[external] alpha {alpha_ext:.3f} epochs {epochs_ext}  "
        f"50% {ext_levels['50'][0]:.3f}  80% {ext_levels['80'][0]:.3f}  95% {ext_levels['95'][0]:.3f}",
        flush=True,
    )
    print(f"[corr] external vs inner {corr:.3f}  blend alpha {alpha_blend:.3f} epochs {epochs_blend}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    for name, plot in (("fast", fast_plot), ("slow", slow_plot)):
        _name, val_plot, group, quant, width = plot
        moved = inner_levels[name]["already_moved"]["80"]["hit_rate"]
        still = inner_levels[name]["not_yet"]["80"]["hit_rate"]
        narrow = terciles[name]["narrow"]["80"]["hit_rate"]
        wide = terciles[name]["wide"]["80"]["hit_rate"]
        save_groups(
            OUT_DIR / f"cs_inner_{name}.png",
            index,
            val_plot,
            group,
            quant,
            late_inner,
            width,
            train,
            (
                f"池外云不动。池内按是否已经动过分开形状，宽度只用{('5日' if name == 'fast' else '63日')}能量。"
                f"已动 {moved:.0%}，未动 {still:.0%}，窄组 {narrow:.0%}，宽组 {wide:.0%}。不是收益。"
            ),
        )

    payload = {
        "cut": "10:00",
        "external_alpha": alpha_ext,
        "external_coverage": pack(ext_levels),
        "blend_alpha": alpha_blend,
        "correlation_external_inner": corr,
        "inner_by_width": inner_levels,
        "full_independent_sum": full_levels,
        "inner_width_terciles": terciles,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
