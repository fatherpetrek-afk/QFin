"""
Keep the external factor cloud. Rebuild only the within-pool piece.

The late relative move splits into (B - mean B) · f_late and the leftover.
The external cloud is the recent-quarter factor path, stretched by the
train-fitted 5-day/63-day energy. The early print does not enter that cloud.

The leftover is conditioned on the name's own early leftover, known at 10:00.
Three uses, all fit on train or on earlier days only: a wider band when that
early leftover is large, a center shifted by a pooled slope, and a separate
historical shape for names that have already moved. A residual factor of the
early leftover is scored only as a mean-squared-error diagnostic.

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
from cs_dynproj_v1 import component_energy, hit_rate
from cs_improve_v1 import cloud_quantiles, save_groups, tercile_groups
from cs_order_v1 import demean_rows, mix_width, order_shock, prices_at_cut, row_mean, standardize
from cs_proj_v1 import systematic_relative, walk_projection
from cs_resid_v1 import daily_last_close, fit_scale
from factor_v1 import K, align_sign, project
from nested_v1 import FAST_WIN, SLOW_WIN

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_inner_v1_out"
SHAPE_Q = np.array([0.025, 0.10, 0.25, 0.75, 0.90, 0.975])
Z_CAP = 3.0


def level_dict(y, quant, mask) -> dict:
    pairs = {"50": (2, 3), "80": (1, 4), "95": (0, 5)}
    return {name: hit_rate(y, quant[:, :, a], quant[:, :, b], mask) for name, (a, b) in pairs.items()}


def pack(levels) -> dict:
    return {name: {"hit_rate": hit, "n": n} for name, (hit, n) in levels.items()}


def project_days(returns: np.ndarray, loading: np.ndarray) -> np.ndarray:
    out = np.full((len(returns), K), np.nan)
    for t in range(len(returns)):
        f_t, _eps = project(returns[t], loading[t])
        out[t] = f_t
    return out


def factor_stretch(factor: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, float, int]:
    fast = component_energy(factor, FAST_WIN)
    slow = component_energy(factor, SLOW_WIN)
    alpha, epochs = fit_scale(
        np.nansum(factor**2, axis=1),
        np.nansum(fast, axis=1),
        np.nansum(slow, axis=1),
        train,
    )
    sigma = np.sqrt(np.maximum(alpha * fast + (1.0 - alpha) * slow, 0.0))
    base = np.nansum(slow, axis=1)
    pred = np.nansum(sigma**2, axis=1)
    stretch = np.full(len(factor), np.nan)
    ok = np.isfinite(base) & np.isfinite(pred) & (base > 1e-12) & (pred > 0)
    stretch[ok] = np.sqrt(pred[ok] / base[ok])
    return stretch, alpha, epochs


def external_cloud(factor, delta, stretch, labels):
    """Quarter cloud of (B - mean B) · f. Today's factor is not in the cloud."""
    n_days, n_names, _k = delta.shape
    quant = np.full((n_days, n_names, len(SHAPE_Q)), np.nan)
    width = np.full((n_days, n_names), np.nan)
    for t in range(SLOW_WIN, n_days):
        if not np.isfinite(stretch[t]):
            continue
        past = factor[t - SLOW_WIN : t]
        good = np.isfinite(past).all(axis=1)
        if int(good.sum()) < SLOW_WIN // 2:
            continue
        db = delta[t]
        good_name = np.isfinite(db).all(axis=1)
        db_fill = np.where(good_name[:, None], db, 0.0)
        scores = (past[good] @ db_fill.T) * stretch[t]
        scores[:, ~good_name] = np.nan
        width[t] = np.sqrt(np.nanmean(scores**2, axis=0))
        if labels[t] == "val":
            quant[t] = np.nanquantile(scores, SHAPE_Q, axis=0).T
    return width, quant


def pooled_slope(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> float:
    mask = train[:, None] & np.isfinite(x) & np.isfinite(y)
    xx = x[mask]
    yy = y[mask]
    denom = float(np.dot(xx, xx))
    if len(xx) < 100 or denom < 1e-18:
        return 0.0
    return float(np.dot(xx, yy) / denom)


def mse_ratio(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    ok = mask & np.isfinite(y) & np.isfinite(pred)
    if int(ok.sum()) == 0:
        return float("nan"), 0
    err = float(np.mean((y[ok] - pred[ok]) ** 2))
    base = float(np.mean(y[ok] ** 2))
    if base <= 0:
        return float("nan"), int(ok.sum())
    return err / base, int(ok.sum())


def capped_z(move: np.ndarray, width: np.ndarray) -> np.ndarray:
    out = np.full(move.shape, np.nan)
    ok = np.isfinite(move) & np.isfinite(width) & (width > 1e-8)
    out[ok] = np.clip(move[ok] / width[ok], -Z_CAP, Z_CAP)
    return out


def scale_width(move, width, z, train):
    """Width times a train-fitted function of today's own early z."""
    energy = row_mean(move**2)
    base = width**2
    blown = base * z**2
    alpha, epochs = fit_scale(energy, row_mean(blown), row_mean(base), train)
    mult = np.full(width.shape, np.nan)
    ok = np.isfinite(width) & np.isfinite(z)
    mult[ok] = np.sqrt(np.maximum(alpha * z[ok] ** 2 + (1.0 - alpha), 0.0))
    return width * mult, alpha, epochs


def state_flags(move: np.ndarray, width: np.ndarray) -> np.ndarray:
    """1 already moved, 0 not yet, -1 unknown. Known once today's move is in."""
    state = np.full(move.shape, -1, dtype=np.int8)
    known = np.isfinite(move) & np.isfinite(width) & (width > 1e-8)
    state[known & (np.abs(move) > width)] = 1
    state[known & (np.abs(move) <= width)] = 0
    return state


def add_center(quant: np.ndarray, center: np.ndarray) -> np.ndarray:
    out = quant + center[:, :, None]
    bad = ~np.isfinite(center)
    out[bad] = np.nan
    return out


def joint_state_quantiles(z_ext, z_inn, w_ext, w_inn, state, days) -> np.ndarray:
    quant = np.full(w_ext.shape + (len(SHAPE_Q),), np.nan)
    for t in days:
        hist_ext = z_ext[:t]
        hist_inn = z_inn[:t]
        hist_state = state[:t]
        for bit in (0, 1):
            use = (hist_state == bit) & np.isfinite(hist_ext) & np.isfinite(hist_inn)
            count = use.sum(axis=0)
            keep = (state[t] == bit) & np.isfinite(w_ext[t]) & np.isfinite(w_inn[t]) & (count >= SLOW_WIN)
            if not np.any(keep):
                continue
            draws = w_ext[t] * hist_ext + w_inn[t] * hist_inn
            draws[~use] = np.nan
            quant[t, keep] = np.nanquantile(draws[:, keep], SHAPE_Q, axis=0).T
    return quant


def residual_factors(panel: np.ndarray, window: int, k: int) -> np.ndarray:
    """Project today's early leftover onto the previous quarter's own basis."""
    n_days, n_names = panel.shape
    out = np.full((n_days, k), np.nan)
    prev = None
    need = max(k + 5, window // 2)
    for t in range(window, n_days):
        block = panel[t - window : t]
        use = np.isfinite(block).sum(axis=0) >= window // 2
        if int(use.sum()) < need or not np.isfinite(panel[t]).any():
            continue
        x = block[:, use]
        x = np.where(np.isfinite(x), x, 0.0)
        x = x - x.mean(axis=0)
        cov = (x.T @ x) / max(x.shape[0] - 1, 1)
        _evals, evecs = np.linalg.eigh(cov)
        full = np.zeros((n_names, k))
        full[use] = evecs[:, -k:]
        full = align_sign(full, prev)
        prev = full
        today = np.where(np.isfinite(panel[t]), panel[t], 0.0)
        out[t] = full.T @ today
    return out


def slice_levels(y, quant, usable, slices) -> dict:
    return {name: pack(level_dict(y, quant, usable & mask)) for name, mask in slices.items()}


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

    width_inner, alpha_inner, epochs_inner = mix_width(late_inner, train)
    width_early, alpha_early, epochs_early = mix_width(early_inner, train)
    z_early = capped_z(early_inner, width_early)
    width_scaled, alpha_scale, epochs_scale = scale_width(late_inner, width_inner, z_early, train)
    gamma = pooled_slope(early_inner, late_inner, train)
    center = gamma * early_inner
    resid = late_inner - center
    width_resid, alpha_resid, epochs_resid = mix_width(resid, train)

    z_ext = standardize(late_ext, width_ext)
    z_inner = standardize(late_inner, width_inner)
    z_scaled = standardize(late_inner, width_scaled)
    z_resid = standardize(resid, width_resid)
    z_zero = np.zeros_like(z_inner)
    state = state_flags(early_inner, width_early)
    val_days = np.flatnonzero(labels == "val")

    quant_inner = cloud_quantiles(z_inner, z_zero, width_inner, z_zero, val_days)
    quant_scaled = cloud_quantiles(z_scaled, z_zero, width_scaled, z_zero, val_days)
    quant_loc = add_center(cloud_quantiles(z_resid, z_zero, width_resid, z_zero, val_days), center)
    quant_state = joint_state_quantiles(z_zero, z_inner, z_zero, width_inner, state, val_days)
    quant_full_base = cloud_quantiles(z_ext, z_inner, width_ext, width_inner, val_days)
    quant_full_scale = cloud_quantiles(z_ext, z_scaled, width_ext, width_scaled, val_days)
    quant_full_loc = add_center(cloud_quantiles(z_ext, z_resid, width_ext, width_resid, val_days), center)
    quant_full_state = joint_state_quantiles(z_ext, z_inner, width_ext, width_inner, state, val_days)

    val = labels == "val"
    usable_ext = val[:, None] & np.isfinite(late_ext) & np.isfinite(quant_ext).all(axis=2)
    usable_inner = val[:, None] & np.isfinite(late_inner) & np.isfinite(quant_inner).all(axis=2)
    usable_scaled = val[:, None] & np.isfinite(late_inner) & np.isfinite(quant_scaled).all(axis=2)
    usable_loc = val[:, None] & np.isfinite(late_inner) & np.isfinite(quant_loc).all(axis=2)
    usable_state = val[:, None] & np.isfinite(late_inner) & np.isfinite(quant_state).all(axis=2)
    usable_full_base = val[:, None] & np.isfinite(late_rel) & np.isfinite(quant_full_base).all(axis=2)
    usable_full_scale = val[:, None] & np.isfinite(late_rel) & np.isfinite(quant_full_scale).all(axis=2)
    usable_full_loc = val[:, None] & np.isfinite(late_rel) & np.isfinite(quant_full_loc).all(axis=2)
    usable_full_state = val[:, None] & np.isfinite(late_rel) & np.isfinite(quant_full_state).all(axis=2)
    slices = {
        "all": np.ones(late_inner.shape, dtype=bool),
        "already_moved": state == 1,
        "not_yet": state == 0,
    }

    ext_levels = level_dict(late_ext, quant_ext, usable_ext)
    inner_levels = {
        "baseline": slice_levels(late_inner, quant_inner, usable_inner, slices),
        "scaled": slice_levels(late_inner, quant_scaled, usable_scaled, slices),
        "location": slice_levels(late_inner, quant_loc, usable_loc, slices),
        "state_shape": slice_levels(late_inner, quant_state, usable_state, slices),
    }
    full_levels = {
        "external_plus_baseline": slice_levels(late_rel, quant_full_base, usable_full_base, slices),
        "external_plus_scaled": slice_levels(late_rel, quant_full_scale, usable_full_scale, slices),
        "external_plus_location": slice_levels(late_rel, quant_full_loc, usable_full_loc, slices),
        "external_plus_state": slice_levels(late_rel, quant_full_state, usable_full_state, slices),
    }

    val_pair = val[:, None] & np.isfinite(late_rel) & np.isfinite(late_ext) & np.isfinite(late_inner)
    energy_late = float(np.mean(late_rel[val_pair] ** 2))
    energy_ext = float(np.mean(late_ext[val_pair] ** 2))
    energy_inner = float(np.mean(late_inner[val_pair] ** 2))
    own_ratio, own_n = mse_ratio(late_inner, center, val[:, None])
    g_resid = residual_factors(early_inner, SLOW_WIN, K)
    resid_shock = order_shock(late_inner, g_resid, SLOW_WIN)
    resid_ratio, resid_n = mse_ratio(late_inner, resid_shock, val[:, None])

    print(
        f"[external] alpha {alpha_ext:.3f} epochs {epochs_ext}  "
        f"80% {ext_levels['80'][0]:.3f} n {ext_levels['80'][1]}",
        flush=True,
    )
    print(
        f"[energy] late {energy_late:.3e} external {energy_ext / energy_late:.3f} "
        f"inner {energy_inner / energy_late:.3f}",
        flush=True,
    )
    print(
        f"[inner] base {alpha_inner:.3f} early {alpha_early:.3f} scale {alpha_scale:.3f} "
        f"resid {alpha_resid:.3f} gamma {gamma:.4f}  "
        f"epochs {epochs_inner}/{epochs_early}/{epochs_scale}/{epochs_resid}",
        flush=True,
    )
    print(f"[mse] own-early/zero {own_ratio:.3f} n {own_n}  residual-factor/zero {resid_ratio:.3f} n {resid_n}", flush=True)
    for name, levels in inner_levels.items():
        hit = levels["all"]["80"]["hit_rate"]
        moved = levels["already_moved"]["80"]["hit_rate"]
        still = levels["not_yet"]["80"]["hit_rate"]
        print(f"[inner 80 {name}] all {hit:.3f} moved {moved:.3f} still {still:.3f}", flush=True)
    for name, levels in full_levels.items():
        hit = levels["all"]["80"]["hit_rate"]
        moved = levels["already_moved"]["80"]["hit_rate"]
        still = levels["not_yet"]["80"]["hit_rate"]
        print(f"[full 80 {name}] all {hit:.3f} moved {moved:.3f} still {still:.3f}", flush=True)

    group, edges = tercile_groups(width_scaled, train)
    val_plot = val & np.isfinite(quant_scaled).any(axis=(1, 2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    save_groups(
        OUT_DIR / "cs_inner_scale.png",
        index,
        val_plot,
        group,
        quant_scaled,
        late_inner,
        width_scaled,
        train,
        (
            f"池外云保持不动，80% 盖住 {ext_levels['80'][0]:.0%}。"
            f"池内宽度随这只股票 10:00 前的残差放大，80% 盖住 {inner_levels['scaled']['all']['80']['hit_rate']:.0%}。"
            "不是收益。"
        ),
    )
    group_state, edges_state = tercile_groups(width_inner, train)
    val_state = val & np.isfinite(quant_state).any(axis=(1, 2))
    save_groups(
        OUT_DIR / "cs_inner_state.png",
        index,
        val_state,
        group_state,
        quant_state,
        late_inner,
        width_inner,
        train,
        (
            "已经走出自己早期带宽的名字，用过去同样状态下的形状。"
            f"80% 盖住 {inner_levels['state_shape']['all']['80']['hit_rate']:.0%}。不是收益。"
        ),
    )

    payload = {
        "cut": "10:00",
        "external_alpha": alpha_ext,
        "external_epochs": epochs_ext,
        "external_energy_share": energy_ext / energy_late,
        "inner_energy_share": energy_inner / energy_late,
        "alpha_inner": alpha_inner,
        "alpha_early": alpha_early,
        "alpha_scale": alpha_scale,
        "alpha_resid": alpha_resid,
        "epochs": {
            "inner": epochs_inner,
            "early": epochs_early,
            "scale": epochs_scale,
            "resid": epochs_resid,
        },
        "gamma_own_early": gamma,
        "mse_ratio_own_early_vs_zero": own_ratio,
        "mse_own_n": own_n,
        "mse_ratio_residual_factor_vs_zero": resid_ratio,
        "mse_residual_factor_n": resid_n,
        "train_scaled_width_terciles": [float(edges[0]), float(edges[1])],
        "train_inner_width_terciles": [float(edges_state[0]), float(edges_state[1])],
        "validation_coverage_external": pack(ext_levels),
        "validation_coverage_inner": inner_levels,
        "validation_coverage_full": full_levels,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
