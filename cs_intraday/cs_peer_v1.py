"""
Peers inside the pool, after the three common factors are removed.

Framework7 assumed the residual lives inside one industry: a few common betas,
companies that still differ, and names that pull on each other. Here the
residual after B f is split by two peer maps, both known before the day:

  GICS sector, leave-one-out.
  The five names whose daily residual tracked this one most closely over the
  previous quarter.

Three questions on validation, all fit on train only:

  1. How much of a name's 10:00-to-close residual moves with its peers on the
     same afternoon. This is description, not forecast.
  2. Whether what the peers did before 10:00, or how far this name ran ahead of
     them, shifts the afternoon residual. Rank correlation by day, and inside
     each sector. Sectors are picked on train, then read on validation.
  3. Whether that shift, used as the center, narrows the band at the same
     coverage. The width and the already-moved shape are the ones from
     cs_inner_v4.

Test bars are not loaded. No equity curve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import label_days, load_fields
from cs_improve_v1 import save_groups, tercile_groups
from cs_inner_v1 import joint_state_quantiles, level_dict, pack, project_days, state_flags
from cs_inner_v4 import shrunk_width
from cs_order_v1 import demean_rows, mix_width, prices_at_cut, standardize
from cs_proj_v1 import systematic_relative, walk_projection
from cs_resid_v1 import daily_last_close, newey_west
from cs_xdist_v1 import style_axes
from nested_v1 import SLOW_WIN

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
UNIVERSE_PATH = ROOT / "universe_sp100.csv"
OUT_DIR = ROOT / "cs_peer_v1_out"
N_NEIGHBORS = 5
MIN_SECTOR_NAMES = 5
SHRINK = 0.5


def load_day_panels(meta: dict) -> dict:
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
    prev = prices.shift(1).reindex(index).to_numpy(np.float64)
    cut = cut_px.reindex(index).to_numpy(np.float64)
    last = prices.reindex(index).to_numpy(np.float64)
    ok = np.isfinite(prev) & np.isfinite(cut) & np.isfinite(last) & (prev > 0) & (cut > 0) & (last > 0)
    r_early = np.full(cut.shape, np.nan)
    r_late = np.full(cut.shape, np.nan)
    r_early[ok] = cut[ok] / prev[ok] - 1.0
    r_late[ok] = last[ok] / cut[ok] - 1.0
    early_rel = demean_rows(r_early)
    late_rel = demean_rows(r_late)
    f_early = project_days(r_early, loading)
    f_late = project_days(r_late, loading)
    early_inner = early_rel - systematic_relative(loading, f_early, r_early)
    late_inner = late_rel - systematic_relative(loading, f_late, r_late)
    return {
        "index": index,
        "labels": labels,
        "early_inner": early_inner,
        "late_inner": late_inner,
    }


def sector_codes(names: list[str]) -> tuple[np.ndarray, list[str]]:
    table = pd.read_csv(UNIVERSE_PATH)
    table["key"] = table["symbol"].str.replace(".", "_", regex=False)
    lookup = dict(zip(table["key"], table["sector"]))
    sectors = [lookup[name] for name in names]
    order = sorted(set(sectors))
    return np.array([order.index(s) for s in sectors]), order


def static_peer_mean(values: np.ndarray, member: np.ndarray, min_peers: int = 2) -> np.ndarray:
    good = np.isfinite(values)
    total = np.where(good, values, 0.0) @ member.T
    count = good.astype(np.float64) @ member.T
    out = np.full(values.shape, np.nan)
    ok = count >= min_peers
    out[ok] = total[ok] / count[ok]
    return out


def neighbor_index(full: np.ndarray, window: int, k: int) -> np.ndarray:
    """Top-k past-quarter residual correlates of each name. Day t sees days before t."""
    n_days, n_names = full.shape
    idx = np.full((n_days, n_names, k), -1, dtype=np.int32)
    need = max(20, (2 * window) // 3)
    for t in range(window, n_days):
        block = full[t - window : t]
        finite = np.isfinite(block)
        seen = finite.sum(axis=0)
        use = seen >= need
        if int(use.sum()) <= k + 1:
            continue
        mean = np.where(finite, block, 0.0).sum(axis=0) / np.maximum(seen, 1)
        centered = np.where(finite, block - mean, 0.0)
        sd = np.sqrt((centered**2).sum(axis=0) / np.maximum(seen - 1, 1))
        sd = np.where(sd > 1e-12, sd, np.nan)
        z = centered / sd
        z[:, ~use] = 0.0
        corr = (z.T @ z) / max(window - 1, 1)
        corr[:, ~use] = -np.inf
        np.fill_diagonal(corr, -np.inf)
        top = np.argsort(-corr, axis=1)[:, :k]
        top[~use] = -1
        idx[t] = top
    return idx


def dynamic_peer_mean(values: np.ndarray, idx: np.ndarray, min_peers: int = 2) -> np.ndarray:
    n_days, n_names, _k = idx.shape
    out = np.full(values.shape, np.nan)
    for t in range(n_days):
        row = idx[t]
        known = row >= 0
        if not known.any():
            continue
        picked = np.where(known, values[t][np.where(known, row, 0)], np.nan)
        count = np.isfinite(picked).sum(axis=1)
        ok = count >= min_peers
        out[t, ok] = np.nanmean(picked[ok], axis=1)
    return out


def rank_corr(x: np.ndarray, y: np.ndarray, min_n: int) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < min_n:
        return np.nan
    xr = np.argsort(np.argsort(x[mask])).astype(np.float64)
    yr = np.argsort(np.argsort(y[mask])).astype(np.float64)
    xr -= xr.mean()
    yr -= yr.mean()
    den = np.sqrt((xr**2).sum() * (yr**2).sum())
    if den < 1e-12:
        return np.nan
    return float((xr * yr).sum() / den)


def daily_ic(signal: np.ndarray, target: np.ndarray, days: np.ndarray, cols=None, min_n: int = 20) -> np.ndarray:
    out = np.full(len(days), np.nan)
    for i, t in enumerate(days):
        x = signal[t] if cols is None else signal[t, cols]
        y = target[t] if cols is None else target[t, cols]
        out[i] = rank_corr(x, y, min_n)
    return out


def daily_spread(signal: np.ndarray, target: np.ndarray, days: np.ndarray) -> np.ndarray:
    out = np.full(len(days), np.nan)
    for i, t in enumerate(days):
        mask = np.isfinite(signal[t]) & np.isfinite(target[t])
        if int(mask.sum()) < 20:
            continue
        sig = signal[t, mask]
        tgt = target[t, mask]
        med = np.median(sig)
        hi = tgt[sig > med]
        lo = tgt[sig <= med]
        if len(hi) and len(lo):
            out[i] = hi.mean() - lo.mean()
    return out


def summarize_ic(ic: np.ndarray) -> dict:
    n, mu, t = newey_west(ic, 1)
    good = ic[np.isfinite(ic)]
    return {
        "n_days": n,
        "ic": mu,
        "t": t,
        "positive_share": float((good > 0).mean()) if len(good) else float("nan"),
    }


def fit_linear(columns: list[np.ndarray], target: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Pooled least squares through the origin on train rows."""
    x = np.stack([c[rows] for c in columns], axis=-1)
    y = target[rows]
    ok = np.isfinite(x).all(axis=-1) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    gram = x.T @ x + 1e-12 * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def apply_linear(columns: list[np.ndarray], coef: np.ndarray) -> np.ndarray:
    out = np.zeros(columns[0].shape)
    for c, b in zip(columns, coef):
        out = out + b * c
    return out


def mse_ratio(y: np.ndarray, pred: np.ndarray, rows: np.ndarray) -> tuple[float, int]:
    ok = rows & np.isfinite(y) & np.isfinite(pred)
    if int(ok.sum()) == 0:
        return float("nan"), 0
    base = float(np.mean(y[ok] ** 2))
    return float(np.mean((y[ok] - pred[ok]) ** 2)) / base, int(ok.sum())


def explained_share(y: np.ndarray, x: np.ndarray, train_rows: np.ndarray, val_rows: np.ndarray) -> dict:
    coef = fit_linear([x], y, train_rows)
    ratio, n = mse_ratio(y, coef[0] * x, val_rows)
    return {"slope_train": float(coef[0]), "val_share": 1.0 - ratio, "n": n}


def band_width(quant: np.ndarray, usable: np.ndarray) -> float:
    return float(np.nanmean((quant[:, :, 4] - quant[:, :, 1])[usable]))


def save_sector_bars(path: Path, sector_names, rows, note: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    ypos = np.arange(len(sector_names))
    for ax, key, title in zip(axes, ("follow", "gap"), ("残差邻居早盘涨跌 → 自己下午", "自己早盘领先同行业多少 → 自己下午")):
        tr = [rows[s][key]["train"]["ic"] for s in sector_names]
        va = [rows[s][key]["val"]["ic"] for s in sector_names]
        ax.barh(ypos + 0.18, tr, height=0.34, color="#b7cce4", label="训练段")
        ax.barh(ypos - 0.18, va, height=0.34, color="#1f4e79", label="验证段")
        ax.axvline(0.0, color="#9aa3ad", lw=0.8)
        ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
        ax.set_xlabel("每天行业内的秩相关，取平均")
        style_axes(ax)
        ax.grid(axis="x", color="#e6ebf0", lw=0.6)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(sector_names)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("同一行业里，早盘的先后能不能说明下午", fontsize=15, color="#1c2430")
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    names = list(meta["names"])
    panels = load_day_panels(meta)
    index = panels["index"]
    labels = panels["labels"]
    early = panels["early_inner"]
    late = panels["late_inner"]
    train = labels == "train"
    val = labels == "val"
    train_rows = train[:, None] & np.ones(late.shape, dtype=bool)
    val_rows = val[:, None] & np.ones(late.shape, dtype=bool)
    train_days = np.flatnonzero(train)
    val_days = np.flatnonzero(val)

    codes, sector_order = sector_codes(names)
    same = (codes[:, None] == codes[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    full = early + late
    prev_full = np.vstack([np.full((1, full.shape[1]), np.nan), full[:-1]])
    neighbors = neighbor_index(full, SLOW_WIN, N_NEIGHBORS)

    peer_early_g = static_peer_mean(early, same)
    peer_late_g = static_peer_mean(late, same)
    peer_prev_g = static_peer_mean(prev_full, same)
    peer_early_d = dynamic_peer_mean(early, neighbors)
    peer_late_d = dynamic_peer_mean(late, neighbors)
    gap_g = early - peer_early_g
    gap_d = early - peer_early_d
    prev_gap_g = prev_full - peer_prev_g

    same_day = {
        "sector_afternoon": explained_share(late, peer_late_g, train_rows, val_rows),
        "neighbor_afternoon": explained_share(late, peer_late_d, train_rows, val_rows),
        "sector_morning": explained_share(early, peer_early_g, train_rows, val_rows),
    }
    print(
        f"[same afternoon] sector explains {same_day['sector_afternoon']['val_share']:.3f}  "
        f"neighbors {same_day['neighbor_afternoon']['val_share']:.3f}  "
        f"(morning sector {same_day['sector_morning']['val_share']:.3f})",
        flush=True,
    )

    signals = {
        "sector_follow": [peer_early_g],
        "neighbor_follow": [peer_early_d],
        "own_morning": [early],
        "sector_gap": [gap_g],
        "neighbor_gap": [gap_d],
        "yesterday_sector_gap": [prev_gap_g],
        "sector_joint": [peer_early_g, early, prev_gap_g],
        "neighbor_joint": [peer_early_d, early],
    }
    forecasts = {}
    preds = {}
    for name, cols in signals.items():
        coef = fit_linear(cols, late, train_rows)
        pred = apply_linear(cols, coef)
        preds[name] = pred
        ratio, n = mse_ratio(late, pred, val_rows)
        ic_tr = summarize_ic(daily_ic(pred, late, train_days))
        ic_va = summarize_ic(daily_ic(pred, late, val_days))
        spread = daily_spread(pred, late, val_days)
        _ns, sp_mu, sp_t = newey_west(spread, 1)
        forecasts[name] = {
            "coef_train": [float(c) for c in coef],
            "val_mse_ratio": ratio,
            "n": n,
            "train_ic": ic_tr,
            "val_ic": ic_va,
            "val_spread_bps": sp_mu * 1e4,
            "val_spread_t": sp_t,
        }
        print(
            f"[{name:22s}] coef {np.round(coef, 4)}  mse/zero {ratio:.4f}  "
            f"IC train {ic_tr['ic']:+.4f} (t {ic_tr['t']:+.1f})  val {ic_va['ic']:+.4f} (t {ic_va['t']:+.1f})  "
            f"spread {sp_mu * 1e4:+.2f} bp (t {sp_t:+.1f})",
            flush=True,
        )

    sector_rows = {}
    picked = []
    for code, sector in enumerate(sector_order):
        cols = np.flatnonzero(codes == code)
        if len(cols) < MIN_SECTOR_NAMES:
            continue
        row = {"n_names": int(len(cols))}
        block = full[train][:, cols]
        finite = np.isfinite(block)
        mean = np.where(finite, block, 0.0).sum(0) / np.maximum(finite.sum(0), 1)
        centered = np.where(finite, block - mean, 0.0)
        sd = np.sqrt((centered**2).sum(0) / np.maximum(finite.sum(0) - 1, 1))
        corr = (centered / sd).T @ (centered / sd) / max(len(block) - 1, 1)
        row["train_residual_corr"] = float(corr[~np.eye(len(cols), dtype=bool)].mean())
        # Inside one sector the leave-one-out sector mean only reverses the rank
        # of the name's own morning, so the follow test uses residual neighbors.
        for key, signal in (("follow", peer_early_d), ("gap", gap_g)):
            row[key] = {
                "train": summarize_ic(daily_ic(signal, late, train_days, cols, MIN_SECTOR_NAMES)),
                "val": summarize_ic(daily_ic(signal, late, val_days, cols, MIN_SECTOR_NAMES)),
            }
            if abs(row[key]["train"]["t"]) >= 2.0:
                picked.append((sector, key, float(np.sign(row[key]["train"]["ic"]))))
        sector_rows[sector] = row
        print(
            f"[sector {sector:24s}] n {len(cols):2d} corr {row['train_residual_corr']:.3f}  "
            f"follow tr {row['follow']['train']['ic']:+.3f}/{row['follow']['train']['t']:+.1f} "
            f"va {row['follow']['val']['ic']:+.3f}/{row['follow']['val']['t']:+.1f}  "
            f"gap tr {row['gap']['train']['ic']:+.3f}/{row['gap']['train']['t']:+.1f} "
            f"va {row['gap']['val']['ic']:+.3f}/{row['gap']['val']['t']:+.1f}",
            flush=True,
        )
    held = []
    for sector, key, sign in picked:
        val_row = sector_rows[sector][key]["val"]
        held.append(
            {
                "sector": sector,
                "signal": key,
                "train_sign": sign,
                "val_ic": val_row["ic"],
                "val_t": val_row["t"],
                "same_sign": bool(np.sign(val_row["ic"]) == sign),
            }
        )
        print(f"[picked on train] {sector} {key} sign {sign:+.0f} -> val {val_row['ic']:+.3f} (t {val_row['t']:+.1f})", flush=True)

    width_early, _a, _e = mix_width(early, train)
    state = state_flags(early, width_early)
    by_state = {}
    for key, bit in (("already_moved", 1), ("not_yet", 0)):
        masked = np.where(state == bit, gap_g, np.nan)
        by_state[key] = {
            "train": summarize_ic(daily_ic(masked, late, train_days, None, MIN_SECTOR_NAMES)),
            "val": summarize_ic(daily_ic(masked, late, val_days, None, MIN_SECTOR_NAMES)),
        }
        print(
            f"[gap by state {key:13s}] train {by_state[key]['train']['ic']:+.4f} (t {by_state[key]['train']['t']:+.1f})  "
            f"val {by_state[key]['val']['ic']:+.4f} (t {by_state[key]['val']['t']:+.1f})",
            flush=True,
        )
    z_zero = np.zeros_like(late)
    slices = {"all": np.ones(late.shape, dtype=bool), "already_moved": state == 1, "not_yet": state == 0}
    dist = {}
    for name, center in (("zero_center", np.zeros_like(late)), ("sector_joint", preds["sector_joint"]), ("neighbor_joint", preds["neighbor_joint"])):
        center = np.where(np.isfinite(center), center, np.nan)
        resid = late - center
        width = shrunk_width(resid, SHRINK)
        z = standardize(resid, width)
        quant = joint_state_quantiles(z_zero, z, z_zero, width, state, val_days) + center[:, :, None]
        usable = val[:, None] & np.isfinite(late) & np.isfinite(quant).all(axis=2)
        levels = {key: pack(level_dict(late, quant, usable & mask)) for key, mask in slices.items()}
        dist[name] = {"coverage": levels, "mean_80_width": band_width(quant, usable), "n": int(usable.sum())}
        print(
            f"[band {name:14s}] 50 {levels['all']['50']['hit_rate']:.3f}  80 {levels['all']['80']['hit_rate']:.3f}  "
            f"95 {levels['all']['95']['hit_rate']:.3f}  mean 80% width {dist[name]['mean_80_width'] * 100:.3f}%",
            flush=True,
        )
        if name == "sector_joint":
            group, _edges = tercile_groups(width, train)
            plot_pack = (quant, width, group)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shown = [s for s in sector_order if s in sector_rows]
    save_sector_bars(
        OUT_DIR / "cs_peer_sector_ic.png",
        shown,
        sector_rows,
        "残差先去掉三个公共因子。浅色是训练段，深色是验证段。两段同号、而且离 0 远，才算站得住。不是收益。",
    )
    quant, width, group = plot_pack
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    save_groups(
        OUT_DIR / "cs_peer_band.png",
        index,
        val & np.isfinite(quant).any(axis=(1, 2)),
        group,
        quant,
        late,
        width,
        train,
        (
            f"中心用同业早盘和自己早盘（训练段定的系数），宽度和形状同 cs_inner_v4。"
            f"80% 盖住 {dist['sector_joint']['coverage']['all']['80']['hit_rate']:.0%}，"
            f"平均 80% 宽度 {dist['sector_joint']['mean_80_width'] * 100:.2f}%，"
            f"不移中心时 {dist['zero_center']['mean_80_width'] * 100:.2f}%。不是收益。"
        ),
    )
    payload = {
        "cut": "10:00",
        "neighbors": N_NEIGHBORS,
        "same_afternoon": same_day,
        "forecasts": forecasts,
        "sectors": sector_rows,
        "picked_on_train": held,
        "gap_by_morning_state": by_state,
        "bands": dist,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
