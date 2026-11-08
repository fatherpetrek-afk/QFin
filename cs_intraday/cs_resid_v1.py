"""
Validation-only cross-section of residuals, grouped by the external scale.

  r = B f + eps
  B is the loading already set before the day (last week, plus the daily
  deviation carried into the day). It is not forecast.
  The external scale is a train-fitted mix of the fast and slow energy of f,
  known before the day. A day is wide when that scale sits at or above the
  median of earlier days.

The signal is the residual over the last h days. The outcome is the residual
over the next h days. Horizons are the clocks already in the model: one day,
the fast window, and the quarter. Test bars are not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from combined_v1 import label_days, load_fields
from factor_v1 import (
    K,
    MIN_DAY_BARS,
    MIN_NAMES_FRAC,
    QUARTER_DAYS,
    RHO,
    align_sign,
    day_loadings,
    project,
    top_loadings,
)
from nested_v1 import FAST_WIN, SLOW_WIN, causal_window, fit_net, hold_weekly

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_resid_v1_out"
HORIZONS = (1, FAST_WIN, SLOW_WIN)


class ScaleMix(nn.Module):
    """One weight on the fast energy. Bias starts at 0, so the weight starts at 1/2."""

    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(1))

    def forward(self, fast, slow):
        alpha = torch.sigmoid(self.raw)
        return alpha * fast + (1.0 - alpha) * slow


def daily_last_close(close: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index = []
    for day, group in close.groupby(close.index.normalize(), sort=True):
        rows.append(group.iloc[::-1].bfill().iloc[0].to_numpy(np.float64))
        index.append(pd.Timestamp(day))
    return pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(index), columns=close.columns)


def walk_daily_residual(bar_returns: np.ndarray, times: pd.DatetimeIndex, daily_ret: pd.DataFrame):
    """Residual of each day's close-to-close cross-section on the loading known that morning."""
    dates = np.array(times.date)
    change = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    day_starts = change
    day_ends = np.append(change[1:], len(times))
    iso = pd.DatetimeIndex(times).isocalendar()
    week_key = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    ret_lookup = {ts.normalize(): daily_ret.loc[ts].to_numpy(np.float64) for ts in daily_ret.index}
    n_names = bar_returns.shape[1]
    b_slow = None
    delta = np.zeros((n_names, K))
    rows = []
    for d in range(len(day_starts)):
        a, b = int(day_starts[d]), int(day_ends[d])
        if b_slow is not None:
            loadings = b_slow + delta
            day = pd.Timestamp(times[a]).normalize()
            if day in ret_lookup:
                r_day_cs = ret_lookup[day]
                f, eps = project(r_day_cs, loadings)
                fitted = np.where(np.isfinite(eps), r_day_cs - eps, np.nan)
                rows.append(
                    {
                        "day": day,
                        "eps": eps,
                        "energy": float(np.nanmean(fitted**2)),
                    }
                )
            r_day = bar_returns[a:b]
            if np.isfinite(r_day).all():
                gram = loadings.T @ loadings + 1e-10 * np.eye(K)
                f_day = r_day @ np.linalg.solve(gram, loadings.T).T
            else:
                f_day = np.full((b - a, K), np.nan)
                for i, t_i in enumerate(range(a, b)):
                    f_i, _ = project(bar_returns[t_i], loadings)
                    f_day[i] = f_i
            realized = day_loadings(f_day, bar_returns[a:b])
            if realized is not None:
                upd = np.isfinite(realized).all(axis=1) & np.isfinite(b_slow).all(axis=1)
                delta[upd] = RHO * delta[upd] + (1.0 - RHO) * (realized[upd] - b_slow[upd])
        week_end = d + 1 == len(day_starts) or week_key[b - 1] != week_key[day_starts[d + 1]]
        if d + 1 >= QUARTER_DAYS and (d + 1 == QUARTER_DAYS or week_end):
            fresh_pair = top_loadings(bar_returns[int(day_starts[d + 1 - QUARTER_DAYS]) : b])
            if fresh_pair is not None:
                fresh, used = fresh_pair
                fresh = align_sign(fresh, b_slow)
                if b_slow is None:
                    delta[:] = 0.0
                else:
                    old_total = b_slow + delta
                    delta = old_total - fresh
                    first = used & (np.max(np.abs(old_total), axis=1) < 1e-8)
                    delta[first] = 0.0
                b_slow = fresh
    order = sorted(rows, key=lambda row: row["day"])
    index = pd.DatetimeIndex([row["day"] for row in order])
    eps = np.vstack([row["eps"] for row in order])
    energy = np.array([row["energy"] for row in order], np.float64)
    return index, eps, energy


def window_sum(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Sum values[start:stop] per name. Names missing too much of the window stay missing."""
    block = values[start:stop]
    finite = np.isfinite(block)
    total = np.where(finite, block, 0.0).sum(axis=0)
    count = finite.sum(axis=0)
    need = max(1, int(np.ceil(MIN_NAMES_FRAC * (stop - start))))
    total[count < need] = np.nan
    return total


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < max(MIN_DAY_BARS, int(MIN_NAMES_FRAC * len(x))):
        return np.nan
    xr = np.argsort(np.argsort(x[mask])).astype(np.float64)
    yr = np.argsort(np.argsort(y[mask])).astype(np.float64)
    xr -= xr.mean()
    yr -= yr.mean()
    den = np.sqrt((xr**2).sum() * (yr**2).sum())
    if den < 1e-12:
        return np.nan
    return float((xr * yr).sum() / den)


def median_spread(signal: np.ndarray, outcome: np.ndarray) -> float:
    mask = np.isfinite(signal) & np.isfinite(outcome)
    if mask.sum() < max(MIN_DAY_BARS, int(MIN_NAMES_FRAC * len(signal))):
        return np.nan
    sig = signal[mask]
    out = outcome[mask]
    med = np.median(sig)
    hi = out[sig >= med]
    lo = out[sig < med]
    if len(hi) == 0 or len(lo) == 0:
        return np.nan
    return float(hi.mean() - lo.mean())


def newey_west(values: np.ndarray, lag: int) -> tuple[int, float, float]:
    z = values[np.isfinite(values)]
    n = len(z)
    if n < 2:
        return n, float("nan"), float("nan")
    mu = float(z.mean())
    err = z - mu
    var = float(np.dot(err, err) / n)
    for ell in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - ell / (lag + 1.0)
        cov = float(np.dot(err[ell:], err[:-ell]) / n)
        var += 2.0 * weight * cov
    se = np.sqrt(var / n) if var > 0 else np.nan
    tstat = float(mu / se) if np.isfinite(se) and se > 0 else float("nan")
    return n, mu, tstat


def fit_scale(energy: np.ndarray, fast: np.ndarray, slow: np.ndarray, train: np.ndarray) -> tuple[float, int]:
    ready = train & np.isfinite(energy) & np.isfinite(fast) & np.isfinite(slow) & (fast > 0) & (slow > 0)
    rows = np.flatnonzero(ready)
    n_fit = int(len(rows) * 0.8)
    if n_fit < 30 or len(rows) - n_fit < 10:
        raise RuntimeError("not enough train days to fit the scale weight")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast_t = torch.tensor(fast[rows], dtype=torch.float32, device=device)
    slow_t = torch.tensor(slow[rows], dtype=torch.float32, device=device)
    energy_t = torch.tensor(energy[rows], dtype=torch.float32, device=device)
    model = ScaleMix().to(device)

    def loss_of(net, idx):
        var = net(fast_t[idx], slow_t[idx])
        log_ratio = torch.log(energy_t[idx] / var + 1e-8)
        abs_log = log_ratio.abs()
        huber = torch.where(abs_log < 1.0, 0.5 * log_ratio**2, abs_log - 0.5)
        return torch.mean(huber)

    epochs, _ = fit_net(model, loss_of, n_fit)
    alpha = float(torch.sigmoid(model.raw).detach().cpu())
    return alpha, epochs


def summarize(ic: np.ndarray, spread: np.ndarray, horizon: int) -> dict:
    n_ic, ic_mean, ic_t = newey_west(ic, horizon - 1)
    n_sp, sp_mean, sp_t = newey_west(spread, horizon - 1)
    sp_ok = spread[np.isfinite(spread)]
    return {
        "n": n_ic,
        "independent_windows": float(n_ic / horizon) if horizon else float("nan"),
        "ic": ic_mean,
        "ic_t": ic_t,
        "spread": sp_mean,
        "spread_bps": float(sp_mean * 1e4) if np.isfinite(sp_mean) else float("nan"),
        "spread_bps_per_day": float(sp_mean * 1e4 / horizon) if np.isfinite(sp_mean) else float("nan"),
        "spread_t": sp_t,
        "spread_positive_share": float(np.mean(sp_ok > 0)) if len(sp_ok) else float("nan"),
        "n_spread": n_sp,
    }


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_end = pd.Timestamp(meta["train_end"])
    val_start = pd.Timestamp(meta["val_eval_start"])
    val_end = pd.Timestamp(meta["val_end"])
    names = meta["names"]
    fields = load_fields(names, val_end)
    assert fields["close"].index.max() <= val_end

    close = fields["close"]
    close = close[close.index.normalize() < val_end.normalize()]
    prices = daily_last_close(close)
    daily_ret = prices.pct_change(fill_method=None)
    bar_returns = close.pct_change(fill_method=None).to_numpy(np.float64)
    print(f"[load] days {len(prices)} bars {len(close)} names {len(names)}", flush=True)

    index, eps, energy = walk_daily_residual(bar_returns, close.index, daily_ret.iloc[1:])
    labels = label_days(index, train_end, val_start)
    fast = causal_window(energy.reshape(-1, 1), FAST_WIN)[:, 0]
    slow = hold_weekly(causal_window(energy.reshape(-1, 1), SLOW_WIN), (index.isocalendar().year * 100 + index.isocalendar().week).to_numpy())[:, 0]
    alpha, epochs = fit_scale(energy, fast, slow, labels == "train")
    var = alpha * fast + (1.0 - alpha) * slow
    sigma = np.sqrt(np.where(var > 0, var, np.nan))
    print(f"[scale] alpha {alpha:.3f} epochs {epochs}", flush=True)

    min_names = max(MIN_DAY_BARS, int(MIN_NAMES_FRAC * eps.shape[1]))
    records = []
    paths = {h: [] for h in HORIZONS}
    for horizon in HORIZONS:
        for t in range(horizon, len(index) - horizon):
            if labels[t] != "val":
                continue
            past_scale = sigma[:t]
            past_scale = past_scale[np.isfinite(past_scale)]
            if len(past_scale) < SLOW_WIN or not np.isfinite(sigma[t]):
                continue
            signal = window_sum(eps, t - horizon, t)
            outcome = window_sum(eps, t, t + horizon)
            if np.isfinite(signal).sum() < min_names or np.isfinite(outcome).sum() < min_names:
                continue
            wide = bool(sigma[t] >= np.median(past_scale))
            ic = spearman(signal, outcome)
            spread = median_spread(signal, outcome)
            row = {"horizon": horizon, "day": index[t], "wide": wide, "ic": ic, "spread": spread}
            records.append(row)
            paths[horizon].append(row)

    frame = pd.DataFrame(records)
    buckets = {"all": np.ones(len(frame), dtype=bool), "narrow": ~frame["wide"].to_numpy(), "wide": frame["wide"].to_numpy()}
    metrics = {"alpha_fast": alpha, "scale_epochs": epochs, "horizons": list(HORIZONS), "by_horizon": {}}
    for horizon in HORIZONS:
        block = frame[frame["horizon"] == horizon]
        metrics["by_horizon"][str(horizon)] = {}
        for name, mask_all in buckets.items():
            part = frame.loc[mask_all & (frame["horizon"] == horizon)]
            metrics["by_horizon"][str(horizon)][name] = summarize(part["ic"].to_numpy(), part["spread"].to_numpy(), horizon)
        print(
            f"[h {horizon}] n {len(block)} wide {int(block['wide'].sum())} "
            f"ic all {metrics['by_horizon'][str(horizon)]['all']['ic']:.4f} "
            f"narrow {metrics['by_horizon'][str(horizon)]['narrow']['ic']:.4f} "
            f"wide {metrics['by_horizon'][str(horizon)]['wide']['ic']:.4f}",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    order = ("all", "narrow", "wide")
    legend = {"all": "全部", "narrow": "窄", "wide": "宽"}
    colors = {"all": "#8a8f98", "narrow": "#1f4e79", "wide": "#c45c26"}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.6), sharex=False)
    x = np.arange(len(HORIZONS))
    width = 0.24
    ax_ic, ax_sp = axes
    for i, name in enumerate(order):
        ic_vals = [metrics["by_horizon"][str(h)][name]["ic"] for h in HORIZONS]
        sp_vals = [metrics["by_horizon"][str(h)][name]["spread_bps_per_day"] for h in HORIZONS]
        ax_ic.bar(x + (i - 1) * width, ic_vals, width=width, color=colors[name], label=legend[name])
        ax_sp.bar(x + (i - 1) * width, sp_vals, width=width, color=colors[name], label=legend[name])
    ax_ic.axhline(0, color="#444444", lw=0.6)
    ax_sp.axhline(0, color="#444444", lw=0.6)
    tick = [f"{h} 日" for h in HORIZONS]
    ax_ic.set_xticks(x, tick)
    ax_sp.set_xticks(x, tick)
    ax_ic.set_ylabel("残差秩相关")
    ax_sp.set_ylabel("截面差（每日基点）")
    ax_ic.set_title("未来残差与刚结束的残差")
    ax_sp.set_title("高于当日中位的名字，减去低于中位的名字")
    ax_ic.legend(frameon=False, ncol=3)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("验证集：按开盘前的外部尺度，看残差的截面差别", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cs_resid_val.png", dpi=140)

    fig2, ax = plt.subplots(figsize=(10.2, 4.2))
    path = pd.DataFrame(paths[1])
    for name, mask in (("窄", ~path["wide"]), ("宽", path["wide"])):
        piece = path.loc[mask].sort_values("day")
        ax.plot(piece["day"], piece["spread"].fillna(0).cumsum() * 1e4, color=colors["narrow" if name == "窄" else "wide"], label=name, lw=1.2)
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_ylabel("累计截面差（基点）")
    ax.set_xlabel("验证集。持有下一个交易日。尺度在开盘前已知")
    ax.set_title("1 日残差截面差的累计路径")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "cs_resid_path.png", dpi=140)
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
