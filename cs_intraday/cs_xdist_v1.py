"""
Predictive distribution of the cross-section, validation only.

After the common move B f is removed, each name has a leftover.
A small net, trained on train days and stopped on late train, maps where
that name sat in yesterday's cross-section into a location and a scale.
The shape of the distribution is the empirical cloud of late-train
standardized leftovers. Test bars are not loaded. No profit is computed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from combined_v1 import label_days, load_fields
from cs_resid_v1 import daily_last_close, fit_scale, walk_daily_residual
from factor_v1 import MIN_DAY_BARS, MIN_NAMES_FRAC
from nested_v1 import FAST_WIN, SLOW_WIN, causal_window, hold_weekly

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_xdist_v1_out"
SHAPE_Q = np.array([0.025, 0.10, 0.25, 0.75, 0.90, 0.975])


class CrossDist(nn.Module):
    """Location and scale of the next leftover. Last layer starts at zero."""

    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 16), nn.Tanh(), nn.Linear(16, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, sigma_own):
        raw = self.net(features)
        shift = raw[..., 0]
        log_mult = raw[..., 1].clamp(-2.0, 2.0)
        sigma = (sigma_own * torch.exp(log_mult)).clamp_min(1e-6)
        mu = sigma_own * shift
        return mu, sigma


def cross_section_position(row: np.ndarray, min_names: int) -> np.ndarray:
    """Percentile in [-1, 1]. Uses only this row."""
    out = np.full(row.shape, np.nan)
    mask = np.isfinite(row)
    if mask.sum() < min_names:
        return out
    order = np.argsort(np.argsort(row[mask])).astype(np.float64)
    pct = order / max(mask.sum() - 1, 1)
    out[mask] = 2.0 * pct - 1.0
    return out


def fit_cross(features, sigma_own, target, valid, train_days, device):
    """features/sigma/target/valid are day x name. train_days indexes days, in time order."""
    n_fit = int(len(train_days) * 0.8)
    if n_fit < 40 or len(train_days) - n_fit < 15:
        raise RuntimeError("not enough train days for the cross-section distribution")
    fit_days = train_days[:n_fit]
    block = features[fit_days]
    mask = valid[fit_days]
    mu_f = np.zeros(block.shape[-1])
    sd_f = np.ones(block.shape[-1])
    for j in range(block.shape[-1]):
        col = block[..., j][mask]
        if len(col) == 0:
            continue
        mu_f[j] = float(np.mean(col))
        sd = float(np.std(col))
        sd_f[j] = sd if sd > 1e-8 else 1.0
    normed = (features - mu_f) / sd_f
    normed = np.where(valid[..., None], normed, 0.0)
    sigma_own = np.where(valid, sigma_own, 1.0)
    target = np.where(valid, target, 0.0)
    feat_t = torch.tensor(normed, dtype=torch.float32, device=device)
    sig_t = torch.tensor(sigma_own, dtype=torch.float32, device=device)
    y_t = torch.tensor(target, dtype=torch.float32, device=device)
    m_t = torch.tensor(valid, dtype=torch.bool, device=device)
    model = CrossDist(features.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loss_of(net, days):
        mu, sigma = net(feat_t[days], sig_t[days])
        mask = m_t[days]
        if int(mask.sum()) < 10:
            return torch.zeros((), device=device)
        z = (y_t[days] - mu) / sigma
        return torch.mean((0.5 * z**2 + torch.log(sigma))[mask])

    model.eval()
    with torch.no_grad():
        best_loss = float(loss_of(model, train_days[n_fit:]))
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    wait = 0
    used = 0
    for _ in range(400):
        model.train()
        opt.zero_grad()
        loss = loss_of(model, train_days[:n_fit])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            held = float(loss_of(model, train_days[n_fit:]))
        used += 1
        if held < best_loss - 1e-8:
            best_loss = held
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 40:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        mu, sigma = model(feat_t, sig_t)
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
    hold = np.zeros(len(target), dtype=bool)
    hold[train_days[n_fit:]] = True
    z_hold = (target - mu_np) / sigma_np
    z_hold = z_hold[hold[:, None] & valid & np.isfinite(z_hold)]
    shape = np.quantile(z_hold, SHAPE_Q)
    return model, mu_f, sd_f, mu_np, sigma_np, shape, used


def coverage(y, lo, hi, mask):
    ok = mask & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if ok.sum() == 0:
        return float("nan"), 0
    inside = (y[ok] >= lo[ok]) & (y[ok] <= hi[ok])
    return float(inside.mean()), int(ok.sum())


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa3ad")
    ax.spines["bottom"].set_color("#9aa3ad")
    ax.tick_params(colors="#3d4652")
    ax.grid(axis="y", color="#e6ebf0", lw=0.6)
    ax.set_axisbelow(True)


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
    index, eps, energy = walk_daily_residual(bar_returns, close.index, daily_ret.iloc[1:])
    labels = label_days(index, train_end, val_start)
    min_names = max(MIN_DAY_BARS, int(MIN_NAMES_FRAC * eps.shape[1]))

    ext_fast = causal_window(energy.reshape(-1, 1), FAST_WIN)[:, 0]
    ext_slow = hold_weekly(
        causal_window(energy.reshape(-1, 1), SLOW_WIN),
        (index.isocalendar().year * 100 + index.isocalendar().week).to_numpy(),
    )[:, 0]
    alpha, scale_epochs = fit_scale(energy, ext_fast, ext_slow, labels == "train")
    ext_sigma = np.sqrt(np.maximum(alpha * ext_fast + (1.0 - alpha) * ext_slow, 0.0))

    fast_var = causal_window(eps**2, FAST_WIN)
    slow_var = hold_weekly(
        causal_window(eps**2, SLOW_WIN),
        (index.isocalendar().year * 100 + index.isocalendar().week).to_numpy(),
    )
    position = np.vstack([cross_section_position(eps[t - 1], min_names) for t in range(1, len(index))])
    # Row t in `position` describes day index t+1, using the previous day's cross-section.
    n = len(index) - 1
    features = np.full((n, eps.shape[1], 4), np.nan)
    sigma_own = np.sqrt(np.maximum(fast_var[1:], 0.0))
    features[..., 0] = position
    features[..., 1] = np.log(fast_var[1:] + 1e-18)
    features[..., 2] = np.log(slow_var[1:] + 1e-18)
    features[..., 3] = np.log(ext_sigma[1:, None] + 1e-18)
    target = eps[1:]
    day_index = index[1:]
    day_labels = labels[1:]
    valid = np.isfinite(features).all(axis=2) & np.isfinite(sigma_own) & (sigma_own > 0) & np.isfinite(target)
    enough = valid.sum(axis=1) >= min_names
    train_days = np.flatnonzero((day_labels == "train") & enough)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat_mu, feat_sd, mu, sigma, shape, epochs = fit_cross(features, sigma_own, target, valid, train_days, device)
    print(f"[train] epochs {epochs} shape {np.round(shape, 3)} device {device}", flush=True)

    bands = mu[..., None] + sigma[..., None] * shape
    # shape order: 2.5, 10, 25, 75, 90, 97.5
    val = (day_labels == "val") & enough & np.isfinite(ext_sigma[1:])
    high = position >= 0
    low = position < 0
    levels = {"50": (2, 3), "80": (1, 4), "95": (0, 5)}
    metrics = {
        "scale_alpha_fast": alpha,
        "scale_epochs": scale_epochs,
        "cross_epochs": epochs,
        "shape_quantiles": {str(q): float(v) for q, v in zip(SHAPE_Q, shape)},
        "coverage": {},
    }
    for name, group in (("all", np.ones_like(high)), ("higher_yesterday", high), ("lower_yesterday", low)):
        metrics["coverage"][name] = {}
        for level, (a, b) in levels.items():
            hit, count = coverage(target, bands[..., a], bands[..., b], val[:, None] & valid & group)
            metrics["coverage"][name][level] = {"hit_rate": hit, "n": count}
            print(f"[cover {name} {level}] {hit:.3f} n {count}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    shades = (
        (0, 5, "#e4edf6", "预测 95%"),
        (1, 4, "#b9cfe6", "预测 80%"),
        (2, 3, "#7ea4cb", "预测 50%"),
    )
    when = day_index[val]
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.8), sharex=True)
    panels = (
        (high, "昨天在池子里偏高的股票", "higher_yesterday", axes[0]),
        (low, "昨天在池子里偏低的股票", "lower_yesterday", axes[1]),
    )
    y_span = []
    drawn = []
    for group, title, key, ax in panels:
        use = val[:, None] & valid & group
        med = np.full(val.sum(), np.nan)
        real_lo = np.full(val.sum(), np.nan)
        real_hi = np.full(val.sum(), np.nan)
        pred = {}
        for a, b, color, label in shades:
            lo_d = np.full(val.sum(), np.nan)
            hi_d = np.full(val.sum(), np.nan)
            for i, t in enumerate(np.flatnonzero(val)):
                m = use[t]
                if m.sum() < 5:
                    continue
                lo_d[i] = np.median(bands[t, m, a])
                hi_d[i] = np.median(bands[t, m, b])
                if a == 0:
                    realized = target[t, m]
                    med[i] = np.median(realized)
                    real_lo[i] = np.quantile(realized, 0.10)
                    real_hi[i] = np.quantile(realized, 0.90)
            pred[(a, b)] = (lo_d, hi_d, color, label)
        for (a, b), (lo_d, hi_d, color, label) in pred.items():
            ax.fill_between(when, lo_d * 100, hi_d * 100, color=color, lw=0, label=label, zorder=1)
        ax.plot(when, real_lo * 100, color="#d06a3a", lw=0.9, ls="--", label="真实的 10% / 90%", zorder=3)
        ax.plot(when, real_hi * 100, color="#d06a3a", lw=0.9, ls="--", zorder=3)
        ax.plot(when, med * 100, color="#16324f", lw=1.25, label="真实中位", zorder=4)
        hit80 = metrics["coverage"][key]["80"]["hit_rate"]
        hit50 = metrics["coverage"][key]["50"]["hit_rate"]
        ax.text(
            0.995,
            0.06,
            f"按每只股票自己的区间，50% 盖住 {hit50:.0%}，80% 盖住 {hit80:.0%}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#3d4652",
        )
        ax.set_ylabel("剩下的涨跌（%）")
        ax.set_title(title, loc="left", color="#1c2430")
        style_axes(ax)
        finite_edge = np.concatenate([pred[(0, 5)][0], pred[(0, 5)][1], real_lo, real_hi])
        finite_edge = finite_edge[np.isfinite(finite_edge)] * 100
        y_span.append(np.quantile(finite_edge, [0.02, 0.98]))
        drawn.append(ax)
    ylim = (min(s[0] for s in y_span), max(s[1] for s in y_span))
    pad = 0.12 * (ylim[1] - ylim[0])
    for ax in drawn:
        ax.set_ylim(ylim[0] - pad, ylim[1] + pad)
    axes[0].legend(frameon=False, ncol=5, loc="upper left", fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("验证集交易日。色带是开盘前的预测分布，虚线是当天这组股票真实落点的 10% 和 90%")
    fig.suptitle("预测分布 vs 真实", fontsize=16, color="#1c2430")
    fig.text(0.5, 0.01, "去掉整个池子一起动的部分之后。图上看主体，不把个别极端点撑满整张图。不是收益。", ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT_DIR / "cs_xdist_val.png", dpi=160)
    plt.close(fig)

    # One picture of the whole cross-section: predicted distribution by yesterday's position.
    pct = (position + 1.0) / 2.0
    val_m = val[:, None] & valid
    x = pct[val_m]
    y = target[val_m] * 100
    pos_grid = np.linspace(-1.0, 1.0, 81)
    raw = np.zeros((len(pos_grid), 4))
    raw[:, 0] = pos_grid
    for j in range(1, 4):
        raw[:, j] = np.nanmedian(features[..., j][val_m])
    grid = torch.tensor((raw - feat_mu) / feat_sd, dtype=torch.float32, device=device)
    base = torch.full((len(pos_grid),), float(np.nanmedian(sigma_own[val_m])), dtype=torch.float32, device=device)
    with torch.no_grad():
        mu_g, sig_g = model(grid, base)
    mu_g = mu_g.cpu().numpy() * 100
    sig_g = sig_g.cpu().numpy() * 100
    smooth = mu_g[:, None] + sig_g[:, None] * shape
    x_grid = (pos_grid + 1.0) / 2.0 * 100
    half = float(np.nanmax(np.abs(smooth[:, [0, 5]])))
    ylim = max(half * 1.25, 1.0)
    fig2, ax = plt.subplots(figsize=(11.2, 6.2))
    shown = (y >= -ylim) & (y <= ylim)
    hb = ax.hist2d(
        x[shown] * 100,
        y[shown],
        bins=[40, 48],
        range=[[0, 100], [-ylim, ylim]],
        cmap="Blues",
        cmin=1,
        norm=matplotlib.colors.PowerNorm(gamma=0.45),
    )
    edge_colors = {0: "#9bb4cc", 1: "#5d86b3", 2: "#1f4e79"}
    for a, b, color, label in shades:
        ax.plot(x_grid, smooth[:, a], color=edge_colors[a], lw=1.15, label=label, zorder=3)
        ax.plot(x_grid, smooth[:, b], color=edge_colors[a], lw=1.15, zorder=3)
    ax.plot(x_grid, 0.5 * (smooth[:, 2] + smooth[:, 3]), color="#16324f", lw=1.5, label="预测中位", zorder=4)
    style_axes(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("昨天在池子里的位置（0 最低，100 最高）")
    ax.set_ylabel("今天剩下的涨跌（%）")
    ax.set_title("预测分布 vs 真实", loc="left", color="#1c2430")
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=8)
    cbar = fig2.colorbar(hb[3], ax=ax, pad=0.015, fraction=0.046)
    cbar.outline.set_visible(False)
    cbar.set_label("真实结果有多密", color="#3d4652")
    cbar.ax.tick_params(colors="#3d4652", labelsize=8)
    fig2.text(0.5, 0.01, "验证集。深色是真实落点较密的地方，色带是开盘前的预测分布。不是收益。", ha="center", fontsize=9, color="#6b7380")
    fig2.tight_layout(rect=(0, 0.04, 1, 1))
    fig2.savefig(OUT_DIR / "cs_xdist_rank.png", dpi=160)
    plt.close(fig2)
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
