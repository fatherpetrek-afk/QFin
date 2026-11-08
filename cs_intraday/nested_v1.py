"""
Train the nested fast/slow model and walk it through validation.

Slow clock: weekly loadings, and a 63-day scale held through the week.
Fast clock: daily deviation of the loadings, and a 5-day scale.
The predictive distribution of the next day is the past shape, stretched by
the nested scale. Test bars are not loaded. Weights stop on late train.
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

from combined_v1 import (
    CHANNELS,
    LABELS,
    day_factor,
    label_days,
    load_fields,
    scale_free_channels,
)
from factor_v1 import K, QUARTER_DAYS, RHO, align_sign, day_loadings, project, top_loadings
from mgf_channels import align_to_market, extract_factor, prepare_calendar

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "nested_v1_out"
FAST_WIN = 5
SLOW_WIN = 63


def walk_loadings(values: np.ndarray, times: pd.DatetimeIndex):
    """Causal close-channel loadings. Returns f1 and the weekly/daily states."""
    n_times, n_names = values.shape
    dates = np.array(times.date)
    change = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    day_starts = change
    day_ends = np.append(change[1:], n_times)
    iso = pd.DatetimeIndex(times).isocalendar()
    week_key = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    f_all = np.full((n_times, K), np.nan)
    b_slow = None
    delta = np.zeros((n_names, K))
    weeks = []
    days = []
    week_energy = []
    for d in range(len(day_starts)):
        a, b = int(day_starts[d]), int(day_ends[d])
        if b_slow is not None:
            loadings = b_slow + delta
            delta_start = delta.copy()
            r_day = values[a:b]
            if np.isfinite(r_day).all():
                gram = loadings.T @ loadings + 1e-10 * np.eye(K)
                proj = np.linalg.solve(gram, loadings.T)
                f_day = r_day @ proj.T
                f_all[a:b] = f_day
            else:
                f_day = np.full((b - a, K), np.nan)
                for i, t_i in enumerate(range(a, b)):
                    f_i, _ = project(values[t_i], loadings)
                    f_day[i] = f_i
                    f_all[t_i] = f_i
            energy = float(np.nanmean(f_day**2)) if np.isfinite(f_day).any() else np.nan
            realized = day_loadings(f_all[a:b], values[a:b])
            if realized is not None:
                upd = np.isfinite(realized).all(axis=1) & np.isfinite(b_slow).all(axis=1)
                delta[upd] = RHO * delta[upd] + (1.0 - RHO) * (realized[upd] - b_slow[upd])
            if np.isfinite(energy):
                days.append(
                    {
                        "time": pd.Timestamp(times[b - 1]),
                        "delta0": delta_start,
                        "energy": energy,
                        "delta1": delta.copy(),
                    }
                )
                week_energy.append(energy)
        week_end = d + 1 == len(day_starts) or week_key[b - 1] != week_key[day_starts[d + 1]]
        if d + 1 >= QUARTER_DAYS and (d + 1 == QUARTER_DAYS or week_end):
            w0 = int(day_starts[d + 1 - QUARTER_DAYS])
            fresh_pair = top_loadings(values[w0:b])
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
                weeks.append(
                    {
                        "time": pd.Timestamp(times[b - 1]),
                        "B": b_slow.copy(),
                        "energy": float(np.mean(week_energy)) if week_energy else 0.0,
                    }
                )
                week_energy = []
    return f_all[:, 0], weeks, days


class SlowMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, K * K))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, B, s):
        A = self.net(s).view(-1, K, K) + torch.eye(K, device=B.device)
        return torch.matmul(B, A)


class FastMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(np.log(RHO / (1.0 - RHO))))

    def forward(self, delta, s):
        rho = torch.sigmoid(self.net(s))
        return delta * rho.unsqueeze(-1)


class ScaleNest(nn.Module):
    """Mix a 5-day scale and a 63-day scale. Output is a weight on the fast scale."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 8), nn.Tanh(), nn.Linear(8, 5))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(np.log(0.35 / 0.65)))

    def forward(self, features):
        return torch.sigmoid(self.net(features))


def fit_net(model, loss_of, n_fit: int, patience: int = 40):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        best_loss = float(loss_of(model, slice(n_fit, None)))
    wait = 0
    used = 0
    for _ in range(400):
        model.train()
        opt.zero_grad()
        loss = loss_of(model, slice(0, n_fit))
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            held = float(loss_of(model, slice(n_fit, None)))
        used += 1
        if held < best_loss - 1e-10:
            best_loss = held
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best)
    return used, best_loss


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def causal_window(x2: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(x2)
    x = np.where(finite, x2, 0.0)
    cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    cc = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(finite, axis=0)])
    out = np.full_like(x2, np.nan)
    for t in range(window, len(x2)):
        cn = cc[t] - cc[t - window]
        out[t] = np.where(cn >= window * 0.8, (cs[t] - cs[t - window]) / np.maximum(cn, 1.0), np.nan)
    return out


def hold_weekly(values: np.ndarray, week_id: np.ndarray) -> np.ndarray:
    """Slow scale changes only at the week boundary, and only using the past."""
    out = np.full_like(values, np.nan)
    held = np.full(values.shape[1], np.nan)
    prev = week_id[0]
    for t in range(len(values)):
        if week_id[t] != prev:
            if np.isfinite(values[t]).all():
                held = values[t].copy()
            prev = week_id[t]
        if np.isfinite(held).all():
            out[t] = held
    return out


def static_bands(f, val, q_lo=0.10, q_hi=0.90):
    """Quantiles of earlier raw factors. This is the band that barely moves."""
    lo = np.full_like(f, np.nan)
    hi = np.full_like(f, np.nan)
    hist = []
    for t in range(len(f)):
        if len(hist) >= 50 and val[t] and np.isfinite(f[t]).all():
            past = np.stack(hist)
            lo[t] = np.quantile(past, q_lo, axis=0)
            hi[t] = np.quantile(past, q_hi, axis=0)
        if np.isfinite(f[t]).all():
            hist.append(f[t].copy())
    return lo, hi


def bands(f, var, val, q_lo=0.10, q_hi=0.90):
    """Walk-forward band of f, width taken from var known before the day."""
    n = len(f)
    lo = np.full_like(f, np.nan)
    hi = np.full_like(f, np.nan)
    z_hist = []
    for t in range(n):
        if not (np.isfinite(var[t]).all() and var[t].min() > 0):
            continue
        scale = np.sqrt(var[t])
        if len(z_hist) >= 50 and val[t] and np.isfinite(f[t]).all():
            past = np.stack(z_hist)
            lo[t] = np.quantile(past, q_lo, axis=0) * scale
            hi[t] = np.quantile(past, q_hi, axis=0) * scale
        if np.isfinite(f[t]).all():
            z_hist.append(f[t] / scale)
    return lo, hi


def hit_rate(f, lo, hi, val):
    m = val & np.isfinite(lo).all(axis=1) & np.isfinite(f).all(axis=1)
    inside = (f[m] >= lo[m]) & (f[m] <= hi[m])
    width = np.mean(hi[m] - lo[m], axis=0)
    return m, inside.mean(axis=0), width, int(m.sum())


def main() -> None:
    print("预计：抽因子大约 1 分钟，4060 上训练嵌套网络不到 10 秒。", flush=True)
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_end = pd.Timestamp(meta["train_end"])
    val_start = pd.Timestamp(meta["val_eval_start"])
    val_end = pd.Timestamp(meta["val_end"])
    names = meta["names"]
    t_all = time.perf_counter()
    fields = load_fields(names, val_end)
    panels = scale_free_channels(fields)
    index = panels["close"].index
    assert index.max() <= val_end
    prepare_calendar(index)
    train_bars = np.asarray(index <= train_end)

    print("[loadings] close channel", flush=True)
    f_close, weeks, day_rows = walk_loadings(panels["close"].to_numpy(np.float64), index)
    f_close = align_to_market(f_close, np.nanmean(panels["close"].to_numpy(np.float64), axis=1), train_bars)
    daily = [day_factor(f_close, index).rename(columns={"f": "close"})[["close", "n"]]]
    for name in CHANNELS:
        if name == "close":
            continue
        print(f"[factor] {name}", flush=True)
        values = panels[name].to_numpy(np.float64)
        f1 = align_to_market(extract_factor(values)[:, 0], np.nanmean(values, axis=1), train_bars)
        daily.append(day_factor(f1, index).rename(columns={"f": name})[[name]])
    days = pd.concat(daily, axis=1).sort_index().dropna()
    days = days[(days["n"] >= 60) & (days.index.normalize() < val_end.normalize())]
    labels = label_days(days.index, train_end, val_start)
    f = days[list(CHANNELS)].to_numpy(np.float64)
    x2 = f**2
    fast_var = causal_window(x2, FAST_WIN)
    slow_raw = causal_window(x2, SLOW_WIN)
    iso = days.index.isocalendar()
    week_id = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    slow_var = hold_weekly(slow_raw, week_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu = time.perf_counter()

    # Nested loadings.
    def kind_of(ts):
        if ts <= train_end:
            return "train"
        if ts < val_start:
            return "embargo"
        return "val"

    slow_rows = []
    for prev, nxt in zip(weeks, weeks[1:]):
        kind = kind_of(nxt["time"])
        if kind == "embargo":
            continue
        slow_rows.append((kind, prev["B"], np.array([prev["energy"], np.linalg.norm(prev["B"])], np.float32), nxt["B"]))
    fast_rows = []
    prev_energy = None
    for row in day_rows:
        kind = kind_of(row["time"])
        if kind == "embargo" or prev_energy is None:
            prev_energy = row["energy"]
            continue
        fast_rows.append((kind, row["delta0"], np.array([prev_energy, np.linalg.norm(row["delta0"])], np.float32), row["delta1"]))
        prev_energy = row["energy"]

    def stack(rows, kind):
        picked = [r for r in rows if r[0] == kind]
        return picked, np.stack([r[1] for r in picked]), np.stack([r[2] for r in picked]), np.stack([r[3] for r in picked])

    slow_tr, Btr, Str, Ytr = stack(slow_rows, "train")
    slow_va, Bva, Sva, Yva = stack(slow_rows, "val")
    n_fit = int(len(slow_tr) * 0.8)
    mu, sd = Str[:n_fit].mean(0), Str[:n_fit].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Str = (Str - mu) / sd
    Sva = (Sva - mu) / sd
    Bt = torch.tensor(Btr, dtype=torch.float32, device=device)
    St = torch.tensor(Str, dtype=torch.float32, device=device)
    Yt = torch.tensor(Ytr, dtype=torch.float32, device=device)
    slow_net = SlowMap().to(device)

    def slow_loss(model, idx):
        return torch.mean((model(Bt[idx], St[idx]) - Yt[idx]) ** 2)

    torch.manual_seed(0)
    ep_s, _ = fit_net(slow_net, slow_loss, n_fit)
    with torch.no_grad():
        pred_b = slow_net(torch.tensor(Bva, dtype=torch.float32, device=device), torch.tensor(Sva, dtype=torch.float32, device=device)).cpu().numpy()
    slow_ratio = mse(pred_b, Yva) / mse(Bva, Yva)

    fast_tr, D0, Sf, D1 = stack(fast_rows, "train")
    fast_va, D0v, Sfv, D1v = stack(fast_rows, "val")
    n_fit_f = int(len(fast_tr) * 0.8)
    mu_f, sd_f = Sf[:n_fit_f].mean(0), Sf[:n_fit_f].std(0)
    sd_f = np.where(sd_f < 1e-8, 1.0, sd_f)
    Sf = (Sf - mu_f) / sd_f
    Sfv = (Sfv - mu_f) / sd_f
    D0t = torch.tensor(D0, dtype=torch.float32, device=device)
    Sft = torch.tensor(Sf, dtype=torch.float32, device=device)
    D1t = torch.tensor(D1, dtype=torch.float32, device=device)
    fast_net = FastMap().to(device)

    def fast_loss(model, idx):
        return torch.mean((model(D0t[idx], Sft[idx]) - D1t[idx]) ** 2)

    torch.manual_seed(0)
    ep_f, _ = fit_net(fast_net, fast_loss, n_fit_f)
    with torch.no_grad():
        pred_d = fast_net(
            torch.tensor(D0v, dtype=torch.float32, device=device),
            torch.tensor(Sfv, dtype=torch.float32, device=device),
        ).cpu().numpy()
    fast_ratio = mse(pred_d, D1v) / mse(D0v, D1v)
    shrink_ratio = mse(RHO * D0v, D1v) / mse(D0v, D1v)

    # Nested scale for the distribution.
    ready = np.isfinite(fast_var).all(1) & np.isfinite(slow_var).all(1) & np.isfinite(f).all(1)
    feat = np.concatenate([np.log(fast_var + 1e-18), np.log(slow_var + 1e-18)], axis=1)
    train_rows = np.flatnonzero(ready & (labels == "train"))
    n_fit_a = int(len(train_rows) * 0.8)
    order = np.concatenate([train_rows[:n_fit_a], train_rows[n_fit_a:]])
    feat_t = torch.tensor(feat[order], dtype=torch.float32, device=device)
    fast_t = torch.tensor(fast_var[order], dtype=torch.float32, device=device)
    slow_t = torch.tensor(slow_var[order], dtype=torch.float32, device=device)
    f_t = torch.tensor(f[order], dtype=torch.float32, device=device)
    scale_net = ScaleNest().to(device)

    def scale_loss(model, idx):
        alpha = model(feat_t[idx])
        var = alpha * fast_t[idx] + (1.0 - alpha) * slow_t[idx]
        log_z2 = torch.log(f_t[idx] ** 2 / var + 1e-8)
        abs_log = log_z2.abs()
        huber = torch.where(abs_log < 1.0, 0.5 * log_z2**2, abs_log - 0.5)
        return torch.mean(huber)

    torch.manual_seed(0)
    ep_a, _ = fit_net(scale_net, scale_loss, n_fit_a)
    with torch.no_grad():
        alpha_all = np.full((len(f), 5), np.nan)
        okf = np.isfinite(feat).all(1)
        alpha_all[okf] = scale_net(torch.tensor(feat[okf], dtype=torch.float32, device=device)).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_sec = time.perf_counter() - t_gpu
    var_dyn = alpha_all * fast_var + (1.0 - alpha_all) * slow_var
    val = labels == "val"
    lo_d, hi_d = bands(f, var_dyn, val)
    lo_s, hi_s = static_bands(f, val)
    m, hit_d, width_d, n_val = hit_rate(f, lo_d, hi_d, val)
    _, hit_s, width_s, _ = hit_rate(f, lo_s, hi_s, val)
    alpha_val = np.nanmean(alpha_all[val], axis=0)

    print(f"[time] nested training {gpu_sec:.2f}s on {device}", flush=True)
    print(f"[loadings] slow mse/persistence {slow_ratio:.3f} in {ep_s} epochs", flush=True)
    print(f"[loadings] fast mse/persistence {fast_ratio:.3f} shrink/persistence {shrink_ratio:.3f} in {ep_f} epochs", flush=True)
    print(f"[scale] epochs {ep_a} val alpha {np.round(alpha_val, 3)}", flush=True)
    for c, name in enumerate(CHANNELS):
        print(f"[band {name}] dynamic {hit_d[c]:.3f} static {hit_s[c]:.3f} width {width_d[c]:.4f} vs {width_s[c]:.4f}", flush=True)

    levels = (0.50, 0.80, 0.90, 0.95)
    coverage = {}
    ribbon = {}
    for nominal in levels:
        tail = (1.0 - nominal) / 2.0
        lo_c, hi_c = bands(f, var_dyn, val, tail, 1.0 - tail)
        lo_sc, hi_sc = static_bands(f, val, tail, 1.0 - tail)
        _, hit_c, width_c, n_c = hit_rate(f, lo_c, hi_c, val)
        _, hit_sc, width_sc, _ = hit_rate(f, lo_sc, hi_sc, val)
        coverage[f"{nominal:.2f}"] = {
            "nominal": nominal,
            "n_days": n_c,
            "dynamic_hit_rate": {name: float(hit_c[c]) for c, name in enumerate(CHANNELS)},
            "static_hit_rate": {name: float(hit_sc[c]) for c, name in enumerate(CHANNELS)},
            "dynamic_mean_width": {name: float(width_c[c]) for c, name in enumerate(CHANNELS)},
            "static_mean_width": {name: float(width_sc[c]) for c, name in enumerate(CHANNELS)},
        }
        dyn_txt = " ".join(f"{name} {hit_c[c]:.3f}" for c, name in enumerate(CHANNELS))
        sta_txt = " ".join(f"{name} {hit_sc[c]:.3f}" for c, name in enumerate(CHANNELS))
        print(f"[coverage {nominal:.0%}] dynamic {dyn_txt}", flush=True)
        print(f"[coverage {nominal:.0%}] static  {sta_txt}", flush=True)
        if nominal in (0.50, 0.80, 0.95):
            ribbon[nominal] = (lo_c, hi_c)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "expected": "factor extraction about 1 minute; nested training under 10 seconds on the 4060",
        "gpu_seconds": gpu_sec,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "slow_epochs": ep_s,
        "fast_epochs": ep_f,
        "scale_epochs": ep_a,
        "slow_mse_ratio_vs_persistence": slow_ratio,
        "fast_mse_ratio_vs_persistence": fast_ratio,
        "fast_shrink_ratio_vs_persistence": shrink_ratio,
        "val_days": n_val,
        "alpha_fast_weight": {name: float(alpha_val[c]) for c, name in enumerate(CHANNELS)},
        "dynamic_hit_rate": {name: float(hit_d[c]) for c, name in enumerate(CHANNELS)},
        "static_hit_rate": {name: float(hit_s[c]) for c, name in enumerate(CHANNELS)},
        "dynamic_mean_width": {name: float(width_d[c]) for c, name in enumerate(CHANNELS)},
        "static_mean_width": {name: float(width_s[c]) for c, name in enumerate(CHANNELS)},
        "coverage_by_confidence": coverage,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    when = days.index[m]
    realized = f[m]
    lo80, hi80 = ribbon[0.80]
    inside80 = (realized >= lo80[m]) & (realized <= hi80[m])
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(5, 1, figsize=(11.4, 11.0), sharex=True)
    shades = (
        (0.95, "#dbe4f0", "预测 95%"),
        (0.80, "#b5c7e0", "预测 80%"),
        (0.50, "#7f9dbe", "预测 50%"),
    )
    for c, (ax, name) in enumerate(zip(axes, CHANNELS)):
        for nominal, color, label in shades:
            lo_c, hi_c = ribbon[nominal]
            ax.fill_between(when, lo_c[m, c], hi_c[m, c], color=color, label=label, zorder=1, linewidth=0)
        ax.plot(when, realized[:, c], color="#1f4e79", lw=1.05, label="真实", zorder=3)
        out = ~inside80[:, c]
        ax.scatter(when[out], realized[out, c], s=18, color="#c45c26", zorder=4, label="真实落在预测 80% 外")
        ax.set_ylabel(f"{LABELS[name]}\n预测 vs 真实")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.99,
            0.90,
            f"预测 80% 盖住真实 {int(inside80[:, c].sum())}/{n_val} 天",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        if c == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=5)
    axes[-1].set_xlabel("验证集交易日。色带是开盘前的预测区间，线是当天真实值")
    fig.suptitle("预测 vs 真实", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nested_val.png", dpi=140)
    print(f"[time] total {time.perf_counter() - t_all:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
