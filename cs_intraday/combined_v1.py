"""
Combined model, validation walk-forward only. Test bars are not loaded.

The forecast is a distribution of the next day's pool state, not a price.
Five channels, all free of the price level:
  open, high, low, close: move versus the previous close
  volume: change in log volume

Each channel is reduced to its causal external factor, then divided by a
scale known before the day. The predictive distribution is that scale times
the empirical shape of earlier days. The daily scale is a causal exponential
average known before the day. A small slow network on the 4060 adjusts how
the five channels move together. Its weights are fit on the early train and
stopped on the late train. Validation is scored only after they are frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from mgf_channels import align_to_market, extract_factor, prepare_calendar

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "5m"
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "combined_v1_out"

CHANNELS = ("open", "high", "low", "close", "volume")
LABELS = {"open": "开盘", "high": "最高", "low": "最低", "close": "收盘", "volume": "成交量"}
DAY_HALFLIFE = 20
SLOW_WINDOW = 21
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_fields(names: list[str], val_end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    frames = []
    for name in names:
        df = pd.read_parquet(
            DATA_DIR / f"{name}.parquet",
            columns=["time", "open", "high", "low", "close", "volume"],
            filters=[("time", "<=", val_end.to_datetime64())],
        )
        df["time"] = pd.to_datetime(df["time"])
        df = df[df["time"] <= val_end]
        frames.append(df.drop_duplicates("time").set_index("time").sort_index())
    index = frames[0].index
    for frame in frames[1:]:
        index = index.union(frame.index)
    index = index.sort_values()
    index = index[index <= val_end]
    fields = {}
    for field in ("open", "high", "low", "close", "volume"):
        cols = {name: frame[field].reindex(index) for name, frame in zip(names, frames)}
        fields[field] = pd.DataFrame(cols, index=index)
    return fields


def scale_free_channels(fields: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prev = fields["close"].shift(1)
    return {
        "open": fields["open"] / prev - 1.0,
        "high": fields["high"] / prev - 1.0,
        "low": fields["low"] / prev - 1.0,
        "close": fields["close"] / prev - 1.0,
        "volume": np.log1p(fields["volume"]).diff(),
    }


def day_factor(f1: np.ndarray, times: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame({"f": f1}, index=times)
    frame["day"] = frame.index.date
    grouped = frame.groupby("day")["f"]
    out = grouped.mean().to_frame("f")
    out["n"] = grouped.count()
    out = out[out["n"] >= 20].copy()
    out.index = pd.to_datetime(out.index)
    return out


def label_days(day_index: pd.DatetimeIndex, train_end, val_start) -> np.ndarray:
    labels = np.full(len(day_index), "drop", dtype=object)
    labels[day_index <= train_end.normalize()] = "train"
    labels[day_index >= val_start.normalize()] = "val"
    # The cut day contains both the train end and the embargo. Leave it out.
    labels[day_index.normalize() == train_end.normalize()] = "drop"
    return labels


def causal_daily_scale(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sigma_t and z_t = f_t / sigma_t. sigma_t uses only earlier days."""
    n, k = f.shape
    sigma = np.full((n, k), np.nan)
    z = np.full((n, k), np.nan)
    var = np.nanmean(f[:DAY_HALFLIFE] ** 2, axis=0)
    lam = float(np.exp(-np.log(2.0) / DAY_HALFLIFE))
    for t in range(DAY_HALFLIFE, n):
        sigma[t] = np.sqrt(np.maximum(var, 1e-18))
        if np.isfinite(f[t]).all():
            z[t] = f[t] / sigma[t]
            var = lam * var + (1.0 - lam) * f[t] ** 2
    return sigma, z


class SlowShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(15, 8), nn.Tanh(), nn.Linear(8, 15))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, past):
        return past + self.net(past)


def vech(mat: np.ndarray) -> np.ndarray:
    idx = np.tril_indices(5)
    return mat[idx]


def fit_net(model, loss_of, n_fit: int, device, patience: int = 40):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        best_loss = float(loss_of(model, slice(n_fit, None)))
    wait = 0
    used = 0
    for epoch in range(400):
        model.train()
        opt.zero_grad()
        loss = loss_of(model, slice(0, n_fit))
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            held = float(loss_of(model, slice(n_fit, None)))
        used = epoch + 1
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


def reliability(p: np.ndarray) -> list[dict]:
    return [{"nominal": lv, "empirical": float(np.mean(p <= lv))} for lv in LEVELS]


def walk_bands(z: np.ndarray, val: np.ndarray, when: pd.DatetimeIndex):
    """10/50/90 percent of earlier days, and where this day landed."""
    times = []
    realized = []
    q10 = []
    q90 = []
    for t in np.flatnonzero(val & np.isfinite(z).all(axis=1)):
        past = z[:t]
        past = past[np.isfinite(past).all(axis=1)]
        if len(past) < 50:
            continue
        lo = np.quantile(past, 0.10, axis=0)
        hi = np.quantile(past, 0.90, axis=0)
        times.append(when[t])
        realized.append(z[t].copy())
        q10.append(lo)
        q90.append(hi)
    return (
        pd.DatetimeIndex(times),
        np.stack(realized),
        np.stack(q10),
        np.stack(q90),
    )


def walk_pit(z: np.ndarray, val: np.ndarray) -> dict:
    """Predictive PIT of each channel and of the joint energy, using only earlier days."""
    n = len(z)
    pits = np.full((n, 5), np.nan)
    joint = np.full(n, np.nan)
    energy = np.full(n, np.nan)
    finite_z = np.isfinite(z).all(axis=1)
    energy[finite_z] = np.sum(z[finite_z] ** 2, axis=1)
    for t in np.flatnonzero(val & np.isfinite(z).all(axis=1)):
        past = z[:t]
        past = past[np.isfinite(past).all(axis=1)]
        if len(past) < 50:
            continue
        pits[t] = (past <= z[t]).mean(axis=0)
        joint[t] = float(np.mean(energy[:t][np.isfinite(energy[:t])] >= energy[t]))
    out = {}
    for c, name in enumerate(CHANNELS):
        p = pits[val, c]
        p = p[np.isfinite(p)]
        out[name] = {"n": int(len(p)), "mean_p": float(np.mean(p)), "levels": reliability(p)}
    g = joint[val]
    g = g[np.isfinite(g)]
    out["joint_energy"] = {"n": int(len(g)), "mean_p": float(np.mean(g)), "levels": reliability(g)}
    return out, pits[val], joint[val]


def mean_outer(block: np.ndarray) -> np.ndarray | None:
    ok = np.isfinite(block).all(axis=1)
    if ok.sum() < 4:
        return None
    x = block[ok]
    return (x.T @ x) / len(x)


def main() -> None:
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_end = pd.Timestamp(meta["train_end"])
    val_start = pd.Timestamp(meta["val_eval_start"])
    val_end = pd.Timestamp(meta["val_end"])
    names = meta["names"]
    print("[load] OHLCV through validation only", flush=True)
    fields = load_fields(names, val_end)
    panels = scale_free_channels(fields)
    index = panels["close"].index
    assert index.max() <= val_end
    prepare_calendar(index)
    train_bars = index <= train_end
    print(f"[factor] {len(index)} bars ending {index.max()}", flush=True)

    daily = []
    for name in CHANNELS:
        print(f"  {name}", flush=True)
        values = panels[name].to_numpy(dtype=np.float64)
        market = np.nanmean(values, axis=1)
        f_all = extract_factor(values)
        f1 = align_to_market(f_all[:, 0].copy(), market, np.asarray(train_bars))
        table = day_factor(f1, index)
        table = table.rename(columns={"f": name})
        keep = [name] if name != "close" else [name, "n"]
        daily.append(table[keep])
    days = pd.concat(daily, axis=1).sort_index().dropna()
    days = days[days["n"] >= 60]
    days = days[days.index.normalize() < val_end.normalize()]
    labels = label_days(days.index, train_end, val_start)
    f = days[list(CHANNELS)].to_numpy(dtype=np.float64)
    sigma, z_base = causal_daily_scale(f)
    ok = np.isfinite(z_base).all(axis=1)
    print(f"[days] {len(days)} usable scale {int(ok.sum())}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}" + (f" {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""), flush=True)

    # Fast clock is the causal scale already inside z_base. A free neural
    # multiplier chases outlier days and changes the unit of the distribution.
    z = z_base
    val = (labels == "val") & np.isfinite(z).all(axis=1)
    print("[walk] validation predictive distribution", flush=True)
    walked, _, _ = walk_pit(z, val)
    band_t, band_z, band_lo, band_hi = walk_bands(z, val, days.index)

    # Slow dependence: past 21 days -> next week's average outer product.
    iso = days.index.isocalendar()
    week_id = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    week_ends = np.flatnonzero(np.r_[week_id[1:] != week_id[:-1], True])
    slow_x = []
    slow_y = []
    slow_kind = []
    for end, nxt in zip(week_ends[:-1], week_ends[1:]):
        start = end - SLOW_WINDOW + 1
        if start < 0:
            continue
        past = mean_outer(z[start : end + 1])
        nxt_start = end + 1
        future = mean_outer(z[nxt_start : nxt + 1])
        if past is None or future is None:
            continue
        kind = labels[nxt]
        if kind == "drop" or labels[end] == "drop":
            continue
        # The target week must sit entirely in one split.
        if not np.all(labels[nxt_start : nxt + 1] == kind):
            continue
        slow_x.append(vech(past))
        slow_y.append(vech(future))
        slow_kind.append(kind)
    slow_x = np.stack(slow_x).astype(np.float32)
    slow_y = np.stack(slow_y).astype(np.float32)
    slow_kind = np.array(slow_kind)
    tr = np.flatnonzero(slow_kind == "train")
    va = np.flatnonzero(slow_kind == "val")
    n_fit_s = int(len(tr) * 0.8)
    order_s = np.concatenate([tr[:n_fit_s], tr[n_fit_s:]])
    x_t = torch.tensor(slow_x[order_s], device=device)
    y_t = torch.tensor(slow_y[order_s], device=device)
    slow = SlowShape().to(device)

    def slow_loss(model, idx):
        return torch.mean((model(x_t[idx]) - y_t[idx]) ** 2)

    torch.manual_seed(0)
    epochs_s, held_s = fit_net(slow, slow_loss, n_fit_s, device)
    slow.eval()
    with torch.no_grad():
        pred_va = slow(torch.tensor(slow_x[va], device=device)).cpu().numpy()
    persist = slow_x[va]
    target = slow_y[va]
    slow_mse_model = float(np.mean((pred_va - target) ** 2))
    slow_mse_persist = float(np.mean((persist - target) ** 2))
    print(
        f"[slow] epochs {epochs_s} val mse ratio {slow_mse_model / slow_mse_persist:.3f}",
        flush=True,
    )
    for name in CHANNELS:
        print(
            f"[val {name}] mean pit {walked[name]['mean_p']:.3f} "
            f"p50={next(r['empirical'] for r in walked[name]['levels'] if r['nominal']==0.5):.3f}",
            flush=True,
        )
    print(f"[val joint] mean p {walked['joint_energy']['mean_p']:.3f}", flush=True)

    scale_check = {}
    for c, name in enumerate(CHANNELS):
        scale_check[name] = {"val_mean_z2": float(np.mean(z[val, c] ** 2))}

    payload = {
        "cut": "series ends at validation end; test bars were not loaded",
        "target": "walk-forward distribution of the next day's five scale-free channel factors",
        "scale": "OHLC versus previous close, volume as log change, then each factor divided by a scale known before the day",
        "val_end": str(val_end),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "fast_scale": "20-day causal EWMA known before the day",
        "slow_epochs": epochs_s,
        "slow_val_mse_ratio_vs_persistence": slow_mse_model / slow_mse_persist,
        "walk_forward": walked,
        "scale_check": scale_check,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    inside = (band_z >= band_lo) & (band_z <= band_hi)
    hit = {name: float(inside[:, c].mean()) for c, name in enumerate(CHANNELS)}
    payload["central_80_hit_rate"] = hit
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(5, 1, figsize=(11.4, 10.2), sharex=True)
    for c, (ax, name) in enumerate(zip(axes, CHANNELS)):
        ax.fill_between(band_t, band_lo[:, c], band_hi[:, c], color="#c5d4e8", label="更早日子的 10%–90%")
        outside = ~inside[:, c]
        ax.scatter(band_t[~outside], band_z[~outside, c], s=9, color="#1f4e79", label="落在区间内", zorder=3)
        ax.scatter(band_t[outside], band_z[outside, c], s=14, color="#c45c26", label="落在区间外", zorder=4)
        ax.axhline(0, color="#dddddd", lw=0.6)
        ax.set_ylabel(LABELS[name])
        ax.set_ylim(-4.5, 4.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.99,
            0.86,
            f"{len(band_t)} 天里 {int(inside[:, c].sum())} 天在带内，名义 80% 该有 {0.8 * len(band_t):.0f} 天",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        if c == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    axes[-1].set_xlabel("验证集交易日，2025-02 至 2025-12")
    fig.suptitle("每一天事先给出的分布，以及当天五个信道实际落在哪里", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_val.png", dpi=140)
    for name in CHANNELS:
        print(f"[band {name}] inside 80% = {hit[name]:.3f}", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
