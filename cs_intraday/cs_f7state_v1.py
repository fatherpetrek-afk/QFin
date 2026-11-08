"""
Framework7's residual profile, rebuilt on the 5-minute residual of this model.

Framework7 assumed that once the common betas are gone, a name's next move
depends on its volume-price profile:

  f1  does volatility move with volume
  f2  how unstable volatility is
  f3  on heavy volume, does price go up or down
  f4  where the close sits inside the recent high-low range (cross-section)

A Kalman filter smooths these four per name, reset every morning, and the
smoothed state is assigned to four profiles. Three fixed corners and a noise
point at the origin are kept as the starting centers. The original score was
p1 ReLU(f3) + p2 ReLU(-f4) + p3 (-f2).

Same clock as Framework7: its 15-minute bar and one-hour window become a
decision every 3 bars and a 12-bar window. The residual is r - B f with the
loading known before the day. The overnight first bar is left out.

Two versions of the profiles:
  fixed  Framework7's centers, softmax of distance, its hand score.
  gmm    a 4-component mixture fit on train states, started at those centers.
         Each profile's mean and shape of the next residual come from train.

Read on validation: rank correlation with the next 15 and 60 minutes of
residual, against plain reversal of the last hour, and the band coverage per
profile. Test bars are not loaded. No equity curve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from combined_v1 import label_days, load_fields
from cs_gmm_v1 import empirical_cdf, hit_rate, mixture_quantiles
from cs_resid_v1 import newey_west
from cs_xdist_v1 import style_axes
from factor_v1 import K, QUARTER_DAYS, RHO, align_sign, day_loadings, project, top_loadings

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_f7state_v1_out"
WINDOW = 12
STEP = 3
HORIZONS = (3, 12)
FIRST_DECISION = pd.Timestamp("10:00").time()
LAST_DECISION = pd.Timestamp("15:00").time()
WIDTH_BARS = 390
Q_KF = 0.5
R_KF = 0.1
SOFT_TEMP = 0.05
CENTERS = np.array(
    [
        [-1.5, 1.5, 1.5, -1.5],
        [-1.5, -1.5, -1.5, 1.5],
        [1.5, -1.5, -1.5, -1.5],
        [0.0, 0.0, 0.0, 0.0],
    ]
)
GMM_FIT_ROWS = 400_000
LEVELS = {"50": (0.25, 0.75), "80": (0.10, 0.90), "95": (0.025, 0.975)}


def walk_bar_residual(bar_returns: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    """r - B f per bar, B fixed before the day. Same update rule as walk_projection."""
    dates = np.array(times.date)
    change = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    day_starts = change
    day_ends = np.append(change[1:], len(times))
    iso = pd.DatetimeIndex(times).isocalendar()
    week_key = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    n_names = bar_returns.shape[1]
    eps = np.full(bar_returns.shape, np.nan)
    b_slow = None
    delta = np.zeros((n_names, K))
    for d in range(len(day_starts)):
        a, b = int(day_starts[d]), int(day_ends[d])
        if b_slow is not None:
            loadings = b_slow + delta
            r_bar = bar_returns[a:b]
            if np.isfinite(r_bar).all():
                gram = loadings.T @ loadings + 1e-10 * np.eye(K)
                f_bar = r_bar @ np.linalg.solve(gram, loadings.T).T
                eps[a:b] = r_bar - f_bar @ loadings.T
            else:
                f_bar = np.full((b - a, K), np.nan)
                for i, t_i in enumerate(range(a, b)):
                    f_i, e_i = project(bar_returns[t_i], loadings)
                    f_bar[i] = f_i
                    eps[t_i] = e_i
            realized = day_loadings(f_bar, bar_returns[a:b])
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
    return eps


def roll_mean(x: np.ndarray, window: int, need_frac: float = 0.75) -> np.ndarray:
    finite = np.isfinite(x)
    zero = np.zeros((1, x.shape[1]))
    total = np.vstack([zero, np.cumsum(np.where(finite, x, 0.0), axis=0)])
    count = np.vstack([zero, np.cumsum(finite, axis=0)])
    out = np.full(x.shape, np.nan)
    s = total[window:] - total[:-window]
    c = count[window:] - count[:-window]
    ok = c >= max(2, int(np.ceil(need_frac * window)))
    block = np.full(s.shape, np.nan)
    block[ok] = s[ok] / c[ok]
    out[window - 1 :] = block
    return out


def roll_cov(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    both = np.isfinite(a) & np.isfinite(b)
    aa = np.where(both, a, np.nan)
    bb = np.where(both, b, np.nan)
    return (roll_mean(aa * bb, window) - roll_mean(aa, window) * roll_mean(bb, window)) * window / (window - 1)


def cross_z(x: np.ndarray) -> np.ndarray:
    good = np.isfinite(x)
    count = good.sum(axis=1, keepdims=True)
    mean = np.where(good, x, 0.0).sum(axis=1, keepdims=True) / np.maximum(count, 1)
    var = np.where(good, (x - mean) ** 2, 0.0).sum(axis=1, keepdims=True) / np.maximum(count - 1, 1)
    out = (x - mean) / (np.sqrt(var) + 1e-12)
    out[(count < 20).ravel()] = np.nan
    return out


def f7_features(eps, volume, high, low, close) -> np.ndarray:
    """Four Framework7 features per bar, 12-bar windows. Bars x names x 4."""
    vr = volume / roll_mean(volume, WINDOW)
    vr = cross_z(vr)
    vv = np.sqrt(np.maximum(roll_cov(eps, eps, WINDOW), 0.0))
    vv = cross_z(vv)
    vov = np.sqrt(np.maximum(roll_cov(vv, vv, WINDOW), 0.0))
    cov_vv_vr = roll_cov(vv, vr, WINDOW)
    cov_close_vr = roll_cov(eps, vr, WINDOW)
    hi = pd.DataFrame(high).rolling(STEP, min_periods=STEP).max().to_numpy()
    lo = pd.DataFrame(low).rolling(STEP, min_periods=STEP).min().to_numpy()
    resistance = (hi - close) / (hi - lo + 1e-12)
    resistance[(hi - lo) <= 0] = np.nan
    resistance_z = cross_z(resistance)
    return np.stack([cov_vv_vr, vov, cov_close_vr, resistance_z], axis=-1)


def forward_sum(eps: np.ndarray, h: int) -> np.ndarray:
    finite = np.isfinite(eps)
    zero = np.zeros((1, eps.shape[1]))
    total = np.vstack([zero, np.cumsum(np.where(finite, eps, 0.0), axis=0)])
    count = np.vstack([zero, np.cumsum(finite, axis=0)])
    out = np.full(eps.shape, np.nan)
    n = len(eps)
    end = np.arange(n) + 1 + h
    ok = end <= n
    idx = np.flatnonzero(ok)
    s = total[end[ok]] - total[idx + 1]
    c = count[end[ok]] - count[idx + 1]
    block = np.where(c == h, s, np.nan)
    out[idx] = block
    return out


def demean(x: np.ndarray) -> np.ndarray:
    good = np.isfinite(x)
    count = good.sum(axis=1, keepdims=True)
    mean = np.where(good, x, 0.0).sum(axis=1, keepdims=True) / np.maximum(count, 1)
    out = x - mean
    out[(count < 20).ravel()] = np.nan
    return out


def kalman_states(z: np.ndarray, day_id: np.ndarray):
    """Framework7's filter: identity transition, Q and R fixed, reset each day.
    Returns the posterior state and its variance scale after the next Q step."""
    n_steps, n_names, _k = z.shape
    states = np.full(z.shape, np.nan)
    p_next = np.full((n_steps, n_names), np.nan)
    x = np.zeros((n_names, 4))
    p = np.ones(n_names)
    for s in range(n_steps):
        if s == 0 or day_id[s] != day_id[s - 1]:
            x = np.zeros((n_names, 4))
            p = np.ones(n_names)
        else:
            p = p + Q_KF
        obs = np.isfinite(z[s]).all(axis=1)
        gain = p / (p + R_KF)
        x[obs] = x[obs] + gain[obs, None] * (z[s, obs] - x[obs])
        p[obs] = (1.0 - gain[obs]) * p[obs]
        states[s, obs] = x[obs]
        p_next[s, obs] = p[obs] + Q_KF
    return states, p_next


def f7_fixed(states: np.ndarray, p_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = states[:, :, None, :] - CENTERS[None, None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1) / (p_next[:, :, None] + 1e-8))
    logits = -dist / SOFT_TEMP
    logits -= np.nanmax(logits, axis=-1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=-1, keepdims=True)
    w0 = np.maximum(states[..., 2], 0.0)
    w1 = np.maximum(-states[..., 3], 0.0)
    w2 = -states[..., 1]
    raw = prob[..., 0] * w0 + prob[..., 1] * w1 + prob[..., 2] * w2
    return raw - np.nanmean(raw, axis=1, keepdims=True), prob


def rank_rows(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if int(m.sum()) < min_n:
            continue
        xr = np.argsort(np.argsort(x[i, m])).astype(np.float64)
        yr = np.argsort(np.argsort(y[i, m])).astype(np.float64)
        xr -= xr.mean()
        yr -= yr.mean()
        den = np.sqrt((xr**2).sum() * (yr**2).sum())
        if den > 1e-12:
            out[i] = float((xr * yr).sum() / den)
    return out


def spread_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if int(m.sum()) < 20:
            continue
        med = np.median(x[i, m])
        hi = y[i, m][x[i, m] > med]
        lo = y[i, m][x[i, m] <= med]
        if len(hi) and len(lo):
            out[i] = hi.mean() - lo.mean()
    return out


def by_day(values: np.ndarray, day_id: np.ndarray, rows: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"d": day_id[rows], "v": values[rows]})
    return frame.groupby("d")["v"].mean().to_numpy()


def score_signal(signal, target, day_id, train_steps, val_steps) -> dict:
    ic = rank_rows(signal, target)
    sp = spread_rows(signal, target)
    out = {}
    for name, rows in (("train", train_steps), ("val", val_steps)):
        ic_d = by_day(ic, day_id, rows)
        sp_d = by_day(sp, day_id, rows)
        n, mu, t = newey_west(ic_d, 1)
        _n2, sp_mu, sp_t = newey_west(sp_d, 1)
        good = ic_d[np.isfinite(ic_d)]
        out[name] = {
            "days": n,
            "ic": mu,
            "t": t,
            "positive_share": float((good > 0).mean()) if len(good) else float("nan"),
            "spread_bps": sp_mu * 1e4,
            "spread_t": sp_t,
        }
    return out


def fit_slope(x: np.ndarray, y: np.ndarray, rows: np.ndarray) -> float:
    xx = x[rows]
    yy = y[rows]
    ok = np.isfinite(xx) & np.isfinite(yy)
    den = float(np.dot(xx[ok], xx[ok]))
    return float(np.dot(xx[ok], yy[ok]) / den) if den > 0 else 0.0


def fit_multi(columns, y, rows) -> np.ndarray:
    x = np.stack([c[rows] for c in columns], axis=-1)
    yy = y[rows]
    ok = np.isfinite(x).all(axis=-1) & np.isfinite(yy)
    gram = x[ok].T @ x[ok] + 1e-12 * np.eye(x.shape[-1])
    return np.linalg.solve(gram, x[ok].T @ yy[ok])


def save_profile_figure(path, table, names, note: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ypos = np.arange(len(names))
    ax = axes[0]
    tr = [table[n]["60"]["train"]["ic"] for n in names]
    va = [table[n]["60"]["val"]["ic"] for n in names]
    ax.barh(ypos + 0.18, tr, height=0.34, color="#b7cce4", label="训练段")
    ax.barh(ypos - 0.18, va, height=0.34, color="#1f4e79", label="验证段")
    ax.axvline(0.0, color="#9aa3ad", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([table[n]["label"] for n in names])
    ax.set_xlabel("每 15 分钟一次的截面秩相关，按天平均")
    ax.set_title("对接下来 1 小时残差的说明力", loc="left", color="#1c2430", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    ax.grid(axis="x", color="#e6ebf0", lw=0.6)
    ax = axes[1]
    tr = [table[n]["60"]["train"]["spread_bps"] for n in names]
    va = [table[n]["60"]["val"]["spread_bps"] for n in names]
    ax.barh(ypos + 0.18, tr, height=0.34, color="#b7cce4")
    ax.barh(ypos - 0.18, va, height=0.34, color="#1f4e79")
    ax.axvline(0.0, color="#9aa3ad", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.set_xlabel("上半减下半，接下来 1 小时，基点（未扣成本）")
    ax.set_title("排在上半的和下半的差多少", loc="left", color="#1c2430", fontsize=11)
    style_axes(ax)
    ax.grid(axis="x", color="#e6ebf0", lw=0.6)
    fig.suptitle("Framework7 的量价画像，放到去掉公共因子之后的 5 分钟残差上", fontsize=14, color="#1c2430")
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_shape_figure(path, shapes, coverage, note: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    colors = ("#d06a3a", "#1f4e79", "#4f8a5b", "#9aa3ad")
    for k, (grid, dens, label) in enumerate(shapes):
        ax.plot(grid, dens, color=colors[k % len(colors)], lw=1.4, label=label)
    ax.set_xlabel("接下来 1 小时残差 ÷ 这只股票自己的宽度")
    ax.set_ylabel("训练段的密度")
    ax.set_title("每个画像在训练段的形状", loc="left", color="#1c2430", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    ax = axes[1]
    labels = [row[0] for row in coverage]
    pooled = [row[1] for row in coverage]
    mixed = [row[2] for row in coverage]
    ypos = np.arange(len(labels))
    ax.barh(ypos + 0.18, pooled, height=0.34, color="#b7cce4", label="一朵云")
    ax.barh(ypos - 0.18, mixed, height=0.34, color="#1f4e79", label="按画像混合")
    ax.axvline(0.80, color="#d06a3a", lw=1.0, ls="--", label="名义 80%")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.6, 0.95)
    ax.set_xlabel("验证段 80% 区间实际盖住的比例")
    ax.set_title("按画像分开以后，带子是否更准", loc="left", color="#1c2430", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    ax.grid(axis="x", color="#e6ebf0", lw=0.6)
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_end = pd.Timestamp(meta["train_end"])
    val_start = pd.Timestamp(meta["val_eval_start"])
    val_end = pd.Timestamp(meta["val_end"])
    fields = load_fields(meta["names"], val_end)
    assert fields["close"].index.max() <= val_end
    keep = fields["close"].index.normalize() < val_end.normalize()
    close = fields["close"][keep]
    high = fields["high"][keep].to_numpy(np.float64)
    low = fields["low"][keep].to_numpy(np.float64)
    volume = fields["volume"][keep].to_numpy(np.float64)
    times = close.index
    bar_ret = close.pct_change(fill_method=None).to_numpy(np.float64)
    eps = walk_bar_residual(bar_ret, times)
    dates = np.array(times.date)
    first_bar = np.r_[True, dates[1:] != dates[:-1]]
    eps[first_bar] = np.nan
    volume = np.where(volume > 0, volume, np.nan)
    print(f"[walk] {time.perf_counter() - t0:.0f}s  bars {len(times)}", flush=True)

    feats = f7_features(eps, volume, high, low, close.to_numpy(np.float64))
    clock = times.time
    minute = np.array([t.hour * 60 + t.minute for t in clock])
    open_min = 9 * 60 + 30
    on_grid = ((minute - open_min) % (5 * STEP) == 0)
    in_hours = np.array([(FIRST_DECISION <= t <= LAST_DECISION) for t in clock])
    steps = np.flatnonzero(on_grid & in_hours)
    day_of_step = pd.DatetimeIndex(times[steps]).normalize()
    day_index = pd.DatetimeIndex(sorted(set(day_of_step)))
    day_id = np.searchsorted(day_index, day_of_step)
    day_label = label_days(day_index, train_end, val_start)[day_id]
    train_steps = day_label == "train"
    val_steps = day_label == "val"

    z = feats[steps]
    z = (z - np.nanmean(z, axis=1, keepdims=True)) / (np.nanstd(z, axis=1, keepdims=True) + 1e-12)
    states, p_next = kalman_states(z, day_id)

    targets = {}
    for h in HORIZONS:
        targets[str(h * 5)] = demean(forward_sum(eps, h)[steps])
    past_hour = demean(-roll_mean(eps, WINDOW)[steps] * WINDOW)
    sq = roll_mean(eps**2, WIDTH_BARS, need_frac=0.5)
    sigma_bar = np.sqrt(sq)
    width = {key: sigma_bar[steps] * np.sqrt(int(key) // 5) for key in targets}

    fixed_score, fixed_prob = f7_fixed(states, p_next)

    rows_train = np.flatnonzero(train_steps)
    flat_states = states[rows_train].reshape(-1, 4)
    usable = np.isfinite(flat_states).all(axis=1)
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(usable)
    pick = rng.choice(pool, size=min(GMM_FIT_ROWS, len(pool)), replace=False)
    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        means_init=CENTERS,
        random_state=0,
        max_iter=300,
    ).fit(flat_states[pick])
    order_flat = states.reshape(-1, 4)
    ok_flat = np.isfinite(order_flat).all(axis=1)
    resp = np.full((len(order_flat), 4), np.nan)
    resp[ok_flat] = gmm.predict_proba(order_flat[ok_flat])
    resp = resp.reshape(states.shape[0], states.shape[1], 4)
    hard = np.where(np.isfinite(resp).all(-1), np.nanargmax(np.nan_to_num(resp, nan=-1), axis=-1), -1)
    print(f"[gmm] converged {gmm.converged_} iters {gmm.n_iter_}  weights {np.round(gmm.weights_, 3)}", flush=True)
    print(f"[gmm] centers\n{np.round(gmm.means_, 2)}", flush=True)

    profile_stats = {}
    gmm_center = {}
    for key, target in targets.items():
        zt = target / width[key]
        zt[~np.isfinite(width[key]) | (width[key] <= 0)] = np.nan
        mu = np.zeros(4)
        for k in range(4):
            m = train_steps[:, None] & (hard == k) & np.isfinite(zt)
            mu[k] = float(np.mean(zt[m])) if m.any() else 0.0
        gmm_center[key] = np.einsum("snk,k->sn", np.nan_to_num(resp), mu) * width[key]
        gmm_center[key][~np.isfinite(resp).all(-1)] = np.nan
        profile_stats[key] = mu

    signals = {
        "reversal": ("过去 1 小时残差反着押", lambda key: past_hour),
        "f7_fixed": ("F7 原样：固定四簇 + 手写打分", lambda key: fixed_score),
        "gmm_center": ("GMM 四簇（F7 起点）学出的中心", lambda key: gmm_center[key]),
    }
    labels_feat = ("f1 波动随量", "f2 波动不稳", "f3 放量涨跌", "f4 收在区间哪里")
    slopes = {}
    for j in range(4):
        name = f"state_f{j + 1}"
        slopes[name] = {}
        signals[name] = (labels_feat[j], None)
    table = {}
    for key, target in targets.items():
        for name, (label, maker) in signals.items():
            if maker is None:
                j = int(name[-1]) - 1
                b = fit_slope(states[..., j], target, train_steps[:, None] & np.ones(target.shape, dtype=bool))
                slopes[name][key] = b
                sig = b * states[..., j]
            else:
                sig = maker(key)
            res = score_signal(sig, target, day_id, train_steps, val_steps)
            table.setdefault(name, {"label": label})[key] = res
            print(
                f"[{key}m {name:11s}] IC train {res['train']['ic']:+.4f} (t {res['train']['t']:+.1f})  "
                f"val {res['val']['ic']:+.4f} (t {res['val']['t']:+.1f})  "
                f"spread val {res['val']['spread_bps']:+.2f} bp (t {res['val']['spread_t']:+.1f})",
                flush=True,
            )
        cols = [past_hour] + [states[..., j] for j in range(4)]
        coef = fit_multi(cols, target, train_steps[:, None] & np.ones(target.shape, dtype=bool))
        combo = sum(c * b for c, b in zip(cols, coef))
        res = score_signal(combo, target, day_id, train_steps, val_steps)
        table.setdefault("reversal_plus_f7", {"label": "反转 + 四个画像量（训练段线性）"})[key] = res
        table["reversal_plus_f7"].setdefault("coef", {})[key] = [float(c) for c in coef]
        print(
            f"[{key}m reversal+f7] coef {np.round(coef, 5)}  IC train {res['train']['ic']:+.4f} (t {res['train']['t']:+.1f})  "
            f"val {res['val']['ic']:+.4f} (t {res['val']['t']:+.1f})  spread val {res['val']['spread_bps']:+.2f} bp",
            flush=True,
        )

    key = "60"
    target = targets[key]
    zt = target / width[key]
    zt[~np.isfinite(width[key]) | (width[key] <= 0)] = np.nan
    grid = np.linspace(-6.0, 6.0, 1201)
    cdfs = np.vstack([empirical_cdf(zt[train_steps[:, None] & (hard == k)], grid) for k in range(4)])
    pooled_cdf = empirical_cdf(zt[train_steps[:, None] & (hard >= 0)], grid)
    val_rows = val_steps[:, None] & (hard >= 0) & np.isfinite(zt)
    r_val = resp[val_rows]
    y_val = zt[val_rows]
    h_val = hard[val_rows]
    coverage_rows = []
    coverage = {}
    for name, (lo_q, hi_q) in LEVELS.items():
        q_mix = mixture_quantiles(r_val, cdfs, grid, np.array([lo_q, hi_q]))
        q_one = mixture_quantiles(np.ones((1, 1)), pooled_cdf[None, :], grid, np.array([lo_q, hi_q]))[0]
        one_lo = np.full(len(y_val), q_one[0])
        one_hi = np.full(len(y_val), q_one[1])
        all_mask = np.ones(len(y_val), dtype=bool)
        coverage[name] = {
            "pooled": hit_rate(y_val, one_lo, one_hi, all_mask),
            "mixture": hit_rate(y_val, q_mix[:, 0], q_mix[:, 1], all_mask),
            "by_profile": {},
        }
        for k in range(4):
            m = h_val == k
            coverage[name]["by_profile"][str(k)] = {
                "pooled": hit_rate(y_val, one_lo, one_hi, m),
                "mixture": hit_rate(y_val, q_mix[:, 0], q_mix[:, 1], m),
                "share": float(m.mean()),
            }
        if name == "80":
            for k in range(4):
                row = coverage[name]["by_profile"][str(k)]
                coverage_rows.append((f"画像 {k + 1}（{row['share']:.0%}）", row["pooled"][0], row["mixture"][0]))
    print(
        f"[band 60m] 80% pooled {coverage['80']['pooled'][0]:.3f}  mixture {coverage['80']['mixture'][0]:.3f}  "
        + "  ".join(
            f"p{k + 1} {coverage['80']['by_profile'][str(k)]['pooled'][0]:.3f}->{coverage['80']['by_profile'][str(k)]['mixture'][0]:.3f}"
            for k in range(4)
        ),
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names_plot = ["reversal", "f7_fixed", "gmm_center", "state_f1", "state_f2", "state_f3", "state_f4", "reversal_plus_f7"]
    save_profile_figure(
        OUT_DIR / "cs_f7state_ic.png",
        table,
        names_plot,
        "每 15 分钟决策一次，10:00 到 15:00。方向只在训练段定。浅色训练段，深色验证段。基点未扣买卖价差（来回约 2 个基点）。不是收益。",
    )
    shapes = []
    for k in range(4):
        sample = zt[train_steps[:, None] & (hard == k)]
        sample = sample[np.isfinite(sample)]
        hist, edges = np.histogram(np.clip(sample, -5, 5), bins=120, range=(-5, 5), density=True)
        shapes.append((0.5 * (edges[1:] + edges[:-1]), hist, f"画像 {k + 1}，均值 {profile_stats[key][k]:+.3f}"))
    save_shape_figure(
        OUT_DIR / "cs_f7state_shape.png",
        shapes,
        coverage_rows,
        "画像由 GMM 在训练段的卡尔曼状态上学出，起点是 F7 写死的四个质心。宽度是这只股票过去 5 天的 5 分钟残差。不是收益。",
    )

    payload = {
        "clock": {"window_bars": WINDOW, "step_bars": STEP, "first": "10:00", "last": "15:00"},
        "kalman": {"Q": Q_KF, "R": R_KF, "reset": "each day"},
        "gmm": {
            "init_centers": CENTERS.tolist(),
            "learned_centers": gmm.means_.tolist(),
            "weights": gmm.weights_.tolist(),
            "converged": bool(gmm.converged_),
            "profile_mean_z": {k: v.tolist() for k, v in profile_stats.items()},
        },
        "state_slopes_train": slopes,
        "signals": table,
        "band_60m": {
            name: {
                "pooled": {"hit_rate": v["pooled"][0], "n": v["pooled"][1]},
                "mixture": {"hit_rate": v["mixture"][0], "n": v["mixture"][1]},
                "by_profile": {
                    k: {
                        "share": row["share"],
                        "pooled": row["pooled"][0],
                        "mixture": row["mixture"][0],
                        "n": row["mixture"][1],
                    }
                    for k, row in v["by_profile"].items()
                },
            }
            for name, v in coverage.items()
        },
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
