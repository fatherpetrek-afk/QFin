"""
What the combined pullback really costs, and ways to trade it more cheaply.

There are no quotes in the data, only 5-minute bars. The spread is read two ways:

  tick     one cent divided by the price. A quoted spread is never narrower.
  bars     Abdi and Ranaldo (2017): the close sits at the bid or the ask, the
           high-low midpoint sits near the true price, so
           spread^2 = 4 E[(c_t - m_t)(c_t - m_{t+1})] on log prices.
           At 5 minutes the price move inside a bar swamps a one-cent spread,
           so this is only reported, not used for costs.

Cost per side for a name is half a spread of one or two ticks, optionally plus
a per-share fee.
Costs are paid name by name on the weights actually traded, so cheap stocks
cost more basis points than expensive ones.

Books, all with at most one unit of gross capital, all flat at the close:

  one shot   each hour's picks held to the close, read hour by hour
  rolling    the book is replaced every hour, the change is paid for
  sleeves    each hour adds one sixth of a book that is held to the close
  smoothed   the book moves halfway to the new picks every hour

The cost-aware rule only keeps names whose expected move to the close, fit on
train, is larger than their own round trip cost. Test bars are not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import label_days, load_fields
from cs_combo_v1 import MIN_SECTOR_NAMES, STEP, WIDTH_BARS, Z_CLIP, fit_mix, row_rank
from cs_combo_v2 import window_sum
from cs_f7state_v1 import demean, f7_features, kalman_states, roll_mean, walk_bar_residual
from cs_peer_v1 import sector_codes, static_peer_mean
from cs_resid_v1 import newey_west
from cs_xdist_v1 import style_axes

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_cost_v1_out"
TICK = 0.01
FEE_PER_SHARE = 0.0035
SPREAD_DAYS = 20
SKIP_OPEN_BARS = 2
TOP_FRAC = 0.2


def bar_spread(close, high, low, first_bar, day_of_bar, n_days):
    """Daily mean of the Abdi-Ranaldo product per name."""
    c = np.log(close)
    m = 0.5 * (np.log(high) + np.log(low))
    prod = np.full(c.shape, np.nan)
    prod[:-1] = (c[:-1] - m[:-1]) * (c[:-1] - m[1:])
    same_day = np.r_[day_of_bar[1:] == day_of_bar[:-1], False]
    position = np.arange(len(c)) - np.flatnonzero(first_bar)[day_of_bar]
    prod[~same_day | (position < SKIP_OPEN_BARS)] = np.nan
    frame = pd.DataFrame(prod)
    frame["d"] = day_of_bar
    return frame.groupby("d").mean().reindex(range(n_days)).to_numpy()


def build(meta):
    names = list(meta["names"])
    train_end = pd.Timestamp(meta["train_end"])
    val_start = pd.Timestamp(meta["val_eval_start"])
    val_end = pd.Timestamp(meta["val_end"])
    fields = load_fields(names, val_end)
    assert fields["close"].index.max() <= val_end
    keep = fields["close"].index.normalize() < val_end.normalize()
    close_df = fields["close"][keep]
    close = close_df.to_numpy(np.float64)
    high = fields["high"][keep].to_numpy(np.float64)
    low = fields["low"][keep].to_numpy(np.float64)
    volume = fields["volume"][keep].to_numpy(np.float64)
    volume = np.where(volume > 0, volume, np.nan)
    times = close_df.index
    bar_ret = close_df.pct_change(fill_method=None).to_numpy(np.float64)
    eps_full = walk_bar_residual(bar_ret, times)
    dates = np.array(times.date)
    first_bar = np.r_[True, dates[1:] != dates[:-1]]
    bar_day = np.cumsum(first_bar) - 1
    eps_feat = eps_full.copy()
    eps_feat[first_bar] = np.nan

    feats = f7_features(eps_feat, volume, high, low, close)
    minute = np.array([t.hour * 60 + t.minute for t in times.time])
    grid15 = ((minute - 570) % (5 * STEP) == 0) & (minute >= 600) & (minute <= 900)
    steps15 = np.flatnonzero(grid15)
    z = feats[steps15]
    with np.errstate(all="ignore"):
        z = (z - np.nanmean(z, axis=1, keepdims=True)) / (np.nanstd(z, axis=1, keepdims=True) + 1e-12)
    states15, _p = kalman_states(z, bar_day[steps15])
    hourly = (minute[steps15] - 600) % 60 == 0
    steps = steps15[hourly]
    states = states15[hourly]

    day_all = pd.DatetimeIndex(sorted(set(times.normalize())))
    day_label_all = label_days(day_all, train_end, val_start)
    day_id = bar_day[steps]
    day_start = np.flatnonzero(first_bar)
    day_end = np.append(day_start[1:], len(times)) - 1
    start_of = day_start[day_id]
    end_of = day_end[day_id]
    next_step = np.r_[steps[1:], 0]
    last_of_day = np.r_[day_id[1:] != day_id[:-1], True]
    next_step = np.where(last_of_day, end_of, np.minimum(next_step, end_of))

    since_close = window_sum(eps_full, np.maximum(start_of - 1, 0), steps)
    since_close[start_of == 0] = np.nan
    sigma_bar = np.sqrt(roll_mean(eps_feat**2, WIDTH_BARS, need_frac=0.5))[steps]
    codes, sector_order = sector_codes(names)
    same = (codes[:, None] == codes[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    moved = demean(since_close)
    gap = moved - static_peer_mean(moved, same)

    daily_spread = bar_spread(close, high, low, first_bar, bar_day, len(day_all))
    return {
        "names": names,
        "hour_label": np.array([f"{m // 60}:{m % 60:02d}" for m in minute[steps]]),
        "day_id": day_id,
        "day_label": day_label_all[day_id],
        "day_label_all": day_label_all,
        "day_dates": day_all,
        "last_of_day": last_of_day,
        "codes": codes,
        "sector_order": sector_order,
        "gap": gap,
        "states": states,
        "f4": states[..., 3],
        "f3": states[..., 2],
        "resid": demean(window_sum(eps_full, steps, end_of)),
        "raw_close": window_sum(bar_ret, steps, end_of),
        "raw_seg": window_sum(bar_ret, steps, next_step),
        "width": sigma_bar * np.sqrt(np.maximum(end_of - steps, 1).astype(np.float64))[:, None],
        "price": close[steps],
        "bar_spread_day": daily_spread,
    }


def target_book(signal, codes, sectors, frac, keep=None):
    """Inside each sector: long the top `frac` with a positive score, short the
    bottom `frac` with a negative score, half the sector's capital on each side,
    sectors equally weighted. Gross is one, or zero when nothing qualifies."""
    w = np.zeros(signal.shape)
    groups = [np.flatnonzero(codes == c) for c in sectors]
    for i in range(len(signal)):
        legs = []
        for cols in groups:
            x = signal[i, cols]
            ok = np.isfinite(x)
            n_all = int(ok.sum())
            if n_all < MIN_SECTOR_NAMES:
                continue
            k = max(1, int(np.floor(frac * n_all)))
            if keep is not None:
                ok &= keep[i, cols]
            idx = cols[ok]
            xs = x[ok]
            order = np.argsort(xs)
            long_side = [j for j in order[::-1][:k] if xs[j] > 0]
            short_side = [j for j in order[:k] if xs[j] < 0]
            if long_side and short_side:
                legs.append((idx[long_side], idx[short_side]))
        for long_idx, short_idx in legs:
            w[i, long_idx] += 0.5 / len(legs) / len(long_idx)
            w[i, short_idx] -= 0.5 / len(legs) / len(short_idx)
    return w


def day_books(w, day_id, kind):
    """Turn the hourly target books into the book actually held in each hour."""
    held = np.zeros_like(w)
    starts = np.r_[0, np.flatnonzero(day_id[1:] != day_id[:-1]) + 1]
    ends = np.r_[starts[1:], len(w)]
    for a, b in zip(starts, ends):
        if kind == "rolling":
            held[a:b] = w[a:b]
        elif kind == "sleeves":
            held[a:b] = np.cumsum(w[a:b], axis=0) / 6.0
        elif kind == "smoothed":
            cur = w[a]
            for i in range(a, b):
                cur = w[i] if i == a else 0.5 * cur + 0.5 * w[i]
                held[i] = cur
    return held


def book_days(held, seg_ret, cost_side, day_id, last_of_day):
    """Per day: gross from holding each hour's book to the next hour, cost of
    every change, exit at the close."""
    gross = np.nansum(held * np.nan_to_num(seg_ret), axis=1)
    prev = np.r_[np.zeros((1, held.shape[1])), held[:-1]]
    first = np.r_[True, day_id[1:] != day_id[:-1]]
    prev[first] = 0.0
    trade = np.abs(held - prev)
    trade[last_of_day] += np.abs(held[last_of_day])
    cost = np.nansum(trade * cost_side, axis=1)
    frame = pd.DataFrame({"d": day_id, "g": gross, "c": cost, "t": trade.sum(axis=1)})
    return frame.groupby("d")[["g", "c", "t"]].sum()


def one_shot_rows(w, ret_close, cost_side):
    gross = np.nansum(w * np.nan_to_num(ret_close), axis=1)
    cost = np.nansum(np.abs(w) * 2.0 * cost_side, axis=1)
    return gross, cost


def nw(values):
    n, mu, t = newey_west(np.asarray(values, dtype=np.float64), 1)
    return {"days": n, "mean_bp": mu * 1e4, "t": t}


def save_figure(path, rows, note):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12.6, 0.55 * len(rows) + 2.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.2, [r["gross"] for r in rows], height=0.38, color="#b7cce4", label="毛利")
    net = [r["net"] for r in rows]
    err = [abs(r["net"] / r["net_t"]) if np.isfinite(r["net_t"]) and abs(r["net_t"]) > 1e-9 else 0.0 for r in rows]
    ax.barh(y - 0.2, net, height=0.38, color=["#1f4e79" if v > 0 else "#d06a3a" for v in net], label="扣成本后",
            xerr=err, error_kw={"elinewidth": 0.8, "ecolor": "#6b7380"})
    ax.axvline(0.0, color="#9aa3ad", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_xlabel("验证段，每天每 1 块钱本金（基点），横线是一倍标准误")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    fig.suptitle("按每只股票自己的价差算成本之后，还剩多少", fontsize=15, color="#1c2430")
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    p = build(meta)
    print(f"[panel] {time.perf_counter() - t0:.0f}s", flush=True)
    train = p["day_label"] == "train"
    val = p["day_label"] == "val"
    codes = p["codes"]
    hours = sorted(set(p["hour_label"]), key=lambda s: int(s.split(":")[0]))

    tick_bp = TICK / p["price"] * 1e4
    val_days = p["day_label_all"] == "val"
    spread_val = 2.0 * np.sqrt(np.maximum(np.nanmean(p["bar_spread_day"][val_days], axis=0), 0.0)) * 1e4
    tick_val = np.nanmedian(tick_bp[val], axis=0)
    report = {"spread": {
        "tick_bp_median_name": float(np.nanmedian(tick_val)),
        "tick_bp_quartiles": [float(q) for q in np.nanpercentile(tick_val, [25, 75])],
        "bar_bp_median_name": float(np.nanmedian(spread_val)),
        "bar_bp_quartiles": [float(q) for q in np.nanpercentile(spread_val, [25, 75])],
        "bar_over_tick_median": float(np.nanmedian(spread_val / tick_val)),
        "per_name": {n: {"tick_bp": float(a), "bar_bp": float(b)} for n, a, b in zip(p["names"], tick_val, spread_val)},
    }}
    print(
        f"[spread] one tick: median name {report['spread']['tick_bp_median_name']:.2f}bp "
        f"(quartiles {np.round(report['spread']['tick_bp_quartiles'], 2)})  bars: median "
        f"{report['spread']['bar_bp_median_name']:.2f}bp (quartiles {np.round(report['spread']['bar_bp_quartiles'], 2)})  "
        f"bars/tick {report['spread']['bar_over_tick_median']:.2f}",
        flush=True,
    )

    cols = [row_rank(-p["gap"]), row_rank(p["f4"]), row_rank(-p["f3"])]
    ztarget = np.clip(p["resid"] / p["width"], -Z_CLIP, Z_CLIP)
    coef = fit_mix(cols, ztarget, train[:, None] & np.ones(ztarget.shape, dtype=bool))
    mix = sum(c * b for c, b in zip(cols, coef))
    expected = mix * p["width"]
    raw_rel = demean(p["raw_close"])
    ok = train[:, None] & np.isfinite(expected) & np.isfinite(raw_rel)
    ex, rc = expected[ok], raw_rel[ok]
    scale = float((ex * rc).sum() / (ex * ex).sum())
    expected = scale * expected
    print(f"[mix] coef {np.round(coef, 4)}  expected-move scale on train {scale:.3f}", flush=True)

    report["spread"]["bar_negative_share"] = float(np.mean(~(spread_val > 0)))
    scenarios = {
        "价差 1 跳": 0.5 * TICK / p["price"],
        "价差 2 跳": TICK / p["price"],
        "价差 2 跳 + 每股佣金": TICK / p["price"] + FEE_PER_SHARE / p["price"],
        "原来的每边 1 个基点": np.full(p["price"].shape, 1e-4),
    }
    big = [c for c in range(len(p["sector_order"])) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    report["coef"] = [float(c) for c in coef]
    report["expected_scale"] = scale
    report["books"] = {}
    rows_fig = []
    for scen, cost_side in scenarios.items():
        cheap = np.abs(expected) > 2.0 * cost_side
        for filt_name, keep in (("全部", None), ("只做预期涨跌够付成本的", cheap)):
            w = target_book(mix, codes, big, TOP_FRAC, keep)
            gross_1, cost_1 = one_shot_rows(w, p["raw_close"], cost_side)
            entry = {}
            for h in hours:
                sel = p["hour_label"] == h
                g = pd.Series(gross_1[sel]).groupby(p["day_id"][sel]).sum()
                c = pd.Series(cost_1[sel]).groupby(p["day_id"][sel]).sum()
                lab = pd.Series(p["day_label"][sel]).groupby(p["day_id"][sel]).first()
                entry[f"one shot {h}"] = {
                    split: {"gross": nw(g[lab == split]), "cost_bp": float(c[lab == split].mean() * 1e4),
                            "net": nw((g - c)[lab == split])} for split in ("train", "val")
                }
            for kind in ("rolling", "sleeves", "smoothed"):
                held = day_books(w, p["day_id"], kind)
                d = book_days(held, p["raw_seg"], cost_side, p["day_id"], p["last_of_day"])
                lab = p["day_label_all"][d.index.to_numpy()]
                entry[kind] = {
                    split: {"gross": nw(d["g"][lab == split]), "cost_bp": float(d["c"][lab == split].mean() * 1e4),
                            "net": nw((d["g"] - d["c"])[lab == split]), "turnover": float(d["t"][lab == split].mean())}
                    for split in ("train", "val")
                }
            best_hour = max(hours, key=lambda h: entry[f"one shot {h}"]["train"]["net"]["mean_bp"])
            entry["train-best one shot"] = best_hour
            report["books"][f"{scen} | {filt_name}"] = entry
            print(f"--- {scen} | {filt_name}", flush=True)
            for key in [f"one shot {h}" for h in hours] + ["rolling", "sleeves", "smoothed"]:
                v, t = entry[key]["val"], entry[key]["train"]
                extra = f" turnover {v['turnover']:.2f}" if "turnover" in v else ""
                print(
                    f"  {key:14s} val gross {v['gross']['mean_bp']:+6.2f} cost {v['cost_bp']:5.2f} net "
                    f"{v['net']['mean_bp']:+6.2f} (t {v['net']['t']:+.1f}) | train net {t['net']['mean_bp']:+6.2f}{extra}",
                    flush=True,
                )
            print(f"  train-best one-shot hour: {best_hour}", flush=True)
            if scen == "价差 2 跳":
                for key, lab in ((f"one shot {best_hour}", f"每天一次（训练段挑的 {best_hour}）"),
                                 ("rolling", "每小时换一遍"), ("sleeves", "每小时加六分之一"), ("smoothed", "每小时往新名单挪一半")):
                    v = entry[key]["val"]
                    rows_fig.append({"label": f"{lab}，{filt_name}", "gross": v["gross"]["mean_bp"],
                                     "net": v["net"]["mean_bp"], "net_t": v["net"]["t"]})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(
        OUT_DIR / "cs_cost_books.png",
        rows_fig,
        "成本：每只股票买卖价差按 2 分钱算，进出各付一半。七个大行业内部做多做空，收盘前全部平掉。不是净值。",
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
