"""
Does the edge survive entering a few minutes late?

The decision bar's close is a trade, and a trade prints at the bid or the ask.
A name that closed at the bid looks like it closed near its low, the close
position signal says buy, and the next print at the mid looks like a gain.
That gain is the spread, not a pullback, and no order could have earned it.

Every trade made at a decision bar is filled here at the close of a later bar
instead, so the return of the bars in between is lost. The exit is still the
closing auction. The book and the costs are the train-picked line from
cs_cost_v2: sleeves, expected move above twice the round trip, a two-tick
spread plus a per-share fee. The mix is also refit without the close position
signal. Test bars are not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_v1 import load_fields
from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP, fit_mix, row_rank
from cs_cost_v1 import FEE_PER_SHARE, TICK, TOP_FRAC, build, day_books, nw, target_book
from cs_f7state_v1 import demean
from cs_xdist_v1 import style_axes

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_cost_v3_out"
THRESHOLD = 2.0
DELAYS = (0, 1, 2)


def decision_bars(meta):
    """Bar positions of the hourly decisions and the 5-minute returns, built the
    same way as cs_cost_v1.build."""
    val_end = pd.Timestamp(meta["val_end"])
    close_df = load_fields(list(meta["names"]), val_end)["close"]
    close_df = close_df[close_df.index.normalize() < val_end.normalize()]
    times = close_df.index
    minute = np.array([t.hour * 60 + t.minute for t in times.time])
    grid = ((minute - 570) % 15 == 0) & (minute >= 600) & (minute <= 900) & ((minute - 600) % 60 == 0)
    return np.flatnonzero(grid), close_df.pct_change(fill_method=None).to_numpy(np.float64)


def delayed_days(held, seg_ret, bar_ret, steps, delay, c_in, c_out, day_id, last_of_day):
    gross = np.nansum(held * np.nan_to_num(seg_ret), axis=1)
    prev = np.r_[np.zeros((1, held.shape[1])), held[:-1]]
    first = np.r_[True, day_id[1:] != day_id[:-1]]
    prev[first] = 0.0
    trade = held - prev
    for lag in range(1, delay + 1):
        gross -= np.nansum(trade * np.nan_to_num(bar_ret[steps + lag]), axis=1)
    cost = np.nansum(np.abs(trade) * c_in, axis=1)
    cost[last_of_day] += np.nansum(np.abs(held[last_of_day]) * c_out[last_of_day], axis=1)
    frame = pd.DataFrame({"d": day_id, "g": gross, "c": cost})
    return frame.groupby("d")[["g", "c"]].sum()


def save_figure(path, table, note):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    for ax, split, title in zip(axes, ("train", "val"), ("训练段", "验证段")):
        x = np.arange(len(DELAYS))
        for offset, mix_name, color in ((-0.2, "三个信号", "#1f4e79"), (0.2, "去掉收盘位置", "#d9a441")):
            vals = [table[mix_name][d][split]["net"]["mean_bp"] for d in DELAYS]
            ts = [table[mix_name][d][split]["net"]["t"] for d in DELAYS]
            errs = [abs(v / t) if np.isfinite(t) and abs(t) > 1e-9 else 0.0 for v, t in zip(vals, ts)]
            ax.bar(x + offset, vals, width=0.38, color=color, label=mix_name, yerr=errs,
                   error_kw={"elinewidth": 0.8, "ecolor": "#6b7380"})
        ax.axhline(0.0, color="#9aa3ad", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"晚 {5 * d} 分钟成交" if d else "决策那一刻成交" for d in DELAYS])
        ax.set_title(title, loc="left", color="#1c2430", fontsize=11)
        style_axes(ax)
    axes[0].set_ylabel("扣成本后，每天每 1 块钱本金（基点）")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("晚几分钟下单，优势还在不在", fontsize=15, color="#1c2430")
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#6b7380")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    p = build(meta)
    steps, bar_ret = decision_bars(meta)
    assert len(steps) == len(p["day_id"])
    print(f"[panel] {time.perf_counter() - t0:.0f}s", flush=True)
    train = p["day_label"] == "train"
    codes = p["codes"]
    big = [c for c in range(len(p["sector_order"])) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    half_spread = TICK / p["price"]
    fee = FEE_PER_SHARE / p["price"]
    c_in, c_out = half_spread + fee, fee
    raw_rel = demean(p["raw_close"])
    ztarget = np.clip(p["resid"] / p["width"], -Z_CLIP, Z_CLIP)
    all_rows = train[:, None] & np.ones(ztarget.shape, dtype=bool)
    ranks = {"gap": row_rank(-p["gap"]), "f4": row_rank(p["f4"]), "f3": row_rank(-p["f3"])}
    labels = p["day_label_all"]

    table = {}
    for mix_name, keys in (("三个信号", ("gap", "f4", "f3")), ("去掉收盘位置", ("gap", "f3"))):
        cols = [ranks[k] for k in keys]
        coef = fit_mix(cols, ztarget, all_rows)
        mix = sum(c * b for c, b in zip(cols, coef))
        expected = mix * p["width"]
        ok = train[:, None] & np.isfinite(expected) & np.isfinite(raw_rel)
        expected = expected * float((expected[ok] * raw_rel[ok]).sum() / (expected[ok] ** 2).sum())
        w = target_book(mix, codes, big, TOP_FRAC, np.abs(expected) > THRESHOLD * (c_in + c_out))
        held = day_books(w, p["day_id"], "sleeves")
        table[mix_name] = {"coef": dict(zip(keys, [float(c) for c in coef]))}
        for delay in DELAYS:
            d = delayed_days(held, p["raw_seg"], bar_ret, steps, delay, c_in, c_out, p["day_id"], p["last_of_day"])
            lab = labels[d.index.to_numpy()]
            table[mix_name][delay] = {
                split: {"gross": nw(d["g"][lab == split]), "net": nw((d["g"] - d["c"])[lab == split]),
                        "cost_bp": float(d["c"][lab == split].mean() * 1e4)} for split in ("train", "val")
            }
            r = table[mix_name][delay]
            print(
                f"[{mix_name} delay {delay} bar] train gross {r['train']['gross']['mean_bp']:+.2f} net "
                f"{r['train']['net']['mean_bp']:+.2f} (t {r['train']['net']['t']:+.1f}) | val gross "
                f"{r['val']['gross']['mean_bp']:+.2f} net {r['val']['net']['mean_bp']:+.2f} (t {r['val']['net']['t']:+.1f})",
                flush=True,
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(
        OUT_DIR / "cs_cost_delay.png",
        table,
        "每小时加六分之一的名单，收盘竞价离场，只做预期涨跌超过来回成本 2 倍的名字。成本：价差 2 分钱进场付一半，加每股 0.35 美分佣金。不是净值。",
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(table, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[time] {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
