"""
One picture: holding the whole pool against the pullback book, after costs.

The pool is the 97 names in equal weight, held from close to close. The book
is the train-picked line from cs_cost_v2, filled one bar late as in cs_cost_v3:
sleeves, expected move above twice the round trip, a two-tick spread plus a
per-share fee on entry, closing-auction exit with the fee only. It holds at
most one unit of gross capital, long and short in equal amounts, and is flat
every night. No leverage on either line.

Train days are where every choice was made, so that part of the curve is
flattering. Test bars are not loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cs_combo_v1 import MIN_SECTOR_NAMES, Z_CLIP, fit_mix, row_rank
from cs_cost_v1 import FEE_PER_SHARE, TICK, TOP_FRAC, build, day_books, target_book
from cs_cost_v3 import THRESHOLD, decision_bars, delayed_days
from cs_f7state_v1 import demean
from cs_xdist_v1 import style_axes

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "cs_curve_v1_out"
DELAY = 1


def pool_daily(meta, day_dates):
    from combined_v1 import load_fields

    val_end = pd.Timestamp(meta["val_end"])
    close = load_fields(list(meta["names"]), val_end)["close"]
    close = close[close.index.normalize() < val_end.normalize()]
    last = close.groupby(close.index.normalize()).last().reindex(day_dates)
    return last.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0).to_numpy()


def stats(r):
    curve = np.cumprod(1.0 + r)
    years = len(r) / 252.0
    peak = np.maximum.accumulate(curve)
    return {
        "total": float(curve[-1] - 1.0),
        "per_year": float(curve[-1] ** (1.0 / years) - 1.0),
        "vol": float(np.std(r) * np.sqrt(252.0)),
        "worst_drop": float((curve / peak - 1.0).min()),
        "ratio": float(np.mean(r) / np.std(r) * np.sqrt(252.0)) if np.std(r) > 0 else float("nan"),
        "up_days": float((r > 0).mean()),
    }


def draw(ax, dates, lines, title):
    for r, color, label, lw in lines:
        ax.plot(dates, (np.cumprod(1.0 + r) - 1.0) * 100.0, color=color, lw=lw, label=label)
    ax.axhline(0.0, color="#9aa3ad", lw=0.8)
    ax.set_title(title, loc="left", color="#1c2430", fontsize=12)
    ax.set_ylabel("累计涨跌（%）")
    style_axes(ax)


def label(name, s):
    return (f"{name}：累计 {s['total'] * 100:+.1f}%，年化 {s['per_year'] * 100:+.1f}%，"
            f"最大回撤 {s['worst_drop'] * 100:.1f}%，年波动 {s['vol'] * 100:.1f}%")


def main() -> None:
    meta = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    p = build(meta)
    steps, bar_ret = decision_bars(meta)
    train = p["day_label"] == "train"
    codes = p["codes"]
    big = [c for c in range(len(p["sector_order"])) if int((codes == c).sum()) >= MIN_SECTOR_NAMES]
    c_in = (TICK + FEE_PER_SHARE) / p["price"]
    c_out = FEE_PER_SHARE / p["price"]
    cols = [row_rank(-p["gap"]), row_rank(p["f4"]), row_rank(-p["f3"])]
    ztarget = np.clip(p["resid"] / p["width"], -Z_CLIP, Z_CLIP)
    coef = fit_mix(cols, ztarget, train[:, None] & np.ones(ztarget.shape, dtype=bool))
    mix = sum(c * b for c, b in zip(cols, coef))
    expected = mix * p["width"]
    raw_rel = demean(p["raw_close"])
    ok = train[:, None] & np.isfinite(expected) & np.isfinite(raw_rel)
    expected = expected * float((expected[ok] * raw_rel[ok]).sum() / (expected[ok] ** 2).sum())
    w = target_book(mix, codes, big, TOP_FRAC, np.abs(expected) > THRESHOLD * (c_in + c_out))
    held = day_books(w, p["day_id"], "sleeves")
    d = delayed_days(held, p["raw_seg"], bar_ret, steps, DELAY, c_in, c_out, p["day_id"], p["last_of_day"])

    all_dates = p["day_dates"]
    labels = p["day_label_all"]
    book = pd.Series(0.0, index=all_dates)
    book.iloc[d.index.to_numpy()] = (d["g"] - d["c"]).to_numpy()
    pool = pd.Series(pool_daily(meta, all_dates), index=all_dates)
    first_trade = all_dates[d.index[(d["g"] != 0).to_numpy()].min()]
    keep = (labels != "drop") & (all_dates >= first_trade)
    book, pool, labels = book[keep], pool[keep], labels[keep]
    val = labels == "val"

    report = {}
    for split, sel in (("all", np.ones(len(book), dtype=bool)), ("train", ~val), ("val", val)):
        report[split] = {"book": stats(book[sel].to_numpy()), "pool": stats(pool[sel].to_numpy()),
                         "corr": float(np.corrcoef(book[sel], pool[sel])[0, 1])}
        print(f"[{split}] book {report[split]['book']}", flush=True)
        print(f"[{split}] pool {report[split]['pool']}  corr {report[split]['corr']:.2f}", flush=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 9.6), gridspec_kw={"height_ratios": [1.15, 1.0]})
    rv, sv = report["val"], report["all"]
    draw(axes[0], book.index[val], [
        (pool[val].to_numpy(), "#9aa3ad", label("大盘长期持有（97 只等权）", rv["pool"]), 1.6),
        (book[val].to_numpy(), "#1f4e79", label("策略（扣成本后）", rv["book"]), 2.2),
    ], f"验证段 {book.index[val][0]:%Y-%m-%d} 到 {book.index[val][-1]:%Y-%m-%d}：参数只在训练段定，但做法是看过这段之后挑的")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    draw(axes[1], book.index, [
        (pool.to_numpy(), "#9aa3ad", label("大盘长期持有", sv["pool"]), 1.3),
        (book.to_numpy(), "#1f4e79", label("策略", sv["book"]), 1.8),
    ], "从头到尾：灰底是训练段，所有参数都在这里定，这部分偏乐观")
    val_start = book.index[val][0]
    axes[1].axvspan(book.index[0], val_start, color="#eef1f5", zorder=0)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("大盘长期持有 vs 策略", fontsize=16, color="#1c2430")
    fig.text(
        0.5, 0.01,
        "策略：每小时在七个大行业里买一批、卖一批同样多的钱，收盘前全部平掉，不过夜、不加杠杆。"
        "晚 5 分钟成交，价差按 2 分钱、每股佣金 0.35 美分扣除。测试集没有打开。",
        ha="center", fontsize=9, color="#6b7380",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "cs_curve.png", dpi=160)
    plt.close(fig)
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
