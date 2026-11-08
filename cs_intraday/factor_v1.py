"""
First-generation generalized-beta split. Causal. No return-prediction head.

  r_t = B_t f_t + eps_t
  B_t = B_slow[week] + delta[day]
  f_t from cross-sectional regression on B_t known before the bar

Time split is chronological 8:1:1. A 60-minute embargo after each cut is
kept out of val/test metrics so a 30-minute forward window cannot cross
the boundary. Loadings at time t use only earlier bars.

theta is fixed in this file. It is not chosen on val or test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "5m"
OUT_DIR = ROOT / "factor_v1_out"
SPLIT_PATH = ROOT / "data" / "splits.json"

K = 3
QUARTER_DAYS = 63
RHO = 0.9
F_HALFLIFE_BARS = 78
COVERAGE = 0.9
EMBARGO_BARS = 12  # 60 minutes
MIN_NAMES_FRAC = 0.8
MIN_DAY_BARS = 20
CHI2_95 = float(chi2.ppf(0.95, K))


def load_returns() -> pd.DataFrame:
    cols = {}
    for path in sorted(DATA_DIR.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["time", "close"])
        df["time"] = pd.to_datetime(df["time"])
        s = df.drop_duplicates("time").set_index("time")["close"].sort_index()
        cols[path.stem] = s
    close = pd.DataFrame(cols).sort_index()
    return close.pct_change(fill_method=None)


def time_split(index: pd.DatetimeIndex) -> dict:
    n = len(index)
    i_train = int(n * 0.8) - 1
    i_val = int(n * 0.9) - 1
    train_end = index[i_train]
    val_end = index[i_val]
    # Embargo sits at the start of val and the start of test.
    val_start = index[min(i_train + 1 + EMBARGO_BARS, n - 1)]
    test_start = index[min(i_val + 1 + EMBARGO_BARS, n - 1)]
    return {
        "train_end": train_end,
        "val_end": val_end,
        "val_eval_start": val_start,
        "test_eval_start": test_start,
        "embargo_bars": EMBARGO_BARS,
    }


def assign_split(times: pd.DatetimeIndex, bounds: dict) -> np.ndarray:
    labels = np.empty(len(times), dtype=object)
    labels[:] = "train"
    labels[times > bounds["train_end"]] = "embargo_val"
    labels[times >= bounds["val_eval_start"]] = "val"
    labels[times > bounds["val_end"]] = "embargo_test"
    labels[times >= bounds["test_eval_start"]] = "test"
    return labels


def top_loadings(window: np.ndarray) -> np.ndarray | None:
    """Top-K loadings in raw-return units, from a T x N return window.

    Names with too few observations inside this window stay at 0 and do not
    enter the correlation. An all-null column would otherwise poison the
    eigenvectors.
    """
    if window.shape[0] < 200:
        return None
    col_n = np.isfinite(window).sum(axis=0)
    use = col_n >= max(50, int(0.8 * window.shape[0]))
    if use.sum() < K + 5:
        return None
    x = window[:, use]
    row_n = np.isfinite(x).sum(axis=1)
    x = x[row_n >= MIN_NAMES_FRAC * use.sum()]
    if x.shape[0] < 50:
        return None
    col_mean = np.nanmean(x, axis=0)
    col_std = np.nanstd(x, axis=0)
    col_std = np.where(~np.isfinite(col_std) | (col_std < 1e-12), 1.0, col_std)
    z = x - col_mean
    z[~np.isfinite(z)] = 0.0
    z = z / col_std
    c = (z.T @ z) / max(z.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(c)
    order = np.argsort(evals)[::-1][:K]
    evecs = evecs[:, order]
    partial = evecs * col_std.reshape(-1, 1)
    loadings = np.zeros((window.shape[1], K))
    loadings[use] = partial
    return loadings, use


def align_sign(new: np.ndarray, old: np.ndarray | None) -> np.ndarray:
    if old is None:
        return new
    out = new.copy()
    for k in range(out.shape[1]):
        if np.dot(out[:, k], old[:, k]) < 0:
            out[:, k] *= -1
    return out


def project(ret_row: np.ndarray, loadings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.isfinite(ret_row) & np.isfinite(loadings).all(axis=1)
    if obs.sum() < max(10, K + 2):
        f = np.full(K, np.nan)
        eps = np.full(ret_row.shape[0], np.nan)
        return f, eps
    b = loadings[obs]
    y = ret_row[obs]
    gram = b.T @ b + 1e-10 * np.eye(K)
    f = np.linalg.solve(gram, b.T @ y)
    fitted = np.full(ret_row.shape[0], np.nan)
    fitted[obs] = b @ f
    eps = np.full(ret_row.shape[0], np.nan)
    eps[obs] = y - fitted[obs]
    return f, eps


def day_loadings(f_day: np.ndarray, r_day: np.ndarray) -> np.ndarray | None:
    """N x K realized loadings from one day's (f, r). Causal only after the close."""
    ok = np.isfinite(f_day).all(axis=1) & (np.isfinite(r_day).mean(axis=1) >= MIN_NAMES_FRAC)
    if ok.sum() < MIN_DAY_BARS:
        return None
    f = f_day[ok]
    r = r_day[ok]
    out = np.full((r.shape[1], K), np.nan)
    for j in range(r.shape[1]):
        m = np.isfinite(r[:, j])
        if m.sum() < MIN_DAY_BARS:
            continue
        gram = f[m].T @ f[m] + 1e-10 * np.eye(K)
        out[j] = np.linalg.solve(gram, f[m].T @ r[m, j])
    if np.isfinite(out).all(axis=1).sum() < max(10, K + 2):
        return None
    return out


def score_split(values, f_all, maha, mask, fwd_ok, times) -> dict:
    # Rebuild epsilon only on this mask. Loadings are not stored bar-by-bar;
    # epsilon = r - projection is recovered from stored f only if we also
    # stored fitted values. Recompute from f is impossible without B.
    # Variance explained is accumulated during the run via maha? No.
    # We need eps. Store it in a module-level buffer filled in run().
    eps = EPS_BUFFER
    r = values[mask]
    e = eps[mask]
    obs = np.isfinite(r) & np.isfinite(e)
    ss_res = np.nansum((e[obs]) ** 2) if obs.any() else np.nan
    ss_tot = np.nansum((r[obs]) ** 2) if obs.any() else np.nan
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot and ss_tot > 0 else float("nan")

    mh = maha[mask]
    mh = mh[np.isfinite(mh)]
    tail = float(np.mean(mh > CHI2_95)) if len(mh) else float("nan")

    ic = residual_ic(values, eps, fwd_ok)
    n_bars = int(mask.sum())
    if n_bars:
        t0 = str(times[mask][0])
        t1 = str(times[mask][-1])
    else:
        t0 = t1 = ""
    return {
        "n_bars": n_bars,
        "start": t0,
        "end": t1,
        "r2": r2,
        "tail_rate_5pct": tail,
        "resid_ic_30m": ic,
    }


def residual_ic(values: np.ndarray, eps: np.ndarray, fwd_ok: np.ndarray) -> dict:
    idx = np.flatnonzero(fwd_ok)[::6]
    if len(idx) < 50:
        return {"n": int(len(idx)), "ic": None, "t": None}
    # Past 1-bar return vs sum of next 6 residual bars. Both ranked cross-sectionally.
    ics = []
    for t in idx:
        x = values[t]
        y = np.nansum(eps[t + 1 : t + 7], axis=0)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 40:
            continue
        xr = pd.Series(x[m]).rank().to_numpy()
        yr = pd.Series(y[m]).rank().to_numpy()
        xr = xr - xr.mean()
        yr = yr - yr.mean()
        den = np.sqrt((xr ** 2).sum() * (yr ** 2).sum())
        if den < 1e-12:
            continue
        ics.append(float((xr * yr).sum() / den))
    if len(ics) < 50:
        return {"n": len(ics), "ic": None, "t": None}
    z = np.asarray(ics)
    tstat = float(z.mean() / (z.std(ddof=1) / np.sqrt(len(z))))
    return {"n": int(len(z)), "ic": float(z.mean()), "t": tstat}


# Filled in run() before scoring. Kept global so score_split can see residuals
# without writing a multi-gigabyte frame.
EPS_BUFFER: np.ndarray | None = None


def run() -> None:
    global EPS_BUFFER
    print("[load] returns", flush=True)
    ret = load_returns()
    bounds = time_split(ret.index)
    labels = assign_split(ret.index, bounds)
    train_mask = labels == "train"
    coverage = np.isfinite(ret.to_numpy()[train_mask]).mean(axis=0)
    keep = coverage >= COVERAGE
    names = ret.columns[keep].tolist()
    dropped = ret.columns[~keep].tolist()
    ret = ret.loc[:, names]
    values = ret.to_numpy(dtype=np.float64)
    times = ret.index
    n_times, n_names = values.shape

    dates = np.array(times.date)
    change = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    day_starts = change
    day_ends = np.append(change[1:], n_times)
    n_days = len(day_starts)
    iso = pd.DatetimeIndex(times).isocalendar()
    week_key = (iso.year.to_numpy() * 100 + iso.week.to_numpy())

    f_all = np.full((n_times, K), np.nan)
    eps_all = np.full((n_times, n_names), np.nan)
    maha = np.full(n_times, np.nan)
    b_slow = None
    delta = np.zeros((n_names, K))
    lam = float(np.exp(-np.log(2) / F_HALFLIFE_BARS))
    v_f = np.eye(K)

    print(f"[run] {n_names} names, {n_days} days", flush=True)
    for d in range(n_days):
        a, b = int(day_starts[d]), int(day_ends[d])
        if b_slow is not None:
            loadings = b_slow + delta
            r_day = values[a:b]
            if np.isfinite(r_day).all():
                gram = loadings.T @ loadings + 1e-10 * np.eye(K)
                proj = np.linalg.solve(gram, loadings.T)
                f_day = r_day @ proj.T
                eps_all[a:b] = r_day - f_day @ loadings.T
                f_all[a:b] = f_day
            else:
                f_day = np.full((b - a, K), np.nan)
                for j, i in enumerate(range(a, b)):
                    f_i, eps_i = project(values[i], loadings)
                    f_day[j] = f_i
                    f_all[i] = f_i
                    eps_all[i] = eps_i
            for j, i in enumerate(range(a, b)):
                f_i = f_all[i]
                if np.isfinite(f_i).all():
                    cov = v_f + 1e-8 * np.eye(K)
                    maha[i] = float(f_i @ np.linalg.solve(cov, f_i))
                    v_f = lam * v_f + (1.0 - lam) * np.outer(f_i, f_i)
            realized = day_loadings(f_all[a:b], values[a:b])
            if realized is not None:
                upd = np.isfinite(realized).all(axis=1) & np.isfinite(b_slow).all(axis=1)
                delta[upd] = RHO * delta[upd] + (1.0 - RHO) * (realized[upd] - b_slow[upd])

        week_end = d + 1 == n_days or week_key[b - 1] != week_key[day_starts[d + 1]]
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
                    first_time = used & (np.max(np.abs(old_total), axis=1) < 1e-8)
                    delta[first_time] = 0.0
                b_slow = fresh

        if (d + 1) % 200 == 0:
            print(f"  day {d + 1}/{n_days}", flush=True)

    active = np.isfinite(f_all).all(axis=1)
    labels = assign_split(times, bounds)
    labels = np.where(active, labels, "warmup")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "rule": "chronological 8:1:1 by bar; 60-minute embargo after each cut excluded from val/test metrics",
        "train_end": str(bounds["train_end"]),
        "val_eval_start": str(bounds["val_eval_start"]),
        "val_end": str(bounds["val_end"]),
        "test_eval_start": str(bounds["test_eval_start"]),
        "series_end": str(times[-1]),
        "embargo_bars": EMBARGO_BARS,
        "first_active": str(times[int(np.flatnonzero(active)[0])]) if active.any() else None,
        "n_names": n_names,
        "names": names,
        "dropped_lt_90pct_train_coverage": dropped,
        "theta": {"K": K, "quarter_days": QUARTER_DAYS, "rho": RHO, "f_halflife_bars": F_HALFLIFE_BARS, "phi": "identity"},
        "counts": {k: int((labels == k).sum()) for k in ("warmup", "train", "embargo_val", "val", "embargo_test", "test")},
    }
    SPLIT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    global EPS_BUFFER
    EPS_BUFFER = eps_all
    metrics = {}
    for split_name in ("train", "val", "test"):
        m = labels == split_name
        fwd_ok = np.zeros(n_times, dtype=bool)
        idx = np.flatnonzero(m)
        if len(idx):
            idx = idx[idx + 6 < n_times]
            fwd_ok[idx] = labels[idx + 6] == split_name
        metrics[split_name] = score_split(values, f_all, maha, m, fwd_ok, times)
        ic = metrics[split_name]["resid_ic_30m"]
        print(
            f"[{split_name}] {metrics[split_name]['start']} → {metrics[split_name]['end']} "
            f"bars={metrics[split_name]['n_bars']} r2={metrics[split_name]['r2']:.3f} "
            f"tail5={metrics[split_name]['tail_rate_5pct']:.3f} ic={ic}",
            flush=True,
        )

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    out = pd.DataFrame(f_all, index=times, columns=[f"f{i+1}" for i in range(K)])
    out["maha"] = maha
    out["split"] = labels
    out.index.name = "time"
    out.to_parquet(OUT_DIR / "f_t.parquet")
    print(f"[wrote] {SPLIT_PATH}")
    print(f"[wrote] {OUT_DIR}")


if __name__ == "__main__":
    run()
