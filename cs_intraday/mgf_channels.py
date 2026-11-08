"""
Approximate the external-factor distribution from its moment generating function.

Five channels, one per OHLCV field. No named family is fitted. The real-axis
MGF is reported for stability. The values actually inverted are the MGF on the
imaginary axis, which is the characteristic function and exists without a
tail assumption. Train builds the transform. Validation and test only score it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest

from factor_v1 import (
    F_HALFLIFE_BARS,
    K,
    QUARTER_DAYS,
    RHO,
    align_sign,
    assign_split,
    day_loadings,
    project,
    time_split,
    top_loadings,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "5m"
SPLIT_PATH = ROOT / "data" / "splits.json"
OUT_DIR = ROOT / "mgf_v1_out"

CHANNELS = ("open", "high", "low", "close", "volume")
LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
U_MAX = 8.0
N_FREQ = 161
MGF_T = np.linspace(-0.4, 0.4, 17)


def load_panels(names: list[str]) -> dict[str, pd.DataFrame]:
    frames = []
    for name in names:
        df = pd.read_parquet(DATA_DIR / f"{name}.parquet")
        df["time"] = pd.to_datetime(df["time"])
        df = df.drop_duplicates("time").set_index("time").sort_index()
        frames.append(df[["open", "high", "low", "close", "volume"]])
    index = frames[0].index
    for frame in frames[1:]:
        index = index.union(frame.index)
    index = index.sort_values()
    fields = {}
    for field in ("open", "high", "low", "close", "volume"):
        cols = {}
        for name, frame in zip(names, frames):
            cols[name] = frame[field].reindex(index)
        fields[field] = pd.DataFrame(cols, index=index)
    return fields


def channel_panels(fields: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prev = fields["close"].shift(1)
    out = {
        "open": fields["open"] / prev - 1.0,
        "high": fields["high"] / prev - 1.0,
        "low": fields["low"] / prev - 1.0,
        "close": fields["close"] / prev - 1.0,
        "volume": np.log1p(fields["volume"]).diff(),
    }
    return out


def extract_factor(values: np.ndarray) -> np.ndarray:
    """Causal K-factor. Same clock as factor_v1. Returns T x K."""
    n_times, n_names = values.shape
    # Day boundaries are unknown here; the caller passes a dummy. Recomputed
    # by the wrapper which has the timestamps. This function receives values
    # only, so day cuts are supplied via the attribute set by extract_on_index.
    day_starts = extract_factor.day_starts
    day_ends = extract_factor.day_ends
    week_key = extract_factor.week_key
    n_days = len(day_starts)
    f_all = np.full((n_times, K), np.nan)
    b_slow = None
    delta = np.zeros((n_names, K))
    for d in range(n_days):
        a, b = int(day_starts[d]), int(day_ends[d])
        if b_slow is not None:
            loadings = b_slow + delta
            r_day = values[a:b]
            if np.isfinite(r_day).all():
                gram = loadings.T @ loadings + 1e-10 * np.eye(K)
                proj = np.linalg.solve(gram, loadings.T)
                f_all[a:b] = r_day @ proj.T
            else:
                for i in range(a, b):
                    f_i, _ = project(values[i], loadings)
                    f_all[i] = f_i
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
    return f_all


def prepare_calendar(index: pd.DatetimeIndex) -> None:
    n_times = len(index)
    dates = np.array(index.date)
    change = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    extract_factor.day_starts = change
    extract_factor.day_ends = np.append(change[1:], n_times)
    iso = pd.DatetimeIndex(index).isocalendar()
    extract_factor.week_key = iso.year.to_numpy() * 100 + iso.week.to_numpy()


def align_to_market(f1: np.ndarray, market: np.ndarray, train: np.ndarray) -> np.ndarray:
    m = train & np.isfinite(f1) & np.isfinite(market)
    if m.sum() < 100:
        return f1
    corr = np.corrcoef(f1[m], market[m])[0, 1]
    if np.isfinite(corr) and corr < 0:
        return -f1
    return f1


def causal_scale(f: np.ndarray) -> np.ndarray:
    z = np.full(len(f), np.nan)
    finite = np.isfinite(f)
    idx = np.flatnonzero(finite)
    if len(idx) <= F_HALFLIFE_BARS + 5:
        return z
    burn = F_HALFLIFE_BARS
    var = float(np.var(f[idx[:burn]]))
    lam = float(np.exp(-np.log(2.0) / F_HALFLIFE_BARS))
    for i in idx[burn:]:
        z[i] = f[i] / np.sqrt(var + 1e-12)
        var = lam * var + (1.0 - lam) * float(f[i] ** 2)
    return z


def empirical_cf(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    out = np.zeros(len(u), dtype=np.complex128)
    n = 0
    for start in range(0, len(z), 25000):
        block = z[start : start + 25000]
        out += np.exp(1j * np.outer(u, block)).sum(axis=1)
        n += len(block)
    return out / max(n, 1)


def empirical_mgf(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Real-axis MGF via logsumexp. Heavy tails make this a max, not a mean."""
    out = np.empty(len(t))
    log_n = np.log(len(z))
    for i, ti in enumerate(t):
        out[i] = float(np.exp(np.logaddexp.reduce(ti * z) - log_n))
    return out


def mgf_tail_share(z: np.ndarray, t: float) -> float:
    """Fraction of the MGF sum coming from the largest 0.1% of |z|."""
    contrib = np.exp(t * z)
    total = contrib.sum()
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    k = max(1, int(0.001 * len(z)))
    top = np.argpartition(np.abs(z), -k)[-k:]
    return float(contrib[top].sum() / total)


def gil_pelaez_cdf(z: np.ndarray, phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """CDF from the characteristic function. t > 0, phi = E[e^{i t z}] on train."""
    # integrand Im(e^{-i t x} phi(t)) / t
    cdf = np.empty(len(z))
    step = 4000
    for start in range(0, len(z), step):
        x = z[start : start + step]
        wave = np.exp(-1j * np.outer(x, t)) * phi
        integrand = np.imag(wave) / t
        cdf[start : start + len(x)] = 0.5 - np.trapz(integrand, t, axis=1) / np.pi
    return cdf


def density_from_cf(phi: np.ndarray, u: np.ndarray, x: np.ndarray) -> np.ndarray:
    du = float(u[1] - u[0])
    wave = np.exp(-1j * np.outer(x, u))
    pdf = np.real(wave @ phi) * du / (2.0 * np.pi)
    pdf = np.clip(pdf, 0.0, None)
    mass = pdf.sum() * float(x[1] - x[0])
    if mass > 0:
        pdf /= mass
    return pdf


def reliability(p: np.ndarray) -> dict:
    rows = []
    for level in LEVELS:
        hit = float(np.mean(p <= level))
        rows.append({"nominal": level, "empirical": hit, "gap": hit - level})
    finite = p[np.isfinite(p)]
    ks = kstest(np.clip(finite, 0.0, 1.0), "uniform")
    return {
        "n": int(len(p)),
        "mean_p": float(np.mean(p)),
        "outside_unit": float(np.mean((p < 0.0) | (p > 1.0))),
        "ks_stat": float(ks.statistic),
        "levels": rows,
    }


def hist_density(z: np.ndarray, edges: np.ndarray) -> list[float]:
    count, _ = np.histogram(z, bins=edges)
    width = float(edges[1] - edges[0])
    total = count.sum() * width
    if total <= 0:
        return [0.0] * len(count)
    return [float(v) for v in count / total]


def main() -> None:
    names = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["names"]
    print("[load] OHLCV", flush=True)
    fields = load_panels(names)
    panels = channel_panels(fields)
    index = panels["close"].index
    prepare_calendar(index)
    labels = assign_split(index, time_split(index))
    train = labels == "train"
    val = labels == "val"
    test = labels == "test"

    u = np.linspace(-U_MAX, U_MAX, N_FREQ)
    t_pos = u[u > 0]
    x_grid = np.linspace(-5.0, 5.0, 41)
    hist_edges = np.linspace(-5.0, 5.0, 22)
    hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    chart = {"frequencies": [round(float(v), 3) for v in u[::4]], "channels": {}}
    summary = {"channels": {}, "joint": {}}
    scaled = {}

    for name in CHANNELS:
        print(f"[factor] {name}", flush=True)
        values = panels[name].to_numpy(dtype=np.float64)
        f_all = extract_factor(values)
        market = np.nanmean(values, axis=1)
        f1 = align_to_market(f_all[:, 0].copy(), market, train)
        z = causal_scale(f1)
        scaled[name] = z
        usable = {
            "train": z[train & np.isfinite(z)],
            "val": z[val & np.isfinite(z)],
            "test": z[test & np.isfinite(z)],
        }
        print(f"  n train/val/test {len(usable['train'])} {len(usable['val'])} {len(usable['test'])}", flush=True)
        phi_train = empirical_cf(usable["train"], u)
        phi_val = empirical_cf(usable["val"], u)
        phi_test = empirical_cf(usable["test"], u)
        phi_pos = empirical_cf(usable["train"], t_pos)
        cdf_val = gil_pelaez_cdf(usable["val"], phi_pos, t_pos)
        cdf_test = gil_pelaez_cdf(usable["test"], phi_pos, t_pos)
        pdf = density_from_cf(phi_train, u, x_grid)
        mgf = empirical_mgf(usable["train"], MGF_T)
        share = {
            str(round(float(t), 2)): mgf_tail_share(usable["train"], float(t))
            for t in (-0.2, -0.1, 0.1, 0.2)
        }
        rel_val = reliability(cdf_val)
        rel_test = reliability(cdf_test)
        # Characteristic-function distance, the imaginary-axis MGF.
        cf_gap = {
            "val": float(np.mean(np.abs(phi_train - phi_val))),
            "test": float(np.mean(np.abs(phi_train - phi_test))),
        }
        summary["channels"][name] = {
            "n": {k: int(len(v)) for k, v in usable.items()},
            "std": {k: float(np.std(v)) for k, v in usable.items()},
            "real_mgf_train": {str(round(float(t), 2)): float(v) for t, v in zip(MGF_T, mgf)},
            "real_mgf_top_0.1pct_share": share,
            "cf_l1_vs_train": cf_gap,
            "val": rel_val,
            "test": rel_test,
        }
        step = 4
        chart["channels"][name] = {
            "cf_real": {
                "train": [round(float(v), 5) for v in np.real(phi_train)[::step]],
                "val": [round(float(v), 5) for v in np.real(phi_val)[::step]],
                "test": [round(float(v), 5) for v in np.real(phi_test)[::step]],
            },
            "cf_imag": {
                "train": [round(float(v), 5) for v in np.imag(phi_train)[::step]],
                "val": [round(float(v), 5) for v in np.imag(phi_val)[::step]],
                "test": [round(float(v), 5) for v in np.imag(phi_test)[::step]],
            },
            "density_x": [round(float(v), 3) for v in x_grid],
            "density_train_inversion": [round(float(v), 5) for v in pdf],
            "hist_x": [round(float(v), 3) for v in hist_centers],
            "hist_val": [round(v, 5) for v in hist_density(usable["val"], hist_edges)],
            "hist_test": [round(v, 5) for v in hist_density(usable["test"], hist_edges)],
        }
        print(
            f"  cf L1 val {cf_gap['val']:.4f} test {cf_gap['test']:.4f} "
            f"KS val {rel_val['ks_stat']:.3f} test {rel_test['ks_stat']:.3f}",
            flush=True,
        )

    # Joint transform along each axis is already above. Add the equal-weight direction.
    print("[joint] diagonal characteristic function", flush=True)
    Z = np.column_stack([scaled[name] for name in CHANNELS])
    finite_row = np.isfinite(Z).all(axis=1)
    direction = np.ones(5) / np.sqrt(5.0)
    joint_gap = {}
    for split_name, mask in ("train", train), ("val", val), ("test", test):
        rows = Z[mask & finite_row]
        proj = rows @ direction
        joint_gap[split_name] = empirical_cf(proj, u)
    summary["joint"] = {
        "direction": "equal weight across the five scaled channel factors",
        "cf_l1_val": float(np.mean(np.abs(joint_gap["train"] - joint_gap["val"]))),
        "cf_l1_test": float(np.mean(np.abs(joint_gap["train"] - joint_gap["test"]))),
        "n": {
            "train": int((train & finite_row).sum()),
            "val": int((val & finite_row).sum()),
            "test": int((test & finite_row).sum()),
        },
    }
    print(
        f"[joint] L1 val {summary['joint']['cf_l1_val']:.4f} test {summary['joint']['cf_l1_test']:.4f}",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "chart_data.json").write_text(json.dumps(chart), encoding="utf-8")
    print(f"[wrote] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
