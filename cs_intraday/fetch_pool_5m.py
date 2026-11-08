"""
Download a second 5-minute panel from Futu OpenD, for a pool other than the S&P 100.

Same request as fetch_sp100_5m.py: 5-minute, forward-adjusted, regular hours,
the last 8 years, one request per name, three names at a time. Each new name
spends one unit of the 30-day history quota, so the list is checked against
the remaining quota before anything is requested.

  python fetch_pool_5m.py --universe universe_flat.csv --out data/5m_flat
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from fetch_sp100_5m import HOST, PORT, WORKERS, ensure_opend, fetch_one

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep-quota", type=int, default=5, help="Stop if fewer than this many units would remain")
    args = parser.parse_args()

    universe = pd.read_csv(ROOT / args.universe)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    end = datetime.today().date().isoformat()
    start = (datetime.today().date() - timedelta(days=365 * 8)).isoformat()

    jobs = []
    for rec in universe.itertuples(index=False):
        symbol = str(rec.symbol).upper()
        dest = out_dir / f"{symbol.replace('.', '_')}.parquet"
        if dest.exists() and len(pd.read_parquet(dest, columns=["time"])):
            continue
        jobs.append((symbol, str(rec.sector), dest))

    ensure_opend()
    from futu import OpenQuoteContext, RET_OK

    ctx = OpenQuoteContext(host=HOST, port=PORT)
    try:
        ret, quota = ctx.get_history_kl_quota(get_detail=False)
    finally:
        ctx.close()
    if ret != RET_OK:
        raise SystemExit(f"quota unavailable: {quota}")
    used, remain = int(quota[0]), int(quota[1])
    print(f"[quota] used {used} remain {remain} | to fetch {len(jobs)} | {start} -> {end}", flush=True)
    if remain - len(jobs) < args.keep_quota:
        raise SystemExit("Not enough history quota for this list.")

    lock = threading.Lock()
    rows = []

    def one(job):
        symbol, sector, dest = job
        row = {"symbol": symbol, "sector": sector, "n_bars": 0, "start": "", "end": "", "error": ""}
        c = OpenQuoteContext(host=HOST, port=PORT)
        try:
            hist = fetch_one(c, f"US.{symbol}", start, end)
            if hist.empty:
                row["error"] = "empty"
            else:
                hist.to_parquet(dest, index=False)
                row.update(n_bars=int(len(hist)), start=str(hist["time"].iloc[0]), end=str(hist["time"].iloc[-1]))
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
            print(f"    ERROR {symbol}: {exc}", flush=True)
        finally:
            c.close()
        with lock:
            rows.append(row)
            pd.DataFrame(rows).to_csv(manifest, index=False)
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            fut.result()
    man = pd.DataFrame(rows)
    print(f"[done] ok {int((man['error'] == '').sum())} failed {int((man['error'] != '').sum())}", flush=True)
    print(man.sort_values("n_bars").head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
