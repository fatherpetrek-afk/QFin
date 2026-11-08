"""
Download the S&P 100 5-minute regular-session panel from Futu OpenD.

Universe: cs_intraday/universe_sp100.csv (Wikipedia constituents as of 2026-09-21).
Bars: 5-minute, forward-adjusted (QFQ), regular trading hours only.
Window: last 8 years (Futu cap for intraday klines). Whatever the account
actually returns is kept; each name is saved on its own so a short listing
does not truncate the others.

One parquet per ticker under cs_intraday/data/5m/. Each name is one request
(OHLCV only) and three names run at a time. Re-running skips files that
already have rows. Pass --force to refetch.

  python fetch_sp100_5m.py
  python fetch_sp100_5m.py --symbols AAPL MSFT
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
UNIVERSE = ROOT / "universe_sp100.csv"
OUT_DIR = ROOT / "data" / "5m"
MANIFEST = ROOT / "data" / "manifest.csv"

# OpenD stays with the original QFin tree on the desktop.
OPEND_DIR = Path(r"C:\Users\ASUS\Desktop\QFin\Futu_OpenD_10.9.6918_Windows\Futu_OpenD_10.9.6918_Windows")
OPEND_EXE = OPEND_DIR / "FutuOpenD.exe"
HOST = "127.0.0.1"
PORT = 11111

# One request per name returns the full 8y of 5m bars (~22s).
# Three overlapping requests stay under the first-page rate limit (~10 / 30s).
WORKERS = 3


def _port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_opend() -> None:
    """Attach to an OpenD that is already logged in.

    Do not launch FutuOpenD.exe from here. The password stored in
    FutuOpenD.xml is rejected by the server, and each automatic start
    spends one of the remaining login attempts.
    """
    if _port_open(HOST, PORT):
        print(f"[opend] already listening on {HOST}:{PORT}")
        return
    raise RuntimeError(
        "FutuOpenD is not listening on 127.0.0.1:11111. "
        f"Open {OPEND_EXE} yourself, log in with the current password, "
        "leave that window open, then re-run this script. "
        "The password in FutuOpenD.xml does not match the account."
    )


def _is_rate_limit(msg: str) -> bool:
    low = msg.lower()
    return ("频率" in msg) or ("too frequent" in low) or ("rate" in low and "limit" in low)


def fetch_one(quote_ctx, code: str, start: str, end: str) -> pd.DataFrame:
    from futu import AuType, KLType, KL_FIELD, RET_OK, Session

    fields = [
        KL_FIELD.DATE_TIME,
        KL_FIELD.OPEN,
        KL_FIELD.HIGH,
        KL_FIELD.LOW,
        KL_FIELD.CLOSE,
        KL_FIELD.TRADE_VOL,
    ]
    last_err = None
    data = None
    for _attempt in range(6):
        ret, data, _next_key = quote_ctx.request_history_kline(
            code,
            start=start,
            end=end,
            ktype=KLType.K_5M,
            autype=AuType.QFQ,
            fields=fields,
            max_count=None,
            extended_time=False,
            session=Session.RTH,
        )
        if ret == RET_OK:
            break
        last_err = str(data)
        if _is_rate_limit(last_err):
            print(f"    rate limit on {code}, sleep 31s", flush=True)
            time.sleep(31)
            continue
        raise RuntimeError(f"Futu history failed for {code}: {last_err}")
    else:
        raise RuntimeError(f"Futu history failed for {code} after retries: {last_err}")

    if data is None or len(data) == 0:
        return pd.DataFrame()
    out = data.drop_duplicates(subset=["time_key"]).sort_values("time_key")
    keep = [c for c in ["time_key", "open", "high", "low", "close", "volume"] if c in out.columns]
    out = out[keep].rename(columns={"time_key": "time"})
    out["time"] = pd.to_datetime(out["time"])
    print(f"    {code} {len(out)} bars  {out['time'].iloc[0]} → {out['time'].iloc[-1]}", flush=True)
    return out


def _parquet_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_")
    return OUT_DIR / f"{safe}.parquet"


def _append_manifest(row: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if MANIFEST.exists():
        prev = pd.read_csv(MANIFEST)
        prev = prev[prev["symbol"] != row["symbol"]]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(MANIFEST, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch S&P 100 5m RTH panel via Futu")
    parser.add_argument("--symbols", nargs="*", default=None, help="Subset, default = full universe")
    parser.add_argument("--force", action="store_true", help="Refetch even if parquet exists")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (default: today-8y)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    universe = pd.read_csv(UNIVERSE)
    if args.symbols:
        want = {s.upper() for s in args.symbols}
        universe = universe[universe["symbol"].str.upper().isin(want)]
        missing = want - set(universe["symbol"].str.upper())
        if missing:
            raise SystemExit(f"Not in universe file: {sorted(missing)}")
    if universe.empty:
        raise SystemExit("Empty universe.")

    end = args.end or datetime.today().date().isoformat()
    start = args.start or (datetime.today().date() - timedelta(days=365 * 8)).isoformat()
    print(f"[fetch] {len(universe)} names | 5m QFQ RTH | {start} → {end}")
    print(f"[fetch] out {OUT_DIR}")

    ensure_opend()
    from futu import OpenQuoteContext, RET_OK

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quote_ctx = OpenQuoteContext(host=HOST, port=PORT)
    try:
        ret, quota = quote_ctx.get_history_kl_quota(get_detail=False)
        if ret == RET_OK:
            print(f"[quota] {quota}")
        else:
            print(f"[quota] unavailable: {quota}")
    finally:
        quote_ctx.close()

    jobs = []
    n = len(universe)
    for i, rec in enumerate(universe.itertuples(index=False), start=1):
        symbol = str(rec.symbol).upper()
        dest = _parquet_path(symbol)
        if dest.exists() and not args.force:
            try:
                n_bars = len(pd.read_parquet(dest, columns=["time"]))
            except Exception:
                n_bars = 0
            if n_bars:
                print(f"[{i}/{n}] {symbol} skip ({n_bars} bars on disk)")
                continue
        jobs.append((i, symbol, str(rec.sector), dest))

    print(f"[fetch] {len(jobs)} to download, {n - len(jobs)} already on disk, workers={WORKERS}")
    manifest_lock = threading.Lock()

    def _one(job):
        i, symbol, sector, dest = job
        code = f"US.{symbol}"
        print(f"[{i}/{n}] {code}", flush=True)
        row = {
            "symbol": symbol,
            "sector": sector,
            "n_bars": 0,
            "start": "",
            "end": "",
            "path": str(dest.relative_to(ROOT)),
            "error": "",
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }
        ctx = OpenQuoteContext(host=HOST, port=PORT)
        try:
            hist = fetch_one(ctx, code, start, end)
            if hist.empty:
                row["error"] = "empty"
                print(f"    WARN empty {code}", flush=True)
            else:
                hist.to_parquet(dest, index=False)
                row["n_bars"] = int(len(hist))
                row["start"] = str(hist["time"].iloc[0])
                row["end"] = str(hist["time"].iloc[-1])
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
            print(f"    ERROR {code}: {exc}", flush=True)
        finally:
            ctx.close()
        with manifest_lock:
            _append_manifest(row)
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futures):
            fut.result()

    if MANIFEST.exists():
        man = pd.read_csv(MANIFEST)
        ok = man[man["error"].fillna("") == ""]
        bad = man[man["error"].fillna("") != ""]
        print(f"[done] ok={len(ok)} empty_or_error={len(bad)} manifest={MANIFEST}")
        if len(ok):
            print(f"[done] earliest start {ok['start'].min()} | latest end {ok['end'].max()}")
            short = ok.nsmallest(5, "n_bars")[["symbol", "n_bars", "start", "end"]]
            print("[done] shortest histories:")
            print(short.to_string(index=False))


if __name__ == "__main__":
    main()
