"""
Split and dividend factors from Futu OpenD, so adjusted bars can be turned back
into the prices that were actually quoted.

The 5-minute panels are forward-adjusted: every price before a split or a
dividend is scaled by the factors of all later events up to the fetch date.
A cent is a cent of the real price, so costs and a price floor need the real
price. get_rehab does not spend the history quota.

  python fetch_rehab.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from fetch_sp100_5m import HOST, PORT, ensure_opend

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "rehab"
UNIVERSES = ("universe_sp100.csv", "universe_flat.csv")
PAUSE = 0.6


def main() -> None:
    ensure_opend()
    from futu import OpenQuoteContext, RET_OK

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted({s for u in UNIVERSES for s in pd.read_csv(ROOT / u)["symbol"]})
    ctx = OpenQuoteContext(host=HOST, port=PORT)
    failed = []
    try:
        for sym in symbols:
            dest = OUT_DIR / f"{sym.replace('.', '_')}.csv"
            if dest.exists():
                continue
            ret, data = ctx.get_rehab(f"US.{sym}")
            if ret != RET_OK:
                failed.append((sym, str(data)))
                print(f"  {sym}: {data}", flush=True)
            else:
                data.to_csv(dest, index=False)
            time.sleep(PAUSE)
    finally:
        ctx.close()
    print(f"[done] {len(symbols) - len(failed)} saved, failed {failed}", flush=True)


if __name__ == "__main__":
    main()
