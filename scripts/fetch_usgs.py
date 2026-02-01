#!/usr/bin/env python3
"""Fetch USGS data and save to CSV.

Example:
  python scripts/fetch_usgs.py --site 09380000 --param 00060 --start 2010-01-01 --end 2024-12-31 --service dv --out data/raw/usgs.csv
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# EN: Allow running as a script without install.
# JP: インストール前でも実行できるようにパスを追加。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrobalance.config import USGSQuery
from hydrobalance.data.usgs_client import fetch_to_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    p.add_argument("--param", required=True, help="USGS parameter code, e.g., 00065 gage height, 00060 discharge.")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--service", default="dv", choices=["dv", "iv"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    q = USGSQuery(site=args.site, parameter_code=args.param, start=args.start, end=args.end, service=args.service)
    fetch_to_csv(q, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
