"""USGS Water Services client.

English:
    This module fetches daily values (dv) or instantaneous values (iv) from the
    USGS Water Services API and returns a tidy pandas.DataFrame.

日本語:
    USGS Water Services APIから日値(dv)または瞬時値(iv)を取得し、
    tidy形式のpandas.DataFrameとして返します。

Notes:
    - We use JSON output because it is easy to parse.
    - The service may return provisional values; treat them accordingly.

References:
    USGS Water Services docs (dv/iv) are linked in the README.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple
import requests
import pandas as pd

from ..config import USGSQuery


class USGSClient:
    """Small wrapper around the USGS Water Services endpoints."""

    BASE = "https://waterservices.usgs.gov/nwis"

    def __init__(self, timeout_s: int = 30):
        self.timeout_s = timeout_s

    def fetch(self, q: USGSQuery) -> pd.DataFrame:
        """Fetch a single time series.

        EN:
            Returns columns: ['datetime', 'value', 'site', 'parameter_code', 'unit'].

        JP:
            返り値の列: ['datetime', 'value', 'site', 'parameter_code', 'unit']。
        """
        if q.service not in {"dv", "iv"}:
            raise ValueError("service must be 'dv' or 'iv'")

        # EN: dv expects startDT/endDT; iv uses the same naming in Water Services.
        # JP: dv/ivともにstartDT/endDTを受け付けます。
        url = f"{self.BASE}/{q.service}/"
        params = {
            "format": "json",
            "sites": q.site,
            "parameterCd": q.parameter_code,
            "startDT": q.start,
            "endDT": q.end,
            # dv: by default, daily mean (statCd=00003) for many params; keep default.
        }
        resp = requests.get(url, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        payload: Dict[str, Any] = resp.json()

        # EN: Water Services JSON has a 'value' section with a timeSeries list.
        # JP: JSONの'value'->'timeSeries'に時系列が入っています。
        ts_list = payload.get("value", {}).get("timeSeries", [])
        if not ts_list:
            raise ValueError(f"No timeSeries found for {q.site} / {q.parameter_code}")

        # Usually one timeSeries for a single site+param.
        series = ts_list[0]
        unit = (
            series.get("variable", {})
            .get("unit", {})
            .get("unitCode", "")
        )

        values = series.get("values", [])
        if not values or not values[0].get("value"):
            raise ValueError("No values returned (empty or missing).")

        rows = []
        for item in values[0]["value"]:
            # datetime is ISO string; value can be string.
            rows.append(
                {
                    "datetime": item["dateTime"],
                    "value": float(item["value"]) if item["value"] not in {"", None} else None,
                    "site": q.site,
                    "parameter_code": q.parameter_code,
                    "unit": unit,
                    "qualifiers": item.get("qualifiers", []),
                }
            )

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
        df = df.sort_values("datetime").reset_index(drop=True)
        return df


def fetch_to_csv(q: USGSQuery, out_csv: str, timeout_s: int = 30) -> str:
    """Convenience helper: fetch and save to CSV."""
    client = USGSClient(timeout_s=timeout_s)
    df = client.fetch(q)
    df.to_csv(out_csv, index=False)
    return out_csv
