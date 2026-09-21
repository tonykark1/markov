#!/usr/bin/env python3
"""Download and merge the current Kenneth French monthly FF5 and momentum factors."""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


def _download_csv_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "markovlab-factor-momentum/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"no CSV found in {url}")
        return archive.read(names[0]).decode("utf-8-sig", errors="replace")


def _monthly_rows(text: str, value_names: list[str]) -> pd.DataFrame:
    rows: list[list[str]] = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        date = row[0].strip()
        if re.fullmatch(r"\d{6}", date) and len(row) >= len(value_names) + 1:
            rows.append([date, *[cell.strip() for cell in row[1 : len(value_names) + 1]]])
    if not rows:
        raise RuntimeError("could not identify monthly factor rows")
    frame = pd.DataFrame(rows, columns=["date", *value_names])
    frame["date"] = pd.PeriodIndex(frame["date"], freq="M").astype(str)
    for column in value_names:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def download_french_factors() -> pd.DataFrame:
    ff5 = _monthly_rows(
        _download_csv_text(FF5_URL),
        ["MKT", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    mom = _monthly_rows(_download_csv_text(MOM_URL), ["MOM"])
    merged = ff5.merge(mom, on="date", how="inner", validate="one_to_one")
    factor_columns = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]
    merged[factor_columns] = merged[factor_columns] / 100.0
    return merged[["date", *factor_columns]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/french_factors_current.csv"))
    args = parser.parse_args()

    frame = download_french_factors()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        f"wrote {len(frame)} monthly observations "
        f"({frame['date'].iloc[0]} to {frame['date'].iloc[-1]}) to {args.output}"
    )


if __name__ == "__main__":
    main()
