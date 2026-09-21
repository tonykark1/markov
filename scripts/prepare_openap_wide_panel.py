#!/usr/bin/env python3
"""Convert a wide Open Asset Pricing long-short return file to the research panel.

This is a fallback for times when the official Google Drive release hits its
public-download quota.  The wide file stores returns in percent, so output is
converted to decimal returns and restricted to a pinned factor universe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def prepare_panel(
    wide_path: Path,
    universe_path: Path,
    *,
    min_months: int = 120,
) -> tuple[pd.DataFrame, list[str]]:
    if min_months < 12:
        raise ValueError("min_months must be >= 12")

    wide = pd.read_csv(wide_path, na_values=["NA"])
    if "date" not in wide.columns:
        raise ValueError("wide factor file must contain a date column")
    wide["date"] = pd.to_datetime(wide["date"], errors="raise")

    requested = [line.strip() for line in universe_path.read_text().splitlines() if line.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("factor universe contains duplicate names")

    available = [name for name in requested if name in wide.columns]
    missing = [name for name in requested if name not in wide.columns]
    if len(available) < 100:
        raise ValueError(
            f"only {len(available)} requested factors are present in the wide file"
        )

    long = wide[["date", *available]].melt(
        id_vars="date", var_name="factor", value_name="ret"
    )
    long["ret"] = pd.to_numeric(long["ret"], errors="coerce") / 100.0
    long = long.dropna(subset=["ret"])

    counts = long.groupby("factor")["ret"].count()
    keep = set(counts[counts >= min_months].index)
    long = long.loc[long["factor"].isin(keep)].copy()
    long = long.sort_values(["factor", "date"]).reset_index(drop=True)
    return long, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wide", type=Path)
    parser.add_argument("universe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-months", type=int, default=120)
    args = parser.parse_args()

    panel, missing = prepare_panel(
        args.wide,
        args.universe,
        min_months=args.min_months,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)

    print(
        f"wrote {panel['factor'].nunique()} factors and {len(panel)} factor-months; "
        f"sample {panel['date'].min().date()} to {panel['date'].max().date()}"
    )
    if missing:
        print("requested names absent from this data vintage: " + ", ".join(missing))


if __name__ == "__main__":
    main()
