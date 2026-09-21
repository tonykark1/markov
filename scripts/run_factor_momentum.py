#!/usr/bin/env python3
"""Run a small FF factor-return momentum sanity check.

The main broad-universe experiment is in ``run_factor_momentum_panel.py``.
This script intentionally excludes the stock momentum factor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from markovlab.factor_momentum import run_factor_momentum_suite

DEFAULT_COLUMNS = ["SMB", "HML", "RMW", "CMA"]


def _load_returns(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required factor columns: {missing}")

    returns = frame[columns].apply(pd.to_numeric, errors="raise")
    if "date" in frame.columns:
        returns.index = pd.PeriodIndex(frame["date"].astype(str), freq="M")
    if returns.abs().stack().quantile(0.99) > 0.75:
        returns = returns / 100.0
    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLUMNS),
        help="comma-separated non-momentum factor-return columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/factor_momentum_ff_sanity"),
    )
    args = parser.parse_args()

    columns = [column.strip() for column in args.columns.split(",") if column.strip()]
    returns = _load_returns(args.csv, columns)
    summary = run_factor_momentum_suite(
        returns,
        specs=(("1m", 1, 0), ("3m", 3, 0), ("6m", 6, 0), ("12m", 12, 0)),
        n_long=1,
        n_short=1,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value: .4f}"))


if __name__ == "__main__":
    main()
