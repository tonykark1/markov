#!/usr/bin/env python3
"""Run fixed-spec factor-momentum tests on a monthly factor CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from markovlab.factor_momentum import (
    cross_sectional_factor_momentum,
    factor_persistence_table,
    run_factor_momentum_suite,
    time_series_factor_momentum,
)

DEFAULT_COLUMNS = ["SMB", "HML", "RMW", "CMA", "MOM"]


def _load_returns(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required factor columns: {missing}")

    returns = frame[columns].apply(pd.to_numeric, errors="raise")
    if "date" in frame.columns:
        returns.index = pd.PeriodIndex(frame["date"].astype(str), freq="M")

    # Kenneth French source files are percentages; downloader output and many research
    # files are decimals. Auto-detect only when magnitudes are unmistakably percentage-like.
    if returns.abs().stack().quantile(0.99) > 0.75:
        returns = returns / 100.0
    return returns


def _suite_with_universe(returns: pd.DataFrame, universe: str) -> pd.DataFrame:
    summary = run_factor_momentum_suite(returns, n_long=1, n_short=1)
    summary.insert(0, "universe", universe)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLUMNS),
        help="comma-separated factor-return columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/factor_momentum"),
        help="output directory",
    )
    args = parser.parse_args()

    columns = [column.strip() for column in args.columns.split(",") if column.strip()]
    full = _load_returns(args.csv, columns)

    suites = [_suite_with_universe(full, "with_MOM")]
    if "MOM" in full.columns and full.shape[1] > 2:
        suites.append(_suite_with_universe(full.drop(columns="MOM"), "without_MOM"))
    summary = pd.concat(suites, ignore_index=True)

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv", index=False)

    primary_label = "12-1"
    primary_lookback = 11
    primary_skip = 1
    persistence = factor_persistence_table(full, lookback=primary_lookback, skip=primary_skip)
    persistence.to_csv(args.output / "persistence_12-1.csv")

    returns_out = pd.DataFrame(index=full.index)
    for universe, data in (
        ("with_MOM", full),
        ("without_MOM", full.drop(columns="MOM") if "MOM" in full else full),
    ):
        ts = time_series_factor_momentum(data, lookback=primary_lookback, skip=primary_skip)[
            "strategy"
        ].iloc[:, 0]
        cs = cross_sectional_factor_momentum(
            data,
            lookback=primary_lookback,
            skip=primary_skip,
            n_long=1,
            n_short=1,
        )["strategy"].iloc[:, 0]
        returns_out[f"{universe}_time_series_{primary_label}"] = ts
        returns_out[f"{universe}_cross_sectional_{primary_label}"] = cs
    returns_out.to_csv(args.output / "returns_12-1.csv", index_label="date")

    display = summary[
        [
            "universe",
            "spec",
            "strategy",
            "n_months",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "hac_t_stat",
            "hac_p_value",
        ]
    ].copy()
    print(display.to_string(index=False, float_format=lambda x: f"{x: .4f}"))


if __name__ == "__main__":
    main()
