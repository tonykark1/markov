#!/usr/bin/env python3
"""Download a broad non-momentum anomaly-factor panel from Open Asset Pricing."""

from __future__ import annotations

import argparse
from pathlib import Path

import openassetpricing as oap
import pandas as pd


def _momentum_like_signals(signal_doc: pd.DataFrame) -> set[str]:
    doc = signal_doc.copy()
    text_columns = [
        column
        for column in ("Acronym", "Acronym2", "LongDescription", "Detailed Definition")
        if column in doc.columns
    ]
    text = doc[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    mask = text.str.contains("momentum", regex=False)
    # Also catch compact acronyms such as Mom12m, IndMom, ResidualMomentum, etc.
    if "Acronym" in doc.columns:
        mask |= doc["Acronym"].fillna("").astype(str).str.lower().str.contains("mom", regex=False)
    return set(doc.loc[mask, "Acronym"].astype(str))


def download_panel(*, min_months: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min_months < 12:
        raise ValueError("min_months must be >= 12")

    source = oap.OpenAP()
    ports = source.dl_port("op", "pandas")
    signal_doc = source.dl_signal_doc("pandas")

    ls = ports.loc[ports["port"].astype(str).str.upper() == "LS", ["signalname", "date", "ret"]].copy()
    ls["date"] = pd.to_datetime(ls["date"])
    ls["ret"] = pd.to_numeric(ls["ret"], errors="coerce") / 100.0
    ls = ls.dropna(subset=["ret"])

    excluded = _momentum_like_signals(signal_doc)
    ls = ls.loc[~ls["signalname"].astype(str).isin(excluded)].copy()

    counts = ls.groupby("signalname")["ret"].count()
    keep = set(counts[counts >= min_months].index.astype(str))
    ls = ls.loc[ls["signalname"].astype(str).isin(keep)].copy()
    ls = ls.rename(columns={"signalname": "factor"})[["date", "factor", "ret"]]
    ls = ls.sort_values(["factor", "date"]).reset_index(drop=True)

    metadata = signal_doc.loc[signal_doc["Acronym"].astype(str).isin(keep)].copy()
    useful = [
        column
        for column in (
            "Acronym",
            "LongDescription",
            "Authors",
            "Year",
            "Journal",
            "Cat.Economic",
            "SampleStartYear",
            "SampleEndYear",
        )
        if column in metadata.columns
    ]
    metadata = metadata[useful].sort_values("Acronym").reset_index(drop=True)
    return ls, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/openap_nonmomentum_factors.csv"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/openap_nonmomentum_factor_metadata.csv"),
    )
    parser.add_argument("--min-months", type=int, default=120)
    args = parser.parse_args()

    panel, metadata = download_panel(min_months=args.min_months)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    metadata.to_csv(args.metadata, index=False)

    print(
        f"wrote {panel['factor'].nunique()} non-momentum factors and {len(panel)} factor-months; "
        f"sample {panel['date'].min().date()} to {panel['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
