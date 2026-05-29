"""Shared helpers for Track A modules — sample-path resolution and DataFrame loading."""

from __future__ import annotations

import os
import sys

import pandas as pd


def sample_dir(output_base: str, sample: str) -> str:
    """Resolve `<output_base>/<sample>`. Created lazily by the loader."""
    return os.path.join(output_base, sample)


def load_sample(sample: str, output_base: str = "results/track_a") -> pd.DataFrame:
    """Load the `loaded.csv` produced by `survey-loader` for a named sample.

    Raises `SystemExit` with a clear message if the sample directory or
    `loaded.csv` is missing — every Track A module needs a loaded sample first.
    """
    path = os.path.join(output_base, sample, "loaded.csv")
    if not os.path.exists(path):
        print(
            f"error: no loaded sample at {path}\n"
            f"  hint: run `knoll-tools survey-loader --synthesize-n 200 --sample {sample}` first",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return pd.read_csv(path)
