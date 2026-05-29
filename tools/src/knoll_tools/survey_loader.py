#!/usr/bin/env python3
"""survey_loader.py — Track A Phase A3 survey loader.

Loads a real survey CSV (`--csv`) or synthesises a plausible 12-item + 6-correlate
dataset (`--synthesize-n N`). The output is a pandas DataFrame written to
`results/track_a/<sample>/loaded.csv`, used by every downstream Track A module.

The synthetic path is **explicitly synthetic** (each output is tagged with a
`synthetic=True` column) so no one mistakes it for real data. It draws a 4-factor
mixture with positive within-subscale loadings and modest between-subscale
correlations, calibrated to Knoll Study 2 marginals.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# 12 items in canonical AS, QS, PS, OS order (3 per subscale).
ITEM_NAMES = [
    "AS1", "AS2", "AS3",
    "QS1", "QS2", "QS3",
    "PS1", "PS2", "PS3",
    "OS1", "OS2", "OS3",
]
SUBSCALE_OF: dict[str, str] = {
    **{f"AS{i+1}": "AS" for i in range(3)},
    **{f"QS{i+1}": "QS" for i in range(3)},
    **{f"PS{i+1}": "PS" for i in range(3)},
    **{f"OS{i+1}": "OS" for i in range(3)},
}
# 6 correlate scales (Knoll Study 2)
CORRELATE_NAMES = [
    "climate_of_silence",
    "job_satisfaction",
    "org_identification",
    "strain",
    "wellbeing",
    "turnover_intention",
]


def synthesize(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthesise a 4-factor 12-item + 6-correlate dataset.

    Items per subscale share a latent factor with loading 0.7; between-subscale
    correlations are 0.3. Correlates have signs matching Knoll Table 2 paper
    values (AS ↔ climate r ≈ .65, PS ⊥ climate, OS ↔ wellbeing r ≈ -.45, etc.).
    """
    rng = np.random.default_rng(seed)
    # Latent factor loadings (rows = items, cols = factors AS / QS / PS / OS)
    loadings = np.zeros((12, 4))
    for i in range(12):
        loadings[i, i // 3] = 0.7
    # Latent factor correlations: AS-QS = .55, AS-OS = .3, QS-OS = .3, PS independent of all
    factor_corr = np.array([
        [1.00, 0.55, 0.30, 0.30],
        [0.55, 1.00, 0.30, 0.30],
        [0.30, 0.30, 1.00, 0.20],
        [0.30, 0.30, 0.20, 1.00],
    ])
    L = np.linalg.cholesky(factor_corr)
    factors = rng.standard_normal((n, 4)) @ L.T

    # Item responses: factor + noise, then Likert-scale to 1..7.
    raw = factors @ loadings.T + 0.45 * rng.standard_normal((n, 12))
    items = np.clip(np.round(raw * 1.2 + 4.0), 1, 7)
    df = pd.DataFrame(items.astype(int), columns=ITEM_NAMES)

    # Correlate scales: built to match Knoll Table 2 marginals.
    climate = 0.65 * factors[:, 0] + 0.40 * factors[:, 1] + 0.35 * factors[:, 3] + 0.45 * rng.standard_normal(n)
    job_sat = -0.50 * factors[:, 0] - 0.20 * factors[:, 1] + 0.6 * rng.standard_normal(n)
    org_id = -0.42 * factors[:, 0] + 0.05 * factors[:, 2] + 0.6 * rng.standard_normal(n)
    strain = 0.30 * factors[:, 0] + 0.30 * factors[:, 1] + 0.25 * factors[:, 3] + 0.6 * rng.standard_normal(n)
    wellbeing = -0.25 * factors[:, 0] - 0.10 * factors[:, 1] - 0.45 * factors[:, 3] + 0.6 * rng.standard_normal(n)
    turn = 0.36 * factors[:, 0] + 0.21 * factors[:, 1] + 0.15 * factors[:, 2] + 0.21 * factors[:, 3] + 0.6 * rng.standard_normal(n)

    df["climate_of_silence"] = climate
    df["job_satisfaction"] = job_sat
    df["org_identification"] = org_id
    df["strain"] = strain
    df["wellbeing"] = wellbeing
    df["turnover_intention"] = turn
    df["synthetic"] = True
    return df


def load_real_csv(path: str) -> pd.DataFrame:
    """Load a real-survey CSV. Expected columns include AS1..OS3 and the 6 correlates."""
    df = pd.read_csv(path)
    missing = [c for c in ITEM_NAMES if c not in df.columns]
    if missing:
        print(
            f"[survey-loader] warning: missing item columns: {missing}",
            file=sys.stderr,
        )
    df["synthetic"] = False
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knoll-tools survey-loader")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="path to a real survey CSV (Phase A3+)")
    src.add_argument(
        "--synthesize-n",
        type=int,
        default=None,
        help="synthesise an N-row dataset (smoke-testing / scaffolding)",
    )
    parser.add_argument(
        "--sample",
        required=True,
        help="sample identifier (output goes to results/track_a/<sample>/loaded.csv)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-base", default="results/track_a", help="base directory for sample outputs"
    )
    args = parser.parse_args(argv)

    if args.csv:
        df = load_real_csv(args.csv)
        src_msg = f"real CSV {args.csv}"
    else:
        df = synthesize(args.synthesize_n, args.seed)
        src_msg = f"synthesised n={args.synthesize_n} seed={args.seed}"

    out_dir = os.path.join(args.output_base, args.sample)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "loaded.csv")
    df.to_csv(out_path, index=False)
    print(f"[survey-loader] wrote {out_path} ({len(df)} rows, {df.shape[1]} cols, source: {src_msg})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
