#!/usr/bin/env python3
"""discriminant_validity.py — Track A: AVE / CR + AVE > squared-r matrix.

Computes Average Variance Extracted (AVE) and Composite Reliability (CR) per
subscale via standardised factor loadings, then a 4×4 matrix showing AVE vs
inter-subscale squared correlations. The "Fornell-Larcker" criterion holds
when each AVE exceeds every off-diagonal squared correlation in its row.

Loads `results/track_a/<sample>/loaded.csv`; writes
`discriminant_validity.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from knoll_tools._track_a_common import load_sample
from knoll_tools.survey_loader import ITEM_NAMES, SUBSCALE_OF

SUBSCALES = ["AS", "QS", "PS", "OS"]


def standardised_loadings(df: pd.DataFrame, sub: str) -> np.ndarray:
    """Stand-in loadings: per-item correlation with the subscale mean score
    (a quick FA-free proxy; replaced by semopy estimates in Phase A4).
    """
    items = [it for it in ITEM_NAMES if SUBSCALE_OF[it] == sub]
    total = df[items].sum(axis=1)
    loadings = []
    for it in items:
        v = pd.to_numeric(df[it], errors="coerce")
        t = pd.to_numeric(total - v, errors="coerce")  # rest-of-subscale total
        mask = v.notna() & t.notna()
        if mask.sum() < 2 or v[mask].std() == 0 or t[mask].std() == 0:
            loadings.append(0.0)
        else:
            loadings.append(float(np.corrcoef(v[mask], t[mask])[0, 1]))
    return np.array(loadings)


def ave_cr(loadings: np.ndarray) -> tuple[float, float]:
    if loadings.size == 0:
        return float("nan"), float("nan")
    sq = loadings**2
    err = 1.0 - sq
    sum_l = float(loadings.sum())
    sum_sq = float(sq.sum())
    sum_err = float(err.sum())
    ave = sum_sq / loadings.size
    cr_denom = sum_l**2 + sum_err
    cr = (sum_l**2) / cr_denom if cr_denom > 0 else float("nan")
    return ave, cr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knoll-tools discriminant-validity")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)
    df = load_sample(args.sample, args.output_base)
    means = {sub: df[[it for it in ITEM_NAMES if SUBSCALE_OF[it] == sub]].mean(axis=1) for sub in SUBSCALES}
    aves: dict[str, float] = {}
    crs: dict[str, float] = {}
    for sub in SUBSCALES:
        loadings = standardised_loadings(df, sub)
        a, c = ave_cr(loadings)
        aves[sub] = a
        crs[sub] = c
    # Squared correlation matrix
    rows = []
    for sub_a in SUBSCALES:
        row = {"subscale": sub_a, "ave": aves[sub_a], "cr": crs[sub_a]}
        for sub_b in SUBSCALES:
            if sub_a == sub_b:
                row[f"r2_{sub_b}"] = aves[sub_a]
            else:
                xa = means[sub_a].to_numpy()
                xb = means[sub_b].to_numpy()
                mask = ~(np.isnan(xa) | np.isnan(xb))
                if mask.sum() < 2 or xa[mask].std() == 0 or xb[mask].std() == 0:
                    r2 = 0.0
                else:
                    r2 = float(np.corrcoef(xa[mask], xb[mask])[0, 1] ** 2)
                row[f"r2_{sub_b}"] = r2
        rows.append(row)
    out = pd.DataFrame(rows)
    out_path = os.path.join(args.output_base, args.sample, "discriminant_validity.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(
        "\nFornell-Larcker criterion holds iff each AVE on the diagonal exceeds every off-diagonal r² in its row."
    )
    print(f"[discriminant-validity] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
