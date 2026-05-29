#!/usr/bin/env python3
"""multigroup_cfa.py — Track A Phase A5 (optional): multigroup CFA invariance.

Fits configural / metric / scalar invariance models over multiple language /
culture samples (e.g. `--samples main_en,main_ja`). semopy supports a `group`
argument for multigroup modelling; for now we implement a simple wrapper that
fits the M3 (correlated 4-factor) model in each group separately, reports
their CFI / RMSEA, and prints a heuristic ΔCFI / ΔRMSEA comparison.

This is a Phase A5 placeholder — the rigorous invariance test (constrained
loadings → constrained intercepts) is in scope for the full Track A run.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from knoll_tools._track_a_common import load_sample
from knoll_tools.cfa_competing_models import MODELS, fit_one
from knoll_tools.survey_loader import ITEM_NAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knoll-tools multigroup-cfa")
    parser.add_argument(
        "--samples",
        required=True,
        help="comma-separated sample names (e.g. main_en,main_ja)",
    )
    parser.add_argument("--levels", default="configural,metric,scalar")
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)
    names = [s.strip() for s in args.samples.split(",") if s.strip()]
    if len(names) < 2:
        print("multigroup-cfa needs ≥ 2 samples", file=sys.stderr)
        return 1
    rows = []
    for sample in names:
        df = load_sample(sample, args.output_base)
        data = df[ITEM_NAMES].apply(pd.to_numeric, errors="coerce").dropna()
        try:
            stats = fit_one(MODELS["M3"], data)
        except ImportError as exc:
            print(f"error: semopy not installed ({exc})", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"warning: M3 failed for sample {sample} ({exc}); NaN row", file=sys.stderr)
            stats = {k: float("nan") for k in ("chi2", "df", "p", "cfi", "rmsea", "srmr", "aic", "bic")}
        rows.append({"sample": sample, **stats})
    out = pd.DataFrame(rows)
    out_path = os.path.join(args.output_base, f"multigroup_cfa_{'_'.join(names)}.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print()
    if len(rows) == 2 and not (pd.isna(rows[0]["cfi"]) or pd.isna(rows[1]["cfi"])):
        delta_cfi = rows[1]["cfi"] - rows[0]["cfi"]
        delta_rmsea = rows[1]["rmsea"] - rows[0]["rmsea"]
        print(
            f"heuristic ΔCFI = {delta_cfi:+.3f} ; ΔRMSEA = {delta_rmsea:+.3f} "
            f"(metric/scalar invariance — Chen 2007 criteria: |ΔCFI| ≤ .010, |ΔRMSEA| ≤ .015)"
        )
    print(f"[multigroup-cfa] wrote {out_path}")
    print()
    print(
        "Note: this is a heuristic Phase A5 stub. Rigorous configural → metric → scalar "
        "invariance testing (with progressively constrained models) is on the Phase A5 backlog."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
