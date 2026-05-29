#!/usr/bin/env python3
"""cfa_competing_models.py — Track A: 5 competing CFA models on the 12 items.

  M1: 1-factor (single global silence)
  M2: 4 uncorrelated factors (AS / QS / PS / OS)
  M3: 4 correlated factors (Knoll Study 1 winner)
  M3b: auxiliary 3-factor (AS + QS merged) — Knoll's closest-fitting competitor
  M4: hierarchical 2nd-order factor (our extension, not in Knoll Study 1)

Loads `results/track_a/<sample>/loaded.csv`; writes `cfa_fit_indices.csv` to the
same directory.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from knoll_tools._track_a_common import load_sample
from knoll_tools.survey_loader import ITEM_NAMES

# semopy model specifications.
MODELS: dict[str, str] = {
    "M1": (
        "Silence =~ "
        + " + ".join(ITEM_NAMES)
    ),
    "M2": (
        "AS =~ AS1 + AS2 + AS3\n"
        "QS =~ QS1 + QS2 + QS3\n"
        "PS =~ PS1 + PS2 + PS3\n"
        "OS =~ OS1 + OS2 + OS3\n"
        "# uncorrelated factors are the semopy default when no DEFINE block adds covariances"
    ),
    "M3": (
        "AS =~ AS1 + AS2 + AS3\n"
        "QS =~ QS1 + QS2 + QS3\n"
        "PS =~ PS1 + PS2 + PS3\n"
        "OS =~ OS1 + OS2 + OS3\n"
        "AS ~~ QS\nAS ~~ PS\nAS ~~ OS\nQS ~~ PS\nQS ~~ OS\nPS ~~ OS"
    ),
    "M3b": (
        "ASQS =~ AS1 + AS2 + AS3 + QS1 + QS2 + QS3\n"
        "PS =~ PS1 + PS2 + PS3\n"
        "OS =~ OS1 + OS2 + OS3\n"
        "ASQS ~~ PS\nASQS ~~ OS\nPS ~~ OS"
    ),
    "M4": (
        "AS =~ AS1 + AS2 + AS3\n"
        "QS =~ QS1 + QS2 + QS3\n"
        "PS =~ PS1 + PS2 + PS3\n"
        "OS =~ OS1 + OS2 + OS3\n"
        "Silence =~ AS + QS + PS + OS"
    ),
}


def fit_one(model_spec: str, data: pd.DataFrame) -> dict[str, float]:
    from semopy import Model, calc_stats

    m = Model(model_spec)
    m.fit(data)
    stats = calc_stats(m)
    # `stats` is a single-row DataFrame in recent semopy versions.
    if isinstance(stats, pd.DataFrame) and not stats.empty:
        row = stats.iloc[0].to_dict()
    elif isinstance(stats, dict):
        row = stats
    else:
        row = {}

    def _get(k: str) -> float:
        for key in (k, k.lower(), k.upper()):
            if key in row:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    pass
        return float("nan")

    return {
        "chi2": _get("chi2"),
        "df": _get("DoF"),
        "p": _get("chi2 p-value"),
        "cfi": _get("CFI"),
        "rmsea": _get("RMSEA"),
        "srmr": _get("SRMR"),
        "aic": _get("AIC"),
        "bic": _get("BIC"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knoll-tools cfa-competing-models")
    parser.add_argument("--sample", required=True)
    parser.add_argument(
        "--models",
        default="M1,M2,M3,M3b,M4",
        help="comma-separated subset of {M1, M2, M3, M3b, M4}",
    )
    parser.add_argument("--output-base", default="results/track_a")
    args = parser.parse_args(argv)
    df = load_sample(args.sample, args.output_base)
    data = df[ITEM_NAMES].apply(pd.to_numeric, errors="coerce").dropna()
    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for name in chosen:
        if name not in MODELS:
            print(f"warning: unknown model {name}; skipping", file=sys.stderr)
            continue
        try:
            stats = fit_one(MODELS[name], data)
        except ImportError as exc:
            print(
                f"error: semopy not installed ({exc}); run `uv sync` first",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:  # noqa: BLE001 — model may fail on tiny / synthetic data
            print(f"warning: {name} failed to fit ({exc}); recording NaN row", file=sys.stderr)
            stats = {k: float("nan") for k in ("chi2", "df", "p", "cfi", "rmsea", "srmr", "aic", "bic")}
        rows.append({"model": name, **stats})
    out = pd.DataFrame(rows)
    out_path = os.path.join(args.output_base, args.sample, "cfa_fit_indices.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print()
    print(f"[cfa-competing-models] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
