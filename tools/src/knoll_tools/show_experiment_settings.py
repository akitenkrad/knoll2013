#!/usr/bin/env python3
"""show_experiment_settings.py — print a results directory's settings.

Reads `config.json` (run) or `sweep_config.json` (sweep) plus `run_metadata.json`
and renders them as a readable table, or as JSON with `--json`. The shared
helpers from `socsim_tools.settings` handle the run-config rendering and the
LLM-metadata block; the sweep-config table (with its β-vector lines) is
repo-specific.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.io import load_run_metadata, resolve_results_dir
from socsim_tools.settings import render_run_config, render_run_metadata

# Config field → display label (left-padded so colons align).
FIELD_LABELS = {
    "decision_mode": "decision_mode    ",
    "n_teams": "n_teams          ",
    "team_size": "team_size        ",
    "n_levels": "n_levels         ",
    "n_employees": "n_employees      ",
    "network_kind": "network_kind     ",
    "network_k": "network_k        ",
    "network_beta": "network_beta     ",
    "prosocial_climate_decoupling": "ps_decoupling    ",
    "p_retaliate": "p_retaliate      ",
    "shock_t": "shock_t          ",
    "shock_magnitude": "shock_magnitude  ",
    "t_max": "t_max            ",
    "runs": "runs             ",
    "seed": "seed (core)      ",
    "llm_temperature": "LLM temperature  ",
    "llm_seed": "LLM seed         ",
    "llm_cache_path": "LLM cache_path   ",
    "output_dir": "output_dir       ",
}


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"no settings file in: {results_dir}\n"
        f"  expected: config.json (run) or sweep_config.json (sweep)"
    )


def render_sweep_config(cfg: dict, source: Path) -> str:
    """Render the sweep-config table (β vectors + decoupling sweep)."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("experiment settings (sweep)")
    lines.append("=" * 70)
    lines.append(f"settings file: {source}")
    lines.append("-" * 70)
    lines.append(f"decision_mode      : {cfg.get('decision_mode', '-')}")
    lines.append(f"n_teams            : {cfg.get('n_teams', '-')}")
    lines.append(f"team_size          : {cfg.get('team_size', '-')}")
    lines.append(f"β_ψ values         : {cfg.get('beta_psafety_values', '-')}")
    lines.append(f"β_f values         : {cfg.get('beta_fear_values', '-')}")
    lines.append(f"β_ρ^PS values      : {cfg.get('beta_rho_ps_values', '-')}")
    lines.append(f"sweep_decoupling   : {cfg.get('sweep_decoupling', '-')}")
    lines.append(f"runs/cell          : {cfg.get('runs', '-')}")
    lines.append(f"t_max              : {cfg.get('t_max', '-')}")
    lines.append(f"seed (base)        : {cfg.get('seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="knoll-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default="results/latest",
        help="results directory (default: results/latest)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of a table.",
    )
    args = parser.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"error: directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    meta = load_run_metadata(results_dir)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "run_metadata": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path, FIELD_LABELS))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_run_metadata(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
