#!/usr/bin/env python3
"""show_experiment_settings.py — print a run directory's settings.

runvault の run ディレクトリの `config.json` (封筒．条件は `parameters` の下) を読み，
実行時に使われた全パラメータを整形表示する．`run` か `sweep` かは `run.json` の
`subcommand` で判別する (`sweep_config.json` はもう書かれない)．LLM 情報 (モデル・
provider・温度) は `run.json` の `llm` ブロック，呼び出し数と cache-hit 率は
`metrics.csv` の run スコープ指標から採る．

run ディレクトリのパスは次で取れる:
    runvault path --experiment knoll --latest --subcommand run --standalone
    runvault path --experiment knoll --latest --subcommand sweep

Usage:
    uv run knoll-tools show-experiment-settings
    uv run knoll-tools show-experiment-settings --results-dir "$(runvault path --experiment knoll --latest --subcommand sweep)"
    uv run knoll-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runvault.read import (
    config_parameters,
    load_run_meta,
    run_scope_metrics,
    runvault_path,
)

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "knoll"

# 動機の内部名 → 論文の略号 (AS / QS / PS / OS)．
MOTIVE_CODES = {
    "acquiescent": "AS",
    "quiescent": "QS",
    "prosocial": "PS",
    "opportunistic": "OS",
}

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
}


def render_run_config(cfg: dict, source: Path, kind: str) -> str:
    """Render the run-config table (Knoll-specific field order)."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"experiment settings ({kind})")
    lines.append("=" * 70)
    lines.append(f"settings file: {source}")
    lines.append("-" * 70)
    for field, label in FIELD_LABELS.items():
        if field in cfg:
            lines.append(f"{label}: {cfg[field]}")
    prior = cfg.get("motive_prior") or {}
    if prior:
        lines.append(
            "motive_prior     : "
            + " ".join(f"{code}={prior[name]}" for name, code in MOTIVE_CODES.items() if name in prior)
        )
    for name, value in (cfg.get("beta") or {}).items():
        lines.append(f"{name:<17}: {value}")
    lines.append("=" * 70)
    return "\n".join(lines)


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


def render_llm(meta: dict, scoped: dict[str, float]) -> str | None:
    """LLM 由来情報．rule モードは LLM 層に触れないので何も出さない．

    移行前は `run_metadata.json` が持っていた．モデル・provider・温度は `run.json` の
    `llm` ブロック，呼び出し数と cache-hit 率は run スコープの指標が正本になった．
    """
    llm = meta.get("llm")
    if llm is None:
        return None
    lines: list[str] = []
    lines.append("LLM provenance")
    lines.append("-" * 70)
    lines.append(f"provider         : {llm.get('provider', '-')}")
    lines.append(f"model            : {llm.get('model_snapshot', '-')}")
    lines.append(f"temperature      : {llm.get('temperature', '-')}")
    calls = scoped.get("llm_calls")
    if calls is not None:
        hits = scoped.get("llm_cache_hits", 0.0)
        rate = scoped.get("llm_cache_hit_rate")
        rate_text = "-" if rate is None else f"{rate * 100:.1f}%"
        lines.append(f"calls / cache-hit: {int(calls)} / {int(hits)} ({rate_text})")
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
        default=None,
        help="run ディレクトリ (省略時は runvault path が返す直近の run)",
    )
    parser.add_argument("--results-root", "--results_root", default="results")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of a table.",
    )
    args = parser.parse_args(argv)

    results_dir = Path(
        args.results_dir
        or runvault_path(
            EXPERIMENT,
            results_root=args.results_root,
            subcommand="run",
            standalone=True,
        )
    )
    if not results_dir.exists():
        print(f"error: directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg = config_parameters(results_dir)
        meta = load_run_meta(results_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    assert cfg is not None and meta is not None  # required=True raises instead
    kind = str(meta["subcommand"])
    scoped = run_scope_metrics(results_dir)
    source = results_dir / "config.json"

    if args.json:
        payload = {
            "source": str(source),
            "kind": kind,
            "config": cfg,
            "llm": meta.get("llm"),
            "run_scope_metrics": scoped,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if kind == "sweep":
        print(render_sweep_config(cfg, source))
    else:
        print(render_run_config(cfg, source, kind))
    llm = render_llm(meta, scoped)
    if llm is not None:
        print(llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
