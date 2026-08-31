#!/usr/bin/env python3
"""スイープの «1 行 1 セル × seed» の表．

run ディレクトリの読み方そのものは `runvault.read` にある．ここに残るのは Knoll 固有
の部分だけ — どの列を持つ表なのか (`beta_rho_ps` / `corr_ps_climate` …) である．
モデルの話であって run ディレクトリの読み方ではないので，共通部品には置かない．

runvault はこの表をディスクに持たない．sweep 親の子 run
(`lineage.parent_run_uid` が親の `run_uid`) を集め，各子の `config.json` の
`parameters`・`run.json` の `rng`・`metrics.csv` の最終ステップと run スコープ指標から
組み直す．列は移行前の `sweep_summary.csv` と同じにしてある．
"""

from __future__ import annotations

import os

import pandas as pd
from runvault.read import (
    config_parameters,
    load_run_meta,
    metrics_wide,
    run_scope_metrics,
    sweep_children,
)

__all__ = ["sweep_summary_table"]

#: 最終ステップの値から作る列 (列名 = metrics.csv の指標名)．
_FINAL_COLUMNS = [
    "silence_rate",
    "motive_mix_as",
    "motive_mix_qs",
    "motive_mix_ps",
    "motive_mix_os",
    "climate_of_silence",
    "kl_divergence_to_knoll",
]

#: run スコープ指標から作る相関の列 (列名 → 指標名)．
_CORR_COLUMNS = {
    f"corr_{motive}_climate": f"corr_{motive}_climate_of_silence"
    for motive in ("ps", "as", "qs", "os")
}


def sweep_summary_table(sweep_dir: str | os.PathLike) -> pd.DataFrame:
    """1 行 1 (セル × seed) のサマリ表を組み直す．

    どの行も `run_dir` を持つので，呼び出し側は条件からディレクトリ名を組み立てなくてよい．
    """
    children = sweep_children(sweep_dir)
    if not children:
        raise SystemExit(
            f"エラー: この sweep 親に紐づく子 run が見つかりません: {sweep_dir}\n"
            "  子 run は lineage.parent_run_uid で親を指します．"
            "親と子が同じ results ルートにあるか確認してください．"
        )

    rows: list[dict] = []
    for child in children:
        params = config_parameters(child) or {}
        beta = params.get("beta") or {}
        meta = load_run_meta(child)
        rng = meta.get("rng") or {}
        scoped = run_scope_metrics(child)
        last = metrics_wide(os.path.join(child, "metrics.csv")).iloc[-1]
        row = {
            "decision_mode": params.get("decision_mode"),
            "beta_psafety": beta.get("beta_psafety"),
            "beta_fear": beta.get("beta_fear"),
            "beta_rho_ps": beta.get("beta_rho_ps"),
            "prosocial_climate_decoupling": params.get("prosocial_climate_decoupling"),
            # 同一セルの何本目かは runvault の rng.replicate_index が持つ．
            "run": rng.get("replicate_index"),
            "seed": rng.get("master_seed"),
            "final_round": int(scoped["final_round"]),
        }
        row.update({column: float(last[column]) for column in _FINAL_COLUMNS})
        row.update({column: float(scoped[name]) for column, name in _CORR_COLUMNS.items()})
        row["run_dir"] = child
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["beta_psafety", "beta_fear", "beta_rho_ps", "prosocial_climate_decoupling", "run"]
        )
        .reset_index(drop=True)
    )
