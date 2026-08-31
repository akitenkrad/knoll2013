[English](cli.md) | [日本語](cli.ja.md)

# CLI reference

`knoll <subcommand> [...flags]`.

## `knoll run`

Run one configuration (rule or LLM mode), one or more times (`--runs`).

| Flag | Default | Description |
|------|---------|-------------|
| `--decision-mode <rule\|llm>` | `rule` | Mutually exclusive: `rule` adds `VoiceDecisionRule`; `llm` adds `VoiceDecisionLlm` (socsim-llm driven). |
| `--n-teams <usize>` | `8` | Number of teams. |
| `--team-size <usize>` | `12` | Employees per team (total = teams × size). |
| `--n-levels <u8>` | `3` | Hierarchical levels (descriptive only). |
| `--network <ws\|er\|ba>` | `watts-strogatz` | Network family. |
| `--network-k <usize>` | `6` | Watts–Strogatz `k` / Barabási–Albert `m`. |
| `--network-beta <f64>` | `0.1` | Watts–Strogatz β / Erdős–Rényi `p`. |
| `--motive-prior-as / -qs / -ps / -os <f64>` | `0.22 / 0.27 / 0.40 / 0.18` | Initial 4-motive marginal (Knoll Study 2 means). |
| `--beta-psafety <f64>` | `1.2` | β_ψ — VOICE coefficient on psychological safety. |
| `--beta-fear <f64>` | `1.5` | β_f — VOICE coefficient on fear (used with negative sign). |
| `--beta-rho <f64>` | `1.0` | β_ρ — VOICE coefficient on perceived peer silence. |
| `--beta-rho-ps <f64>` | `0.1` | **β_ρ^{PS}** — PS-row coefficient on ρ inside the motive softmax. The critical knob. |
| `--prosocial-climate-decoupling` | `false` | Force β_ρ^{PS} = 0 and omit climate cue in PS persona prompt — mechanises Knoll's PS-climate independence finding. |
| `--p-retaliate <f64>` | `0.05` | Per-agent per-step retaliation probability. |
| `--shock-t <u64>` | (none) | Optional exogenous σ shock step. |
| `--shock-magnitude <f64>` | `0.3` | σ-shock magnitude. |
| `--t-max <u64>` | `36` | Maximum simulation step. |
| `--runs <usize>` | `1` | Independent runs (different seeds; output = the *last* run). |
| `--seed <u64>` | `42` | Root seed (governs the deterministic socsim core). |
| `--llm-temperature <f32>` | `0.0` | LLM generation temperature. |
| `--llm-seed <u64>` | `0` | LLM seed offset; per-(agent, t) seed is derived. |
| `--llm-cache-path <path>` | `.llm_cache/cache.json` | Prompt→response cache (LLM mode only). |
| `--output-dir <path>` | `results` | runvault results root; the run directory is named under `<root>/knoll/`. |

One invocation is one [runvault](https://github.com/akitenkrad/rs-runvault) run. runvault creates and names the directory, so there is no timestamped folder and no `latest` symlink to maintain.

Outputs (under `results/knoll/run_{stamp}_{config}_{exec}/`):

- `run.json` — identity, `rng` (the `master_seed` that actually governed the recorded execution, and its `replicate_index`), the `llm` block in LLM mode, and the paper + targets.
- `config.json` — the envelope; the conditions are under `parameters`.
- `metrics.csv` — long form, `(name, step, step_unit, scope, value)`:
  - per step (`step_unit=step`, `scope=run`): `silence_rate`, `motive_mix_{as,qs,ps,os}`, `subscale_proxy_{as,qs,ps,os}`, `climate_of_silence`, `issue_salience`, `kl_divergence_to_knoll`.
  - one value for the whole run (no step): `n_units`, `final_round`, `llm_calls`, `llm_cache_hits`, `llm_cache_hit_rate` (LLM mode only — a rate over zero calls is undefined, so the row is absent in rule mode), and the 24 `corr_{as,qs,ps,os}_{climate_of_silence,fear,psafety,ivt,harm,self_gain}` values.
- `events.jsonl` — one `x.knoll2013.agent` line per employee: the final-step state. Per-employee values cannot live in `metrics.csv`, whose primary key is `(name, step, step_unit, scope)`; and `expression` / `motive` are labels, not numbers.
- `status.json` — the run's state and `duration_sec` (wall-clock time is not a metric).

## `knoll sweep`

Cartesian product over `β_ψ × β_f × β_ρ^{PS} × prosocial_decoupling × seeds`. Each cell × seed is a genuinely separate execution, so it becomes a **child run** under one sweep **parent**.

| Flag | Default | Description |
|------|---------|-------------|
| `--decision-mode <rule\|llm>` | `rule` | Sweep over β is most meaningful for `rule`. |
| `--beta-psafety-values <csv>` | `0.6,1.2,2.0` | β_ψ sweep values. |
| `--beta-fear-values <csv>` | `0.5,1.5,2.5` | β_f sweep values. |
| `--beta-rho-ps-values <csv>` | `0.0,0.1,0.3` | β_ρ^{PS} sweep values. |
| `--sweep-decoupling <bool>` | `true` | Whether to also sweep `prosocial_climate_decoupling ∈ {false, true}`. |
| `--runs <usize>` | `5` | Runs (independent seeds) per cell. |
| `--t-max <u64>` | `36` | Maximum simulation step per run. |
| `--seed <u64>` | `42` | Base seed; per-cell seed derived. |
| `--output-dir <path>` | `results` | runvault results root. |

Outputs:

- `results/knoll/sweep_{stamp}_…/` — the parent. Its `parameters` hold the grid itself. It has **no** `master_seed`: a sweep is driven by a list of seeds, not one, and the base seed reaches the execution hash through `/parameters.seed`.
- `results/knoll/run_{stamp}_…/` — one child per (cell × seed), pointing at the parent through `lineage.parent_run_uid`. A child writes the same files as a hand-run `run`, with `parameters` of the same shape (so the same conditions give the same `config_hash`), `master_seed` = the derived cell seed and `rng.replicate_index` = which repeat of the cell it is.

There is no `sweep_summary.csv`: every column it had is in the children (conditions in `parameters`, seeds in `run.json`, the final-step values and the correlations in `metrics.csv`). `knoll-tools visualize-sweep` rebuilds the table from them.

## `knoll reproduce`

Emits reflexive 12-item self-ratings for population-CFA reproduction of the Knoll Study 1 measurement model; pairs with `cfa-analysis` and `reproduce-paper` in `knoll-tools` on the Python side.

## `knoll-tools` (Python)

`knoll-tools <subcommand> [...flags]`. Subcommands:

- **Track B (ABM)**: `visualize`, `visualize-sweep`, `show-experiment-settings`, `cfa-analysis` (population CFA over emitted self-ratings), `reproduce-paper` (3-way paper / Track A / Track B comparison)
- **Track A (psychometrics)**: `survey-loader`, `descriptive-stats`, `efa-4factor`, `cfa-competing-models`, `reliability-analysis`, `nomological-network`, `discriminant-validity`, `robustness-checks`, `multigroup-cfa`

Run `knoll-tools <subcommand> --help` for per-subcommand flags.

---
*This file was generated by Claude Code.*
