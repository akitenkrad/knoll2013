<p align="center"><img src="docs/assets/hero.svg" width="100%"></p>

**English** | [日本語](README.ja.md)

# Knoll & van Dick (2013) — Four Forms of Employee Silence

A two-track replication of **Knoll & van Dick (2013), "Do I Hear the Whistle…? A First Attempt to Measure Four Forms of Employee Silence and Their Correlates"** (*Journal of Business Ethics*, 113(2), 349–362; DOI: 10.1007/s10551-012-1308-4).

- **Track A — psychometric replication** (Python `knoll-tools`): EFA / CFA / α / nomological-r-matrix analyses of an independent sample.
- **Track B — generative ABM** (Rust `knoll` on the [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) library): a 4-motive silence simulation on a Watts–Strogatz team network. A **rule** decision mode (multinomial-logistic ablation) and an **LLM** decision mode (`socsim-llm`, Ollama-first → OpenAI fallback) are mutually exclusive via `--decision-mode {rule|llm}`.

## Two-layer determinism

LLM output is **outside** socsim's bit-reproducibility, so the design splits into two layers:

- **Deterministic socsim core** — employee initialisation, Watts–Strogatz network generation, scheduling, the 8 non-decision mechanisms, the rule-mode `voice_decision_rule`. Given a seed this reproduces bit-for-bit. The `--decision-mode rule` path lives entirely here and makes **zero LLM calls**.
- **Non-deterministic LLM layer** — `voice_decision` only. Pseudo-determinised by `socsim-llm`'s `CachingClient` (a `hash(prompt+model)` → response cache), `temperature=0` and a fixed `(agent_id, t)`-derived seed. Provider order is **Ollama first → OpenAI fallback** via `socsim-llm`'s `FallbackClient`.

The cache — not the model — is the reproducibility mechanism: a warm cache replays identical responses. Each run records the model, endpoint and temperature in `run.json`'s `llm` block, and the call / cache-hit counts as run-scope metrics.

## Install & Quick start

```bash
# Build the Rust simulation (fetches socsim incl. socsim-llm with Ollama+OpenAI backends).
cargo build --release

# === Rule mode (no LLM) — ablation baseline ===
cargo run --release -- run --decision-mode rule \
    --n-teams 8 --team-size 12 \
    --motive-prior-as 0.22 --motive-prior-qs 0.27 \
    --motive-prior-ps 0.40 --motive-prior-os 0.18 \
    --prosocial-climate-decoupling \
    --t-max 36 --runs 30 --seed 42

# === LLM mode (Ollama first) ===
#   ollama pull llama3.1
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.1
cargo run --release -- run --decision-mode llm \
    --llm-cache-path runs/knoll_cache.json \
    --t-max 36 --runs 10 --seed 42

# === Sensitivity sweep (β group × prosocial_decoupling × seeds) ===
cargo run --release -- sweep \
    --beta-psafety-values "0.6,1.2,2.0" \
    --beta-fear-values    "0.5,1.5,2.5" \
    --beta-rho-ps-values  "0.0,0.1,0.3" \
    --runs 20 --seed 42

# Python visualization & analysis tools (workspace root)
uv sync
uv run knoll-tools visualize                          # motive_mix + KL + motive×climate bar
uv run knoll-tools visualize-sweep                    # β heatmap + PS-decoupling response curve
uv run knoll-tools show-experiment-settings           # a run directory's config + LLM provenance

# === Track A synthetic-data smoke (no real data required) ===
uv run knoll-tools survey-loader --synthesize-n 200 --sample synth
uv run knoll-tools descriptive-stats     --sample synth
uv run knoll-tools efa-4factor           --sample synth --rotation varimax
uv run knoll-tools reliability-analysis  --sample synth
uv run knoll-tools nomological-network   --sample synth --bootstrap 500
uv run knoll-tools cfa-competing-models  --sample synth --models M1,M2,M3,M3b,M4
```

## Repository layout

```
knoll2013/
├── simulation/                       # Track B (Rust socsim ABM)
│   ├── Cargo.toml                    # socsim-{core,engine,net,mechanisms,metrics,llm} + runvault git deps
│   ├── src/
│   │   ├── lib.rs / main.rs          # CLI: run / sweep / reproduce
│   │   ├── config.rs                 # Config / DecisionMode / BetaGroup / MotivePrior / NetworkKind
│   │   ├── world.rs                  # SilenceWorld + Employee + Team + Motive + Expression
│   │   ├── mechanisms.rs             # 9 mechanisms × 6 phases; rule vs LLM decision (mutually exclusive)
│   │   ├── prompts.rs                # LLM persona templates + decision JSON parser
│   │   ├── llm.rs                    # socsim-llm shared-harness re-export shim
│   │   ├── simulation.rs             # init_world + run_with_client (no file writing)
│   │   ├── record.rs                 # runvault: paper metadata, metrics, agent events
│   │   └── metrics.rs                # silence_rate / motive_mix / climate / KL / Pearson r
│   └── tests/integration_test.rs     # rule + scripted-LLM smoke tests
├── tools/                            # Python knoll-tools (Track A + Track B)
│   └── src/knoll_tools/{cli,visualize,visualize_sweep,show_experiment_settings,sweep_summary,
│                        survey_loader,descriptive_stats,efa_4factor,cfa_competing_models,
│                        reliability_analysis,nomological_network,discriminant_validity,
│                        robustness_checks,multigroup_cfa,cfa_analysis,reproduce_paper}.py
├── survey/                           # Track A instrument (12-item EN/JA + translation log + IRB protocol)
│   ├── knoll_12item_en.yaml / knoll_12item_ja.yaml
│   ├── translation_log.md            # Brislin 1970 translation process
│   └── irb_protocol.md               # IRB submission protocol
├── docs/                             # bilingual: architecture, cli, usecases, visualization, reproduction
├── data_external/                    # raw survey CSVs (gitignored — never commit)
└── results/                          # runvault run directories (gitignored)
    └── knoll/
        ├── run_{stamp}_{config}_{exec}/    # one subcommand invocation = one run
        │   ├── run.json              # identity, seeds, LLM block, paper + targets
        │   ├── config.json           # envelope; the conditions live under `parameters`
        │   ├── metrics.csv           # long: (name, step, step_unit, scope, value)
        │   ├── events.jsonl          # x.knoll2013.agent — final per-agent state
        │   └── status.json           # state + duration_sec
        ├── sweep_{stamp}_…/          # sweep parent; children point at it via lineage
        └── figures/{run_slug}/       # drawn after the run, so outside it
```

## Documentation

- [Architecture](docs/architecture.md) — world state, 9-mechanism × 6-phase table, two-track diagram
- [CLI reference](docs/cli.md) — `run` / `sweep` / `reproduce` flags
- [Usecases](docs/usecases.md) — Track A vs Track B use cases
- [Visualization](docs/visualization.md) — what the Python tools produce
- [Reproduction](docs/reproduction.md) — how the model maps to the Knoll 2013 Study 1 / 2 numbers

## References

- Knoll, M., & van Dick, R. (2013). Do I Hear the Whistle…? A First Attempt to Measure Four Forms of Employee Silence and Their Correlates. *Journal of Business Ethics*, 113(2), 349–362.
- Simulation engine: [socsim (rs-social-simulation-tools)](https://github.com/akitenkrad/rs-social-simulation-tools).

## License

MIT — see [LICENSE](LICENSE).

---
*This file was generated by Claude Code.*
