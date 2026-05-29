[English](architecture.md) | [日本語](architecture.ja.md)

# Architecture

## World state — `SilenceWorld`

A team-organised, fully-connected (within team) + Watts–Strogatz (across team) social network of employees. Each employee carries an 8-dimensional context vector and a 4-way silence motive.

- `employees: BTreeMap<AgentId, Employee>` — sorted-key activation order = determinism.
- `teams: Vec<Team>` — per-team `supervisor_openness`, `knowledge_stock`, `climate`.
- `network: SocialNetwork` — Watts–Strogatz / Erdős–Rényi / Barabási–Albert (CLI-selected via `--network`).
- `issue_salience: f64` — σ(t), updated by `IssueSalience`.
- `climate_of_silence: f64` — C(t), updated by `ClimateSilence`.
- `retaliation_this_step: Vec<AgentId>` — set by `RetaliationEvent`, consumed by `FearAppraisal` + `PsafetyUpdate`.

Per-`Employee`: `level, tenure, team, private_concern, expression, silence_motive, fear, psych_safety, ivt_strength, perceived_silence, harm_perception, self_gain, voice_threshold`.

`WorldState::agent_ids()` returns the BTreeMap keys (sorted `AgentId`s) as the canonical activation order.

## 9 mechanisms × 6 phases

| #  | Mechanism                | Phase        | Role |
|----|--------------------------|--------------|------|
| 1  | `IssueSalience`          | Environment  | Update σ(t) with mean-reversion + optional shock at `shock_t`. |
| 2  | `RetaliationEvent`       | Environment  | With probability `p_retaliate` per agent, mark them as retaliated this step. |
| 3  | `FearAppraisal`          | Decision     | `f_i ← clamp(f_i + α·retaliated − γ·max(u_team, 0), 0, 1)`. |
| 4a | `VoiceDecisionRule` ★   | Decision     | **Phase B1 ablation.** Multinomial-logistic over 8 features → Bernoulli VOICE + softmax motive. |
| 4b | `VoiceDecisionLlm` ★    | Decision     | **Phase B2.** `socsim-llm` driven; persona + context → JSON `{decision, motive, rationale}`. |
| 5  | `SilenceSpiral`          | Interaction  | `ρ_i ← neighbour silence ratio` (Noelle-Neumann 1974). |
| 6  | `PrefalseCascade`        | Interaction  | Silent agent flips to VOICE if neighbour-VOICE ratio > θ_i (Kuran 1995). |
| 7  | `OrgPerformance`         | Reward       | Per-team `K_k(t)` ← decay + VOICE_share_k − OS_share_k. |
| 8  | `PsafetyUpdate`          | PostStep     | `ψ_i ← clamp(ψ_i + η·voiced − ν·retaliated, 0, 1)`. |
| 9  | `ClimateSilence`         | PostStep     | Update `world.climate_of_silence` + per-team `team.climate`. |

★ = mutually exclusive. The driver wires *exactly one* of `voice_decision_rule` and `voice_decision` based on `cfg.decision_mode`.

## Two-track architecture diagram

```
┌──────────────────────────── Track A (psychometric) ───────────────────────┐
│                                                                            │
│   data_external/*.csv     ┌─→ survey_loader.py ─→ loaded.csv             │
│   (real survey CSV)       │                          │                    │
│                           │                          ├─→ descriptive_stats │
│   --synthesize-n 200 ─────┘                          ├─→ efa_4factor      │
│   (smoke fallback)                                   ├─→ cfa_competing_… │
│                                                      ├─→ reliability     │
│                                                      ├─→ nomological_net │
│                                                      ├─→ discriminant   │
│                                                      ├─→ robustness     │
│                                                      └─→ multigroup_cfa  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── Track B (socsim ABM) ─────────────────────────┐
│                                                                            │
│   Config + seed                                                            │
│         │                                                                  │
│         v                                                                  │
│   init_world ─→ SilenceWorld (BTreeMap<AgentId, Employee>, SocialNetwork) │
│         │                                                                  │
│         v                                                                  │
│   SimulationBuilder ── 9 mechanisms × 6 phases × t_max steps              │
│         │                                                                  │
│         │   ┌─ rule:  VoiceDecisionRule  (deterministic)                  │
│         ├──→│                                                              │
│         │   └─ llm:   VoiceDecisionLlm   (socsim-llm: Ollama→OpenAI)      │
│         v                                                                  │
│   metrics.csv | agents.csv | correlations.csv | run_metadata.json         │
│         │                                                                  │
│         v                                                                  │
│   knoll-tools visualize / visualize-sweep / show-experiment-settings      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## RNG streams (determinism)

```text
RNG_WORLD_INIT = 0   // employee attributes + Watts–Strogatz network generation
RNG_ENGINE     = 1   // scheduler + non-LLM mechanism RNG
RNG_LLM_ROOT   = 2   // per-(agent_id, t) LLM seed root for VoiceDecisionLlm
```

- `init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]))`
- Engine seed: `derive_seed(root, &[RNG_ENGINE])`
- LLM per-call seed: `derive_seed(derive_seed(root, &[RNG_LLM_ROOT]), &[2, agent_id, t])`

## Update semantics

Synchronous within each phase: the decision mechanism snapshots all employees at step start, computes new expressions/motives from the snapshot, then writes them in one batch. Network-interaction mechanisms (`SilenceSpiral`, `PrefalseCascade`) read the just-finalised expressions and write `perceived_silence` / promote silent → VOICE for next step's `ρ_i`.

`Scheduler = RandomActivationScheduler` (shuffles the agent activation order each step using `RNG_ENGINE`). Because the decision mechanism processes the snapshot rather than the per-agent order, scheduling shuffling does not affect correctness — it only determines the LLM call order in `VoiceDecisionLlm` (still deterministic because `temperature=0` + per-(agent, t) seed).

## Equations

VOICE Bernoulli (rule mode):

```text
P(VOICE) = σ( β0 + β_ψ ψ + β_u u + β_σ σ − β_f f − β_ι ι − β_ρ ρ )
```

SILENCE motive softmax (rule mode), per row k ∈ {AS, QS, PS, OS}:

```text
P(m = k | SILENCE) ∝ exp( β_k · x )
x = (f, ψ, ι, ρ, u, σ, h, g)
```

Sign constraints (design §4.4 Table):

| motive | β_f | β_ψ | β_ι | β_ρ | β_u | β_h | β_g |
|--------|-----|-----|-----|-----|-----|-----|-----|
| AS     |  0  |  −  |  +  |  +  |  −  |  0  |  0  |
| QS     |  +  |  −  |  0  |  +  |  −  |  0  |  0  |
| PS     |  0  |  0  |  0  | ≈0  |  +  |  +  |  0  |
| OS     |  0  |  −  |  0  |  0  |  0  |  −  |  +  |

The PS row's `β_ρ` is the *critical knob*: `--prosocial-climate-decoupling` forces it to zero, mechanising Knoll's central PS-climate independence finding.

KL divergence to Knoll Study 2 marginals (the headline match indicator):

```text
KL(π_emp || π_abm) = Σ_m π_emp_m · log(π_emp_m / π_abm_m)
```

with π_emp = (.22, .27, .40, .18) for (AS, QS, PS, OS).

---
*This file was generated by Claude Code.*
