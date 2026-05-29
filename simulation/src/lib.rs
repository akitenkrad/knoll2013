//! Knoll & van Dick (2013) — Four-form employee silence simulation.
//!
//! This crate is **Track B** of the knoll2013 replication: a socsim-based ABM
//! that generates motive-stratified silence patterns (`AS`/`QS`/`PS`/`OS`)
//! on a Watts–Strogatz small-world team network.
//!
//! Two **mutually exclusive** decision modes are wired by `config.decision_mode`:
//!
//! - `--decision-mode rule` — `VoiceDecisionRule` mechanism: a multinomial
//!   logistic regression with sign-constrained `β` over the 8 contextual
//!   features (`f, ψ, ι, ρ, u, σ, h, g`). The `prosocial_climate_decoupling`
//!   flag forces `β_ρ^{PS} = 0` to mechanise the paper's central PS-climate
//!   independence finding.
//! - `--decision-mode llm` — `VoiceDecisionLlm` mechanism: an LLM (Ollama-first,
//!   OpenAI fallback) decides VOICE/SILENCE and the silence motive, given an
//!   employee persona and the same 8-feature local context. All LLM access is
//!   funnelled through `socsim-llm`'s shared harness (`build_live_client_from_settings`,
//!   `LiveClient`, `llm_config`, `wrap_client`).
//!
//! Eight further (non-decision) mechanisms run unconditionally each step:
//! `IssueSalience`, `RetaliationEvent`, `FearAppraisal`, `SilenceSpiral`,
//! `PrefalseCascade`, `OrgPerformance`, `PsafetyUpdate`, `ClimateSilence` —
//! laid out across socsim's 6-phase fixed loop (see `mechanisms.rs`).
//!
//! See `simulation/src/main.rs` for the `run` / `sweep` / `reproduce` CLI.
//! Scope is documented in the project README under "Phase Status".

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod prompts;
pub mod simulation;
pub mod world;
