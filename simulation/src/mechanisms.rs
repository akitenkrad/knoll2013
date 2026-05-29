//! 9 mechanisms across the socsim 6-phase loop, plus the seeded-motive
//! BTreeMap used by the LLM path's persona selection.
//!
//! | # | Mechanism             | Phase        | Role |
//! |---|-----------------------|--------------|------|
//! | 1 | `IssueSalience`       | Environment  | Update σ(t); fire optional shock at `shock_t` |
//! | 2 | `RetaliationEvent`    | Environment  | With probability `p_retaliate` mark agents touched by a retaliation event |
//! | 3 | `FearAppraisal`       | Decision     | Threat appraisal: `f_i ← (1-α) f_i + α (retaliated?1:0) − γ u_team` |
//! | 4 | `VoiceDecisionRule` / `VoiceDecisionLlm` | Decision | **★ mutually exclusive**: rule = multinomial-logistic ablation; llm = `socsim-llm` driven |
//! | 5 | `SilenceSpiral`       | Interaction  | ρ_i ← neighbour silence ratio (Noelle-Neumann 1974) |
//! | 6 | `PrefalseCascade`     | Interaction  | Silent agent flips to VOICE if neighbour-VOICE ratio > θ_i (Kuran 1995) |
//! | 7 | `OrgPerformance`      | Reward       | Team `K_k(t)` ← decay + PS/OS contribution |
//! | 8 | `PsafetyUpdate`       | PostStep     | ψ_i ← ψ_i + η (voiced?+1:0) − ν (retaliated?+1:0) |
//! | 9 | `ClimateSilence`      | PostStep     | Aggregate C(t) and per-team climate |
//!
//! The decision mechanisms **snapshot all employees at step start** and apply
//! the new expressions/motives from the snapshot (synchronous update,
//! design §4.4 "Update semantics").

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;
use socsim_core::{
    derive_seed, AgentId, Mechanism, Phase, Result, SocsimError, StepContext, WorldState,
};
use socsim_llm::MetadataCollector;

use crate::config::{BetaGroup, LlmSettings, MotivePrior};
use crate::llm::{llm_config, SilenceClient};
use crate::prompts::{build_silence_prompt, parse_voice_decision, persona_for};
use crate::world::{Expression, Motive, SilenceWorld};

// --------------------------------------------------------------------------- //
// Shared LLM client / metadata wrappers (mirrors mou2024 / wang2025)
// --------------------------------------------------------------------------- //

/// Shared LLM client between driver + mechanism (`Rc<RefCell>` pattern).
pub type SharedClient = Rc<RefCell<SilenceClient>>;
/// Shared metadata collector for cache-hit rate / call count.
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;

/// Per-agent seeded persona motive, populated at world init and consumed by
/// the LLM-decision mechanism for prompt construction. Carried alongside the
/// world (not on it) because it is LLM-mode-only persona data.
pub type SeededMotives = Rc<BTreeMap<AgentId, Motive>>;

// --------------------------------------------------------------------------- //
// 1. IssueSalience  (Environment)
// --------------------------------------------------------------------------- //

/// Updates `world.issue_salience` with mild mean-reversion plus an optional
/// step-`shock_t` exogenous bump. Wholly deterministic given the run seed.
pub struct IssueSalience {
    decay: f64,
    target: f64,
    shock_t: Option<u64>,
    shock_magnitude: f64,
}

impl IssueSalience {
    pub fn new(shock_t: Option<u64>, shock_magnitude: f64) -> Self {
        IssueSalience {
            decay: 0.10,
            target: 0.5,
            shock_t,
            shock_magnitude,
        }
    }
}

impl Mechanism<SilenceWorld> for IssueSalience {
    fn name(&self) -> &str {
        "issue_salience"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        // mean-reverting toward `target`
        let sigma = ctx.world.issue_salience;
        let mut new_sigma = sigma + self.decay * (self.target - sigma);
        // optional shock
        if let Some(t_shock) = self.shock_t {
            if ctx.clock.t() == t_shock {
                new_sigma = (new_sigma + self.shock_magnitude).clamp(0.0, 1.0);
            }
        }
        ctx.world.issue_salience = new_sigma.clamp(0.0, 1.0);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 2. RetaliationEvent  (Environment)
// --------------------------------------------------------------------------- //

/// With probability `p_retaliate` per agent, mark them as retaliated this step.
/// Drives downstream fear / psafety updates.
pub struct RetaliationEvent {
    p_retaliate: f64,
}

impl RetaliationEvent {
    pub fn new(p_retaliate: f64) -> Self {
        RetaliationEvent {
            p_retaliate: p_retaliate.clamp(0.0, 1.0),
        }
    }
}

impl Mechanism<SilenceWorld> for RetaliationEvent {
    fn name(&self) -> &str {
        "retaliation_event"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        ctx.world.retaliation_this_step.clear();
        if self.p_retaliate <= 0.0 {
            return Ok(());
        }
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            if ctx.rng.gen::<f64>() < self.p_retaliate {
                ctx.world.retaliation_this_step.push(id);
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 3. FearAppraisal  (Decision)
// --------------------------------------------------------------------------- //

/// Threat appraisal: `f_i ← clamp(f_i + α·retaliated − γ·max(u_team,0), 0, 1)`.
pub struct FearAppraisal {
    alpha: f64,
    gamma: f64,
}

impl FearAppraisal {
    pub fn new() -> Self {
        FearAppraisal {
            alpha: 0.30,
            gamma: 0.10,
        }
    }
}

impl Default for FearAppraisal {
    fn default() -> Self {
        Self::new()
    }
}

impl Mechanism<SilenceWorld> for FearAppraisal {
    fn name(&self) -> &str {
        "fear_appraisal"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let retaliated: std::collections::HashSet<AgentId> =
            ctx.world.retaliation_this_step.iter().copied().collect();
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            let team_idx = ctx.world.employees[&id].team;
            let u = ctx.world.teams[team_idx].supervisor_openness;
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            let r = if retaliated.contains(&id) { 1.0 } else { 0.0 };
            let new_f = (emp.fear + self.alpha * r - self.gamma * u.max(0.0)).clamp(0.0, 1.0);
            emp.fear = new_f;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 4a. VoiceDecisionRule  (Decision) — multinomial-logistic ablation
// --------------------------------------------------------------------------- //

/// Sigmoid (logistic) function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// 8-feature context vector `(f, ψ, ι, ρ, u, σ, h, g)` for an agent.
fn context_vec(world: &SilenceWorld, id: AgentId) -> [f64; 8] {
    let emp = &world.employees[&id];
    let team = &world.teams[emp.team];
    let rho = emp.perceived_silence; // last step's ρ (synchronous update)
    [
        emp.fear,
        emp.psych_safety,
        emp.ivt_strength,
        rho,
        team.supervisor_openness,
        world.issue_salience,
        emp.harm_perception,
        emp.self_gain,
    ]
}

/// Multinomial-logistic motive softmax row dot products
/// (sign-constrained per design doc §4.4 Table).
///
/// Index of `x` = `(f, ψ, ι, ρ, u, σ, h, g)`.
fn motive_row_logit(motive: Motive, x: &[f64; 8], beta: &BetaGroup, ps_decoupling: bool) -> f64 {
    // Magnitudes are mid-range constants chosen so that motive shares respond
    // smoothly to the corresponding context dimension. Signs are fixed per the
    // design doc Table 2 motive × β sign constraints.
    let beta_rho_ps_effective = if ps_decoupling { 0.0 } else { beta.beta_rho_ps };
    match motive {
        // AS: -ψ + ι + ρ − u
        Motive::Acquiescent => {
            0.0 + (-beta.beta_psafety) * x[1]
                + beta.beta_ivt * x[2]
                + beta.beta_rho * x[3]
                + (-beta.beta_supervisor) * x[4]
        }
        // QS: +f − ψ + ρ − u
        Motive::Quiescent => {
            0.0 + beta.beta_fear * x[0]
                + (-beta.beta_psafety) * x[1]
                + beta.beta_rho * x[3]
                + (-beta.beta_supervisor) * x[4]
        }
        // PS: +u + h + β_ρ^{PS} · ρ (β_ρ^{PS} is the critical knob)
        Motive::Prosocial => {
            0.0 + beta.beta_supervisor * x[4] + 1.0 * x[6] + beta_rho_ps_effective * x[3]
        }
        // OS: -ψ − h + g
        Motive::Opportunistic => (-beta.beta_psafety) * x[1] - x[6] + 1.5 * x[7],
    }
}

/// Softmax over the 4 motives, returning shares in (AS, QS, PS, OS) order.
fn motive_softmax(x: &[f64; 8], beta: &BetaGroup, ps_decoupling: bool) -> [f64; 4] {
    let logits = [
        motive_row_logit(Motive::Acquiescent, x, beta, ps_decoupling),
        motive_row_logit(Motive::Quiescent, x, beta, ps_decoupling),
        motive_row_logit(Motive::Prosocial, x, beta, ps_decoupling),
        motive_row_logit(Motive::Opportunistic, x, beta, ps_decoupling),
    ];
    let m = logits
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let exps: [f64; 4] = [
        (logits[0] - m).exp(),
        (logits[1] - m).exp(),
        (logits[2] - m).exp(),
        (logits[3] - m).exp(),
    ];
    let s: f64 = exps.iter().sum();
    if s <= 0.0 {
        return [0.25; 4];
    }
    [exps[0] / s, exps[1] / s, exps[2] / s, exps[3] / s]
}

/// Sample an index in 0..4 from a categorical distribution.
fn sample_categorical(probs: &[f64; 4], u: f64) -> usize {
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    3
}

/// `voice_decision_rule` mechanism — the multinomial-logistic Phase B1 ablation.
pub struct VoiceDecisionRule {
    beta: BetaGroup,
    ps_decoupling: bool,
}

impl VoiceDecisionRule {
    pub fn new(beta: BetaGroup, ps_decoupling: bool) -> Self {
        VoiceDecisionRule {
            beta,
            ps_decoupling,
        }
    }
}

impl Mechanism<SilenceWorld> for VoiceDecisionRule {
    fn name(&self) -> &str {
        "voice_decision_rule"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        // Synchronous update: snapshot first, then write the new expressions/motives.
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let mut snapshot: Vec<([f64; 8], AgentId)> = Vec::with_capacity(ids.len());
        for id in &ids {
            snapshot.push((context_vec(ctx.world, *id), *id));
        }

        let mut updates: Vec<(AgentId, Expression, Option<Motive>)> = Vec::with_capacity(ids.len());
        for (x, id) in snapshot {
            // VOICE Bernoulli
            let voice_logit = self.beta.voice_intercept
                + self.beta.beta_psafety * x[1]
                + self.beta.beta_supervisor * x[4]
                + self.beta.beta_salience * x[5]
                - self.beta.beta_fear * x[0]
                - self.beta.beta_ivt * x[2]
                - self.beta.beta_rho * x[3];
            let p_voice = sigmoid(voice_logit);
            let u_voice: f64 = ctx.rng.gen();
            if u_voice < p_voice {
                updates.push((id, Expression::Voice, None));
            } else {
                let probs = motive_softmax(&x, &self.beta, self.ps_decoupling);
                let u_motive: f64 = ctx.rng.gen();
                let idx = sample_categorical(&probs, u_motive);
                updates.push((id, Expression::Silence, Some(Motive::ALL[idx])));
            }
        }
        for (id, expr, m) in updates {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            emp.expression = expr;
            emp.silence_motive = m;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 4b. VoiceDecisionLlm  (Decision) — LLM-driven
// --------------------------------------------------------------------------- //

/// LLM-driven voice decision (Phase B2).
///
/// Uses the shared `socsim-llm` harness. Per-agent prompt seeded with one of
/// the 4 persona templates (round-robin from the motive prior at world init).
/// `temperature=0` + a `(agent_id, t)`-derived LLM seed + the prompt→response
/// cache pseudo-determinise generation.
pub struct VoiceDecisionLlm {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    seeded_motives: SeededMotives,
    ps_decoupling: bool,
    /// `derive_seed` root for the (agent_id, t) LLM seed stream.
    llm_seed_root: u64,
}

impl VoiceDecisionLlm {
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        seeded_motives: SeededMotives,
        ps_decoupling: bool,
        llm_seed_root: u64,
    ) -> Self {
        VoiceDecisionLlm {
            client,
            metadata,
            settings,
            seeded_motives,
            ps_decoupling,
            llm_seed_root,
        }
    }
}

impl Mechanism<SilenceWorld> for VoiceDecisionLlm {
    fn name(&self) -> &str {
        "voice_decision"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let t = ctx.clock.t();
        // Snapshot prompts before mutating (synchronous update).
        let mut prompts: Vec<(AgentId, String, u64)> = Vec::with_capacity(ids.len());
        for id in ids {
            let seeded = *self.seeded_motives.get(&id).unwrap_or(&Motive::Acquiescent);
            let persona = persona_for(seeded);
            let prompt = build_silence_prompt(ctx.world, id, persona, seeded, self.ps_decoupling);
            // Per-(agent, t) LLM seed — design §4.4 RNG streams.
            let llm_seed = derive_seed(self.llm_seed_root, &[2, id.0, t]);
            prompts.push((id, prompt, llm_seed));
        }

        let mut updates: Vec<(AgentId, Expression, Option<Motive>)> =
            Vec::with_capacity(prompts.len());
        for (id, prompt, llm_seed) in prompts {
            let mut cfg = llm_config(&self.settings);
            cfg.seed = llm_seed;
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client.complete(&prompt, &cfg).map_err(|e| {
                    SocsimError::Mechanism(format!("voice_decision LLM call failed: {e}"))
                })?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            let verdict = parse_voice_decision(&text);
            updates.push((id, verdict.expression, verdict.motive));
        }
        for (id, expr, m) in updates {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            emp.expression = expr;
            emp.silence_motive = m;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 5. SilenceSpiral  (Interaction)
// --------------------------------------------------------------------------- //

/// `ρ_i ← neighbour silence ratio` (Noelle-Neumann 1974). The synchronous
/// update is implicit here: we read `expression` (which the decision phase
/// has already finalised) and write `perceived_silence` (read next step by
/// the Decision phase).
pub struct SilenceSpiral;

impl Mechanism<SilenceWorld> for SilenceSpiral {
    fn name(&self) -> &str {
        "silence_spiral"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let mut new_rho: Vec<(AgentId, f64)> = Vec::with_capacity(ids.len());
        for id in ids {
            new_rho.push((id, ctx.world.neighbour_silence_ratio(id)));
        }
        for (id, r) in new_rho {
            if let Some(e) = ctx.world.employees.get_mut(&id) {
                e.perceived_silence = r;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 6. PrefalseCascade  (Interaction)
// --------------------------------------------------------------------------- //

/// Threshold cascade (Kuran 1995): a silent agent flips to VOICE if their
/// neighbour-VOICE ratio exceeds `θ_i` (`voice_threshold`).
pub struct PrefalseCascade;

impl Mechanism<SilenceWorld> for PrefalseCascade {
    fn name(&self) -> &str {
        "prefalse_cascade"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let mut flips: Vec<AgentId> = Vec::new();
        for id in ids {
            let emp = &ctx.world.employees[&id];
            if emp.expression == Expression::Silence {
                let rv = ctx.world.neighbour_voice_ratio(id);
                if rv > emp.voice_threshold {
                    flips.push(id);
                }
            }
        }
        for id in flips {
            if let Some(e) = ctx.world.employees.get_mut(&id) {
                e.expression = Expression::Voice;
                e.silence_motive = None;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 7. OrgPerformance  (Reward)
// --------------------------------------------------------------------------- //

/// Per-team knowledge-stock update.
///
/// `K_k(t+1) = (1-δ) K_k(t) + voice_share_k − os_share_k`
/// (VOICE contributes; OS drains; PS / AS / QS are neutral on `K`).
pub struct OrgPerformance {
    delta: f64,
}

impl OrgPerformance {
    pub fn new() -> Self {
        OrgPerformance { delta: 0.10 }
    }
}

impl Default for OrgPerformance {
    fn default() -> Self {
        Self::new()
    }
}

impl Mechanism<SilenceWorld> for OrgPerformance {
    fn name(&self) -> &str {
        "org_performance"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let n_teams = ctx.world.teams.len();
        let mut sizes = vec![0u64; n_teams];
        let mut voice_cnt = vec![0u64; n_teams];
        let mut os_cnt = vec![0u64; n_teams];
        for e in ctx.world.employees.values() {
            sizes[e.team] += 1;
            match e.expression {
                Expression::Voice => voice_cnt[e.team] += 1,
                Expression::Silence => {
                    if e.silence_motive == Some(Motive::Opportunistic) {
                        os_cnt[e.team] += 1;
                    }
                }
                Expression::Neutral => {}
            }
        }
        for k in 0..n_teams {
            let n = sizes[k].max(1) as f64;
            let v = voice_cnt[k] as f64 / n;
            let o = os_cnt[k] as f64 / n;
            let team = &mut ctx.world.teams[k];
            team.knowledge_stock = ((1.0 - self.delta) * team.knowledge_stock + v - o).max(0.0);
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 8. PsafetyUpdate  (PostStep)
// --------------------------------------------------------------------------- //

/// `ψ_i ← clamp(ψ_i + η (voiced?+1:0) − ν (retaliated?+1:0), 0, 1)`.
pub struct PsafetyUpdate {
    eta: f64,
    nu: f64,
}

impl PsafetyUpdate {
    pub fn new() -> Self {
        PsafetyUpdate {
            eta: 0.05,
            nu: 0.15,
        }
    }
}

impl Default for PsafetyUpdate {
    fn default() -> Self {
        Self::new()
    }
}

impl Mechanism<SilenceWorld> for PsafetyUpdate {
    fn name(&self) -> &str {
        "psafety_update"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let retaliated: std::collections::HashSet<AgentId> =
            ctx.world.retaliation_this_step.iter().copied().collect();
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            let voiced = matches!(emp.expression, Expression::Voice);
            let was_retaliated = retaliated.contains(&id);
            let delta =
                self.eta * (voiced as i32 as f64) - self.nu * (was_retaliated as i32 as f64);
            emp.psych_safety = (emp.psych_safety + delta).clamp(0.0, 1.0);
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 9. ClimateSilence  (PostStep)
// --------------------------------------------------------------------------- //

/// Updates `world.climate_of_silence` and per-team `team.climate` to the
/// current step's values.
pub struct ClimateSilence;

impl Mechanism<SilenceWorld> for ClimateSilence {
    fn name(&self) -> &str {
        "climate_silence"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        ctx.world.climate_of_silence = crate::metrics::climate_of_silence(ctx.world);
        let per_team = crate::metrics::team_climates(ctx.world);
        for (k, c) in per_team.into_iter().enumerate() {
            if let Some(team) = ctx.world.teams.get_mut(k) {
                team.climate = c;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// Seeded-motive bootstrap helper (used by `simulation::init_world`)
// --------------------------------------------------------------------------- //

/// Deterministically assign each agent a seeded persona motive by sampling
/// from `motive_prior`. Used by the LLM path's prompt construction.
pub fn seed_motives(
    ids: &[AgentId],
    prior: &MotivePrior,
    rng: &mut socsim_core::SimRng,
) -> BTreeMap<AgentId, Motive> {
    let probs = prior.normalised();
    let mut out = BTreeMap::new();
    for &id in ids {
        let u: f64 = rng.gen();
        let idx = sample_categorical(&probs, u);
        out.insert(id, Motive::ALL[idx]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BetaGroup;

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn motive_softmax_sums_to_one() {
        let beta = BetaGroup::default();
        let x = [0.3, 0.5, 0.3, 0.5, 0.0, 0.5, 0.3, 0.2];
        let p = motive_softmax(&x, &beta, false);
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ps_decoupling_blocks_rho_contribution_to_ps_row() {
        let beta = BetaGroup::default();
        // x with high ρ; PS row should not benefit when decoupling is on.
        let x_high_rho = [0.3, 0.5, 0.3, 0.9, 0.0, 0.5, 0.3, 0.2];
        let with = motive_softmax(&x_high_rho, &beta, false);
        let without = motive_softmax(&x_high_rho, &beta, true);
        // With decoupling, PS share should be ≤ the with-coupling case (β_ρ_ps > 0).
        assert!(
            without[Motive::Prosocial.index()] <= with[Motive::Prosocial.index()] + 1e-9,
            "ps_decoupling did not constrain PS row: with={:?} without={:?}",
            with,
            without
        );
    }

    #[test]
    fn sample_categorical_respects_distribution() {
        // Deterministic edge cases.
        let p = [0.5, 0.0, 0.0, 0.5];
        assert_eq!(sample_categorical(&p, 0.1), 0);
        assert_eq!(sample_categorical(&p, 0.6), 3);
    }
}
