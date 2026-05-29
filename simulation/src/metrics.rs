//! Aggregate metrics for the Knoll & van Dick (2013) four-form silence model.
//!
//! - **silence_rate** — fraction of employees with `expression == Silence`.
//! - **motive_mix_{AS,QS,PS,OS}** — within-silent motive share (Σ = 1 when
//!   any silence exists; 0 otherwise).
//! - **subscale_proxy_{AS,QS,PS,OS}** — per-agent motive-frequency mapped to
//!   a Likert 1–7 scale (used as a sub-scale rating proxy in lieu of the
//!   reflexive self-rating; aggregated to a population mean).
//! - **climate_of_silence** — `C(t) = (1/N) Σ 1[b̂_i = Silence ∧ b_i < 0]`
//!   (Morrison 2000).
//! - **corr_motive_correlate** — Pearson correlation between motive-mix and
//!   a correlate (here: climate of silence) over a vector of agent-step rows.
//! - **kl_divergence_to_knoll** — KL(π_emp ‖ π_abm) where π_emp is Knoll
//!   Study 2 subscale means.

use crate::world::{Expression, SilenceWorld};

/// Knoll Study 2 subscale means (AS=.22, QS=.27, PS=.40, OS=.18 — design §4.4).
/// Used as the reference distribution for `kl_divergence_to_knoll`.
pub const KNOLL_REFERENCE_MIX: [f64; 4] = [0.22, 0.27, 0.40, 0.18];

/// Fraction of employees in `Silence` expression.
pub fn silence_rate(world: &SilenceWorld) -> f64 {
    let n = world.n_employees();
    if n == 0 {
        return 0.0;
    }
    let silent = world
        .employees
        .values()
        .filter(|e| e.expression == Expression::Silence)
        .count();
    silent as f64 / n as f64
}

/// 4-vector `(AS, QS, PS, OS)` of within-silent motive shares (Σ = 1 if any
/// silence exists; the all-zero vector otherwise).
pub fn motive_mix(world: &SilenceWorld) -> [f64; 4] {
    let mut counts = [0u64; 4];
    let mut total = 0u64;
    for e in world.employees.values() {
        if e.expression == Expression::Silence {
            if let Some(m) = e.silence_motive {
                counts[m.index()] += 1;
                total += 1;
            }
        }
    }
    if total == 0 {
        return [0.0; 4];
    }
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = counts[i] as f64 / total as f64;
    }
    out
}

/// Per-motive Likert-7 proxy `(AS, QS, PS, OS)` aggregated as a population
/// mean. We map: an employee's per-motive frequency × 6 + 1 → Likert 1..7,
/// then average across employees. Acts as a sub-scale rating stand-in until
/// Phase B3's reflexive self-rating ships.
pub fn subscale_proxy(motive_mix: [f64; 4]) -> [f64; 4] {
    [
        motive_mix[0] * 6.0 + 1.0,
        motive_mix[1] * 6.0 + 1.0,
        motive_mix[2] * 6.0 + 1.0,
        motive_mix[3] * 6.0 + 1.0,
    ]
}

/// Climate of silence `C(t) = (1/N) Σ 1[Silence ∧ private_concern < 0]`
/// (Morrison & Milliken 2000).
pub fn climate_of_silence(world: &SilenceWorld) -> f64 {
    let n = world.n_employees();
    if n == 0 {
        return 0.0;
    }
    let cnt = world
        .employees
        .values()
        .filter(|e| e.expression == Expression::Silence && e.private_concern < 0.0)
        .count();
    cnt as f64 / n as f64
}

/// Per-team climate-of-silence values, one entry per team in `world.teams`.
pub fn team_climates(world: &SilenceWorld) -> Vec<f64> {
    let n_teams = world.teams.len();
    let mut counts = vec![0u64; n_teams];
    let mut sizes = vec![0u64; n_teams];
    for e in world.employees.values() {
        sizes[e.team] += 1;
        if e.expression == Expression::Silence && e.private_concern < 0.0 {
            counts[e.team] += 1;
        }
    }
    let mut out = vec![0.0; n_teams];
    for k in 0..n_teams {
        out[k] = if sizes[k] == 0 {
            0.0
        } else {
            counts[k] as f64 / sizes[k] as f64
        };
    }
    out
}

/// Pearson correlation between paired `x` and `y` vectors. Returns 0 on
/// degenerate inputs (length mismatch / fewer than 2 points / zero variance).
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        sxy / denom
    }
}

/// KL divergence `D_KL(π_emp ‖ π_abm)`.
///
/// Tiny floor on `π_abm` to avoid `log(0)`; `π_emp = 0` rows contribute 0
/// regardless (standard convention).
pub fn kl_divergence_to_knoll(motive_mix: [f64; 4]) -> f64 {
    let p_emp = KNOLL_REFERENCE_MIX;
    let p_abm = motive_mix;
    let eps = 1e-9;
    let mut acc = 0.0;
    for i in 0..4 {
        let pe = p_emp[i];
        if pe <= 0.0 {
            continue;
        }
        let pa = p_abm[i].max(eps);
        acc += pe * (pe / pa).ln();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Employee, Motive, SilenceWorld, Team};
    use socsim_core::{AgentId, SimClock, SimRng};
    use socsim_net::SocialNetwork;
    use std::collections::BTreeMap;

    fn mini_world(exprs: &[Expression], motives: &[Option<Motive>]) -> SilenceWorld {
        assert_eq!(exprs.len(), motives.len());
        let mut rng = SimRng::from_seed(0);
        let ids: Vec<AgentId> = (0..exprs.len()).map(|i| AgentId(i as u64)).collect();
        let net = SocialNetwork::erdos_renyi(&ids, 0.5, &mut rng);
        let mut emps: BTreeMap<AgentId, Employee> = BTreeMap::new();
        for (i, &id) in ids.iter().enumerate() {
            let mut e = Employee::neutral(0, 0, 0);
            e.expression = exprs[i];
            e.silence_motive = motives[i];
            e.private_concern = -0.5; // negative → counts toward climate
            emps.insert(id, e);
        }
        SilenceWorld::new(SimClock::new(1), emps, vec![Team::default()], net)
    }

    #[test]
    fn motive_mix_sums_to_one_when_any_silence() {
        let w = mini_world(
            &[
                Expression::Silence,
                Expression::Silence,
                Expression::Silence,
                Expression::Voice,
            ],
            &[
                Some(Motive::Acquiescent),
                Some(Motive::Quiescent),
                Some(Motive::Prosocial),
                None,
            ],
        );
        let mix = motive_mix(&w);
        let total: f64 = mix.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "mix sum = {total}");
    }

    #[test]
    fn motive_mix_zero_when_no_silence() {
        let w = mini_world(&[Expression::Voice, Expression::Voice], &[None, None]);
        let mix = motive_mix(&w);
        assert_eq!(mix, [0.0; 4]);
    }

    #[test]
    fn climate_counts_only_disagreeing_silent() {
        let w = mini_world(
            &[Expression::Silence, Expression::Silence],
            &[Some(Motive::Acquiescent), Some(Motive::Prosocial)],
        );
        // Both employees have private_concern = -0.5 (disagreeing).
        assert!((climate_of_silence(&w) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_zero_variance_is_zero() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0];
        assert_eq!(pearson(&x, &y), 0.0);
    }

    #[test]
    fn kl_zero_when_matching() {
        let kl = kl_divergence_to_knoll(KNOLL_REFERENCE_MIX);
        assert!(kl.abs() < 1e-12, "kl = {kl}");
    }
}
