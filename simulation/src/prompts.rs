//! LLM prompt construction and response parsing for `voice_decision`.
//!
//! The LLM is asked, given an employee persona and the 8-feature local
//! context, to return a JSON decision of the form
//!
//! ```json
//! {
//!   "decision": "voice" | "silence",
//!   "motive":   "acquiescent" | "quiescent" | "prosocial" | "opportunistic" | null,
//!   "rationale": "free-form short string"
//! }
//! ```
//!
//! When `decision = silence`, `motive` is required; when `decision = voice`,
//! `motive` is `null`. Parse failures fall back to `Silence + None` (counted
//! against the "parse-failure rate ≤ 5%" Phase-B1 acceptance criterion).
//!
//! The persona seeded for each agent encodes Knoll's 4-motive priors:
//! AS-leaning, QS-leaning, PS-leaning, OS-leaning. Under
//! `prosocial_climate_decoupling`, the PS persona omits any reference to the
//! organisational climate of silence (= the LLM-mode analogue of forcing
//! `β_ρ^{PS} = 0` in the rule-based ablation).

use serde::Deserialize;
use serde_json::Value;

use crate::world::{Employee, Expression, Motive, SilenceWorld};

// --------------------------------------------------------------------------- //
// Persona library
// --------------------------------------------------------------------------- //

/// 4 persona templates, one per Knoll motive. Round-robin assignment based on
/// the motive prior at world init seeds the LLM's silence motive distribution.
pub const PERSONAS: [&str; 4] = [
    // AS — acquiescent: resigned, gave-up
    "a long-tenured employee who has stopped speaking up because nothing ever \
     changes; they perceive their voice as ineffective and have accepted that.",
    // QS — quiescent: fearful, self-protective
    "an employee who privately worries about retaliation from supervisors or \
     colleagues; they keep concerns to themselves to avoid negative consequences.",
    // PS — prosocial: protective of others
    "a colleague who withholds critical comments to spare others' feelings, \
     protect co-workers from embarrassment, or keep peers out of trouble.",
    // OS — opportunistic: strategic / self-interested
    "an ambitious individual who selectively withholds information when sharing \
     it could create extra work for them or weaken their personal advantage.",
];

/// Persona for `motive` (the 4-template library indexed by motive order).
pub fn persona_for(motive: Motive) -> &'static str {
    PERSONAS[motive.index()]
}

// --------------------------------------------------------------------------- //
// Prompt construction
// --------------------------------------------------------------------------- //

/// Build the voice-decision prompt for `agent_id` from the world.
///
/// If `prosocial_climate_decoupling = true` AND the agent's seeded persona is
/// the PS persona, the climate/neighbour-silence sentence is *omitted* — the
/// LLM-mode mirror of `β_ρ^{PS} = 0`.
pub fn build_silence_prompt(
    world: &SilenceWorld,
    agent_id: socsim_core::AgentId,
    persona: &str,
    seeded_motive: Motive,
    prosocial_climate_decoupling: bool,
) -> String {
    let emp = &world.employees[&agent_id];
    let team = &world.teams[emp.team];
    let rho = world.neighbour_silence_ratio(agent_id);
    let sigma = world.issue_salience;

    let context = format_context(emp, team.supervisor_openness, sigma, rho);
    let climate_line = if prosocial_climate_decoupling && seeded_motive == Motive::Prosocial {
        // PS persona: omit climate cue (mirror of β_ρ^{PS} = 0)
        String::new()
    } else {
        format!(
            "Around you, {neigh_pct:.0}% of your colleagues are currently silent \
             about workplace concerns.\n",
            neigh_pct = rho * 100.0
        )
    };

    format!(
        "You are {persona}\n\n\
         Today an ethically questionable practice has arisen at work. You must \
         decide whether to SPEAK UP (voice) or REMAIN SILENT.\n\n\
         Your inner state:\n\
         {context}\n\
         {climate_line}\n\
         Reply with a SINGLE JSON object on one line:\n\
         {{\"decision\": \"voice\" | \"silence\", \
            \"motive\": \"acquiescent\" | \"quiescent\" | \"prosocial\" | \"opportunistic\" | null, \
            \"rationale\": \"short reason\"}}\n\
         Rules: if decision = voice, motive must be null. If decision = silence, \
         motive must be one of the four labels (acquiescent / quiescent / prosocial / opportunistic). \
         Output JSON only.",
    )
}

fn format_context(emp: &Employee, supervisor_openness: f64, sigma: f64, rho: f64) -> String {
    format!(
        "  fear of consequences      f = {f:.2}\n\
         \x20 psychological safety      ψ = {psi:.2}\n\
         \x20 implicit-voice theory     ι = {iota:.2}\n\
         \x20 supervisor openness       u = {u:+.2}\n\
         \x20 issue salience            σ = {sigma:.2}\n\
         \x20 harm-to-others concern    h = {h:.2}\n\
         \x20 self-gain opportunity     g = {g:.2}\n\
         \x20 perceived peer silence    ρ = {rho:.2}\n",
        f = emp.fear,
        psi = emp.psych_safety,
        iota = emp.ivt_strength,
        u = supervisor_openness,
        sigma = sigma,
        h = emp.harm_perception,
        g = emp.self_gain,
        rho = rho,
    )
}

// --------------------------------------------------------------------------- //
// Response parsing
// --------------------------------------------------------------------------- //

/// Parsed voice-decision verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceDecisionVerdict {
    pub expression: Expression,
    pub motive: Option<Motive>,
    pub rationale: String,
    /// True if the LLM response failed to parse and we fell back to the
    /// "silence + None" default — counted against the parse-failure rate.
    pub parse_failed: bool,
}

#[derive(Deserialize)]
struct RawDecision {
    decision: Option<String>,
    motive: Option<String>,
    rationale: Option<String>,
}

/// Parse an LLM response into a verdict.
///
/// Lenient: extracts the first `{...}` JSON object substring, accepts mixed-case
/// labels, and on any failure falls back to `Silence + None` (parse_failed = true).
pub fn parse_voice_decision(text: &str) -> VoiceDecisionVerdict {
    let fallback = VoiceDecisionVerdict {
        expression: Expression::Silence,
        motive: None,
        rationale: String::new(),
        parse_failed: true,
    };

    let json_str = match extract_json_object(text) {
        Some(s) => s,
        None => return fallback,
    };

    // Try the typed parse first.
    if let Ok(raw) = serde_json::from_str::<RawDecision>(&json_str) {
        return finalise_verdict(raw);
    }
    // Fall back to Value parse for non-strict JSON.
    if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
        let raw = RawDecision {
            decision: val
                .get("decision")
                .and_then(|v| v.as_str().map(str::to_string)),
            motive: val.get("motive").and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    v.as_str().map(str::to_string)
                }
            }),
            rationale: val
                .get("rationale")
                .and_then(|v| v.as_str().map(str::to_string)),
        };
        return finalise_verdict(raw);
    }
    fallback
}

fn finalise_verdict(raw: RawDecision) -> VoiceDecisionVerdict {
    let decision = raw
        .decision
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let expression = match decision.as_str() {
        "voice" | "speak" | "speak_up" => Expression::Voice,
        "silence" | "silent" | "withhold" => Expression::Silence,
        _ => {
            return VoiceDecisionVerdict {
                expression: Expression::Silence,
                motive: None,
                rationale: raw.rationale.unwrap_or_default(),
                parse_failed: true,
            };
        }
    };

    let motive = if expression == Expression::Voice {
        None
    } else {
        match raw
            .motive
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("acquiescent") | Some("as") => Some(Motive::Acquiescent),
            Some("quiescent") | Some("qs") => Some(Motive::Quiescent),
            Some("prosocial") | Some("ps") => Some(Motive::Prosocial),
            Some("opportunistic") | Some("os") => Some(Motive::Opportunistic),
            // Silence without a recognised motive → parse_failed, default None
            _ => {
                return VoiceDecisionVerdict {
                    expression: Expression::Silence,
                    motive: None,
                    rationale: raw.rationale.unwrap_or_default(),
                    parse_failed: true,
                };
            }
        }
    };

    VoiceDecisionVerdict {
        expression,
        motive,
        rationale: raw.rationale.unwrap_or_default(),
        parse_failed: false,
    }
}

/// Extract the first balanced `{...}` substring from `text`. Returns `None` if
/// no balanced object is found.
fn extract_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_voice() {
        let v = parse_voice_decision(r#"{"decision": "voice", "motive": null, "rationale": "ok"}"#);
        assert_eq!(v.expression, Expression::Voice);
        assert_eq!(v.motive, None);
        assert!(!v.parse_failed);
    }

    #[test]
    fn parses_canonical_silence_with_motive() {
        let v = parse_voice_decision(
            r#"{"decision":"silence","motive":"prosocial","rationale":"protect peer"}"#,
        );
        assert_eq!(v.expression, Expression::Silence);
        assert_eq!(v.motive, Some(Motive::Prosocial));
        assert!(!v.parse_failed);
    }

    #[test]
    fn tolerates_surrounding_text() {
        let v = parse_voice_decision(
            r#"Sure, here is my answer: {"decision":"silence","motive":"qs","rationale":"fear"}. End."#,
        );
        assert_eq!(v.expression, Expression::Silence);
        assert_eq!(v.motive, Some(Motive::Quiescent));
    }

    #[test]
    fn unknown_decision_falls_back() {
        let v = parse_voice_decision(r#"{"decision":"???","motive":"as","rationale":""}"#);
        assert!(v.parse_failed);
        assert_eq!(v.expression, Expression::Silence);
        assert_eq!(v.motive, None);
    }

    #[test]
    fn silence_with_unknown_motive_falls_back() {
        let v = parse_voice_decision(r#"{"decision":"silence","motive":"foo","rationale":""}"#);
        assert!(v.parse_failed);
        assert_eq!(v.expression, Expression::Silence);
        assert_eq!(v.motive, None);
    }

    #[test]
    fn no_json_object_falls_back() {
        let v = parse_voice_decision("no json here");
        assert!(v.parse_failed);
        assert_eq!(v.expression, Expression::Silence);
    }

    #[test]
    fn personas_indexed_by_motive() {
        // Persona library uses canonical (AS, QS, PS, OS) order; persona_for
        // must hit the right slot.
        for m in Motive::ALL {
            assert_eq!(persona_for(m), PERSONAS[m.index()]);
        }
    }
}
