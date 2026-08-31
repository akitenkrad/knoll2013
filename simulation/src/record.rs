//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` でも `sweep` の子でも同一なので，ここ 1 箇所で
//! 組み立てる．ステップごとの指標，run 全体を 1 つの値で表す指標，動機 × correlate の
//! 相関，エージェント 1 体ごとの最終状態の書き方もここに集める．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::simulation::{AgentRow, CorrelationRow, MetricsRow, SimulationResult};

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "knoll";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "knoll2013";
/// 分野．従業員属性・Watts–Strogatz 網・9 機構がいずれも乱数駆動なので `simulation`
/// (= `master_seed` が必須)．`--decision-mode llm` では LLM が意思決定を担うが，
/// 測っているのはモデルの安全性ではなく組織集団に創発するサイレンスなので
/// `llm-safety` ではない．LLM 側の同一性は `run.json` の `llm` ブロックが持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．
///
/// このモデルの刻みは socsim エンジンの離散ステップ (`--t-max`) そのもので，
/// 論文の調査ウェーブのような外部の単位には対応しない．語彙では `step`．
const T_UNIT: &str = "step";

/// 指標の粒度．ステップ指標も相関も母集団全体の集約なので `run`．
const SCOPE: &str = "run";

/// エージェント 1 体の最終状態を表す実験固有のイベント種別．コア語彙に無いので
/// `x.<repo_id>.<name>` を使う．
pub const AGENT_EVENT: &str = "x.knoll2013.agent";

/// この再現実験が対象としている論文．
///
/// `run` も `sweep` の子も同じ主張を対象とする — 掃引は Table 2 の相関パターンが
/// $\beta$ 群のどこで再現されるかを見るためのもので，別の対象を持たない．
pub fn replication() -> Replication {
    let mut work = Work::doi("10.1007/s10551-012-1308-4")
        .title(
            "Do I Hear the Whistle…? A First Attempt to Measure Four Forms of Employee Silence \
             and Their Correlates",
        )
        .year(2013)
        .source_version("published");
    // vault 側の同定にも使えるよう paper-id も残す (work_id は DOI 側)．
    work.paper_id = Some("P00001813".to_string());
    Replication::new(work)
        .target(Target::table("study2-table2", "Table 2"))
        .target(Target::claim(
            "prosocial-climate-independence",
            "Prosocial silence alone is uncorrelated with the climate of silence",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/knoll2013/設計書.md")
}

// --------------------------------------------------------------------------- //
// LLM ブロック
// --------------------------------------------------------------------------- //

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める．推測しているのは分類だけで，値そのものは記録から採る．
///
/// `model_snapshot` に入るのは `llama3.1` のような動くエイリアスであることが多い．
/// socsim-llm はスナップショット id を持たないので，持っていない値を作らずに
/// 名乗られた名前を書く．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // ペルソナ prompt はエージェントごとに組み立てられ，固定の system prompt を
        // 持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// --------------------------------------------------------------------------- //
// シミュレーション 1 本ぶんの記録
// --------------------------------------------------------------------------- //

/// シミュレーション 1 本ぶんを run へ書く (`run` サブコマンドと `sweep` の子で共通)．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    for m in &result.metrics_rows {
        log_step(run, m);
    }
    log_run_scope(run, result);
    log_correlations(run, &result.correlation_rows);
    log_agents(run, result.final_round, &result.agent_rows);
}

/// [`MetricsRow`] の 12 フィールドを 1 ステップぶんまとめて書く．
///
/// `t` は時間軸そのものなので値としては書かない．動機 mix と下位尺度プロキシは
/// 動機ごとに 4 本の指標に割る — カテゴリに番号を振ったのではなく，動機ごとの割合と
/// いう «ステップごとの数» が 4 つあるだけである (mix の 4 本の和は 1，沈黙が
/// 1 人もいなければ 0)．
fn log_step(run: &mut Run, m: &MetricsRow) {
    run.log_metrics_at(
        m.t,
        T_UNIT,
        SCOPE,
        &[
            ("silence_rate", m.silence_rate),
            ("motive_mix_as", m.motive_mix_as),
            ("motive_mix_qs", m.motive_mix_qs),
            ("motive_mix_ps", m.motive_mix_ps),
            ("motive_mix_os", m.motive_mix_os),
            ("subscale_proxy_as", m.subscale_proxy_as),
            ("subscale_proxy_qs", m.subscale_proxy_qs),
            ("subscale_proxy_ps", m.subscale_proxy_ps),
            ("subscale_proxy_os", m.subscale_proxy_os),
            ("climate_of_silence", m.climate_of_silence),
            ("issue_salience", m.issue_salience),
            ("kl_divergence_to_knoll", m.kl_divergence_to_knoll),
        ],
    )
    .unwrap_or_else(|e| panic!("step {} の指標の記録に失敗: {e}", m.t));
}

/// run 全体を 1 つの値で表す指標．
///
/// `n_units` は予約指標名で «観測主体の数» — このモデルでは従業員の数である．
/// 実行時間は `status.json` の `duration_sec` が正本なので指標にはしない．
fn log_run_scope(run: &mut Run, result: &SimulationResult) {
    let calls = result.metadata.total();
    let mut values: Vec<(&str, f64)> = vec![
        ("n_units", result.agent_rows.len() as f64),
        ("final_round", result.final_round as f64),
        ("llm_calls", calls as f64),
        ("llm_cache_hits", result.metadata.cache_hits() as f64),
    ];
    // 呼び出しが 1 本も無いときの cache-hit 率は «0» ではなく «定義できない»．
    // rule モードは LLM を 1 度も呼ばないので，率の行そのものを書かない．
    if calls > 0 {
        values.push(("llm_cache_hit_rate", result.metadata.cache_hit_rate()));
    }
    run.log_metrics(SCOPE, &values)
        .expect("run スコープの指標の記録に失敗");
}

/// 動機 × correlate の Pearson $r$ を run スコープの指標として書く．
///
/// 動機 (AS/QS/PS/OS) と correlate (風土・fear・ψ・ι・harm・self-gain) は，値そのもの
/// ではなく «どの数か» を指す名前なので，`motive_mix_as` と同じく名前に畳む．24 セルは
/// それぞれ別の名前を名乗るので主キー (name, step, step_unit, scope) は衝突しない．
/// run 全体で 1 つしか無い値なので `step` は持たない (最終ステップの状態から求める
/// 集約であって，ステップごとの系列ではない)．
///
/// `metrics::pearson` は分散 0 などの退化した入力に 0 を返す契約なので，動機が 1 人も
/// いなかったセルも 0 の行になる．欠測を 0 で埋めているのではなく，その関数が返した
/// 値をそのまま書いている (契約を変えるのは機構側の話)．
fn log_correlations(run: &mut Run, rows: &[CorrelationRow]) {
    let names: Vec<String> = rows
        .iter()
        .map(|r| format!("corr_{}_{}", r.motive.to_ascii_lowercase(), r.correlate))
        .collect();
    let values: Vec<(&str, f64)> = names
        .iter()
        .zip(rows)
        .map(|(name, row)| (name.as_str(), row.pearson_r))
        .collect();
    run.log_metrics(SCOPE, &values)
        .expect("動機 × correlate 相関の記録に失敗");
}

// --------------------------------------------------------------------------- //
// エージェント 1 体ごとの最終状態
// --------------------------------------------------------------------------- //

/// `events.jsonl` に書くエージェント 1 体の最終状態．
///
/// この行は `metrics.csv` には置けない．主キーが (name, step, step_unit, scope) なので，
/// 従業員ごとの `fear` を並べると全行が同じキーを名乗り，従業員どうしが衝突する
/// (`scope=agent` にしても行を分ける列が無い)．かといって従業員 1 人を子 run に割るのも
/// 実態と違う — 1 回の実行は 1 つの組織を丸ごと回すのであって，起きていない N 本の実行を
/// 主張することになる．従業員は «1 回の実行の中で観測された対象» なので，予約語
/// `unit_id` (観測の主体) を持つイベントとして書く．
///
/// `expression` と `motive` はラベルであって数ではないので，そもそも指標にできない．
/// 沈黙していない従業員に沈黙動機は無いので，欄そのものを落とす (旧 `agents.csv` は
/// `-` という番兵を書いていた．欠測は «無い» と書く方が後から見分けられる)．
///
/// コア語彙の `observation` ではなく実験固有の種別にしてある．`observation` は到達時間
/// の観測 1 点という意味を持つ行で，ここに書くのは «run の終端での状態» だからである．
#[derive(Serialize)]
struct AgentEvent<'a> {
    unit_id: String,
    t: u64,
    t_unit: &'static str,
    agent_id: u64,
    team: usize,
    level: u8,
    tenure: u32,
    expression: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    motive: Option<&'a str>,
    fear: f64,
    psafety: f64,
    ivt: f64,
    perceived_silence: f64,
    harm: f64,
    self_gain: f64,
    private_concern: f64,
}

/// 従業員 1 人につき 1 行書く．
fn log_agents(run: &mut Run, final_round: u64, rows: &[AgentRow]) {
    for row in rows {
        let event = AgentEvent {
            unit_id: format!("agent-{}", row.agent_id),
            t: final_round,
            t_unit: T_UNIT,
            agent_id: row.agent_id,
            team: row.team,
            level: row.level,
            tenure: row.tenure,
            expression: &row.expression,
            motive: row.motive.as_deref(),
            fear: row.fear,
            psafety: row.psafety,
            ivt: row.ivt,
            perceived_silence: row.perceived_silence,
            harm: row.harm,
            self_gain: row.self_gain,
            private_concern: row.private_concern,
        };
        run.log_event(AGENT_EVENT, &event)
            .unwrap_or_else(|e| panic!("agent {} の記録に失敗: {e}", row.agent_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runvault::meta::TargetKind;

    #[test]
    fn the_work_id_agrees_with_the_doi() {
        let research: runvault::meta::Research = replication().into();
        let work = research.work.expect("再現実験なので work がある");
        assert_eq!(work.work_id, "doi:10.1007/s10551-012-1308-4");
        assert_eq!(work.doi.as_deref(), Some("10.1007/s10551-012-1308-4"));
        assert_eq!(work.paper_id.as_deref(), Some("P00001813"));
    }

    #[test]
    fn the_targets_are_the_table_and_the_headline_claim() {
        let research: runvault::meta::Research = replication().into();
        assert_eq!(research.targets.len(), 2);
        assert!(matches!(research.targets[0].kind, TargetKind::Table));
        assert!(matches!(research.targets[1].kind, TargetKind::Claim));
    }

    #[test]
    fn a_run_that_reproduces_the_paper_passes_the_research_checks() {
        let research: runvault::meta::Research = replication().into();
        runvault::verify::check_research(&research).expect("research の検査に失敗");
    }

    #[test]
    fn the_provider_comes_from_the_endpoint() {
        assert_eq!(llm_block("m", "mock://scripted", 0.0).provider, "mock");
        assert_eq!(
            llm_block("m", "https://api.openai.com/v1", 0.0).provider,
            "openai"
        );
        assert_eq!(
            llm_block("m", "http://localhost:11434", 0.0).provider,
            "ollama"
        );
    }
}
