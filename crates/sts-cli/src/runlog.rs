//! Saving a run played in the CLI so it can be watched in the real game
//! (`tools/trace/playback.py` replays it through CommunicationMod).
//!
//! Each step records which screen it was taken on and the CommunicationMod command(s) that do
//! the same thing in the game. Map moves and deck-card selections are stored by meaning (map x,
//! deck positions) because the game's choice indices for them depend on its own state.

use serde::Serialize;
use serde_json::{json, Value};
use sts_core::commands::run_step;
use sts_core::run::{Run, RunAction};

pub use sts_core::commands::{parse_seed, seed_string};

#[derive(Serialize)]
pub struct RunLog {
    /// The character's name for CommunicationMod's `start` ("ironclad", "silent"...).
    pub character: String,
    pub seed: i64,
    pub seed_string: String,
    pub ascension: i32,
    pub neow_full: bool,
    pub steps: Vec<Value>,
}

impl RunLog {
    pub fn new(run: &Run) -> RunLog {
        RunLog {
            character: run.config.character.name().to_string(),
            seed: run.config.seed,
            seed_string: seed_string(run.config.seed),
            ascension: run.config.ascension,
            neow_full: run.config.neow_full,
            steps: Vec::new(),
        }
    }

    /// Records `a`, taken from the run's current state (call before `run.step`).
    pub fn record(&mut self, run: &Run, a: &RunAction) {
        let mut s = run_step(run, a);
        // Where the CLI was, so the playback can tell if the game drifted away from it.
        let hp = run.combat.as_ref().map_or(run.hp, |c| c.player.hp);
        s["floor"] = json!(run.floor);
        s["hp"] = json!(hp);
        self.steps.push(s);
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))
    }
}

/// A saved run worked out again by the current engine from its seed and decisions alone: every
/// step's game commands (`sts_core::commands`), and the floor and HP the engine has there now,
/// which is what playback compares the game with. So a batch keeps playing back right after a
/// fix to the engine's commands or game logic. Returns the new log, how the engine's run ends
/// (`{"won", "act", "floor", "hp", "max_hp", "gold"}`, as python/watch/soak_runs.py saves it) and the
/// first step whose floor or HP changed since the run was saved, if any. Fails only if a saved
/// decision can't be taken any more (the engine reaches a different screen or refuses it).
pub fn recommand(log: &Value) -> Result<(Value, Value, Option<String>), String> {
    let seed = log["seed"].as_i64().ok_or("no seed")?;
    let mut config = sts_core::run::RunConfig::new(seed);
    config.character = match log["character"].as_str() {
        Some(n) => sts_core::character::Character::from_name(n).ok_or(format!("unknown character {n}"))?,
        None => sts_core::character::Character::Ironclad,
    };
    config.ascension = log["ascension"].as_i64().unwrap_or(0) as i32;
    config.neow_full = log["neow_full"].as_bool().unwrap_or(true);
    config.final_act = log["final_act"].as_bool().unwrap_or(true);
    let mut run = Run::new(config);
    let mut out = log.clone();
    let mut changed = None;
    let steps = out["steps"].as_array_mut().ok_or("no steps")?;
    for (i, step) in steps.iter_mut().enumerate() {
        let a: RunAction = serde_json::from_value(step["action"].clone()).map_err(|e| format!("step {i}: {e}"))?;
        let mut s = run_step(&run, &a);
        s["floor"] = json!(run.floor);
        s["hp"] = json!(run.combat.as_ref().map_or(run.hp, |c| c.player.hp));
        if s["kind"] != step["kind"] {
            return Err(format!("step {i}: saved on {}, the engine is on {}", step["kind"], s["kind"]));
        }
        if changed.is_none() && (&s["floor"], &s["hp"]) != (&step["floor"], &step["hp"]) {
            changed = Some(format!(
                "from step {i}: saved at floor {} hp {}, the engine is at floor {} hp {}",
                step["floor"], step["hp"], s["floor"], s["hp"]
            ));
        }
        run.step(&a).map_err(|e| format!("step {i} ({a:?}): {e}"))?;
        *step = s;
    }
    let won = run.outcome == Some(sts_core::run::RunOutcome::Won);
    let end = json!({"won": won, "act": run.act, "floor": run.floor, "hp": run.hp, "max_hp": run.max_hp, "gold": run.gold});
    Ok((out, end, changed))
}
