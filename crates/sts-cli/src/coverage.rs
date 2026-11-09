//! `sts-cli coverage <recording and soak folders...>`: which cards, relics, potions, powers,
//! monsters and events the recordings and soak runs that match the real game have exercised,
//! for docs/checklist's real-game boxes (`tools/trace/tick_coverage.py` ticks them).
//!
//! What counts:
//! - Fight traces (`trace-*.jsonl`, tools/trace/record.py) and run traces (`run-*.jsonl`,
//!   record_run.py) whose replay matches the game; soak runs the soak found matched
//!   (`summary.jsonl`) and that the current engine still plays from their saved decisions to the
//!   same end (`batch.json`).
//! - Cards played (Reflex and Tactician when a card discards them, Deus Ex Machina when drawn;
//!   curses and statuses when in hand as cards are played or the turn ends) and potions drunk
//!   (Fairy in a Bottle when it saved the player);
//!   the powers on anyone after each step; each monster's moves
//!   (the move every monster still standing makes when the turn ends); events' options taken.
//! - A relic counts when it changed something. A fight trace replayed without it no longer
//!   matches the game; or a room of a matching run, played again from its start without the
//!   relic, comes out differently (anything but the relics themselves: HP, gold, deck, potions,
//!   the screen and its options, the RNG streams, the fight's piles, energy, powers and
//!   monsters). Relics whose whole effect is on pickup count when picked up in a matching run.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sts_core::cards::CardId;
use sts_core::combat::{Combat, PlayerAction};
use sts_core::relics::RelicId;
use sts_core::run::{Run, RunAction, RunConfig, Screen};

use crate::replay::replay_observed;
use crate::run_replay::replay_run_observed;

/// Told about every engine step of a replay.
pub trait Observer {
    fn combat_before(&mut self, _c: &Combat, _a: &PlayerAction) {}
    fn combat_after(&mut self, _c: &Combat) {}
    fn run_start(&mut self, _cfg: &RunConfig) {}
    /// The recorder's HP override (record_run.py sets 400 after Neow).
    fn run_hp(&mut self, _hp: i32, _max_hp: i32) {}
    /// The recorder's relic list (record_run.py --relics).
    fn run_relics(&mut self, _ids: &[RelicId]) {}
    fn run_before(&mut self, _r: &Run, _a: &RunAction) {}
    fn run_after(&mut self, _r: &Run) {}
}

pub struct NoObserver;
impl Observer for NoObserver {}

/// Relics whose whole effect happens when they are picked up (`onEquip`).
const ON_PICKUP: &[RelicId] = &[
    RelicId::Astrolabe,
    RelicId::BottledFlame,
    RelicId::BottledLightning,
    RelicId::BottledTornado,
    RelicId::CallingBell,
    RelicId::Cauldron,
    RelicId::DollysMirror,
    RelicId::EmptyCage,
    RelicId::Mango,
    RelicId::Necronomicon,
    RelicId::OldCoin,
    RelicId::Orrery,
    RelicId::PandorasBox,
    RelicId::Pear,
    RelicId::PotionBelt,
    RelicId::Strawberry,
    RelicId::TinyHouse,
    RelicId::Waffle,
    RelicId::WarPaint,
    RelicId::Whetstone,
];

/// Relics that do nothing in a game (Circlet stands in for a duplicate; the others are only
/// sound or display): they count when held through a matching fight or picked up in a matching
/// run.
const NO_EFFECT: &[RelicId] = &[RelicId::Circlet, RelicId::CultistMask, RelicId::FrozenEye, RelicId::SpiritPoop];

#[derive(Clone)]
enum Step {
    Act(RunAction),
    Hp(i32, i32),
    Relics(Vec<RelicId>),
}

fn bump(m: &mut BTreeMap<String, u32>, k: &str) {
    *m.entry(k.to_string()).or_default() += 1;
}

/// What one recording, or all of them, exercised.
#[derive(Default)]
pub struct Coverage {
    cards: BTreeMap<String, u32>,
    cards_upgraded: BTreeMap<String, u32>,
    /// Curses and statuses in hand when a card is played or the turn ends (they can't be played;
    /// their effects work from the hand).
    cards_held: BTreeMap<String, u32>,
    potions: BTreeMap<String, u32>,
    powers: BTreeMap<String, u32>,
    monsters: BTreeMap<String, BTreeSet<u8>>,
    events: BTreeMap<String, BTreeSet<String>>,
    relics: BTreeMap<String, u32>,
    picked_up: BTreeSet<RelicId>,
    /// Reflex and Tactician in hand before a step (they work when a card or choice discards them).
    discard_triggers: Vec<(u32, &'static str)>,
    /// Fairy in a Bottle slots before a step that doesn't use or discard a potion.
    fairies: Vec<usize>,
    cfg: Option<RunConfig>,
    steps: Vec<Step>,
    relics_before: Vec<RelicId>,
}

impl Observer for Coverage {
    fn combat_before(&mut self, c: &Combat, a: &PlayerAction) {
        if matches!(a, PlayerAction::PlayCard { .. } | PlayerAction::EndTurn) {
            for card in &c.hand {
                if matches!(card.card_type(), sts_core::cards::CardType::Curse | sts_core::cards::CardType::Status) {
                    bump(&mut self.cards_held, card.data().java_id);
                }
            }
        }
        self.fairies = if matches!(a, PlayerAction::UsePotion { .. } | PlayerAction::DiscardPotion { .. }) {
            Vec::new()
        } else {
            (0..c.potions.len()).filter(|&i| c.potions[i] == Some(sts_core::potions::PotionId::FairyPotion)).collect()
        };
        self.discard_triggers = if matches!(a, PlayerAction::EndTurn) {
            Vec::new()
        } else {
            c.hand.iter().filter(|h| matches!(h.id, CardId::Reflex | CardId::Tactician)).map(|h| (h.instance, h.data().java_id)).collect()
        };
        match a {
            PlayerAction::PlayCard { hand_index, .. } => {
                if let Some(card) = c.hand.get(*hand_index) {
                    bump(&mut self.cards, card.data().java_id);
                    if card.upgraded {
                        bump(&mut self.cards_upgraded, card.data().java_id);
                    }
                }
            }
            PlayerAction::UsePotion { slot, .. } => {
                if let Some(Some(p)) = c.potions.get(*slot) {
                    bump(&mut self.potions, p.java_id());
                }
            }
            PlayerAction::EndTurn => {
                // As the engine queues them: a half-dead monster (Darkling, Awakened One) acts too.
                for &m in &c.order {
                    let mon = &c.monsters[m];
                    if !mon.is_dead_or_escaped() || mon.half_dead {
                        self.monsters.entry(mon.id.java_id().to_string()).or_default().insert(mon.next_move);
                    }
                }
            }
            _ => {}
        }
    }

    fn combat_after(&mut self, c: &Combat) {
        // Reflex and Tactician can't be played: they count when discarded from the hand by a card
        // or a choice; Deus Ex Machina when drawing it exhausted it.
        for (inst, id) in std::mem::take(&mut self.discard_triggers) {
            if c.discard_pile.iter().any(|d| d.instance == inst) {
                bump(&mut self.cards, id);
            }
        }
        // Fairy in a Bottle can't be drunk: it counts when it saved the player.
        for i in std::mem::take(&mut self.fairies) {
            if c.potions.get(i) == Some(&None) {
                bump(&mut self.potions, sts_core::potions::PotionId::FairyPotion.java_id());
            }
        }
        if c.exhaust_pile.iter().any(|x| x.id == CardId::DeusExMachina) {
            bump(&mut self.cards, CardId::DeusExMachina.data().java_id);
        }
        for p in &c.player.powers {
            bump(&mut self.powers, p.java_id());
        }
        for &m in &c.order {
            if !c.monsters[m].is_dead_or_escaped() {
                for p in &c.monsters[m].powers {
                    bump(&mut self.powers, p.java_id());
                }
            }
        }
    }

    fn run_start(&mut self, cfg: &RunConfig) {
        self.cfg = Some(cfg.clone());
    }

    fn run_hp(&mut self, hp: i32, max_hp: i32) {
        self.steps.push(Step::Hp(hp, max_hp));
    }

    fn run_relics(&mut self, ids: &[RelicId]) {
        self.steps.push(Step::Relics(ids.to_vec()));
    }

    fn run_before(&mut self, r: &Run, a: &RunAction) {
        self.steps.push(Step::Act(a.clone()));
        self.relics_before = r.relics.iter().map(|x| x.id).collect();
        match (&r.screen, a) {
            (Screen::Event { event }, RunAction::EventChoice { index }) => {
                self.events.entry(event.id.key().to_string()).or_default().insert(format!("{}/{}", event.screen, index));
            }
            (Screen::Combat, RunAction::Combat { action }) => {
                if let Some(c) = &r.combat {
                    self.combat_before(c, action);
                }
            }
            (_, RunAction::UsePotion { slot }) => {
                if let Some(Some(p)) = r.potions.get(*slot) {
                    bump(&mut self.potions, p.java_id());
                }
            }
            _ => {}
        }
    }

    fn run_after(&mut self, r: &Run) {
        if let Some(c) = &r.combat {
            self.combat_after(c);
        }
        for x in &r.relics {
            if !self.relics_before.contains(&x.id) {
                self.picked_up.insert(x.id);
            }
        }
    }
}

impl Coverage {
    fn merge(&mut self, o: Coverage) {
        for (a, b) in [
            (&mut self.cards, o.cards),
            (&mut self.cards_upgraded, o.cards_upgraded),
            (&mut self.cards_held, o.cards_held),
            (&mut self.potions, o.potions),
            (&mut self.powers, o.powers),
            (&mut self.relics, o.relics),
        ] {
            for (k, v) in b {
                *a.entry(k).or_default() += v;
            }
        }
        for (k, v) in o.monsters {
            self.monsters.entry(k).or_default().extend(v);
        }
        for (k, v) in o.events {
            self.events.entry(k).or_default().extend(v);
        }
    }

    /// A matching run: its relics' effects (see the module notes), then fold it in.
    fn finish_run(mut self, total: &mut Coverage) {
        if let Some(cfg) = self.cfg.take() {
            let mut found = run_relic_effects(&cfg, &self.steps);
            found.extend(self.picked_up.iter().copied().filter(|r| ON_PICKUP.contains(r) || NO_EFFECT.contains(r)));
            for r in found {
                bump(&mut self.relics, r.data().java_id);
            }
        }
        total.merge(self);
    }
}

fn apply(run: &mut Run, s: &Step) -> Result<(), String> {
    match s {
        Step::Act(a) => run.step(a),
        Step::Hp(hp, max) => {
            run.max_hp = *max;
            run.hp = *hp;
            Ok(())
        }
        Step::Relics(ids) => {
            crate::run_replay::set_relics(run, ids);
            Ok(())
        }
    }
}

/// Everything about the run but its relics, hashed.
fn fingerprint(r: &Run) -> u64 {
    let mut h = DefaultHasher::new();
    (r.act, r.floor, r.hp, r.max_hp, r.gold, r.card_blizz, r.potion_blizz).hash(&mut h);
    for c in &r.deck {
        (c.id, c.times_upgraded, c.misc, c.bottled).hash(&mut h);
    }
    format!("{:?}{:?}", r.potions, r.keys).hash(&mut h);
    serde_json::to_string(&r.screen).unwrap_or_default().hash(&mut h);
    if !matches!(r.screen, Screen::Combat) {
        format!("{:?}", r.legal_actions()).hash(&mut h);
    }
    let rng = match &r.combat {
        Some(c) => &c.rng,
        None => &r.rng,
    };
    for s in [&rng.monster, &rng.map, &rng.event, &rng.merchant, &rng.card, &rng.treasure, &rng.relic, &rng.potion, &rng.monster_hp, &rng.ai, &rng.shuffle, &rng.card_random, &rng.misc] {
        s.counter.hash(&mut h);
    }
    if let Some(c) = &r.combat {
        let p = &c.player;
        (p.hp, p.max_hp, p.block, p.gold, c.energy, c.turn).hash(&mut h);
        format!("{:?}{:?}{:?}", p.powers, p.orbs, p.stance).hash(&mut h);
        for pile in [&c.hand, &c.draw_pile, &c.discard_pile, &c.exhaust_pile] {
            for card in pile.iter() {
                (card.id, card.cost_for_turn, card.upgraded).hash(&mut h);
            }
            0u8.hash(&mut h);
        }
        for m in &c.monsters {
            (m.hp, m.max_hp, m.block, m.next_move, m.is_dying, m.is_escaping).hash(&mut h);
            format!("{:?}", m.powers).hash(&mut h);
        }
        format!("{:?}{:?}", c.outcome, c.pending.as_ref().map(|p| &p.options)).hash(&mut h);
    }
    h.finish()
}

/// The relics that changed something in a matching run: each room (from one map move to the
/// next) is played again from its start without the relic, and compared step by step.
fn run_relic_effects(cfg: &RunConfig, steps: &[Step]) -> BTreeSet<RelicId> {
    let mut found = BTreeSet::new();
    let starts: Vec<usize> = (0..steps.len()).filter(|&i| matches!(steps[i], Step::Act(RunAction::Path { .. }))).collect();
    let mut run = Run::new(cfg.clone());
    let mut at = 0;
    for (n, &start) in starts.iter().enumerate() {
        while at < start {
            if apply(&mut run, &steps[at]).is_err() {
                return found;
            }
            at += 1;
        }
        let end = starts.get(n + 1).copied().unwrap_or(steps.len());
        let held: Vec<RelicId> = run.relics.iter().map(|r| r.id).filter(|r| !found.contains(r)).collect();
        if held.is_empty() {
            continue;
        }
        let mut base = run.clone();
        let mut prints = Vec::new();
        for s in &steps[start..end] {
            if apply(&mut base, s).is_err() {
                break;
            }
            prints.push(fingerprint(&base));
        }
        for r in held {
            let mut without = run.clone();
            without.relics.retain(|x| x.id != r);
            for (k, s) in steps[start..end].iter().enumerate() {
                if k >= prints.len() {
                    break;
                }
                if apply(&mut without, s).is_err() || fingerprint(&without) != prints[k] {
                    found.insert(r);
                    break;
                }
            }
        }
    }
    found
}

/// The relics a matching fight trace needed: replayed without each one (taken out of the first
/// recorded state, which the replay builds the fight from), it no longer matches.
fn fight_relic_effects(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let Some((si, first)) = lines.iter().enumerate().find_map(|(i, l)| {
        let v: Value = serde_json::from_str(l).ok()?;
        (v["type"] == "state").then_some((i, v))
    }) else {
        return found;
    };
    let relics: Vec<Value> = first["state"]["relics"].as_array().cloned().unwrap_or_default();
    for (k, r) in relics.iter().enumerate() {
        let mut v = first.clone();
        let rest: Vec<Value> = relics.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, x)| x.clone()).collect();
        v["state"]["relics"] = Value::Array(rest);
        let mut copy: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        copy[si] = v.to_string();
        if !replay_observed(&copy.join("\n"), &mut NoObserver).ok {
            if let Some(id) = r["id"].as_str() {
                found.push(id.to_string());
            }
        }
    }
    found
}

#[derive(Default)]
struct Tally {
    fights: (u32, u32),
    runs: (u32, u32),
    soak: (u32, u32),
}

fn fight_file(path: &Path, total: &mut Coverage, t: &mut Tally) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    let mut cov = Coverage::default();
    t.fights.1 += 1;
    if replay_observed(&text, &mut cov).ok {
        t.fights.0 += 1;
        for r in fight_relic_effects(&text) {
            bump(&mut cov.relics, &r);
        }
        for r in NO_EFFECT {
            if text.contains(&format!("\"id\": \"{}\"", r.data().java_id)) {
                bump(&mut cov.relics, r.data().java_id);
            }
        }
        total.merge(cov);
    }
}

fn run_file(path: &Path, total: &mut Coverage, t: &mut Tally) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    let mut cov = Coverage::default();
    t.runs.1 += 1;
    let res = replay_run_observed(&text, false, &mut cov);
    if res.ok {
        t.runs.0 += 1;
        cov.finish_run(total);
    }
}

/// A soak folder: `batch.json` (here or in the parent of a shard) and the `summary.jsonl` of
/// each shard.
fn soak_dir(dir: &Path, total: &mut Coverage, t: &mut Tally, seen: &mut BTreeSet<String>) {
    let batch_path = [dir.join("batch.json"), dir.parent().map(|p| p.join("batch.json")).unwrap_or_default()].into_iter().find(|p| p.exists());
    let Some(batch_path) = batch_path else { return };
    let Ok(batch) = std::fs::read_to_string(&batch_path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string())) else { return };
    let mut summaries = vec![dir.join("summary.jsonl")];
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with("shard-") {
                summaries.push(e.path().join("summary.jsonl"));
            }
        }
    }
    for sp in summaries {
        let Ok(text) = std::fs::read_to_string(&sp) else { continue };
        for line in text.lines() {
            let Ok(s) = serde_json::from_str::<Value>(line) else { continue };
            if s["status"] != "matched" {
                continue;
            }
            let Some(idx) = s["index"].as_u64() else { continue };
            let Some(entry) = batch["runs"].get(idx as usize) else { continue };
            let log = &entry["log"];
            let key = format!("{}/{}/{}", log["seed"], log["ascension"], log["steps"].as_array().map_or(0, |a| a.len()));
            if !seen.insert(key) {
                continue;
            }
            t.soak.1 += 1;
            let mut cov = Coverage::default();
            if let Ok(end) = simulate_saved(log, &mut cov) {
                let want = &s["engine"];
                let same = ["act", "floor", "hp", "max_hp", "gold"].iter().all(|k| end[*k] == want[*k]);
                if same {
                    t.soak.0 += 1;
                    cov.finish_run(total);
                }
            }
        }
    }
}

/// A saved run (`sts-cli run --save`, a soak batch's `log`) played again from its decisions.
fn simulate_saved(log: &Value, obs: &mut Coverage) -> Result<Value, String> {
    let seed = log["seed"].as_i64().ok_or("no seed")?;
    let mut config = RunConfig::new(seed);
    if let Some(n) = log["character"].as_str() {
        config.character = sts_core::character::Character::from_name(n).ok_or(format!("unknown character {n}"))?;
    }
    config.ascension = log["ascension"].as_i64().unwrap_or(0) as i32;
    config.neow_full = log["neow_full"].as_bool().unwrap_or(true);
    config.final_act = log["final_act"].as_bool().unwrap_or(true);
    obs.run_start(&config);
    let mut run = Run::new(config);
    for (i, step) in log["steps"].as_array().ok_or("no steps")?.iter().enumerate() {
        let a: RunAction = serde_json::from_value(step["action"].clone()).map_err(|e| format!("step {i}: {e}"))?;
        obs.run_before(&run, &a);
        run.step(&a).map_err(|e| format!("step {i}: {e}"))?;
        obs.run_after(&run);
    }
    let won = run.outcome == Some(sts_core::run::RunOutcome::Won);
    Ok(json!({"won": won, "act": run.act, "floor": run.floor, "hp": run.hp, "max_hp": run.max_hp, "gold": run.gold}))
}

fn walk(path: &Path, total: &mut Coverage, t: &mut Tally, seen: &mut BTreeSet<String>) {
    if path.is_dir() {
        if path.join("summary.jsonl").exists() || path.join("batch.json").exists() {
            soak_dir(path, total, t, seen);
            return;
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(path).map(|rd| rd.flatten().map(|e| e.path()).collect()).unwrap_or_default();
        files.sort();
        for f in files {
            walk(&f, total, t, seen);
        }
    } else {
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if name.starts_with("trace-") && name.ends_with(".jsonl") {
            fight_file(path, total, t);
        } else if name.starts_with("run-") && name.ends_with(".jsonl") {
            run_file(path, total, t);
        }
    }
}

/// `sts-cli coverage <paths...>`: prints the coverage as JSON.
pub fn coverage_cmd(paths: &[String]) -> Result<(), String> {
    let mut total = Coverage::default();
    let mut t = Tally::default();
    let mut seen = BTreeSet::new();
    for p in paths {
        walk(Path::new(p), &mut total, &mut t, &mut seen);
    }
    let out = json!({
        "sources": {
            "fight_traces_matching": [t.fights.0, t.fights.1],
            "run_traces_matching": [t.runs.0, t.runs.1],
            "soak_runs_still_matching": [t.soak.0, t.soak.1],
        },
        "cards": total.cards,
        "cards_upgraded": total.cards_upgraded,
        "cards_held": total.cards_held,
        "potions": total.potions,
        "powers": total.powers,
        "relics": total.relics,
        "monsters": total.monsters,
        "events": total.events,
    });
    println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
    Ok(())
}
