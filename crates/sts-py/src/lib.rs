//! Python bindings: a batch of games stepped together, in parallel across cores.
//!
//! ```python
//! import sts
//! envs = sts.Envs(64, mode="fight", seed=0)
//! counts = envs.num_legal()                 # legal actions per game
//! status = envs.step([0] * len(envs))       # 0 running, 1 won, 2 lost (then auto-reset)
//! ```
//!
//! Each game keeps its legal actions from the last step, so `step` takes an index into that list
//! (`legal_actions(i)` shows them). A finished game starts a new episode straight away;
//! `last_episode(i)` describes the one that ended.

mod advise;
mod human;
mod probe;
mod practice;
mod scenarios;
mod search;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict};
use rayon::prelude::*;
use serde_json::json;
use sts_core::character::Character;
use sts_core::commands::{combat_commands, run_step, seed_string};
use sts_core::encounters;
use sts_core::rng::StsRandom;
use sts_core::run::map::RoomType;
use sts_core::run::{Run, RunAction, RunConfig, RunOutcome, Screen};
use sts_core::{Combat, CombatConfig, Outcome, PlayerAction};
use sts_encode::agent::{self, fight_slot_action};
use sts_encode::{encode, encode_run};

#[derive(Clone)]
struct Settings {
    fight: bool,
    seed: i64,
    ascension: i32,
    encounters: Vec<String>,
    random_decks: bool,
    final_act: bool,
    n: usize,
    /// A fight past this many turns (or an episode past `max_steps` decisions) ends as a loss,
    /// so a stalemate can't hang a batch.
    max_turns: i32,
    max_steps: u64,
    /// Keep each fight's moves as CommunicationMod commands (and its deck), reported by
    /// `last_episode`, so a fight can be watched in the real game.
    record: bool,
    /// `practice_decks` fights set up as the modded game's `harness` command sets them up
    /// (`practice::harness_fight`), so with `record` they can be played back in the game.
    harness: bool,
    /// With `record`: each combat move also keeps the state it was taken in (`move_state`), and
    /// an episode its relics and room, for analysing play (python/analysis/practice_report.py).
    detail: bool,
    /// Training only (runs): the chance that arriving at a shop opens its card removal at once,
    /// paid as usual, with the network choosing the card. The environment takes that step, not
    /// the network, so the policy's own moves stay its own; it gives the network practice at
    /// removal it never chooses by itself. 0 plays the real game.
    forced_purge: f64,
    /// The same, for a shop visited with a curse in the deck (the larger of the two applies):
    /// practice at the removal that matters most.
    forced_purge_curse: f64,
    /// Training only (runs): the share of games that play practice fights instead of runs (the
    /// first `practice * n` games): a run set up partway through an act with a deck built around
    /// an archetype (block, exhaust, strength), in one of that act's fights, ending with the
    /// fight (see `practice`). A share of games, not of episodes, since a practice fight is about
    /// 30 decisions and a run about 600.
    practice: f64,
    /// Training only (runs): the share of games, after the practice ones, whose runs start with
    /// an archetype's engine already in the deck (`SEEDS`: Corruption+ and Feel No Pain, or
    /// Barricade+, Impervious and Body Slam), so card choices get feedback on building around it.
    seeded: f64,
    /// Practice fights only of this archetype (`practice::ARCHETYPES` name), if set.
    practice_archetype: Option<String>,
    /// Practice fights from `practice::build_deck`, for any character, instead of the Ironclad's
    /// archetype decks: "mix" (a third of each deck kind) or one kind ("grown", "engine",
    /// "random"), if set.
    practice_decks: Option<String>,
    /// `practice_decks` fights only in this act (1-4), if set.
    practice_act: Option<i32>,
    /// Who plays. Practice fights, seeded runs and boss scenarios are built from Ironclad cards
    /// and only exist for the Ironclad so far.
    character: Character,
    /// Practice fights draw their ascension evenly from this range (inclusive), if set, instead of
    /// the runs' ascension: harder fights punish sloppy play.
    practice_ascension_range: Option<(i32, i32)>,
    /// The chance a practice fight is the act's boss, if set (otherwise 15%).
    practice_boss: Option<f64>,
    /// `practice_decks` fights' acts and bosses weighted (`practice::Focus`), if set.
    practice_focus: Option<practice::Focus>,
    /// Seeded runs only from these of the character's seeds (by name), if set.
    seed_names: Option<Vec<String>>,
    /// Seeded runs start with these cards instead of a named seed (`Envs::set_seed_cards`), if set: what a card is worth to a network.
    seed_cards: Option<Vec<(sts_core::cards::CardId, bool)>>,
    /// Runs start holding the three keys (`Envs::set_free_keys`): an evaluation of what a network would do at the Heart, keys' cost aside.
    free_keys: bool,
    /// Runs leave the keys to the hand-written rule (`RunConfig::key_rule`, `Envs::set_key_rule`).
    key_rule: Option<sts_core::run::KeyRule>,
    /// The act a seeded run gets its cards in, on its first map screen there (`Envs::set_seed_act`; 1: from the start).
    seed_act: i32,
    /// Boss fights from winning human runs (`scenarios`, `Envs::set_scenarios`), and the share of
    /// practice fights that play one of them instead of an archetype deck. Picked at random, or
    /// with `scenario_sequential` in turn (episode by episode, game by game: evaluation). One with
    /// `play_on` isn't a practice fight at all, but a normal run that starts in that fight.
    scenarios: std::sync::Arc<Vec<scenarios::Scenario>>,
    scenario_share: f64,
    scenario_sequential: bool,
    /// Runs: each episode draws its ascension evenly from this range (inclusive) instead of
    /// `ascension` (practice fights and seeded runs too).
    ascension_range: Option<(i32, i32)>,
    /// With `ascension_range`, this share of runs draws its ascension evenly from 0 to the range's
    /// LO - 1 instead: keeps play below the range from drifting while training aims at it.
    ascension_low_share: f64,
    /// Training only (practice fights): a lost practice fight is played again, the same fight (its
    /// seed and scenario: the same deck, relics, potions, HP and encounter), until a replay wins or
    /// it has been played again this many times (0: never; jack, 2026-10-30: "can we replay each
    /// lost fight up to 8 times while practising and then train off the winning ones?"). With
    /// `retry_reseed` each replay draws otherwise (the fight's own streams reseeded: `fight_seed`;
    /// jack: "yeah sure relay the fights with a differnt seed"); without, the same draws and monster
    /// rolls while the moves agree (6% of replays won: the network played much the same moves).
    retry_losses: u32,
    retry_reseed: bool,
    /// Training only (runs): human boss states (`Envs::set_late_starts`) that `late_share` of the games neither practice fights
    /// nor seeded start every episode from. A share of games, like `practice`: a late start is shorter than a run.
    late_starts: std::sync::Arc<Vec<scenarios::Scenario>>,
    late_share: f64,
}

/// Moves in one combat turn past which ending the turn is the only legal move (2026-10-24: an
/// infinite combo against the Heart, whose Invincible caps the damage it takes a turn, played
/// 4,000 cards at it on 1 HP and held up a whole eval batch). A turn is rarely past 20.
const MAX_TURN_MOVES: u32 = 100;

/// The engines seeded runs start with: (name, cards as (id, upgraded)).
const SEEDS: [(&str, &[(sts_core::cards::CardId, bool)]); 3] = [
    ("exhaust", &[(sts_core::cards::CardId::Corruption, true), (sts_core::cards::CardId::FeelNoPain, false)]),
    ("block", &[(sts_core::cards::CardId::Barricade, true), (sts_core::cards::CardId::Impervious, false), (sts_core::cards::CardId::BodySlam, false)]),
    ("self-damage", &[(sts_core::cards::CardId::Rupture, true), (sts_core::cards::CardId::Bloodletting, false), (sts_core::cards::CardId::Combust, false)]),
];

/// The Silent's seeds: each engine of `practice::SILENT_ARCHETYPES` started with its key cards, and
/// `CONTROL_SEED`, two good cards that are no engine: a seed is free cards, so an engine's runs are
/// read against the control's (`python/analysis/seed_rates.py`).
const SILENT_SEEDS: [(&str, &[(sts_core::cards::CardId, bool)]); 5] = {
    use sts_core::cards::CardId::*;
    [
        ("poison", &[(NoxiousFumes, true), (Catalyst, false)]),
        ("shiv", &[(InfiniteBlades, true), (Accuracy, false), (BladeDance, false)]),
        ("discard", &[(Acrobatics, true), (Tactician, false), (Eviscerate, false)]),
        ("dex", &[(Footwork, true), (LegSweep, false), (Backflip, false)]),
        (CONTROL_SEED, &[(Dash, true), (Predator, false)]),
    ]
};

/// A yardstick, not an engine to train on: seeded runs draw it only when `set_seeds` names it.
const CONTROL_SEED: &str = "control";

/// The Watcher's seeds (jack, 2026-10-28): the stance engine's two powers, which the run network
/// took far less often than human winners (Mental Fortress 5% of offers against 46%; 75% after
/// the stance seeds) and which pay from the first fight with the starter's Eruption and Vigilance;
/// and the Mantra engine's scaling, Deva Form+ (not ethereal) and Devotion+ (Divinity every fourth
/// turn), which it never takes (0% of offers against human winners' 51% and 22%).
const WATCHER_SEEDS: [(&str, &[(sts_core::cards::CardId, bool)]); 2] = {
    use sts_core::cards::CardId::*;
    [("stance", &[(MentalFortress, true), (Rushdown, false)]), ("mantra", &[(DevaForm, true), (Devotion, true)])]
};

/// Puts a seed's cards in a run's deck.
fn give_cards(r: &mut Run, cards: &[(sts_core::cards::CardId, bool)]) {
    for &(id, up) in cards {
        let mut c = r.new_card(id);
        if up {
            c.upgrade();
        }
        r.deck.push(c);
    }
}

/// The name a run seeded with `Envs::set_seed_cards` carries.
const CUSTOM_SEED: &str = "cards";

/// The name a late start (`Envs::set_late_starts`) carries where a seeded run carries its seed's.
const LATE_START: &str = "late";

/// A character's seeds (none yet for the Defect).
fn seeds_for(c: Character) -> &'static [(&'static str, &'static [(sts_core::cards::CardId, bool)])] {
    match c {
        Character::Ironclad => &SEEDS,
        Character::Silent => &SILENT_SEEDS,
        Character::Watcher => &WATCHER_SEEDS,
        Character::Defect => &[],
    }
}

enum Game {
    Fight(Box<Combat>, Vec<PlayerAction>),
    Run(Box<Run>, Vec<RunAction>),
}

impl Game {
    fn num_legal(&self) -> usize {
        match self {
            Game::Fight(_, l) => l.len(),
            Game::Run(_, l) => l.len(),
        }
    }

    fn is_over(&self) -> bool {
        match self {
            Game::Fight(c, _) => c.is_over(),
            Game::Run(r, _) => r.is_over(),
        }
    }
}

struct Slot {
    index: usize,
    episode: u64,
    seed: i64,
    steps: u64,
    game: Game,
    last: Option<serde_json::Value>,
    /// Options toggled so far in a pending card selection (slot actions only).
    picked: Vec<usize>,
    /// The fight's encounter (empty for runs).
    encounter: String,
    /// With `record`: the fight's starting deck and the commands of each move so far.
    deck: Vec<(String, i32)>,
    moves: Vec<serde_json::Value>,
    /// With `harness`: the game's harness commands that set the fight up.
    harness: Option<Vec<serde_json::Value>>,
    /// A Smoke Bomb was drunk: the fight ends in an escape, which a practice fight counts as a
    /// loss (jack, 2026-10-24: escaping had scored as a win at full HP); the summary's `hp_escaped` keeps the HP it
    /// escaped with, for train_runs.practice_reward's survival credit (2026-11-05).
    smoked: bool,
    /// A practice fight's HP when it started (0 otherwise), for the HP it cost.
    start_hp: i32,
    /// The combat turn being played ((floor, turn); a fight's floor is 0) and the moves made in it
    /// so far, for `MAX_TURN_MOVES`.
    turn_at: (i32, i32),
    turn_moves: u32,
    /// Runs: card rewards skipped on the current reward screen (see `encode_run::options`).
    skipped: Vec<Vec<(i32, i32)>>,
    /// Runs with `forced_purge`: rolls for it (apart from the game's own streams), and the floor
    /// of the last shop rolled for, so a visit rolls once.
    explore: StsRandom,
    shop_floor: i32,
    /// A practice fight's archetype (runs with `practice`), or None for a run.
    practice: Option<&'static str>,
    /// A boss scenario's index in `Settings::scenarios`, and the floor its last boss is on.
    scenario: Option<(usize, i32)>,
    /// A seeded run's engine (runs with `seeded`), or `LATE_START`.
    seeded: Option<&'static str>,
    /// Its cards, until `Settings::seed_act` comes.
    late_seed: Vec<(sts_core::cards::CardId, bool)>,
    /// Seeded runs: cards of the engine's archetype the last step added to the deck (counted
    /// towards `SYNERGY_CAP`), and how many this run has added so far.
    synergy: u32,
    synergy_total: u32,
    /// What the last step did that training shapes rewards on (`Envs::habits`).
    habit: Habit,
    /// A practice fight's replay count (`retry_losses`): 0 for a fight played the first time.
    retry: u32,
}

/// What one run step did, for reward shaping (training only; see train_runs.py): cards added to
/// and removed from the deck and its size after, the HP share a rest was taken at, a Recall taken
/// without the Sapphire key (the three keys are needed for Act 4, and the network never takes
/// that one), an elite fight won, a card upgraded at a rest site, a potion drunk in a fight that
/// isn't an elite or a boss, a shop left without the card removal it offered and could pay for,
/// and basic Strikes upgraded on an upgrade grid (a rest site's or an event's).
#[derive(Clone, Copy, Default)]
struct Habit {
    added: u32,
    removed: u32,
    deck: u32,
    rest_hp: Option<f32>,
    recall_wasted: bool,
    elite_won: bool,
    smithed: bool,
    hallway_potion: bool,
    removal_skipped: bool,
    strikes_upgraded: u32,
    /// The cards the step added that belong to the deck's leading archetype (`archetype_counts`; a core card of any
    /// archetype if the deck had none), the cards it added that don't, and whether that archetype was already complete.
    arch_in: u32,
    arch_off: u32,
    arch_complete: bool,
}

/// The deck's leading archetype (of `practice::archetypes_for` its character's): of those with a core card in `deck`, the one
/// with the most of its cards (a core card counting twice); and whether it is complete (two core cards and four support).
fn leading_archetype(c: sts_core::character::Character, deck: &[sts_core::cards::CardId]) -> Option<(&'static practice::Archetype, bool)> {
    practice::archetypes_for(c).iter()
        .map(|a| (a, deck.iter().filter(|id| a.core.contains(id)).count(), deck.iter().filter(|id| a.support.contains(id)).count()))
        .filter(|&(_, core, _)| core > 0)
        .max_by_key(|&(_, core, support)| 2 * core + support)
        .map(|(a, core, support)| (a, core >= 2 && support >= 4))
}

/// The run before a step, as `Habit::of` compares it.
struct Before {
    /// The deck's card ids (empty in a fight, which doesn't change the deck).
    deck: Vec<sts_core::cards::CardId>,
    combat: bool,
    room: Option<RoomType>,
    hp: i32,
    max_hp: i32,
    sapphire: bool,
    smith_grid: bool,
    /// A grid that upgrades the card picked (a rest site's or an event's).
    upgrade_grid: bool,
    /// The shop's card removal is on offer and affordable (a legal move).
    removal_open: bool,
}

impl Habit {
    const FIELDS: usize = 13;

    /// Compares the run before and after `a`.
    fn of(b: &Before, a: &RunAction, r: &Run) -> Habit {
        let elite_won = b.combat && b.room == Some(RoomType::Elite) && !matches!(r.screen, Screen::Combat) && r.outcome != Some(RunOutcome::Died);
        if b.combat {
            // A fight doesn't change the deck.
            let potion = matches!(a, RunAction::Combat { action: sts_core::PlayerAction::UsePotion { .. } });
            let hallway = !matches!(b.room, Some(RoomType::Elite) | Some(RoomType::Boss));
            return Habit { deck: r.deck.len() as u32, elite_won, hallway_potion: potion && hallway, ..Habit::default() };
        }
        let mut left: std::collections::HashMap<sts_core::cards::CardId, i32> = std::collections::HashMap::new();
        for &id in &b.deck {
            *left.entry(id).or_default() += 1;
        }
        let mut added = 0;
        let mut new_cards = Vec::new();
        for c in &r.deck {
            match left.get_mut(&c.id) {
                Some(n) if *n > 0 => *n -= 1,
                _ => {
                    added += 1;
                    new_cards.push(c.id);
                }
            }
        }
        let leading = leading_archetype(r.config.character, &b.deck);
        let fits = |id: &sts_core::cards::CardId| match leading {
            Some((a, _)) => a.core.contains(id) || a.support.contains(id),
            None => practice::archetypes_for(r.config.character).iter().any(|a| a.core.contains(id)),
        };
        let arch_in = new_cards.iter().filter(|id| fits(id)).count() as u32;
        let removed = left.values().map(|&n| n.max(0) as u32).sum();
        Habit {
            added,
            removed,
            deck: r.deck.len() as u32,
            rest_hp: matches!(a, RunAction::Rest).then(|| b.hp as f32 / b.max_hp.max(1) as f32),
            recall_wasted: matches!(a, RunAction::Recall) && !b.sapphire,
            elite_won,
            smithed: b.smith_grid && matches!(a, RunAction::Select { .. }),
            hallway_potion: false,
            removal_skipped: b.removal_open && matches!(a, RunAction::Proceed),
            arch_in,
            arch_off: added - arch_in,
            arch_complete: leading.is_some_and(|(_, complete)| complete),
            strikes_upgraded: match a {
                RunAction::Select { indices } if b.upgrade_grid => {
                    indices.iter().filter(|&&i| b.deck.get(i).is_some_and(|id| id.data().name == "Strike")).count() as u32
                }
                _ => 0,
            },
        }
    }

    fn row(&self) -> [f32; Habit::FIELDS] {
        let f = |b: bool| b as u32 as f32;
        [self.added as f32, self.removed as f32, self.deck as f32, self.rest_hp.unwrap_or(-1.0), f(self.recall_wasted), f(self.elite_won), f(self.smithed),
         f(self.hallway_potion), f(self.removal_skipped), self.strikes_upgraded as f32, self.arch_in as f32, self.arch_off as f32,
         f(self.arch_complete)]
    }
}

/// The run before a step, for `Habit::of` (`legal`: its legal moves).
fn habit_before(r: &Run, legal: &[RunAction]) -> Before {
    let combat = matches!(r.screen, Screen::Combat);
    Before {
        deck: if combat { Vec::new() } else { r.deck.iter().map(|c| c.id).collect() },
        combat,
        room: r.room,
        hp: r.hp,
        max_hp: r.max_hp,
        sapphire: r.keys.sapphire,
        smith_grid: matches!(&r.screen, Screen::Grid { purpose: sts_core::run::GridPurpose::Smith, .. }),
        upgrade_grid: matches!(&r.screen, Screen::Grid { purpose: sts_core::run::GridPurpose::Smith | sts_core::run::GridPurpose::Upgrade, .. }),
        removal_open: matches!(r.screen, Screen::Shop { .. }) && legal.contains(&RunAction::Purge),
    }
}

/// The most archetype cards a seeded run is credited with (`Envs::synergy_picks`), so the
/// reward for them can't grow by bloating the deck.
const SYNERGY_CAP: u32 = 6;

/// Cards in the deck that belong to archetype `name` (its core or support in
/// `practice::ARCHETYPES`).
fn synergy_cards(r: &Run, name: &str) -> u32 {
    let Some(arch) = practice::archetypes_for(r.config.character).iter().find(|a| a.name == name) else { return 0 };
    r.deck.iter().filter(|c| arch.core.contains(&c.id) || arch.support.contains(&c.id)).count() as u32
}

impl Slot {
    fn new(s: &Settings, index: usize, episode: u64) -> Result<Slot, String> {
        Slot::build(s, index, episode, None)
    }

    /// Game `index`'s episode `episode`; with `again`, a lost practice fight played again: its seed,
    /// its boss scenario's index (if it was one) and the replay's count, so the same fight is set up
    /// (with that replay's own draws if `retry_reseed`: `fight_seed`).
    fn build(s: &Settings, index: usize, episode: u64, again: Option<(i64, Option<usize>, u32)>) -> Result<Slot, String> {
        let seed = again.map_or_else(|| s.seed.wrapping_add((episode * s.n as u64 + index as u64) as i64), |(seed, _, _)| seed);
        let fight_seed = again.filter(|_| s.retry_reseed).map(|(seed, _, retry)| fight_seed(seed, retry));
        let mut encounter = String::new();
        let mut deck = Vec::new();
        let mut practice = None;
        let mut scenario = None;
        let mut seeded = None;
        let mut late_seed = Vec::new();
        let mut harness = None;
        let game = if s.fight {
            let mut picker = StsRandom::new(seed ^ 0x5DEE_CE66);
            let enc = &s.encounters[picker.random(s.encounters.len() as i32 - 1) as usize];
            encounter = enc.clone();
            let mut cfg = CombatConfig::new(s.character, seed, enc);
            cfg.ascension = s.ascension;
            if s.random_decks {
                let n = 8 + picker.random(14);
                let pool = s.character.library_order();
                cfg.deck = (0..n)
                    .map(|_| {
                        let id = pool[picker.random(pool.len() as i32 - 1) as usize];
                        (id, if picker.random_boolean() { 1 } else { 0 })
                    })
                    .collect();
            }
            if s.record {
                deck = cfg.deck.iter().map(|&(id, up)| (id.data().java_id.to_string(), up)).collect();
            }
            let c = Combat::new(&cfg)?;
            let legal = c.legal_actions();
            Game::Fight(Box::new(c), legal)
        } else {
            let mut roll = StsRandom::new(seed ^ 0x5052_4143_5449_4345); // "PRACTICE"
            let ascension = match s.ascension_range {
                Some((lo, _)) if lo > 0 && s.ascension_low_share > 0.0
                    && (StsRandom::new(seed ^ 0x4C4F_5741_5343_454E).random_float() as f64) < s.ascension_low_share => // "LOWASCEN"
                {
                    StsRandom::new(seed ^ 0x4153_4345_4E53_494F).random(lo - 1)
                }
                Some((lo, hi)) => lo + StsRandom::new(seed ^ 0x4153_4345_4E53_494F).random(hi - lo), // "ASCENSIO"
                None => s.ascension,
            };
            // Archetype practice fights and seeded runs are the Ironclad's only so far; another
            // character's practice fights must all be its own human boss scenarios. Until they are
            // loaded (`set_scenarios`, which starts every game again), its practice games play runs.
            let in_practice = (index as f64) < s.practice * s.n as f64;
            let is_seeded = !in_practice && (index as f64) < (s.practice + s.seeded) * s.n as f64;
            // The first `late_share` of the games left.
            let taken = (s.practice + s.seeded).min(1.0);
            let is_late = !in_practice && !is_seeded && !s.late_starts.is_empty() && (index as f64) < (taken + s.late_share * (1.0 - taken)) * s.n as f64;
            let ironclad = s.character == Character::Ironclad;
            let decks = s.practice_decks.is_some();
            if !ironclad && ((is_seeded && seeds_for(s.character).is_empty()) || (!decks && in_practice && !s.scenarios.is_empty() && s.scenario_share < 1.0)) {
                return Err(format!(
                    "practice fights other than boss scenarios (--scenario-share 1) need practice_decks, and seeded runs need a character with seeds, for {:?}",
                    s.character
                ));
            }
            if in_practice && (ironclad || decks || !s.scenarios.is_empty()) {
                let ascension = match s.practice_ascension_range {
                    Some((lo, hi)) => lo + StsRandom::new(seed ^ 0x5052_4143_4153_434E).random(hi - lo), // "PRACASCN"
                    None => ascension,
                };
                let n = s.scenarios.len();
                let from_scenario = n > 0 && (StsRandom::new(seed ^ 0x5343_454E_4152_494F).random_float() as f64) < s.scenario_share; // "SCENARIO"
                let mut built = None;
                // A boss the engine can't set up falls back to another scenario (a few tries), then to
                // an archetype fight (the Ironclad's only).
                for attempt in 0..if from_scenario { 8 } else { 0 } {
                    let i = match again {
                        Some((_, Some(i), _)) if attempt == 0 => i,
                        _ if s.scenario_sequential && attempt == 0 => ((episode * s.n as u64 + index as u64) % n as u64) as usize,
                        _ => roll.random(n as i32 - 1) as usize,
                    };
                    let sc = &s.scenarios[i];
                    // (A scenario's setup is all in its line: a replay with other draws builds it on its own seed.)
                    let mut r = if sc.play_on { sc.build_play_on(seed, s.final_act) } else { sc.build(fight_seed.or(sc.seed).unwrap_or(seed)) };
                    if sc.play_on && s.free_keys {
                        r.keys = sts_core::run::Keys { ruby: true, emerald: true, sapphire: true };
                    }
                    if sc.play_on {
                        r.config.key_rule = s.key_rule;
                    }
                    if matches!(r.screen, Screen::Combat) {
                        scenario = Some((i, sc.end_floor()));
                        built = Some((r, sc.kind(), sc.bosses.join(" + ")));
                        break;
                    }
                    if s.scenario_sequential {
                        break;
                    }
                }
                if built.is_none() && !ironclad && !decks {
                    return Err(format!("no boss scenario could be set up ({:?})", s.character));
                }
                let (mut r, arch, key) = match (built, &s.practice_decks) {
                    (Some(b), _) => b,
                    (None, Some(d)) => {
                        let kind = practice::DeckKind::from_name(d);
                        practice::build_deck_with(s.character, seed, ascension, &mut roll, kind, s.practice_act, s.practice_boss, s.practice_focus.as_ref(), fight_seed)?
                    }
                    (None, None) => practice::build(seed, ascension, &mut roll, s.practice_archetype.as_deref(), s.practice_boss)?,
                };
                if s.harness && s.practice_decks.is_some() && scenario.is_none() {
                    let (h, commands) = practice::harness_fight(&r, &key)?;
                    r = h;
                    harness = Some(commands);
                }
                practice = if scenario.is_some_and(|(i, _)| s.scenarios[i].play_on) { None } else { Some(arch) };
                encounter = key;
                let legal = r.legal_actions();
                Game::Run(Box::new(r), legal)
            } else {
                // A fight the engine can't set up falls back to another line (a few tries), then to a plain run, as a scenario's does.
                let late = (0..if is_late { 8 } else { 0 }).find_map(|_| {
                    let r = s.late_starts[roll.random(s.late_starts.len() as i32 - 1) as usize].build_play_on(seed, s.final_act);
                    matches!(r.screen, Screen::Combat).then_some(r)
                });
                seeded = late.as_ref().map(|_| LATE_START);
                let mut r = late.unwrap_or_else(|| {
                    let mut cfg = RunConfig::new(seed);
                    cfg.ascension = ascension;
                    cfg.final_act = s.final_act;
                    cfg.character = s.character;
                    Run::new(cfg)
                });
                if s.free_keys {
                    r.keys = sts_core::run::Keys { ruby: true, emerald: true, sapphire: true };
                }
                r.config.key_rule = s.key_rule;
                if is_seeded {
                    let seeds: Vec<_> = seeds_for(s.character)
                        .iter()
                        .copied()
                        .filter(|(n, _)| s.seed_names.as_ref().map_or(*n != CONTROL_SEED, |names| names.iter().any(|x| x == n)))
                        .collect();
                    let (name, cards) = match &s.seed_cards {
                        Some(cards) => (CUSTOM_SEED, cards.as_slice()),
                        None => seeds[roll.random(seeds.len() as i32 - 1) as usize],
                    };
                    if s.seed_act > 1 {
                        late_seed = cards.to_vec();
                    } else {
                        give_cards(&mut r, cards);
                    }
                    seeded = Some(name);
                }
                let legal = r.legal_actions();
                Game::Run(Box::new(r), legal)
            }
        };
        let explore = StsRandom::new(seed ^ 0x2545_F491_4F6C_DD1D);
        let start_hp = match &game {
            Game::Run(r, _) if practice.is_some() => r.combat.as_ref().map_or(r.hp, |c| c.player.hp),
            _ => 0,
        };
        Ok(Slot { index, episode, seed, steps: 0, game, last: None, picked: Vec::new(), encounter, deck, moves: Vec::new(), harness, smoked: false, start_hp, turn_at: (-1, -1), turn_moves: 0, skipped: Vec::new(), explore, shop_floor: -1, practice, scenario, seeded, late_seed, synergy: 0, synergy_total: 0, habit: Habit::default(), retry: 0 })
    }

    /// A seeded run whose cards wait for an act: gives them on its first map screen there (after the step's `count_synergy`: a gift isn't credited as a pick).
    fn give_late_seed(&mut self, s: &Settings) {
        if self.late_seed.is_empty() {
            return;
        }
        if let Game::Run(r, _) = &mut self.game {
            if r.act >= s.seed_act && matches!(r.screen, Screen::Map) {
                give_cards(r, &self.late_seed);
                self.late_seed.clear();
            }
        }
    }

    /// After a run step of a seeded run: credits the archetype cards it added to the deck (any
    /// way: card rewards, shops, events), up to `SYNERGY_CAP` a run. `before` is the count before
    /// the step.
    fn count_synergy(&mut self, before: Option<u32>) {
        let (Some(before), Some(name), Game::Run(r, _)) = (before, self.seeded, &self.game) else { return };
        let gained = synergy_cards(r, name).saturating_sub(before).min(SYNERGY_CAP.saturating_sub(self.synergy_total));
        self.synergy = gained;
        self.synergy_total += gained;
    }

    /// After a step: counts the moves of the combat turn and, past `MAX_TURN_MOVES`, leaves only
    /// ending the turn legal.
    fn cap_turn(&mut self) {
        let turn = match &self.game {
            Game::Fight(c, _) => Some((0, c.turn)),
            Game::Run(r, _) => r.combat.as_ref().filter(|_| matches!(r.screen, Screen::Combat)).map(|c| (r.floor, c.turn)),
        };
        let Some(turn) = turn else {
            self.turn_moves = 0;
            return;
        };
        if turn == self.turn_at {
            self.turn_moves += 1;
        } else {
            self.turn_at = turn;
            self.turn_moves = 1;
        }
        if self.turn_moves < MAX_TURN_MOVES {
            return;
        }
        match &mut self.game {
            Game::Fight(_, legal) => {
                if legal.contains(&PlayerAction::EndTurn) {
                    *legal = vec![PlayerAction::EndTurn];
                }
            }
            Game::Run(_, legal) => {
                let end = RunAction::Combat { action: PlayerAction::EndTurn };
                if legal.contains(&end) {
                    *legal = vec![end];
                }
            }
        }
    }

    /// After a run step: on arriving at a shop, `forced_purge` may open its card removal.
    fn maybe_force_purge(&mut self, s: &Settings) -> Result<(), String> {
        let Game::Run(r, legal) = &mut self.game else { return Ok(()) };
        if (s.forced_purge <= 0.0 && s.forced_purge_curse <= 0.0) || !matches!(r.screen, Screen::Shop { .. }) || r.floor == self.shop_floor {
            return Ok(());
        }
        self.shop_floor = r.floor;
        let has_curse = r.deck.iter().any(|c| c.card_type() == sts_core::cards::CardType::Curse);
        let chance = if has_curse { s.forced_purge.max(s.forced_purge_curse) } else { s.forced_purge };
        if (self.explore.random_float() as f64) < chance && legal.contains(&RunAction::Purge) {
            if s.record {
                self.moves.push(record_run_step(r, &RunAction::Purge));
            }
            r.step(&RunAction::Purge)?;
            *legal = r.legal_actions();
        }
        Ok(())
    }

    /// Takes legal action `k`; returns 0 while running, 1 for a win, 2 for a loss. A finished
    /// game is replaced by the next episode.
    fn step(&mut self, s: &Settings, k: usize) -> Result<u32, String> {
        self.synergy = 0;
        self.habit = Habit::default();
        match &mut self.game {
            Game::Fight(c, legal) => {
                let a = legal.get(k).ok_or_else(|| format!("game {}: action {k} out of range ({} legal)", self.index, legal.len()))?.clone();
                if s.record {
                    self.moves.push(record_move(c, &a));
                }
                c.step(&a)?;
                *legal = if c.is_over() { Vec::new() } else { c.legal_actions() };
                self.picked.clear();
            }
            Game::Run(r, legal) => {
                let a = legal.get(k).ok_or_else(|| format!("game {}: action {k} out of range ({} legal)", self.index, legal.len()))?.clone();
                if s.record {
                    self.moves.push(record_step(self.harness.is_some(), s.detail, r, &a));
                }
                self.smoked |= smoke_bomb(r, &a);
                let before = self.seeded.map(|n| synergy_cards(r, n));
                let habit = habit_before(r, legal);
                r.step(&a)?;
                self.habit = Habit::of(&habit, &a, r);
                *legal = if r.is_over() { Vec::new() } else { r.legal_actions() };
                self.count_synergy(before);
                self.give_late_seed(s);
            }
        }
        self.cap_turn();
        self.maybe_force_purge(s)?;
        self.finish_step(s)
    }

    /// Takes fixed action slot `a` (see `encode`). A card selection is made one pick at a
    /// time; it is sent to the engine when it is full or confirmed.
    /// Takes fixed action slot `a` (see `encode` and `encode_run`). A card selection is made
    /// one pick at a time; it is sent to the engine when it is full or confirmed.
    fn step_slot(&mut self, s: &Settings, a: usize) -> Result<u32, String> {
        self.synergy = 0;
        self.habit = Habit::default();
        let index = self.index;
        let bad = |e: String| format!("game {index}: {e}");
        match &mut self.game {
            Game::Fight(c, legal) => {
                let Some(action) = fight_slot_action(c, legal, &mut self.picked, a).map_err(bad)? else {
                    self.steps += 1;
                    return Ok(0);
                };
                if s.record {
                    self.moves.push(record_move(c, &action));
                }
                c.step(&action)?;
                *legal = if c.is_over() { Vec::new() } else { c.legal_actions() };
            }
            Game::Run(r, legal) => {
                let action = agent::slot_action(r, legal, &mut self.picked, &self.skipped, a).map_err(bad)?;
                let Some(action) = action else {
                    self.steps += 1;
                    return Ok(0);
                };
                if s.record {
                    self.moves.push(record_step(self.harness.is_some(), s.detail, r, &action));
                }
                self.smoked |= smoke_bomb(r, &action);
                agent::before_step(&mut self.skipped, r, &action);
                let before = self.seeded.map(|n| synergy_cards(r, n));
                let habit = habit_before(r, legal);
                r.step(&action)?;
                self.habit = Habit::of(&habit, &action, r);
                agent::after_step(&mut self.skipped, r);
                *legal = if r.is_over() { Vec::new() } else { r.legal_actions() };
                self.picked.clear();
                self.count_synergy(before);
                self.give_late_seed(s);
            }
        }
        self.cap_turn();
        self.maybe_force_purge(s)?;
        self.finish_step(s)
    }

    fn finish_step(&mut self, s: &Settings) -> Result<u32, String> {
        self.steps += 1;
        let timed_out = self.steps >= s.max_steps
            || match &self.game {
                Game::Fight(c, _) => c.turn > s.max_turns,
                Game::Run(r, _) => r.combat.as_ref().is_some_and(|c| matches!(r.screen, Screen::Combat) && c.turn > s.max_turns),
            };
        // A practice fight is over when its fight is; a boss scenario when its last boss's is.
        let practice_over = match (&self.game, self.scenario) {
            (Game::Run(r, _), Some((_, end_floor))) if self.practice.is_some() => scenarios::over(r, end_floor),
            (Game::Run(r, _), None) => self.practice.is_some() && !matches!(r.screen, Screen::Combat),
            _ => false,
        };
        if !self.game.is_over() && !timed_out && !practice_over {
            return Ok(0);
        }
        let (status, summary) = match &self.game {
            Game::Fight(c, _) => {
                let won = c.outcome == Some(Outcome::Victory);
                let v = json!({"won": won, "hp": c.player.hp, "max_hp": c.player.max_hp, "turns": c.turn, "timed_out": timed_out});
                (if won { 1 } else { 2 }, v)
            }
            Game::Run(r, _) if self.practice.is_some() => {
                // Won if the fight ended without the player dying (and not by the turn limit or a
                // Smoke Bomb's escape).
                let won = practice_over && r.outcome != Some(RunOutcome::Died) && !self.smoked;
                let mut v = json!({"practice": self.practice, "won": won, "act": r.act, "floor": r.floor, "hp": if won { r.hp } else { 0 },
                               "max_hp": r.max_hp, "hp_start": self.start_hp, "room": r.room.map(|x| format!("{x:?}")), "timed_out": timed_out, "escaped": self.smoked,
                               "hp_escaped": if self.smoked { r.hp } else { 0 }, "potions": r.potions.iter().filter(|p| p.is_some()).count()});
                if let Some((i, end_floor)) = self.scenario {
                    // Bosses beaten: all if won, else those on floors before the one it ended on.
                    let first = s.scenarios[i].floor;
                    v["scenario"] = json!(i);
                    v["bosses_beaten"] = json!(if won { end_floor - first + 1 } else { (r.floor - first).max(0) });
                }
                (if won { 1 } else { 2 }, v)
            }
            Game::Run(r, _) => {
                let won = r.outcome == Some(RunOutcome::Won);
                let keys = r.keys.ruby as u8 + r.keys.emerald as u8 + r.keys.sapphire as u8;
                let v = json!({"won": won, "act": r.act, "floor": r.floor, "hp": r.hp, "max_hp": r.max_hp, "gold": r.gold, "timed_out": timed_out, "keys": keys,
                               "seeded": self.seeded, "synergy_picks": self.synergy_total, "ascension": r.config.ascension,
                               "died_in": if r.outcome == Some(RunOutcome::Died) { r.encounter.clone() } else { None },
                               "potions": r.potions.iter().filter(|p| p.is_some()).count(),
                               "archetype_cards": practice::archetypes_for(r.config.character).iter().map(|a| (a.name.to_string(), json!(synergy_cards(r, a.name)))).collect::<serde_json::Map<_, _>>()});
                (if won { 1 } else { 2 }, v)
            }
        };
        let mut summary = summary;
        if self.practice.is_some() {
            summary["retry"] = json!(self.retry);
        } else if let Some((i, _)) = self.scenario {
            summary["scenario"] = json!(i); // a run that played on from it
        }
        summary["seed"] = json!(self.seed);
        if !self.encounter.is_empty() {
            summary["encounter"] = json!(self.encounter);
        }
        if s.record && s.fight {
            summary["seed_string"] = json!(seed_string(self.seed));
            summary["deck"] = json!(self.deck);
            summary["moves"] = json!(std::mem::take(&mut self.moves));
        }
        if let (true, Game::Run(r, _)) = (s.record, &self.game) {
            summary["deck"] = json!(r.deck.iter().map(|c| c.name().to_string()).collect::<Vec<_>>());
            if s.detail {
                summary["relics"] = json!(r.relics.iter().map(|x| x.id.java_id()).collect::<Vec<_>>());
            }
        }
        if let (true, Some(commands)) = (s.record, &self.harness) {
            // In tools/trace/playback_fights.py's format: the game's start and harness commands,
            // then the fight's moves.
            if let Game::Run(r, _) = &self.game {
                summary["character"] = json!(format!("{:?}", r.config.character).to_lowercase());
                summary["ascension"] = json!(r.config.ascension);
            }
            summary["seed_string"] = json!(seed_string(self.seed));
            summary["harness"] = json!(commands);
            summary["moves"] = json!(std::mem::take(&mut self.moves));
        } else if s.record && !s.fight {
            // In tools/trace/playback.py's format (like `sts-cli run --save`).
            summary["log"] = json!({
                "seed": self.seed,
                "seed_string": seed_string(self.seed),
                "ascension": match &self.game { Game::Run(r, _) => r.config.ascension, _ => s.ascension },
                "character": format!("{:?}", s.character).to_lowercase(),
                "neow_full": true,
                "steps": std::mem::take(&mut self.moves),
            });
        }
        summary["steps"] = json!(self.steps);
        // A lost practice fight is played again (`retry_losses`) until a replay wins.
        let next = if self.practice.is_some() && status == 2 && self.retry < s.retry_losses {
            Slot { retry: self.retry + 1, ..Slot::build(s, self.index, self.episode + 1, Some((self.seed, self.scenario.map(|(i, _)| i), self.retry + 1)))? }
        } else {
            Slot::new(s, self.index, self.episode + 1)?
        };
        // The last step's synergy is still read after it (`Envs::synergy_picks`).
        *self = Slot { last: Some(summary), synergy: self.synergy, ..next };
        Ok(status)
    }
}

/// Probe fights' first rows (`probe`), encoded in parallel into `ints` and `floats` (room for at least
/// `probes.len()` rows; the rest untouched): their meta as (source, option, fight, draw).
fn write_probes(py: Python<'_>, probes: &[(usize, probe::Probe, Run)], ints: &mut [i32], floats: &mut [f32]) -> Vec<(usize, i32, u8, u32)> {
    let (ni, nf) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len());
    let n = probes.len();
    py.detach(|| {
        ints[..n * ni].par_chunks_mut(ni).zip(floats[..n * nf].par_chunks_mut(nf)).zip(probes.par_iter()).for_each(|((i, f), (_, _, r))| {
            i.fill(0);
            f.fill(0.0);
            let mut mask = vec![0u8; encode::ACTIONS];
            encode_run::encode_run(r, &r.legal_actions(), &[], &[], i, f, &mut mask);
        });
    });
    probes.iter().map(|(s, p, _)| (*s, p.option, p.fight, p.draw)).collect()
}

/// `write_probes` with only the first `keep.0` ints and `keep.1` floats of each row kept (rows that long in
/// `ints` and `floats`): what the fight-outcome head reads (the fight's fields and the run token's), so the lookup
/// writes and moves about 40% of each row. Each row is encoded into a scratch row first (one per thread).
fn write_probes_prefix(py: Python<'_>, probes: &[(usize, probe::Probe, Run)], keep: (usize, usize), ints: &mut [i32], floats: &mut [f32])
    -> Vec<(usize, i32, u8, u32)> {
    let (ni, nf) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len());
    let (ki, kf, n) = (keep.0.min(ni), keep.1.min(nf), probes.len());
    py.detach(|| {
        ints[..n * ki].par_chunks_mut(ki).zip(floats[..n * kf].par_chunks_mut(kf)).zip(probes.par_iter()).for_each_init(
            || (vec![0i32; ni], vec![0f32; nf], vec![0u8; encode::ACTIONS]),
            |(si, sf, sm), ((i, f), (_, _, r))| {
                si.fill(0);
                sf.fill(0.0);
                sm.fill(0);
                encode_run::encode_run(r, &r.legal_actions(), &[], &[], si, sf, sm);
                i.copy_from_slice(&si[..ki]);
                f.copy_from_slice(&sf[..kf]);
            },
        );
    });
    probes.iter().map(|(s, p, _)| (*s, p.option, p.fight, p.draw)).collect()
}

/// Probe fights' first rows encoded into fresh arrays: (meta, ints, floats).
fn encode_probes(py: Python<'_>, probes: Vec<(usize, probe::Probe, Run)>) -> (Vec<(usize, i32, u8, u32)>, Vec<i32>, Vec<f32>) {
    let (ni, nf) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len());
    let (mut ints, mut floats) = (vec![0i32; probes.len() * ni], vec![0f32; probes.len() * nf]);
    let meta = write_probes(py, &probes, &mut ints, &mut floats);
    (meta, ints, floats)
}

/// The seed of replay `retry` (1, 2...) of the practice fight set up from `seed`, with `retry_reseed`:
/// its draws and monster rolls.
fn fight_seed(seed: i64, retry: u32) -> i64 {
    seed ^ (retry as i64).wrapping_mul(0x9E37_79B9_7F4A_7C15_u64 as i64)
}

/// One decision of a recorded run, with where the engine was (so playback can spot drift).
fn record_run_step(r: &Run, a: &RunAction) -> serde_json::Value {
    let mut v = run_step(r, a);
    v["floor"] = json!(r.floor);
    v["hp"] = json!(r.combat.as_ref().map_or(r.hp, |c| c.player.hp));
    v
}

/// The run step drinks a Smoke Bomb (it can only be drunk where it works: an escape).
fn smoke_bomb(r: &Run, a: &RunAction) -> bool {
    match (a, &r.combat) {
        (RunAction::Combat { action: PlayerAction::UsePotion { slot, .. } }, Some(c)) => c.potions.get(*slot) == Some(&Some(sts_core::potions::PotionId::SmokeBomb)),
        _ => false,
    }
}

/// A recorded run step: with `harness` (a fight to play back with the harness command), a fight
/// move as `record_move` records it; with `detail`, a combat move also keeps its state.
fn record_step(harness: bool, detail: bool, r: &Run, a: &RunAction) -> serde_json::Value {
    let mut v = match (harness, a, &r.combat) {
        (true, RunAction::Combat { action }, Some(c)) => record_move(c, action),
        _ => record_run_step(r, a),
    };
    if let (true, RunAction::Combat { action }, Some(c)) = (detail, a, &r.combat) {
        v["state"] = move_state(c, action);
    }
    v
}

/// The state a combat move was taken in, compactly: the turn, HP, block, energy, the attack
/// damage coming (intents' damage x hits), the hand ([name, type, cost, playable]), orbs, the
/// player's powers and stance, the monsters alive ([name, HP, block, intent damage, hits]), the
/// piles' sizes, and what the move played, targeted, drank or chose.
fn move_state(c: &Combat, a: &PlayerAction) -> serde_json::Value {
    let o = c.observe();
    let incoming: i32 = o.monsters.iter().filter(|m| m.alive && m.intent_damage > 0).map(|m| m.intent_damage * m.intent_hits.max(1)).sum();
    let card = |v: &sts_core::combat::observe::CardView| json!([v.name, v.card_type, v.cost, v.playable]);
    let monster = |t: &Option<usize>| t.and_then(|t| o.monsters.iter().find(|m| m.id == t)).map(|m| m.name.clone());
    let what = match a {
        PlayerAction::PlayCard { hand_index, target } => json!({"card": o.hand.get(*hand_index).map(card), "target": monster(target)}),
        PlayerAction::UsePotion { slot, target } => {
            json!({"potion": c.potions.get(*slot).copied().flatten().map(|p| p.java_id()), "target": monster(target)})
        }
        PlayerAction::DiscardPotion { slot } => json!({"discard": c.potions.get(*slot).copied().flatten().map(|p| p.java_id())}),
        PlayerAction::Choose { indices } => {
            let ch = o.choice.as_ref();
            json!({"choice": ch.map(|ch| ch.kind.clone()), "chose": indices.iter().filter_map(|&i| ch.and_then(|ch| ch.options.get(i)).map(|v| v.name.clone())).collect::<Vec<_>>()})
        }
        PlayerAction::EndTurn => json!({"end": true}),
    };
    json!({"turn": o.turn, "hp": o.player.hp, "max_hp": o.player.max_hp, "block": o.player.block, "energy": o.player.energy, "incoming": incoming,
           "hand": o.hand.iter().map(card).collect::<Vec<_>>(), "orbs": o.player.orbs.iter().map(|x| x.id.clone()).collect::<Vec<_>>(),
           "powers": o.player.powers.iter().map(|p| json!([p.id, p.amount])).collect::<Vec<_>>(), "stance": o.player.stance,
           "monsters": o.monsters.iter().filter(|m| m.alive).map(|m| json!([m.name, m.hp, m.block, m.intent_damage, m.intent_hits])).collect::<Vec<_>>(),
           "draw": o.draw_pile.len(), "discard": o.discard_pile.len(), "exhaust": o.exhaust_pile.len(), "move": what})
}

/// One move of a recorded fight: its game commands and the state it was taken in.
fn record_move(c: &Combat, a: &PlayerAction) -> serde_json::Value {
    json!({"cmds": combat_commands(c, a), "turn": c.turn, "hp": c.player.hp, "action": a})
}

/// A batch of games (fights or whole runs) stepped together.
#[pyclass(module = "sts")]
struct Envs {
    settings: Settings,
    slots: Vec<Slot>,
    /// The fight look-ahead's copies (`playout_start`).
    playouts: Vec<search::Playout>,
}

/// The bytes of numbers (`i32`, `f32`) as Python reads them ("<i4", "<f4"), without converting
/// one by one (that took 30 ms of every observation of 2,048 games).
fn le_bytes<T: Copy>(v: &[T]) -> &[u8] {
    #[cfg(not(target_endian = "little"))]
    compile_error!("the arrays handed to Python are little-endian");
    // SAFETY: plain numbers have no padding, and any byte is a valid u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A bytearray's bytes as 4-byte numbers, in place (`observe_into`).
fn numbers_mut<T: Copy>(b: &mut [u8]) -> PyResult<&mut [T]> {
    #[cfg(not(target_endian = "little"))]
    compile_error!("the arrays handed to Python are little-endian");
    // SAFETY: plain numbers (i32, f32) have no invalid bit patterns; the alignment is checked.
    let (head, numbers, tail) = unsafe { b.align_to_mut::<T>() };
    if !head.is_empty() || !tail.is_empty() {
        return Err(err("observe_into: a buffer isn't aligned for its numbers".into()));
    }
    Ok(numbers)
}

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pymethods]
impl Envs {
    /// `mode` is "fight" (single fights) or "run" (whole runs). Fights pick an encounter from
    /// `encounters` (default: all of them) and, with `random_decks`, a random deck of the
    /// character's cards. `character`: "ironclad" (default), "silent", "defect" or "watcher".
    /// `practice_decks` ("mix", "grown", "engine" or "random") makes practice fights
    /// `practice::build_deck`'s, for any character, `practice_act` (1-4) holding them to one act.
    /// `harness` sets those fights up as the modded game's harness command does (A0-10), so with
    /// `record` their episodes can be played back in the game (python/watch/watch_practice.py).
    /// `detail` (with `record`) keeps each combat move's state, for analysing play.
    #[new]
    #[pyo3(signature = (n, mode="fight", seed=0, ascension=0, encounters=None, random_decks=true, final_act=true, max_turns=100, max_steps=20000, record=false, forced_purge=0.0, forced_purge_curse=0.0, practice=0.0, seeded=0.0, ascension_range=None, practice_archetype=None, practice_ascension_range=None, practice_boss=None, character="ironclad", practice_decks=None, practice_act=None, harness=false, detail=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        mode: &str,
        seed: i64,
        ascension: i32,
        encounters: Option<Vec<String>>,
        random_decks: bool,
        final_act: bool,
        max_turns: i32,
        max_steps: u64,
        record: bool,
        forced_purge: f64,
        forced_purge_curse: f64,
        practice: f64,
        seeded: f64,
        ascension_range: Option<(i32, i32)>,
        practice_archetype: Option<String>,
        practice_ascension_range: Option<(i32, i32)>,
        practice_boss: Option<f64>,
        character: &str,
        practice_decks: Option<String>,
        practice_act: Option<i32>,
        harness: bool,
        detail: bool,
    ) -> PyResult<Envs> {
        let character = Character::from_name(character).ok_or_else(|| err(format!("unknown character {character:?}")))?;
        if let Some(d) = &practice_decks {
            if d != "mix" && practice::DeckKind::from_name(d).is_none() {
                return Err(err(format!("practice_decks must be \"mix\", \"grown\", \"engine\" or \"random\", not {d:?}")));
            }
        }
        if practice_act.is_some_and(|a| !(1..=4).contains(&a)) {
            return Err(err(format!("practice_act must be 1-4, not {practice_act:?}")));
        }
        if let Some(a) = &practice_archetype {
            if !practice::ARCHETYPES.iter().any(|x| x.name == a) {
                return Err(err(format!("unknown practice archetype {a:?}")));
            }
        }
        let fight = match mode {
            "fight" => true,
            "run" => false,
            other => return Err(err(format!("mode must be \"fight\" or \"run\", not {other:?}"))),
        };
        if n == 0 {
            return Err(err("n must be at least 1".into()));
        }
        let encounters = encounters.unwrap_or_else(|| encounters::all().iter().map(|s| s.to_string()).collect());
        if encounters.is_empty() {
            return Err(err("encounters is empty".into()));
        }
        let settings = Settings { fight, seed, ascension: ascension.clamp(0, 20), encounters, random_decks, final_act, n, max_turns, max_steps, record, harness, detail, forced_purge, forced_purge_curse, practice, seeded, practice_archetype, practice_decks, practice_act,
                                  practice_ascension_range: practice_ascension_range.map(|(lo, hi)| (lo.clamp(0, 20), hi.clamp(lo.clamp(0, 20), 20))),
                                  practice_boss, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, scenarios: Default::default(), scenario_share: 0.0, scenario_sequential: false, character,
                                  ascension_range: ascension_range.map(|(lo, hi)| (lo.clamp(0, 20), hi.clamp(lo.clamp(0, 20), 20))), ascension_low_share: 0.0, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let slots = (0..n).into_par_iter().map(|i| Slot::new(&settings, i, 0)).collect::<Result<Vec<_>, _>>().map_err(err)?;
        Ok(Envs { settings, slots, playouts: Vec::new() })
    }

    fn __len__(&self) -> usize {
        self.slots.len()
    }

    /// Changes `forced_purge` and `forced_purge_curse` (training can fade them out).
    /// The share of games playing practice fights, from each game's next episode (runs only; see
    /// `Settings::practice`).
    fn set_practice(&mut self, p: f64) {
        self.settings.practice = p;
    }

    /// The share of games (after the practice ones) whose runs start seeded, from each game's
    /// next episode (see `Settings::seeded`).
    fn set_seeded(&mut self, p: f64) {
        self.settings.seeded = p;
    }

    /// Seeded runs only from these of the character's seeds (`WATCHER_SEEDS`... names), from each
    /// game's next episode (None: all of them but `CONTROL_SEED`).
    #[pyo3(signature = (names=None))]
    fn set_seeds(&mut self, names: Option<Vec<String>>) -> PyResult<()> {
        if let Some(ns) = &names {
            let known: Vec<&str> = seeds_for(self.settings.character).iter().map(|(n, _)| *n).collect();
            if ns.is_empty() || ns.iter().any(|n| !known.contains(&n.as_str())) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("seeds: some of {known:?}, not {ns:?}")));
            }
        }
        self.settings.seed_names = names;
        Ok(())
    }

    /// Seeded runs start with these cards instead of a named seed, from each game's next episode (None: named seeds
    /// again). Names as the game shows them, "+" for upgraded: what a card is worth to a network (`seed_rates.py --cards`).
    #[pyo3(signature = (cards=None))]
    fn set_seed_cards(&mut self, cards: Option<Vec<String>>) -> PyResult<()> {
        self.settings.seed_cards = cards
            .map(|names| {
                names
                    .iter()
                    .map(|n| {
                        let (base, up) = n.strip_suffix('+').map_or((n.as_str(), false), |b| (b, true));
                        sts_core::cards::CardId::from_name(base)
                            .map(|id| (id, up))
                            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no card named {n:?}")))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        Ok(())
    }

    /// The act seeded runs get their cards in, on its first map screen there, from each game's next episode (1: from the start):
    /// a run that plays Act 1 as it does, then is given an engine to build on.
    fn set_seed_act(&mut self, act: i32) {
        self.settings.seed_act = act;
    }

    /// Runs (and play-on scenarios) start holding the three keys, from each game's next episode: what a network would do at the
    /// Heart, keys' cost aside. For evaluations only.
    fn set_free_keys(&mut self, on: bool) {
        self.settings.free_keys = on;
    }

    /// Runs from the next episode on leave the keys to the hand-written rule, for a network that trained with `set_free_keys`
    /// (`RunConfig::key_rule`). A result with it says so: the keys aren't the network's.
    #[pyo3(signature = (on, recall_hp_act2=101, recall_hp_act3=60))]
    fn set_key_rule(&mut self, on: bool, recall_hp_act2: i32, recall_hp_act3: i32) {
        self.settings.key_rule = on.then_some(sts_core::run::KeyRule { recall_hp_act2, recall_hp_act3 });
    }

    /// Weights for `practice_decks` fights, from each game's next episode (`practice::Focus`): the
    /// chance of each act 1-4 (None: 30/30/30/10%), a weight for each boss named (1 for the
    /// others) and the chance a fight not sent to the boss is an elite (None: the usual roll).
    #[pyo3(signature = (act_weights=None, boss_weights=None, elite_share=None))]
    fn set_practice_focus(&mut self, act_weights: Option<Vec<f64>>, boss_weights: Option<Vec<(String, f64)>>, elite_share: Option<f64>) -> PyResult<()> {
        if elite_share.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("elite_share: a chance in 0..1, not {elite_share:?}")));
        }
        let act_weights = match act_weights {
            Some(w) if w.len() == 4 && w.iter().all(|x| *x >= 0.0) && w.iter().sum::<f64>() > 0.0 => Some([w[0], w[1], w[2], w[3]]),
            Some(w) => return Err(pyo3::exceptions::PyValueError::new_err(format!("act_weights: four weights, not all 0, not {w:?}"))),
            None => None,
        };
        let boss_weights = boss_weights.unwrap_or_default();
        if let Some((b, _)) = boss_weights.iter().find(|(b, w)| *w < 0.0 || !sts_core::encounters::all().contains(&b.as_str())) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("boss_weights: no such encounter, or a negative weight: {b}")));
        }
        self.settings.practice_focus = Some(practice::Focus { act_weights, boss_weights, elite_share });
        Ok(())
    }

    /// Practice fights, from each game's next episode: a lost one is played again, the same fight,
    /// until a replay wins or it has been replayed `k` times (0: never; `Settings::retry_losses`),
    /// with other draws and monster rolls each time if `reseed`. The episode summary's "retry"
    /// counts the replay (0: played the first time).
    #[pyo3(signature = (k, reseed=true))]
    fn set_retry_losses(&mut self, k: u32, reseed: bool) {
        self.settings.retry_losses = k;
        self.settings.retry_reseed = reseed;
    }

    /// Each game's current replay count (`set_retry_losses`): 0 for a fight played the first time
    /// (and for runs), so training can play replays otherwise (train_runs.py --retry-temp).
    fn retries(&self) -> Vec<u32> {
        self.slots.iter().map(|s| s.retry).collect()
    }

    /// Each run's ascension drawn from `lo..=hi` from each game's next episode (None: the fixed
    /// `ascension`).
    #[pyo3(signature = (range=None))]
    fn set_ascension_range(&mut self, range: Option<(i32, i32)>) {
        self.settings.ascension_range = range.map(|(lo, hi)| (lo.clamp(0, 20), hi.clamp(lo.clamp(0, 20), 20)));
    }

    /// With an ascension range: the share of runs that draw their ascension from below it (0 to
    /// LO - 1) instead, from each game's next episode.
    #[pyo3(signature = (share=0.0))]
    fn set_ascension_low_share(&mut self, share: f64) {
        self.settings.ascension_low_share = share.clamp(0.0, 1.0);
    }

    /// For each game, what its last step did (runs; `Habit`): cards added, cards removed, deck
    /// size, the HP share a rest was taken at (-1: no rest), a Recall without the Sapphire key
    /// (0/1), an elite won (0/1), a card upgraded at a rest site (0/1), a potion drunk in a fight
    /// that isn't an elite or a boss (0/1), a shop left with an affordable removal unused (0/1),
    /// basic Strikes upgraded on an upgrade grid. Flat, `Habit::FIELDS` per game. Training shapes
    /// rewards on them.
    fn habits(&self) -> Vec<f32> {
        self.slots.iter().flat_map(|s| s.habit.row()).collect()
    }

    /// For each game: the archetype cards its last step added to a seeded run's deck (0 for
    /// other runs; at most `SYNERGY_CAP` a run in all). Training rewards them.
    fn synergy_picks(&self) -> Vec<u32> {
        self.slots.iter().map(|s| s.synergy).collect()
    }

    /// Boss scenarios (`human_bosses.py` JSON lines) for practice fights: `share` of practice
    /// fights play one, from each game's next episode; `sequential` takes them in turn instead of
    /// at random (evaluation). Returns how many lines were skipped (cards or bosses the engine
    /// doesn't have). Every game's current episode starts again (with the same episode number),
    /// so the first ones play scenarios too. With `play_on` each is a normal run after its fight
    /// (not a practice fight, no end at its last boss; `scenario` in its summary is its index in
    /// the lines kept), with the batch's `final_act` and no keys given.
    #[pyo3(signature = (lines, share=1.0, sequential=false, play_on=false))]
    fn set_scenarios(&mut self, lines: Vec<String>, share: f64, sequential: bool, play_on: bool) -> PyResult<usize> {
        // Only this batch's character's scenarios (others count as skipped).
        let character = self.settings.character;
        let parsed: Vec<_> = lines.iter().filter_map(|l| scenarios::Scenario::parse(l).ok()).filter(|sc| sc.character == character).map(|sc| scenarios::Scenario { play_on, ..sc }).collect();
        let skipped = lines.len() - parsed.len();
        self.settings.scenarios = std::sync::Arc::new(parsed);
        self.settings.scenario_share = share;
        self.settings.scenario_sequential = sequential;
        let settings = &self.settings;
        let episodes: Vec<u64> = self.slots.iter().map(|s| s.episode).collect();
        self.slots = episodes.into_par_iter().enumerate().map(|(i, e)| Slot::new(settings, i, e)).collect::<Result<Vec<_>, _>>().map_err(err)?;
        Ok(skipped)
    }

    /// Human boss states (`human_bosses.py` JSON lines) that `share` of the games neither practice fights nor seeded (the first of
    /// them, as with `practice` and `seeded`) start every episode from, from each game's next episode: a random line's fight, then
    /// a normal run (`Scenario::build_play_on`: not a practice fight, no end at its boss, no keys given, the line's ascension),
    /// named "late" where a seeded run has its seed's name. Returns how many lines were skipped: another character's, what
    /// the engine doesn't have, and any fight but an act's boss, which wouldn't go on from there.
    fn set_late_starts(&mut self, lines: Vec<String>, share: f64) -> PyResult<usize> {
        if !(0.0..=1.0).contains(&share) {
            return Err(err(format!("late starts: a share in 0..1, not {share}")));
        }
        let character = self.settings.character;
        let parsed: Vec<_> = lines.iter().filter_map(|l| scenarios::Scenario::parse(l).ok()).filter(|sc| sc.character == character && sc.room == RoomType::Boss && sc.act <= 3).collect();
        let skipped = lines.len() - parsed.len();
        self.settings.late_starts = std::sync::Arc::new(parsed);
        self.settings.late_share = share;
        Ok(skipped)
    }

    /// For each game in a fight: each monster slot's next move (the one its intent stands for, the
    /// last of its `move_history`) as a move vocabulary id (`encode::move_id`), in the encoding's
    /// slot order (`encode::monster_slots`); 0 for an empty slot or a game outside a fight. Flat,
    /// `MAX_MONSTERS` per game. Training labels each fight turn with the move a monster shows on the
    /// turn after (`train_runs.py --next-move`); the network itself never sees it.
    fn monster_next_moves(&self) -> Vec<i32> {
        let mut out = vec![0; self.slots.len() * encode::MAX_MONSTERS];
        for (i, s) in self.slots.iter().enumerate() {
            let c = match &s.game {
                Game::Fight(c, _) => Some(c.as_ref()),
                Game::Run(r, _) => r.combat.as_deref().filter(|_| matches!(r.screen, Screen::Combat)),
            };
            let Some(c) = c else { continue };
            for (j, &m) in encode::monster_slots(c).iter().enumerate() {
                if let Some(&mv) = c.monsters[m].move_history.last() {
                    out[i * encode::MAX_MONSTERS + j] = encode::move_id(c.monsters[m].id, mv);
                }
            }
        }
        out
    }

    /// For each game: whether its current episode is a practice fight.
    fn practice_mask(&self) -> Vec<bool> {
        self.slots.iter().map(|s| s.practice.is_some()).collect()
    }

    /// For each game: whether its current episode is a late start (`set_late_starts`). Training keeps their habits out of its counts.
    fn late_mask(&self) -> Vec<bool> {
        self.slots.iter().map(|s| s.seeded == Some(LATE_START)).collect()
    }

    #[pyo3(signature = (p, curse=None))]
    fn set_forced_purge(&mut self, p: f64, curse: Option<f64>) {
        self.settings.forced_purge = p;
        if let Some(c) = curse {
            self.settings.forced_purge_curse = c;
        }
    }

    /// Starts a fresh first episode in every game.
    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        let s = self.settings.clone();
        let slots = py.detach(|| (0..s.n).into_par_iter().map(|i| Slot::new(&s, i, 0)).collect::<Result<Vec<_>, _>>()).map_err(err)?;
        self.slots = slots;
        Ok(())
    }

    /// Number of legal actions in each game.
    fn num_legal(&self) -> Vec<usize> {
        self.slots.iter().map(|s| s.game.num_legal()).collect()
    }

    /// Game `i`'s legal actions, as JSON strings, in index order.
    fn legal_actions(&self, i: usize) -> PyResult<Vec<String>> {
        let s = self.slots.get(i).ok_or_else(|| err(format!("no game {i}")))?;
        Ok(match &s.game {
            Game::Fight(_, l) => l.iter().map(|a| serde_json::to_string(a).unwrap()).collect(),
            Game::Run(_, l) => l.iter().map(|a| serde_json::to_string(a).unwrap()).collect(),
        })
    }

    /// Game `i`'s full observation as a JSON string.
    fn observe(&self, i: usize) -> PyResult<String> {
        let s = self.slots.get(i).ok_or_else(|| err(format!("no game {i}")))?;
        Ok(match &s.game {
            Game::Fight(c, _) => serde_json::to_string(&c.observe()).unwrap(),
            Game::Run(r, _) => serde_json::to_string(&r.observe()).unwrap(),
        })
    }

    /// Steps every game with its chosen action index (in parallel, without the GIL). Returns
    /// one status per game: 0 running, 1 won, 2 lost (that game has already been reset).
    fn step(&mut self, py: Python<'_>, actions: Vec<usize>) -> PyResult<Vec<u32>> {
        if actions.len() != self.slots.len() {
            return Err(err(format!("expected {} actions, got {}", self.slots.len(), actions.len())));
        }
        let s = self.settings.clone();
        let slots = &mut self.slots;
        py.detach(|| slots.par_iter_mut().zip(actions.par_iter()).map(|(slot, &k)| slot.step(&s, k)).collect::<Result<Vec<_>, _>>()).map_err(err)
    }

    /// Steps every game with a uniformly random legal action (for benchmarks and smoke tests).
    fn step_random(&mut self, py: Python<'_>, seed: i64) -> PyResult<Vec<u32>> {
        let s = self.settings.clone();
        let slots = &mut self.slots;
        py.detach(|| {
            slots
                .par_iter_mut()
                .map(|slot| {
                    let n = slot.game.num_legal() as i32;
                    let mut r = StsRandom::new(seed ^ (slot.index as i64).wrapping_mul(0x9E37_79B9) ^ slot.steps as i64);
                    let k = r.random(n - 1) as usize;
                    slot.step(&s, k)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(err)
    }

    /// What each game in `roots` (`(game, action slots)`) would look like after each of its action slots, without changing
    /// it (a copy takes the slot, nothing redrawn): the copies' arrays as `observe_arrays`, and `(root, slot, over)` for
    /// each row (`over`: the copy's run ended). A slot that only toggles a card selection shows the selection after it.
    /// For judging a choice by the deck network's value of where each option leads (python/search/value_peek.py).
    fn peek_after<'py>(&self, py: Python<'py>, roots: Vec<(usize, Vec<usize>)>)
        -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<(u32, u32, bool)>)> {
        if self.settings.fight {
            return Err(err("peek_after needs mode=\"run\"".into()));
        }
        let (ni, nf, na) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len(), encode::ACTIONS);
        let jobs: Vec<(u32, usize, usize)> = roots.iter().enumerate()
            .flat_map(|(i, (g, slots))| slots.iter().map(move |&a| (i as u32, *g, a))).collect();
        for &(_, g, _) in &jobs {
            if g >= self.slots.len() {
                return Err(err(format!("no game {g}")));
            }
        }
        let slots = &self.slots;
        let rows: Vec<(Vec<i32>, Vec<f32>, Vec<u8>, bool)> = py.detach(|| {
            jobs.par_iter().map(|&(_, g, a)| {
                let slot = &slots[g];
                let Game::Run(run, legal) = &slot.game else { return Err(format!("game {g} is not a run")) };
                let (mut r, mut legal, mut picked, mut skipped) = (run.as_ref().clone(), legal.clone(), slot.picked.clone(), slot.skipped.clone());
                if let Some(action) = agent::slot_action(&r, &legal, &mut picked, &skipped, a).map_err(|e| format!("game {g}: {e}"))? {
                    agent::before_step(&mut skipped, &r, &action);
                    r.step(&action).map_err(|e| format!("game {g}: {e}"))?;
                    agent::after_step(&mut skipped, &r);
                    legal = if r.is_over() { Vec::new() } else { r.legal_actions() };
                    picked.clear();
                }
                let (mut i, mut f, mut m) = (vec![0i32; ni], vec![0f32; nf], vec![0u8; na]);
                if !r.is_over() {
                    encode_run::encode_run(&r, &legal, &picked, &skipped, &mut i, &mut f, &mut m);
                }
                Ok((i, f, m, r.is_over()))
            }).collect::<Result<Vec<_>, String>>()
        }).map_err(err)?;
        let mut ints = Vec::with_capacity(rows.len() * ni);
        let mut floats = Vec::with_capacity(rows.len() * nf);
        let mut mask = Vec::with_capacity(rows.len() * na);
        let mut meta = Vec::with_capacity(rows.len());
        for ((i, f, m, over), &(root, _, a)) in rows.into_iter().zip(&jobs) {
            ints.extend(i);
            floats.extend(f);
            mask.extend(m);
            meta.push((root, a as u32, over));
        }
        Ok((PyBytes::new(py, le_bytes(&ints)), PyBytes::new(py, le_bytes(&floats)), PyBytes::new(py, &mask), meta))
    }

    /// Every game as arrays (runs add the `run_ints`/`run_floats` fields after the fight's):
    /// (ints, floats, mask) as raw little-endian bytes of shape
    /// (n, INTS) int32, (n, FLOATS) float32 and (n, ACTIONS) uint8. `python/core/sts_env.py` wraps
    /// them in numpy arrays; `layout()` names the fields.
    fn observe_arrays<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let run_fields = if self.settings.fight { (0, 0) } else { (encode_run::run_int_len(), encode_run::run_float_len()) };
        let (ni, nf, na) = (encode::int_len() + run_fields.0, encode::float_len() + run_fields.1, encode::ACTIONS);
        let n = self.slots.len();
        let mut ints = vec![0i32; n * ni];
        let mut floats = vec![0f32; n * nf];
        let mut mask = vec![0u8; n * na];
        let slots = &self.slots;
        py.detach(|| {
            ints.par_chunks_mut(ni)
                .zip(floats.par_chunks_mut(nf))
                .zip(mask.par_chunks_mut(na))
                .zip(slots.par_iter())
                .for_each(|(((i, f), m), slot)| {
                    match &slot.game {
                        Game::Fight(c, legal) => encode::encode_fight(c, legal, &slot.picked, i, f, m),
                        Game::Run(r, legal) => encode_run::encode_run(r, legal, &slot.picked, &slot.skipped, i, f, m),
                    }
                });
        });
        Ok((PyBytes::new(py, le_bytes(&ints)), PyBytes::new(py, le_bytes(&floats)), PyBytes::new(py, &mask)))
    }

    /// `observe_arrays` written into the caller's bytearrays (ints, floats, mask, of the sizes it
    /// returns), which the caller reuses step after step (`RunEnv(reuse_obs=True)`): nothing is
    /// allocated. On the training machine (WSL) fresh arrays and Python's copies of them took 11-25 ms a step at
    /// 2,048 games (4 ms on the Mac), mostly the page faults of fresh memory (2026-10-27). Every
    /// byte is written: zeros, then the encoding, as into fresh arrays.
    fn observe_into(&self, py: Python<'_>, ints: &Bound<'_, PyByteArray>, floats: &Bound<'_, PyByteArray>, mask: &Bound<'_, PyByteArray>) -> PyResult<()> {
        let run_fields = if self.settings.fight { (0, 0) } else { (encode_run::run_int_len(), encode_run::run_float_len()) };
        let (ni, nf, na) = (encode::int_len() + run_fields.0, encode::float_len() + run_fields.1, encode::ACTIONS);
        let n = self.slots.len();
        if ints.len() != n * ni * 4 || floats.len() != n * nf * 4 || mask.len() != n * na {
            return Err(err(format!("observe_into: buffers of {}, {} and {} bytes, not {}, {} and {}", ints.len(), floats.len(), mask.len(),
                                   n * ni * 4, n * nf * 4, n * na)));
        }
        // SAFETY: the bytearrays are held by this call, and nothing resizes them while it runs.
        let (ib, fb, mb) = unsafe { (ints.as_bytes_mut(), floats.as_bytes_mut(), mask.as_bytes_mut()) };
        let (ints, floats) = (numbers_mut::<i32>(ib)?, numbers_mut::<f32>(fb)?);
        let slots = &self.slots;
        py.detach(|| {
            ints.par_chunks_mut(ni)
                .zip(floats.par_chunks_mut(nf))
                .zip(mb.par_chunks_mut(na))
                .zip(slots.par_iter())
                .for_each(|(((i, f), m), slot)| {
                    i.fill(0);
                    f.fill(0.0);
                    m.fill(0);
                    match &slot.game {
                        Game::Fight(c, legal) => encode::encode_fight(c, legal, &slot.picked, i, f, m),
                        Game::Run(r, legal) => encode_run::encode_run(r, legal, &slot.picked, &slot.skipped, i, f, m),
                    }
                });
        });
        Ok(())
    }

    /// Steps every fight with one fixed action slot each (see `layout()["actions"]`). Returns
    /// statuses like `step`.
    fn step_slots(&mut self, py: Python<'_>, actions: Vec<usize>) -> PyResult<Vec<u32>> {
        if actions.len() != self.slots.len() {
            return Err(err(format!("expected {} actions, got {}", self.slots.len(), actions.len())));
        }
        let s = self.settings.clone();
        let slots = &mut self.slots;
        py.detach(|| slots.par_iter_mut().zip(actions.par_iter()).map(|(slot, &a)| slot.step_slot(&s, a)).collect::<Result<Vec<_>, _>>()).map_err(err)
    }

    /// Starts the fight look-ahead (see `search.rs`) from each `(game, first moves)` in `roots`:
    /// `k` redrawn copies of the game for each first move (action slots), each with that move
    /// taken; each then goes on for `turns` turns, at most `max_moves` moves. With `fight_only`,
    /// a run's playout also ends with its fight, as a practice fight's does (reported like one,
    /// kind 3), instead of going on as a leaf. A game that isn't waiting for a fight move outside
    /// a card selection gets none. Replaces any playouts still going; returns `(root, first
    /// move, copy)` for each playout, in order.
    #[pyo3(signature = (roots, k=8, turns=1, seed=0, max_moves=30, fight_only=false))]
    #[allow(clippy::too_many_arguments)]
    fn playout_start(&mut self, py: Python<'_>, roots: Vec<(usize, Vec<usize>)>, k: usize, turns: i32, seed: i64, max_moves: u32, fight_only: bool)
        -> PyResult<Vec<(u32, u32, u32)>> {
        if self.settings.fight {
            return Err(err("the look-ahead needs mode=\"run\"".into()));
        }
        let slots = &self.slots;
        let settings = &self.settings;
        for (g, _) in &roots {
            if *g >= slots.len() {
                return Err(err(format!("no game {g}")));
            }
        }
        let per_root: Vec<Vec<search::Playout>> = py.detach(|| {
            roots.par_iter().enumerate().map(|(i, (g, firsts))| {
                let slot = &slots[*g];
                let Game::Run(r, _) = &slot.game else { return Err(format!("game {g} is not a run")) };
                let s = seed.wrapping_add(slot.seed.wrapping_mul(31)).wrapping_add(slot.steps as i64 * 1_000_003);
                let practice = match slot.practice {
                    Some(_) => Some(slot.scenario.map(|(x, end)| (settings.scenarios[x].floor, end))),
                    None if fight_only => Some(None),
                    None => None,
                };
                search::start(i as u32, r, firsts, k, search::Horizon { turns, max_moves }, s, practice).map_err(|e| format!("game {g}: {e}"))
            }).collect::<Result<_, _>>()
        }).map_err(err)?;
        self.playouts = per_root.into_iter().flatten().collect();
        Ok(self.playouts.iter().map(|p| (p.root, p.first, p.det)).collect())
    }

    /// The playouts still going, as arrays (like `observe_arrays`), and their indices.
    fn playout_observe<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<u32>)> {
        let (ni, nf, na) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len(), encode::ACTIONS);
        let live: Vec<&search::Playout> = self.playouts.iter().filter(|p| p.end.is_none()).collect();
        let index = self.playouts.iter().enumerate().filter(|(_, p)| p.end.is_none()).map(|(i, _)| i as u32).collect();
        let n = live.len();
        let (mut ints, mut floats, mut mask) = (vec![0i32; n * ni], vec![0f32; n * nf], vec![0u8; n * na]);
        py.detach(|| {
            ints.par_chunks_mut(ni).zip(floats.par_chunks_mut(nf)).zip(mask.par_chunks_mut(na)).zip(live.par_iter())
                .for_each(|(((i, f), m), p)| encode_run::encode_run(&p.run, &p.legal, &p.picked, &[], i, f, m));
        });
        Ok((PyBytes::new(py, le_bytes(&ints)), PyBytes::new(py, le_bytes(&floats)), PyBytes::new(py, &mask), index))
    }

    /// `playout_observe` into reused buffers (as `observe_into`): the playouts still going, their
    /// rows written at the start of `ints`, `floats` and `mask` (bytearrays with room for at least
    /// that many rows), and their indices. Returns (rows written, indices). Saves allocating and
    /// copying every round's arrays (2026-10-30: 8 of the look-ahead's 31 seconds).
    fn playout_observe_into(&self, py: Python<'_>, ints: &Bound<'_, PyByteArray>, floats: &Bound<'_, PyByteArray>, mask: &Bound<'_, PyByteArray>)
        -> PyResult<(usize, Vec<u32>)> {
        let (ni, nf, na) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len(), encode::ACTIONS);
        let live: Vec<&search::Playout> = self.playouts.iter().filter(|p| p.end.is_none()).collect();
        let index = self.playouts.iter().enumerate().filter(|(_, p)| p.end.is_none()).map(|(i, _)| i as u32).collect();
        let n = live.len();
        if ints.len() < n * ni * 4 || floats.len() < n * nf * 4 || mask.len() < n * na {
            return Err(err(format!("playout_observe_into: buffers of {}, {} and {} bytes, {} rows need {}, {} and {}", ints.len(), floats.len(), mask.len(),
                                   n, n * ni * 4, n * nf * 4, n * na)));
        }
        // SAFETY: the bytearrays are held by this call, and nothing resizes them while it runs.
        let (ib, fb, mb) = unsafe { (ints.as_bytes_mut(), floats.as_bytes_mut(), mask.as_bytes_mut()) };
        let (ints, floats) = (numbers_mut::<i32>(&mut ib[..n * ni * 4])?, numbers_mut::<f32>(&mut fb[..n * nf * 4])?);
        let mb = &mut mb[..n * na];
        py.detach(|| {
            ints.par_chunks_mut(ni).zip(floats.par_chunks_mut(nf)).zip(mb.par_chunks_mut(na)).zip(live.par_iter())
                .for_each(|(((i, f), m), p)| {
                    i.fill(0);
                    f.fill(0.0);
                    m.fill(0);
                    encode_run::encode_run(&p.run, &p.legal, &p.picked, &[], i, f, m)
                });
        });
        Ok((n, index))
    }

    /// Steps each playout still going with an action slot (one per index `playout_observe`
    /// returned, in its order). Returns how many are still going.
    fn playout_step(&mut self, py: Python<'_>, actions: Vec<usize>) -> PyResult<usize> {
        let live = self.playouts.iter().filter(|p| p.end.is_none()).count();
        if actions.len() != live {
            return Err(err(format!("expected {live} actions, got {}", actions.len())));
        }
        let mut next = actions.into_iter();
        let actions: Vec<Option<usize>> = self.playouts.iter().map(|p| if p.end.is_none() { next.next() } else { None }).collect();
        let playouts = &mut self.playouts;
        py.detach(|| playouts.par_iter_mut().zip(actions.par_iter()).try_for_each(|(p, a)| a.map_or(Ok(()), |a| p.step(a)))).map_err(err)?;
        Ok(self.playouts.iter().filter(|p| p.end.is_none()).count())
    }

    /// Playout `i`'s full observation as a JSON string (for debugging).
    fn playout_json(&self, i: usize) -> PyResult<String> {
        let p = self.playouts.get(i).ok_or_else(|| err(format!("no playout {i}")))?;
        Ok(serde_json::to_string(&p.run.observe()).unwrap())
    }

    /// Ends the look-ahead: how each playout ended, in order, as (kind: 0 a leaf to score, 1
    /// died, 2 won the run, 3 a practice fight over; then for 3: won, HP, max HP, potions held,
    /// bosses beaten), and the arrays (like `observe_arrays`, no mask) of the leaves, in order.
    /// A playout still going counts as a leaf.
    fn playout_results<'py>(&mut self, py: Python<'py>) -> PyResult<(Vec<(u8, bool, i32, i32, u32, i32)>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let playouts = std::mem::take(&mut self.playouts);
        let (ni, nf) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len());
        let ends: Vec<search::End> = playouts.iter().map(|p| p.end.unwrap_or(search::End::Leaf)).collect();
        let leaves: Vec<&search::Playout> = playouts.iter().zip(&ends).filter(|(_, e)| **e == search::End::Leaf).map(|(p, _)| p).collect();
        let n = leaves.len();
        let (mut ints, mut floats) = (vec![0i32; n * ni], vec![0f32; n * nf]);
        py.detach(|| {
            ints.par_chunks_mut(ni).zip(floats.par_chunks_mut(nf)).zip(leaves.par_iter()).for_each(|((i, f), p)| {
                let mut mask = vec![0u8; encode::ACTIONS];
                encode_run::encode_run(&p.run, &p.legal, &p.picked, &[], i, f, &mut mask);
            });
        });
        let meta = ends.iter().map(|e| match *e {
            search::End::Practice { won, hp, max_hp, potions, bosses_beaten } => (3, won, hp, max_hp, potions, bosses_beaten),
            e => (e.code(), false, 0, 0, 0, 0),
        }).collect();
        Ok((meta, PyBytes::new(py, le_bytes(&ints)), PyBytes::new(py, le_bytes(&floats))))
    }

    /// `playout_results` into reused buffers (as `playout_observe_into`): how each playout ended,
    /// and the leaves' arrays written at the start of `ints` and `floats` (bytearrays with room for
    /// every playout's row). Returns (the ends, as `playout_results` gives them; leaves written).
    fn playout_results_into(&mut self, py: Python<'_>, ints: &Bound<'_, PyByteArray>, floats: &Bound<'_, PyByteArray>)
        -> PyResult<(Vec<(u8, bool, i32, i32, u32, i32)>, usize)> {
        let playouts = std::mem::take(&mut self.playouts);
        let (ni, nf) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len());
        let ends: Vec<search::End> = playouts.iter().map(|p| p.end.unwrap_or(search::End::Leaf)).collect();
        let leaves: Vec<&search::Playout> = playouts.iter().zip(&ends).filter(|(_, e)| **e == search::End::Leaf).map(|(p, _)| p).collect();
        let n = leaves.len();
        if ints.len() < n * ni * 4 || floats.len() < n * nf * 4 {
            return Err(err(format!("playout_results_into: buffers of {} and {} bytes, {n} leaves need {} and {}", ints.len(), floats.len(), n * ni * 4, n * nf * 4)));
        }
        // SAFETY: the bytearrays are held by this call, and nothing resizes them while it runs.
        let (ib, fb) = unsafe { (ints.as_bytes_mut(), floats.as_bytes_mut()) };
        let (ints, floats) = (numbers_mut::<i32>(&mut ib[..n * ni * 4])?, numbers_mut::<f32>(&mut fb[..n * nf * 4])?);
        py.detach(|| {
            ints.par_chunks_mut(ni).zip(floats.par_chunks_mut(nf)).zip(leaves.par_iter()).for_each(|((i, f), p)| {
                i.fill(0);
                f.fill(0.0);
                let mut mask = vec![0u8; encode::ACTIONS];
                encode_run::encode_run(&p.run, &p.legal, &p.picked, &[], i, f, &mut mask);
            });
        });
        let meta = ends.iter().map(|e| match *e {
            search::End::Practice { won, hp, max_hp, potions, bosses_beaten } => (3, won, hp, max_hp, potions, bosses_beaten),
            e => (e.code(), false, 0, 0, 0, 0),
        }).collect();
        Ok((meta, n))
    }

    /// The deck network's lookup (`probe`): for each game in `games` at a card reward, every option (-1:
    /// skipping; else the card at that index added) at the start of each probe fight (`probe::fights`: this
    /// act's boss, the next act's bosses), from `draws` openings. Returns (meta as (game, option, fight,
    /// draw), the probes' first rows as arrays like `observe_arrays`, no mask), for the fight network's
    /// fight-outcome head. Games not at a card reward give none.
    fn probe_card_rewards<'py>(&self, py: Python<'py>, games: Vec<usize>, draws: u32, seed: i64)
        -> PyResult<(Vec<(usize, i32, u8, u32)>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let mut probes = Vec::new();
        for g in games {
            let slot = self.slots.get(g).ok_or_else(|| err(format!("no game {g}")))?;
            if let Game::Run(r, _) = &slot.game {
                probes.extend(probe::card_reward(r, draws, seed).into_iter().map(|(p, run)| (g, p, run)));
            }
        }
        let (meta, ints, floats) = encode_probes(py, probes);
        Ok((meta, PyBytes::new(py, le_bytes(&ints)), PyBytes::new(py, le_bytes(&floats))))
    }

    /// `probe_card_rewards` into reused buffers (as `playout_observe_into`): the probes' rows written at the start of
    /// `ints` and `floats`, each only its first `keep` (ints, floats) if given (`write_probes_prefix`). Returns (meta,
    /// rows written), or an error naming the rows needed if the buffers are too small (nothing written then).
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (games, draws, seed, ints, floats, keep=None))]
    fn probe_card_rewards_into(&self, py: Python<'_>, games: Vec<usize>, draws: u32, seed: i64, ints: &Bound<'_, PyByteArray>,
                               floats: &Bound<'_, PyByteArray>, keep: Option<(usize, usize)>) -> PyResult<(Vec<(usize, i32, u8, u32)>, usize)> {
        let (full_i, full_f) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len());
        let (ni, nf) = keep.map_or((full_i, full_f), |(a, b)| (a.min(full_i), b.min(full_f)));
        if let Some(g) = games.iter().find(|&&g| g >= self.slots.len()) {
            return Err(err(format!("no game {g}")));
        }
        // The probe fights set up in parallel, game by game (each a few dozen copies of the run started in a fight).
        let slots = &self.slots;
        let probes: Vec<(usize, probe::Probe, Run)> = py.detach(|| {
            games.par_iter().flat_map_iter(|&g| match &slots[g].game {
                Game::Run(r, _) => probe::card_reward(r, draws, seed).into_iter().map(|(p, run)| (g, p, run)).collect::<Vec<_>>(),
                _ => Vec::new(),
            }).collect()
        });
        let n = probes.len();
        if ints.len() < n * ni * 4 || floats.len() < n * nf * 4 {
            return Err(err(format!("probe_card_rewards_into: room for {} rows, {n} needed", (ints.len() / (ni * 4)).min(floats.len() / (nf * 4)))));
        }
        // SAFETY: the bytearrays are held by this call, and nothing resizes them while it runs.
        let (ib, fb) = unsafe { (ints.as_bytes_mut(), floats.as_bytes_mut()) };
        let (ints, floats) = (numbers_mut::<i32>(&mut ib[..n * ni * 4])?, numbers_mut::<f32>(&mut fb[..n * nf * 4])?);
        let meta = if keep.is_some() { write_probes_prefix(py, &probes, (ni, nf), ints, floats) } else { write_probes(py, &probes, ints, floats) };
        Ok((meta, n))
    }

    /// The deck network's whole lookup (`probe::options`): for each game in `games` at a screen it probes (a card
    /// reward, a rest site, an upgrade or removal grid, a shop, a boss relic, a reward's relic or potion), the run as
    /// it stands and each option applied at the start of each probe fight from `draws` openings, written into reused
    /// buffers: each probe's first `keep` (ints, floats) into `ints` and `floats`, its meta as five i32 (game, option,
    /// sub, fight, draw) into `meta`. Each game's probes are encoded as they are set up, its games in parallel (no copies
    /// of the runs kept), then moved into the buffers. Returns the rows written and the options left to a copy of their
    /// card's probes as (game, option, the option probed), or an error naming the rows needed if the buffers are too
    /// small (nothing written then).
    #[allow(clippy::too_many_arguments)]
    fn probe_options_into(&self, py: Python<'_>, games: Vec<usize>, draws: u32, seed: i64, ints: &Bound<'_, PyByteArray>,
                          floats: &Bound<'_, PyByteArray>, meta: &Bound<'_, PyByteArray>, keep: (usize, usize))
        -> PyResult<(usize, Vec<(usize, i32, i32)>)> {
        let (ni, nf) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len());
        let (ki, kf) = (keep.0.min(ni), keep.1.min(nf));
        if let Some(g) = games.iter().find(|&&g| g >= self.slots.len()) {
            return Err(err(format!("no game {g}")));
        }
        let slots = &self.slots;
        let per_game: Vec<(Vec<i32>, Vec<i32>, Vec<f32>, Vec<(i32, i32)>)> = py.detach(|| {
            games.par_iter().map(|&g| {
                let (mut m, mut ri, mut rf, mut al) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                if let Game::Run(r, _) = &slots[g].game {
                    let (mut si, mut sf, mut sm) = (vec![0i32; ni], vec![0f32; nf], vec![0u8; encode::ACTIONS]);
                    al = probe::options_each(r, &slots[g].picked, &slots[g].skipped, draws, seed, |p, run| {
                        si.fill(0);
                        sf.fill(0.0);
                        encode_run::encode_run(&run, &run.legal_actions(), &[], &[], &mut si, &mut sf, &mut sm);
                        ri.extend_from_slice(&si[..ki]);
                        rf.extend_from_slice(&sf[..kf]);
                        m.extend_from_slice(&[g as i32, p.option, p.sub, p.fight as i32, p.draw as i32]);
                    });
                }
                (m, ri, rf, al)
            }).collect()
        });
        let n: usize = per_game.iter().map(|(m, _, _, _)| m.len() / 5).sum();
        let aliases: Vec<(usize, i32, i32)> =
            games.iter().zip(&per_game).flat_map(|(&g, (_, _, _, al))| al.iter().map(move |&(k, k0)| (g, k, k0))).collect();
        if ints.len() < n * ki * 4 || floats.len() < n * kf * 4 || meta.len() < n * 5 * 4 {
            return Err(err(format!("probe_options_into: room for {} rows, {n} needed",
                                   (ints.len() / (ki * 4)).min(floats.len() / (kf * 4)).min(meta.len() / 20))));
        }
        // SAFETY: the bytearrays are held by this call, and nothing resizes them while it runs.
        let (ib, fb, mb) = unsafe { (ints.as_bytes_mut(), floats.as_bytes_mut(), meta.as_bytes_mut()) };
        let (mut ints, mut floats, mut meta) =
            (numbers_mut::<i32>(&mut ib[..n * ki * 4])?, numbers_mut::<f32>(&mut fb[..n * kf * 4])?, numbers_mut::<i32>(&mut mb[..n * 20])?);
        // Each game's rows into its own stretch of the buffers, the games in parallel.
        let mut parts = Vec::with_capacity(per_game.len());
        for (m, ri, rf, _) in &per_game {
            let (a, rest_i) = std::mem::take(&mut ints).split_at_mut(ri.len());
            let (b, rest_f) = std::mem::take(&mut floats).split_at_mut(rf.len());
            let (c, rest_m) = std::mem::take(&mut meta).split_at_mut(m.len());
            (ints, floats, meta) = (rest_i, rest_f, rest_m);
            parts.push((a, b, c, ri, rf, m));
        }
        py.detach(|| parts.into_par_iter().for_each(|(a, b, c, ri, rf, m)| {
            a.copy_from_slice(ri);
            b.copy_from_slice(rf);
            c.copy_from_slice(m);
        }));
        Ok((n, aliases))
    }

    /// The seed of game `i`'s current episode.
    fn episode_seed(&self, i: usize) -> PyResult<i64> {
        Ok(self.slots.get(i).ok_or_else(|| err(format!("no game {i}")))?.seed)
    }

    /// A JSON summary of the last episode game `i` finished (None before the first).
    fn last_episode(&self, i: usize) -> PyResult<Option<String>> {
        let s = self.slots.get(i).ok_or_else(|| err(format!("no game {i}")))?;
        Ok(s.last.as_ref().map(|v| v.to_string()))
    }
}

/// Why a scenario line can't be played here (`Envs.set_scenarios` skips such lines), or None.
#[pyfunction]
fn scenario_problem(line: &str) -> Option<String> {
    scenarios::Scenario::parse(line).err()
}

/// The array layout: field names and sizes, the action slots, and the size of each ID
/// vocabulary (embedding tables need at least that many rows; IDs only grow as characters are
/// added).
#[pyfunction]
fn layout<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ints", encode::INT_FIELDS.to_vec())?;
    d.set_item("floats", encode::FLOAT_FIELDS.to_vec())?;
    d.set_item("run_ints", encode_run::RUN_INT_FIELDS.to_vec())?;
    d.set_item("run_floats", encode_run::RUN_FLOAT_FIELDS.to_vec())?;
    d.set_item("option_kinds", encode_run::OPTION_KINDS.to_vec())?;
    d.set_item("retired_option_kinds", encode_run::RETIRED_OPTION_KINDS.to_vec())?;
    d.set_item("screens", encode_run::SCREENS.to_vec())?;
    // Card names by encoded id - 1 (the encodings store a card as its id + 1, 0 = none).
    d.set_item("card_names", sts_core::cards::CARD_DATA.iter().map(|c| c.name).collect::<Vec<_>>())?;
    // The game's IDs (as its run history files write them), in the same order.
    d.set_item("card_java_ids", sts_core::cards::CARD_DATA.iter().map(|c| c.java_id).collect::<Vec<_>>())?;
    d.set_item("card_types", sts_core::cards::CARD_DATA.iter().map(|c| format!("{:?}", c.card_type)).collect::<Vec<_>>())?;
    d.set_item("relic_java_ids", sts_core::relics::RELIC_DATA.iter().map(|r| r.java_id).collect::<Vec<_>>())?;
    // Choice kinds by encoded id - 1, and Poison's encoded id (the "monster_power" fields).
    d.set_item("choice_kinds", encode::CHOICE_KINDS.to_vec())?;
    d.set_item("poison_power", sts_core::powers::PowerId::Poison as usize + 1)?;
    let actions = PyDict::new(py);
    actions.set_item("play", (encode::PLAY, encode::MAX_HAND, encode::MAX_MONSTERS + 1))?;
    actions.set_item("potion", (encode::POTION, encode::MAX_POTIONS, encode::MAX_MONSTERS + 1))?;
    actions.set_item("end_turn", encode::END_TURN)?;
    actions.set_item("pick", (encode::PICK, encode::MAX_OPTIONS))?;
    actions.set_item("confirm", encode::CONFIRM)?;
    actions.set_item("count", encode::ACTIONS)?;
    d.set_item("actions", actions)?;
    let vocab = PyDict::new(py);
    vocab.set_item("card", sts_core::cards::ALL_CARDS.iter().map(|c| *c as usize).max().unwrap_or(0) + 2)?;
    vocab.set_item("power", sts_core::powers::PowerId::LAST as usize + 2)?;
    vocab.set_item("monster", sts_core::monsters::MonsterId::CorruptHeart as usize + 2)?;
    vocab.set_item("relic", sts_core::relics::RELIC_DATA.len() + 1)?;
    vocab.set_item("potion", sts_core::potions::POTION_DATA.len() + 1)?;
    vocab.set_item("intent", 18)?;
    vocab.set_item("choice_kind", encode::CHOICE_KINDS.len() + 1)?;
    vocab.set_item("choice_pile", 6)?;
    vocab.set_item("character", 5)?;
    for (k, n) in encode::vocab_sizes().into_iter().chain(encode_run::vocab_sizes()) {
        vocab.set_item(k, n)?;
    }
    d.set_item("vocab", vocab)?;
    Ok(d)
}

/// Encounter keys grouped by act (1-4), event fights included.
#[pyfunction]
fn encounters_by_act() -> Vec<Vec<String>> {
    let v = |l: Vec<&str>| l.into_iter().map(|s| s.to_string()).collect();
    vec![
        v(encounters::all_act1()),
        v(encounters::all_act2()),
        v(encounters::all_act3()),
        v(vec!["Shield and Spear", "The Heart"]),
    ]
}

/// Every encounter key the engine can build.
#[pyfunction]
fn all_encounters() -> Vec<String> {
    encounters::all().iter().map(|s| s.to_string()).collect()
}

/// A character's engines (`practice::archetypes_for`): the names its engine practice decks, seeded runs and deck stats use.
#[pyfunction]
fn archetypes(character: &str) -> PyResult<Vec<String>> {
    let c = Character::from_name(character).ok_or_else(|| err(format!("unknown character {character:?}")))?;
    Ok(practice::archetypes_for(c).iter().map(|a| a.name.to_string()).collect())
}

/// Encodes human decision records (JSON strings from `python/human/human_runs.py`) as the run network
/// sees them: returns (ints bytes, floats bytes, mask bytes, the action slot the player took per
/// sample, the record each sample came from, and (record, reason) for records left out). A rest
/// site's smith or toke gives two samples (the choice, then the card).
/// The deck network's lookup on human card decisions (`human_runs.py` records of kind "card"; others give
/// none): each record's run rebuilt at its card reward (`human::samples`), probed as `Envs.probe_card_rewards`
/// probes a game. Returns (meta as (record, option, fight, draw), ints, floats); a record the rebuild can't
/// express gives none.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn probe_human_cards<'py>(py: Python<'py>, records: Vec<String>, draws: u32, seed: i64)
    -> PyResult<(Vec<(usize, i32, u8, u32)>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let probes: Vec<(usize, probe::Probe, Run)> = py.detach(|| {
        records.par_iter().enumerate().flat_map_iter(|(k, r)| {
            let rec: Option<serde_json::Value> = serde_json::from_str(r).ok();
            let runs = rec.filter(|v| v["kind"] == "card").and_then(|v| human::samples(&v).ok()).unwrap_or_default();
            runs.into_iter().flat_map(move |(run, _)| probe::card_reward(&run, draws, seed).into_iter().map(move |(p, x)| (k, p, x)))
        }).collect()
    });
    let (meta, ints, floats) = encode_probes(py, probes);
    Ok((meta, PyBytes::new(py, le_bytes(&ints)), PyBytes::new(py, le_bytes(&floats))))
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn encode_human<'py>(
    py: Python<'py>,
    records: Vec<String>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<usize>, Vec<usize>, Vec<(usize, String)>)> {
    let (ni, nf, na) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len(), encode::ACTIONS);
    let rows: Vec<Result<Vec<(Vec<i32>, Vec<f32>, Vec<u8>, usize)>, String>> = py.detach(|| {
        records
            .par_iter()
            .map(|r| {
                let rec: serde_json::Value = serde_json::from_str(r).map_err(|e| e.to_string())?;
                let mut out = Vec::new();
                for (run, target) in human::samples(&rec)? {
                    let (mut i, mut f, mut m) = (vec![0i32; ni], vec![0f32; nf], vec![0u8; na]);
                    let slot = human::encode(&run, &target, &mut i, &mut f, &mut m).ok_or_else(|| format!("{target:?} not allowed in the rebuilt state"))?;
                    out.push((i, f, m, slot));
                }
                Ok(out)
            })
            .collect()
    });
    let (mut ib, mut fb, mut mb, mut slots, mut source, mut skipped) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (k, r) in rows.into_iter().enumerate() {
        match r {
            Ok(samples) => {
                for (i, f, m, slot) in samples {
                    ib.extend_from_slice(le_bytes(&i));
                    fb.extend_from_slice(le_bytes(&f));
                    mb.extend(m);
                    slots.push(slot);
                    source.push(k);
                }
            }
            Err(e) => skipped.push((k, e)),
        }
    }
    Ok((PyBytes::new(py, &ib), PyBytes::new(py, &fb), PyBytes::new(py, &mb), slots, source, skipped))
}

/// The in-game advisor: a CommunicationMod `game_state` (JSON) and its available commands as the
/// run network's inputs (ints, floats, mask as `observe_arrays` lays them out for one game) and
/// a label for every allowed action slot (see `advise`).
#[pyfunction]
#[pyo3(name = "advise")]
fn advise_state<'py>(py: Python<'py>, game_state: &str, available: Vec<String>) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<(usize, String)>)> {
    let gs: serde_json::Value = serde_json::from_str(game_state).map_err(|e| err(e.to_string()))?;
    let a = advise::advise(&gs, &available).map_err(err)?;
    Ok((PyBytes::new(py, le_bytes(&a.ints)), PyBytes::new(py, le_bytes(&a.floats)), PyBytes::new(py, &a.mask), a.labels))
}

#[pymodule]
fn sts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Envs>()?;
    m.add_function(wrap_pyfunction!(all_encounters, m)?)?;
    m.add_function(wrap_pyfunction!(archetypes, m)?)?;
    m.add_function(wrap_pyfunction!(layout, m)?)?;
    m.add_function(wrap_pyfunction!(scenario_problem, m)?)?;
    m.add_function(wrap_pyfunction!(encounters_by_act, m)?)?;
    m.add_function(wrap_pyfunction!(encode_human, m)?)?;
    m.add_function(wrap_pyfunction!(probe_human_cards, m)?)?;
    m.add_function(wrap_pyfunction!(advise_state, m)?)?;
    m.add("ENCODING_VERSION", encode::ENCODING_VERSION)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fight's monsters report the move their intent stands for, in the encoding's slot order;
    /// a game between fights reports none.
    #[test]
    fn monster_next_moves_follow_the_slots() {
        let mut s = Settings { fight: true, seed: 3, ascension: 0, encounters: vec!["2 Louse".into()], random_decks: false, final_act: true, n: 1, max_turns: 100, max_steps: 20000,
                               record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 0.0, seeded: 0.0, practice_archetype: None, practice_ascension_range: None, practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0,
                               scenarios: Default::default(), scenario_share: 0.0, scenario_sequential: false, character: Character::Ironclad, practice_decks: None, practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let envs = Envs { slots: vec![Slot::new(&s, 0, 0).unwrap()], settings: s.clone(), playouts: Vec::new() };
        let moves = envs.monster_next_moves();
        let Game::Fight(c, _) = &envs.slots[0].game else { unreachable!() };
        let slots = encode::monster_slots(c);
        assert_eq!(moves.len(), encode::MAX_MONSTERS);
        for (j, &m) in slots.iter().enumerate() {
            assert_eq!(moves[j], encode::move_id(c.monsters[m].id, *c.monsters[m].move_history.last().unwrap()));
        }
        assert!(moves[slots.len()..].iter().all(|&x| x == 0));
        s.fight = false;
        let envs = Envs { slots: vec![Slot::new(&s, 0, 0).unwrap()], settings: s, playouts: Vec::new() };
        assert!(envs.monster_next_moves().iter().all(|&x| x == 0));
    }

    /// The archetype counts a step reports: with no archetype in the deck, a core card of any starts one (in) and anything
    /// else is off; once Rupture leads, its support is in and an unrelated card off; complete at two core and four support.
    #[test]
    fn habits_report_cards_for_the_leading_archetype() {
        use sts_core::cards::CardId::*;
        let mut r = Run::new(RunConfig::new(1));
        let add = |r: &mut Run, ids: &[sts_core::cards::CardId]| {
            let b = habit_before(r, &r.legal_actions());
            for &id in ids {
                let c = r.new_card(id);
                r.deck.push(c);
            }
            Habit::of(&b, &RunAction::Proceed, r)
        };
        let h = add(&mut r, &[Rupture, Anger]);
        assert_eq!((h.arch_in, h.arch_off, h.arch_complete), (1, 1, false));
        assert_eq!(leading_archetype(r.config.character, &r.deck.iter().map(|c| c.id).collect::<Vec<_>>()).unwrap().0.name, "self-damage");
        let h = add(&mut r, &[Offering, Hemokinesis, Whirlwind]);
        assert_eq!((h.arch_in, h.arch_off, h.arch_complete), (2, 1, false));
        let h = add(&mut r, &[Combust, Reaper, Feed]);
        assert_eq!((h.arch_in, h.arch_off, h.arch_complete), (3, 0, false));
        let h = add(&mut r, &[Clothesline]);
        assert_eq!((h.arch_in, h.arch_off, h.arch_complete), (0, 1, true));
        assert_eq!(Habit::FIELDS, h.row().len());
    }

    /// The habits a step reports: a card added (and its deck size), a card removed, a rest at full
    /// HP, a Recall without the Sapphire key.
    #[test]
    fn habits_report_deck_changes_rests_and_recalls() {
        let mut r = Run::new(RunConfig::new(1));
        let n = r.deck.len() as u32;
        let b = habit_before(&r, &r.legal_actions());
        let c = r.new_card(sts_core::cards::CardId::Inflame);
        r.deck.push(c);
        let h = Habit::of(&b, &RunAction::Proceed, &r);
        assert_eq!((h.added, h.removed, h.deck), (1, 0, n + 1));
        let b = habit_before(&r, &r.legal_actions());
        r.deck.remove(0);
        let h = Habit::of(&b, &RunAction::Proceed, &r);
        assert_eq!((h.added, h.removed, h.deck), (0, 1, n));

        r.screen = Screen::Rest;
        r.room = Some(RoomType::Rest);
        r.hp = r.max_hp;
        let b = habit_before(&r, &r.legal_actions());
        r.step(&RunAction::Rest).unwrap();
        let h = Habit::of(&b, &RunAction::Rest, &r);
        assert_eq!(h.rest_hp, Some(1.0));
        assert!(!h.recall_wasted);

        r.screen = Screen::Rest;
        let b = habit_before(&r, &r.legal_actions());
        r.step(&RunAction::Recall).unwrap();
        let h = Habit::of(&b, &RunAction::Recall, &r);
        assert!(h.recall_wasted && h.rest_hp.is_none());

        // Upgrading at a rest site: Smith, then the card on the grid.
        r.screen = Screen::Rest;
        r.step(&RunAction::Smith).unwrap();
        let b = habit_before(&r, &r.legal_actions());
        let Screen::Grid { options, .. } = &r.screen else { panic!("the upgrade grid") };
        let pick = RunAction::Select { indices: vec![options[0]] };
        r.step(&pick).unwrap();
        let h = Habit::of(&b, &pick, &r);
        assert!(h.smithed);
        assert_eq!((h.added, h.removed), (0, 0));

        // Leaving a shop whose removal it could pay for (the Before a shop with Purge legal gives).
        let b = Before { removal_open: true, ..habit_before(&r, &[]) };
        assert!(Habit::of(&b, &RunAction::Proceed, &r).removal_skipped);
        assert!(!Habit::of(&b, &RunAction::Purge, &r).removal_skipped);
        assert!(!habit_before(&r, &[RunAction::Purge]).removal_open, "not in a shop");
        assert_eq!(h.row().len(), Habit::FIELDS);
    }

    /// Seeded Watcher runs start with one of the two seeds on top of the starting deck: Mental
    /// Fortress+ and Rushdown (stance), or Deva Form+ and Devotion+ (mantra); `seed_names` keeps
    /// them to the ones named.
    #[test]
    fn watcher_seeded_runs_start_with_an_engine() {
        use sts_core::cards::CardId;
        let s = Settings { fight: false, seed: 3, ascension: 20, encounters: Vec::new(), random_decks: false, final_act: true, n: 1, max_turns: 100, max_steps: 20000,
                           record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 0.0, seeded: 1.0, practice_archetype: None, practice_ascension_range: None, practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0,
                           scenarios: Default::default(), scenario_share: 0.0, scenario_sequential: false, character: Character::Watcher, practice_decks: None, practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let mut seen = std::collections::HashSet::new();
        for episode in 0..20 {
            let slot = Slot::new(&s, 0, episode).unwrap();
            let name = slot.seeded.expect("seeded");
            seen.insert(name);
            let Game::Run(r, _) = &slot.game else { unreachable!() };
            let plain = sts_core::run::Run::new({
                let mut c = sts_core::run::RunConfig::new(slot.seed);
                c.ascension = 20;
                c.character = Character::Watcher;
                c
            });
            assert_eq!(r.deck.len(), plain.deck.len() + 2);
            let (a, b) = if name == "stance" { (CardId::MentalFortress, CardId::Rushdown) } else { (CardId::DevaForm, CardId::Devotion) };
            assert!(r.deck.iter().any(|c| c.id == a && c.upgraded), "{name}");
            assert!(r.deck.iter().any(|c| c.id == b && c.upgraded == (name == "mantra")), "{name}");
            assert_eq!(synergy_cards(r, name), 2);
        }
        assert_eq!(seen.len(), 2, "both seeds in 20 runs");
        let only = Settings { seed_names: Some(vec!["mantra".to_string()]), ..s };
        for episode in 0..10 {
            assert_eq!(Slot::new(&only, 0, episode).unwrap().seeded, Some("mantra"));
        }
    }

    /// Random seeded runs: the archetype cards credited step by step add up to the run's
    /// `synergy_picks`, stay within the cap, and some runs pick some up.
    #[test]
    fn seeded_runs_count_synergy_picks() {
        let s = Settings { fight: false, seed: 7, ascension: 0, encounters: Vec::new(), random_decks: false, final_act: true, n: 1, max_turns: 100, max_steps: 20000,
                           record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 0.0, seeded: 1.0, practice_archetype: None, practice_ascension_range: None, practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0,
                           scenarios: Default::default(), scenario_share: 0.0, scenario_sequential: false, character: Character::Ironclad, practice_decks: None, practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let mut roll = StsRandom::new(1);
        let mut credited_runs = 0;
        for episode in 0..20 {
            let mut slot = Slot::new(&s, 0, episode).unwrap();
            assert!(slot.seeded.is_some());
            let mut sum = 0;
            loop {
                let Game::Run(_, legal) = &slot.game else { unreachable!() };
                let k = roll.random(legal.len() as i32 - 1) as usize;
                let status = slot.step(&s, k).unwrap();
                sum += slot.synergy;
                if status != 0 {
                    break;
                }
            }
            let picks = slot.last.as_ref().unwrap()["synergy_picks"].as_u64().unwrap() as u32;
            assert_eq!(sum, picks);
            assert!(picks <= SYNERGY_CAP);
            // The final deck's archetype cards include the seed's own.
            let last = slot.last.as_ref().unwrap();
            assert!(last["archetype_cards"][last["seeded"].as_str().unwrap()].as_u64().unwrap() >= 2);
            credited_runs += (picks > 0) as u32;
        }
        assert!(credited_runs > 0);
    }

    /// With an ascension range and a low share, about that share of runs play below the range and
    /// the rest in it; without the share every run plays in it.
    #[test]
    fn low_ascension_share_plays_below_the_range() {
        let mut s = Settings { fight: false, seed: 13, ascension: 0, encounters: Vec::new(), random_decks: false, final_act: true, n: 1, max_turns: 100, max_steps: 20000,
                               record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 0.0, seeded: 0.0, practice_archetype: None, practice_ascension_range: None, practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1,
                               ascension_range: Some((17, 20)), ascension_low_share: 0.25, scenarios: Default::default(), scenario_share: 0.0, scenario_sequential: false, character: Character::Silent, practice_decks: None, practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let ascensions = |s: &Settings| (0..200).map(|episode| match &Slot::new(s, 0, episode).unwrap().game {
            Game::Run(r, _) => r.config.ascension,
            _ => unreachable!(),
        }).collect::<Vec<_>>();
        let low = ascensions(&s).iter().filter(|&&a| a < 17).count();
        assert!((25..=75).contains(&low), "{low} of 200 below A17");
        assert!(ascensions(&s).iter().all(|&a| (0..=20).contains(&a)));
        s.ascension_low_share = 0.0;
        assert!(ascensions(&s).iter().all(|&a| (17..=20).contains(&a)));
    }

    /// The Silent's practice fights are her own human boss scenarios: played as the Silent, in
    /// the boss fight; before any are loaded they're plain runs. Her seeded runs start with one of
    /// her engines.
    #[test]
    fn silent_practice_fights_are_her_boss_scenarios() {
        let line = r#"{"character": "THE_SILENT", "ascension": 20, "act": 2, "floor": 33, "hp": 50, "max_hp": 70, "bosses": ["Champ"],
            "deck": [["Strike_G", 0], ["Defend_G", 1], ["Neutralize", 1], ["Survivor", 0], ["Noxious Fumes", 1], ["Catalyst", 0]],
            "relics": ["Ring of the Snake"], "potions": ["Poison Potion"]}"#;
        let mut s = Settings { fight: false, seed: 5, ascension: 20, encounters: Vec::new(), random_decks: false, final_act: true, n: 4, max_turns: 100, max_steps: 20000,
                               record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 1.0, seeded: 0.0, practice_archetype: None, practice_ascension_range: None, practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0,
                               scenarios: Default::default(), scenario_share: 1.0, scenario_sequential: false, character: Character::Silent, practice_decks: None, practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        // No scenarios yet: plain Silent runs.
        let slot = Slot::new(&s, 0, 0).unwrap();
        assert!(slot.practice.is_none());
        let Game::Run(r, _) = &slot.game else { unreachable!() };
        assert_eq!(r.config.character, Character::Silent);
        assert!(r.deck.iter().all(|c| c.id.data().name != "Bash"));
        // With hers loaded: her boss fight.
        let sc = scenarios::Scenario::parse(line).unwrap();
        assert_eq!(sc.character, Character::Silent);
        s.scenarios = std::sync::Arc::new(vec![sc]);
        let slot = Slot::new(&s, 1, 0).unwrap();
        assert_eq!(slot.practice, Some("boss2"));
        let Game::Run(r, _) = &slot.game else { unreachable!() };
        assert_eq!(r.config.character, Character::Silent);
        assert!(matches!(r.screen, Screen::Combat));
        assert!(r.combat.as_ref().unwrap().hand.len() + r.combat.as_ref().unwrap().draw_pile.len() >= 6);
        // Seeded runs start with one of her engines (never an Ironclad card) and count its cards.
        s.practice = 0.0;
        s.seeded = 1.0;
        for episode in 0..8 {
            let slot = Slot::new(&s, 0, episode).unwrap();
            let name = slot.seeded.unwrap();
            let Game::Run(r, _) = &slot.game else { unreachable!() };
            assert!(r.deck.iter().all(|c| c.id.data().color != sts_core::cards::CardColor::Red), "{name}");
            assert!(synergy_cards(r, name) >= 2, "{name}: {}", synergy_cards(r, name));
        }
        // The control seed comes only when named: Dash+ and Predator, no engine's cards to count.
        for episode in 0..64 {
            assert_ne!(Slot::new(&s, 0, episode).unwrap().seeded, Some(CONTROL_SEED));
        }
        s.seed_names = Some(vec![CONTROL_SEED.to_string()]);
        let slot = Slot::new(&s, 0, 0).unwrap();
        assert_eq!(slot.seeded, Some(CONTROL_SEED));
        let Game::Run(r, _) = &slot.game else { unreachable!() };
        assert!(r.deck.iter().any(|c| c.id == sts_core::cards::CardId::Dash && c.times_upgraded == 1));
        assert!(r.deck.iter().any(|c| c.id == sts_core::cards::CardId::Predator && c.times_upgraded == 0));
        assert_eq!(synergy_cards(r, CONTROL_SEED), 0);
        s.seed_names = None;
        // Cards instead of a named seed, and the three keys from the start.
        s.seed_cards = Some(vec![(sts_core::cards::CardId::Catalyst, true)]);
        s.free_keys = true;
        let slot = Slot::new(&s, 0, 0).unwrap();
        assert_eq!(slot.seeded, Some(CUSTOM_SEED));
        let Game::Run(r, _) = &slot.game else { unreachable!() };
        assert!(r.deck.iter().any(|c| c.id == sts_core::cards::CardId::Catalyst && c.times_upgraded == 1));
        assert!(r.keys.ruby && r.keys.emerald && r.keys.sapphire);
        s.seed_cards = None;
        s.free_keys = false;
        // A seed's cards can wait for an act's first map screen.
        s.seed_names = Some(vec![CONTROL_SEED.to_string()]);
        s.seed_act = 2;
        let mut slot = Slot::new(&s, 0, 0).unwrap();
        let has_dash = |slot: &Slot| matches!(&slot.game, Game::Run(r, _) if r.deck.iter().any(|c| c.id == sts_core::cards::CardId::Dash));
        assert!(!has_dash(&slot));
        slot.give_late_seed(&s);
        assert!(!has_dash(&slot), "not in Act 1");
        if let Game::Run(r, _) = &mut slot.game {
            r.act = 2;
            r.screen = Screen::Map;
        }
        slot.give_late_seed(&s);
        assert!(has_dash(&slot) && slot.late_seed.is_empty() && slot.seeded == Some(CONTROL_SEED));
        s.seed_names = None;
        s.seed_act = 1;
        // A line's own seed sets the fight up the same in any game (the same draws and monsters).
        s.practice = 1.0;
        s.seeded = 0.0;
        s.scenario_sequential = true;
        s.scenarios = std::sync::Arc::new((0..2).map(|_| scenarios::Scenario::parse(&line.replacen('{', r#"{"seed": 77, "#, 1)).unwrap()).collect());
        let draws = |slot: Slot| {
            let Game::Run(r, _) = &slot.game else { unreachable!() };
            let c = r.combat.as_ref().unwrap();
            (c.hand.iter().chain(&c.draw_pile).map(|x| x.id).collect::<Vec<_>>(), c.monsters[0].hp)
        };
        let (a, b) = (Slot::new(&s, 0, 0).unwrap(), Slot::new(&s, 1, 0).unwrap());
        assert_ne!(a.seed, b.seed);
        assert_eq!(draws(a), draws(b));
        s.scenario_sequential = false;
        // Her Act 1 boss and elite fights too.
        s.practice = 1.0;
        s.seeded = 0.0;
        for (line, kind) in [(line.replace(r#""act": 2, "floor": 33"#, r#""act": 1, "floor": 16"#).replace("Champ", "Hexaghost"), "boss1"),
                             (line.replace(r#""act": 2, "floor": 33"#, r#""act": 1, "room": "elite", "floor": 6"#).replace("Champ", "Lagavulin"), "elite1"),
                             (line.replace(r#""floor": 33"#, r#""room": "elite", "floor": 25"#).replace("Champ", "Book of Stabbing"), "elite2")] {
            let sc = scenarios::Scenario::parse(&line).unwrap();
            assert_eq!(sc.kind(), kind);
            s.scenarios = std::sync::Arc::new(vec![sc]);
            let slot = Slot::new(&s, 1, 0).unwrap();
            assert_eq!(slot.practice, Some(kind));
            let Game::Run(r, _) = &slot.game else { unreachable!() };
            assert!(matches!(r.screen, Screen::Combat), "{kind}");
            assert_eq!(r.room, Some(if kind.starts_with("elite") { RoomType::Elite } else { RoomType::Boss }));
        }
        // Archetype practice fights (no scenarios as practice) are refused for her.
        s.practice = 1.0;
        s.seeded = 0.0;
        s.scenario_share = 0.5;
        assert!(Slot::new(&s, 0, 0).is_err());
    }

    /// A boss scenario with `play_on` isn't a practice fight: past its boss the same game goes on
    /// as a normal run, and its summary is a run's, with the scenario's index. Without it the
    /// episode ends with the boss, as ever.
    #[test]
    fn scenario_plays_on_as_a_normal_run_only_if_asked() {
        let line = r#"{"character": "THE_SILENT", "ascension": 20, "act": 1, "floor": 16, "hp": 60, "max_hp": 70, "bosses": ["Hexaghost"],
            "deck": [["Strike_G", 0], ["Defend_G", 1], ["Neutralize", 1], ["Survivor", 0]], "relics": ["Ring of the Snake"], "potions": []}"#;
        let sc = scenarios::Scenario::parse(line).unwrap();
        let settings = |sc: scenarios::Scenario| Settings { fight: false, seed: 5, ascension: 20, encounters: Vec::new(), random_decks: false, final_act: true, n: 1, max_turns: 100, max_steps: 20000,
                               record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 1.0, seeded: 0.0, practice_archetype: None, practice_ascension_range: None, practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0,
                               scenarios: std::sync::Arc::new(vec![sc]), scenario_share: 1.0, scenario_sequential: true, character: Character::Silent, practice_decks: None, practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        // Random moves against the boss at 1 HP, the player kept alive, until the episode ends or it's in Act 2: the
        // summary if it ended.
        let beat_the_boss = |s: &Settings| {
            let mut slot = Slot::new(s, 0, 0).unwrap();
            let mut rng = StsRandom::new(3);
            let Game::Run(r, _) = &mut slot.game else { unreachable!() };
            r.combat.as_mut().unwrap().monsters.iter_mut().for_each(|m| m.hp = 1);
            for _ in 0..5000 {
                let Game::Run(r, _) = &mut slot.game else { unreachable!() };
                if r.act == 2 {
                    return None;
                }
                if let Some(c) = r.combat.as_mut() {
                    c.player.hp = c.player.max_hp;
                }
                let n = slot.game.num_legal() as i32;
                if slot.step(s, rng.random(n - 1) as usize).unwrap() != 0 {
                    return slot.last.clone();
                }
            }
            panic!("neither ended nor reached Act 2");
        };
        // A practice fight: over with the boss.
        let s = settings(sc.clone());
        assert_eq!(Slot::new(&s, 0, 0).unwrap().practice, Some("boss1"));
        let summary = beat_the_boss(&s).unwrap();
        assert_eq!((summary["practice"].as_str(), summary["won"].as_bool(), summary["floor"].as_i64()), (Some("boss1"), Some(true), Some(16)));
        // Playing on: a normal run, in the same fight at the same HP, on to Act 2.
        let s = settings(scenarios::Scenario { play_on: true, ..sc });
        let slot = Slot::new(&s, 0, 0).unwrap();
        assert_eq!((slot.practice, slot.start_hp), (None, 0));
        let Game::Run(r, _) = &slot.game else { unreachable!() };
        assert!(matches!(r.screen, Screen::Combat) && r.config.final_act && !r.keys.ruby);
        assert_eq!((r.act, r.floor, r.hp, r.room), (1, 16, 60, Some(RoomType::Boss)));
        assert_eq!(beat_the_boss(&s), None);
        // Its summary at the end (here it dies): a run's, naming the scenario.
        let mut slot = Slot::new(&s, 0, 0).unwrap();
        let mut rng = StsRandom::new(3);
        let summary = loop {
            let n = slot.game.num_legal() as i32;
            if slot.step(&s, rng.random(n - 1) as usize).unwrap() != 0 {
                break slot.last.clone().unwrap();
            }
        };
        assert!(summary.get("practice").is_none() && summary.get("retry").is_none());
        assert!(summary["floor"].as_i64().unwrap() >= 16 && summary["keys"].is_u64());
        assert_eq!((summary["scenario"].as_u64(), summary["encounter"].as_str()), (Some(0), Some("Hexaghost")));
    }

    /// Late starts: of the games neither practice fights nor seeded, the first `late_share` start every episode from a
    /// random line, as normal runs named "late": no practice mask, no keys, no end with the fight, nothing credited for the deck
    /// they start with. The other games are as they'd be without them, seed by seed.
    #[test]
    fn late_starts_are_normal_runs_from_human_states() {
        use sts_core::cards::CardId;
        use sts_core::relics::RelicId;
        let line = |floor: i32, act: i32, boss: &str, hp: i32| {
            format!(r#"{{"character": "THE_SILENT", "ascension": 20, "act": {act}, "floor": {floor}, "hp": {hp}, "max_hp": 70, "bosses": ["{boss}"], "gold": 321, "purges": 1,
                "deck": [["Strike_G", 0], ["Defend_G", 1], ["Neutralize", 1], ["Survivor", 0], ["Noxious Fumes", 1], ["Catalyst", 0]], "relics": ["Ring of the Snake", "Vajra"], "potions": ["Poison Potion"]}}"#)
        };
        let lines: Vec<_> = [line(16, 1, "Hexaghost", 41), line(33, 2, "Champ", 52)].iter().map(|l| scenarios::Scenario::parse(l).unwrap()).collect();
        let plain = Settings { fight: false, seed: 5, ascension: 20, encounters: Vec::new(), random_decks: false, final_act: true, n: 8, max_turns: 100, max_steps: 20000,
                               record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 0.25, seeded: 0.25, practice_archetype: None, practice_ascension_range: None, practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0,
                               scenarios: Default::default(), scenario_share: 0.0, scenario_sequential: false, character: Character::Silent, practice_decks: Some("mix".into()), practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let s = Settings { late_starts: std::sync::Arc::new(lines), late_share: 0.5, ..plain.clone() };
        // Two games of practice, two seeded, and half of the four left: every episode alike.
        for episode in 0..12 {
            let of = |i| Slot::new(&s, i, episode).unwrap();
            assert!((0..2).all(|i| of(i).practice.is_some()));
            assert!((2..4).all(|i| of(i).seeded.is_some_and(|n| n != LATE_START)));
            assert!((4..6).all(|i| of(i).seeded == Some(LATE_START) && of(i).practice.is_none()));
            assert!((6..8).all(|i| of(i).seeded.is_none() && of(i).practice.is_none()));
        }
        // The other games aren't touched, nor anyone with no share.
        let view = |slot: &Slot| match &slot.game {
            Game::Run(r, _) => (slot.seed, serde_json::to_string(&r.observe()).unwrap()),
            _ => unreachable!(),
        };
        let none = Settings { late_share: 0.0, ..s.clone() };
        for i in 0..8 {
            assert_eq!(view(&Slot::new(&none, i, 3).unwrap()), view(&Slot::new(&plain, i, 3).unwrap()));
            if !(4..6).contains(&i) {
                assert_eq!(view(&Slot::new(&s, i, 3).unwrap()), view(&Slot::new(&plain, i, 3).unwrap()));
            }
        }
        // In the line's fight, with what it holds, a random line each episode.
        let mut floors = std::collections::BTreeMap::new();
        for episode in 0..40 {
            let slot = Slot::new(&s, 4, episode).unwrap();
            assert_eq!((slot.practice, slot.scenario, slot.start_hp, slot.encounter.as_str()), (None, None, 0, ""));
            let Game::Run(r, _) = &slot.game else { unreachable!() };
            assert!(matches!(r.screen, Screen::Combat) && r.config.final_act && r.config.character == Character::Silent);
            assert!(!r.keys.ruby && !r.keys.emerald && !r.keys.sapphire);
            let (act, hp, boss) = if r.floor == 16 { (1, 41, "Hexaghost") } else { (2, 52, "Champ") };
            assert_eq!((r.act, r.hp, r.max_hp, r.gold, r.purge_cost, r.room, r.boss.as_str()), (act, hp, 70, 321, 100, Some(RoomType::Boss), boss));
            assert_eq!(r.deck.len(), 6);
            assert!(r.deck.iter().any(|c| c.id == CardId::Catalyst) && r.relics.iter().any(|x| x.id == RelicId::Vajra));
            assert_eq!(synergy_cards(r, LATE_START), 0, "no card is credited as a pick");
            *floors.entry(r.floor).or_insert(0) += 1;
        }
        assert!(floors[&16] >= 10 && floors[&33] >= 10, "{floors:?}");
        // A start at a boss: past the boss, with the boss at 1 HP and the player kept alive, it goes on to the next act, and that
        // act's map has the burning elite (the final act on, no key held), as a run's has.
        let mut rng = StsRandom::new(3);
        for (floor, act, boss) in [(16, 1, "Hexaghost"), (33, 2, "Champ")] {
            let only = Settings { late_starts: std::sync::Arc::new(vec![scenarios::Scenario::parse(&line(floor, act, boss, 41)).unwrap()]), ..s.clone() };
            let mut slot = Slot::new(&only, 4, 0).unwrap();
            let Game::Run(r, _) = &mut slot.game else { unreachable!() };
            r.combat.as_mut().unwrap().monsters.iter_mut().for_each(|m| m.hp = 1);
            for step in 0.. {
                assert!(step < 5000, "never reached the next act");
                let Game::Run(r, _) = &mut slot.game else { unreachable!() };
                if r.act == act + 1 {
                    assert!(matches!(r.screen, Screen::Map) && slot.seeded == Some(LATE_START));
                    assert!(r.map.rows.iter().flatten().any(|n| n.has_emerald_key), "no burning elite in act {}", r.act);
                    break;
                }
                if let Some(c) = r.combat.as_mut() {
                    c.player.hp = c.player.max_hp;
                }
                let n = slot.game.num_legal() as i32;
                assert_eq!(slot.step(&only, rng.random(n - 1) as usize).unwrap(), 0, "it doesn't end with the fight");
                if step == 0 {
                    assert_eq!((slot.habit.added, slot.habit.removed), (0, 0), "the deck it started with isn't added to it");
                }
            }
        }
        // Played to its end (here it dies): a run's summary, named, and the game's next episode is a late start too.
        let mut slot = Slot::new(&s, 5, 0).unwrap();
        let summary = loop {
            let n = slot.game.num_legal() as i32;
            if slot.step(&s, rng.random(n - 1) as usize).unwrap() != 0 {
                break slot.last.clone().unwrap();
            }
        };
        assert_eq!(summary["seeded"].as_str(), Some(LATE_START));
        assert!(summary.get("practice").is_none() && summary.get("scenario").is_none() && summary.get("retry").is_none());
        assert!(summary["floor"].as_i64().unwrap() >= 16 && summary["keys"].is_u64() && summary["synergy_picks"] == 0);
        assert_eq!((slot.seeded, slot.episode), (Some(LATE_START), 1));
    }

    /// Upgrading a basic Strike on a smithing grid is counted (training penalises it); another
    /// card isn't.
    #[test]
    fn smithing_a_strike_is_counted() {
        for character in [Character::Ironclad, Character::Silent] {
            let mut config = RunConfig::new(1);
            config.character = character;
            let mut r = Run::new(config);
            let strike = r.deck.iter().position(|c| c.id.data().name == "Strike").unwrap();
            let other = r.deck.iter().position(|c| c.id.data().name != "Strike").unwrap();
            r.screen = Screen::Grid { purpose: sts_core::run::GridPurpose::Smith, options: (0..r.deck.len()).collect(), count: 1 };
            let b = habit_before(&r, &[]);
            assert_eq!(Habit::of(&b, &RunAction::Select { indices: vec![strike] }, &r).strikes_upgraded, 1);
            assert_eq!(Habit::of(&b, &RunAction::Select { indices: vec![other] }, &r).strikes_upgraded, 0);
        }
    }

    /// The action slots `encode_run` marks legal.
    fn legal_slots(r: &Run, legal: &[RunAction], picked: &[usize]) -> Vec<usize> {
        let (mut i, mut f, mut m) = (vec![0; encode::int_len() + encode_run::run_int_len()], vec![0.0; encode::float_len() + encode_run::run_float_len()], vec![0u8; encode::ACTIONS]);
        encode_run::encode_run(r, legal, picked, &[], &mut i, &mut f, &mut m);
        (0..encode::ACTIONS).filter(|&a| m[a] != 0).collect()
    }

    /// The look-ahead from fight decisions of random runs, its copies played on at random: every
    /// playout ends, ending the turn at once stops at the next turn's start, and the redrawn
    /// copies don't share a draw order.
    #[test]
    fn playouts_end_and_redraw() {
        let s = Settings { fight: false, seed: 11, ascension: 0, encounters: Vec::new(), random_decks: false, final_act: true, n: 1, max_turns: 100, max_steps: 20000,
                           record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 0.0, seeded: 0.0, practice_archetype: None, practice_ascension_range: None, practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0,
                           scenarios: Default::default(), scenario_share: 0.0, scenario_sequential: false, character: Character::Ironclad, practice_decks: None, practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let mut roll = StsRandom::new(3);
        let (mut searched, mut differ, mut playouts) = (0, 0, 0);
        let mut ends = [0usize; 4];
        for episode in 0..6 {
            let mut slot = Slot::new(&s, 0, episode).unwrap();
            loop {
                let Game::Run(r, legal) = &slot.game else { unreachable!() };
                if search::searchable(r) {
                    let firsts = legal_slots(r, legal, &[]);
                    assert!(firsts.contains(&encode::END_TURN));
                    let turn = r.combat.as_ref().unwrap().turn;
                    let mut ps = search::start(0, r, &firsts, 4, search::Horizon { turns: 1, max_moves: 200 }, episode as i64, None).unwrap();
                    searched += 1;
                    playouts += ps.len();
                    for p in &mut ps {
                        while p.end.is_none() {
                            let options = legal_slots(&p.run, &p.legal, &p.picked);
                            p.step(options[roll.random(options.len() as i32 - 1) as usize]).unwrap();
                        }
                        ends[p.end.unwrap().code() as usize] += 1;
                    }
                    // Ending the turn at once: the next turn's hand, from each copy's own shuffle.
                    let hands: Vec<Vec<i32>> = ps.iter().filter(|p| p.first as usize == encode::END_TURN).filter_map(|p| {
                        let c = p.run.combat.as_ref().filter(|_| p.end == Some(search::End::Leaf) && matches!(p.run.screen, Screen::Combat))?;
                        assert_eq!(c.turn, turn + 1);
                        Some(c.hand.iter().map(|x| x.id as i32).collect())
                    }).collect();
                    differ += hands.windows(2).any(|w| w[0] != w[1]) as usize;
                }
                let k = roll.random(legal.len() as i32 - 1) as usize;
                if slot.step(&s, k).unwrap() != 0 {
                    break;
                }
            }
        }
        eprintln!("{searched} searches, {playouts} playouts, ends {ends:?}");
        assert!(searched > 10);
        assert!(ends[0] > 0 && ends[1] > 0);
        assert!(differ > searched / 4, "copies drew the same hands: {differ} of {searched}");
    }

    /// The Defect's practice fights come from `practice::build_deck` with `practice_decks`: her
    /// cards, labelled by deck kind or engine, in the asked act; without it they are refused.
    #[test]
    fn defect_practice_fights_from_deck_kinds() {
        let mut s = Settings { fight: false, seed: 3, ascension: 10, encounters: Vec::new(), random_decks: false, final_act: true, n: 4, max_turns: 100, max_steps: 20000,
                               record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 1.0, seeded: 0.0, practice_archetype: None, practice_ascension_range: None,
                               practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0, scenarios: Default::default(), scenario_share: 0.0,
                               scenario_sequential: false, character: Character::Defect, practice_decks: Some("mix".into()), practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let engines: Vec<&str> = practice::DEFECT_ARCHETYPES.iter().map(|a| a.name).collect();
        let mut labels = std::collections::HashSet::new();
        for episode in 0..40 {
            let slot = Slot::new(&s, (episode % 4) as usize, episode).unwrap();
            let label = slot.practice.unwrap();
            assert!(label == "grown" || label == "random" || engines.contains(&label), "{label}");
            labels.insert(if engines.contains(&label) { "engine" } else { label });
            let Game::Run(r, _) = &slot.game else { unreachable!() };
            assert_eq!(r.config.character, Character::Defect);
            assert!(matches!(r.screen, Screen::Combat));
            assert!(r.deck.iter().all(|c| matches!(c.id.data().color, sts_core::cards::CardColor::Blue | sts_core::cards::CardColor::Colorless)), "{label}");
        }
        assert_eq!(labels.len(), 3, "{labels:?}");
        s.practice_decks = Some("engine".into());
        s.practice_act = Some(4);
        for episode in 0..8 {
            let slot = Slot::new(&s, 0, episode).unwrap();
            assert!(engines.contains(&slot.practice.unwrap()));
            let Game::Run(r, _) = &slot.game else { unreachable!() };
            assert_eq!(r.act, 4);
        }
        s.practice_decks = None;
        assert!(Slot::new(&s, 0, 0).is_err() || Slot::new(&s, 0, 0).unwrap().practice.is_none());
    }

    /// A practice fight left by a Smoke Bomb is a loss: no HP kept, nothing for escaping.
    #[test]
    fn a_smoke_bomb_escape_loses_a_practice_fight() {
        use sts_core::potions::PotionId;
        let s = Settings { fight: false, seed: 21, ascension: 0, encounters: Vec::new(), random_decks: false, final_act: true, n: 1, max_turns: 100, max_steps: 20000,
                           record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 1.0, seeded: 0.0, practice_archetype: None, practice_ascension_range: None,
                           practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0, scenarios: Default::default(), scenario_share: 0.0,
                           scenario_sequential: false, character: Character::Defect, practice_decks: Some("mix".into()), practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let mut escapes = 0;
        for episode in 0..10 {
            let mut slot = Slot::new(&s, 0, episode).unwrap();
            let Game::Run(r, legal) = &mut slot.game else { unreachable!() };
            r.combat.as_mut().unwrap().potions[0] = Some(PotionId::SmokeBomb);
            *legal = r.legal_actions();
            // Not where it can't be drunk (bosses).
            let Some(k) = legal.iter().position(|a| matches!(a, RunAction::Combat { action: PlayerAction::UsePotion { slot: 0, .. } })) else { continue };
            assert_eq!(slot.step(&s, k).unwrap(), 2, "an escape is a loss");
            let ep = slot.last.as_ref().unwrap();
            assert_eq!((ep["won"].as_bool(), ep["escaped"].as_bool(), ep["hp"].as_i64()), (Some(false), Some(true), Some(0)));
            assert!(ep["hp_escaped"].as_i64().unwrap() > 0, "the HP it escaped with is kept apart");
            assert!(ep["hp_start"].as_i64().unwrap() >= 1, "the fight's starting HP: {ep}");
            escapes += 1;
        }
        assert!(escapes > 5, "{escapes} escapes");
    }

    /// With `retry_losses`, a lost practice fight comes back as the same fight (seed, deck, relics,
    /// potions, HP, encounter) until a replay wins or the replays run out; a won one, or one
    /// replayed enough, gives way to a new fight. With `retry_reseed` a replay draws otherwise;
    /// without, as the fight did.
    #[test]
    fn lost_practice_fights_are_played_again() {
        for reseed in [false, true] {
            let s = Settings { fight: false, seed: 9, ascension: 20, encounters: Vec::new(), random_decks: false, final_act: true, n: 1, max_turns: 100, max_steps: 20000,
                               record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 1.0, seeded: 0.0, practice_archetype: None, practice_ascension_range: None,
                               practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0, scenarios: Default::default(), scenario_share: 0.0,
                               scenario_sequential: false, character: Character::Ironclad, practice_decks: Some("mix".into()), practice_act: None, retry_losses: 2, retry_reseed: reseed, late_starts: Default::default(), late_share: 0.0 };
            let fight = |sl: &Slot| match &sl.game {
                Game::Run(r, _) => (sl.seed, sl.encounter.clone(), sl.start_hp, r.deck.iter().map(|c| c.name().to_string()).collect::<Vec<_>>(),
                                    r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), r.potions.clone()),
                _ => unreachable!(),
            };
            // The fight's draws and monster rolls: the hand and the draw pile at the start, the monsters' HP.
            let draws = |sl: &Slot| match &sl.game {
                Game::Run(r, _) => {
                    let c = r.combat.as_ref().unwrap();
                    (c.hand.iter().chain(c.draw_pile.iter()).map(|x| x.name().to_string()).collect::<Vec<_>>(), c.monsters.iter().map(|m| m.hp).collect::<Vec<_>>())
                }
                _ => unreachable!(),
            };
            let mut slot = Slot::new(&s, 0, 0).unwrap();
            let mut pick = StsRandom::new(4);
            let (mut replayed, mut gave_way, mut drew_otherwise) = (0, 0, 0);
            // Each fight as it started.
            let (mut before, mut before_draws) = (fight(&slot), draws(&slot));
            while replayed < 6 || gave_way < 4 {
                let retry = slot.retry;
                let n = match &slot.game { Game::Run(_, l) => l.len(), _ => unreachable!() };
                let status = slot.step(&s, pick.random(n as i32 - 1) as usize).unwrap();
                if status == 0 {
                    continue;
                }
                let started = (fight(&slot), draws(&slot));
                assert_eq!(slot.last.as_ref().unwrap()["retry"].as_u64(), Some(retry as u64));
                if status == 2 && retry < 2 {
                    assert_eq!(fight(&slot), before, "a lost fight is played again with the same setup");
                    assert_eq!(slot.retry, retry + 1);
                    drew_otherwise += (draws(&slot) != before_draws) as u32;
                    replayed += 1;
                } else {
                    assert_ne!(slot.seed, before.0, "a new fight after a win or the last replay");
                    assert_eq!(slot.retry, 0);
                    gave_way += 1;
                }
                (before, before_draws) = started;
            }
            if reseed {
                assert!(drew_otherwise * 2 > replayed, "replays draw otherwise: {drew_otherwise} of {replayed}");
            } else {
                assert_eq!(drew_otherwise, 0, "replays draw as the fight did");
            }
        }
    }

    /// Past `MAX_TURN_MOVES` moves in a combat turn, ending the turn is the only legal move.
    #[test]
    fn a_turn_past_its_move_cap_can_only_end() {
        let s = Settings { fight: false, seed: 5, ascension: 0, encounters: Vec::new(), random_decks: false, final_act: true, n: 1, max_turns: 100, max_steps: 20000,
                           record: false, harness: false, detail: false, forced_purge: 0.0, forced_purge_curse: 0.0, practice: 1.0, seeded: 0.0, practice_archetype: None, practice_ascension_range: None,
                           practice_boss: None, practice_focus: None, seed_names: None, seed_cards: None, free_keys: false, key_rule: None, seed_act: 1, ascension_range: None, ascension_low_share: 0.0, scenarios: Default::default(), scenario_share: 0.0,
                           scenario_sequential: false, character: Character::Defect, practice_decks: Some("mix".into()), practice_act: None, retry_losses: 0, retry_reseed: false, late_starts: Default::default(), late_share: 0.0 };
        let end = RunAction::Combat { action: PlayerAction::EndTurn };
        let mut slot = (0..20)
            .map(|e| Slot::new(&s, 0, e).unwrap())
            .find(|sl| matches!(&sl.game, Game::Run(_, legal) if legal.len() > 1 && legal.contains(&end)))
            .unwrap();
        let Game::Run(r, _) = &slot.game else { unreachable!() };
        slot.turn_at = (r.floor, r.combat.as_ref().unwrap().turn);
        slot.turn_moves = MAX_TURN_MOVES - 2;
        slot.cap_turn();
        let Game::Run(_, legal) = &slot.game else { unreachable!() };
        assert!(legal.len() > 1, "one move under the cap");
        slot.cap_turn();
        let Game::Run(_, legal) = &slot.game else { unreachable!() };
        assert_eq!(legal, &vec![end.clone()]);
        // Ending the turn starts the next turn's count again (or ends the fight).
        slot.step(&s, 0).unwrap();
        assert!(slot.turn_moves <= 1, "{}", slot.turn_moves);
    }
}
