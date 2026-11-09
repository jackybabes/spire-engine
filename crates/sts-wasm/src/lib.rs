//! The engine for the browser (milestone 6, `web/`): `sts-core` compiled to WebAssembly behind a
//! JSON interface, like `sts-cli serve`. No wasm-bindgen: JavaScript writes a command into memory
//! from `alloc`, calls `call(ptr, len)` and reads the reply (JSON) at `out_ptr()`, `call`'s
//! return value long (`web/engine.js` does this).
//!
//! A session is a run: its config and its decisions. Any step is reached by replaying them from
//! the seed (`Run` snapshots every `SNAPSHOT` steps keep that quick), so stepping back costs
//! nothing extra and a new decision part-way through simply drops the ones after it.
//!
//! Commands (`{"cmd": ...}`):
//! - `new_run`: `seed` (number or game seed string), `ascension`, `character`, `final_act`,
//!   `neow_full` (defaults: 0, "ironclad", true, true).
//! - `load`: `log`, a saved run (`sts-cli run --save`, `python/watch/watch_runs.py`,
//!   `python/watch/soak_runs.py`'s logs): replays it and stands at its start. `by` (optional): who took
//!   its decisions when the run doesn't say (e.g. "network").
//! - `goto`: `index`, stand before decision `index` (0 to the number of decisions).
//! - `step`: `index` into the legal actions (or `action`, a `RunAction`): takes it here, dropping
//!   any later decisions.
//! - `view`: the current position. `log`: every decision with its act, floor, map position, HP,
//!   label, action and who took it (`by`: "you" for the page's own, else the saved run's).
//! - `save`: the run up to here in the saved-run format (`tools/trace/watch.sh` plays it);
//!   `all`: every decision, also those after here.
//! - `names`: display names by game id, for relics, potions, cards and monsters (needs no run).
//! - `info`: what tooltips are built from (needs no run): every card's type, rarity, colour, target,
//!   cost, damage, block, magic number and keywords, fresh and upgraded; relic tiers and classes;
//!   potion rarities, potencies and targets; monster names.
//! - `net`: the network's weights file's metadata (`null` before `load_net`).
//! - `advise`: the network's `k` (default 3) likeliest decisions here, each a legal action's
//!   `index` with its `label` and `probability` (a card selection's first pick's, its later picks
//!   the network's own; decisions that read the same, like two copies of a card, count as one),
//!   its run `value`, and `pick`, the legal action the network itself takes here (`net_step`'s: its
//!   likeliest single action slot, which in a close call can differ from the likeliest decision).
//! - `net_step`: the network takes up to `count` (default 1) decisions from here, dropping any
//!   later ones, each `by` "network"; replies with the view and how many it `took`.
//!
//! The network (`crates/sts-net`) comes from a release weights file (`python/release/export_release.py`),
//! handed over as bytes: `alloc` room for them, write them, `load_net(ptr, len)` (its reply, the
//! metadata or an error, is at `out_ptr()` as for `call`).
//!
//! Every reply but `log`, `save` and `names` is the view: the step, the number of decisions, the
//! run's observation (`Run::observe`), the relics with their counters, the potions held now, and
//! each legal action with its label, where the last decision leaves the run (`ending`: act,
//! floor, outcome), a pending choice's options as pile indices (`choice_options`), and in a fight the draw (in name order), discard and exhaust
//! piles.

use std::cell::RefCell;

use serde_json::{json, Value};
use sts_core::character::Character;
use sts_core::commands::{parse_seed, run_step, seed_string};
use sts_core::combat::ChoicePile;
use sts_core::run::observe::{RewardView, ScreenView};
use sts_core::run::{Run, RunAction, RunConfig, Screen};
use sts_core::PlayerAction;
use sts_encode::agent::Skipped;
use sts_encode::encode_run;
use sts_net::{play, Net};

/// A snapshot of the run every this many decisions.
const SNAPSHOT: usize = 64;

pub struct Session {
    config: RunConfig,
    actions: Vec<RunAction>,
    /// Who took each decision: "you" (in this page), or what the saved run says ("network", ...).
    by: Vec<String>,
    /// Where the last decision leaves the run: act, floor and outcome.
    ending: Value,
    /// Per decision, from the run before it: act, floor, map position, HP, label and action.
    log: Vec<Value>,
    cursor: usize,
    run: Run,
    /// `snapshots[k]`: the run before decision `k * SNAPSHOT`.
    snapshots: Vec<Run>,
    /// Per decision: the card reward it skipped (`encode_run::card_reward_key`), and whether it
    /// left the reward screens. The network sees the card rewards skipped on the current reward
    /// screen (`Session::skipped`).
    skip_keys: Vec<Option<Vec<(i32, i32)>>>,
    left_rewards: Vec<bool>,
}

fn potion_name(p: Option<sts_core::potions::PotionId>) -> String {
    p.map_or("an empty slot".into(), |p| p.data().name.to_string())
}

/// Words from a Rust name: `JawWorm` -> "Jaw Worm", `HundredGold / None` -> "Hundred Gold / None".
fn words(name: &str) -> String {
    let mut out = String::new();
    let mut prev: Option<char> = None;
    for ch in name.chars() {
        if ch.is_ascii_uppercase() && prev.is_some_and(|p| p.is_ascii_lowercase() || p.is_ascii_digit()) {
            out.push(' ');
        }
        out.push(ch);
        prev = Some(ch);
    }
    out
}

/// Where a map room leads, in words.
fn room_name(r: sts_core::run::map::RoomType) -> &'static str {
    use sts_core::run::map::RoomType::*;
    match r {
        Monster => "a fight",
        Elite => "an elite",
        Event => "an unknown room",
        Rest => "a rest site",
        Shop => "a shop",
        Treasure => "a treasure room",
        Boss => "the boss",
    }
}

/// One of Neow's blessings in words.
fn neow_text(o: &sts_core::run::NeowOption) -> String {
    use sts_core::run::{NeowDrawback as D, NeowReward as R};
    let reward = match o.reward {
        R::RandomColorless2 => "choose a rare colourless card",
        R::ThreeCards => "choose a card for your deck",
        R::OneRandomRareCard => "get a random rare card",
        R::RemoveCard => "remove a card",
        R::UpgradeCard => "upgrade a card",
        R::RandomColorless => "choose a colourless card",
        R::TransformCard => "transform a card",
        R::ThreeSmallPotions => "get 3 random potions",
        R::RandomCommonRelic => "get a random common relic",
        R::TenPercentHpBonus => "gain 10% max HP",
        R::HundredGold => "gain 100 gold",
        R::ThreeEnemyKill => "the enemies of your next 3 fights start at 1 HP",
        R::RemoveTwo => "remove 2 cards",
        R::TransformTwoCards => "transform 2 cards",
        R::OneRareRelic => "get a random rare relic",
        R::ThreeRareCards => "choose a rare card",
        R::TwoFiftyGold => "gain 250 gold",
        R::TwentyPercentHpBonus => "gain 20% max HP",
        R::BossRelic => "swap your starting relic for a random boss relic",
    };
    let drawback = match o.drawback {
        D::None => "",
        D::TenPercentHpLoss => ", but lose 10% max HP",
        D::NoGold => ", but lose all your gold",
        D::Curse => ", but gain a curse",
        D::PercentDamage => ", but lose 30% of your HP",
    };
    let mut s = format!("{reward}{drawback}");
    s[..1].make_ascii_uppercase();
    s
}

/// A monster's name from its id: `JawWorm` -> "Jaw Worm".
fn monster_name(id: sts_core::monsters::MonsterId) -> String {
    words(&format!("{id:?}"))
}

/// The potions held now (the fight's own during one).
fn potions_now(run: &Run) -> Vec<Option<sts_core::potions::PotionId>> {
    match (&run.screen, &run.combat) {
        (Screen::Combat, Some(c)) => c.potions.clone(),
        _ => run.potions.clone(),
    }
}

/// What `a` does from `run`, in words.
pub fn describe(run: &Run, a: &RunAction) -> String {
    let o = run.observe();
    let potions = potions_now(run);
    let monster = |t: &Option<usize>| match (t, &run.combat) {
        (Some(t), Some(c)) => format!(" on {}", c.monsters.get(*t).map_or("?".into(), |m| monster_name(m.id))),
        _ => String::new(),
    };
    match a {
        RunAction::Neow { option } => match &run.screen {
            Screen::Neow { options } => format!("Neow: {}", options.get(*option).map_or("?".into(), neow_text)),
            _ => "Neow".into(),
        },
        RunAction::Path { x } => match &o.screen {
            ScreenView::Map { next } => {
                let room = next.iter().find(|c| c.x == *x).and_then(|c| c.room).map_or("the boss", room_name);
                format!("Go to {room} (column {})", x + 1)
            }
            _ => format!("Go to column {}", x + 1),
        },
        RunAction::Combat { action } => {
            let Some(c) = &run.combat else { return format!("{action:?}") };
            match action {
                PlayerAction::PlayCard { hand_index, target } => {
                    format!("Play {}{}", c.hand.get(*hand_index).map_or("?".into(), |x| x.name()), monster(target))
                }
                PlayerAction::EndTurn => "End turn".into(),
                PlayerAction::UsePotion { slot, target } => format!("Drink {}{}", potion_name(potions.get(*slot).copied().flatten()), monster(target)),
                PlayerAction::DiscardPotion { slot } => format!("Discard {}", potion_name(potions.get(*slot).copied().flatten())),
                PlayerAction::Choose { indices } => {
                    let Some(p) = &c.pending else { return "Choose".into() };
                    let pile: Vec<String> = match p.pile {
                        ChoicePile::Hand => c.hand.iter().map(|x| x.name()).collect(),
                        ChoicePile::Discard => c.discard_pile.iter().map(|x| x.name()).collect(),
                        ChoicePile::Exhaust => c.exhaust_pile.iter().map(|x| x.name()).collect(),
                        ChoicePile::Draw => c.draw_pile.iter().map(|x| x.name()).collect(),
                        ChoicePile::Offer => p.offer.iter().map(|x| x.name()).collect(),
                    };
                    if indices.is_empty() {
                        return "Choose nothing".into();
                    }
                    let names: Vec<String> = indices.iter().map(|&i| pile.get(i).cloned().unwrap_or("?".into())).collect();
                    format!("Choose {}", names.join(", "))
                }
            }
        }
        RunAction::TakeReward { index } => match &o.screen {
            ScreenView::Rewards { items } => match items.get(*index) {
                Some(RewardView::Gold { amount }) | Some(RewardView::StolenGold { amount }) => format!("Take {amount} gold"),
                Some(RewardView::Relic { name, .. }) => format!("Take {name}"),
                Some(RewardView::Potion { name, .. }) => format!("Take {name}"),
                Some(RewardView::Cards { .. }) => "Open the card reward".into(),
                Some(RewardView::EmeraldKey) => "Take the Emerald key".into(),
                Some(RewardView::SapphireKey) => "Take the Sapphire key".into(),
                None => "Take a reward".into(),
            },
            _ => "Take a reward".into(),
        },
        RunAction::PickCard { index } => match &o.screen {
            ScreenView::CardReward { cards } => format!("Pick {}", cards.get(*index).map_or("?".into(), |c| c.name.clone())),
            _ => "Pick a card".into(),
        },
        RunAction::SkipCards => "Skip the cards".into(),
        RunAction::SingingBowl => "Singing Bowl: +2 max HP".into(),
        RunAction::Select { indices } => {
            let names: Vec<String> = indices.iter().map(|&i| run.deck.get(i).map_or("?".into(), |c| c.name())).collect();
            let what = match &o.screen {
                ScreenView::Grid { purpose, .. } => words(&format!("{purpose:?}")),
                _ => "Select".into(),
            };
            if names.is_empty() { format!("{what}: nothing") } else { format!("{what}: {}", names.join(", ")) }
        }
        RunAction::Rest => "Rest".into(),
        RunAction::Smith => "Smith (upgrade a card)".into(),
        RunAction::Lift => "Lift (Girya)".into(),
        RunAction::Toke => "Toke (Peace Pipe)".into(),
        RunAction::Dig => "Dig (Shovel)".into(),
        RunAction::Recall => "Recall (the Ruby key)".into(),
        RunAction::UsePotion { slot } => format!("Drink {}", potion_name(potions.get(*slot).copied().flatten())),
        RunAction::DiscardPotion { slot } => format!("Discard {}", potion_name(potions.get(*slot).copied().flatten())),
        RunAction::BuyCard { index } | RunAction::BuyRelic { index } | RunAction::BuyPotion { index } => match &o.screen {
            ScreenView::Shop { cards, relics, potions, .. } => {
                let list = match a {
                    RunAction::BuyCard { .. } => cards,
                    RunAction::BuyRelic { .. } => relics,
                    _ => potions,
                };
                match list.get(*index).cloned().flatten() {
                    Some(item) => format!("Buy {} for {}", item.name, item.price),
                    None => "Buy".into(),
                }
            }
            _ => "Buy".into(),
        },
        RunAction::Purge => match &o.screen {
            ScreenView::Shop { purge_cost: Some(p), .. } => format!("Pay {p} to remove a card"),
            _ => "Remove a card".into(),
        },
        RunAction::EventChoice { index } => match &o.screen {
            ScreenView::Event { id, options, .. } => format!("{id}: {}", options.get(*index).map_or("?".into(), |x| x.label.clone())),
            _ => "Event option".into(),
        },
        RunAction::BossRelic { index } => match &o.screen {
            ScreenView::BossRelic { relics } => format!("Boss relic: {}", relics.get(*index).cloned().unwrap_or_default()),
            _ => "Boss relic".into(),
        },
        RunAction::Proceed => "Proceed".into(),
    }
}

/// The saved-run format's step for `a` from `run` (as `crates/sts-py`'s recorder writes it):
/// what the game is sent, and where the engine was.
fn saved_step(run: &Run, a: &RunAction) -> Value {
    let mut v = run_step(run, a);
    v["floor"] = json!(run.floor);
    v["hp"] = json!(run.combat.as_ref().map_or(run.hp, |c| c.player.hp));
    v
}

/// A card in a pile, for the viewer: what `info` needs to describe it.
fn pile_card(c: &sts_core::cards::Card) -> Value {
    json!({"id": c.data().java_id, "name": c.name(), "upgraded": c.upgraded, "cost": c.cost, "card_type": format!("{:?}", c.card_type()),
           "base_damage": c.base_damage, "base_block": c.base_block, "base_magic": c.base_magic})
}

fn character_name(c: Character) -> String {
    format!("{c:?}").to_lowercase()
}

impl Session {
    pub fn new(config: RunConfig) -> Session {
        let run = Run::new(config.clone());
        Session { config, actions: Vec::new(), by: Vec::new(), ending: Value::Null, log: Vec::new(), cursor: 0, snapshots: vec![run.clone()], run,
                  skip_keys: Vec::new(), left_rewards: Vec::new() }
    }

    /// A saved run, replayed to its end (to label every decision), standing at its start. Each
    /// decision's `by` is the step's own, else the run's, else `by` (the page's guess, e.g. a
    /// gallery run's "network"), else "saved".
    pub fn load(log: &Value, by: Option<&str>) -> Result<Session, String> {
        let run_by = log["by"].as_str().or(by).unwrap_or("saved");
        let seed = match &log["seed"] {
            Value::Number(n) => n.as_i64().ok_or("bad seed")?,
            Value::String(s) => parse_seed(s).ok_or("bad seed")?,
            _ => return Err("the run has no seed".into()),
        };
        let mut config = RunConfig::new(seed);
        if let Some(n) = log["character"].as_str() {
            config.character = Character::from_name(n).ok_or(format!("unknown character {n}"))?;
        }
        config.ascension = log["ascension"].as_i64().unwrap_or(0) as i32;
        config.neow_full = log["neow_full"].as_bool().unwrap_or(true);
        config.final_act = log["final_act"].as_bool().unwrap_or(true);
        let mut s = Session::new(config);
        for (i, step) in log["steps"].as_array().ok_or("the run has no steps")?.iter().enumerate() {
            let a: RunAction = serde_json::from_value(step["action"].clone()).map_err(|e| format!("decision {i}: {e}"))?;
            s.push(a, step["by"].as_str().unwrap_or(run_by)).map_err(|e| format!("decision {i}: {e}"))?;
        }
        s.goto(0)?;
        Ok(s)
    }

    /// Takes `a` at the end of the decisions (the cursor must be there).
    fn push(&mut self, a: RunAction, by: &str) -> Result<(), String> {
        let entry = json!({"act": self.run.act, "floor": self.run.floor, "pos": self.run.pos, "hp": self.run.combat.as_ref().map_or(self.run.hp, |c| c.player.hp),
                           "label": describe(&self.run, &a), "action": a, "by": by});
        let skip_key = match (&a, &self.run.screen) {
            (RunAction::SkipCards, Screen::CardReward { cards, .. }) => Some(encode_run::card_reward_key(cards)),
            _ => None,
        };
        self.run.step(&a)?;
        self.skip_keys.push(skip_key);
        self.left_rewards.push(!matches!(self.run.screen, Screen::Rewards { .. } | Screen::CardReward { .. }));
        self.actions.push(a);
        self.by.push(by.to_string());
        self.ending = json!({"act": self.run.act, "floor": self.run.floor, "outcome": self.run.outcome});
        self.log.push(entry);
        self.cursor += 1;
        if self.cursor.is_multiple_of(SNAPSHOT) && self.snapshots.len() == self.cursor / SNAPSHOT {
            self.snapshots.push(self.run.clone());
        }
        Ok(())
    }

    /// Stands before decision `index`.
    pub fn goto(&mut self, index: usize) -> Result<(), String> {
        let index = index.min(self.actions.len());
        let k = (index / SNAPSHOT).min(self.snapshots.len() - 1);
        // From the nearest snapshot at or before `index` (or from here, if that's nearer).
        if !(self.cursor <= index && self.cursor >= k * SNAPSHOT) {
            self.run = self.snapshots[k].clone();
            self.cursor = k * SNAPSHOT;
        }
        while self.cursor < index {
            self.run.step(&self.actions[self.cursor])?;
            self.cursor += 1;
        }
        Ok(())
    }

    /// The card rewards skipped on the current reward screen, as training keeps them
    /// (`sts_encode::agent`): those since the last decision that left the reward screens.
    pub fn skipped(&self) -> Skipped {
        let mut from = self.cursor;
        while from > 0 && !self.left_rewards[from - 1] {
            from -= 1;
        }
        self.skip_keys[from..self.cursor].iter().flatten().cloned().collect()
    }

    /// Takes `a` here, dropping any later decisions.
    pub fn step(&mut self, a: RunAction) -> Result<(), String> {
        self.step_by(a, "you")
    }

    /// Takes `a` here for `by` ("you", "network"), dropping any later decisions.
    pub fn step_by(&mut self, a: RunAction, by: &str) -> Result<(), String> {
        if self.run.is_over() {
            return Err("the run is over".into());
        }
        if legal_index(&self.run.legal_actions(), &a).is_none() {
            return Err(format!("not a legal action now: {a:?}"));
        }
        self.actions.truncate(self.cursor);
        self.by.truncate(self.cursor);
        self.log.truncate(self.cursor);
        self.skip_keys.truncate(self.cursor);
        self.left_rewards.truncate(self.cursor);
        self.snapshots.truncate(self.cursor / SNAPSHOT + 1);
        self.push(a, by)
    }

    pub fn view(&self) -> Value {
        let legal = if self.run.is_over() { Vec::new() } else { self.run.legal_actions() };
        json!({
            "step": self.cursor,
            "steps": self.actions.len(),
            "seed": self.config.seed,
            "seed_string": seed_string(self.config.seed),
            "character": character_name(self.config.character),
            "ascension": self.config.ascension,
            "obs": self.run.observe(),
            "potions_now": potions_now(&self.run).iter().map(|p| p.map(|p| p.data().name)).collect::<Vec<_>>(),
            "relics": self.run.relics.iter().map(|r| json!({"id": r.id.java_id(), "name": r.id.data().name, "counter": r.counter, "used_up": r.used_up})).collect::<Vec<_>>(),
            "legal": legal.iter().map(|a| json!({"action": a, "label": describe(&self.run, a)})).collect::<Vec<_>>(),
            "outcome": self.run.outcome,
            "ending": self.ending,
            // A fight's pending choice: the pile index of each option (what `choose` takes).
            "choice_options": self.run.combat.as_ref().and_then(|c| c.pending.as_ref()).map(|p| p.options.clone()),
            "neow": match &self.run.screen {
                Screen::Neow { options } => Some(options.iter().map(neow_text).collect::<Vec<_>>()),
                _ => None,
            },
            "piles": self.run.combat.as_ref().map(|c| {
                let mut draw: Vec<Value> = c.draw_pile.iter().map(pile_card).collect();
                // In name order: the draw pile's order is hidden.
                draw.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
                json!({"draw": draw, "discard": c.discard_pile.iter().map(pile_card).collect::<Vec<_>>(),
                       "exhaust": c.exhaust_pile.iter().map(pile_card).collect::<Vec<_>>()})
            }),
        })
    }

    pub fn log(&self) -> Value {
        json!({"steps": self.log})
    }

    /// The decisions up to here (`all`: every one), in the saved-run format, each with its `by`.
    pub fn save(&self, all: bool) -> Result<Value, String> {
        let mut run = self.snapshots[0].clone();
        let mut steps = Vec::new();
        let end = if all { self.actions.len() } else { self.cursor };
        for (a, by) in self.actions[..end].iter().zip(&self.by) {
            let mut step = saved_step(&run, a);
            step["by"] = json!(by);
            steps.push(step);
            run.step(a)?;
        }
        Ok(json!({
            "seed": self.config.seed,
            "seed_string": seed_string(self.config.seed),
            "character": character_name(self.config.character),
            "ascension": self.config.ascension,
            "neow_full": self.config.neow_full,
            "final_act": self.config.final_act,
            "steps": steps,
        }))
    }

    pub fn run(&self) -> &Run {
        &self.run
    }
}

/// Handles one command (see the module docs); errors come back as `{"error": ...}`.
pub fn handle(session: &mut Option<Session>, cmd: &Value) -> Value {
    match handle_inner(session, cmd) {
        Ok(v) => v,
        Err(e) => json!({"error": e}),
    }
}

/// Display names by game id.
fn names() -> Value {
    let map = |pairs: Vec<(&str, String)>| pairs.into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>();
    json!({
        "relics": map(sts_core::relics::RELIC_DATA.iter().map(|d| (d.java_id, d.name.to_string())).collect()),
        "potions": map(sts_core::potions::POTION_DATA.iter().map(|d| (d.java_id, d.name.to_string())).collect()),
        "cards": map(sts_core::cards::CARD_DATA.iter().map(|d| (d.java_id, d.name.to_string())).collect()),
        "monsters": map(sts_core::monsters::MonsterId::all().map(|m| (m.java_id(), monster_name(m))).collect()),
    })
}

/// A card's numbers and keywords, as a fresh copy has them (`upgraded`: after one upgrade).
fn card_numbers(c: &sts_core::cards::Card) -> Value {
    json!({"cost": c.cost, "damage": c.base_damage, "block": c.base_block, "magic": c.base_magic,
           "exhaust": c.exhaust, "ethereal": c.ethereal, "innate": c.innate, "retain": c.self_retain || c.retain})
}

/// What the viewer's tooltips are built from (`web/text.js` holds the words): every card's type,
/// rarity, colour, target and numbers before and after an upgrade; every relic's tier and class;
/// every potion's rarity, potency and whether it takes a target; monster names.
fn info() -> Value {
    use sts_core::cards::Card;
    let cards: serde_json::Map<String, Value> = sts_core::cards::CARD_DATA
        .iter()
        .map(|d| {
            let base = Card::new(d.id, 0, 0);
            let mut up = base.clone();
            up.upgrade();
            let mut v = card_numbers(&base);
            v["name"] = json!(d.name);
            v["type"] = json!(format!("{:?}", d.card_type));
            v["rarity"] = json!(format!("{:?}", d.rarity));
            v["color"] = json!(format!("{:?}", d.color));
            v["target"] = json!(format!("{:?}", d.target));
            v["up"] = card_numbers(&up);
            (d.java_id.to_string(), v)
        })
        .collect();
    let relics: serde_json::Map<String, Value> = sts_core::relics::RELIC_DATA
        .iter()
        .map(|d| (d.java_id.to_string(), json!({"name": d.name, "tier": format!("{:?}", d.tier), "class": format!("{:?}", d.class)})))
        .collect();
    let potions: serde_json::Map<String, Value> = sts_core::potions::POTION_DATA
        .iter()
        .map(|d| (d.java_id.to_string(), json!({"name": d.name, "rarity": format!("{:?}", d.rarity), "potency": d.potency, "target": d.needs_target})))
        .collect();
    json!({
        "cards": cards,
        "relics": relics,
        "potions": potions,
        "monsters": names()["monsters"],
        "interface": INTERFACE,
        "encoding_version": sts_encode::encode::ENCODING_VERSION,
    })
}

/// The version of this JSON interface (the commands and the replies the page reads). The page
/// runs older engines too (a network keeps the engine it was released with, `web/networks.json`),
/// so raise this when a command or a reply field the page relies on is added, and have the page
/// check it before using one. Engines from before it existed report none: version 1.
const INTERFACE: u32 = 1;

fn handle_inner(session: &mut Option<Session>, cmd: &Value) -> Result<Value, String> {
    let name = cmd["cmd"].as_str().ok_or("no cmd")?;
    if name == "names" {
        return Ok(names());
    }
    if name == "info" {
        return Ok(info());
    }
    if name == "net" {
        return Ok(NET.with(|n| n.borrow().as_ref().map_or(Value::Null, |net| json!(net.metadata))));
    }
    if name == "new_run" {
        let seed = match &cmd["seed"] {
            Value::Number(n) => n.as_i64().ok_or("bad seed")?,
            Value::String(s) => parse_seed(s).ok_or("bad seed")?,
            _ => 1,
        };
        let mut config = RunConfig::new(seed);
        config.ascension = cmd["ascension"].as_i64().unwrap_or(0).clamp(0, 20) as i32;
        if let Some(n) = cmd["character"].as_str() {
            config.character = Character::from_name(n).ok_or(format!("unknown character {n}"))?;
        }
        config.final_act = cmd["final_act"].as_bool().unwrap_or(true);
        config.neow_full = cmd["neow_full"].as_bool().unwrap_or(true);
        *session = Some(Session::new(config));
        return Ok(session.as_ref().unwrap().view());
    }
    if name == "load" {
        *session = Some(Session::load(&cmd["log"], cmd["by"].as_str())?);
        return Ok(session.as_ref().unwrap().view());
    }
    let s = session.as_mut().ok_or("no run: send new_run or load first")?;
    match name {
        "goto" => s.goto(cmd["index"].as_u64().ok_or("no index")? as usize)?,
        "step" => {
            let a = match cmd["index"].as_u64() {
                Some(i) => s.run.legal_actions().get(i as usize).cloned().ok_or(format!("no legal action {i}"))?,
                None => serde_json::from_value(cmd["action"].clone()).map_err(|e| format!("bad action: {e}"))?,
            };
            s.step(a)?;
        }
        "view" => {}
        "advise" => return NET.with(|n| advise(s, n.borrow().as_ref().ok_or("no network loaded")?, cmd["k"].as_u64().unwrap_or(3) as usize)),
        "net_step" => {
            let took = NET.with(|n| -> Result<usize, String> {
                let n = n.borrow();
                let net = n.as_ref().ok_or("no network loaded")?;
                let mut took = 0;
                while took < cmd["count"].as_u64().unwrap_or(1) as usize && !s.run.is_over() {
                    let a = play::choose(net, &s.run, &s.run.legal_actions(), &s.skipped())?;
                    s.step_by(a, "network")?;
                    took += 1;
                }
                Ok(took)
            })?;
            let mut v = s.view();
            v["took"] = json!(took);
            return Ok(v);
        }
        "log" => return Ok(s.log()),
        "save" => return s.save(cmd["all"].as_bool().unwrap_or(false)),
        other => return Err(format!("unknown cmd {other:?}")),
    }
    Ok(s.view())
}

/// The network's advice at the session's position (see the `advise` command).
fn advise(s: &Session, net: &Net, k: usize) -> Result<Value, String> {
    if s.run.is_over() {
        return Err("the run is over".into());
    }
    let legal = s.run.legal_actions();
    // A few more than asked: decisions that read the same (two copies of a card) count as one.
    let (choices, value) = play::advise(net, &s.run, &legal, &s.skipped(), k + 4)?;
    let mut out: Vec<Value> = Vec::new();
    for c in choices {
        let label = describe(&s.run, &c.action);
        match out.iter_mut().find(|o| o["label"] == label) {
            Some(o) => o["probability"] = json!(o["probability"].as_f64().unwrap_or(0.0) + c.probability as f64),
            None => out.push(json!({"index": legal_index(&legal, &c.action), "label": label, "probability": c.probability})),
        }
    }
    out.sort_by(|a, b| b["probability"].as_f64().unwrap_or(0.0).total_cmp(&a["probability"].as_f64().unwrap_or(0.0)));
    out.truncate(k);
    // The network's own move (`net_step`'s): its likeliest single action slot, which in a close
    // call can differ from the likeliest decision once copies of a card count together.
    let own = play::choose(net, &s.run, &legal, &s.skipped())?;
    Ok(json!({"step": s.cursor, "value": value, "choices": out, "pick": legal_index(&legal, &own)}))
}

/// `a`'s place among the legal actions. A card selection matches in any order of its cards: the
/// engine lists a selection's cards in the screen's order (Seek's by name and rarity), while the
/// network picks them in index order, as training does (and the engine takes either).
fn legal_index(legal: &[RunAction], a: &RunAction) -> Option<usize> {
    fn cards(a: &RunAction) -> Option<Vec<usize>> {
        match a {
            RunAction::Combat { action: PlayerAction::Choose { indices } } | RunAction::Select { indices } => {
                let mut v = indices.clone();
                v.sort_unstable();
                Some(v)
            }
            _ => None,
        }
    }
    legal.iter().position(|l| l == a).or_else(|| {
        let want = cards(a)?;
        legal.iter().position(|l| std::mem::discriminant(l) == std::mem::discriminant(a) && cards(l).as_ref() == Some(&want))
    })
}

thread_local! {
    static NET: RefCell<Option<Net>> = const { RefCell::new(None) };
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Room for a command of `len` bytes (JavaScript writes it there, then calls `call`).
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut v: Vec<u8> = Vec::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Frees what `alloc` gave.
///
/// # Safety
/// `p` and `len` must come from one `alloc` call, freed once.
#[no_mangle]
pub unsafe extern "C" fn dealloc(p: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(p, 0, len));
}

/// Runs the command (JSON) at `p`, `len` bytes long; returns the reply's length (at `out_ptr()`).
///
/// # Safety
/// `p` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn call(p: *const u8, len: usize) -> usize {
    let input = std::slice::from_raw_parts(p, len);
    let reply = match serde_json::from_slice::<Value>(input) {
        Ok(cmd) => SESSION.with(|s| handle(&mut s.borrow_mut(), &cmd)),
        Err(e) => json!({"error": format!("bad JSON: {e}")}),
    };
    let bytes = serde_json::to_vec(&reply).unwrap_or_default();
    OUT.with(|o| {
        *o.borrow_mut() = bytes;
        o.borrow().len()
    })
}

/// Loads the network from a release weights file (`len` bytes at `p`); returns the reply's length
/// (at `out_ptr()`): the file's metadata, or an error (another encoding's file is refused).
///
/// # Safety
/// `p` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn load_net(p: *const u8, len: usize) -> usize {
    let bytes = std::slice::from_raw_parts(p, len);
    let reply = match Net::from_safetensors(bytes) {
        Ok(net) => {
            let meta = json!(net.metadata);
            NET.with(|n| *n.borrow_mut() = Some(net));
            meta
        }
        Err(e) => json!({"error": e}),
    };
    let bytes = serde_json::to_vec(&reply).unwrap_or_default();
    OUT.with(|o| {
        *o.borrow_mut() = bytes;
        o.borrow().len()
    })
}

/// Where the last reply is.
#[no_mangle]
pub extern "C" fn out_ptr() -> *const u8 {
    OUT.with(|o| o.borrow().as_ptr())
}

#[cfg(test)]
mod tests {
    /// A card selection is legal in any order of its cards (Seek lists its grid by name, the
    /// network picks in index order), and nothing else is.
    #[test]
    fn a_card_selection_matches_in_any_order() {
        let choose = |i: &[usize]| RunAction::Combat { action: PlayerAction::Choose { indices: i.to_vec() } };
        let select = |i: &[usize]| RunAction::Select { indices: i.to_vec() };
        let legal = vec![choose(&[14, 2]), choose(&[14, 5]), select(&[3, 1]), RunAction::Combat { action: PlayerAction::EndTurn }];
        assert_eq!(legal_index(&legal, &choose(&[2, 14])), Some(0));
        assert_eq!(legal_index(&legal, &choose(&[14, 5])), Some(1));
        assert_eq!(legal_index(&legal, &select(&[1, 3])), Some(2));
        assert_eq!(legal_index(&legal, &choose(&[2, 5])), None);
        assert_eq!(legal_index(&legal, &choose(&[1, 3])), None);
        assert_eq!(legal_index(&legal, &RunAction::Combat { action: PlayerAction::EndTurn }), Some(3));
    }

    /// `info` covers every card, relic and potion, with an upgraded copy that differs from the
    /// fresh one where the game's upgrade changes a number (Bash: 8 then 10 damage).
    #[test]
    fn info_has_every_card_relic_and_potion() {
        let v = handle(&mut None, &json!({"cmd": "info"}));
        assert_eq!(v["cards"].as_object().unwrap().len(), sts_core::cards::CARD_DATA.len());
        assert_eq!(v["relics"].as_object().unwrap().len(), sts_core::relics::RELIC_DATA.len());
        assert_eq!(v["potions"].as_object().unwrap().len(), sts_core::potions::POTION_DATA.len());
        let bash = &v["cards"]["Bash"];
        assert_eq!((bash["damage"].as_i64(), bash["up"]["damage"].as_i64(), bash["magic"].as_i64()), (Some(8), Some(10), Some(2)));
        assert_eq!(bash["type"], "Attack");
        assert_eq!(v["cards"]["Limit Break"]["exhaust"], true);
        assert_eq!(v["cards"]["Limit Break"]["up"]["exhaust"], false);
        assert_eq!(v["potions"]["Fire Potion"]["potency"], 20);
        assert_eq!(v["potions"]["Fire Potion"]["target"], true);
        assert_eq!(v["interface"], INTERFACE);
        assert_eq!(v["encoding_version"], sts_encode::encode::ENCODING_VERSION);
    }

    use super::*;
    use sts_core::rng::StsRandom;

    /// Random play for `n` decisions from a new run, through `handle` like JavaScript.
    fn play(seed: i64, character: &str, n: usize) -> Option<Session> {
        let mut s = None;
        let v = handle(&mut s, &json!({"cmd": "new_run", "seed": seed, "character": character, "ascension": 10}));
        assert!(v.get("error").is_none(), "{v}");
        let mut roll = StsRandom::new(seed);
        for _ in 0..n {
            let legal = s.as_ref().unwrap().view()["legal"].as_array().unwrap().len();
            if legal == 0 {
                break;
            }
            let v = handle(&mut s, &json!({"cmd": "step", "index": roll.random(legal as i32 - 1)}));
            assert!(v.get("error").is_none(), "{v}");
        }
        s
    }

    /// Deck cards carry their own numbers: Searing Blow upgraded three times hits for 12 + 4 + 5 + 6.
    #[test]
    fn deck_cards_carry_their_own_numbers() {
        let mut s = Session::new(RunConfig::new(1));
        let mut c = s.run.new_card(sts_core::cards::CardId::SearingBlow);
        for _ in 0..3 {
            c.upgrade();
        }
        s.run.deck.push(c);
        let v = s.view();
        let card = v["obs"]["deck"].as_array().unwrap().iter().find(|c| c["id"] == "Searing Blow").unwrap().clone();
        assert_eq!((card["name"].as_str(), card["base_damage"].as_i64()), (Some("Searing Blow+3"), Some(27)));
    }

    fn tiny_net() -> Net {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../sts-net/tests/data/tiny.safetensors");
        Net::from_safetensors(&std::fs::read(path).unwrap()).unwrap()
    }

    /// The session's network play (`net_step`) is the agent's own (`sts_net::play`, as training
    /// plays), decision for decision; at any position the session's skipped card rewards are the
    /// ones training would hold there, so `advise` sees what the network saw. The runs mix the
    /// network's decisions (a quarter) with random ones (which skip card rewards often).
    #[test]
    fn network_play_and_advice_match_the_agent() {
        NET.with(|n| *n.borrow_mut() = Some(tiny_net()));
        let net = tiny_net();
        let mut skips = 0;
        for (seed, asc) in [(11, 0), (12, 20), (13, 10), (14, 5), (15, 15)] {
            let mut s = None;
            handle(&mut s, &json!({"cmd": "new_run", "seed": seed, "ascension": asc}));
            let mut roll = StsRandom::new(seed);
            for _ in 0..900 {
                let legal = s.as_ref().unwrap().view()["legal"].as_array().unwrap().len();
                if legal == 0 {
                    break;
                }
                let v = if roll.random(3) == 0 {
                    handle(&mut s, &json!({"cmd": "net_step"}))
                } else {
                    // At random, but in a fight a card whenever one can be played (else the runs
                    // end in the first fights), and card rewards skipped half the time.
                    let all = s.as_ref().unwrap().run.legal_actions();
                    let plays: Vec<usize> = (0..all.len()).filter(|&i| matches!(all[i], RunAction::Combat { action: PlayerAction::PlayCard { .. } })).collect();
                    let skip = all.iter().position(|a| *a == RunAction::SkipCards);
                    let i = match skip {
                        Some(k) if roll.random_boolean() => k,
                        _ if !plays.is_empty() => plays[roll.random(plays.len() as i32 - 1) as usize],
                        _ => roll.random(legal as i32 - 1) as usize,
                    };
                    handle(&mut s, &json!({"cmd": "step", "index": i}))
                };
                assert!(v.get("error").is_none(), "{v}");
            }
            let mut s = s.unwrap();
            let took = s.actions.len();
            // The same run replayed by the agent, its skipped rewards kept as training does.
            let mut cfg = RunConfig::new(seed);
            cfg.ascension = asc as i32;
            let mut run = Run::new(cfg);
            let mut skipped: Skipped = Vec::new();
            let mut held = vec![skipped.clone()];
            for i in 0..took {
                if s.by[i] == "network" {
                    let a = play::choose(&net, &run, &run.legal_actions(), &skipped).unwrap();
                    assert_eq!(a, s.actions[i], "seed {seed}: decision {i}");
                }
                play::step(&mut run, &mut skipped, &s.actions[i]).unwrap();
                held.push(skipped.clone());
            }
            skips += held.iter().filter(|k| !k.is_empty()).count();
            assert!(s.by.iter().any(|b| b == "network") && s.by.iter().any(|b| b == "you"));
            for _ in 0..80 {
                let i = roll.random(took as i32) as usize;
                s.goto(i).unwrap();
                assert_eq!(s.skipped(), held[i], "seed {seed}: skipped rewards at decision {i}");
                if s.run.is_over() {
                    continue;
                }
                let mut t = Some(s);
                let v = handle(&mut t, &json!({"cmd": "advise", "k": 3}));
                s = t.unwrap();
                let choices = v["choices"].as_array().unwrap_or_else(|| panic!("{v}"));
                let legal = s.run.legal_actions();
                assert!(!choices.is_empty() && choices.len() <= 3);
                let probs: Vec<f64> = choices.iter().map(|c| c["probability"].as_f64().unwrap()).collect();
                assert!(probs.windows(2).all(|w| w[0] >= w[1]) && probs.iter().sum::<f64>() <= 1.0 + 1e-4);
                for c in choices {
                    assert!(c["index"].as_u64().is_some_and(|k| (k as usize) < legal.len()), "{v}");
                }
                // The network's own decision there is its pick.
                if i < took && s.by[i] == "network" {
                    let taken = legal_index(&legal, &s.actions[i]).unwrap();
                    assert_eq!(v["pick"], taken, "seed {seed}, decision {i}: {v}");
                }
            }
        }
        assert!(skips > 0, "no card reward was skipped");
        NET.with(|n| *n.borrow_mut() = None);
    }

    /// Saving a played run and loading it back reaches the same place, and every step on the way
    /// matches going there directly (snapshots and replays agree).
    #[test]
    fn save_load_and_goto_agree() {
        for (seed, character) in [(3, "ironclad"), (4, "silent"), (5, "defect"), (6, "watcher")] {
            let s = play(seed, character, 700).unwrap();
            let end = s.view();
            let saved = s.save(false).unwrap();
            let mut t = Session::load(&saved, None).unwrap();
            assert_eq!(t.view()["step"], 0);
            t.goto(usize::MAX).unwrap();
            let back = t.view();
            assert_eq!(back["obs"]["floor"], end["obs"]["floor"], "{character}");
            assert_eq!(back["obs"]["hp"], end["obs"]["hp"]);
            assert_eq!(back["obs"]["deck"], end["obs"]["deck"]);
            assert_eq!(back["steps"], end["steps"]);
            // Backwards, forwards and across snapshots: the same observations as a straight replay.
            let n = t.view()["steps"].as_u64().unwrap() as usize;
            for i in [n / 2, 1, n - 1, SNAPSHOT, SNAPSHOT - 1, 2 * SNAPSHOT + 3, 0, n] {
                let i = i.min(n);
                t.goto(i).unwrap();
                let mut u = Session::load(&saved, None).unwrap();
                for k in 0..i {
                    u.run.step(&u.actions[k].clone()).unwrap();
                }
                u.cursor = i;
                assert_eq!(t.view()["obs"].to_string(), u.view()["obs"].to_string(), "{character} step {i}");
            }
            // Every saved step carries the engine's action and what to send the game (commands, or
            // the map column or grid cards that playback resolves).
            assert!(saved["steps"].as_array().unwrap().iter().all(|x| x.get("action").is_some()
                && (x.get("cmds").is_some() || x.get("path_x").is_some() || x.get("grid_deck").is_some())));
            assert_eq!(t.log()["steps"].as_array().unwrap().len(), n);
        }
    }

    /// The saved A20 win (python/results/watch, played back in the real game before) replays to
    /// its win, and saving it again gives back the same decisions and game commands. It ends in
    /// the victory room (floor 52, after the second boss on 51), as the game's does.
    #[test]
    fn a_saved_win_replays() {
        saved_win_replays("a20_win_L7WC.json", 52);
        // The Silent's (silent-runs-5 update 800, seed 70ES).
        saved_win_replays("silent_a20_win_sl5.json", 52);
    }

    fn saved_win_replays(file: &str, floor: i32) {
        let text = std::fs::read_to_string(format!("{}/../../python/results/watch/{file}", env!("CARGO_MANIFEST_DIR"))).unwrap();
        let log: Value = serde_json::from_str(&text).unwrap();
        let mut s = Session::load(&log, Some("network")).unwrap();
        s.goto(usize::MAX).unwrap();
        let v = s.view();
        assert_eq!(v["outcome"], "Won", "{file}");
        assert_eq!(v["obs"]["floor"], floor);
        let again = s.save(false).unwrap();
        for (a, b) in again["steps"].as_array().unwrap().iter().zip(log["steps"].as_array().unwrap()) {
            assert_eq!(a["action"], b["action"]);
            assert_eq!(a["cmds"], b["cmds"]);
            assert_eq!(a["floor"], b["floor"]);
            assert_eq!(a["hp"], b["hp"]);
        }
        assert_eq!(again["steps"].as_array().unwrap().len(), log["steps"].as_array().unwrap().len());
    }

    /// A decision part-way through drops the ones after it.
    #[test]
    fn stepping_back_and_choosing_again_branches() {
        let mut s = play(9, "ironclad", 120);
        let n = s.as_ref().unwrap().view()["steps"].as_u64().unwrap();
        handle(&mut s, &json!({"cmd": "goto", "index": 40}));
        let v = handle(&mut s, &json!({"cmd": "step", "index": 0}));
        assert!(v.get("error").is_none(), "{v}");
        assert_eq!(v["step"], 41);
        assert_eq!(v["steps"], 41);
        assert_eq!(v["ending"]["floor"], v["obs"]["floor"]);
        assert!(n > 41);
        assert!(handle(&mut s, &json!({"cmd": "step", "index": 999}))["error"].is_string());
        // Who took each decision survives saving every decision and loading them back.
        let mut t = play(9, "ironclad", 60).unwrap();
        t.goto(20).unwrap();
        assert_eq!(t.save(true).unwrap()["steps"].as_array().unwrap().len(), 60);
        assert_eq!(t.save(false).unwrap()["steps"].as_array().unwrap().len(), 20);
        let back = Session::load(&t.save(true).unwrap(), Some("network")).unwrap();
        assert!(back.log()["steps"].as_array().unwrap().iter().all(|e| e["by"] == "you"));
        let mut saved = t.save(true).unwrap();
        for st in saved["steps"].as_array_mut().unwrap() {
            st.as_object_mut().unwrap().remove("by");
        }
        let back = Session::load(&saved, Some("network")).unwrap();
        assert!(back.log()["steps"].as_array().unwrap().iter().all(|e| e["by"] == "network"));
        assert!(handle(&mut s, &json!({"cmd": "nonsense"}))["error"].is_string());
    }

    /// Every legal action has a label, all through random runs of each character.
    #[test]
    fn every_legal_action_is_labelled() {
        for (seed, character) in [(11, "ironclad"), (12, "silent"), (13, "defect"), (14, "watcher")] {
            let s = play(seed, character, 900).unwrap();
            let log = s.log();
            for e in log["steps"].as_array().unwrap() {
                let l = e["label"].as_str().unwrap();
                assert!(!l.is_empty() && !l.contains('?'), "{character}: {l}");
            }
        }
    }
}
