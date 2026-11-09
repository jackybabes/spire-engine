//! Replays combat traces recorded from the real game (tools/trace/record.py) through the
//! engine and reports the first state that differs.

use std::collections::BTreeMap;

use serde_json::Value;
use sts_core::cards::CardId;
use sts_core::combat::{ChoicePile, Combat, CombatConfig, PlayerAction};
use sts_core::relics::RelicId as Relic;
use sts_core::Outcome;

pub struct ReplayResult {
    pub ok: bool,
    pub steps: usize,
    pub message: String,
}

pub(crate) fn s(v: &Value) -> String {
    match v {
        Value::String(x) => x.clone(),
        other => other.to_string(),
    }
}

pub(crate) fn int(v: &Value, k: &str) -> i64 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(-999)
}

/// A normalized, comparable snapshot. Keys are paths; values are rendered strings.
type Snap = BTreeMap<String, String>;

fn powers_game(v: &Value) -> String {
    let mut out: Vec<String> = v
        .as_array()
        .map(|a| {
            a.iter()
                .map(|p| {
                    let ja = p.get("just_applied").and_then(|x| x.as_bool()).map(|b| if b { "*" } else { "" }).unwrap_or("");
                    // Each Bomb's ID is "TheBomb" and a running number, so several can tick.
                    let id = s(&p["id"]);
                    let id = if id.starts_with("TheBomb") && id[7..].chars().all(|c| c.is_ascii_digit()) { "TheBomb".to_string() } else { id };
                    format!("{}={}{}", id, int(p, "amount"), ja)
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out.join(",")
}

fn powers_engine(ps: &[sts_core::powers::Power], with_just_applied: bool) -> String {
    let mut out: Vec<String> = ps
        .iter()
        .map(|p| {
            let has_ja = matches!(p.id, sts_core::powers::PowerId::Vulnerable | sts_core::powers::PowerId::Weakened | sts_core::powers::PowerId::Frail);
            // CommunicationMod reports Ritual's skipFirst through the same flag.
            let flagged = if p.id == sts_core::powers::PowerId::Ritual { p.skip_first } else { has_ja && p.just_applied };
            let ja = if with_just_applied && flagged { "*" } else { "" };
            format!("{}={}{}", p.java_id(), p.amount, ja)
        })
        .collect();
    out.sort();
    out.join(",")
}

fn pile_game(v: &Value) -> String {
    v.as_array()
        .map(|a| a.iter().map(|c| format!("{}+{}:{}", s(&c["id"]), int(c, "upgrades"), int(c, "cost"))).collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

fn pile_engine(cards: &[sts_core::cards::Card]) -> String {
    cards.iter().map(|c| format!("{}+{}:{}", c.data().java_id, c.times_upgraded, c.cost_for_turn)).collect::<Vec<_>>().join(" ")
}

fn strip_costs(pile: &str) -> String {
    pile.split(' ').map(|c| c.rsplit_once(':').map_or(c, |(a, _)| a)).collect::<Vec<_>>().join(" ")
}

pub(crate) fn snap_game(gs: &Value) -> Snap {
    let mut m = Snap::new();
    let cs = &gs["combat_state"];
    let p = &cs["player"];
    m.insert("player.hp".into(), int(p, "current_hp").to_string());
    m.insert("player.max_hp".into(), int(p, "max_hp").to_string());
    m.insert("player.block".into(), int(p, "block").to_string());
    m.insert("player.energy".into(), int(p, "energy").to_string());
    m.insert("player.powers".into(), powers_game(&p["powers"]));
    // Orb slots: id, passive and evoke numbers (an empty slot's numbers don't matter). An empty
    // slot made by `new EmptyOrbSlot()` (the slots at combat start, and those left behind by an
    // evoke) has no ID; only the constructor with a position sets "Empty".
    let orbs: Vec<String> = p["orbs"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|o| if o["id"].is_null() || s(&o["id"]) == "Empty" { "Empty".to_string() } else { format!("{}:{}/{}", s(&o["id"]), int(o, "passive_amount"), int(o, "evoke_amount")) })
                .collect()
        })
        .unwrap_or_default();
    m.insert("player.orbs".into(), orbs.join(","));
    // The stance (recordings from before the harness reported it are all Neutral).
    let stance = if p["stance"].is_string() { s(&p["stance"]) } else { "Neutral".to_string() };
    m.insert("player.stance".into(), stance);
    m.insert("turn".into(), int(cs, "turn").to_string());
    for k in ["hand", "draw_pile"] {
        m.insert(k.into(), pile_game(&cs[k]));
    }
    // A card's cost resets when its Soul animation lands on the discard pile, or when its exhaust
    // animation ends (ExhaustCardEffect), which may be after the next decision; compare those
    // piles without costs.
    m.insert("discard_pile".into(), strip_costs(&pile_game(&cs["discard_pile"])));
    m.insert("exhaust_pile".into(), strip_costs(&pile_game(&cs["exhaust_pile"])));
    let hm = gs["harness"]["monsters"].as_array().cloned().unwrap_or_default();
    for (i, mon) in cs["monsters"].as_array().cloned().unwrap_or_default().iter().enumerate() {
        let pre = format!("m{i}");
        m.insert(format!("{pre}.id"), s(&mon["id"]));
        m.insert(format!("{pre}.hp"), format!("{}/{}", int(mon, "current_hp"), int(mon, "max_hp")));
        m.insert(format!("{pre}.block"), int(mon, "block").to_string());
        m.insert(format!("{pre}.gone"), mon["is_gone"].to_string());
        // A dead monster's powers linger until its death animation ends, which depends on
        // frame timing, so only living monsters' powers are compared.
        if !mon["is_gone"].as_bool().unwrap_or(false) {
            m.insert(format!("{pre}.powers"), powers_game(&mon["powers"]));
        }
        // Runic Dome hides intents, and CommunicationMod then leaves the move out.
        if mon.get("move_id").is_some() {
            m.insert(format!("{pre}.move"), format!("{} dmg {}", int(mon, "move_id"), int(mon, "move_base_damage")));
        }
        if let Some(h) = hm.get(i) {
            let hist: Vec<String> = h["move_history"].as_array().map(|a| a.iter().map(s).collect()).unwrap_or_default();
            m.insert(format!("{pre}.history"), hist.join(","));
            let outs: Vec<String> = h["damage_outputs"].as_array().map(|a| a.iter().map(s).collect()).unwrap_or_default();
            if !mon["is_gone"].as_bool().unwrap_or(false) {
                m.insert(format!("{pre}.damage_outputs"), outs.join(","));
            }
        }
    }
    for k in ["monster_hp", "ai", "shuffle", "card_random", "misc"] {
        m.insert(format!("rng.{k}"), int(&gs["harness"]["rng"][k], "counter").to_string());
    }
    m
}

pub(crate) fn snap_engine(c: &Combat, with_just_applied: bool) -> Snap {
    let mut m = Snap::new();
    m.insert("player.hp".into(), c.player.hp.to_string());
    m.insert("player.max_hp".into(), c.player.max_hp.to_string());
    m.insert("player.block".into(), c.player.block.to_string());
    m.insert("player.energy".into(), c.energy.to_string());
    m.insert("player.powers".into(), powers_engine(&c.player.powers, with_just_applied));
    let orbs: Vec<String> = c
        .player
        .orbs
        .iter()
        .map(|o| match o {
            Some(o) => format!("{}:{}/{}", o.id.java_id(), c.orb_passive(o), c.orb_evoke(o)),
            None => "Empty".to_string(),
        })
        .collect();
    m.insert("player.orbs".into(), orbs.join(","));
    m.insert("player.stance".into(), c.player.stance.java_id().to_string());
    m.insert("turn".into(), c.turn.to_string());
    m.insert("hand".into(), pile_engine(&c.hand));
    m.insert("draw_pile".into(), pile_engine(&c.draw_pile));
    m.insert("discard_pile".into(), strip_costs(&pile_engine(&c.discard_pile)));
    m.insert("exhaust_pile".into(), strip_costs(&pile_engine(&c.exhaust_pile)));
    for (i, &id) in c.order.iter().enumerate() {
        let mon = &c.monsters[id];
        let pre = format!("m{i}");
        m.insert(format!("{pre}.id"), mon.id.java_id().to_string());
        m.insert(format!("{pre}.hp"), format!("{}/{}", mon.hp, mon.max_hp));
        m.insert(format!("{pre}.block"), mon.block.to_string());
        let gone = mon.is_dead_or_escaped();
        m.insert(format!("{pre}.gone"), gone.to_string());
        if !gone {
            m.insert(format!("{pre}.powers"), powers_engine(&mon.powers, with_just_applied));
        }
        let mv = if mon.mv.next_move == sts_core::monsters::NO_MOVE { -1 } else { mon.mv.next_move as i64 };
        if !c.has_relic(sts_core::relics::RelicId::RunicDome) {
            m.insert(format!("{pre}.move"), format!("{} dmg {}", mv, mon.mv.base_damage));
        }
        m.insert(format!("{pre}.history"), mon.move_history.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        if !gone {
            m.insert(format!("{pre}.damage_outputs"), mon.damage.iter().map(|d| d.output.to_string()).collect::<Vec<_>>().join(","));
        }
    }
    let r = &c.rng;
    m.insert("rng.monster_hp".into(), r.monster_hp.counter.to_string());
    m.insert("rng.ai".into(), r.ai.counter.to_string());
    m.insert("rng.shuffle".into(), r.shuffle.counter.to_string());
    m.insert("rng.card_random".into(), r.card_random.counter.to_string());
    m.insert("rng.misc".into(), r.misc.counter.to_string());
    m
}

pub(crate) fn diff(game: &Snap, engine: &Snap) -> Vec<String> {
    let mut out = Vec::new();
    for (k, gv) in game {
        let ev = engine.get(k).cloned().unwrap_or_else(|| "<missing>".into());
        if &ev != gv {
            out.push(format!("  {k}:\n    game:   {gv}\n    engine: {ev}"));
        }
    }
    for k in engine.keys() {
        if !game.contains_key(k) {
            out.push(format!("  {k}: only in engine ({})", engine[k]));
        }
    }
    out
}

fn config_from_trace(setup: &Value, first: &Value) -> Result<CombatConfig, String> {
    let h = &first["harness"];
    let seed: i64 = s(&h["seed"]).parse().map_err(|_| "bad seed")?;
    let character = sts_core::character::Character::from_name(&s(&first["class"])).unwrap_or_default();
    let mut cfg = CombatConfig::new(character, seed, &s(&setup["encounter"]));
    cfg.floor = int(h, "floor") as i32;
    cfg.ascension = int(first, "ascension_level") as i32;
    cfg.deck = first["deck"]
        .as_array()
        .ok_or("no deck")?
        .iter()
        .map(|c| {
            let id = s(&c["id"]);
            CardId::from_java_id(&id).map(|cid| (cid, int(c, "upgrades") as i32)).ok_or(format!("card {id} not ported"))
        })
        .collect::<Result<_, _>>()?;
    let p = &first["combat_state"]["player"];
    cfg.player_hp = int(p, "current_hp") as i32;
    cfg.player_max_hp = int(p, "max_hp") as i32;
    cfg.gold = int(first, "gold") as i32;
    cfg.relics = first["relics"]
        .as_array()
        .ok_or("no relics")?
        .iter()
        .map(|r| Relic::from_name(&s(&r["id"])).ok_or(format!("relic {} not ported", s(&r["id"]))))
        .collect::<Result<_, _>>()?;
    // The first recorded state is after turn 1's atTurnStart, so take counters from onEquip
    // instead of from the state.
    cfg.relic_counters = Some(
        cfg.relics
            .iter()
            .map(|&r| match r {
                Relic::HappyFlower | Relic::IncenseBurner | Relic::Sundial | Relic::TinyChest => 0,
                Relic::DuVuDoll => cfg.deck.iter().filter(|(c, _)| c.data().card_type == sts_core::cards::CardType::Curse).count() as i32,
                _ => sts_core::relics::RelicState::new(r).counter,
            })
            .collect(),
    );
    cfg.potions = first["potions"]
        .as_array()
        .map(|a| a.iter().map(|p| sts_core::potions::PotionId::from_java_id(&s(&p["id"]))).collect())
        .unwrap_or_default();
    Ok(cfg)
}

/// `picks` holds the cards chosen so far in a selection that takes several (the game picks them
/// one `choose` at a time); the selection goes to the engine once it is complete or confirmed.
pub(crate) fn to_action(cmd: &str, c: &Combat, picks: &mut Vec<usize>) -> Result<Option<PlayerAction>, String> {
    let t: Vec<&str> = cmd.split_whitespace().collect();
    match t.first().copied() {
        Some("play") => {
            let i: usize = t.get(1).ok_or("play needs a card")?.parse().map_err(|_| "bad card index")?;
            let target = match t.get(2) {
                Some(x) => {
                    let pos: usize = x.parse().map_err(|_| "bad target")?;
                    Some(*c.order.get(pos).ok_or("target out of range")?)
                }
                None => None,
            };
            Ok(Some(PlayerAction::PlayCard { hand_index: if i == 0 { 9 } else { i - 1 }, target }))
        }
        Some("end") => Ok(Some(PlayerAction::EndTurn)),
        Some("potion") => {
            let slot: usize = t.get(2).ok_or("potion needs a slot")?.parse().map_err(|_| "bad potion slot")?;
            match t.get(1).copied() {
                Some("use") => {
                    let target = match t.get(3) {
                        Some(x) => {
                            let pos: usize = x.parse().map_err(|_| "bad target")?;
                            Some(*c.order.get(pos).ok_or("target out of range")?)
                        }
                        None => None,
                    };
                    Ok(Some(PlayerAction::UsePotion { slot, target }))
                }
                Some("discard") => Ok(Some(PlayerAction::DiscardPotion { slot })),
                other => Err(format!("unsupported potion command {other:?}")),
            }
        }
        Some("choose") => {
            let k: usize = t.get(1).ok_or("choose needs an index")?.parse().map_err(|_| "bad choice")?;
            let p = c.pending.as_ref().ok_or("game is choosing but engine has no pending selection")?;
            // A hand selection's list leaves out the cards already picked.
            let left: Vec<usize> = p.options.iter().copied().filter(|i| p.pile != ChoicePile::Hand || !picks.contains(i)).collect();
            let idx = *left.get(k).ok_or("choice out of range")?;
            picks.push(idx);
            // An exact-count selection resolves at its last pick; an any-number one (Scry, Purity)
            // waits for the confirm, as the game's screen does, even with every card picked: its
            // effects (a Scry's draw, and Confusion's cost roll on it) come after the confirm.
            if picks.len() >= p.count && !p.any_number {
                return Ok(Some(PlayerAction::Choose { indices: std::mem::take(picks) }));
            }
            Ok(None)
        }
        // Confirming what was picked of an any-number selection, or nothing (e.g. Gambling Chip
        // keeping the hand).
        Some("confirm") if c.pending.as_ref().is_some_and(|p| p.any_number) && !picks.is_empty() => {
            Ok(Some(PlayerAction::Choose { indices: std::mem::take(picks) }))
        }
        Some("confirm") if c.pending.as_ref().is_some_and(|p| p.any_number && p.can_pick_zero) => Ok(Some(PlayerAction::Choose { indices: Vec::new() })),
        // Skipping a skippable card choice (Nilry's Codex).
        Some("skip") if c.pending.as_ref().is_some_and(|p| p.can_pick_zero) => Ok(Some(PlayerAction::Choose { indices: Vec::new() })),
        Some("confirm") | Some("wait") => Ok(None),
        other => Err(format!("unsupported command {other:?}")),
    }
}

/// Replays one trace file.
pub fn replay(text: &str) -> ReplayResult {
    replay_observed(text, &mut crate::coverage::NoObserver)
}

/// `replay`, telling `obs` about every engine step (`sts-cli coverage`).
pub fn replay_observed(text: &str, obs: &mut dyn crate::coverage::Observer) -> ReplayResult {
    match replay_inner(text, obs) {
        Ok((steps, msg)) => ReplayResult { ok: msg.is_empty(), steps, message: msg },
        Err(e) => ReplayResult { ok: false, steps: 0, message: e },
    }
}

fn replay_inner(text: &str, obs: &mut dyn crate::coverage::Observer) -> Result<(usize, String), String> {
    let recs: Vec<Value> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    let setup = recs.iter().find(|r| r["type"] == "setup").ok_or("no setup record")?["setup"].clone();
    let first = recs.iter().find(|r| r["type"] == "state").ok_or("no states")?["state"].clone();
    let cfg = config_from_trace(&setup, &first)?;
    let mut c = Combat::new(&cfg)?;
    // The fight sits in a run the harness started, whose potion and card streams have been drawn
    // from already (fights reseed only the per-floor streams): take them from the recorded state,
    // for Alchemize and Entropic Brew, and Foreign Influence's picks.
    let recorded = |k: &str| -> Option<sts_core::rng::StsRandom> {
        let p = first["harness"]["rng"].get(k)?;
        let part = |k: &str| s(&p[k]).parse::<i64>().map(|v| v as u64);
        match (part("s0"), part("s1")) {
            (Ok(s0), Ok(s1)) => Some(sts_core::rng::StsRandom { rng: sts_core::rng::Xs128::from_state(s0, s1), counter: int(p, "counter") as i32 }),
            _ => None,
        }
    };
    if let Some(r) = recorded("potion") {
        c.rng.potion = r;
    }
    if let Some(r) = recorded("card") {
        c.rng.card = r;
    }
    // STS_REPLAY_DEBUG=1: print every command with the engine's hand and piles after it.
    let debug = std::env::var_os("STS_REPLAY_DEBUG").is_some();
    let mut picks = Vec::new();
    let mut steps = 0;
    let mut last_cmd = String::from("(combat start)");
    let mut last_screen = String::new();
    let mut i = 0;
    while i < recs.len() {
        let r = &recs[i];
        match r["type"].as_str() {
            Some("state") => {
                let gs = &r["state"];
                let screen = s(&gs["screen_type"]);
                // Compare only at plain combat decisions (not mid-selection screens).
                let next_cmd = recs.get(i + 1).filter(|n| n["type"] == "command").map(|n| s(&n["command"])).unwrap_or_default();
                // A discovery pick reaches the hand through ShowCardAndAddToHandEffect, which may
                // still be animating at the next state.
                let after_discovery = last_screen == "CARD_REWARD";
                last_screen = screen.clone();
                if c.smoked {
                    // Smoke Bomb: the game lets the player act during the 2.5 s escape; the
                    // engine ends the fight once the queue empties.
                    return Ok((steps, String::new()));
                }
                if screen == "NONE" && c.pending.is_none() && !next_cmd.starts_with("confirm") && !after_discovery {
                    let d = diff(&snap_game(gs), &snap_engine(&c, true));
                    if !d.is_empty() {
                        return Ok((steps, format!("after `{last_cmd}` (step {steps}):\n{}", d.join("\n"))));
                    }
                }
            }
            Some("command") => {
                let cmd = s(&r["command"]);
                if let Some(a) = to_action(&cmd, &c, &mut picks)? {
                    obs.combat_before(&c, &a);
                    c.step(&a).map_err(|e| format!("engine rejected `{cmd}` at step {steps}: {e}"))?;
                    obs.combat_after(&c);
                    steps += 1;
                }
                if debug {
                    eprintln!("{cmd:12} -> hand [{}] draw {} discard [{}] energy {} block {}", pile_engine(&c.hand), c.draw_pile.len(), pile_engine(&c.discard_pile), c.energy, c.player.block);
                }
                last_cmd = cmd;
            }
            Some("end") => {
                let gs = &r["state"];
                let game_hp = int(gs, "current_hp");
                let game_dead = s(&gs["screen_type"]) == "GAME_OVER" || game_hp <= 0;
                let expected = if game_dead { Outcome::Defeat } else { Outcome::Victory };
                if c.outcome != Some(expected) {
                    return Ok((steps, format!("fight ended in the game ({expected:?}, hp {game_hp}) but engine outcome is {:?}", c.outcome)));
                }
                if !game_dead && game_hp != c.player.hp as i64 {
                    return Ok((steps, format!("final hp differs: game {game_hp}, engine {}", c.player.hp)));
                }
            }
            _ => {}
        }
        i += 1;
    }
    Ok((steps, String::new()))
}
