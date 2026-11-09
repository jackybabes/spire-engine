//! Replays whole-run traces recorded from the real game (tools/trace/record_run.py) through the
//! engine's `Run` and reports the first state that differs.

use serde_json::Value;
use sts_core::rngs::Rngs;
use sts_core::run::map::RoomType;
use sts_core::run::{normalize_label, Reward, Run, RunAction, RunConfig, Screen};

use crate::replay::{diff, int, s, snap_engine, snap_game, to_action, ReplayResult};

type Snap = std::collections::BTreeMap<String, String>;

fn list(v: &Value) -> String {
    v.as_array().map(|a| a.iter().map(s).collect::<Vec<_>>().join(",")).unwrap_or_default()
}

fn symbol(r: Option<RoomType>) -> &'static str {
    match r {
        Some(RoomType::Monster) => "M",
        Some(RoomType::Elite) => "E",
        Some(RoomType::Event) => "?",
        Some(RoomType::Rest) => "R",
        Some(RoomType::Shop) => "$",
        Some(RoomType::Treasure) => "T",
        Some(RoomType::Boss) => "B",
        None => " ",
    }
}

fn run_rngs(run: &Run) -> &Rngs {
    match &run.combat {
        Some(c) => &c.rng,
        None => &run.rng,
    }
}

/// Run-level state that both sides expose on every screen.
fn snap_run_game(gs: &Value) -> Snap {
    let h = &gs["harness"];
    let mut m = Snap::new();
    m.insert("floor".into(), int(gs, "floor").to_string());
    m.insert("gold".into(), int(gs, "gold").to_string());
    m.insert("deck".into(), list(&h["master_deck"]));
    m.insert("relics".into(), gs["relics"].as_array().map(|a| a.iter().map(|r| s(&r["id"])).collect::<Vec<_>>().join(",")).unwrap_or_default());
    let pots: Vec<String> = gs["potions"].as_array().map(|a| a.iter().map(|p| s(&p["id"])).filter(|p| p != "Potion Slot").collect()).unwrap_or_default();
    m.insert("potions".into(), pots.join(","));
    m.insert("boss".into(), s(&gs["act_boss"]));
    m.insert("card_blizz".into(), int(h, "card_blizz").to_string());
    m.insert("potion_blizz".into(), int(h, "potion_blizz").to_string());
    m.insert("monster_list".into(), list(&h["monster_list"]));
    m.insert("elite_list".into(), list(&h["elite_list"]));
    for t in ["common", "uncommon", "rare", "shop", "boss"] {
        m.insert(format!("relic_pool.{t}"), list(&h["relic_pools"][t]));
    }
    for k in ["monster", "event", "merchant", "card", "treasure", "relic", "potion", "monster_hp", "ai", "shuffle", "card_random", "misc"] {
        m.insert(format!("rng.{k}"), int(&h["rng"][k], "counter").to_string());
    }
    m
}

fn snap_run_engine(run: &Run) -> Snap {
    let mut m = Snap::new();
    m.insert("floor".into(), run.floor.to_string());
    let gold = run.combat.as_ref().map_or(run.gold, |c| c.player.gold);
    m.insert("gold".into(), gold.to_string());
    m.insert("deck".into(), run.deck.iter().map(|c| format!("{}+{}", c.data().java_id, c.times_upgraded)).collect::<Vec<_>>().join(","));
    let relics = run.combat.as_ref().map_or(&run.relics, |c| &c.player.relics);
    m.insert("relics".into(), relics.iter().map(|r| r.id.java_id()).collect::<Vec<_>>().join(","));
    let potions = run.combat.as_ref().map_or(&run.potions, |c| &c.potions);
    m.insert("potions".into(), potions.iter().flatten().map(|p| p.data().java_id).collect::<Vec<_>>().join(","));
    m.insert("boss".into(), run.boss.clone());
    m.insert("card_blizz".into(), run.card_blizz.to_string());
    m.insert("potion_blizz".into(), run.potion_blizz.to_string());
    m.insert("monster_list".into(), run.monster_list.join(","));
    m.insert("elite_list".into(), run.elite_list.join(","));
    let p = &run.pools;
    for (t, pool) in [("common", &p.common_relics), ("uncommon", &p.uncommon_relics), ("rare", &p.rare_relics), ("shop", &p.shop_relics), ("boss", &p.boss_relics)] {
        m.insert(format!("relic_pool.{t}"), pool.iter().map(|r| r.java_id()).collect::<Vec<_>>().join(","));
    }
    let r = run_rngs(run);
    for (k, v) in [
        ("monster", &r.monster),
        ("event", &r.event),
        ("merchant", &r.merchant),
        ("card", &r.card),
        ("treasure", &r.treasure),
        ("relic", &r.relic),
        ("potion", &r.potion),
        ("monster_hp", &r.monster_hp),
        ("ai", &r.ai),
        ("shuffle", &r.shuffle),
        ("card_random", &r.card_random),
        ("misc", &r.misc),
    ] {
        m.insert(format!("rng.{k}"), v.counter.to_string());
    }
    m
}

/// Screen contents, keyed by the game's screen type.
fn snap_screen_game(gs: &Value) -> Snap {
    let ss = &gs["screen_state"];
    let mut m = Snap::new();
    m.insert("hp".into(), format!("{}/{}", int(gs, "current_hp"), int(gs, "max_hp")));
    match s(&gs["screen_type"]).as_str() {
        "EVENT" if ss["event_id"] != "Neow Event" => {
            m.insert("event.id".into(), s(&ss["event_id"]));
            let opts: Vec<String> = ss["options"].as_array().map(|a| a.iter().map(|o| if o["disabled"].as_bool().unwrap_or(false) { "0".into() } else { "1".into() }).collect()).unwrap_or_default();
            m.insert("event.options".into(), opts.join(""));
        }
        "EVENT" => {
            let opts: Vec<String> = ss["options"].as_array().map(|a| a.iter().map(|o| normalize_label(&s(&o["text"]))).collect()).unwrap_or_default();
            m.insert("neow.options".into(), opts.join(" | "));
        }
        "MAP" => {
            let nodes: Vec<String> = ss["next_nodes"].as_array().map(|a| a.iter().map(|n| format!("{},{}{}", int(n, "x"), int(n, "y"), s(&n["symbol"]))).collect()).unwrap_or_default();
            m.insert("map.next".into(), nodes.join(" "));
        }
        "COMBAT_REWARD" => {
            let items: Vec<String> = ss["rewards"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|r| match s(&r["reward_type"]).as_str() {
                            "GOLD" => format!("GOLD {}", int(r, "gold")),
                            "STOLEN_GOLD" => format!("STOLEN_GOLD {}", int(r, "gold")),
                            "RELIC" => format!("RELIC {}", s(&r["relic"]["id"])),
                            "POTION" => format!("POTION {}", s(&r["potion"]["id"])),
                            other => other.to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            m.insert("rewards".into(), items.join(", "));
        }
        "CARD_REWARD" => {
            let cards: Vec<String> = ss["cards"].as_array().map(|a| a.iter().map(|c| format!("{}+{}", s(&c["id"]), int(c, "upgrades"))).collect()).unwrap_or_default();
            m.insert("card_reward".into(), cards.join(","));
        }
        "BOSS_REWARD" => {
            let relics: Vec<String> = ss["relics"].as_array().map(|a| a.iter().map(|r| s(&r["id"])).collect()).unwrap_or_default();
            m.insert("boss_relics".into(), relics.join(","));
        }
        "SHOP_SCREEN" => {
            let items = |k: &str, card: bool| -> String {
                ss[k]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|x| if card { format!("{}+{}@{}", s(&x["id"]), int(x, "upgrades"), int(x, "price")) } else { format!("{}@{}", s(&x["id"]), int(x, "price")) })
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .unwrap_or_default()
            };
            m.insert("shop.cards".into(), items("cards", true));
            m.insert("shop.relics".into(), items("relics", false));
            m.insert("shop.potions".into(), items("potions", false));
            m.insert("shop.purge".into(), format!("{} {}", ss["purge_available"], int(ss, "purge_cost")));
        }
        _ => {}
    }
    m
}

fn snap_screen_engine(run: &Run, neow_hp: (i32, i32), shop_open: bool) -> Snap {
    let mut m = Snap::new();
    m.insert("hp".into(), format!("{}/{}", run.hp, run.max_hp));
    match &run.screen {
        Screen::Neow { options } => {
            let opts: Vec<String> = options.iter().map(|o| normalize_label(&o.label(neow_hp.1, neow_hp.0))).collect();
            m.insert("neow.options".into(), opts.join(" | "));
        }
        Screen::Map => {
            let y = run.pos.map_or(0, |(_, y)| y + 1);
            let nodes: Vec<String> = if y >= 15 {
                Vec::new()
            } else {
                run.reachable_nodes().iter().map(|&x| format!("{x},{y}{}", symbol(run.map.node(x, y).room))).collect()
            };
            m.insert("map.next".into(), nodes.join(" "));
        }
        Screen::Rewards { items } => {
            let items: Vec<String> = items
                .iter()
                .map(|r| match r {
                    Reward::Gold(g) => format!("GOLD {}", run.gold_with_bonus(*g)),
                    Reward::StolenGold(g) => format!("STOLEN_GOLD {g}"),
                    Reward::Relic(id) => format!("RELIC {}", id.java_id()),
                    Reward::Potion(p) => format!("POTION {}", p.data().java_id),
                    Reward::Cards(_) => "CARD".into(),
                    Reward::EmeraldKey => "EMERALD_KEY".into(),
                    Reward::SapphireKey => "SAPPHIRE_KEY".into(),
                })
                .collect();
            m.insert("rewards".into(), items.join(", "));
        }
        Screen::CardReward { cards, .. } => {
            m.insert("card_reward".into(), cards.iter().map(|c| format!("{}+{}", c.data().java_id, c.times_upgraded)).collect::<Vec<_>>().join(","));
        }
        Screen::BossRelic { relics } => {
            m.insert("boss_relics".into(), relics.iter().map(|r| r.java_id()).collect::<Vec<_>>().join(","));
        }
        Screen::Event { event } => {
            m.insert("event.id".into(), event.id.key().to_string());
            m.insert("event.options".into(), event.options.iter().map(|&o| if o { "1" } else { "0" }).collect::<String>());
        }
        Screen::Shop { shop } if shop_open => {
            let cards: Vec<String> = shop.cards.iter().flatten().map(|c| format!("{}+{}@{}", c.card.data().java_id, c.card.times_upgraded, c.price)).collect();
            m.insert("shop.cards".into(), cards.join(","));
            let relics: Vec<String> = shop.relics.iter().flatten().map(|r| format!("{}@{}", r.relic.java_id(), r.price)).collect();
            m.insert("shop.relics".into(), relics.join(","));
            let potions: Vec<String> = shop.potions.iter().flatten().map(|p| format!("{}@{}", p.potion.data().java_id, p.price)).collect();
            m.insert("shop.potions".into(), potions.join(","));
            m.insert("shop.purge".into(), format!("{} {}", shop.purge_available, shop.purge_cost));
        }
        _ => {}
    }
    m
}

/// The game's `getAvailableShopItems` order (what `choose i` on the shop screen picks).
fn shop_items(run: &Run, shop: &sts_core::run::shop::Shop) -> Vec<RunAction> {
    let mut out = Vec::new();
    if shop.purge_available && run.gold >= shop.purge_cost {
        out.push(RunAction::Purge);
    }
    for (i, c) in shop.cards.iter().enumerate() {
        if c.as_ref().is_some_and(|c| c.price <= run.gold) {
            out.push(RunAction::BuyCard { index: i });
        }
    }
    for (i, r) in shop.relics.iter().enumerate() {
        if r.as_ref().is_some_and(|r| r.price <= run.gold) {
            out.push(RunAction::BuyRelic { index: i });
        }
    }
    for (i, p) in shop.potions.iter().enumerate() {
        if p.as_ref().is_some_and(|p| p.price <= run.gold) {
            out.push(RunAction::BuyPotion { index: i });
        }
    }
    out
}

/// Whether the game's screen is the one the engine is on (so their contents can be compared).
fn same_screen(gs: &Value, run: &Run) -> bool {
    let screen = s(&gs["screen_type"]);
    let in_combat = gs["room_phase"] == "COMBAT" && gs.get("combat_state").is_some();
    match &run.screen {
        Screen::Neow { options } => screen == "EVENT" && gs["screen_state"]["options"].as_array().map_or(0, |a| a.len()) == options.len(),
        Screen::Map => screen == "MAP",
        Screen::Combat => in_combat,
        Screen::Rewards { .. } => screen == "COMBAT_REWARD",
        Screen::CardReward { .. } => screen == "CARD_REWARD",
        Screen::Grid { .. } => screen == "GRID",
        Screen::Rest => screen == "REST" && gs["choice_list"].as_array().is_some_and(|a| !a.is_empty()),
        Screen::BossRelic { .. } => screen == "BOSS_REWARD",
        Screen::Shop { .. } => screen == "SHOP_SCREEN" || screen == "SHOP_ROOM",
        Screen::Event { .. } => screen == "EVENT" && gs["screen_state"]["event_id"] != "Neow Event",
        Screen::Unsupported { .. } | Screen::Over => false,
    }
}

/// Translates a recorded command (sent while the game showed `gs`) into engine actions.
fn to_run_actions(cmd: &str, gs: &Value, run: &Run, grid_picks: &mut Vec<usize>) -> Result<Vec<RunAction>, String> {
    let screen = s(&gs["screen_type"]);
    let t: Vec<&str> = cmd.split_whitespace().collect();
    let arg = t.get(1).copied().unwrap_or("");
    let idx = || arg.parse::<usize>().map_err(|_| format!("bad index in `{cmd}`"));
    Ok(match (&run.screen, screen.as_str(), t.first().copied().unwrap_or("")) {
        (s, _, "potion") if !matches!(s, Screen::Combat) => {
            let slot: usize = t.get(2).ok_or("potion needs a slot")?.parse().map_err(|_| "bad potion slot")?;
            match t.get(1).copied() {
                Some("use") => vec![RunAction::UsePotion { slot }],
                _ => vec![RunAction::DiscardPotion { slot }],
            }
        }
        (Screen::Combat, _, _) => {
            let c = run.combat.as_ref().ok_or("engine has no combat")?;
            to_action(cmd, c, grid_picks)?.map(|a| vec![RunAction::Combat { action: a }]).unwrap_or_default()
        }
        (Screen::Neow { options }, "EVENT", "choose") if gs["screen_state"]["options"].as_array().map_or(0, |a| a.len()) == options.len() => {
            vec![RunAction::Neow { option: idx()? }]
        }
        (Screen::Map, "MAP", "choose") => {
            if arg == "boss" {
                vec![RunAction::Path { x: run.next_nodes()[0] }]
            } else {
                let n = &gs["screen_state"]["next_nodes"][idx()?];
                vec![RunAction::Path { x: int(n, "x") as i32 }]
            }
        }
        (Screen::Event { event }, "EVENT", "choose") => {
            let enabled: Vec<usize> = (0..event.options.len()).filter(|&i| event.options[i]).collect();
            vec![RunAction::EventChoice { index: *enabled.get(idx()?).ok_or("event choice out of range")? }]
        }
        (Screen::Shop { shop }, "SHOP_SCREEN", "choose") => {
            let items = shop_items(run, shop);
            vec![items.get(idx()?).cloned().ok_or("shop choice out of range")?]
        }
        (Screen::Shop { .. }, "SHOP_ROOM", "proceed") => vec![RunAction::Proceed],
        (Screen::Rewards { .. }, "COMBAT_REWARD", "choose") => vec![RunAction::TakeReward { index: idx()? }],
        (Screen::Rewards { .. }, "COMBAT_REWARD", "proceed") => vec![RunAction::Proceed],
        (Screen::CardReward { .. }, "CARD_REWARD", "choose") => vec![RunAction::PickCard { index: idx()? }],
        (Screen::CardReward { .. }, "CARD_REWARD", "skip" | "proceed") => vec![RunAction::SkipCards],
        (Screen::Rest, "REST", "choose") => match arg {
            "rest" => vec![RunAction::Rest],
            "smith" => vec![RunAction::Smith],
            "lift" => vec![RunAction::Lift],
            "toke" => vec![RunAction::Toke],
            "dig" => vec![RunAction::Dig],
            "recall" => vec![RunAction::Recall],
            other => return Err(format!("unsupported rest option {other}")),
        },
        (Screen::BossRelic { .. }, "BOSS_REWARD", "choose") => vec![RunAction::BossRelic { index: idx()? }],
        (Screen::BossRelic { .. }, "BOSS_REWARD", "skip") => vec![RunAction::Proceed],
        // The Library's cards are a grid in the game and event options in the engine.
        (Screen::Event { event }, "GRID", "choose") if event.id == sts_core::run::events::EventId::TheLibrary => {
            vec![RunAction::EventChoice { index: idx()? }]
        }
        (Screen::Grid { count, .. }, "GRID", "choose") => {
            let card = &gs["screen_state"]["cards"][idx()?];
            let uuid = s(&card["uuid"]);
            let deck = gs["deck"].as_array().ok_or("no deck in state")?;
            let pos = deck.iter().position(|c| s(&c["uuid"]) == uuid).ok_or("grid card not in deck")?;
            grid_picks.push(pos);
            if grid_picks.len() == *count {
                vec![RunAction::Select { indices: std::mem::take(grid_picks) }]
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    })
}

pub fn replay_run(text: &str, verbose: bool) -> ReplayResult {
    replay_run_observed(text, verbose, &mut crate::coverage::NoObserver)
}

/// `replay_run`, telling `obs` about every engine step (`sts-cli coverage`).
pub fn replay_run_observed(text: &str, verbose: bool, obs: &mut dyn crate::coverage::Observer) -> ReplayResult {
    match replay_run_inner(text, verbose, obs) {
        Ok((steps, msg)) => ReplayResult { ok: msg.is_empty() || msg.starts_with("stopped:"), steps, message: msg },
        Err(e) => ReplayResult { ok: false, steps: 0, message: e },
    }
}

fn replay_run_inner(text: &str, verbose: bool, obs: &mut dyn crate::coverage::Observer) -> Result<(usize, String), String> {
    let recs: Vec<Value> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    let states: Vec<&Value> = recs.iter().filter(|r| r["type"] == "state").map(|r| &r["state"]).collect();
    let first = *states.first().ok_or("no states")?;
    let seed: i64 = s(&first["harness"]["seed"]).parse().map_err(|_| "bad seed")?;
    let mut cfg = RunConfig::new(seed);
    cfg.character = sts_core::character::Character::from_name(&s(&first["class"])).unwrap_or_default();
    cfg.ascension = int(first, "ascension_level") as i32;
    // Four blessings once the profile has beaten a boss, two before.
    let neow_count = states
        .iter()
        .filter(|g| s(&g["screen_type"]) == "EVENT")
        .map(|g| g["screen_state"]["options"].as_array().map_or(0, |a| a.len()))
        .max()
        .unwrap_or(0);
    cfg.neow_full = neow_count >= 4;
    // The final act was on if a rest site offered Recall or a key was ever offered.
    cfg.final_act = recs.iter().any(|r| r["type"] == "setup" && r["final_act"] == true)
        || states.iter().any(|g| {
            g["choice_list"].as_array().is_some_and(|a| a.iter().any(|c| c == "recall"))
                || g["screen_state"]["rewards"]
                    .as_array()
                    .is_some_and(|a| a.iter().any(|r| r["reward_type"] == "EMERALD_KEY" || r["reward_type"] == "SAPPHIRE_KEY"))
        });
    let mut run = Run::new(cfg.clone());
    obs.run_start(&cfg);
    let neow_hp = (run.hp, run.max_hp);
    let mut steps = 0;
    let mut last_cmd = String::from("(run start)");
    let mut prev_state: Option<&Value> = None;
    let mut grid_picks = Vec::new();
    let mut compared = 0;
    let mut after_neow = false;
    for (i, r) in recs.iter().enumerate() {
        match r["type"].as_str() {
            Some("state") => {
                let gs = &r["state"];
                // A state sent while the game was busy (only `wait` offered) is not a decision point.
                let busy = r["available"].as_array().is_some_and(|a| !a.iter().any(|c| matches!(c.as_str(), Some("play" | "end" | "choose" | "confirm" | "proceed" | "skip" | "leave" | "return" | "potion"))));
                if busy {
                    continue;
                }
                if let Some(p) = prev_state {
                    after_neow = s(&p["screen_type"]) == "EVENT" || (after_neow && s(&gs["screen_type"]) == "MAP" && int(gs, "floor") == 0);
                }
                prev_state = Some(gs);
                if !same_screen(gs, &run) {
                    continue;
                }
                let mut d = Vec::new();
                if let (Screen::Combat, Some(c)) = (&run.screen, &run.combat) {
                    let next_cmd = recs.get(i + 1).filter(|n| n["type"] == "command").map(|n| s(&n["command"])).unwrap_or_default();
                    if s(&gs["screen_type"]) == "NONE" && c.pending.is_none() && !next_cmd.starts_with("confirm") {
                        d.extend(diff(&snap_game(gs), &snap_engine(c, true)));
                    }
                } else {
                    let shop_open = s(&gs["screen_type"]) == "SHOP_SCREEN";
                    d.extend(diff(&snap_screen_game(gs), &snap_screen_engine(&run, neow_hp, shop_open)));
                }
                let mut run_diff = diff(&snap_run_game(gs), &snap_run_engine(&run));
                if after_neow {
                    // Neow's cards arrive through ShowCardAndObtainEffect, which can still be
                    // animating on the first map screen; later screens check the deck.
                    run_diff.retain(|l| !l.starts_with("  deck:"));
                }
                if last_cmd == "confirm" {
                    // A grid's transform adds its new card through ShowCardAndObtainEffect too: right
                    // after the confirm the game's deck may still lack cards the engine has (and
                    // only that); the next screens check the deck again.
                    let game_deck = snap_run_game(gs).get("deck").cloned().unwrap_or_default();
                    let engine_deck = snap_run_engine(&run).get("deck").cloned().unwrap_or_default();
                    if sub_multiset(&game_deck, &engine_deck) {
                        run_diff.retain(|l| !l.starts_with("  deck:"));
                    }
                }
                d.extend(run_diff);
                compared += 1;
                if !d.is_empty() {
                    return Ok((steps, format!("floor {} after `{last_cmd}` (step {steps}):\n{}", run.floor, d.join("\n"))));
                }
            }
            Some("command") => {
                let cmd = s(&r["command"]);
                let gs = prev_state.ok_or("command before any state")?;
                if let Some(json) = cmd.strip_prefix("harness ") {
                    // The recorder's HP override (record_run.py), mirrored into the engine.
                    let v: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
                    if let Some(m) = v.get("max_hp").and_then(|x| x.as_i64()) {
                        run.max_hp = m as i32;
                    }
                    if let Some(h) = v.get("hp").and_then(|x| x.as_i64()) {
                        run.hp = h as i32;
                    }
                    obs.run_hp(run.hp, run.max_hp);
                    // record_run.py --relics: the harness replaces the relics with this list, each
                    // new one picked up (instantObtain, so its onEquip runs).
                    if let Some(list) = v.get("relics").and_then(|x| x.as_array()) {
                        let ids: Vec<sts_core::relics::RelicId> =
                            list.iter().map(|r| sts_core::relics::RelicId::from_java_id(&s(r)).ok_or(format!("relic {} not ported", s(r)))).collect::<Result<_, _>>()?;
                        set_relics(&mut run, &ids);
                        obs.run_relics(&ids);
                    }
                    last_cmd = cmd;
                    continue;
                }
                let acts = to_run_actions(&cmd, gs, &run, &mut grid_picks)?;
                for a in acts {
                    if verbose {
                        eprintln!("  floor {} {:?}: {a:?}", run.floor, std::mem::discriminant(&run.screen));
                    }
                    obs.run_before(&run, &a);
                    run.step(&a).map_err(|e| format!("engine rejected `{cmd}` ({a:?}) at step {steps}: {e}"))?;
                    obs.run_after(&run);
                    steps += 1;
                }
                last_cmd = cmd;
                if let Screen::Unsupported { what } = &run.screen {
                    return Ok((steps, format!("stopped: engine reached an unsupported room ({what}) on floor {}; {compared} states compared", run.floor)));
                }
            }
            Some("end") => {
                return Ok((steps, format!("stopped: {} on floor {}; {compared} states compared", s(&r["why"]), run.floor)));
            }
            _ => {}
        }
    }
    Ok((steps, "stopped: trace ended".into()))
}

/// The harness's relic list: the relics kept in its order, new ones picked up.
pub(crate) fn set_relics(run: &mut Run, ids: &[sts_core::relics::RelicId]) {
    let old = std::mem::take(&mut run.relics);
    for &id in ids {
        match old.iter().find(|r| r.id == id) {
            Some(r) => run.relics.push(r.clone()),
            None => run.obtain_relic(id),
        }
    }
}

/// Whether every card of `small` (a comma-separated deck) is in `big`, counting copies.
fn sub_multiset(small: &str, big: &str) -> bool {
    let mut rest: Vec<&str> = big.split(',').collect();
    for c in small.split(',').filter(|c| !c.is_empty()) {
        match rest.iter().position(|x| *x == c) {
            Some(i) => {
                rest.remove(i);
            }
            None => return false,
        }
    }
    true
}
