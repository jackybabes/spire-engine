//! The in-game advisor's half of the encoding: a CommunicationMod game state (the JSON the real
//! game sends) turned into the run network's inputs, with a readable label for every allowed
//! action slot.
//!
//! Between fights the state is rebuilt as an engine `Run` on that screen (deck, relics, potions,
//! HP, gold, map...) and encoded by `encode_run`, as in training. In a fight the fight fields are
//! written straight from the JSON, mirroring `encode::encode_fight` field for field, with the
//! game's own playability and targets for the mask (the engine's run fields come from the rebuilt
//! `Run`). Some inputs can only be approximated from the JSON: a card's damage, block and magic
//! are its base numbers, an event's internal screen and numbers are unknown, cards and attacks
//! played this turn are 0, the last card type played is unknown (0), as are a card's Retain given
//! in the fight (Meditate, Well-Laid Plans), the Lightning channeled this combat (Thunder
//! Strike's hits) and an extra turn coming (Vault).

use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::Value;
use sts_core::cards::{Card, CardId, CARD_DATA};
use sts_core::combat::{Orb, OrbId, Stance};
use sts_core::monsters::MonsterId;
use sts_core::potions::{PotionId, POTION_DATA};
use sts_core::powers::{Power, PowerId};
use sts_core::relics::{RelicId, RelicState, RELIC_DATA};
use sts_core::run::map::{Map, MapNode, RoomType, MAP_HEIGHT, MAP_WIDTH};
use sts_core::run::shop::{Shop, ShopCard, ShopPotion, ShopRelic};
use sts_core::run::events::{Event, EventId};
use sts_core::run::{normalize_label, GridPurpose, NeowDrawback, NeowOption, NeowReward, Reward, Run, RunAction, RunConfig, Screen};

use sts_core::{Combat, CombatConfig};

use crate::encode::{self, Writer, CONFIRM, END_TURN, HAND_FEATS, MAX_HAND, MAX_MONSTERS, MAX_MONSTER_POWERS, MAX_OPTIONS, MAX_ORBS, MAX_PILE, MAX_PLAYER_POWERS, MAX_POTIONS, MAX_RELICS, MOVES_SEEN, PICK, PLAY, POTION, SCALARS, TARGETS};
use crate::encode_run::{self, OptAction};

/// The network's inputs for one game state, and what each allowed slot does.
pub struct Advice {
    pub ints: Vec<i32>,
    pub floats: Vec<f32>,
    pub mask: Vec<u8>,
    /// (action slot, label) for every allowed slot.
    pub labels: Vec<(usize, String)>,
}

struct Names {
    cards: HashMap<&'static str, CardId>,
    relics: HashMap<&'static str, RelicId>,
    potions: HashMap<&'static str, PotionId>,
    powers: HashMap<&'static str, PowerId>,
    monsters: HashMap<&'static str, MonsterId>,
    events: HashMap<&'static str, EventId>,
}

fn names() -> &'static Names {
    static N: OnceLock<Names> = OnceLock::new();
    N.get_or_init(|| Names {
        cards: CARD_DATA.iter().map(|c| (c.java_id, c.id)).collect(),
        relics: RELIC_DATA.iter().map(|r| (r.java_id, r.id)).collect(),
        potions: POTION_DATA.iter().map(|p| (p.java_id, p.id)).collect(),
        powers: PowerId::all().map(|p| (Power::new(p, 0).java_id(), p)).collect(),
        monsters: MonsterId::all().map(|m| (m.java_id(), m)).collect(),
        events: EventId::all().map(|e| (e.key(), e)).collect(),
    })
}

/// `AbstractMonster.Intent`, in the engine's (and the game's) order.
const INTENTS: [&str; 17] = [
    "ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND", "BUFF", "DEBUFF", "STRONG_DEBUFF", "DEBUG", "DEFEND", "DEFEND_DEBUFF",
    "DEFEND_BUFF", "ESCAPE", "MAGIC", "NONE", "SLEEP", "STUN", "UNKNOWN",
];

fn int(v: &Value) -> i32 {
    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).unwrap_or(0) as i32
}

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

fn arr(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}

/// A card from its JSON (id and upgrades), or None for a card the engine doesn't have.
fn card(run: &mut Run, v: &Value) -> Option<Card> {
    let id = *names().cards.get(s(&v["id"]))?;
    let mut c = run.new_card(id);
    for _ in 0..int(&v["upgrades"]) {
        c.upgrade();
    }
    if let Some(m) = v["misc"].as_i64() {
        c.misc = m as i32;
    }
    Some(c)
}

fn card_name(v: &Value) -> String {
    let n = s(&v["name"]);
    if n.is_empty() { s(&v["id"]).to_string() } else { n.to_string() }
}

fn room_type(name: &str) -> Option<RoomType> {
    Some(match name {
        "MonsterRoom" => RoomType::Monster,
        "MonsterRoomElite" => RoomType::Elite,
        "MonsterRoomBoss" => RoomType::Boss,
        "EventRoom" | "NeowRoom" => RoomType::Event,
        "RestRoom" => RoomType::Rest,
        "ShopRoom" => RoomType::Shop,
        "TreasureRoom" | "TreasureRoomBoss" => RoomType::Treasure,
        _ => return None,
    })
}

fn symbol_room(sym: &str) -> Option<RoomType> {
    Some(match sym {
        "M" => RoomType::Monster,
        "E" => RoomType::Elite,
        "?" => RoomType::Event,
        "R" => RoomType::Rest,
        "$" => RoomType::Shop,
        "T" => RoomType::Treasure,
        "B" => RoomType::Boss,
        _ => return None,
    })
}

fn room_word(r: Option<RoomType>) -> &'static str {
    match r {
        Some(RoomType::Monster) => "fight",
        Some(RoomType::Elite) => "elite",
        Some(RoomType::Event) => "? room",
        Some(RoomType::Rest) => "rest site",
        Some(RoomType::Shop) => "shop",
        Some(RoomType::Treasure) => "treasure",
        Some(RoomType::Boss) => "boss",
        None => "room",
    }
}

/// The run as far as the game state shows it (on the map screen, set per screen after).
fn base_run(gs: &Value) -> (Run, HashMap<String, usize>) {
    let mut cfg = RunConfig::new(gs["seed"].as_i64().unwrap_or(0));
    cfg.ascension = int(&gs["ascension_level"]);
    // `class`: IRONCLAD, THE_SILENT, DEFECT or WATCHER.
    cfg.character = sts_core::character::Character::from_name(s(&gs["class"])).unwrap_or_default();
    cfg.final_act = true;
    let mut run = Run::new(cfg);
    run.act = int(&gs["act"]).max(1);
    run.floor = int(&gs["floor"]);
    run.max_hp = int(&gs["max_hp"]).max(1);
    run.hp = int(&gs["current_hp"]).clamp(0, run.max_hp);
    run.gold = int(&gs["gold"]);
    run.room = room_type(s(&gs["room_type"]));
    if let Some(b) = gs["act_boss"].as_str() {
        run.boss = b.to_string();
    }
    run.keys.ruby = gs["keys"]["ruby"].as_bool().unwrap_or(false);
    run.keys.emerald = gs["keys"]["emerald"].as_bool().unwrap_or(false);
    run.keys.sapphire = gs["keys"]["sapphire"].as_bool().unwrap_or(false);
    run.deck.clear();
    let mut by_uuid = HashMap::new();
    for v in arr(&gs["deck"]) {
        if let Some(c) = card(&mut run, v) {
            by_uuid.insert(s(&v["uuid"]).to_string(), run.deck.len());
            run.deck.push(c);
        }
    }
    run.relics.clear();
    for v in arr(&gs["relics"]) {
        if let Some(&id) = names().relics.get(s(&v["id"])) {
            let mut r = RelicState::new(id);
            r.counter = int(&v["counter"]);
            run.relics.push(r);
        }
    }
    run.potions = arr(&gs["potions"]).iter().map(|p| names().potions.get(s(&p["id"])).copied()).collect();
    if let Some(c) = gs["screen_state"]["purge_cost"].as_i64() {
        run.purge_cost = c as i32;
    }
    run.screen = Screen::Map;
    (run, by_uuid)
}

/// The act's map from the game state (`map`: nodes with their children).
fn build_map(gs: &Value) -> Map {
    let mut rows: Vec<Vec<MapNode>> = (0..MAP_HEIGHT as i32)
        .map(|y| (0..MAP_WIDTH as i32).map(|x| MapNode { x, y, room: None, edges: Vec::new(), parents: Vec::new(), has_emerald_key: false }).collect())
        .collect();
    for n in arr(&gs["map"]) {
        let (x, y) = (int(&n["x"]), int(&n["y"]));
        if !(0..MAP_WIDTH as i32).contains(&x) || !(0..MAP_HEIGHT as i32).contains(&y) {
            continue;
        }
        let node = &mut rows[y as usize][x as usize];
        node.room = symbol_room(s(&n["symbol"]));
        node.edges = arr(&n["children"]).iter().map(|c| (int(&c["x"]), int(&c["y"]))).collect();
        node.edges.sort_unstable();
        node.parents = arr(&n["parents"]).iter().map(|c| (int(&c["x"]), int(&c["y"]))).collect();
    }
    Map { rows }
}

/// The Neow option whose button text is `text` (the game's labels; HP as now).
fn neow_option(text: &str, max_hp: i32, hp: i32) -> Option<NeowOption> {
    let want = normalize_label(text);
    for reward in NeowReward::all() {
        for drawback in NeowDrawback::all() {
            let o = NeowOption { reward, drawback };
            if normalize_label(&o.label(max_hp, hp)) == want {
                return Some(o);
            }
        }
    }
    None
}

/// The game state as an engine screen (between fights), or None when there's no decision the
/// network makes (a chest, the shop room, a finished screen).
fn set_screen(run: &mut Run, gs: &Value, by_uuid: &HashMap<String, usize>, picked: &mut Vec<usize>) -> Option<Vec<String>> {
    let st = &gs["screen_state"];
    let mut event_labels = Vec::new();
    run.screen = match s(&gs["screen_type"]) {
        "CARD_REWARD" => {
            let cards: Vec<Card> = arr(&st["cards"]).iter().filter_map(|c| card(run, c)).collect();
            let skip = st["skip_available"].as_bool().unwrap_or(true);
            Screen::CardReward { item: if skip { Some(0) } else { None }, cards }
        }
        "COMBAT_REWARD" => {
            let mut items = Vec::new();
            for r in arr(&st["rewards"]) {
                items.push(match s(&r["reward_type"]) {
                    "GOLD" => Reward::Gold(int(&r["gold"])),
                    "STOLEN_GOLD" => Reward::StolenGold(int(&r["gold"])),
                    "RELIC" => match names().relics.get(s(&r["relic"]["id"])) {
                        Some(&id) => Reward::Relic(id),
                        None => continue,
                    },
                    "POTION" => match names().potions.get(s(&r["potion"]["id"])) {
                        Some(&id) => Reward::Potion(id),
                        None => continue,
                    },
                    "CARD" => {
                        let c = run.new_card(CardId::StrikeR);
                        Reward::Cards(vec![c.clone(), c.clone(), c])
                    }
                    "EMERALD_KEY" => Reward::EmeraldKey,
                    "SAPPHIRE_KEY" => Reward::SapphireKey,
                    _ => continue,
                });
            }
            Screen::Rewards { items }
        }
        "MAP" => {
            run.map = build_map(gs);
            let cur = &st["current_node"];
            run.pos = if cur.is_object() && int(&cur["y"]) >= 0 { Some((int(&cur["x"]), int(&cur["y"]))) } else { None };
            Screen::Map
        }
        "REST" => Screen::Rest,
        "BOSS_REWARD" => Screen::BossRelic { relics: arr(&st["relics"]).iter().filter_map(|r| names().relics.get(s(&r["id"])).copied()).collect() },
        "GRID" => {
            let options: Vec<usize> = arr(&st["cards"]).iter().filter_map(|c| by_uuid.get(s(&c["uuid"])).copied()).collect();
            *picked = arr(&st["selected_cards"]).iter().filter_map(|c| by_uuid.get(s(&c["uuid"])).copied()).collect();
            let purpose = if st["for_upgrade"].as_bool() == Some(true) {
                if run.room == Some(RoomType::Rest) { GridPurpose::Smith } else { GridPurpose::Upgrade }
            } else if st["for_transform"].as_bool() == Some(true) {
                GridPurpose::Transform
            } else if run.room == Some(RoomType::Shop) {
                GridPurpose::ShopPurge
            } else if run.room == Some(RoomType::Rest) {
                GridPurpose::Toke
            } else {
                GridPurpose::Remove
            };
            Screen::Grid { purpose, options, count: int(&st["num_cards"]).max(1) as usize }
        }
        "SHOP_SCREEN" => {
            let cards = arr(&st["cards"]).iter().map(|c| card(run, c).map(|card| ShopCard { card, price: int(&c["price"]) })).collect();
            let relics = arr(&st["relics"]).iter().map(|r| names().relics.get(s(&r["id"])).map(|&relic| ShopRelic { relic, price: int(&r["price"]) })).collect();
            let potions = arr(&st["potions"]).iter().map(|p| names().potions.get(s(&p["id"])).map(|&potion| ShopPotion { potion, price: int(&p["price"]) })).collect();
            let purge_available = st["purge_available"].as_bool().unwrap_or(false);
            Screen::Shop { shop: Box::new(Shop { cards, relics, potions, purge_available, purge_cost: run.purge_cost }) }
        }
        "EVENT" => {
            let opts = arr(&st["options"]);
            event_labels = opts.iter().map(|o| normalize_label(s(&o["text"]))).collect();
            if s(&st["event_id"]) == "Neow Event" || s(&st["event_name"]) == "Neow" {
                let options: Vec<NeowOption> = opts.iter().filter_map(|o| neow_option(s(&o["text"]), run.max_hp, run.hp)).collect();
                if options.len() != opts.len() || options.is_empty() {
                    return None; // talking to Neow, or a mini blessing the labels don't match
                }
                Screen::Neow { options }
            } else {
                let id = names().events.get(s(&st["event_id"])).or_else(|| names().events.get(s(&st["event_name"])))?;
                let mut e = Event::new(*id);
                e.options = opts.iter().map(|o| !o["disabled"].as_bool().unwrap_or(false)).collect();
                e.labels = event_labels.clone();
                Screen::Event { event: Box::new(e) }
            }
        }
        _ => return None,
    };
    Some(event_labels)
}

fn describe_run(run: &Run, a: &OptAction, event_labels: &[String]) -> String {
    match a {
        OptAction::GridPick(d) => run.deck.get(*d).map_or("a card".into(), |c| format!("{} {}", grid_verb(&run.screen), c.name())),
        OptAction::Run(a) => match (a, &run.screen) {
            (RunAction::Neow { option }, Screen::Neow { options }) => normalize_label(&options[*option].label(run.max_hp, run.hp)),
            (RunAction::Path { x }, _) => {
                let y = run.pos.map_or(0, |(_, y)| y + 1);
                let room = if (y as usize) < MAP_HEIGHT { run.map.node(*x, y).room } else { Some(RoomType::Boss) };
                format!("Go to the {} (column {})", room_word(room), x + 1)
            }
            (RunAction::TakeReward { index }, Screen::Rewards { items }) => match &items[*index] {
                Reward::Gold(n) | Reward::StolenGold(n) => format!("Take {n} gold"),
                Reward::Relic(r) => format!("Take {}", r.data().name),
                Reward::Potion(p) => format!("Take {}", p.data().name),
                Reward::Cards(_) => "Open the card reward".into(),
                Reward::EmeraldKey => "Take the emerald key".into(),
                Reward::SapphireKey => "Take the sapphire key".into(),
            },
            (RunAction::PickCard { index }, Screen::CardReward { cards, .. }) => format!("Take {}", cards[*index].name()),
            (RunAction::SingingBowl, _) => "Singing Bowl (+2 max HP)".into(),
            (RunAction::Rest, _) => "Rest".into(),
            (RunAction::Smith, _) => "Upgrade a card".into(),
            (RunAction::Lift, _) => "Lift (Girya)".into(),
            (RunAction::Toke, _) => "Remove a card (Peace Pipe)".into(),
            (RunAction::Dig, _) => "Dig (Shovel)".into(),
            (RunAction::Recall, _) => "Recall (ruby key)".into(),
            (RunAction::BossRelic { index }, Screen::BossRelic { relics }) => format!("Take {}", relics[*index].data().name),
            (RunAction::BuyCard { index }, Screen::Shop { shop }) => shop.cards[*index].as_ref().map_or("Buy a card".into(), |c| format!("Buy {} ({}g)", c.card.name(), c.price)),
            (RunAction::BuyRelic { index }, Screen::Shop { shop }) => shop.relics[*index].as_ref().map_or("Buy a relic".into(), |r| format!("Buy {} ({}g)", r.relic.data().name, r.price)),
            (RunAction::BuyPotion { index }, Screen::Shop { shop }) => shop.potions[*index].as_ref().map_or("Buy a potion".into(), |p| format!("Buy {} ({}g)", p.potion.data().name, p.price)),
            (RunAction::Purge, Screen::Shop { shop }) => format!("Remove a card ({}g)", shop.purge_cost),
            (RunAction::EventChoice { index }, _) => event_labels.get(*index).cloned().unwrap_or_else(|| format!("Option {}", index + 1)),
            (a, _) => format!("{a:?}"),
        },
    }
}

fn grid_verb(screen: &Screen) -> &'static str {
    match screen {
        Screen::Grid { purpose: GridPurpose::Upgrade | GridPurpose::Smith, .. } => "Upgrade",
        Screen::Grid { purpose: GridPurpose::Transform, .. } => "Transform",
        _ => "Remove",
    }
}

fn confirm_label(run: &Run) -> String {
    match &run.screen {
        Screen::CardReward { .. } => "Skip the cards".into(),
        Screen::Shop { .. } => "Leave the shop".into(),
        Screen::BossRelic { .. } => "Skip the boss relic".into(),
        Screen::Rewards { .. } => "Leave the rewards".into(),
        Screen::Grid { .. } => "Confirm / skip".into(),
        _ => "Proceed".into(),
    }
}

/// Encodes a CommunicationMod `game_state` with its `available_commands`.
pub fn advise(gs: &Value, available: &[String]) -> Result<Advice, String> {
    let (ni, nf) = (encode::int_len() + encode_run::run_int_len(), encode::float_len() + encode_run::run_float_len());
    let mut out = Advice { ints: vec![0; ni], floats: vec![0.0; nf], mask: vec![0; encode::ACTIONS], labels: Vec::new() };
    let (mut run, by_uuid) = base_run(gs);
    // A fight's own screens: its turns, and card choices made during it.
    let in_fight = gs["combat_state"].is_object()
        && s(&gs["room_phase"]) == "COMBAT"
        && matches!(s(&gs["screen_type"]), "NONE" | "HAND_SELECT" | "GRID" | "CARD_REWARD");
    if in_fight {
        run.screen = Screen::Combat;
        encode_run::encode_run(&run, &[], &[], &[], &mut out.ints, &mut out.floats, &mut out.mask);
        fight(&run, gs, available, &mut out);
        return Ok(out);
    }
    let mut picked = Vec::new();
    let Some(event_labels) = set_screen(&mut run, gs, &by_uuid, &mut picked) else {
        return Ok(out);
    };
    let legal = run.legal_actions();
    encode_run::encode_run(&run, &legal, &picked, &[], &mut out.ints, &mut out.floats, &mut out.mask);
    let opts = encode_run::options(&run, &picked, &[]);
    for (k, o) in opts.iter().enumerate() {
        if out.mask[PICK + k] == 1 {
            out.labels.push((PICK + k, describe_run(&run, &o.action, &event_labels)));
        }
    }
    if out.mask[CONFIRM] == 1 {
        out.labels.push((CONFIRM, confirm_label(&run)));
    }
    for slot in 0..MAX_POTIONS {
        let name = run.potions.get(slot).copied().flatten().map_or("a potion".into(), |p| p.data().name.to_string());
        if out.mask[POTION + slot * TARGETS] == 1 {
            out.labels.push((POTION + slot * TARGETS, format!("Drink {name}")));
        }
        if out.mask[POTION + slot * TARGETS + 1] == 1 {
            out.labels.push((POTION + slot * TARGETS + 1, format!("Discard {name}")));
        }
    }
    Ok(out)
}

/// What a card choice in a fight is for (`encode::CHOICE_KINDS`), from the card or relic that
/// asked for it, and which pile its options come from.
fn choice_kind(cs: &Value, screen: &str) -> (i32, i32) {
    let by = s(&cs["card_in_play"]["id"]);
    let kind = match by {
        "Armaments" => "Armaments",
        "Dual Wield" => "DualWield",
        "Warcry" => "PutOnDeck",
        "Headbutt" => "DiscardToTopOfDeck",
        "Exhume" => "Exhume",
        "Discovery" => "Discovery",
        "Forethought" => "Forethought",
        "Secret Technique" | "Secret Weapon" => "FromDeckToHand",
        "True Grit" | "Burning Pact" | "Purity" => "Exhaust",
        _ => match screen {
            "CARD_REWARD" => "Discover",
            "HAND_SELECT" => "Discard",
            _ => "",
        },
    };
    let k = ["Exhaust", "Discard", "PutOnDeck", "Armaments", "DualWield", "DiscardToTopOfDeck", "Exhume", "GamblingChip", "Discover", "Discovery", "Codex", "DiscardToHand", "FromDeckToHand", "Forethought"]
        .iter()
        .position(|n| *n == kind)
        .map_or(0, |i| i as i32 + 1);
    let pile = match (screen, kind) {
        ("HAND_SELECT", _) => 1,
        (_, "DiscardToTopOfDeck") => 3,
        (_, "Exhume") => 4,
        (_, "FromDeckToHand") => 2,
        ("CARD_REWARD", _) => 5,
        _ => 3,
    };
    (k, pile)
}

/// A stand-in engine fight with the game's player (powers, block), hand, piles and monsters (HP,
/// block, powers), monster `j` at index `j`: enough for the numbers the hand's cards would show
/// (`Combat::card_display`, `card_damage_against`).
/// The player's stance (`player.stance`, the harness's addition; Neutral without it).
fn stance(p: &Value) -> Stance {
    match s(&p["stance"]) {
        "Calm" => Stance::Calm,
        "Wrath" => Stance::Wrath,
        "Divinity" => Stance::Divinity,
        _ => Stance::Neutral,
    }
}

/// The orb slots, front first (`player.orbs`): an empty slot has no ID or "Empty".
fn orbs(p: &Value) -> Vec<Option<Orb>> {
    arr(&p["orbs"])
        .iter()
        .map(|o| {
            let id = match s(&o["id"]) {
                "Lightning" => OrbId::Lightning,
                "Frost" => OrbId::Frost,
                "Dark" => OrbId::Dark,
                "Plasma" => OrbId::Plasma,
                _ => return None,
            };
            let mut orb = Orb::new(id);
            orb.dark_evoke = int(&o["evoke_amount"]);
            Some(orb)
        })
        .collect()
}

fn view_combat(run: &Run, p: &Value, hand: &[(Value, Option<Card>)], draw: &[(Value, Option<Card>)], discard: &[(Value, Option<Card>)], monsters: &[&Value]) -> Option<Combat> {
    let n = names();
    let powers = |v: &Value| arr(v).iter().filter_map(|x| Some(Power::new(*n.powers.get(s(&x["id"]))?, int(&x["amount"])))).collect::<Vec<_>>();
    let mut c = Combat::new(&CombatConfig::ironclad(1, "Jaw Worm")).ok()?;
    c.player.powers = powers(&p["powers"]);
    c.player.block = int(&p["block"]);
    c.player.relics = run.relics.clone();
    c.player.stance = stance(p);
    c.player.orbs = orbs(p);
    c.energy = int(&p["energy"]);
    let cards = |l: &[(Value, Option<Card>)]| l.iter().filter_map(|(_, c)| c.clone()).collect::<Vec<_>>();
    c.hand = cards(hand);
    c.draw_pile = cards(draw);
    c.discard_pile = cards(discard);
    let template = c.monsters.first()?.clone();
    c.monsters = monsters
        .iter()
        .map(|m| {
            let mut x = template.clone();
            x.hp = int(&m["current_hp"]);
            x.max_hp = int(&m["max_hp"]);
            x.block = int(&m["block"]);
            x.powers = powers(&m["powers"]);
            x
        })
        .collect();
    c.order = (0..c.monsters.len()).collect();
    Some(c)
}

/// The fight fields and mask from the game's `combat_state` (see `encode::encode_fight`).
fn fight(run: &Run, gs: &Value, available: &[String], out: &mut Advice) {
    let cs = &gs["combat_state"];
    let p = &cs["player"];
    let n = names();
    let mut scratch = run.clone();
    let card_of = |v: &Value, r: &mut Run| card(r, v);
    let pile = |key: &str, r: &mut Run| -> Vec<(Value, Option<Card>)> { arr(&cs[key]).iter().map(|v| (v.clone(), card_of(v, r))).collect() };
    let hand = pile("hand", &mut scratch);
    let mut draw = pile("draw_pile", &mut scratch);
    let discard = pile("discard_pile", &mut scratch);
    let exhaust = pile("exhaust_pile", &mut scratch);
    draw.sort_by_key(|(_, c)| c.as_ref().map_or((i32::MAX, 0), |c| (c.id as i32, c.times_upgraded)));
    let monsters: Vec<&Value> = arr(&cs["monsters"]).iter().filter(|m| !m["is_gone"].as_bool().unwrap_or(false)).take(MAX_MONSTERS).collect();
    let dome = run.has_relic(RelicId::RunicDome);
    let id_of = |c: &Option<Card>| c.as_ref().map_or(0, |c| c.id as i32 + 1);

    // A card choice in the fight: a hand selection, a grid (Headbutt, Exhume...) or a card
    // reward screen (Discovery and the like).
    let screen = s(&gs["screen_type"]);
    let st = &gs["screen_state"];
    let choosing = matches!(screen, "HAND_SELECT" | "GRID" | "CARD_REWARD");
    let (options, selected): (Vec<(Value, Option<Card>)>, Vec<String>) = if choosing {
        let opts = match screen {
            "HAND_SELECT" => arr(&st["hand"]),
            _ => arr(&st["cards"]),
        };
        let sel = arr(if screen == "HAND_SELECT" { &st["selected"] } else { &st["selected_cards"] }).iter().map(|c| s(&c["uuid"]).to_string()).collect();
        (opts.iter().map(|v| (v.clone(), card_of(v, &mut scratch))).collect(), sel)
    } else {
        (Vec::new(), Vec::new())
    };
    let (kind, pile_id) = if choosing { choice_kind(cs, screen) } else { (0, 0) };
    let count = if choosing { int(&st["max_cards"]).max(int(&st["num_cards"])).max(1) } else { 0 };
    let any_number = choosing && st["any_number"].as_bool().unwrap_or(false);

    let mut w = Writer { buf: &mut out.ints[..encode::int_len()], at: 0 };
    w.field(1, [run.config.character as i32 + 1]);
    w.field(MAX_HAND, hand.iter().map(|(_, c)| id_of(c)));
    w.field(MAX_PLAYER_POWERS, arr(&p["powers"]).iter().map(|x| n.powers.get(s(&x["id"])).map_or(0, |&id| id as i32 + 1)));
    w.field(MAX_MONSTERS, monsters.iter().map(|m| n.monsters.get(s(&m["id"])).map_or(0, |&id| id as i32 + 1)));
    w.field(MAX_MONSTERS, monsters.iter().map(|m| if dome { 0 } else { INTENTS.iter().position(|i| *i == s(&m["intent"])).map_or(0, |i| i as i32 + 1) }));
    let mut mp = vec![0i32; MAX_MONSTERS * MAX_MONSTER_POWERS];
    for (j, m) in monsters.iter().enumerate() {
        for (k, x) in arr(&m["powers"]).iter().take(MAX_MONSTER_POWERS).enumerate() {
            mp[j * MAX_MONSTER_POWERS + k] = n.powers.get(s(&x["id"])).map_or(0, |&id| id as i32 + 1);
        }
    }
    w.field(mp.len(), mp);
    w.field(MAX_RELICS, run.relics.iter().map(|r| r.id as i32 + 1));
    w.field(MAX_POTIONS, run.potions.iter().map(|p| p.map_or(0, |p| p as i32 + 1)));
    w.field(MAX_PILE, draw.iter().map(|(_, c)| id_of(c)));
    w.field(MAX_PILE, discard.iter().map(|(_, c)| id_of(c)));
    w.field(MAX_PILE, exhaust.iter().map(|(_, c)| id_of(c)));
    w.field(1, [kind]);
    w.field(1, [pile_id]);
    w.field(MAX_OPTIONS, options.iter().map(|(_, c)| id_of(c)));
    w.field(1, [stance(p) as i32 + 1]);
    w.field(1, [0]); // the last card type played isn't in the JSON
    let slots = orbs(p);
    w.field(MAX_ORBS, slots.iter().map(|o| encode::orb_id(o.as_ref())));
    let mut moves = vec![0i32; MAX_MONSTERS * MOVES_SEEN];
    for (j, m) in monsters.iter().enumerate() {
        let Some(&id) = n.monsters.get(s(&m["id"])) else { continue };
        for (k, key) in ["last_move_id", "second_last_move_id"].iter().enumerate().take(MOVES_SEEN) {
            if let Some(mv) = m[*key].as_i64() {
                moves[j * MOVES_SEEN + k] = encode::move_id(id, mv as u8);
            }
        }
    }
    w.field(moves.len(), moves);

    let energy_master = 3 + run
        .relics
        .iter()
        .filter(|r| {
            use RelicId::*;
            matches!(r.id, BustedCrown | CoffeeDripper | CursedKey | Ectoplasm | FusionHammer | MarkOfPain | PhilosopherStone | RunicDome | Sozu | VelvetChoker)
        })
        .count() as i32;
    let picked = options.iter().filter(|(v, _)| selected.contains(&s(&v["uuid"]).to_string())).count();
    let mut f = Writer { buf: &mut out.floats[..encode::float_len()], at: 0 };
    f.field(
        SCALARS,
        [
            int(&p["current_hp"]) as f32,
            int(&p["max_hp"]) as f32,
            int(&p["block"]) as f32,
            int(&p["energy"]) as f32,
            energy_master as f32,
            int(&cs["turn"]) as f32,
            hand.len() as f32,
            draw.len() as f32,
            discard.len() as f32,
            exhaust.len() as f32,
            run.gold as f32,
            run.config.ascension as f32,
            0.0,
            count as f32,
            any_number as i32 as f32,
            picked as f32,
            0.0,
            int(&cs["cards_discarded_this_turn"]) as f32,
            slots.len() as f32,
            0.0,
        ],
    );
    // The hand's numbers as the game shows them, from a stand-in engine fight.
    let view = view_combat(run, p, &hand, &draw, &discard, &monsters);
    let mut hf = vec![0f32; MAX_HAND * HAND_FEATS];
    let mut against = vec![0f32; MAX_HAND * MAX_MONSTERS];
    let mut total = vec![0f32; MAX_HAND * MAX_MONSTERS];
    let targets = monsters.iter().filter(|m| !m["half_dead"].as_bool().unwrap_or(false)).count();
    for (h, (v, c)) in hand.iter().take(MAX_HAND).enumerate() {
        let playable = v["is_playable"].as_bool().unwrap_or(false);
        let targeted = v["has_target"].as_bool().unwrap_or(false);
        let row = match (c, &view) {
            (Some(c), Some(view)) => {
                let mut c = c.clone();
                c.cost_for_turn = int(&v["cost"]);
                for j in 0..monsters.len() {
                    against[h * MAX_MONSTERS + j] = view.card_damage_against(&c, j) as f32;
                    total[h * MAX_MONSTERS + j] = encode::play_damage_against(view, &c, j, targets);
                }
                encode::hand_row(view, &c, playable, targeted, v["exhausts"].as_bool().unwrap_or(false))
            }
            _ => {
                let mut row = [0f32; HAND_FEATS];
                row[0] = int(&v["cost"]) as f32;
                row[1] = int(&v["upgrades"]) as f32;
                row[5] = playable as i32 as f32;
                row[6] = targeted as i32 as f32;
                row
            }
        };
        hf[h * HAND_FEATS..(h + 1) * HAND_FEATS].copy_from_slice(&row);
    }
    f.field(hf.len(), hf);
    f.field(MAX_PLAYER_POWERS, arr(&p["powers"]).iter().map(|x| int(&x["amount"]) as f32));
    let mut mon = vec![0f32; MAX_MONSTERS * 8];
    let mut mon_pow = vec![0f32; MAX_MONSTERS * MAX_MONSTER_POWERS];
    let targetable = |m: &Value| !m["half_dead"].as_bool().unwrap_or(false);
    for (j, m) in monsters.iter().enumerate() {
        let attack = int(&m["move_base_damage"]) >= 0 && m["move_base_damage"].is_number() && !dome && s(&m["intent"]).starts_with("ATTACK");
        mon[j * 8..(j + 1) * 8].copy_from_slice(&[
            int(&m["current_hp"]) as f32,
            int(&m["max_hp"]) as f32,
            int(&m["block"]) as f32,
            if attack { int(&m["move_adjusted_damage"]).max(0) as f32 } else { 0.0 },
            if attack { int(&m["move_hits"]).max(1) as f32 } else { 0.0 },
            targetable(m) as i32 as f32,
            m["half_dead"].as_bool().unwrap_or(false) as i32 as f32,
            0.0,
        ]);
        for (k, x) in arr(&m["powers"]).iter().take(MAX_MONSTER_POWERS).enumerate() {
            mon_pow[j * MAX_MONSTER_POWERS + k] = int(&x["amount"]) as f32;
        }
    }
    f.field(mon.len(), mon);
    f.field(mon_pow.len(), mon_pow);
    f.field(MAX_RELICS, run.relics.iter().map(|r| r.counter as f32));
    let ups = |l: &[(Value, Option<Card>)]| l.iter().map(|(v, _)| int(&v["upgrades"]) as f32).collect::<Vec<_>>();
    f.field(MAX_PILE, ups(&draw));
    f.field(MAX_PILE, ups(&discard));
    f.field(MAX_PILE, ups(&exhaust));
    let mut of = vec![0f32; MAX_OPTIONS * 3];
    for (k, (v, _)) in options.iter().take(MAX_OPTIONS).enumerate() {
        let chosen = selected.contains(&s(&v["uuid"]).to_string());
        of[k * 3..(k + 1) * 3].copy_from_slice(&[int(&v["upgrades"]) as f32, int(&v["cost"]) as f32, chosen as i32 as f32]);
    }
    f.field(of.len(), of);
    f.field(against.len(), against);
    f.field(total.len(), total);
    let orb_numbers = |key: &str| arr(&p["orbs"]).iter().map(|o| if o["id"].is_null() || s(&o["id"]) == "Empty" { 0.0 } else { int(&o[key]) as f32 }).collect::<Vec<_>>();
    f.field(MAX_ORBS, orb_numbers("passive_amount"));
    f.field(MAX_ORBS, orb_numbers("evoke_amount"));

    // The mask and labels, from what the game allows.
    let can = |c: &str| available.iter().any(|a| a == c);
    if choosing {
        for (k, (v, _)) in options.iter().take(MAX_OPTIONS).enumerate() {
            if !selected.contains(&s(&v["uuid"]).to_string()) && can("choose") {
                out.mask[PICK + k] = 1;
                out.labels.push((PICK + k, format!("Choose {}", card_name(v))));
            }
        }
        if can("confirm") || can("skip") {
            out.mask[CONFIRM] = 1;
            out.labels.push((CONFIRM, if can("confirm") { "Confirm".into() } else { "Skip".into() }));
        }
        return;
    }
    let mon_name = |j: usize| s(&monsters[j]["name"]).to_string();
    if can("play") {
        for (h, (v, _)) in hand.iter().take(MAX_HAND).enumerate() {
            if !v["is_playable"].as_bool().unwrap_or(false) {
                continue;
            }
            if v["has_target"].as_bool().unwrap_or(false) {
                for j in 0..monsters.len() {
                    if targetable(monsters[j]) {
                        out.mask[PLAY + h * TARGETS + 1 + j] = 1;
                        out.labels.push((PLAY + h * TARGETS + 1 + j, format!("Play {} → {}", card_name(v), mon_name(j))));
                    }
                }
            } else {
                out.mask[PLAY + h * TARGETS] = 1;
                out.labels.push((PLAY + h * TARGETS, format!("Play {}", card_name(v))));
            }
        }
    }
    if can("potion") {
        for (slot, pv) in arr(&gs["potions"]).iter().take(MAX_POTIONS).enumerate() {
            if !pv["can_use"].as_bool().unwrap_or(false) {
                continue;
            }
            let name = s(&pv["name"]).to_string();
            if pv["requires_target"].as_bool().unwrap_or(false) {
                for j in 0..monsters.len() {
                    if targetable(monsters[j]) {
                        out.mask[POTION + slot * TARGETS + 1 + j] = 1;
                        out.labels.push((POTION + slot * TARGETS + 1 + j, format!("Drink {name} → {}", mon_name(j))));
                    }
                }
            } else {
                out.mask[POTION + slot * TARGETS] = 1;
                out.labels.push((POTION + slot * TARGETS, format!("Drink {name}")));
            }
        }
    }
    if can("end") {
        out.mask[END_TURN] = 1;
        out.labels.push((END_TURN, "End turn".into()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sts_core::cards::CardTarget;
    use sts_core::{Combat, CombatConfig, PlayerAction};

    /// A fight as CommunicationMod would report it (the fields `advise` reads).
    fn game_json(c: &Combat, legal: &[PlayerAction]) -> Value {
        let card = |h: Option<usize>, x: &Card| {
            let playable = h.is_some_and(|h| legal.iter().any(|a| matches!(a, PlayerAction::PlayCard { hand_index, .. } if *hand_index == h)));
            json!({"id": x.data().java_id, "name": x.name(), "upgrades": x.times_upgraded, "cost": x.cost_for_turn, "uuid": format!("{}", x.uuid),
                   "is_playable": playable, "has_target": matches!(x.target(), CardTarget::Enemy | CardTarget::SelfAndEnemy)})
        };
        let powers = |ps: &[Power]| ps.iter().map(|p| json!({"id": p.java_id(), "amount": p.amount})).collect::<Vec<_>>();
        let monsters: Vec<Value> = c
            .order
            .iter()
            .map(|&m| {
                let x = &c.monsters[m];
                let h = &x.move_history;
                let back = |k: usize| if h.len() >= k { json!(h[h.len() - k]) } else { Value::Null };
                json!({"id": x.id.java_id(), "name": x.id.java_id(), "current_hp": x.hp, "max_hp": x.max_hp, "block": x.block,
                       "intent": INTENTS[x.intent as usize], "move_base_damage": x.intent_base_dmg, "move_adjusted_damage": x.intent_dmg,
                       "move_hits": if x.is_multi_dmg { x.intent_multi } else { 1 }, "half_dead": x.half_dead,
                       "is_gone": x.is_dying || x.is_escaping, "powers": powers(&x.powers),
                       "last_move_id": back(2), "second_last_move_id": back(3)})
            })
            .collect();
        json!({
            "screen_type": "NONE", "room_phase": "COMBAT", "seed": 1, "ascension_level": 0, "act": 1, "floor": 3, "class": c.character.class_name(),
            "max_hp": c.player.max_hp, "current_hp": c.player.hp, "gold": c.player.gold, "deck": [],
            "relics": c.player.relics.iter().map(|r| json!({"id": r.id.data().java_id, "counter": r.counter})).collect::<Vec<_>>(),
            "potions": [],
            "combat_state": {
                "player": {"current_hp": c.player.hp, "max_hp": c.player.max_hp, "block": c.player.block, "energy": c.energy, "powers": powers(&c.player.powers),
                           "stance": c.player.stance.java_id(),
                           "orbs": c.player.orbs.iter().map(|o| match o {
                               Some(o) => json!({"id": o.id.java_id(), "passive_amount": c.orb_passive(o), "evoke_amount": c.orb_evoke(o)}),
                               None => json!({"id": "Empty", "passive_amount": 0, "evoke_amount": 0}),
                           }).collect::<Vec<_>>()},
                "hand": c.hand.iter().enumerate().map(|(h, x)| card(Some(h), x)).collect::<Vec<_>>(),
                "draw_pile": c.draw_pile.iter().map(|x| card(None, x)).collect::<Vec<_>>(),
                "discard_pile": c.discard_pile.iter().map(|x| card(None, x)).collect::<Vec<_>>(),
                "exhaust_pile": c.exhaust_pile.iter().map(|x| card(None, x)).collect::<Vec<_>>(),
                "monsters": monsters, "turn": c.turn, "cards_discarded_this_turn": c.total_discarded_this_turn,
            }
        })
    }

    /// The fight fields `advise` writes from the game's JSON match the engine's own encoding of
    /// the same fight (apart from the hand's magic numbers, which `advise` takes as the card's
    /// base numbers, and the cards and attacks played this turn and the last card type played,
    /// which the JSON doesn't have).
    #[test]
    fn a_fight_from_the_game_json_encodes_like_the_engine() {
        use sts_core::character::Character;
        let (fi, ff) = (encode::int_len(), encode::float_len());
        let at = |fields: &[(&str, usize)], name: &str| fields.iter().take_while(|f| f.0 != name).map(|f| f.1).sum::<usize>();
        let hand_at = at(encode::FLOAT_FIELDS, "hand");
        let last_type = at(encode::INT_FIELDS, "last_played_type");
        let mut checked = 0;
        let fights = [
            (Character::Ironclad, "Jaw Worm"),
            (Character::Ironclad, "Gremlin Nob"),
            (Character::Ironclad, "3 Sentries"),
            (Character::Ironclad, "Slavers"),
            (Character::Ironclad, "Book of Stabbing"),
            (Character::Ironclad, "Reptomancer"),
            (Character::Defect, "Awakened One"),
            (Character::Watcher, "Time Eater"),
            (Character::Watcher, "Gremlin Gang"),
            (Character::Defect, "Lots of Slimes"),
        ];
        for (n, (ch, key)) in fights.iter().enumerate() {
            let mut cfg = CombatConfig::new(*ch, n as i64 + 5, key);
            cfg.deck.extend(match ch {
                Character::Defect => [(CardId::Zap, 0), (CardId::Barrage, 1), (CardId::Coolheaded, 0), (CardId::Defragment, 0)],
                Character::Watcher => [(CardId::Eruption, 1), (CardId::Tantrum, 0), (CardId::FlyingSleeves, 0), (CardId::InnerPeace, 0)],
                _ => [(CardId::Inflame, 0), (CardId::Bash, 1), (CardId::ShrugItOff, 0), (CardId::Disarm, 0)],
            });
            let mut c = Combat::new(&cfg).unwrap();
            let mut rng = sts_core::rng::StsRandom::new(n as i64);
            for _ in 0..25 {
                if c.is_over() || c.pending.is_some() {
                    break;
                }
                let legal = c.legal_actions();
                let mut want = (vec![0; fi], vec![0f32; ff], vec![0u8; encode::ACTIONS]);
                encode::encode_fight(&c, &legal, &[], &mut want.0, &mut want.1, &mut want.2);
                let gs = game_json(&c, &legal);
                let a = advise(&gs, &["play".into(), "end".into()]).unwrap();
                for (i, (got, exp)) in a.ints[..fi].iter().zip(&want.0).enumerate() {
                    if i != last_type {
                        assert_eq!(got, exp, "{key}: int {i} differs");
                    }
                }
                for (i, (got, exp)) in a.floats[..ff].iter().zip(&want.1).enumerate() {
                    let in_hand = (hand_at..hand_at + MAX_HAND * HAND_FEATS).contains(&i) && (i - hand_at) % HAND_FEATS == 4;
                    if !in_hand && i != 12 && i != 16 {
                        assert_eq!(got, exp, "{key}: float {i} differs");
                    }
                }
                assert_eq!(a.mask, want.2, "{key}: mask differs");
                checked += 1;
                let a = legal[rng.random(legal.len() as i32 - 1) as usize].clone();
                c.step(&a).unwrap();
            }
        }
        assert!(checked > 50, "{checked}");
    }
}
