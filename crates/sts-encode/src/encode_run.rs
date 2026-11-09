//! Fixed-size arrays for whole runs: the fight arrays (filled during combat, zero otherwise)
//! followed by the run's own fields, with the same action slots as fights.
//!
//! Outside combat every screen is a list of up to 64 options, each one token's worth of IDs and
//! numbers (a card, relic, potion, Neow blessing, map node, event option, shop item, rest
//! action...), and the action slots mean:
//! - `pick(k)`: take option `k`. On a deck grid (remove, upgrade, transform...) that needs several
//!   cards, each pick adds one card and the choice is made when enough are picked;
//! - `confirm`: proceed, skip or leave (rewards, card rewards, boss relics, shops, rest sites
//!   without options);
//! - `potion(p, 0)`: drink potion `p` outside combat; `potion(p, 1)`: discard it.
//!
//! Map nodes carry what lies beyond them: for each room type, the fewest and most rooms of that
//! type on any path from the node to the act's boss.

use sts_core::cards::Card;
use sts_core::encounters;
use sts_core::run::map::{RoomType, MAP_HEIGHT, MAP_WIDTH};
use sts_core::run::{GridPurpose, KeyRule, Reward, Run, RunAction, Screen};
use sts_core::PlayerAction;

use crate::encode::{self, Writer, CONFIRM, MAX_OPTIONS, MAX_POTIONS, MAX_RELICS, PICK, POTION};

pub const MAX_DECK: usize = 80;
const RUN_SCALARS: usize = 18;
const OPT_FEATS: usize = 19;
/// Room types counted along map paths.
const PATH_ROOMS: [RoomType; 6] = [RoomType::Monster, RoomType::Elite, RoomType::Event, RoomType::Rest, RoomType::Shop, RoomType::Treasure];

/// (name, length) of each integer field after the fight's, in row order.
pub const RUN_INT_FIELDS: &[(&str, usize)] = &[
    ("screen", 1),
    ("boss", 1),
    ("room", 1),
    ("event", 1),
    ("event_screen", 1),
    ("grid_purpose", 1),
    ("run_relic", MAX_RELICS),
    ("run_potion", MAX_POTIONS),
    ("deck", MAX_DECK),
    ("opt_kind", MAX_OPTIONS),
    ("opt_card", MAX_OPTIONS),
    ("opt_relic", MAX_OPTIONS),
    ("opt_potion", MAX_OPTIONS),
    ("opt_neow", MAX_OPTIONS),
    ("opt_drawback", MAX_OPTIONS),
    ("opt_room", MAX_OPTIONS),
    ("opt_index", MAX_OPTIONS),
];

/// (name, length) of each float field after the fight's, in row order.
pub const RUN_FLOAT_FIELDS: &[(&str, usize)] = &[
    // hp, max hp, gold, floor, act, ascension, deck size, ruby/emerald/sapphire keys, purge
    // cost, potion slots, empty potion slots, cards picked on a grid, cards a grid takes, the
    // event's first three numbers.
    ("run_scalars", RUN_SCALARS),
    ("run_relic_counter", MAX_RELICS),
    ("deck_upgrades", MAX_DECK),
    // upgrades, cost, price, affordable, amount (gold, cards in a reward), already picked,
    // emerald key on the node, then for each of monster/elite/event/rest/shop/treasure the
    // fewest and the most rooms of that type on the paths from the node to the boss.
    ("opt", MAX_OPTIONS * OPT_FEATS),
    // The deck network's lookup (jack, 2026-10-30; python/core/lookup.py): at a card reward, each
    // card option's gains over skipping in the fight network's win chance and HP share, against
    // this act's boss and then the next act's bosses; and the deck as it stands' own four scores.
    // The engine leaves them at zero: the training code writes them after observing.
    ("opt_lookup", MAX_OPTIONS * LOOKUP_FEATS),
    ("run_lookup", LOOKUP_FEATS),
];

/// Numbers per option in "opt_lookup" (and in "run_lookup").
pub const LOOKUP_FEATS: usize = 4;

/// Option kinds (`opt_kind`, 0 = no option). New kinds go at the end so a checkpoint's ids keep
/// their meaning (`python/core/run_model.py` grows the embedding when loading an older one).
/// "grid_card" is no longer used: a grid's cards carry the kind of what the grid does with the
/// chosen card (`grid_kind`), shared by every grid with that effect.
pub const OPTION_KINDS: &[&str] = &[
    "neow", "path", "gold", "stolen_gold", "relic", "potion", "cards", "emerald_key", "sapphire_key", "pick_card",
    "singing_bowl", "grid_card", "rest", "smith", "lift", "toke", "dig", "recall", "boss_relic", "shop_card",
    "shop_relic", "shop_potion", "shop_purge", "event_option", "grid_remove", "grid_upgrade", "grid_transform",
    "grid_duplicate", "grid_bottle", "grid_offer",
];

/// Option kinds no longer produced (kept so later ids don't move).
pub const RETIRED_OPTION_KINDS: &[&str] = &["grid_card"];

/// The option kind of a grid's cards: what the grid does with the chosen card.
fn grid_kind(p: GridPurpose) -> &'static str {
    match p {
        GridPurpose::Remove | GridPurpose::ShopPurge | GridPurpose::Toke | GridPurpose::RemoveAndUpgrade => "grid_remove",
        GridPurpose::Upgrade | GridPurpose::Smith => "grid_upgrade",
        GridPurpose::Transform | GridPurpose::EventTransform | GridPurpose::Astrolabe => "grid_transform",
        GridPurpose::Duplicate => "grid_duplicate",
        GridPurpose::Bottle(_) => "grid_bottle",
        GridPurpose::Bonfire => "grid_offer",
    }
}

pub const SCREENS: &[&str] = &["neow", "map", "combat", "rewards", "card_reward", "grid", "rest", "boss_relic", "shop", "event", "unsupported", "over"];

const GRID_PURPOSES: usize = 12;

pub fn run_int_len() -> usize {
    RUN_INT_FIELDS.iter().map(|f| f.1).sum()
}

pub fn run_float_len() -> usize {
    RUN_FLOAT_FIELDS.iter().map(|f| f.1).sum()
}

fn kind(name: &str) -> i32 {
    OPTION_KINDS.iter().position(|k| *k == name).expect("option kind") as i32 + 1
}

fn screen_id(s: &Screen) -> i32 {
    1 + match s {
        Screen::Neow { .. } => 0,
        Screen::Map => 1,
        Screen::Combat => 2,
        Screen::Rewards { .. } => 3,
        Screen::CardReward { .. } => 4,
        Screen::Grid { .. } => 5,
        Screen::Rest => 6,
        Screen::BossRelic { .. } => 7,
        Screen::Shop { .. } => 8,
        Screen::Event { .. } => 9,
        Screen::Unsupported { .. } => 10,
        Screen::Over => 11,
    }
}

fn grid_purpose_id(p: GridPurpose) -> i32 {
    1 + match p {
        GridPurpose::Upgrade => 0,
        GridPurpose::Remove => 1,
        GridPurpose::Transform => 2,
        GridPurpose::Smith => 3,
        GridPurpose::Astrolabe => 4,
        GridPurpose::Bottle(_) => 5,
        GridPurpose::Toke => 6,
        GridPurpose::ShopPurge => 7,
        GridPurpose::EventTransform => 8,
        GridPurpose::Bonfire => 9,
        GridPurpose::Duplicate => 10,
        GridPurpose::RemoveAndUpgrade => 11,
    }
}

fn room_id(r: RoomType) -> i32 {
    1 + match r {
        RoomType::Monster => 0,
        RoomType::Elite => 1,
        RoomType::Event => 2,
        RoomType::Rest => 3,
        RoomType::Shop => 4,
        RoomType::Treasure => 5,
        RoomType::Boss => 6,
    }
}

/// What taking an option does.
#[derive(Clone, Debug)]
pub enum OptAction {
    Run(RunAction),
    /// Add deck card `i` to a grid selection.
    GridPick(usize),
}

#[derive(Clone, Debug)]
pub struct Opt {
    kind: i32,
    card: i32,
    relic: i32,
    potion: i32,
    neow: i32,
    drawback: i32,
    room: i32,
    index: i32,
    f: [f32; OPT_FEATS],
    pub action: OptAction,
    /// A card reward already looked at and skipped: shown, but not taken again.
    blocked: bool,
}

impl Opt {
    fn new(kind_name: &str, action: RunAction) -> Opt {
        Opt { kind: kind(kind_name), card: 0, relic: 0, potion: 0, neow: 0, drawback: 0, room: 0, index: 0, f: [0.0; OPT_FEATS], action: OptAction::Run(action), blocked: false }
    }

    fn card(mut self, c: &Card) -> Opt {
        self.card = c.id as i32 + 1;
        self.f[0] = c.times_upgraded as f32;
        self.f[1] = c.cost as f32;
        self
    }

    fn price(mut self, price: i32, gold: i32) -> Opt {
        self.f[2] = price as f32;
        self.f[3] = (price <= gold) as i32 as f32;
        self
    }

    fn amount(mut self, n: i32) -> Opt {
        self.f[4] = n as f32;
        self
    }
}

/// For every node of the act's map: (fewest, most) rooms of each `PATH_ROOMS` type on the
/// paths from the node (itself included) to the boss.
fn path_counts(run: &Run) -> Vec<Vec<([f32; 6], [f32; 6])>> {
    let mut out = vec![vec![([0.0; 6], [0.0; 6]); MAP_WIDTH]; MAP_HEIGHT];
    if run.map.rows.len() < MAP_HEIGHT {
        return out;
    }
    for y in (0..MAP_HEIGHT).rev() {
        for x in 0..MAP_WIDTH {
            let n = run.map.node(x as i32, y as i32);
            let next: Vec<(i32, i32)> = n.edges.iter().copied().filter(|e| (e.1 as usize) == y + 1 && (e.1 as usize) < MAP_HEIGHT).collect();
            let (mut lo, mut hi) = if next.is_empty() {
                ([0.0; 6], [0.0; 6])
            } else {
                let mut lo = [f32::MAX; 6];
                let mut hi = [0.0f32; 6];
                for (ex, ey) in next {
                    let (l, h) = out[ey as usize][ex as usize];
                    for t in 0..6 {
                        lo[t] = lo[t].min(l[t]);
                        hi[t] = hi[t].max(h[t]);
                    }
                }
                (lo, hi)
            };
            if let Some(r) = n.room {
                if let Some(t) = PATH_ROOMS.iter().position(|&p| p == r) {
                    lo[t] += 1.0;
                    hi[t] += 1.0;
                }
            }
            out[y][x] = (lo, hi);
        }
    }
    out
}

/// A card reward's cards, to recognise one that was already skipped.
pub fn card_reward_key(cards: &[Card]) -> Vec<(i32, i32)> {
    cards.iter().map(|c| (c.id as i32, c.times_upgraded)).collect()
}

/// The options of the current (non-combat) screen, in slot order. `skipped` holds the card
/// rewards skipped on this reward screen: skipping leaves a card reward on the list, and opening
/// it again only shows the same cards, so it is not offered again (greedy play would loop).
pub fn options(run: &Run, picked: &[usize], skipped: &[Vec<(i32, i32)>]) -> Vec<Opt> {
    let gold = run.gold;
    let mut out = Vec::new();
    match &run.screen {
        Screen::Neow { options } => {
            for (k, o) in options.iter().enumerate() {
                let mut opt = Opt::new("neow", RunAction::Neow { option: k });
                opt.neow = o.reward as i32 + 1;
                opt.drawback = o.drawback as i32 + 1;
                out.push(opt);
            }
        }
        Screen::Map => {
            let y = run.pos.map_or(0, |(_, y)| y + 1);
            let counts = if (y as usize) < MAP_HEIGHT { path_counts(run) } else { Vec::new() };
            for x in run.reachable_nodes() {
                let mut opt = Opt::new("path", RunAction::Path { x });
                if (y as usize) < MAP_HEIGHT {
                    let n = run.map.node(x, y);
                    opt.room = n.room.map_or(0, room_id);
                    opt.f[6] = n.has_emerald_key as i32 as f32;
                    let (lo, hi) = counts[y as usize][x as usize];
                    opt.f[7..13].copy_from_slice(&lo);
                    opt.f[13..19].copy_from_slice(&hi);
                } else {
                    opt.room = room_id(RoomType::Boss);
                }
                out.push(opt);
            }
        }
        Screen::Rewards { items } => {
            for (i, it) in items.iter().enumerate() {
                let a = RunAction::TakeReward { index: i };
                out.push(match it {
                    Reward::Gold(n) => Opt::new("gold", a).amount(*n),
                    Reward::StolenGold(n) => Opt::new("stolen_gold", a).amount(*n),
                    Reward::Relic(r) => {
                        let mut o = Opt::new("relic", a);
                        o.relic = *r as i32 + 1;
                        o
                    }
                    Reward::Potion(p) => {
                        let mut o = Opt::new("potion", a);
                        o.potion = *p as i32 + 1;
                        o
                    }
                    Reward::Cards(cs) => {
                        let mut o = Opt::new("cards", a).amount(cs.len() as i32);
                        if skipped.contains(&card_reward_key(cs)) {
                            o.blocked = true;
                            o.f[5] = 1.0;
                        }
                        o
                    }
                    Reward::EmeraldKey => Opt::new("emerald_key", a),
                    Reward::SapphireKey => Opt::new("sapphire_key", a),
                });
            }
        }
        Screen::CardReward { cards, .. } => {
            for (k, c) in cards.iter().enumerate() {
                out.push(Opt::new("pick_card", RunAction::PickCard { index: k }).card(c));
            }
            if run.has_relic(sts_core::relics::RelicId::SingingBowl) {
                out.push(Opt::new("singing_bowl", RunAction::SingingBowl));
            }
        }
        Screen::Grid { options, purpose, .. } => {
            for &d in options {
                let Some(c) = run.deck.get(d) else { continue };
                let mut o = Opt::new(grid_kind(*purpose), RunAction::Proceed).card(c);
                o.action = OptAction::GridPick(d);
                o.f[5] = picked.contains(&d) as i32 as f32;
                out.push(o);
            }
        }
        Screen::Rest => {
            for a in run.campfire_options() {
                let name = match a {
                    RunAction::Rest => "rest",
                    RunAction::Smith => "smith",
                    RunAction::Lift => "lift",
                    RunAction::Toke => "toke",
                    RunAction::Dig => "dig",
                    RunAction::Recall => "recall",
                    _ => continue,
                };
                out.push(Opt::new(name, a));
            }
        }
        Screen::BossRelic { relics } => {
            for (k, r) in relics.iter().enumerate() {
                let mut o = Opt::new("boss_relic", RunAction::BossRelic { index: k });
                o.relic = *r as i32 + 1;
                out.push(o);
            }
        }
        Screen::Shop { shop } => {
            for (i, c) in shop.cards.iter().enumerate() {
                if let Some(c) = c {
                    out.push(Opt::new("shop_card", RunAction::BuyCard { index: i }).card(&c.card).price(c.price, gold));
                }
            }
            for (i, r) in shop.relics.iter().enumerate() {
                if let Some(r) = r {
                    let mut o = Opt::new("shop_relic", RunAction::BuyRelic { index: i }).price(r.price, gold);
                    o.relic = r.relic as i32 + 1;
                    out.push(o);
                }
            }
            for (i, p) in shop.potions.iter().enumerate() {
                if let Some(p) = p {
                    let mut o = Opt::new("shop_potion", RunAction::BuyPotion { index: i }).price(p.price, gold);
                    o.potion = p.potion as i32 + 1;
                    out.push(o);
                }
            }
            if shop.purge_available {
                out.push(Opt::new("shop_purge", RunAction::Purge).price(shop.purge_cost, gold));
            }
        }
        Screen::Event { event } => {
            for k in 0..event.options.len() {
                let mut o = Opt::new("event_option", RunAction::EventChoice { index: k });
                o.index = k as i32 + 1;
                out.push(o);
            }
        }
        Screen::Combat | Screen::Unsupported { .. } | Screen::Over => {}
    }
    out.truncate(MAX_OPTIONS);
    if let Some(rule) = run.config.key_rule {
        key_rule(run, rule, &mut out);
    }
    out
}

/// `RunConfig::key_rule`, hand-written, for a network that trained holding the three keys and has never met them: its
/// inputs say it holds them (the key inputs here, the burning elite on later acts' maps, where those games have
/// none), and the rule gets them in Act 3, where a chest relic and a smith cost the least: the map moves only towards the burning elite while it's
/// missing, the chest's Sapphire key over its relic, and Recall at the first rest with `rule`'s share of HP (in Act 2 too, if `rule` says), at Act 3's last rest whatever
/// the HP. An Emerald key is always taken. Otherwise the other two are out of sight. (`key_to_take` keeps the reward screen open.)
fn key_rule(run: &Run, rule: KeyRule, opts: &mut [Opt]) {
    let only = |opts: &mut [Opt], name: &str| {
        for o in opts.iter_mut().filter(|o| o.kind != kind(name)) {
            o.blocked = true;
        }
    };
    let hide = |opts: &mut [Opt], name: &str| {
        for o in opts.iter_mut().filter(|o| o.kind == kind(name)) {
            o.blocked = true;
            o.kind = 0;
        }
    };
    match &run.screen {
        Screen::Map => {
            if run.act > 1 {
                opts.iter_mut().for_each(|o| o.f[6] = 0.0);
            }
            let y = run.pos.map_or(0, |(_, y)| y + 1) as usize;
            if run.act == 3 && !run.keys.emerald && y < MAP_HEIGHT {
                let to = towards_burning_elite(run);
                let on_way = |o: &Opt| matches!(o.action, OptAction::Run(RunAction::Path { x }) if to[y][x as usize]);
                if opts.iter().any(on_way) {
                    for o in opts.iter_mut() {
                        o.blocked |= !on_way(o);
                    }
                }
            }
        }
        Screen::Rewards { items } => {
            if items.contains(&Reward::EmeraldKey) {
                only(opts, "emerald_key");
            } else if run.act == 3 {
                if items.contains(&Reward::SapphireKey) {
                    only(opts, "sapphire_key");
                }
            } else {
                hide(opts, "sapphire_key");
            }
        }
        Screen::Rest => {
            let last = run.pos.is_some_and(|(_, y)| y as usize == MAP_HEIGHT - 1);
            // (Recall alone: with Coffee Dripper and nothing to smith, the site offers nothing else.)
            let alone = opts.iter().all(|o| o.kind == kind("recall"));
            let from = match run.act {
                2 => rule.recall_hp_act2,
                3 if last => 0,
                3 => rule.recall_hp_act3,
                _ => 101,
            };
            if alone || (!run.keys.ruby && run.hp * 100 >= run.max_hp * from) {
                only(opts, "recall");
            } else {
                hide(opts, "recall");
            }
        }
        _ => {}
    }
}

/// The key rule won't leave the reward screen without this key.
fn key_to_take(run: &Run) -> bool {
    run.config.key_rule.is_some() && matches!(&run.screen, Screen::Rewards { items } if items.contains(&Reward::EmeraldKey) || (run.act == 3 && items.contains(&Reward::SapphireKey)))
}

/// For each node of the act's map, whether the burning elite can be reached from it.
fn towards_burning_elite(run: &Run) -> [[bool; MAP_WIDTH]; MAP_HEIGHT] {
    let mut to = [[false; MAP_WIDTH]; MAP_HEIGHT];
    for y in (0..MAP_HEIGHT).rev() {
        for x in 0..MAP_WIDTH {
            let n = run.map.node(x as i32, y as i32);
            to[y][x] = n.has_emerald_key || n.edges.iter().any(|&(ex, ey)| (ey as usize) < MAP_HEIGHT && to[ey as usize][ex as usize]);
        }
    }
    to
}

/// How many cards the current grid takes (0 when not on a grid).
fn grid_count(run: &Run) -> usize {
    match &run.screen {
        Screen::Grid { options, count, .. } => (*count).min(options.len()),
        _ => 0,
    }
}

/// The combat actions among a run's legal actions.
pub fn combat_legal(legal: &[RunAction]) -> Vec<PlayerAction> {
    legal
        .iter()
        .filter_map(|a| match a {
            RunAction::Combat { action } => Some(action.clone()),
            _ => None,
        })
        .collect()
}

/// Encodes `run` into a row: the fight fields (during combat), then the run fields.
pub fn encode_run(run: &Run, legal: &[RunAction], picked: &[usize], skipped: &[Vec<(i32, i32)>], ints: &mut [i32], floats: &mut [f32], mask: &mut [u8]) {
    let (fi, ff) = (encode::int_len(), encode::float_len());
    let (fight_ints, run_ints) = ints.split_at_mut(fi);
    let (fight_floats, run_floats) = floats.split_at_mut(ff);
    let in_combat = matches!(run.screen, Screen::Combat);
    if let (true, Some(c)) = (in_combat, run.combat.as_ref()) {
        encode::encode_fight(c, &combat_legal(legal), picked, fight_ints, fight_floats, mask);
    }

    let event = match &run.screen {
        Screen::Event { event } => Some(event),
        _ => None,
    };
    let opts = if in_combat { Vec::new() } else { options(run, picked, skipped) };

    let mut w = Writer { buf: run_ints, at: 0 };
    w.field(1, [screen_id(&run.screen)]);
    w.field(1, [encounters::all().iter().position(|e| *e == run.boss).map_or(0, |i| i as i32 + 1)]);
    w.field(1, [run.room.map_or(0, room_id)]);
    w.field(1, [event.map_or(0, |e| e.id as i32 + 1)]);
    w.field(1, [event.map_or(0, |e| (e.screen as i32).min(15) + 1)]);
    w.field(1, [match &run.screen {
        Screen::Grid { purpose, .. } => grid_purpose_id(*purpose),
        _ => 0,
    }]);
    w.field(MAX_RELICS, run.relics.iter().map(|r| r.id as i32 + 1));
    w.field(MAX_POTIONS, run.potions.iter().map(|p| p.map_or(0, |p| p as i32 + 1)));
    w.field(MAX_DECK, run.deck.iter().map(|c| c.id as i32 + 1));
    w.field(MAX_OPTIONS, opts.iter().map(|o| o.kind));
    w.field(MAX_OPTIONS, opts.iter().map(|o| o.card));
    w.field(MAX_OPTIONS, opts.iter().map(|o| o.relic));
    w.field(MAX_OPTIONS, opts.iter().map(|o| o.potion));
    w.field(MAX_OPTIONS, opts.iter().map(|o| o.neow));
    w.field(MAX_OPTIONS, opts.iter().map(|o| o.drawback));
    w.field(MAX_OPTIONS, opts.iter().map(|o| o.room));
    w.field(MAX_OPTIONS, opts.iter().map(|o| o.index));

    let mut f = Writer { buf: run_floats, at: 0 };
    let n = event.map_or([0; 4], |e| e.n);
    f.field(
        RUN_SCALARS,
        [
            run.hp as f32,
            run.max_hp as f32,
            run.gold as f32,
            run.floor as f32,
            run.act as f32,
            run.config.ascension as f32,
            run.deck.len() as f32,
            (run.keys.ruby || run.config.key_rule.is_some()) as i32 as f32,
            (run.keys.emerald || run.config.key_rule.is_some()) as i32 as f32,
            (run.keys.sapphire || run.config.key_rule.is_some()) as i32 as f32,
            run.purge_cost as f32,
            run.potions.len() as f32,
            run.potions.iter().filter(|p| p.is_none()).count() as f32,
            if in_combat { 0.0 } else { picked.len() as f32 },
            grid_count(run) as f32,
            n[0] as f32,
            n[1] as f32,
            n[2] as f32,
        ],
    );
    f.field(MAX_RELICS, run.relics.iter().map(|r| r.counter as f32));
    f.field(MAX_DECK, run.deck.iter().map(|c| c.times_upgraded as f32));
    f.field(MAX_OPTIONS * OPT_FEATS, opts.iter().flat_map(|o| o.f));

    if in_combat {
        return;
    }
    let count = grid_count(run);
    for (k, o) in opts.iter().enumerate() {
        let ok = !o.blocked
            && match &o.action {
            OptAction::Run(a) => legal.contains(a),
            OptAction::GridPick(d) => !picked.contains(d) && picked.len() < count,
        };
        if ok {
            mask[PICK + k] = 1;
        }
    }
    let discard = discard_useful(run);
    let gold = gold_on_offer(run);
    for a in legal {
        match a {
            RunAction::Proceed if gold || key_to_take(run) => {}
            RunAction::Proceed | RunAction::SkipCards => mask[CONFIRM] = 1,
            RunAction::UsePotion { slot } if *slot < MAX_POTIONS => mask[POTION + slot * encode::TARGETS] = 1,
            RunAction::DiscardPotion { slot } if *slot < MAX_POTIONS && discard => mask[POTION + slot * encode::TARGETS + 1] = 1,
            _ => {}
        }
    }
}

/// Whether discarding a potion can help: only to make room, with every slot full and a potion on
/// offer (a reward, or one for sale in a shop). Anywhere else the network may not discard (it threw
/// away good potions all the time); the engine itself still allows it.
pub fn discard_useful(run: &Run) -> bool {
    run.potions.iter().all(|p| p.is_some())
        && match &run.screen {
            Screen::Rewards { items } => items.iter().any(|r| matches!(r, Reward::Potion(_))),
            Screen::Shop { shop } => !shop.potions.is_empty(),
            _ => false,
        }
}

/// Whether gold is still on the reward screen. Gold never hurts, so the network may not leave it
/// (it sometimes did); the engine itself still allows it.
pub fn gold_on_offer(run: &Run) -> bool {
    matches!(&run.screen, Screen::Rewards { items } if items.iter().any(|r| matches!(r, Reward::Gold(_) | Reward::StolenGold(_))))
}

/// The run action for slot `a` outside combat; `None` while a grid selection is still being
/// picked (the pick is added to `picked`).
pub fn resolve(run: &Run, legal: &[RunAction], picked: &mut Vec<usize>, skipped: &[Vec<(i32, i32)>], a: usize) -> Result<Option<RunAction>, String> {
    let bad = || format!("action slot {a} is not allowed on this screen");
    if a == CONFIRM {
        return legal.iter().find(|l| matches!(l, RunAction::Proceed | RunAction::SkipCards)).cloned().map(Some).ok_or_else(bad);
    }
    if (POTION..POTION + MAX_POTIONS * encode::TARGETS).contains(&a) {
        let (slot, t) = ((a - POTION) / encode::TARGETS, (a - POTION) % encode::TARGETS);
        let want = match t {
            0 => RunAction::UsePotion { slot },
            1 => RunAction::DiscardPotion { slot },
            _ => return Err(bad()),
        };
        return if legal.contains(&want) { Ok(Some(want)) } else { Err(bad()) };
    }
    if !(PICK..CONFIRM).contains(&a) {
        return Err(bad());
    }
    let opts = options(run, picked, skipped);
    let o = opts.get(a - PICK).filter(|o| !o.blocked).ok_or_else(bad)?;
    match &o.action {
        OptAction::Run(x) => {
            if legal.contains(x) {
                Ok(Some(x.clone()))
            } else {
                Err(bad())
            }
        }
        OptAction::GridPick(d) => {
            let count = grid_count(run);
            if picked.contains(d) || picked.len() >= count {
                return Err(bad());
            }
            picked.push(*d);
            if picked.len() < count {
                return Ok(None);
            }
            // Legal selections list deck indices in the grid's order.
            let Screen::Grid { options, .. } = &run.screen else { return Err(bad()) };
            let mut indices = std::mem::take(picked);
            indices.sort_by_key(|i| options.iter().position(|o| o == i));
            let want = RunAction::Select { indices };
            if legal.contains(&want) {
                Ok(Some(want))
            } else {
                Err(format!("grid selection {want:?} is not legal"))
            }
        }
    }
}

pub fn vocab_sizes() -> Vec<(&'static str, usize)> {
    vec![
        ("option_kind", OPTION_KINDS.len() + 1),
        ("screen", SCREENS.len() + 1),
        ("encounter", encounters::all().len() + 1),
        ("room", 8),
        ("event", sts_core::run::events::EventId::WindingHalls as usize + 2),
        ("event_screen", 17),
        ("event_index", MAX_OPTIONS + 1),
        ("grid_purpose", GRID_PURPOSES + 1),
        ("neow", sts_core::run::NeowReward::BossRelic as usize + 2),
        ("neow_drawback", sts_core::run::NeowDrawback::PercentDamage as usize + 2),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts_core::potions::PotionId;
    use sts_core::run::RunConfig;

    /// Discarding is offered only to make room for a potion on offer, with every slot full.
    #[test]
    fn potions_are_discarded_only_to_make_room() {
        let mut r = Run::new(RunConfig::new(1));
        let n = r.potions.len();
        r.potions = vec![Some(PotionId::BloodPotion); n];
        r.screen = Screen::Rewards { items: vec![Reward::Gold(10), Reward::Potion(PotionId::Elixir)] };
        assert!(discard_useful(&r));
        r.screen = Screen::Rewards { items: vec![Reward::Gold(10)] };
        assert!(!discard_useful(&r));
        r.screen = Screen::Map;
        assert!(!discard_useful(&r));
        r.screen = Screen::Rewards { items: vec![Reward::Potion(PotionId::Elixir)] };
        r.potions[0] = None;
        assert!(!discard_useful(&r));
    }

    fn rows(r: &Run) -> (Vec<i32>, Vec<f32>, Vec<u8>) {
        let legal = r.legal_actions();
        let (mut ints, mut floats, mut mask) = (vec![0; encode::int_len() + run_int_len()], vec![0.0; encode::float_len() + run_float_len()], vec![0; encode::ACTIONS]);
        encode_run(r, &legal, &[], &[], &mut ints, &mut floats, &mut mask);
        (ints, floats, mask)
    }

    /// The moves open to the network: the options' kinds, and whether it can leave.
    fn open(r: &Run) -> (Vec<i32>, bool) {
        let (_, _, mask) = rows(r);
        let opts = options(r, &[], &[]);
        (opts.iter().enumerate().filter(|(k, _)| mask[PICK + k] == 1).map(|(_, o)| o.kind).collect(), mask[CONFIRM] == 1)
    }

    fn with_key_rule(act: i32) -> Run {
        let mut cfg = RunConfig::new(1);
        cfg.key_rule = Some(KeyRule { recall_hp_act2: 101, recall_hp_act3: 60 });
        let mut r = Run::new(cfg);
        r.act = act;
        r
    }

    /// The rule's rests: Recall in Act 3 from 60% of HP, or at the last rest; otherwise it's out of sight.
    #[test]
    fn key_rule_recalls_in_act_3() {
        let rest = |act: i32, hp: i32, y: i32| {
            let mut r = with_key_rule(act);
            r.screen = Screen::Rest;
            r.pos = Some((0, y));
            r.max_hp = 100;
            r.hp = hp;
            r
        };
        assert_eq!(open(&rest(3, 60, 5)).0, vec![kind("recall")]);
        assert_eq!(open(&rest(3, 20, 14)).0, vec![kind("recall")]);
        let low = open(&rest(3, 59, 5)).0;
        assert!(!low.is_empty() && !low.contains(&kind("recall")));
        let early = rest(2, 100, 14);
        assert!(!open(&early).0.contains(&kind("recall")));
        assert!(!options(&early, &[], &[]).iter().any(|o| o.kind == kind("recall")), "out of sight");
        let mut held = rest(3, 100, 5);
        held.keys.ruby = true;
        assert!(!open(&held).0.is_empty() && !open(&held).0.contains(&kind("recall")));
        // A rule that recalls in Act 2 too, from its own share of HP.
        let mut act2 = rest(2, 90, 5);
        act2.config.key_rule = Some(KeyRule { recall_hp_act2: 90, recall_hp_act3: 75 });
        assert_eq!(open(&act2).0, vec![kind("recall")]);
        act2.hp = 89;
        assert!(!open(&act2).0.contains(&kind("recall")));
        // Nothing else to do at the site: Recall, whatever the act, so that some move stays open.
        let mut alone = rest(1, 10, 5);
        alone.deck.clear();
        alone.relics.push(sts_core::relics::RelicState::new(sts_core::relics::RelicId::CoffeeDripper));
        assert_eq!(open(&alone).0, vec![kind("recall")]);
    }

    /// The rule's reward screens: the Sapphire key over the chest's relic in Act 3, out of sight before; an Emerald key always.
    #[test]
    fn key_rule_takes_rewards() {
        let chest = vec![Reward::Relic(sts_core::relics::RelicId::Anchor), Reward::SapphireKey];
        let mut r = with_key_rule(3);
        r.screen = Screen::Rewards { items: chest.clone() };
        assert_eq!(open(&r), (vec![kind("sapphire_key")], false));
        let mut r = with_key_rule(2);
        r.screen = Screen::Rewards { items: chest };
        assert_eq!(open(&r), (vec![kind("relic")], true));
        assert!(!options(&r, &[], &[]).iter().any(|o| o.kind == kind("sapphire_key")));
        let mut r = with_key_rule(1);
        r.screen = Screen::Rewards { items: vec![Reward::Gold(20), Reward::EmeraldKey] };
        assert_eq!(open(&r), (vec![kind("emerald_key")], false));
        r.step(&RunAction::TakeReward { index: 1 }).unwrap();
        assert!(r.keys.emerald);
        assert_eq!(open(&r), (vec![kind("gold")], false), "gold can't be left");
    }

    /// The rule's map in Act 3: only moves towards the burning elite, until there are none. Later acts' burning elites and the keys
    /// are hidden from the inputs; the rule off, they aren't.
    #[test]
    fn key_rule_goes_for_the_burning_elite() {
        let mut r = with_key_rule(3);
        r.screen = Screen::Map;
        r.pos = None;
        let to = towards_burning_elite(&r);
        let starts = r.reachable_nodes();
        assert!(starts.len() >= 2);
        assert!(to[0].iter().any(|t| *t), "the map holds one");
        let want = starts.iter().filter(|x| to[0][**x as usize]).count();
        assert!(want >= 1 && want == open(&r).0.len(), "{want} of {starts:?}");
        assert!(options(&r, &[], &[]).iter().all(|o| o.f[6] == 0.0));

        let key_at = encode::float_len() + 7;
        let (_, floats, _) = rows(&r);
        assert_eq!(floats[key_at..key_at + 3], [1.0; 3]);
        r.keys.emerald = true;
        assert_eq!(open(&r).0.len(), starts.len(), "nothing to go for");

        let mut off = Run::new(RunConfig::new(1));
        off.act = 3;
        off.screen = Screen::Map;
        off.pos = None;
        assert_eq!(open(&off).0.len(), off.reachable_nodes().len());
        let (_, floats, _) = rows(&off);
        assert_eq!(floats[key_at..key_at + 3], [0.0; 3]);
    }

    /// A reward screen can't be left while gold is on it; once taken, it can.
    #[test]
    fn gold_is_never_left() {
        let mut r = Run::new(RunConfig::new(1));
        let leave = |r: &Run| {
            let legal = r.legal_actions();
            let (mut ints, mut floats, mut mask) = (vec![0; encode::int_len() + run_int_len()], vec![0.0; encode::float_len() + run_float_len()], vec![0; encode::ACTIONS]);
            encode_run(r, &legal, &[], &[], &mut ints, &mut floats, &mut mask);
            mask[CONFIRM] == 1
        };
        for gold in [Reward::Gold(15), Reward::StolenGold(30)] {
            r.screen = Screen::Rewards { items: vec![gold, Reward::Potion(PotionId::Elixir)] };
            assert!(gold_on_offer(&r) && !leave(&r));
            r.step(&RunAction::TakeReward { index: 0 }).unwrap();
            assert!(!gold_on_offer(&r) && leave(&r));
        }
    }
}
