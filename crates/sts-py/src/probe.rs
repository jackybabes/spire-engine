//! The deck network's lookup (jack, 2026-10-30: "once this is done we use it as lookup for deck network ...
//! Whenever decisions check the improvement to scores"): at a card reward, each option (skipping: the deck as
//! it is; or each card offered, added) set up at the start of probe fights, for the fight network's
//! fight-outcome head to score (python/core/lookup.py): this act's boss, and the next act's bosses (each of the
//! three; the Heart after Act 3). Each probe is played from a few redrawn openings (the fight's own streams
//! reseeded per draw: the shuffle, the monsters' first moves and HP), so one hand doesn't decide it.
//!
//! A probe fight is the run as it stands (deck, relics, potions, HP, max HP, ascension) moved to the boss's
//! floor of that act and into its boss room; for the Heart, Act 4 with the final act on and the three keys.
//! Nothing else changes: no rest, no upgrades, no cards from the floors between.
//!
//! `options` probes the other choices that change what a fight starts with the same way (2026-10-30, the weekend's
//! work on the deck network): each option applied to a copy of the run, then probed. At a rest site resting
//! (the HP), lifting, digging; Smith and Peace Pipe by the card each leads to (`Probe::sub`: the best is the
//! option's); a grid's card upgraded, removed, bottled or copied; a shop's cards, relics and potions bought,
//! its removal by the card removed; a boss relic taken; a reward's relic or potion taken. The copy's random
//! streams start afresh first, so an option with a random outcome (a relic that transforms, Shovel's relic) is
//! one sample of it, never the run's own future.
//!
//! An event's options and Neow's are probed too: each applied to whole copies of the run with fresh streams, one sample of its
//! outcome per opening (the probes average over the outcomes as well as the openings).
//!
//! At the map (Acts 1-3), a path into an elite is probed against each of the act's elites and a path into a hallway
//! fight against each of its strong hallway encounters, from the run as it stands on the next floor (one opening each:
//! averaged over the encounters): how dangerous that room is now (python/core/lookup.py writes the win chance and HP
//! left themselves there, not gains).

use sts_core::cards::Card;
use sts_core::encounters::{ACT1_ELITES, ACT1_STRONG, ACT2_BOSSES, ACT2_ELITES, ACT2_STRONG, ACT3_BOSSES, ACT3_ELITES, ACT3_STRONG};
use sts_core::rngs::Rngs;
use sts_core::run::map::{Map, RoomType, MAP_HEIGHT};
use sts_core::run::{GridPurpose, Reward, Run, RunAction, Screen};
use sts_encode::encode_run::{self, OptAction};

/// A probe's fight: the act and the boss.
#[derive(Clone, Debug, PartialEq)]
pub struct Fight {
    pub act: i32,
    pub key: String,
}

/// The fights a run at `run.act` is probed against: this act's boss (`run.boss`), then the next act's bosses
/// (Act 3's: the Heart). Act 4: the Heart alone.
pub fn fights(run: &Run) -> Vec<Fight> {
    let this = Fight { act: run.act.min(4), key: if run.act >= 4 { "The Heart".to_string() } else { run.boss.clone() } };
    let next: Vec<Fight> = match run.act {
        1 => ACT2_BOSSES.iter().map(|k| Fight { act: 2, key: k.to_string() }).collect(),
        2 => ACT3_BOSSES.iter().map(|k| Fight { act: 3, key: k.to_string() }).collect(),
        3 => vec![Fight { act: 4, key: "The Heart".to_string() }],
        _ => Vec::new(),
    };
    std::iter::once(this).chain(next).collect()
}

/// How many openings fight `f` of `fights` is read from: this act's boss `draws`; each of the next act's bosses enough
/// that they share about `draws` between them (one each at two draws: their scores are averaged over the bosses anyway,
/// so a third of the probes for the same noise in that mean; 2026-10-30, the lookup at every choice).
pub fn fight_draws(fights: &[Fight], f: usize, draws: u32) -> u32 {
    if f == 0 { draws } else { draws.div_ceil((fights.len() - 1).max(1) as u32) }
}

/// What makes two deck cards the same to a fight (copies of one card probe alike).
fn same_card(a: &Card, b: &Card) -> bool {
    a.id == b.id && a.times_upgraded == b.times_upgraded && a.misc == b.misc && a.bottled == b.bottled
}

/// `run` (with `extra` added to its deck) at the start of `fight`, its fight's streams reseeded from `seed`;
/// None if the fight didn't start.
pub fn fight_run(run: &Run, extra: Option<&Card>, fight: &Fight, seed: i64) -> Option<Run> {
    let mut r = run.clone();
    if let Some(c) = extra {
        r.deck.push(c.clone());
    }
    r.combat = None;
    r.screen = Screen::Map;
    r.act = fight.act;
    r.floor = [16, 33, 50, 55][(fight.act.clamp(1, 4) - 1) as usize];
    if fight.act == 4 {
        r.config.final_act = true;
        r.keys.ruby = true;
        r.keys.emerald = true;
        r.keys.sapphire = true;
    }
    r.boss = fight.key.clone();
    r.boss_list = Vec::new();
    r.outcome = None; // (an option that ended the run: the last boss relic without the keys)
    r.rng.reseed_fight(seed);
    r.start_fight(&fight.key, RoomType::Boss);
    matches!(r.screen, Screen::Combat).then_some(r)
}

/// `run` without what a probe fight doesn't use (the map, the act's monster, elite and event lists), so the few dozen
/// copies a card reward's probes make are cheap (2026-10-30: copying the whole run was most of the lookup's time). The
/// screen is kept: `options` applies the screen's options to it (`fight_run` leaves it for the boss room).
fn light(run: &Run) -> Run {
    let mut r = run.clone();
    r.map = Map { rows: Vec::new() };
    for list in [&mut r.monster_list, &mut r.elite_list, &mut r.event_list, &mut r.shrine_list, &mut r.special_events] {
        list.clear();
    }
    r.combat = None;
    r
}

/// One probe: which option (-1: skip, the deck as it is; else the card offered at that index; in `options`, the
/// option's index in `encode_run::options`), for an option that leads to a choice of a card (Smith, Peace Pipe, the
/// merchant's removal) which card (`sub`: its index in that grid's options; -1 none), which fight (an index into
/// `fights`), which draw.
#[derive(Clone, Copy, Debug)]
pub struct Probe {
    pub option: i32,
    pub sub: i32,
    pub fight: u8,
    pub draw: u32,
}

/// The probes of a run at a card reward: every option against every fight, `draws` openings each (the same
/// openings for every option: draw d is seeded `seed + d` whatever the option, so options differ by the
/// card, not the luck). Empty if `run` isn't at a card reward.
pub fn card_reward(run: &Run, draws: u32, seed: i64) -> Vec<(Probe, Run)> {
    let mut out = Vec::new();
    card_reward_each(run, draws, seed, |p, r| out.push((p, r)));
    out
}

/// `card_reward`, each probe handed to `emit` as it is set up (the caller encodes it and drops it: no pile of copies).
pub fn card_reward_each(run: &Run, draws: u32, seed: i64, mut emit: impl FnMut(Probe, Run)) {
    let Screen::CardReward { cards, .. } = &run.screen else { return };
    let fights = fights(run);
    let base = light(run);
    for option in -1..cards.len() as i32 {
        let extra = (option >= 0).then(|| &cards[option as usize]);
        for (f, fight) in fights.iter().enumerate() {
            for d in 0..fight_draws(&fights, f, draws) {
                if let Some(r) = fight_run(&base, extra, fight, seed.wrapping_add(d as i64 * 7919)) {
                    emit(Probe { option, sub: -1, fight: f as u8, draw: d }, r);
                }
            }
        }
    }
}

/// Grids whose single card the lookup probes: upgraded, removed, bottled, copied.
fn probed_grid(p: GridPurpose) -> bool {
    matches!(p, GridPurpose::Upgrade | GridPurpose::Smith | GridPurpose::Remove | GridPurpose::Toke | GridPurpose::ShopPurge
        | GridPurpose::Bottle(_) | GridPurpose::Duplicate)
}

/// The encounters a path into a room of this type is probed against: the act's elites, or its strong hallway fights.
fn room_fights(act: i32, room: RoomType) -> &'static [&'static str] {
    match (act, room) {
        (1, RoomType::Elite) => &ACT1_ELITES,
        (1, RoomType::Monster) => &ACT1_STRONG,
        (2, RoomType::Elite) => &ACT2_ELITES,
        (2, RoomType::Monster) => &ACT2_STRONG,
        (3, RoomType::Elite) => &ACT3_ELITES,
        (3, RoomType::Monster) => &ACT3_STRONG,
        _ => &[],
    }
}

/// `run` (a light copy) at the start of the hallway or elite fight `key` on the next floor, its fight's streams reseeded
/// from `seed`; None if the fight didn't start.
fn room_run(run: &Run, key: &str, room: RoomType, seed: i64) -> Option<Run> {
    let mut r = run.clone();
    r.combat = None;
    r.screen = Screen::Map;
    r.floor += 1;
    r.outcome = None;
    r.rng.reseed_fight(seed);
    r.start_fight(key, room);
    matches!(r.screen, Screen::Combat).then_some(r)
}

/// The map's probes (see the module docs): for each path option into an elite or a hallway fight, the run against each
/// of the act's encounters of that kind; a second path into the same kind of room is an alias of the first.
fn map_options(run: &Run, picked: &[usize], skipped: &[Vec<(i32, i32)>], seed: i64, mut emit: impl FnMut(Probe, Run)) -> Vec<(i32, i32)> {
    let mut aliases = Vec::new();
    let y = run.pos.map_or(0, |(_, y)| y + 1);
    if y as usize >= MAP_HEIGHT || run.map.rows.len() < MAP_HEIGHT {
        return aliases; // (the boss next: probed at every other choice already)
    }
    let base = light(run);
    let mut first: Vec<(RoomType, i32)> = Vec::new();
    for (k, o) in encode_run::options(run, picked, skipped).iter().enumerate() {
        let OptAction::Run(RunAction::Path { x }) = &o.action else { continue };
        let Some(room) = run.map.node(*x, y).room else { continue };
        let keys = room_fights(run.act, room);
        if keys.is_empty() {
            continue;
        }
        if let Some(&(_, k0)) = first.iter().find(|(r, _)| *r == room) {
            aliases.push((k as i32, k0));
            continue;
        }
        first.push((room, k as i32));
        for (f, key) in keys.iter().enumerate() {
            if let Some(p) = room_run(&base, key, room, seed.wrapping_add(f as i64 * 104_729)) {
                emit(Probe { option: k as i32, sub: -1, fight: f as u8, draw: 0 }, p);
            }
        }
    }
    aliases
}

/// Whether `options` probes `run`'s screen (`picked`: the cards already picked on a grid).
pub fn probed(run: &Run, picked: &[usize]) -> bool {
    match &run.screen {
        Screen::Map => run.act <= 3,
        Screen::CardReward { .. } | Screen::Rest | Screen::BossRelic { .. } | Screen::Shop { .. } | Screen::Rewards { .. } => true,
        Screen::Event { .. } | Screen::Neow { .. } => true,
        Screen::Grid { purpose, count, .. } => *count == 1 && picked.is_empty() && probed_grid(*purpose),
        _ => false,
    }
}

/// `run` (a light copy) after `action`, its random streams fresh from `seed`; None if the action isn't possible.
fn after(run: &Run, action: &RunAction, seed: i64) -> Option<Run> {
    let mut r = run.clone();
    r.rng = Rngs::new(seed);
    r.step(action).ok()?;
    Some(r)
}

/// The probes of every option of `run`'s screen that changes what a fight starts with (see the module docs), against
/// the fights of `run` as it stands, `draws` openings each (the same for every option, as `card_reward`); the deck as it
/// stands is option -1. Card rewards as `card_reward`. Empty on a screen `probed` doesn't take.
#[cfg_attr(not(test), allow(dead_code))]
pub fn options(run: &Run, picked: &[usize], skipped: &[Vec<(i32, i32)>], draws: u32, seed: i64) -> Vec<(Probe, Run)> {
    let mut out = Vec::new();
    options_each(run, picked, skipped, draws, seed, |p, r| out.push((p, r)));
    out
}

/// `options`, each probe handed to `emit` as it is set up. Returns the options not probed because a card the same as
/// theirs was (a grid's copies of one card: (option, the option probed for it)).
pub fn options_each(run: &Run, picked: &[usize], skipped: &[Vec<(i32, i32)>], draws: u32, seed: i64, mut emit: impl FnMut(Probe, Run))
    -> Vec<(i32, i32)> {
    let mut aliases = Vec::new();
    if matches!(run.screen, Screen::CardReward { .. }) {
        card_reward_each(run, draws, seed, emit);
        return aliases;
    }
    if !probed(run, picked) {
        return aliases;
    }
    if matches!(run.screen, Screen::Map) {
        return map_options(run, picked, skipped, seed, emit);
    }
    // The options it probes (a reward screen of gold and cards alone: none, and no probes at all).
    let actions: Vec<(usize, RunAction)> = encode_run::options(run, picked, skipped).iter().enumerate().filter_map(|(k, o)| {
        let action = match &o.action {
            OptAction::GridPick(d) => RunAction::Select { indices: vec![*d] },
            OptAction::Run(a) => match (a, &run.screen) {
                (RunAction::TakeReward { index }, Screen::Rewards { items }) => match items.get(*index) {
                    Some(Reward::Relic(_) | Reward::Potion(_)) => a.clone(),
                    _ => return None,
                },
                (RunAction::Rest | RunAction::Smith | RunAction::Lift | RunAction::Toke | RunAction::Dig | RunAction::BossRelic { .. }
                    | RunAction::BuyCard { .. } | RunAction::BuyRelic { .. } | RunAction::BuyPotion { .. } | RunAction::Purge, _) => a.clone(),
                (RunAction::EventChoice { .. } | RunAction::Neow { .. }, _) => a.clone(),
                _ => return None,
            },
        };
        Some((k, action))
    }).collect();
    // An event's or Neow's option: its outcome is random, so each opening is a fresh sample of it (below).
    let random = matches!(run.screen, Screen::Event { .. } | Screen::Neow { .. });
    if actions.is_empty() {
        return aliases;
    }
    let fights = fights(run);
    let base = light(run);
    let mut push = |option: i32, sub: i32, r: &Run| {
        for (f, fight) in fights.iter().enumerate() {
            for d in 0..fight_draws(&fights, f, draws) {
                if let Some(p) = fight_run(r, None, fight, seed.wrapping_add(d as i64 * 7919)) {
                    emit(Probe { option, sub, fight: f as u8, draw: d }, p);
                }
            }
        }
    };
    push(-1, -1, &base);
    let fresh = seed ^ 0x6C6F_6F6B; // ("look"): the copies' streams, the same for every option
    if random {
        // Each option played on a whole copy of the run (events and Neow read more of it than a light copy keeps) under a panic
        // guard (paths the training games rarely take), once per opening with its own streams, each sample against the fights
        // of that opening: the probes' mean is over the outcomes too. A fight the option starts, or a screen it opens, shows as
        // the run it leaves at that point.
        for (k, action) in actions {
            for d in 0..draws {
                let sample = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut r = run.clone();
                    r.rng = Rngs::new(fresh.wrapping_add((d as i64 + 1) * 104_729));
                    r.step(&action).ok()?;
                    Some(light(&r))
                }));
                let Ok(Some(r)) = sample else { continue };
                for (f, fight) in fights.iter().enumerate() {
                    if d < fight_draws(&fights, f, draws) {
                        if let Some(p) = fight_run(&r, None, fight, seed.wrapping_add(d as i64 * 7919)) {
                            emit(Probe { option: k as i32, sub: -1, fight: f as u8, draw: d }, p);
                        }
                    }
                }
            }
        }
        return aliases;
    }
    let mut grid_seen: Vec<(usize, i32)> = Vec::new(); // (deck index, option) of the grid's cards probed so far
    for (k, action) in actions {
        if let RunAction::Select { indices } = &action {
            // A grid's copy of a card already probed: that option's scores.
            let d = indices[0];
            if let Some(&(_, k0)) = grid_seen.iter().find(|(e, _)| same_card(&run.deck[*e], &run.deck[d])) {
                aliases.push((k as i32, k0));
                continue;
            }
            grid_seen.push((d, k as i32));
        }
        let Some(r) = after(&base, &action, fresh) else { continue };
        match &r.screen {
            // Smith, Peace Pipe, the merchant's removal: each card it could take (one of each kind), the best the option's.
            Screen::Grid { purpose, options, count: 1 } if probed_grid(*purpose) && !matches!(run.screen, Screen::Grid { .. }) => {
                for (j, &d) in options.iter().enumerate() {
                    if options[..j].iter().any(|&e| same_card(&r.deck[e], &r.deck[d])) {
                        continue;
                    }
                    if let Some(r2) = after(&r, &RunAction::Select { indices: vec![d] }, fresh) {
                        push(k as i32, j as i32, &r2);
                    }
                }
            }
            _ => push(k as i32, -1, &r),
        }
    }
    aliases
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts_core::cards::CardId;
    use sts_core::run::RunConfig;

    fn at_card_reward(act: i32) -> Run {
        let mut cfg = RunConfig::new(3);
        cfg.ascension = 20;
        let mut r = Run::new(cfg);
        r.act = act;
        r.floor = [5, 22, 39, 54][(act - 1) as usize];
        r.boss = ["Hexaghost", "Champ", "Time Eater", "The Heart"][(act - 1) as usize].to_string();
        let cards = vec![r.new_card(CardId::Inflame), r.new_card(CardId::ShrugItOff)];
        r.screen = Screen::CardReward { item: None, cards };
        r
    }

    /// Every option against this act's boss and the next act's, from the same openings; the fights start with
    /// the deck plus the option's card, at the boss's floor.
    #[test]
    fn card_reward_probes_every_option_against_the_bosses() {
        for act in 1..=4 {
            let r = at_card_reward(act);
            let fights = fights(&r);
            assert_eq!(fights.len(), [4, 4, 2, 1][(act - 1) as usize], "act {act}");
            let probes = card_reward(&r, 2, 11);
            let per_option: u32 = (0..fights.len()).map(|f| fight_draws(&fights, f, 2)).sum();
            assert_eq!(per_option, [5, 5, 4, 2][(act - 1) as usize], "act {act}: this boss twice, the next act's once each");
            assert_eq!(probes.len() as u32, 3 * per_option, "act {act}: every probe starts");
            for (p, run) in &probes {
                let c = run.combat.as_ref().unwrap();
                let fight = &fights[p.fight as usize];
                assert_eq!(run.act, fight.act);
                assert_eq!(run.deck.len(), r.deck.len() + (p.option >= 0) as usize);
                if p.option >= 0 {
                    let want = [CardId::Inflame, CardId::ShrugItOff][p.option as usize];
                    assert_eq!(run.deck.last().unwrap().id, want);
                }
                assert!(c.outcome.is_none() && !c.monsters.is_empty());
            }
            // The same opening for skipping and for a card: the same monsters' HP.
            let hp = |o: i32, d: u32| probes.iter().find(|(p, _)| p.option == o && p.fight == 0 && p.draw == d)
                .map(|(_, r)| r.combat.as_ref().unwrap().monsters.iter().map(|m| m.hp).collect::<Vec<_>>()).unwrap();
            assert_eq!(hp(-1, 0), hp(1, 0), "act {act}");
        }
        assert!(card_reward(&Run::new(RunConfig::new(1)), 2, 1).is_empty(), "not at a card reward");
    }

    /// The other choices: resting probes the run healed; Smith each card it could upgrade (as a grid's upgrade
    /// would); a boss relic the run with it; every option against the same fights from the same openings.
    #[test]
    fn options_probe_rests_smiths_and_boss_relics() {
        let mut r = at_card_reward(1);
        r.hp = 30;
        r.screen = Screen::Rest;
        let probes = options(&r, &[], &[], 2, 5);
        let fs = fights(&r);
        let per = (0..fs.len()).map(|f| fight_draws(&fs, f, 2)).sum::<u32>() as i32; // (probes an option)
        let opts = encode_run::options(&r, &[], &[]);
        let rest = opts.iter().position(|o| matches!(o.action, OptAction::Run(RunAction::Rest))).unwrap() as i32;
        let smith = opts.iter().position(|o| matches!(o.action, OptAction::Run(RunAction::Smith))).unwrap() as i32;
        let of = |k: i32| probes.iter().filter(|(p, _)| p.option == k).collect::<Vec<_>>();
        assert_eq!(of(-1).len() as i32, per, "the run as it stands");
        assert!(of(-1).iter().all(|(_, run)| run.combat.as_ref().unwrap().player.hp == 30));
        assert_eq!(of(rest).len() as i32, per);
        assert!(of(rest).iter().all(|(p, run)| p.sub == -1 && run.combat.as_ref().unwrap().player.hp > 30), "rested");
        let kinds = 3; // (the starting deck's Strikes, Defends and Bash: one of each kind)
        assert_eq!(of(smith).len() as i32, kinds * per, "Smith: every kind of card it could upgrade");
        for (p, run) in of(smith) {
            let ups = run.deck.iter().filter(|c| c.times_upgraded > 0).count();
            assert!(p.sub >= 0 && ups == 1, "one card upgraded");
        }
        // The same fight and draw start the same for every option (the monsters' HP).
        let hp = |k: i32| of(k).iter().find(|(p, _)| p.fight == 0 && p.draw == 1)
            .map(|(_, run)| run.combat.as_ref().unwrap().monsters.iter().map(|m| m.hp).collect::<Vec<_>>()).unwrap();
        assert_eq!(hp(-1), hp(rest));

        let mut b = at_card_reward(1);
        b.screen = Screen::BossRelic { relics: vec![sts_core::relics::RelicId::CursedKey, sts_core::relics::RelicId::FusionHammer] };
        let probes = options(&b, &[], &[], 1, 5);
        assert_eq!(probes.iter().filter(|(p, _)| p.option == 1).count(), probes.iter().filter(|(p, _)| p.option == -1).count());
        assert!(probes.iter().any(|(p, run)| p.option == 0 && run.relics.iter().any(|x| x.id == sts_core::relics::RelicId::CursedKey)));
        assert!(probes.iter().all(|(p, run)| p.option != 1 || run.relics.iter().any(|x| x.id == sts_core::relics::RelicId::FusionHammer)));
        assert!(probes.iter().all(|(_, run)| run.combat.is_some()), "every probe starts its fight");
        let mut not_probed = Run::new(RunConfig::new(1));
        not_probed.screen = Screen::Combat;
        assert!(options(&not_probed, &[], &[], 2, 1).is_empty(), "not a probed screen");

        // A removal grid: one probe set per kind of card, the copies its aliases.
        let mut g = at_card_reward(1);
        let opts: Vec<usize> = (0..g.deck.len()).collect();
        g.screen = Screen::Grid { purpose: GridPurpose::Remove, options: opts, count: 1 };
        let mut probes = Vec::new();
        let aliases = options_each(&g, &[], &[], 2, 5, |p, r| probes.push((p, r)));
        let probed: std::collections::BTreeSet<i32> = probes.iter().map(|(p, _)| p.option).filter(|&o| o >= 0).collect();
        let kinds = g.deck.iter().enumerate().filter(|(i, c)| !g.deck[..*i].iter().any(|e| same_card(e, c))).count();
        assert_eq!(probed.len(), kinds, "one of each kind (Strike, Defend, Bash, Ascender's Bane at A20)");
        assert_eq!(aliases.len(), g.deck.len() - kinds);
        for (k, k0) in aliases {
            assert!(probed.contains(&k0) && same_card(&g.deck[k as usize], &g.deck[k0 as usize]));
        }
    }

    /// Neow (a new run's first screen): the run as it stands and each blessing, every probe starting its fight; the blessings'
    /// samples differ by opening (a fresh outcome each).
    #[test]
    fn neow_blessings_are_probed() {
        let mut cfg = RunConfig::new(4);
        cfg.ascension = 20;
        let r = Run::new(cfg);
        assert!(matches!(r.screen, Screen::Neow { .. }), "a new run starts at Neow");
        let probes = options(&r, &[], &[], 2, 7);
        let opts: std::collections::BTreeSet<i32> = probes.iter().map(|(p, _)| p.option).collect();
        assert!(opts.contains(&-1) && opts.len() >= 3, "{opts:?}");
        assert!(probes.iter().all(|(_, run)| run.combat.is_some()));
    }

    /// The map: a path into an elite against each of the act's elites, into a hallway fight against its strong
    /// encounters, on the next floor with the run's HP; a second path into the same kind of room an alias.
    #[test]
    fn map_paths_probe_the_rooms_they_lead_to() {
        let mut cfg = RunConfig::new(9);
        cfg.ascension = 20;
        let mut r = Run::new(cfg);
        r.screen = Screen::Map;
        r.hp = 40;
        // Walk the map until a choice leads to an elite or a hallway fight.
        let mut found = false;
        for _ in 0..14 {
            let mut probes = Vec::new();
            let aliases = options_each(&r, &[], &[], 2, 3, |p, run| probes.push((p, run)));
            let opts = encode_run::options(&r, &[], &[]);
            let y = r.pos.map_or(0, |(_, y)| y + 1);
            for (k, o) in opts.iter().enumerate() {
                let OptAction::Run(RunAction::Path { x }) = &o.action else { continue };
                let room = r.map.node(*x, y).room;
                let mine: Vec<_> = probes.iter().filter(|(p, _)| p.option == k as i32).collect();
                match room {
                    Some(RoomType::Elite) | Some(RoomType::Monster) if !aliases.iter().any(|(a, _)| *a == k as i32) => {
                        let keys = room_fights(1, room.unwrap());
                        assert_eq!(mine.len(), keys.len(), "every encounter of that kind");
                        for (p, run) in &mine {
                            assert_eq!(run.combat.as_ref().unwrap().player.hp, 40);
                            assert_eq!(run.room, room);
                            assert_eq!(run.floor, r.floor + 1);
                            assert!(p.fight < keys.len() as u8);
                        }
                        found = true;
                    }
                    _ => assert!(mine.is_empty(), "no probes for {room:?}"),
                }
            }
            // On to the next floor along the first path (its room skipped: the map screen again).
            let Some(OptAction::Run(a)) = opts.first().map(|o| o.action.clone()) else { break };
            let RunAction::Path { x } = a else { break };
            r.pos = Some((x, y));
            r.floor += 1;
        }
        assert!(found, "some path led to an elite or a hallway fight");
    }
}
