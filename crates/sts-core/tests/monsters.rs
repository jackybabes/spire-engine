//! Every monster, checked against its class in the decompiled game (`monsters/`): HP and damage at
//! each ascension threshold, the move rules of `getMove`, and the powers and mechanics each one
//! brings. The recordings and soaks checked the same monsters in whole fights.

use sts_core::action::{Action, Cr};
use sts_core::cards::CardId;
use sts_core::character::Character;
use sts_core::combat::{Combat, CombatConfig, PlayerAction};
use sts_core::encounters;
use sts_core::monsters::{Monster, MonsterId};
use sts_core::potions::PotionId;
use sts_core::powers::{Power, PowerId};
use sts_core::rng::StsRandom;
use sts_core::rngs::Rngs;

// ---- HP and damage ----

type Row = (i32, Option<(i32, i32)>, &'static [Option<i32>]);

/// (monster, an encounter it appears in, rows from the first ascension each applies: (ascension,
/// HP range, damage bases in `damage` order, None where rolled)). Read off the constructors.
#[rustfmt::skip]
const STATS: &[(MonsterId, Option<&str>, &[Row])] = &[
    (MonsterId::AcidSlimeL, Some("Large Slime"), &[(0, Some((65, 69)), &[Some(11), Some(16)]), (2, Some((65, 69)), &[Some(12), Some(18)]), (7, Some((68, 72)), &[Some(12), Some(18)])]),
    (MonsterId::AcidSlimeM, Some("Small Slimes"), &[(0, Some((28, 32)), &[Some(7), Some(10)]), (2, Some((28, 32)), &[Some(8), Some(12)]), (7, Some((29, 34)), &[Some(8), Some(12)])]),
    (MonsterId::AcidSlimeS, Some("Lots of Slimes"), &[(0, Some((8, 12)), &[Some(3)]), (2, Some((8, 12)), &[Some(4)]), (7, Some((9, 13)), &[Some(4)])]),
    (MonsterId::Cultist, Some("Cultist"), &[(0, Some((48, 54)), &[Some(6)]), (7, Some((50, 56)), &[Some(6)])]),
    (MonsterId::FungiBeast, Some("2 Fungi Beasts"), &[(0, Some((22, 28)), &[Some(6)]), (7, Some((24, 28)), &[Some(6)])]),
    (MonsterId::GremlinFat, Some("Gremlin Gang"), &[(0, Some((13, 17)), &[Some(4)]), (2, Some((13, 17)), &[Some(5)]), (7, Some((14, 18)), &[Some(5)])]),
    (MonsterId::GremlinNob, Some("Gremlin Nob"), &[(0, Some((82, 86)), &[Some(14), Some(6)]), (3, Some((82, 86)), &[Some(16), Some(8)]), (8, Some((85, 90)), &[Some(16), Some(8)])]),
    (MonsterId::GremlinThief, Some("Gremlin Gang"), &[(0, Some((10, 14)), &[Some(9)]), (2, Some((10, 14)), &[Some(10)]), (7, Some((11, 15)), &[Some(10)])]),
    (MonsterId::GremlinTsundere, Some("Gremlin Gang"), &[(0, Some((12, 15)), &[Some(6)]), (2, Some((12, 15)), &[Some(8)]), (7, Some((13, 17)), &[Some(8)])]),
    (MonsterId::GremlinWarrior, Some("Gremlin Gang"), &[(0, Some((20, 24)), &[Some(4)]), (2, Some((20, 24)), &[Some(5)]), (7, Some((21, 25)), &[Some(5)])]),
    (MonsterId::GremlinWizard, Some("Gremlin Gang"), &[(0, Some((21, 25)), &[Some(25)]), (2, Some((21, 25)), &[Some(30)]), (7, Some((22, 26)), &[Some(30)])]),
    (MonsterId::Hexaghost, Some("Hexaghost"), &[(0, Some((250, 250)), &[Some(5), Some(6), Some(-1), Some(2)]), (4, Some((250, 250)), &[Some(6), Some(6), Some(-1), Some(3)]), (9, Some((264, 264)), &[Some(6), Some(6), Some(-1), Some(3)])]),
    (MonsterId::JawWorm, Some("Jaw Worm"), &[(0, Some((40, 44)), &[Some(11), Some(7)]), (2, Some((40, 44)), &[Some(12), Some(7)]), (7, Some((42, 46)), &[Some(12), Some(7)])]),
    (MonsterId::Lagavulin, Some("Lagavulin"), &[(0, Some((109, 111)), &[Some(18)]), (3, Some((109, 111)), &[Some(20)]), (8, Some((112, 115)), &[Some(20)])]),
    (MonsterId::Looter, Some("Looter"), &[(0, Some((44, 48)), &[Some(10), Some(12)]), (2, Some((44, 48)), &[Some(11), Some(14)]), (7, Some((46, 50)), &[Some(11), Some(14)])]),
    (MonsterId::LouseDefensive, Some("3 Louse"), &[(0, Some((11, 17)), &[None]), (7, Some((12, 18)), &[None])]),
    (MonsterId::LouseNormal, Some("3 Louse"), &[(0, Some((10, 15)), &[None]), (7, Some((11, 16)), &[None])]),
    (MonsterId::Sentry, Some("3 Sentries"), &[(0, Some((38, 42)), &[Some(9)]), (3, Some((38, 42)), &[Some(10)]), (8, Some((39, 45)), &[Some(10)])]),
    (MonsterId::SlaverBlue, Some("Blue Slaver"), &[(0, Some((46, 50)), &[Some(12), Some(7)]), (2, Some((46, 50)), &[Some(13), Some(8)]), (7, Some((48, 52)), &[Some(13), Some(8)])]),
    (MonsterId::SlaverRed, Some("Red Slaver"), &[(0, Some((46, 50)), &[Some(13), Some(8)]), (2, Some((46, 50)), &[Some(14), Some(9)]), (7, Some((48, 52)), &[Some(14), Some(9)])]),
    (MonsterId::SlimeBoss, Some("Slime Boss"), &[(0, Some((140, 140)), &[Some(9), Some(35)]), (4, Some((140, 140)), &[Some(10), Some(38)]), (9, Some((150, 150)), &[Some(10), Some(38)])]),
    (MonsterId::SpikeSlimeL, Some("Large Slime"), &[(0, Some((64, 70)), &[Some(16)]), (2, Some((64, 70)), &[Some(18)]), (7, Some((67, 73)), &[Some(18)])]),
    (MonsterId::SpikeSlimeM, Some("Small Slimes"), &[(0, Some((28, 32)), &[Some(8)]), (2, Some((28, 32)), &[Some(10)]), (7, Some((29, 34)), &[Some(10)])]),
    (MonsterId::SpikeSlimeS, Some("Lots of Slimes"), &[(0, Some((10, 14)), &[Some(5)]), (2, Some((10, 14)), &[Some(6)]), (7, Some((11, 15)), &[Some(6)])]),
    (MonsterId::TheGuardian, Some("The Guardian"), &[(0, Some((240, 240)), &[Some(32), Some(9), Some(5), Some(8)]), (4, Some((240, 240)), &[Some(36), Some(10), Some(5), Some(8)]), (9, Some((250, 250)), &[Some(36), Some(10), Some(5), Some(8)])]),
    (MonsterId::SphericGuardian, Some("Spheric Guardian"), &[(0, Some((20, 20)), &[Some(10)]), (2, Some((20, 20)), &[Some(11)])]),
    (MonsterId::Chosen, Some("Chosen"), &[(0, Some((95, 99)), &[Some(18), Some(10), Some(5)]), (2, Some((95, 99)), &[Some(21), Some(12), Some(6)]), (7, Some((98, 103)), &[Some(21), Some(12), Some(6)])]),
    (MonsterId::ShelledParasite, Some("Shell Parasite"), &[(0, Some((68, 72)), &[Some(6), Some(18), Some(10)]), (2, Some((68, 72)), &[Some(7), Some(21), Some(12)]), (7, Some((70, 75)), &[Some(7), Some(21), Some(12)])]),
    (MonsterId::Byrd, Some("3 Byrds"), &[(0, Some((25, 31)), &[Some(1), Some(12), Some(3)]), (2, Some((25, 31)), &[Some(1), Some(14), Some(3)]), (7, Some((26, 33)), &[Some(1), Some(14), Some(3)])]),
    (MonsterId::Mugger, Some("2 Thieves"), &[(0, Some((48, 52)), &[Some(10), Some(16)]), (2, Some((48, 52)), &[Some(11), Some(18)]), (7, Some((50, 54)), &[Some(11), Some(18)])]),
    (MonsterId::BanditPointy, Some("Masked Bandits"), &[(0, Some((30, 30)), &[Some(5)]), (2, Some((30, 30)), &[Some(6)]), (7, Some((34, 34)), &[Some(6)])]),
    (MonsterId::BanditBear, Some("Masked Bandits"), &[(0, Some((38, 42)), &[Some(18), Some(9)]), (2, Some((38, 42)), &[Some(20), Some(10)]), (7, Some((40, 44)), &[Some(20), Some(10)])]),
    (MonsterId::BanditLeader, Some("Masked Bandits"), &[(0, Some((35, 39)), &[Some(15), Some(10)]), (2, Some((35, 39)), &[Some(17), Some(12)]), (7, Some((37, 41)), &[Some(17), Some(12)])]),
    (MonsterId::SnakePlant, Some("Snake Plant"), &[(0, Some((75, 79)), &[Some(7)]), (2, Some((75, 79)), &[Some(8)]), (7, Some((78, 82)), &[Some(8)])]),
    (MonsterId::Snecko, Some("Snecko"), &[(0, Some((114, 120)), &[Some(15), Some(8)]), (2, Some((114, 120)), &[Some(18), Some(10)]), (7, Some((120, 125)), &[Some(18), Some(10)])]),
    (MonsterId::Centurion, Some("Centurion and Healer"), &[(0, Some((76, 80)), &[Some(12), Some(6)]), (2, Some((76, 80)), &[Some(14), Some(7)]), (7, Some((78, 83)), &[Some(14), Some(7)])]),
    (MonsterId::Healer, Some("Centurion and Healer"), &[(0, Some((48, 56)), &[Some(8)]), (2, Some((48, 56)), &[Some(9)]), (7, Some((50, 58)), &[Some(9)])]),
    (MonsterId::Taskmaster, Some("Slavers"), &[(0, Some((54, 60)), &[Some(4), Some(7)]), (8, Some((57, 64)), &[Some(4), Some(7)])]),
    (MonsterId::GremlinLeader, Some("Gremlin Leader"), &[(0, Some((140, 148)), &[Some(6)]), (8, Some((145, 155)), &[Some(6)])]),
    (MonsterId::BookOfStabbing, Some("Book of Stabbing"), &[(0, Some((160, 164)), &[Some(6), Some(21)]), (3, Some((160, 164)), &[Some(7), Some(24)]), (8, Some((168, 172)), &[Some(7), Some(24)])]),
    (MonsterId::Champ, Some("Champ"), &[(0, Some((420, 420)), &[Some(16), Some(10), Some(12)]), (4, Some((420, 420)), &[Some(18), Some(10), Some(14)]), (9, Some((440, 440)), &[Some(18), Some(10), Some(14)])]),
    (MonsterId::TheCollector, Some("Collector"), &[(0, Some((282, 282)), &[Some(18)]), (4, Some((282, 282)), &[Some(21)]), (9, Some((300, 300)), &[Some(21)])]),
    (MonsterId::TorchHead, None, &[(0, Some((38, 40)), &[Some(7)]), (9, Some((40, 45)), &[Some(7)])]),
    (MonsterId::BronzeAutomaton, Some("Automaton"), &[(0, Some((300, 300)), &[Some(7), Some(45)]), (4, Some((300, 300)), &[Some(8), Some(50)]), (9, Some((320, 320)), &[Some(8), Some(50)])]),
    (MonsterId::BronzeOrb, None, &[(0, Some((52, 58)), &[Some(8)]), (9, Some((54, 60)), &[Some(8)])]),
    (MonsterId::Darkling, Some("3 Darklings"), &[(0, Some((48, 56)), &[Some(8), None]), (2, Some((48, 56)), &[Some(9), None]), (7, Some((50, 59)), &[Some(9), None])]),
    (MonsterId::OrbWalker, Some("Orb Walker"), &[(0, Some((90, 96)), &[Some(10), Some(15)]), (2, Some((90, 96)), &[Some(11), Some(16)]), (7, Some((92, 102)), &[Some(11), Some(16)])]),
    (MonsterId::Spiker, Some("3 Shapes"), &[(0, Some((42, 56)), &[Some(7)]), (2, Some((42, 56)), &[Some(9)]), (7, Some((44, 60)), &[Some(9)])]),
    (MonsterId::Repulsor, Some("3 Shapes"), &[(0, Some((29, 35)), &[Some(11)]), (2, Some((29, 35)), &[Some(13)]), (7, Some((31, 38)), &[Some(13)])]),
    (MonsterId::Exploder, Some("3 Shapes"), &[(0, Some((30, 30)), &[Some(9)]), (2, Some((30, 30)), &[Some(11)]), (7, Some((30, 35)), &[Some(11)])]),
    (MonsterId::SpireGrowth, Some("Spire Growth"), &[(0, Some((170, 170)), &[Some(16), Some(22)]), (2, Some((170, 170)), &[Some(18), Some(25)]), (7, Some((190, 190)), &[Some(18), Some(25)])]),
    (MonsterId::Transient, Some("Transient"), &[(0, Some((999, 999)), &[Some(30), Some(40), Some(50), Some(60), Some(70), Some(80), Some(90)]), (2, Some((999, 999)), &[Some(40), Some(50), Some(60), Some(70), Some(80), Some(90), Some(100)])]),
    (MonsterId::Maw, Some("Maw"), &[(0, Some((300, 300)), &[Some(25), Some(5)]), (2, Some((300, 300)), &[Some(30), Some(5)])]),
    (MonsterId::WrithingMass, Some("Writhing Mass"), &[(0, Some((160, 160)), &[Some(32), Some(7), Some(15), Some(10)]), (2, Some((160, 160)), &[Some(38), Some(9), Some(16), Some(12)]), (7, Some((175, 175)), &[Some(38), Some(9), Some(16), Some(12)])]),
    (MonsterId::GiantHead, Some("Giant Head"), &[(0, Some((500, 500)), &[Some(13), Some(30), Some(35), Some(40), Some(45), Some(50), Some(55), Some(60)]), (3, Some((500, 500)), &[Some(13), Some(40), Some(45), Some(50), Some(55), Some(60), Some(65), Some(70)]), (8, Some((520, 520)), &[Some(13), Some(40), Some(45), Some(50), Some(55), Some(60), Some(65), Some(70)])]),
    (MonsterId::Nemesis, Some("Nemesis"), &[(0, Some((185, 185)), &[Some(45), Some(6)]), (3, Some((185, 185)), &[Some(45), Some(7)]), (8, Some((200, 200)), &[Some(45), Some(7)])]),
    (MonsterId::Reptomancer, Some("Reptomancer"), &[(0, Some((180, 190)), &[Some(13), Some(30)]), (3, Some((180, 190)), &[Some(16), Some(34)]), (8, Some((190, 200)), &[Some(16), Some(34)])]),
    (MonsterId::SnakeDagger, Some("Reptomancer"), &[(0, Some((20, 25)), &[Some(9), Some(25), Some(25)])]),
    (MonsterId::TimeEater, Some("Time Eater"), &[(0, Some((456, 456)), &[Some(7), Some(26)]), (4, Some((456, 456)), &[Some(8), Some(32)]), (9, Some((480, 480)), &[Some(8), Some(32)])]),
    (MonsterId::Donu, Some("Donu and Deca"), &[(0, Some((250, 250)), &[Some(10)]), (4, Some((250, 250)), &[Some(12)]), (9, Some((265, 265)), &[Some(12)])]),
    (MonsterId::Deca, Some("Donu and Deca"), &[(0, Some((250, 250)), &[Some(10)]), (4, Some((250, 250)), &[Some(12)]), (9, Some((265, 265)), &[Some(12)])]),
    (MonsterId::AwakenedOne, Some("Awakened One"), &[(0, Some((300, 300)), &[Some(20), Some(6), Some(40), Some(18), Some(10)]), (9, Some((320, 320)), &[Some(20), Some(6), Some(40), Some(18), Some(10)])]),
    (MonsterId::SpireShield, Some("Shield and Spear"), &[(0, Some((110, 110)), &[Some(12), Some(34)]), (3, Some((110, 110)), &[Some(14), Some(38)]), (8, Some((125, 125)), &[Some(14), Some(38)])]),
    (MonsterId::SpireSpear, Some("Shield and Spear"), &[(0, Some((160, 160)), &[Some(5), Some(10)]), (3, Some((160, 160)), &[Some(6), Some(10)]), (8, Some((180, 180)), &[Some(6), Some(10)])]),
    (MonsterId::CorruptHeart, Some("The Heart"), &[(0, Some((750, 750)), &[Some(40), Some(2)]), (4, Some((750, 750)), &[Some(45), Some(2)]), (9, Some((800, 800)), &[Some(45), Some(2)])]),
];

/// The monsters of `id` made at `asc` over many seeds (by their encounter, or their constructor
/// for the two that only appear when spawned).
fn made(id: MonsterId, encounter: Option<&str>, asc: i32) -> Vec<Monster> {
    let mut out = Vec::new();
    for seed in 0..400i64 {
        match encounter {
            Some(key) => {
                let group = encounters::create(key, &mut Rngs::new(seed), asc).unwrap();
                out.extend(group.into_iter().filter(|m| m.id == id));
            }
            None => {
                let mut rng = StsRandom::new(seed);
                out.push(match id {
                    MonsterId::TorchHead => sts_core::monsters::city::torch_head(&mut rng, asc, 0.0),
                    _ => sts_core::monsters::city::bronze_orb(&mut rng, asc, 0.0, 0),
                });
            }
        }
    }
    assert!(!out.is_empty(), "{id:?} never made by {encounter:?}");
    out
}

#[test]
fn every_monster_has_the_games_hp_and_damage_at_each_ascension() {
    for &(id, encounter, rows) in STATS {
        for (i, &(asc, hp, damage)) in rows.iter().enumerate() {
            // The row holds from its ascension up to the next row's.
            let until = rows.get(i + 1).map_or(20, |r| r.0 - 1);
            for a in [asc, until] {
                let ms = made(id, encounter, a);
                if let Some((lo, hi)) = hp {
                    let min = ms.iter().map(|m| m.max_hp).min().unwrap();
                    let max = ms.iter().map(|m| m.max_hp).max().unwrap();
                    assert_eq!((min, max), (lo, hi), "{id:?} HP at A{a}");
                }
                for m in &ms {
                    let bases: Vec<i32> = m.damage.iter().map(|d| d.base).collect();
                    assert_eq!(bases.len(), damage.len(), "{id:?} damage at A{a}: {bases:?}");
                    for (j, d) in damage.iter().enumerate() {
                        if let Some(d) = d {
                            assert_eq!(bases[j], *d, "{id:?} damage {j} at A{a}: {bases:?}");
                        }
                    }
                }
            }
        }
    }
}

/// The damage some monsters roll on `monsterHpRng` when they are made.
#[test]
fn rolled_monster_damage() {
    // (monster, encounter, slot, A0 range, A2 range)
    let cases = [
        (MonsterId::LouseNormal, "3 Louse", 0, (5, 7), (6, 8)),
        (MonsterId::LouseDefensive, "3 Louse", 0, (5, 7), (6, 8)),
        (MonsterId::Darkling, "3 Darklings", 1, (7, 11), (9, 13)),
    ];
    for (id, key, slot, a0, a2) in cases {
        for (asc, (lo, hi)) in [(0, a0), (1, a0), (2, a2), (20, a2)] {
            let ms = made(id, Some(key), asc);
            let min = ms.iter().map(|m| m.damage[slot].base).min().unwrap();
            let max = ms.iter().map(|m| m.damage[slot].base).max().unwrap();
            assert_eq!((min, max), (lo, hi), "{id:?} at A{asc}");
        }
    }
}

// ---- Move rules (`getMove`, and the moves `takeTurn` sets) ----

/// One fight where the player only ends turns (100,000 HP, no cards played): each monster's moves,
/// its `nextMove` at the start of every player turn, by its index in the fight.
fn passive_moves(encounter: &str, asc: i32, seed: i64, turns: usize) -> Vec<(MonsterId, Vec<u8>)> {
    let mut cfg = CombatConfig::new(Character::Ironclad, seed, encounter);
    cfg.ascension = asc;
    cfg.deck = vec![(CardId::DefendR, 0); 10];
    cfg.relics.clear();
    cfg.player_hp = 100_000;
    cfg.player_max_hp = 100_000;
    let mut c = Combat::new(&cfg).unwrap();
    let mut log: Vec<(MonsterId, Vec<u8>)> = Vec::new();
    for _ in 0..turns {
        if c.is_over() {
            break;
        }
        for (i, m) in c.monsters.iter().enumerate() {
            if log.len() <= i {
                log.push((m.id, Vec::new()));
            }
            if !m.is_dead_or_escaped() {
                log[i].1.push(m.next_move);
            }
        }
        c.step(&PlayerAction::EndTurn).unwrap();
    }
    log
}

/// The move sequences of every `id` in `encounter` over 60 seeds.
fn sequences(encounter: &str, id: MonsterId, asc: i32, turns: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for seed in 0..60 {
        out.extend(passive_moves(encounter, asc, seed, turns).into_iter().filter(|(m, _)| *m == id).map(|(_, s)| s));
    }
    assert!(!out.is_empty(), "no {id:?} in {encounter}");
    out
}

/// The longest run of `mv` in `seq`.
fn longest_run(seq: &[u8], mv: u8) -> usize {
    let (mut best, mut cur) = (0, 0);
    for &m in seq {
        cur = if m == mv { cur + 1 } else { 0 };
        best = best.max(cur);
    }
    best
}

/// Checks `seqs` never repeat a move more than the given number of times in a row, use only the
/// moves listed, and (when the fights are long enough) use each of them.
fn assert_runs(what: &str, seqs: &[Vec<u8>], max_runs: &[(u8, usize)]) {
    for (mv, max) in max_runs {
        let longest = seqs.iter().map(|s| longest_run(s, *mv)).max().unwrap();
        assert!(longest <= *max, "{what}: move {mv} {longest} times in a row (at most {max})");
        assert!(longest > 0, "{what}: move {mv} never used");
    }
    for s in seqs {
        assert!(s.iter().all(|m| max_runs.iter().any(|(mv, _)| mv == m)), "{what}: unexpected move in {s:?}");
    }
}

/// Every sequence starts with `prefix` and then cycles `cycle`.
fn assert_cycle(what: &str, seqs: &[Vec<u8>], prefix: &[u8], cycle: &[u8]) {
    for s in seqs {
        for (i, &m) in s.iter().enumerate() {
            let want = if i < prefix.len() { prefix[i] } else { cycle[(i - prefix.len()) % cycle.len()] };
            assert_eq!(m, want, "{what}: move {i} of {s:?}");
        }
    }
}

#[test]
fn exordium_move_rules() {
    use MonsterId::*;
    // Slimes. Acid: 1 Corrosive Spit, 2 Tackle, 4 Lick; Spike: 1 Flame Tackle, 4 Lick.
    for asc in [0, 17] {
        let s = sequences("Large Slime", AcidSlimeL, asc, 10);
        let rules: &[(u8, usize)] = if asc >= 17 { &[(1, 2), (2, 2), (4, 1)] } else { &[(1, 2), (2, 1), (4, 2)] };
        assert_runs(&format!("Acid Slime (L) A{asc}"), &s, rules);
        let s = sequences("Small Slimes", AcidSlimeM, asc, 10);
        assert_runs(&format!("Acid Slime (M) A{asc}"), &s, rules);
        let s = sequences("Large Slime", SpikeSlimeL, asc, 10);
        let rules: &[(u8, usize)] = if asc >= 17 { &[(1, 2), (4, 1)] } else { &[(1, 2), (4, 2)] };
        assert_runs(&format!("Spike Slime (L) A{asc}"), &s, rules);
        let s = sequences("Small Slimes", SpikeSlimeM, asc, 10);
        assert_runs(&format!("Spike Slime (M) A{asc}"), &s, rules);
        assert_cycle("Spike Slime (S)", &sequences("Lots of Slimes", SpikeSlimeS, asc, 6), &[], &[1]);
    }
    // Acid Slime (S) alternates Tackle and Lick; from A17 it opens with Lick.
    for s in sequences("Lots of Slimes", AcidSlimeS, 0, 8) {
        let other = if s[0] == 1 { 2 } else { 1 };
        assert_cycle("Acid Slime (S)", &[s.clone()], &[], &[s[0], other]);
    }
    assert_cycle("Acid Slime (S) A17", &sequences("Lots of Slimes", AcidSlimeS, 17, 8), &[], &[2, 1]);
    // Cultist: Incantation, then Dark Strike forever.
    assert_cycle("Cultist", &sequences("Cultist", Cultist, 0, 6), &[3], &[1]);
    // Fungi Beast: Bite (1) at most twice in a row, Grow (2) never twice.
    assert_runs("Fungi Beast", &sequences("2 Fungi Beasts", FungiBeast, 0, 12), &[(1, 2), (2, 1)]);
    // Gremlins: Fat always Smash, Thief and Warrior always attack, Shield Gremlin protects while
    // others live, Wizard charges twice (three times after the first blast) then blasts.
    for asc in [0, 17] {
        assert_cycle("Fat Gremlin", &sequences("Gremlin Gang", GremlinFat, asc, 5), &[], &[2]);
        assert_cycle("Sneaky Gremlin", &sequences("Gremlin Gang", GremlinThief, asc, 5), &[], &[1]);
        assert_cycle("Mad Gremlin", &sequences("Gremlin Gang", GremlinWarrior, asc, 5), &[], &[1]);
        assert_cycle("Shield Gremlin", &sequences("Gremlin Gang", GremlinTsundere, asc, 5), &[], &[1]);
    }
    assert_cycle("Gremlin Wizard", &sequences("Gremlin Gang", GremlinWizard, 0, 11), &[2, 2, 1], &[2, 2, 2, 1]);
    assert_cycle("Gremlin Wizard A17", &sequences("Gremlin Gang", GremlinWizard, 17, 8), &[2, 2], &[1]);
    // Gremlin Nob: Bellow, then Rush (1) at most twice in a row; from A18 Skull Bash (2) every
    // third turn.
    let s = sequences("Gremlin Nob", GremlinNob, 0, 12);
    assert!(s.iter().all(|s| s[0] == 3 && !s[1..].contains(&3)));
    assert_runs("Gremlin Nob", &s.iter().map(|s| s[1..].to_vec()).collect::<Vec<_>>(), &[(1, 2), (2, 12)]);
    assert_cycle("Gremlin Nob A18", &sequences("Gremlin Nob", GremlinNob, 18, 10), &[3], &[2, 1, 1]);
    // Hexaghost: Activate, Divider, then the six orbs' cycle with Inferno last.
    assert_cycle("Hexaghost", &sequences("Hexaghost", Hexaghost, 0, 16), &[5, 1], &[4, 2, 4, 3, 2, 4, 6]);
    // Jaw Worm: Chomp first; then never Chomp or Bellow twice, Thrash at most twice.
    let s = sequences("Jaw Worm", JawWorm, 0, 14);
    assert!(s.iter().all(|s| s[0] == 1));
    assert_runs("Jaw Worm", &s, &[(1, 1), (2, 1), (3, 2)]);
    // Lagavulin: asleep three turns, then Attack, Attack, Siphon Soul.
    assert_cycle("Lagavulin", &sequences("Lagavulin", Lagavulin, 0, 12), &[5, 5, 5], &[3, 3, 1]);
    // Looter: two Mugs, then Smoke Bomb (2) and escape (3), or Lunge (4) first.
    for s in sequences("Looter", Looter, 0, 8) {
        assert!(s == [1, 1, 2, 3] || s == [1, 1, 4, 2, 3], "Looter: {s:?}");
    }
    // Louses: Bite (3) and Grow/Spit Web (4), each at most twice; from A17 4 never twice.
    for (id, asc, rules) in [
        (LouseNormal, 0, &[(3u8, 2usize), (4, 2)][..]),
        (LouseDefensive, 0, &[(3, 2), (4, 2)][..]),
        (LouseNormal, 17, &[(3, 2), (4, 1)][..]),
        (LouseDefensive, 17, &[(3, 2), (4, 1)][..]),
    ] {
        assert_runs(&format!("{id:?} A{asc}"), &sequences("3 Louse", id, asc, 12), rules);
    }
    // Sentries alternate Bolt (3) and Beam (4); the first and third open with Bolt.
    for seed in 0..10 {
        let log = passive_moves("3 Sentries", 0, seed, 8);
        for (i, (_, s)) in log.iter().enumerate() {
            let first = if i % 2 == 0 { 3 } else { 4 };
            assert_cycle("Sentry", &[s.clone()], &[], &[first, 7 - first]);
        }
    }
    // Slavers. Blue: Stab (1) and Rake (4) at most twice in a row (Rake once from A17).
    assert_runs("Blue Slaver", &sequences("Blue Slaver", SlaverBlue, 0, 12), &[(1, 2), (4, 2)]);
    assert_runs("Blue Slaver A17", &sequences("Blue Slaver", SlaverBlue, 17, 12), &[(1, 2), (4, 1)]);
    // Red: Stab first, Entangle (2) at most once, Stab and Scrape (3) at most twice in a row.
    for asc in [0, 17] {
        let s = sequences("Red Slaver", SlaverRed, asc, 12);
        assert!(s.iter().all(|s| s[0] == 1 && s.iter().filter(|&&m| m == 2).count() <= 1));
        let scrape = if asc >= 17 { 1 } else { 2 };
        assert_runs(&format!("Red Slaver A{asc}"), &s, &[(1, 2), (2, 1), (3, scrape)]);
    }
    // Slime Boss: Goop Spray, Preparing, Slam.
    assert_cycle("Slime Boss", &sequences("Slime Boss", SlimeBoss, 0, 9), &[], &[4, 2, 1]);
    // The Guardian, left alone: Charging Up, Fierce Bash, Vent Steam, Whirlwind.
    assert_cycle("The Guardian", &sequences("The Guardian", TheGuardian, 0, 12), &[], &[6, 2, 7, 5]);
}

#[test]
fn city_move_rules() {
    use MonsterId::*;
    // Masked Bandits: the Bear hugs then alternates Lunge and Maul; Romeo mocks, then Agonizing
    // Slash and Cross Slash (twice from A17); Pointy always attacks.
    assert_cycle("Bear", &sequences("Masked Bandits", BanditBear, 0, 9), &[2], &[3, 1]);
    assert_cycle("Romeo", &sequences("Masked Bandits", BanditLeader, 0, 9), &[2], &[3, 1]);
    assert_cycle("Romeo A17", &sequences("Masked Bandits", BanditLeader, 17, 10), &[2], &[3, 1, 1]);
    assert_cycle("Pointy", &sequences("Masked Bandits", BanditPointy, 0, 6), &[], &[1]);
    // Book of Stabbing: Multi-Stab (1) at most twice in a row, Single Stab (2) never twice.
    assert_runs("Book of Stabbing", &sequences("Book of Stabbing", BookOfStabbing, 0, 14), &[(1, 2), (2, 1)]);
    // Bronze Automaton: Spawn Orbs, then Flail and Boost, Hyper Beam every fifth move, then
    // Stunned (Boost from A19).
    assert_cycle("Bronze Automaton", &sequences("Automaton", BronzeAutomaton, 0, 14), &[4], &[1, 5, 1, 5, 2, 3]);
    assert_cycle("Bronze Automaton A19", &sequences("Automaton", BronzeAutomaton, 19, 14), &[4, 1, 5, 1, 5, 2], &[5, 1, 5, 1, 5, 2]);
    // Bronze Orbs: Stasis (3) at most once, Beam (1) and Support Beam (2) at most twice in a row.
    let s = sequences("Automaton", BronzeOrb, 0, 12);
    assert!(s.iter().all(|s| s.iter().filter(|&&m| m == 3).count() <= 1));
    assert_runs("Bronze Orb", &s, &[(1, 2), (2, 2), (3, 1)]);
    // Byrds in flight: Peck (1) at most twice in a row, Swoop (3) and Caw (6) never twice.
    assert_runs("Byrd", &sequences("3 Byrds", Byrd, 0, 12), &[(1, 2), (3, 1), (6, 1)]);
    // Centurion beside the Mystic: Slash (1) and Defend (2) at most twice in a row. The Mystic
    // (nobody hurt): Attack (1) and Buff (3) at most twice, Attack never twice from A17.
    assert_runs("Centurion", &sequences("Centurion and Healer", Centurion, 0, 12), &[(1, 2), (2, 2)]);
    assert_runs("Mystic", &sequences("Centurion and Healer", Healer, 0, 12), &[(1, 2), (3, 2)]);
    assert_runs("Mystic A17", &sequences("Centurion and Healer", Healer, 17, 12), &[(1, 1), (3, 2)]);
    // Champ (never hurt): Taunt (6) every fourth move and only then, no move twice in a row,
    // Defensive Stance (2) at most twice and never followed by Gloat (5).
    for asc in [0, 19] {
        for s in sequences("Champ", Champ, asc, 16) {
            for (i, &m) in s.iter().enumerate() {
                assert_eq!(m == 6, i % 4 == 3, "Champ A{asc}: {s:?}");
                assert!([1, 2, 4, 5, 6].contains(&m), "Champ A{asc}: {s:?}");
            }
            assert!(s.windows(2).all(|w| w[0] != w[1] && !(w[0] == 2 && w[1] == 5)), "Champ A{asc}: {s:?}");
            assert!(s.iter().filter(|&&m| m == 2).count() <= 2, "Champ A{asc}: {s:?}");
        }
    }
    // Chosen: Poke, Hex, then alternately Debilitate (3) or Drain (2), and Zap (1) or Poke (5);
    // from A17 Hex comes first.
    for (asc, prefix) in [(0, &[5u8, 4][..]), (17, &[4][..])] {
        for s in sequences("Chosen", Chosen, asc, 12) {
            assert_eq!(&s[..prefix.len()], prefix, "Chosen A{asc}");
            for (i, &m) in s[prefix.len()..].iter().enumerate() {
                assert!(if i % 2 == 0 { m == 2 || m == 3 } else { m == 1 || m == 5 }, "Chosen A{asc}: {s:?}");
            }
        }
    }
    // Gremlin Leader with both gremlins alive: Encourage (3) and Stab (4) alternate.
    for s in sequences("Gremlin Leader", GremlinLeader, 0, 10) {
        assert!(s[0] == 3 || s[0] == 4);
        assert_cycle("Gremlin Leader", &[s.clone()], &[], &[s[0], 7 - s[0]]);
    }
    // Mugger: two Mugs, then Smoke Bomb (2) and escape (3), or Lunge (4) first.
    for s in sequences("2 Thieves", Mugger, 0, 8) {
        assert!(s == [1, 1, 2, 3] || s == [1, 1, 4, 2, 3], "Mugger: {s:?}");
    }
    // Shelled Parasite: Double Strike (2) or Suck (3) first (Fell, 1, from A17); Fell never twice,
    // the others at most twice in a row.
    for asc in [0, 17] {
        let s = sequences("Shell Parasite", ShelledParasite, asc, 12);
        assert!(s.iter().all(|s| if asc >= 17 { s[0] == 1 } else { s[0] == 2 || s[0] == 3 }));
        assert_runs(&format!("Shelled Parasite A{asc}"), &s, &[(1, 1), (2, 2), (3, 2)]);
    }
    // Snake Plant: Chomp (1) at most twice in a row, Enfeebling Spores (2) never twice; from A17
    // at least two Chomps between Spores.
    assert_runs("Snake Plant", &sequences("Snake Plant", SnakePlant, 0, 12), &[(1, 2), (2, 1)]);
    for s in sequences("Snake Plant", SnakePlant, 17, 12) {
        let spores: Vec<usize> = (0..s.len()).filter(|&i| s[i] == 2).collect();
        assert!(spores.windows(2).all(|w| w[1] - w[0] >= 3), "Snake Plant A17: {s:?}");
    }
    // Snecko: Perplexing Glare first and never again; Bite (2) at most twice in a row.
    let s = sequences("Snecko", Snecko, 0, 12);
    assert!(s.iter().all(|s| s[0] == 1 && !s[1..].contains(&1)));
    assert_runs("Snecko", &s.iter().map(|s| s[1..].to_vec()).collect::<Vec<_>>(), &[(2, 2), (3, 12)]);
    // Spheric Guardian: Activate, Attack/Debuff, then Slam and Harden alternate.
    assert_cycle("Spheric Guardian", &sequences("Spheric Guardian", SphericGuardian, 0, 10), &[2, 4], &[1, 3]);
    assert_cycle("Taskmaster", &sequences("Slavers", Taskmaster, 0, 6), &[], &[2]);
    // The Collector: Spawn, then Mega Debuff on the fourth move only; Fireball (2) at most twice in
    // a row, Buff (3) never twice; Revive (5) needs a dead Torch Head.
    for s in sequences("Collector", TheCollector, 0, 12) {
        assert!(s.iter().enumerate().all(|(i, &m)| (m == 4) == (i == 3) && (m == 1) == (i == 0)), "Collector: {s:?}");
    }
    let rest: Vec<Vec<u8>> = sequences("Collector", TheCollector, 0, 12).iter().map(|s| s[1..].to_vec()).collect();
    assert_runs("Collector", &rest, &[(2, 2), (3, 1), (4, 1)]);
    assert_cycle("Torch Head", &sequences("Collector", TorchHead, 0, 8), &[], &[1]);
}

#[test]
fn beyond_move_rules() {
    use MonsterId::*;
    // Awakened One, first form: Slash first; Soul Strike (2) never twice, Slash (1) at most twice.
    let s = sequences("Awakened One", AwakenedOne, 0, 12);
    assert!(s.iter().all(|s| s[0] == 1));
    assert_runs("Awakened One", &s, &[(1, 2), (2, 1)]);
    // Darklings: Harden (2) or Nip (3) first; Chomp (1) only for the first and third, never twice;
    // Harden never twice; Nip at most twice in a row.
    for seed in 0..60 {
        for (i, (_, s)) in passive_moves("3 Darklings", 0, seed, 12).iter().enumerate() {
            assert!(s[0] == 2 || s[0] == 3, "Darkling: {s:?}");
            assert!(i != 1 || !s.contains(&1), "middle Darkling: {s:?}");
            assert!(longest_run(s, 1) <= 1 && longest_run(s, 2) <= 1 && longest_run(s, 3) <= 2, "Darkling: {s:?}");
        }
    }
    // Donu and Deca take turns: Deca beams first, Donu buffs first.
    assert_cycle("Deca", &sequences("Donu and Deca", Deca, 0, 8), &[], &[0, 2]);
    assert_cycle("Donu", &sequences("Donu and Deca", Donu, 0, 8), &[], &[2, 0]);
    // Exploder: two attacks, then it explodes.
    for s in sequences("3 Shapes", Exploder, 0, 8) {
        assert_eq!(s, [1, 1, 2]);
    }
    // Giant Head: Glare (1) or Count (3) for four moves (three from A18), each at most twice in a
    // row, then It Is Time (2) forever.
    for (asc, n) in [(0, 4), (18, 3)] {
        for s in sequences("Giant Head", GiantHead, asc, 12) {
            assert!(s[..n].iter().all(|&m| m == 1 || m == 3) && s[n..].iter().all(|&m| m == 2), "Giant Head A{asc}: {s:?}");
            assert!(longest_run(&s, 1) <= 2 && longest_run(&s, 3) <= 2, "Giant Head A{asc}: {s:?}");
        }
    }
    // Maw: Roar first; Nom (5) and Slam (3) never twice; Drool (4) always follows Nom.
    for s in sequences("Maw", Maw, 0, 14) {
        assert_eq!(s[0], 2);
        assert!(!s[1..].contains(&2));
        for w in s.windows(2) {
            assert!(!(w[0] == 5 && w[1] != 4) && !(w[0] == 3 && w[1] == 3), "Maw: {s:?}");
        }
    }
    // Nemesis: Debuff (4) or Attack (2) first; Scythe (3) at most every other move, Debuff never
    // twice, Attack at most twice in a row.
    for s in sequences("Nemesis", Nemesis, 0, 14) {
        assert!(s[0] == 2 || s[0] == 4);
        let scythes: Vec<usize> = (0..s.len()).filter(|&i| s[i] == 3).collect();
        assert!(scythes.windows(2).all(|w| w[1] - w[0] >= 2), "Nemesis: {s:?}");
    }
    assert_runs("Nemesis", &sequences("Nemesis", Nemesis, 0, 14), &[(2, 2), (3, 1), (4, 1)]);
    assert_runs("Orb Walker", &sequences("Orb Walker", OrbWalker, 0, 12), &[(1, 2), (2, 2)]);
    // Reptomancer: Summon first; Snake Strike (1) and Big Bite (3) never twice, Summon at most twice.
    let s = sequences("Reptomancer", Reptomancer, 0, 14);
    assert!(s.iter().all(|s| s[0] == 2));
    assert_runs("Reptomancer", &s, &[(1, 1), (2, 2), (3, 1)]);
    // Daggers stab, then explode (a dagger summoned near the end has had one move).
    for s in sequences("Reptomancer", SnakeDagger, 0, 14) {
        assert!([1, 2].starts_with(&s) && !s.is_empty(), "Dagger: {s:?}");
    }
    assert_runs("Repulsor", &sequences("4 Shapes", Repulsor, 0, 12), &[(1, 12), (2, 1)]);
    // Spiker: Cut (1) never twice until it has spiked up six times, then only Cut.
    for s in sequences("4 Shapes", Spiker, 0, 16) {
        let sixth = s.iter().enumerate().filter(|(_, &m)| m == 2).nth(5).map_or(s.len(), |(i, _)| i);
        assert!(s[..sixth].windows(2).all(|w| w != [1, 1]) && s[sixth + 1..].iter().all(|&m| m == 1), "Spiker: {s:?}");
    }
    // Spire Growth: Constrict (2) once (the player keeps Constricted), first from A17; Quick
    // Tackle (1) and Smash (3) at most twice in a row.
    assert_runs("Spire Growth", &sequences("Spire Growth", SpireGrowth, 0, 12), &[(1, 2), (2, 1), (3, 2)]);
    for s in sequences("Spire Growth", SpireGrowth, 17, 12) {
        assert_eq!(s[0], 2);
        assert_eq!(s.iter().filter(|&&m| m == 2).count(), 1);
    }
    // Time Eater (never below half): Reverberate (2) at most twice in a row, Head Slam (4) and
    // Ripple (3) never twice.
    assert_runs("Time Eater", &sequences("Time Eater", TimeEater, 0, 14), &[(2, 2), (3, 1), (4, 1)]);
    assert_cycle("Transient", &sequences("Transient", Transient, 0, 8), &[], &[1]);
    // Writhing Mass: Implant (4) at most once; no move twice in a row, except Big Hit (0) after
    // the Implant (its 10% roll in the Implant branch doesn't check the last move).
    for s in sequences("Writhing Mass", WrithingMass, 0, 16) {
        assert!(s.iter().filter(|&&m| m == 4).count() <= 1, "Writhing Mass: {s:?}");
        for i in 1..s.len() {
            assert!(s[i] != s[i - 1] || (s[i] == 0 && s[..i].contains(&4)), "Writhing Mass: {s:?}");
        }
    }
}

#[test]
fn ending_move_rules() {
    use MonsterId::*;
    // The Heart: Debilitate, then threes of Blood Shots (1) and Echo (2) in either order, and Buff (4).
    for s in sequences("The Heart", CorruptHeart, 0, 13) {
        assert_eq!(s[0], 3);
        for i in 1..s.len() {
            match (i - 1) % 3 {
                0 => assert!(s[i] == 1 || s[i] == 2, "Heart: {s:?}"),
                1 => assert_eq!(s[i], 3 - s[i - 1], "Heart: {s:?}"),
                _ => assert_eq!(s[i], 4, "Heart: {s:?}"),
            }
        }
    }
    // Spire Shield: Bash (1) and Fortify (2) in either order, then Smash (3).
    for s in sequences("Shield and Spear", SpireShield, 0, 12) {
        for i in 0..s.len() {
            match i % 3 {
                0 => assert!(s[i] == 1 || s[i] == 2, "Shield: {s:?}"),
                1 => assert_eq!(s[i], 3 - s[i - 1], "Shield: {s:?}"),
                _ => assert_eq!(s[i], 3, "Shield: {s:?}"),
            }
        }
    }
    // Spire Spear: Burn Strike first; Skewer (3) every third move from the second; the move after
    // a random Burn Strike (1) or Piercer (2) is the other one.
    for s in sequences("Shield and Spear", SpireSpear, 0, 12) {
        assert_eq!(s[0], 1);
        for i in 1..s.len() {
            match i % 3 {
                1 => assert_eq!(s[i], 3, "Spear: {s:?}"),
                2 => assert!(s[i] == 1 || s[i] == 2, "Spear: {s:?}"),
                _ => assert_eq!(s[i], 3 - s[i - 1], "Spear: {s:?}"),
            }
        }
    }
}

// ---- Powers and mechanics ----

/// A 300-HP Ironclad fight against `encounter` at `asc` (the first seed whose group has a `want`),
/// with no relics, exactly `hand` in hand and `draw` in the draw pile (top last).
fn fight_vs(encounter: &str, asc: i32, want: Option<MonsterId>, hand: &[CardId], draw: &[CardId]) -> Combat {
    for seed in 0..200 {
        let mut cfg = CombatConfig::new(Character::Ironclad, seed, encounter);
        cfg.ascension = asc;
        cfg.deck = vec![(CardId::DefendR, 0); 10];
        cfg.relics.clear();
        cfg.player_hp = 300;
        cfg.player_max_hp = 300;
        let mut c = Combat::new(&cfg).unwrap();
        if want.map_or(true, |id| c.monsters.iter().any(|m| m.id == id)) {
            set_piles(&mut c, hand, draw);
            return c;
        }
    }
    panic!("no {want:?} in {encounter}");
}

fn set_piles(c: &mut Combat, hand: &[CardId], draw: &[CardId]) {
    c.hand.clear();
    c.draw_pile.clear();
    c.discard_pile.clear();
    for &id in hand {
        let card = c.new_card(id);
        c.hand.push(card);
    }
    for &id in draw {
        let card = c.new_card(id);
        c.draw_pile.push(card);
    }
}

fn mon(c: &Combat, id: MonsterId) -> usize {
    *c.order.iter().find(|&&m| c.monsters[m].id == id && !c.monsters[m].is_dead_or_escaped()).unwrap_or_else(|| panic!("no {id:?}"))
}

fn play(c: &mut Combat, id: CardId, target: Option<usize>) {
    let i = c.hand.iter().position(|card| card.id == id).unwrap_or_else(|| panic!("{id:?} not in hand"));
    c.step(&PlayerAction::PlayCard { hand_index: i, target }).unwrap();
}

fn end_turn(c: &mut Combat) {
    c.step(&PlayerAction::EndTurn).unwrap();
}

fn power(c: &Combat, who: Cr, id: PowerId) -> i32 {
    c.power_amount(who, id)
}

fn can_play(c: &Combat, hand_index: usize) -> bool {
    c.legal_actions().iter().any(|a| matches!(a, PlayerAction::PlayCard { hand_index: h, .. } if *h == hand_index))
}

/// Applies `power` to `target` through `ApplyPowerAction` (from the first monster), letting the
/// engine run it: the player drinks a Liquid Memories over an empty discard pile, which does
/// nothing else.
fn apply(c: &mut Combat, target: Cr, power: Power) {
    let source = Some(Cr::Monster(c.order[0]));
    c.push_bot(Action::ApplyPower { target, source, amount: power.amount, power });
    let saved = std::mem::take(&mut c.discard_pile);
    c.potions = vec![Some(PotionId::LiquidMemories), None, None];
    c.step(&PlayerAction::UsePotion { slot: 0, target: None }).unwrap();
    let added = std::mem::replace(&mut c.discard_pile, saved);
    c.discard_pile.extend(added);
}

fn cards(c: &Combat, id: CardId) -> usize {
    c.hand.iter().chain(&c.draw_pile).chain(&c.discard_pile).filter(|card| card.id == id).count()
}

#[test]
fn gremlin_nob_enrages_on_skills() {
    // Bellow: Anger 2 (3 from A18); then each Skill gives that much Strength.
    for (asc, anger) in [(0, 2), (18, 3)] {
        let mut c = fight_vs("Gremlin Nob", asc, None, &[], &[CardId::DefendR; 10]);
        let m = c.order[0];
        end_turn(&mut c);
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Anger), anger);
        set_piles(&mut c, &[CardId::DefendR, CardId::StrikeR], &[]);
        play(&mut c, CardId::StrikeR, Some(m));
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), 0);
        play(&mut c, CardId::DefendR, None);
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), anger);
    }
}

#[test]
fn mad_gremlin_gets_angry_when_hit() {
    for (asc, angry) in [(0, 1), (17, 2)] {
        let mut c = fight_vs("Gremlin Gang", asc, Some(MonsterId::GremlinWarrior), &[CardId::StrikeR, CardId::Bash], &[]);
        let m = mon(&c, MonsterId::GremlinWarrior);
        c.monsters[m].hp = 50;
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Angry), angry);
        play(&mut c, CardId::StrikeR, Some(m));
        play(&mut c, CardId::Bash, Some(m));
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), 2 * angry);
    }
}

#[test]
fn shield_and_spear_surround_the_player() {
    let mut c = fight_vs("Shield and Spear", 0, None, &[CardId::StrikeR, CardId::StrikeR], &[]);
    let (shield, spear) = (mon(&c, MonsterId::SpireShield), mon(&c, MonsterId::SpireSpear));
    assert!(c.player.has_power(PowerId::Surrounded));
    // Facing the Spear, the Shield is behind: Back Attack, 50% more damage.
    assert!(c.monsters[shield].has_power(PowerId::BackAttack) && !c.monsters[spear].has_power(PowerId::BackAttack));
    assert_eq!(c.monsters[shield].damage[0].output, 18);
    // Striking the Shield turns the player round.
    play(&mut c, CardId::StrikeR, Some(shield));
    assert!(c.monsters[spear].has_power(PowerId::BackAttack) && !c.monsters[shield].has_power(PowerId::BackAttack));
    assert_eq!(c.monsters[spear].damage[0].output, 7);
    // With one of them dead, the player is no longer surrounded.
    c.monsters[shield].hp = 1;
    play(&mut c, CardId::StrikeR, Some(shield));
    assert!(!c.player.has_power(PowerId::Surrounded) && !c.monsters[spear].has_power(PowerId::BackAttack));
}

#[test]
fn the_heart_beats_and_caps_damage() {
    for (asc, beat, cap) in [(0, 1, 300), (19, 2, 200)] {
        let mut c = fight_vs("The Heart", asc, None, &[CardId::StrikeR, CardId::StrikeR], &[CardId::DefendR; 10]);
        let m = c.order[0];
        assert_eq!((power(&c, Cr::Monster(m), PowerId::Invincible), power(&c, Cr::Monster(m), PowerId::BeatOfDeath)), (cap, beat));
        c.player.powers.push(Power::new(PowerId::Strength, 1000));
        let (hp, php) = (c.monsters[m].hp, c.player.hp);
        play(&mut c, CardId::StrikeR, Some(m));
        assert_eq!((c.monsters[m].hp, c.player.hp), (hp - cap, php - beat), "A{asc}");
        play(&mut c, CardId::StrikeR, Some(m));
        assert_eq!((c.monsters[m].hp, c.player.hp), (hp - cap, php - 2 * beat), "A{asc}");
        end_turn(&mut c);
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Invincible), cap);
    }
}

#[test]
fn player_side_powers_from_monsters() {
    let defends = [CardId::DefendR; 10];
    // Buffer: the next HP loss is prevented (Jaw Worm's Chomp).
    let mut c = fight_vs("Jaw Worm", 0, None, &[], &defends);
    apply(&mut c, Cr::Player, Power::new(PowerId::Buffer, 1));
    end_turn(&mut c);
    assert_eq!(c.player.hp, 300);
    assert!(!c.player.has_power(PowerId::Buffer));
    // Confusion: a drawn card's cost becomes cardRandomRng 0-3.
    let mut c = fight_vs("Jaw Worm", 0, None, &[CardId::ShrugItOff], &[CardId::Bash]);
    c.player.powers.push(Power::flag(PowerId::Confusion));
    let cost = c.rng.card_random.clone().random(3);
    play(&mut c, CardId::ShrugItOff, None);
    assert_eq!((c.hand[0].id, c.hand[0].cost, c.hand[0].cost_for_turn), (CardId::Bash, cost, cost));
    // Draw Reduction: one card fewer. It goes at the end of the second round, but the next turn's
    // DrawCardAction takes `gameHandSize` before the queued removal runs, so two draws are short.
    let mut c = fight_vs("Jaw Worm", 0, None, &[], &defends);
    c.player.hp = 10_000;
    apply(&mut c, Cr::Player, Power::draw_reduction(1));
    let mut hands = Vec::new();
    for _ in 0..3 {
        end_turn(&mut c);
        hands.push(c.hand.len());
    }
    assert_eq!(hands, [4, 4, 5]);
    // Entangled: no Attacks until the turn ends.
    let mut c = fight_vs("Jaw Worm", 0, None, &[CardId::StrikeR], &[CardId::StrikeR; 10]);
    apply(&mut c, Cr::Player, Power::new(PowerId::Entangled, 1));
    assert!(!can_play(&c, 0));
    end_turn(&mut c);
    assert!(can_play(&c, 0));
    // Hex: each non-Attack shuffles a Dazed into the draw pile.
    let mut c = fight_vs("Jaw Worm", 0, None, &[CardId::StrikeR, CardId::DefendR], &defends);
    apply(&mut c, Cr::Player, Power::new(PowerId::Hex, 1));
    let m = c.order[0];
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(cards(&c, CardId::Dazed), 0);
    play(&mut c, CardId::DefendR, None);
    assert_eq!(cards(&c, CardId::Dazed), 1);
    // Next Turn Block.
    let mut c = fight_vs("Jaw Worm", 0, None, &[], &defends);
    apply(&mut c, Cr::Player, Power::new(PowerId::NextTurnBlock, 3));
    end_turn(&mut c);
    assert_eq!(c.player.block, 3);
}

#[test]
fn spire_growth_constricts() {
    // From A17 Constrict comes first: 12 damage at the end of every player turn.
    let mut c = fight_vs("Spire Growth", 17, None, &[], &[CardId::DefendR; 10]);
    let m = c.order[0];
    end_turn(&mut c);
    assert_eq!(power(&c, Cr::Player, PowerId::Constricted), 12);
    let hp = c.player.hp;
    let attack = c.monsters[m].intent_dmg;
    assert!(c.monsters[m].intent.is_attack() && attack > 0);
    end_turn(&mut c);
    assert_eq!(c.player.hp, hp - 12 - attack);
}

#[test]
fn awakened_one_curiosity_regenerates_and_awakens() {
    // Curiosity: Strength for each Power played (1, 2 from A19); A4 starts with 2 Strength.
    for (asc, base, cur, regen) in [(0, 0, 1, 10), (4, 2, 1, 10), (19, 2, 2, 15)] {
        let mut c = fight_vs("Awakened One", asc, None, &[CardId::Inflame], &[CardId::DefendR; 10]);
        let m = mon(&c, MonsterId::AwakenedOne);
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), base);
        play(&mut c, CardId::Inflame, None);
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), base + cur, "A{asc}");
        // Regenerate at the end of its turn.
        c.monsters[m].hp -= 50;
        let hp = c.monsters[m].hp;
        end_turn(&mut c);
        assert_eq!(c.monsters[m].hp, hp + regen, "A{asc}");
    }
    // Unawakened: dying in the first form rebirths it at full HP, without Curiosity.
    let mut c = fight_vs("Awakened One", 0, None, &[CardId::StrikeR], &[CardId::DefendR; 10]);
    let m = mon(&c, MonsterId::AwakenedOne);
    c.monsters[m].hp = 1;
    play(&mut c, CardId::StrikeR, Some(m));
    assert!(c.monsters[m].half_dead && c.outcome.is_none());
    assert_eq!(c.monsters[m].next_move, 3);
    assert!(!c.monsters[m].has_power(PowerId::Curiosity) && !c.monsters[m].has_power(PowerId::Unawakened));
    end_turn(&mut c);
    assert!(!c.monsters[m].half_dead);
    assert_eq!(c.monsters[m].hp, 300);
    assert_eq!(c.monsters[m].next_move, 5);
}

#[test]
fn louse_curls_up_once() {
    let mut c = fight_vs("3 Louse", 0, None, &[CardId::StrikeR, CardId::StrikeR], &[]);
    let m = c.order[0];
    let curl = power(&c, Cr::Monster(m), PowerId::CurlUp);
    assert!((3..=7).contains(&curl), "{curl}");
    c.monsters[m].hp = 50;
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!((c.monsters[m].hp, c.monsters[m].block), (44, curl));
    assert!(!c.monsters[m].has_power(PowerId::CurlUp));
}

#[test]
fn shapes_explode_and_spike() {
    // Exploder: Explosive 3 counts down; at 1 it explodes for 30 and dies.
    let mut c = fight_vs("3 Shapes", 0, Some(MonsterId::Exploder), &[], &[CardId::DefendR; 10]);
    let m = mon(&c, MonsterId::Exploder);
    let mut counts = Vec::new();
    for _ in 0..3 {
        counts.push(power(&c, Cr::Monster(m), PowerId::Explosive));
        end_turn(&mut c);
    }
    assert_eq!(counts, [3, 2, 1]);
    assert!(c.monsters[m].is_dead_or_escaped());
}

#[test]
fn transient_fades_and_shifts() {
    // Fading: gone after 5 turns (6 from A17).
    for (asc, turns) in [(0, 5), (17, 6)] {
        let mut c = fight_vs("Transient", asc, None, &[], &[CardId::DefendR; 10]);
        c.player.hp = 10_000;
        for t in 1..=turns {
            assert!(c.outcome.is_none(), "A{asc} turn {t}");
            end_turn(&mut c);
        }
        assert!(c.monsters[c.order[0]].is_dead_or_escaped(), "A{asc}");
    }
    // Shifting: damage taken is lost as Strength until the end of the turn.
    let mut c = fight_vs("Transient", 0, None, &[CardId::StrikeR], &[CardId::DefendR; 10]);
    let m = c.order[0];
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!((power(&c, Cr::Monster(m), PowerId::Strength), power(&c, Cr::Monster(m), PowerId::Shackled)), (-6, 6));
    end_turn(&mut c);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), 0);
}

#[test]
fn byrds_fly_until_hit_enough() {
    for (asc, flight) in [(0, 3), (17, 4)] {
        let mut c = fight_vs("3 Byrds", asc, None, &[CardId::StrikeR; 4], &[]);
        c.energy = 10;
        let m = c.order[0];
        c.monsters[m].hp = 100;
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Flight), flight);
        // Half damage from attacks; each hit takes one Flight; at none it is grounded and stunned.
        for i in 1..=flight {
            play(&mut c, CardId::StrikeR, Some(m));
            assert_eq!(c.monsters[m].hp, 100 - 3 * i, "A{asc}");
        }
        assert!(!c.monsters[m].has_power(PowerId::Flight));
        assert_eq!(c.monsters[m].next_move, 4);
    }
}

#[test]
fn monster_buff_powers() {
    // Orb Walker: 3 Strength (5 from A17) at the end of every round.
    for (asc, up) in [(0, 3), (17, 5)] {
        let mut c = fight_vs("Orb Walker", asc, None, &[], &[CardId::DefendR; 10]);
        let m = c.order[0];
        end_turn(&mut c);
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), up);
    }
    // Intangible: at most 1 damage per hit.
    let mut c = fight_vs("Nemesis", 0, None, &[CardId::StrikeR], &[]);
    let m = c.order[0];
    apply(&mut c, Cr::Monster(m), Power::intangible(1));
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 1);
    // Snake Plant's Malleable: 3 Block when hit, one more each hit, back to 3 after its turn.
    let mut c = fight_vs("Snake Plant", 0, None, &[CardId::StrikeR, CardId::StrikeR], &[CardId::DefendR; 10]);
    let m = c.order[0];
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!((c.monsters[m].hp, c.monsters[m].block), (hp - 6, 3));
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!((c.monsters[m].hp, c.monsters[m].block), (hp - 9, 4));
    end_turn(&mut c);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Malleable), 3);
    // Writhing Mass: Malleable too, and Reactive rerolls its intent when hit.
    let mut c = fight_vs("Writhing Mass", 0, None, &[CardId::StrikeR], &[]);
    let m = c.order[0];
    let ai = c.rng.ai.counter;
    play(&mut c, CardId::StrikeR, Some(m));
    assert!(c.rng.ai.counter > ai);
    assert_eq!(c.monsters[m].block, 3);
    // Giant Head's Slow: 10% more damage taken per card played this round.
    let mut c = fight_vs("Giant Head", 0, None, &[CardId::DefendR, CardId::DefendR, CardId::StrikeR], &[CardId::DefendR; 10]);
    let m = c.order[0];
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::DefendR, None);
    play(&mut c, CardId::DefendR, None);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Slow), 2);
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 7);
    end_turn(&mut c);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Slow), 0);
}

#[test]
fn the_guardian_shifts_modes() {
    let mut c = fight_vs("The Guardian", 0, None, &[CardId::Bludgeon], &[CardId::StrikeR; 10]);
    let m = c.order[0];
    assert_eq!(power(&c, Cr::Monster(m), PowerId::ModeShift), 30);
    play(&mut c, CardId::Bludgeon, Some(m));
    // 30 damage taken: Defensive Mode, 20 Block, Close Up next.
    assert!(!c.monsters[m].has_power(PowerId::ModeShift));
    assert_eq!((c.monsters[m].block, c.monsters[m].next_move), (20, 1));
    end_turn(&mut c);
    // Sharp Hide: 3 damage to the player for each Attack played.
    assert_eq!(power(&c, Cr::Monster(m), PowerId::SharpHide), 3);
    let hp = c.player.hp;
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.player.hp, hp - 3);
    // Roll Attack, Twin Slam, then back to Offensive Mode with a threshold 10 higher.
    end_turn(&mut c);
    end_turn(&mut c);
    assert!(!c.monsters[m].has_power(PowerId::SharpHide));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::ModeShift), 40);
}

#[test]
fn darklings_regrow_until_all_are_down() {
    let mut c = fight_vs("3 Darklings", 0, None, &[CardId::StrikeR], &[CardId::DefendR; 10]);
    let m = c.order[0];
    c.monsters[m].hp = 1;
    play(&mut c, CardId::StrikeR, Some(m));
    assert!(c.monsters[m].half_dead);
    // Its view says so (the viewer keeps it on the field, as the game does), not alive.
    let v = c.observe().monsters.into_iter().find(|v| v.id == m).unwrap();
    assert!(v.half_dead && !v.alive);
    assert_eq!(c.monsters[m].next_move, 4);
    end_turn(&mut c);
    assert_eq!(c.monsters[m].next_move, 5);
    end_turn(&mut c);
    assert!(!c.monsters[m].half_dead && c.monsters[m].has_power(PowerId::Regrow));
    assert_eq!(c.monsters[m].hp, c.monsters[m].max_hp / 2);
    // All three down at once: the fight is won.
    let mut c = fight_vs("3 Darklings", 0, None, &[CardId::StrikeR; 3], &[]);
    for i in 0..3 {
        let m = c.order[i];
        c.monsters[m].hp = 1;
        play(&mut c, CardId::StrikeR, Some(m));
    }
    assert_eq!(c.outcome, Some(sts_core::combat::Outcome::Victory));
}

#[test]
fn minions_leave_with_their_leader() {
    let mut c = fight_vs("Gremlin Leader", 0, None, &[CardId::StrikeR], &[]);
    let l = mon(&c, MonsterId::GremlinLeader);
    for &m in &c.order {
        assert_eq!(m != l, c.monsters[m].has_power(PowerId::Minion));
    }
    c.monsters[l].hp = 1;
    play(&mut c, CardId::StrikeR, Some(l));
    assert!(c.order.iter().all(|&m| c.monsters[m].is_dead_or_escaped()));
    // Reptomancer's daggers die with it.
    let mut c = fight_vs("Reptomancer", 0, None, &[CardId::StrikeR], &[]);
    let r = mon(&c, MonsterId::Reptomancer);
    c.monsters[r].hp = 1;
    play(&mut c, CardId::StrikeR, Some(r));
    assert!(c.order.iter().all(|&m| c.monsters[m].is_dead_or_escaped()));
    assert_eq!(c.outcome, Some(sts_core::combat::Outcome::Victory));
}

#[test]
fn book_of_stabbing_wounds_with_every_hit() {
    let mut c = fight_vs("Book of Stabbing", 0, None, &[], &[CardId::DefendR; 10]);
    let m = c.order[0];
    assert!(c.monsters[m].has_power(PowerId::PainfulStabs));
    let hits = if c.monsters[m].next_move == 1 { c.monsters[m].intent_multi.max(1) } else { 1 };
    end_turn(&mut c);
    assert_eq!(cards(&c, CardId::Wound) as i32, hits);
}

#[test]
fn slimes_split_at_half_health() {
    for (id, child) in [(MonsterId::AcidSlimeL, MonsterId::AcidSlimeM), (MonsterId::SpikeSlimeL, MonsterId::SpikeSlimeM), (MonsterId::SlimeBoss, MonsterId::SpikeSlimeL)] {
        let key = if id == MonsterId::SlimeBoss { "Slime Boss" } else { "Large Slime" };
        let mut c = fight_vs(key, 0, Some(id), &[CardId::StrikeR], &[CardId::DefendR; 10]);
        let m = mon(&c, id);
        assert!(c.monsters[m].has_power(PowerId::Split));
        c.monsters[m].hp = c.monsters[m].max_hp / 2 + 3;
        play(&mut c, CardId::StrikeR, Some(m));
        let left = c.monsters[m].hp;
        assert_eq!(c.monsters[m].next_move, 3, "{id:?}");
        end_turn(&mut c);
        assert!(c.monsters[m].is_dead_or_escaped());
        let kids: Vec<&Monster> = c.monsters.iter().filter(|x| !x.is_dead_or_escaped()).collect();
        assert_eq!(kids.len(), 2, "{id:?}");
        assert!(kids.iter().all(|k| k.max_hp == left && k.hp == left), "{id:?}");
        assert!(kids.iter().any(|k| k.id == child), "{id:?}");
    }
}

#[test]
fn fungi_beasts_leave_spores() {
    let mut c = fight_vs("2 Fungi Beasts", 0, None, &[CardId::StrikeR, CardId::StrikeR], &[]);
    let (a, b) = (c.order[0], c.order[1]);
    assert_eq!(power(&c, Cr::Monster(a), PowerId::SporeCloud), 2);
    c.monsters[a].hp = 1;
    play(&mut c, CardId::StrikeR, Some(a));
    assert_eq!(power(&c, Cr::Player, PowerId::Vulnerable), 2);
    // Not when the fight ends with it.
    c.monsters[b].hp = 1;
    play(&mut c, CardId::StrikeR, Some(b));
    assert_eq!(power(&c, Cr::Player, PowerId::Vulnerable), 2);
}

#[test]
fn bronze_orbs_take_a_card_into_stasis() {
    let mut c = fight_vs("Automaton", 0, None, &[], &[CardId::DefendR; 10]);
    let mut orb = None;
    for _ in 0..6 {
        end_turn(&mut c);
        orb = c.order.iter().copied().find(|&m| c.monsters[m].has_power(PowerId::Stasis));
        if orb.is_some() {
            break;
        }
    }
    let orb = orb.expect("an orb used Stasis");
    assert_eq!(cards(&c, CardId::DefendR), 9);
    let card = c.new_card(CardId::StrikeR);
    c.hand.push(card);
    c.monsters[orb].hp = 1;
    play(&mut c, CardId::StrikeR, Some(orb));
    assert_eq!(cards(&c, CardId::DefendR), 10);
}

#[test]
fn thieves_steal_gold() {
    for (asc, steal) in [(0, 15), (17, 20)] {
        let mut c = fight_vs("Looter", asc, None, &[], &[CardId::DefendR; 10]);
        let gold = c.player.gold;
        assert_eq!(power(&c, Cr::Monster(c.order[0]), PowerId::Thievery), steal);
        end_turn(&mut c);
        assert_eq!((c.player.gold, c.stolen_gold), (gold - steal, steal), "A{asc}");
    }
}

#[test]
fn time_eater_ends_the_turn_after_twelve_cards() {
    let mut c = fight_vs("Time Eater", 0, None, &[CardId::StrikeR, CardId::StrikeR], &[CardId::DefendR; 10]);
    let m = c.order[0];
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::TimeWarp), 1);
    // Ten more cards later: the twelfth.
    c.monsters[m].powers.iter_mut().find(|p| p.id == PowerId::TimeWarp).unwrap().amount = 11;
    let turn = c.turn;
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.turn, turn + 1);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), 2);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::TimeWarp), 0);
}

#[test]
fn lagavulin_wakes_when_hurt() {
    let mut c = fight_vs("Lagavulin", 0, None, &[CardId::StrikeR], &[CardId::DefendR; 10]);
    let m = c.order[0];
    assert_eq!((c.monsters[m].block, power(&c, Cr::Monster(m), PowerId::Metallicize)), (8, 8));
    c.monsters[m].block = 0;
    play(&mut c, CardId::StrikeR, Some(m));
    // Stunned this turn, then out: Metallicize gone, attacking.
    assert_eq!(c.monsters[m].next_move, 4);
    end_turn(&mut c);
    assert!(!c.monsters[m].has_power(PowerId::Metallicize));
    assert_eq!(c.monsters[m].next_move, 3);
}

#[test]
fn hexaghost_divider_scales_with_player_hp() {
    let mut c = fight_vs("Hexaghost", 0, None, &[], &[CardId::DefendR; 10]);
    let m = c.order[0];
    end_turn(&mut c);
    // Activate sets Divider: 6 hits of player HP / 12 + 1.
    assert_eq!(c.monsters[m].next_move, 1);
    assert_eq!((c.monsters[m].intent_base_dmg, c.monsters[m].intent_multi), (300 / 12 + 1, 6));
}

#[test]
fn growing_attacks() {
    // Giant Head's It Is Time: 30, then 5 more each time, up to 60 (40 to 70 from A3).
    for (asc, first) in [(0, 30), (3, 40)] {
        let mut c = fight_vs("Giant Head", asc, None, &[], &[CardId::DefendR; 10]);
        c.player.hp = 100_000;
        let m = c.order[0];
        let mut seen = Vec::new();
        for _ in 0..13 {
            if c.monsters[m].next_move == 2 {
                seen.push(c.monsters[m].intent_base_dmg);
            }
            end_turn(&mut c);
        }
        let want: Vec<i32> = (0..seen.len() as i32).map(|i| (first + 5 * i).min(first + 30)).collect();
        assert_eq!(seen, want, "A{asc}");
    }
    // Transient: 30, 40, 50... (40 from A2).
    let mut c = fight_vs("Transient", 0, None, &[], &[CardId::DefendR; 10]);
    let m = c.order[0];
    let mut seen = Vec::new();
    for _ in 0..4 {
        seen.push(c.monsters[m].intent_base_dmg);
        end_turn(&mut c);
    }
    assert_eq!(seen, [30, 40, 50, 60]);
    // Book of Stabbing: one more stab each Multi-Stab (`stabCount` starts at 1, the first is 2).
    let mut c = fight_vs("Book of Stabbing", 0, None, &[], &[CardId::DefendR; 10]);
    let m = c.order[0];
    let mut stabs = Vec::new();
    for _ in 0..10 {
        if c.monsters[m].next_move == 1 {
            stabs.push(c.monsters[m].intent_multi.max(1));
        }
        end_turn(&mut c);
    }
    assert!(stabs.windows(2).all(|w| w[1] == w[0] + 1) && stabs[0] == 2, "{stabs:?}");
}

#[test]
fn monsters_set_up_at_the_start() {
    // (encounter, monster, power, amount) from usePreBattleAction.
    use MonsterId::*;
    let cases = [
        ("Spheric Guardian", SphericGuardian, PowerId::Artifact, 3),
        ("Spheric Guardian", SphericGuardian, PowerId::Barricade, -1),
        ("Shell Parasite", ShelledParasite, PowerId::PlatedArmor, 14),
        ("Automaton", BronzeAutomaton, PowerId::Artifact, 3),
        ("3 Sentries", Sentry, PowerId::Artifact, 1),
        ("Donu and Deca", Donu, PowerId::Artifact, 2),
        ("Donu and Deca", Deca, PowerId::Artifact, 2),
        ("4 Shapes", Spiker, PowerId::Thorns, 3),
    ];
    for (key, id, p, amount) in cases {
        let c = fight_vs(key, 0, Some(id), &[], &[]);
        let m = mon(&c, id);
        assert_eq!(c.monsters[m].power(p).map(|x| x.amount), Some(amount), "{id:?} {p:?}");
    }
    let c = fight_vs("Spheric Guardian", 0, None, &[], &[]);
    assert_eq!(c.monsters[c.order[0]].block, 40);
    let c = fight_vs("Shell Parasite", 0, None, &[], &[]);
    assert_eq!(c.monsters[c.order[0]].block, 14);
}

#[test]
fn shield_gremlin_attacks_once_alone() {
    // Protect while another monster lives; alone it Bashes (6, 8 from A2) every turn.
    let mut c = fight_vs("Gremlin Gang", 0, Some(MonsterId::GremlinTsundere), &[], &[CardId::DefendR; 10]);
    let g = mon(&c, MonsterId::GremlinTsundere);
    for &m in &c.order.clone() {
        if m != g {
            c.monsters[m].hp = 1;
            let card = c.new_card(CardId::StrikeR);
            c.hand.push(card);
            c.energy = 3;
            play(&mut c, CardId::StrikeR, Some(m));
        }
    }
    assert_eq!(c.monsters[g].next_move, 1);
    end_turn(&mut c);
    assert_eq!((c.monsters[g].next_move, c.monsters[g].intent_base_dmg), (2, 6));
    let hp = c.player.hp;
    end_turn(&mut c);
    assert_eq!(c.player.hp, hp - 6);
    assert_eq!(c.monsters[g].next_move, 2);
}

#[test]
fn awakened_one_second_form_moves() {
    // Reborn: Dark Echo (40) first, then Sludge (6: 18 and a Void in the draw pile) and Tackle
    // (8: 10 x 3), each at most twice in a row.
    for seed in 0..20 {
        let mut c = fight_vs("Awakened One", 0, None, &[CardId::StrikeR], &[CardId::DefendR; 10]);
        c.player.hp = 100_000;
        c.player.max_hp = 100_000;
        let m = mon(&c, MonsterId::AwakenedOne);
        c.rng.ai = StsRandom::new(seed);
        c.monsters[m].hp = 1;
        play(&mut c, CardId::StrikeR, Some(m));
        end_turn(&mut c);
        let mut seq = Vec::new();
        for _ in 0..12 {
            seq.push(c.monsters[m].next_move);
            let voids = c.draw_pile.iter().chain(&c.hand).chain(&c.discard_pile).chain(&c.exhaust_pile).filter(|x| x.id == CardId::Void).count();
            let sludge = c.monsters[m].next_move == 6;
            end_turn(&mut c);
            let now = c.draw_pile.iter().chain(&c.hand).chain(&c.discard_pile).chain(&c.exhaust_pile).filter(|x| x.id == CardId::Void).count();
            if sludge {
                assert!(now > voids, "Sludge adds a Void");
            }
        }
        assert_eq!(seq[0], 5, "{seq:?}");
        assert!(seq[1..].iter().all(|&x| x == 6 || x == 8), "{seq:?}");
        assert!(longest_run(&seq, 6) <= 2 && longest_run(&seq, 8) <= 2, "{seq:?}");
    }
}
