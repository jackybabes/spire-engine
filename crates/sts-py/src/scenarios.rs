//! Training only: boss fights set up from human runs (`python/human/human_bosses.py`): the deck, relics,
//! potions and HP a winning player brought to a boss, so the network practises the fights that end
//! its runs with decks known to win them. A scenario is a run placed on its boss floor, in the
//! fight; at A20 the Act 3 scenario is both bosses back to back (the reward screen between them
//! included), and the episode ends when its last boss does. An Act 4 one (the Heart, or Shield and
//! Spear) is a run with the final act and the three keys. With `play_on` (evaluation: can a network
//! win with this deck? `python/analysis/play_on.py`) the fight only starts a normal run: it goes on
//! through the acts, with no keys it didn't take itself. Late starts in training (`Envs.set_late_starts`) are the same runs.

use serde_json::Value;
use sts_core::cards::CardId;
use sts_core::encounters;
use sts_core::potions::PotionId;
use sts_core::relics::{RelicId, RelicState};
use sts_core::run::map::RoomType;
use sts_core::run::{Run, RunConfig, Screen};

#[derive(Clone, Debug)]
pub struct Scenario {
    /// Whose run it is (human_bosses.py's `character`; the Ironclad's if missing).
    pub character: sts_core::character::Character,
    pub ascension: i32,
    pub act: i32,
    /// A boss fight (the default) or an elite fight (`"room": "elite"`).
    pub room: RoomType,
    pub floor: i32,
    pub hp: i32,
    pub max_hp: i32,
    pub bosses: Vec<String>,
    pub deck: Vec<(CardId, i32)>,
    pub relics: Vec<RelicId>,
    pub potions: Vec<PotionId>,
    /// The seed to play it on, if the line gives one (`"seed"`), instead of its game's: the same
    /// fight with a card removed, played in another game (`python/analysis/fights_by_card.py`).
    pub seed: Option<i64>,
    /// Only for `play_on`: the gold brought and the shop removals made so far (`human_bosses.py
    /// --all-outcomes`; older lines don't say).
    pub gold: Option<i32>,
    pub purges: i32,
    /// Plays on as a normal run after its fight (`Envs.set_scenarios`); not set by the line.
    pub play_on: bool,
}

impl Scenario {
    /// A scenario from human_bosses.py's JSON, or why it can't be played here.
    pub fn parse(line: &str) -> Result<Scenario, String> {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("bad scenario: {e}"))?;
        let int = |k: &str| {
            v[k].as_i64()
                .map(|x| x as i32)
                .ok_or_else(|| format!("no {k}"))
        };
        let strs = |k: &str| -> Vec<String> {
            v[k].as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        };
        let (ascension, act, floor, hp, max_hp) = (
            int("ascension")?,
            int("act")?,
            int("floor")?,
            int("hp")?,
            int("max_hp")?,
        );
        let (bosses, relic_ids, potion_ids) = (strs("bosses"), strs("relics"), strs("potions"));
        let deck = v["deck"]
            .as_array()
            .ok_or("no deck")?
            .iter()
            .map(|c| {
                let id = c[0].as_str().unwrap_or("");
                CardId::from_java_id(id)
                    .map(|card| (card, c[1].as_i64().unwrap_or(0) as i32))
                    .ok_or_else(|| format!("unknown card {id}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut relics = relic_ids
            .iter()
            .map(|id| RelicId::from_java_id(id).ok_or_else(|| format!("unknown relic {id}")))
            .collect::<Result<Vec<_>, _>>()?;
        // A boss relic that replaces the starter takes its slot. The scenarios list both (written
        // before 2026-10-27): Holy Water beside Pure Water started fights with 4 Miracles, not 3;
        // Frozen Core beside Cracked Core channelled an extra Lightning (5% of the Defect's).
        for new in relics.clone() {
            if let Some(old) = new.replaces_starter() {
                if let Some(i) = relics.iter().position(|&r| r == old) {
                    relics.retain(|&r| r != new);
                    relics[i] = new;
                }
            }
        }
        let potions = potion_ids
            .iter()
            .filter_map(|id| PotionId::from_java_id(id))
            .collect();
        let room = match v["room"].as_str() {
            Some("elite") => RoomType::Elite,
            _ => RoomType::Boss,
        };
        let bosses_of = match (room, act) {
            (RoomType::Boss, 1) => &encounters::ACT1_BOSSES[..],
            (RoomType::Boss, 2) => &encounters::ACT2_BOSSES[..],
            (RoomType::Boss, 3) => &encounters::ACT3_BOSSES[..],
            (RoomType::Boss, 4) => &["The Heart"][..],
            (_, 1) => &encounters::ACT1_ELITES[..],
            (_, 2) => &encounters::ACT2_ELITES[..],
            (_, 3) => &encounters::ACT3_ELITES[..],
            (_, 4) => &["Shield and Spear"][..],
            (_, a) => return Err(format!("no fights for act {a}")),
        };
        if bosses.is_empty() || bosses.iter().any(|b| !bosses_of.contains(&b.as_str())) || (room == RoomType::Elite && bosses.len() > 1) {
            return Err(format!("unknown fight in {bosses:?}"));
        }
        if hp <= 0 || max_hp <= 0 {
            return Err("no HP".into());
        }
        let character = match v["character"].as_str() {
            Some(c) => sts_core::character::Character::from_name(c).ok_or_else(|| format!("unknown character {c}"))?,
            None => sts_core::character::Character::Ironclad,
        };
        Ok(Scenario {
            character,
            ascension,
            act,
            room,
            floor,
            hp: hp.min(max_hp),
            max_hp,
            bosses,
            deck,
            relics,
            potions,
            seed: v["seed"].as_i64(),
            gold: v["gold"].as_i64().map(|x| x as i32),
            purges: v["purges"].as_i64().unwrap_or(0) as i32,
            play_on: false,
        })
    }

    /// The floor whose end ends the scenario: the last boss's.
    pub fn end_floor(&self) -> i32 {
        self.floor + self.bosses.len() as i32 - 1
    }

    /// What kind of fight it is, for the stats: "boss1".."boss4" or "elite1".."elite4".
    pub fn kind(&self) -> &'static str {
        match (self.room, self.act) {
            (RoomType::Boss, 1) => "boss1",
            (RoomType::Boss, 2) => "boss2",
            (RoomType::Boss, 3) => "boss3",
            (RoomType::Boss, _) => "boss4",
            (_, 1) => "elite1",
            (_, 2) => "elite2",
            (_, 3) => "elite3",
            _ => "elite4",
        }
    }

    /// The run on the boss floor, in the first boss fight. Relics go in without their pickup
    /// effects (the deck already has what they did); potion slots follow the ascension and Potion
    /// Belt.
    pub fn build(&self, seed: i64) -> Run {
        let mut run = self.place(seed);
        run.start_fight(&self.bosses[0], self.room);
        run
    }

    /// `build` for a run that plays on: what only matters past the fight comes from the line too
    /// (gold, the removal price), held relics aren't offered again (`Pools::new` takes out the
    /// starter only), and the final act is as in the runs it's compared with. Bottled cards and
    /// relic counters still start afresh, and the events seen so far aren't remembered.
    pub fn build_play_on(&self, seed: i64, final_act: bool) -> Run {
        let mut run = self.place(seed);
        run.config.final_act |= final_act;
        run.gold = self.gold.unwrap_or(run.gold);
        run.purge_cost += 25 * self.purges;
        let pools = &mut run.pools;
        for pool in [&mut pools.common_relics, &mut pools.uncommon_relics, &mut pools.rare_relics, &mut pools.shop_relics, &mut pools.boss_relics] {
            pool.retain(|r| !self.relics.contains(r));
        }
        run.start_fight(&self.bosses[0], self.room);
        run
    }

    /// The run on the boss floor, before its fight.
    fn place(&self, seed: i64) -> Run {
        let mut cfg = RunConfig::new(seed);
        cfg.ascension = self.ascension;
        // Act 4 is reached with the final act on and the three keys.
        cfg.final_act = self.act == 4;
        cfg.character = self.character;
        let mut run = Run::new(cfg);
        run.act = self.act;
        run.floor = self.floor;
        if self.act == 4 {
            run.keys.ruby = true;
            run.keys.emerald = true;
            run.keys.sapphire = true;
        }
        run.deck.clear();
        for &(id, up) in &self.deck {
            let mut c = run.new_card(id);
            for _ in 0..up {
                if c.can_upgrade() {
                    c.upgrade();
                }
            }
            run.deck.push(c);
        }
        run.relics = self
            .relics
            .iter()
            .map(|&id| {
                let mut r = RelicState::new(id);
                if id == RelicId::NeowsLament {
                    r.counter = -2; // used up long before a boss
                    r.used_up = true;
                }
                r
            })
            .collect();
        let slots = if self.ascension >= 11 { 2 } else { 3 }
            + if self.relics.contains(&RelicId::PotionBelt) {
                2
            } else {
                0
            };
        run.potions = (0..slots).map(|i| self.potions.get(i).copied()).collect();
        run.max_hp = self.max_hp;
        run.hp = self.hp;
        run.gold = 100;
        if self.room == RoomType::Boss {
            run.boss = self.bosses[0].clone();
        } else if self.act == 4 {
            run.boss = "The Heart".to_string();
        }
        // A second boss (A20, Act 3) is the next boss room's (`after_beyond_boss`): entering a
        // boss room takes the first name off the list, and two left means another boss.
        run.boss_list = if self.bosses.len() > 1 {
            vec![self.bosses[1].clone(), self.bosses[1].clone()]
        } else {
            Vec::new()
        };
        run
    }
}

/// Whether a scenario's episode is over: out of the fight with its last boss behind it, or the
/// run over.
pub fn over(run: &Run, end_floor: i32) -> bool {
    run.is_over() || (!matches!(run.screen, Screen::Combat) && run.floor >= end_floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An A20 Act 3 scenario plays both bosses: after the first, the reward screen, then the
    /// second boss on the next floor; the episode is over only after that. An Act 2 scenario is
    /// over when its boss is.
    #[test]
    fn a20_act3_scenario_has_two_bosses() {
        let line = r#"{"ascension": 20, "act": 3, "floor": 50, "hp": 400, "max_hp": 400, "bosses": ["Time Eater", "Donu and Deca"],
            "deck": [["Strike_R", 1], ["Strike_R", 0], ["Defend_R", 0], ["Bash", 1], ["Demon Form", 1], ["Whirlwind", 1]],
            "relics": ["Burning Blood", "Potion Belt", "NeowsBlessing"], "potions": ["Fire Potion"]}"#;
        let s = Scenario::parse(line).unwrap();
        let mut run = s.build(7);
        // Neow's Lament (used up by build) back on, for the fight already started and the next:
        // monsters at 1 HP, so random play wins both.
        let lament = run
            .relics
            .iter_mut()
            .find(|r| r.id == RelicId::NeowsLament)
            .unwrap();
        lament.counter = 1;
        lament.used_up = false;
        let lament = lament.clone();
        let c = run.combat.as_mut().unwrap();
        for m in c.monsters.iter_mut() {
            m.hp = 1;
        }
        *c.player
            .relics
            .iter_mut()
            .find(|r| r.id == RelicId::NeowsLament)
            .unwrap() = lament;
        assert!(matches!(run.screen, Screen::Combat));
        assert_eq!(run.potions.len(), 4);
        assert_eq!(run.deck.iter().filter(|c| c.times_upgraded > 0).count(), 4);
        let mut rng = sts_core::rng::StsRandom::new(1);
        let mut seen_floor_51_fight = false;
        for _ in 0..20000 {
            if over(&run, s.end_floor()) {
                break;
            }
            if matches!(run.screen, Screen::Combat) && run.floor == 51 {
                seen_floor_51_fight = true;
            }
            let legal = run.legal_actions();
            let a = legal[rng.random(legal.len() as i32 - 1) as usize].clone();
            run.step(&a).unwrap();
        }
        assert!(over(&run, s.end_floor()));
        assert!(seen_floor_51_fight, "never reached the second boss");
        // The second boss ends Act 3, and with no final act, the run: won, in the victory room
        // after it (floor 52), as in the game.
        assert_eq!(
            (run.floor, run.outcome),
            (52, Some(sts_core::run::RunOutcome::Won))
        );
    }

    #[test]
    fn act2_scenario_ends_with_its_boss() {
        let line = r#"{"ascension": 20, "act": 2, "floor": 33, "hp": 60, "max_hp": 80, "bosses": ["Champ"],
            "deck": [["Strike_R", 0], ["Defend_R", 0], ["Bash", 0]], "relics": ["Burning Blood"], "potions": ["Fire Potion", "Block Potion", "Swift Potion"]}"#;
        let s = Scenario::parse(line).unwrap();
        let mut run = s.build(3);
        assert_eq!(
            run.potions,
            vec![Some(PotionId::FirePotion), Some(PotionId::BlockPotion)]
        );
        assert!(!over(&run, s.end_floor()));
        run.combat
            .as_mut()
            .unwrap()
            .monsters
            .iter_mut()
            .for_each(|m| m.hp = 1);
        let mut rng = sts_core::rng::StsRandom::new(1);
        while !over(&run, s.end_floor()) {
            let legal = run.legal_actions();
            run.step(&legal[rng.random(legal.len() as i32 - 1) as usize].clone())
                .unwrap();
        }
        assert!(!run.is_over() && run.floor == 33);
    }

    /// A scenario that plays on is a normal run from its boss on: Act 1's boss beaten, the chest,
    /// Act 2 and its boss, Act 3 and both its bosses, and a win on floor 52 with no keys given (with
    /// all three, on to the Heart). What matters past the fight is the line's; `build` is as ever.
    #[test]
    fn a_run_placed_on_a_boss_plays_on_to_the_end() {
        let line = r#"{"character": "THE_SILENT", "ascension": 20, "act": 1, "floor": 16, "hp": 60, "max_hp": 70, "bosses": ["Hexaghost"], "gold": 321, "purges": 2,
            "deck": [["Strike_G", 0], ["Defend_G", 0], ["Neutralize", 1], ["Survivor", 0]], "relics": ["Ring of the Snake", "Vajra", "Runic Pyramid", "NeowsBlessing"], "potions": []}"#;
        let s = Scenario::parse(line).unwrap();
        let plain = s.build(5);
        assert_eq!((plain.gold, plain.purge_cost, plain.config.final_act), (100, 75, false));
        assert!(plain.pools.common_relics.contains(&RelicId::Vajra) && plain.pools.boss_relics.contains(&RelicId::RunicPyramid));
        for all_keys in [false, true] {
            let mut run = s.build_play_on(5, true);
            assert_eq!((run.gold, run.purge_cost, run.config.final_act, run.keys.all()), (321, 125, true, false));
            assert!(matches!(run.screen, Screen::Combat));
            assert_eq!(run.combat.as_ref().unwrap().player.gold, 321);
            let pools = &run.pools;
            assert!(![&pools.common_relics, &pools.uncommon_relics, &pools.rare_relics, &pools.shop_relics, &pools.boss_relics]
                .iter()
                .any(|pool| pool.contains(&RelicId::Vajra) || pool.contains(&RelicId::RunicPyramid)));
            if all_keys {
                run.keys.ruby = true;
                run.keys.emerald = true;
                run.keys.sapphire = true;
            }
            // Neow's Lament back on for every fight (monsters at 1 HP) and the player kept alive, so random play goes through.
            for r in run.relics.iter_mut().chain(run.combat.as_mut().unwrap().player.relics.iter_mut()) {
                if r.id == RelicId::NeowsLament {
                    r.counter = 1000;
                    r.used_up = false;
                }
            }
            run.combat.as_mut().unwrap().monsters.iter_mut().for_each(|m| m.hp = 1);
            let mut rng = sts_core::rng::StsRandom::new(1);
            let (mut acts, mut boss_floors) = (vec![run.act], Vec::new());
            for _ in 0..200_000 {
                if run.is_over() {
                    break;
                }
                if acts.last() != Some(&run.act) {
                    acts.push(run.act);
                }
                if run.room == Some(RoomType::Boss) && matches!(run.screen, Screen::Combat) && boss_floors.last() != Some(&run.floor) {
                    boss_floors.push(run.floor);
                }
                if let Some(c) = run.combat.as_mut() {
                    c.player.hp = c.player.max_hp;
                }
                run.hp = run.max_hp;
                // (No key by chance: with all three the run goes on to Act 4.)
                let legal: Vec<_> = run.legal_actions().into_iter().filter(|a| !matches!(a, sts_core::run::RunAction::Recall)).collect();
                run.step(&legal[rng.random(legal.len() as i32 - 1) as usize].clone()).unwrap();
            }
            assert_eq!(run.outcome, Some(sts_core::run::RunOutcome::Won), "{acts:?} {boss_floors:?}");
            if all_keys {
                assert_eq!((acts, boss_floors.last()), (vec![1, 2, 3, 4], Some(&run.floor)), "the Heart's floor is the last");
            } else {
                assert_eq!((acts, boss_floors, run.floor), (vec![1, 2, 3], vec![16, 33, 50, 51], 52));
            }
        }
    }

    /// An Act 4 scenario is the Heart (or Shield and Spear) with the final act and the keys, and
    /// its episode ends with that fight.
    #[test]
    fn act_4_scenarios_are_the_heart_with_the_keys() {
        let line = r#"{"character": "DEFECT", "ascension": 20, "act": 4, "floor": 55, "hp": 70, "max_hp": 75, "bosses": ["The Heart"],
                       "deck": [["Strike_B", 0], ["Defend_B", 1], ["Zap", 1], ["Dualcast", 0], ["Glacier", 1], ["Defragment", 1]],
                       "relics": ["Cracked Core", "Vajra"], "potions": ["Regen Potion"]}"#;
        let sc = Scenario::parse(line).unwrap();
        assert_eq!((sc.kind(), sc.end_floor()), ("boss4", 55));
        let run = sc.build(9);
        assert!(matches!(run.screen, Screen::Combat), "{:?}", run.screen);
        assert!(run.config.final_act && run.keys.ruby && run.keys.emerald && run.keys.sapphire);
        assert_eq!((run.act, run.floor, run.hp, run.boss.as_str()), (4, 55, 70, "The Heart"));
        let c = run.combat.as_ref().unwrap();
        assert!(c.monsters.iter().any(|m| format!("{:?}", m.id).contains("Heart")), "{:?}", c.monsters.iter().map(|m| m.id).collect::<Vec<_>>());
        let elite = Scenario::parse(&line.replace(r#""bosses": ["The Heart"]"#, r#""bosses": ["Shield and Spear"], "room": "elite""#).replace("55", "54")).unwrap();
        assert_eq!(elite.kind(), "elite4");
        assert!(matches!(elite.build(9).screen, Screen::Combat));
        assert!(Scenario::parse(&line.replace("The Heart", "Hexaghost")).is_err(), "only Act 4's fights in Act 4");
    }

    /// A boss relic that replaces the starter takes its slot, as in the game: the human runs list
    /// both (Holy Water beside Pure Water: 4 Miracles at the start, the game's 3).
    #[test]
    fn a_replacing_boss_relic_takes_the_starters_slot() {
        let line = r#"{"character": "WATCHER", "ascension": 20, "act": 2, "floor": 33, "hp": 60, "max_hp": 70, "bosses": ["Champ"],
            "deck": [["Strike_P", 0], ["Defend_P", 0], ["Eruption", 0], ["Vigilance", 0]],
            "relics": ["PureWater", "Anchor", "HolyWater"], "potions": []}"#;
        let s = Scenario::parse(line).unwrap();
        assert_eq!(s.relics, [RelicId::HolyWater, RelicId::Anchor]);
        let run = s.build(1);
        let c = run.combat.as_ref().unwrap();
        assert_eq!(c.hand.iter().filter(|x| x.id == CardId::Miracle).count(), 3);
    }

    /// A check of a whole scenario file, not run by default:
    ///
    ///     STS_SCENARIOS=<plain .jsonl> cargo test --release -p sts-py scenario_file -- --ignored --nocapture
    ///
    /// The scenarios it can't play here and why, the potions it drops as unknown, and how the
    /// fights it builds start: stance, Mantra, the Miracles in hand (the Watcher's, 2026-10-27).
    #[test]
    #[ignore]
    fn scenario_file() {
        use std::collections::BTreeMap;
        let path = std::env::var("STS_SCENARIOS").expect("STS_SCENARIOS: a plain .jsonl of scenarios");
        let text = std::fs::read_to_string(&path).unwrap();
        let (mut n, mut ok, mut wrong_deck) = (0, 0, 0);
        let (mut rejected, mut potions, mut kinds) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        let (mut stances, mut mantra, mut miracles) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            n += 1;
            let sc = match Scenario::parse(line) {
                Ok(sc) => sc,
                Err(e) => {
                    *rejected.entry(e).or_insert(0) += 1;
                    continue;
                }
            };
            ok += 1;
            let v: Value = serde_json::from_str(line).unwrap();
            for p in v["potions"].as_array().into_iter().flatten().filter_map(|p| p.as_str()) {
                if PotionId::from_java_id(p).is_none() {
                    *potions.entry(p.to_string()).or_insert(0) += 1;
                }
            }
            *kinds.entry(sc.kind()).or_insert(0) += 1;
            let run = sc.build(n as i64);
            wrong_deck += (run.deck.len() != sc.deck.len()) as i32;
            let c = run.combat.as_ref().expect("in the fight");
            *stances.entry(format!("{:?}", c.player.stance)).or_insert(0) += 1;
            *mantra.entry(c.player.power(sts_core::powers::PowerId::Mantra).map_or(0, |p| p.amount)).or_insert(0) += 1;
            *miracles.entry(c.hand.iter().filter(|x| x.id == CardId::Miracle).count()).or_insert(0) += 1;
        }
        println!("{path}: {n} scenarios, {ok} playable; decks built short: {wrong_deck}");
        println!("by kind: {kinds:?}");
        println!("rejected: {rejected:?}");
        println!("potions dropped as unknown: {potions:?}");
        println!("starting stance: {stances:?}; Mantra: {mantra:?}; Miracles in hand: {miracles:?}");
    }
}
