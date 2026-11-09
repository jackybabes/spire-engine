//! Training only: practice fights, a run set up partway through an act (deck, relics, HP and
//! potions to match) and started in one of that act's fights; the episode ends when the fight
//! does. The network plays it with the same inputs as any fight in a run.
//!
//! Two builders:
//! - `build`, the Ironclad's archetype decks (block, exhaust, strength): the run network only took
//!   cards it already fought well with, so it never held such a deck and never learned to fight
//!   with one;
//! - `build_deck`, any character's fights for a network learning to fight from scratch (the
//!   Defect's defect-fights-0, `notes/defect-training-plan.md`): a grown starting deck, an engine deck or a
//!   random deck (`DeckKind`), sized by act from human runs (`ACT_COUNTS`), in Acts 1-4.

use sts_core::cards::{CardId, CardType, RED_LIBRARY_ORDER};
use sts_core::character::Character;
use sts_core::encounters;
use sts_core::relics::{RelicId, RelicTier};
use sts_core::rng::StsRandom;
use sts_core::run::map::RoomType;
use sts_core::run::pools::{random_potion, Pools};
use sts_core::run::{Run, RunConfig, Screen};
use serde_json::json;

pub struct Archetype {
    pub name: &'static str,
    /// What the deck is built around (two to four of these).
    pub core: &'static [CardId],
    /// Cards that work with the core.
    pub support: &'static [CardId],
}

use CardId::*;

pub const ARCHETYPES: &[Archetype] = &[
    Archetype {
        name: "block",
        core: &[Barricade, Entrench, BodySlam, Juggernaut],
        support: &[ShrugItOff, FlameBarrier, Impervious, Metallicize, GhostlyArmor, PowerThrough, IronWave, TrueGrit, Sentinel, SecondWind, Armaments, Disarm],
    },
    Archetype {
        name: "exhaust",
        core: &[Corruption, DarkEmbrace, FeelNoPain, Evolve],
        support: &[SecondWind, BurningPact, TrueGrit, FiendFire, Sentinel, Exhume, SeverSoul, Offering, PowerThrough, ShrugItOff, GhostlyArmor, Impervious,
                   FireBreathing, WildStrike, RecklessCharge, Immolate],
    },
    Archetype {
        name: "strength",
        core: &[LimitBreak, DemonForm, Inflame, SpotWeakness],
        support: &[HeavyBlade, TwinStrike, SwordBoomerang, Pummel, Whirlwind, Flex, Reaper, Uppercut, Disarm, ShrugItOff, Bludgeon, Shockwave],
    },
    // Losing HP on purpose (jack, 2026-11-01: "Some training for self damage decks would be good"): Rupture turns each card's
    // HP loss into Strength; Combust, Brutality, Bloodletting, Offering and Hemokinesis pay it; Reaper and Feed win it back.
    Archetype {
        name: "self-damage",
        core: &[Rupture, Combust, Brutality, Bloodletting],
        support: &[Offering, Hemokinesis, Reaper, Feed, Inflame, HeavyBlade, SwordBoomerang, TwinStrike, Pummel, ShrugItOff, FlameBarrier, Impervious],
    },
];

/// The Silent's engines, from the decks of her 5,590 A20 runs that beat the Heart (of 234,664 A20 runs in the 2020 dump;
/// `python/human/winner_decks.py`, 2026-11-05), replacing the wiki's four (2026-10-20). Pairs held together beyond what decks of
/// the same sizes would hold by chance: **shiv** (Accuracy with Blade Dance 4.1, Storm of Steel 3.9, Finisher 3.7, Infinite
/// Blades 3.5, Cloak and Dagger 2.6), **discard** (Eviscerate with Sneaky Strike 3.2 and Reflex 1.9, Prepared with Reflex 1.9,
/// Concentrate with Expertise 2.1 and Reflex 1.7, Tools of the Trade with Sneaky Strike 2.0), **hits** (Envenom with Riddle With
/// Holes 2.4, Finisher 1.8 and Accuracy 2.1; Choke with Finisher 2.1 and Riddle With Holes 2.0), **big_turn** (Phantasmal Killer
/// with Terror 2.2, Eviscerate 2.1, Outmaneuver 2.0, Doppelganger 1.8, Skewer 1.8). **poison** pairs little (its cards are
/// common in every kind of winning deck: Catalyst 51%, Noxious Fumes 55%, Crippling Poison 42%, Bouncing Flask 37%, Corpse
/// Explosion 33%, each 2.5-3.6 times its share of A20 decks) and is held apart from the Shivs (Accuracy with Catalyst 0.44,
/// Deadly Poison 0.45, Bouncing Flask 0.45). **dex**: the cards in most winning decks of every kind (Backflip 75%, Acrobatics
/// 66%, Footwork 62%, Well-Laid Plans 54%, Piercing Wail 54%; Wraith Form 6.1 and After Image 4.2 times their share), so
/// support for every engine too. Blade Dance and Cloak and Dagger alone are no more common in winning decks than in all (0.99,
/// 1.29): Shivs win only with Accuracy. Seeded runs and the synergy reward count them (`archetypes_for`); `build_deck`'s engine
/// decks build them.
pub const SILENT_ARCHETYPES: &[Archetype] = &[
    Archetype {
        name: "poison",
        core: &[Catalyst, NoxiousFumes, CorpseExplosion, BouncingFlask],
        support: &[DeadlyPoison, CripplingPoison, PoisonedStab, Burst, Malaise, WellLaidPlans, Footwork, Backflip, PiercingWail, LegSweep, Blur,
                   EscapePlan, Acrobatics, AfterImage, WraithForm],
    },
    Archetype {
        name: "shiv",
        core: &[Accuracy, BladeDance, CloakAndDagger, InfiniteBlades],
        support: &[StormOfSteel, Finisher, AThousandCuts, AfterImage, Envenom, Footwork, Backflip, Acrobatics, PiercingWail, LegSweep, Blur,
                   WellLaidPlans],
    },
    Archetype {
        name: "discard",
        core: &[Eviscerate, Reflex, Tactician, ToolsOfTheTrade],
        support: &[Prepared, Acrobatics, CalculatedGamble, Concentrate, SneakyStrike, Expertise, Backflip, WellLaidPlans, Footwork, Adrenaline,
                   PiercingWail, GrandFinale],
    },
    Archetype {
        name: "hits",
        core: &[Envenom, Choke, RiddleWithHoles, Finisher],
        support: &[BladeDance, Accuracy, HeelHook, AThousandCuts, CloakAndDagger, Footwork, Backflip, Acrobatics, PiercingWail, AfterImage, LegSweep,
                   Blur],
    },
    Archetype {
        name: "big_turn",
        core: &[PhantasmalKiller, Terror, Doppelganger, Skewer],
        support: &[Burst, Adrenaline, Concentrate, Outmaneuver, Expertise, Eviscerate, Footwork, Backflip, Acrobatics, PiercingWail, CalculatedGamble,
                   WellLaidPlans],
    },
    Archetype {
        name: "dex",
        core: &[Footwork, WraithForm, AfterImage, LegSweep],
        support: &[Backflip, Acrobatics, PiercingWail, Malaise, Blur, EscapePlan, DodgeAndRoll, Deflect, Caltrops, WellLaidPlans, Adrenaline,
                   CalculatedGamble],
    },
];

/// The Silent's generally good cards (`build_deck`'s engine decks add a couple): in most of her A20 Heart winners' decks, or
/// taken most often when offered (Adrenaline 76%, Wraith Form 67%, After Image 66%, Footwork 65%, Catalyst 64%).
const SILENT_GOOD: &[CardId] = &[Footwork, Backflip, Acrobatics, WellLaidPlans, PiercingWail, Adrenaline, WraithForm, AfterImage, CalculatedGamble,
                                 Catalyst, NoxiousFumes];

/// The Defect's engines, after the wiki's Defect strategy (slay-the-spire.fandom.com/wiki/Defect):
/// Focus stacking, the Frost wall, Lightning, Dark orbs, zero-cost cards and powers. Not exclusive,
/// and grown into rather than forced; `build_deck`'s engine decks and the per-boss deck stats use
/// them (seeded runs stay off).
pub const DEFECT_ARCHETYPES: &[Archetype] = &[
    Archetype {
        name: "focus",
        core: &[Defragment, BiasedCognition, Capacitor, Consume],
        support: &[Loop, Coolheaded, Glacier, ColdSnap, BallLightning, Electrodynamics, Seek, Skim, CompileDriver, Fission, EchoForm],
    },
    Archetype {
        name: "frost",
        core: &[Glacier, Coolheaded, ColdSnap, Defragment],
        support: &[Capacitor, Loop, Barrage, Blizzard, Chill, ReinforcedBody, BootSequence, ConserveBattery, GoForTheEyes, Leap],
    },
    Archetype {
        name: "lightning",
        core: &[BallLightning, Electrodynamics, Storm, StaticDischarge],
        support: &[ThunderStrike, Tempest, Consume, LockOn, SelfRepair, Aggregate, Turbo, DoubleEnergy, Capacitor],
    },
    Archetype {
        name: "dark",
        core: &[Darkness, DoomAndGloom, Recursion, MultiCast],
        support: &[Aggregate, Turbo, DoubleEnergy, Glacier, Coolheaded, Loop, Capacitor, Consume],
    },
    Archetype {
        name: "zero_cost",
        core: &[Claw, AllForOne, Scrape, FTL],
        support: &[BeamCell, GoForTheEyes, Streamline, Rebound, Hologram, WhiteNoise, Reboot, Seek, Claw],
    },
    Archetype {
        name: "powers",
        core: &[Storm, CreativeAI, Heatsinks, EchoForm],
        support: &[ForceField, Amplify, Recycle, Buffer, SelfRepair, StaticDischarge, Defragment, Capacitor, Loop],
    },
];

/// Generally good Defect cards, in any deck: the ones human players take most often when offered
/// (the 2020 dump, 1.5M Defect runs, 2026-10-24).
const DEFECT_GOOD: &[CardId] = &[Defragment, Electrodynamics, EchoForm, Capacitor, SelfRepair, Loop, WhiteNoise, Glacier, CreativeAI, Buffer];

/// The Watcher's engines, from the decks of the 5,832 A20 Watcher runs that beat the Heart in the
/// 2020 dump (2026-10-27, against all 88,646 A20 Watcher runs; `python/human/winner_decks.py`).
/// Stances: Rushdown, Mental Fortress, Tantrum and Inner Peace are in 44-64% of those decks, about
/// three times their share of A20 decks, in every kind of winning deck. The others are pairs held
/// together beyond what decks of the same sizes would hold by chance: Mantra (Pray with Prostrate
/// 3.05 times, Brilliance with Pray 2.73, with Prostrate 2.48, Devotion with Brilliance 2.0),
/// Scry (Just Lucky with Weave 2.77, Nirvana with Weave 2.56), Retain (Establishment with Battle
/// Hymn 1.95, Windmill Strike 1.73 and Perseverance 1.66; Swivel with Windmill Strike 1.85) and
/// Master Reality's made cards (with Study 2.74, Pray 2.28, Battle Hymn 2.09, Evaluate 1.97).
pub const WATCHER_ARCHETYPES: &[Archetype] = &[
    Archetype {
        name: "stance",
        core: &[Rushdown, MentalFortress, Tantrum, InnerPeace],
        support: &[FearNoEvil, FlurryOfBlows, EmptyMind, EmptyFist, EmptyBody, Crescendo, Tranquility, Indignation, SimmeringFury, CutThroughFate,
                   TalkToTheHand, Wallop, Vault, LikeWater],
    },
    Archetype {
        name: "mantra",
        core: &[Prostrate, Pray, Devotion, Brilliance],
        support: &[Worship, MasterReality, Study, Protect, Blasphemy, Sanctity, Wallop, TalkToTheHand, DevaForm, InnerPeace, Meditate, Halt],
    },
    Archetype {
        name: "scry",
        core: &[Weave, Nirvana, JustLucky, Foresight],
        support: &[ThirdEye, CutThroughFate, Consecrate, TalkToTheHand, Wallop, Protect, FearNoEvil, InnerPeace, EmptyBody],
    },
    Archetype {
        name: "retain",
        core: &[Establishment, WindmillStrike, SandsOfTime, Perseverance],
        support: &[Swivel, Meditate, BattleHymn, Protect, SpiritShield, Alpha, FlyingSleeves, DeceiveReality, TalkToTheHand],
    },
    Archetype {
        name: "master_reality",
        core: &[MasterReality, Study, Evaluate, BattleHymn],
        support: &[Pray, ConjureBlade, DeceiveReality, ForeignInfluence, ReachHeaven, Establishment, Collect, Prostrate, Brilliance],
    },
];

/// Generally good Watcher cards: the ones the A20 Heart winners above hold most and take most
/// often when offered (Talk to the Hand and Vault 79% of the time, Scrawl 70%, Rushdown 64%,
/// Omniscience 62%; Fasting in 41% of their decks, four times its share of A20 decks).
const WATCHER_GOOD: &[CardId] = &[TalkToTheHand, Vault, Scrawl, Omniscience, Rushdown, Tantrum, LessonLearned, Wallop, FearNoEvil, Fasting];

/// A character's archetypes (the Ironclad's practice ones, the other characters' engines).
pub fn archetypes_for(c: Character) -> &'static [Archetype] {
    match c {
        Character::Ironclad => ARCHETYPES,
        Character::Silent => SILENT_ARCHETYPES,
        Character::Defect => DEFECT_ARCHETYPES,
        Character::Watcher => WATCHER_ARCHETYPES,
    }
}

/// A character's generally good cards (`build_deck`'s engine decks add a couple).
fn good_cards(c: Character) -> &'static [CardId] {
    match c {
        Character::Silent => SILENT_GOOD,
        Character::Defect => DEFECT_GOOD,
        Character::Watcher => WATCHER_GOOD,
        Character::Ironclad => &[],
    }
}

/// What a `build_deck` practice fight's deck is (jack, 2026-10-24: a third each for defect-fights-0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeckKind {
    /// The starting deck plus random cards of the character's, as many as a run has by the act.
    Grown,
    /// The starting deck less some Strikes and Defends, a few of one engine's core cards, its
    /// support and a couple of generally good cards.
    Engine,
    /// Random cards of the character's, as many as a deck has by the act.
    Random,
}

impl DeckKind {
    pub const ALL: [DeckKind; 3] = [DeckKind::Grown, DeckKind::Engine, DeckKind::Random];

    pub fn name(self) -> &'static str {
        match self {
            DeckKind::Grown => "grown",
            DeckKind::Engine => "engine",
            DeckKind::Random => "random",
        }
    }

    pub fn from_name(s: &str) -> Option<DeckKind> {
        DeckKind::ALL.into_iter().find(|k| k.name() == s)
    }
}

/// What a run holds at an act's fights, as (low, high) ranges: the 25th to 75th percentile of
/// human A20 Defect winners' card rewards after hallway, elite and boss fights (the 2020 dump,
/// 2026-10-24). Relics count the starting one; basics are the starting Strikes and Defends left.
/// Potions are a guess.
struct ActCounts {
    deck: (i32, i32),
    upgraded: (i32, i32),
    relics: (i32, i32),
    basics: (i32, i32),
    potions: (i32, i32),
    /// Boss relics among the relics (one after each of the Act 1 and 2 bosses).
    boss_relics: i32,
}

const ACT_COUNTS: [ActCounts; 4] = [
    ActCounts { deck: (11, 19), upgraded: (0, 4), relics: (1, 6), basics: (5, 8), potions: (0, 2), boss_relics: 0 },
    ActCounts { deck: (18, 26), upgraded: (3, 9), relics: (6, 13), basics: (2, 7), potions: (0, 2), boss_relics: 1 },
    ActCounts { deck: (22, 30), upgraded: (6, 12), relics: (12, 18), basics: (1, 6), potions: (1, 3), boss_relics: 2 },
    ActCounts { deck: (23, 32), upgraded: (9, 15), relics: (16, 21), basics: (0, 4), potions: (1, 3), boss_relics: 2 },
];

/// The exhaust decks practice fights build, after the wiki's Ironclad archetypes
/// (slay-the-spire.fandom.com/wiki/Ironclad): Corruption + Dark Embrace (a skill deck that burns
/// through itself, Entrench, then Body Slam), Juggernaut + Corruption + Feel No Pain (every skill
/// blocks and hits), and Status + Evolve (status-making cards with Evolve, Fire Breathing and
/// exhaust to clear them). Each takes a few of the attacks real decks pick up too (`ATTACKS`).
pub const EXHAUST_BUILDS: &[Archetype] = &[
    Archetype {
        name: "exhaust",
        core: &[Corruption, DarkEmbrace, Entrench, BodySlam],
        support: &[Impervious, ShrugItOff, Sentinel, BurningPact, Offering, Bloodletting, TrueGrit, ShrugItOff, Impervious, FlameBarrier,
                   GhostlyArmor, SecondWind, BodySlam, Barricade, DualWield, SeeingRed],
    },
    Archetype {
        name: "exhaust",
        core: &[Corruption, FeelNoPain, Juggernaut],
        support: &[ShrugItOff, Impervious, GhostlyArmor, PowerThrough, FlameBarrier, TrueGrit, Sentinel, SecondWind, IronWave, BurningPact,
                   Offering, ShrugItOff, IronWave, DarkEmbrace, Entrench],
    },
    Archetype {
        name: "exhaust",
        core: &[Evolve, FireBreathing, FeelNoPain],
        support: &[WildStrike, PowerThrough, RecklessCharge, Immolate, TrueGrit, SeverSoul, SecondWind, FiendFire, ShrugItOff, BattleTrance,
                   PommelStrike, FireBreathing, Evolve, DarkEmbrace],
    },
];

/// Attacks real decks of any kind pick up, for the exhaust builds' damage.
const ATTACKS: &[CardId] = &[PommelStrike, Carnage, Uppercut, Headbutt, TwinStrike, Hemokinesis, IronWave, SwordBoomerang, Bludgeon,
                             Clothesline, ThunderClap, Cleave, Immolate, FiendFire];

/// Relics a practice deck may carry: ones that only act in fights and need no setup screen.
const RELICS: &[RelicId] = &[
    RelicId::Anchor,
    RelicId::BagOfMarbles,
    RelicId::BronzeScales,
    RelicId::Lantern,
    RelicId::OddlySmoothStone,
    RelicId::Vajra,
    RelicId::Orichalcum,
    RelicId::PenNib,
    RelicId::Akabeko,
    RelicId::HornCleat,
    RelicId::HappyFlower,
    RelicId::Kunai,
    RelicId::Shuriken,
    RelicId::OrnamentalFan,
    RelicId::LetterOpener,
    RelicId::MercuryHourglass,
    RelicId::SelfFormingClay,
    RelicId::PaperFrog,
    RelicId::RedSkull,
    RelicId::GremlinHorn,
    RelicId::MeatOnTheBone,
    RelicId::FossilizedHelix,
    RelicId::Torii,
    RelicId::TungstenRod,
    RelicId::Pocketwatch,
    RelicId::InkBottle,
    RelicId::StoneCalendar,
    RelicId::BirdFacedUrn,
    RelicId::CharonsAshes,
    RelicId::DeadBranch,
    RelicId::ChampionsBelt,
    RelicId::Calipers,
    RelicId::IceCream,
    RelicId::RunicPyramid,
    RelicId::CaptainsWheel,
    RelicId::ArtOfWar,
    RelicId::Sundial,
    RelicId::BloodVial,
    RelicId::Nunchaku,
];

/// Thins the Strikes and Defends in `v` to a count drawn from `range` (at least two out, never
/// below none), as a run loses them.
fn trim_basics(v: &mut Vec<CardId>, rng: &mut StsRandom, range: (i32, i32)) {
    let basics = v.iter().filter(|&&c| is_basic(c)).count() as i32;
    let keep = rng.random_range(range.0, range.1).min(basics - 2).max(0);
    while v.iter().filter(|&&c| is_basic(c)).count() as i32 > keep {
        let at: Vec<usize> = (0..v.len()).filter(|&i| is_basic(v[i])).collect();
        v.remove(at[rng.random(at.len() as i32 - 1) as usize]);
    }
}

fn pick<T: Copy>(rng: &mut StsRandom, xs: &[T]) -> T {
    xs[rng.random(xs.len() as i32 - 1) as usize]
}

/// Where `build_deck`'s practice fights go, if set (jack, 2026-10-28: the Watcher network's A20
/// runs die most at the Act 3 bosses, the Time Eater alone ending 17%): the chance of each act
/// (1-4, any scale), a weight for each boss named (1 for the others), and the chance a fight that
/// isn't sent to the boss is an elite (jack, 2026-10-30: "do More elites"; Act 4's is Shield and
/// Spear), the rest rolling as usual.
#[derive(Clone, Debug, Default)]
pub struct Focus {
    pub act_weights: Option<[f64; 4]>,
    pub boss_weights: Vec<(String, f64)>,
    pub elite_share: Option<f64>,
}

impl Focus {
    fn boss_weight(&self, boss: &str) -> f64 {
        self.boss_weights.iter().find(|(b, _)| b == boss).map_or(1.0, |&(_, w)| w)
    }

    /// An act drawn by `act_weights` from one roll in 0..1.
    fn act(&self, w: &[f64; 4], roll: f64) -> i32 {
        let total: f64 = w.iter().sum();
        let mut acc = 0.0;
        for (i, x) in w.iter().enumerate() {
            acc += x / total;
            if roll < acc {
                return i as i32 + 1;
            }
        }
        4
    }

    /// One of `bosses`, by their weights.
    fn boss(&self, rng: &mut StsRandom, bosses: &[&'static str]) -> &'static str {
        let total: f64 = bosses.iter().map(|b| self.boss_weight(b)).sum();
        let roll = rng.random_float() as f64 * total;
        let mut acc = 0.0;
        for b in bosses {
            acc += self.boss_weight(b);
            if roll < acc {
                return b;
            }
        }
        bosses[bosses.len() - 1]
    }
}

/// `lo..=hi` from the act's row of `[act1, act2, act3]` ranges.
fn count(rng: &mut StsRandom, act: i32, ranges: [(i32, i32); 3]) -> i32 {
    let (lo, hi) = ranges[(act - 1) as usize];
    rng.random_range(lo, hi)
}

/// A practice run in a fight: (the run, its archetype, the encounter). `rng` picks everything;
/// the run's own streams come from `seed`. `only` restricts the archetype (by name); block decks
/// built that way hold Barricade three times in four, the card whose timing matters most.
/// `boss_share`, if set, is the chance the fight is the act's boss (otherwise 15%).
pub fn build(seed: i64, ascension: i32, rng: &mut StsRandom, only: Option<&str>, boss_share: Option<f64>) -> Result<(Run, &'static str, String), String> {
    let arch = match only {
        Some(name) => ARCHETYPES.iter().find(|a| a.name == name).ok_or_else(|| format!("practice: no archetype {name:?}"))?,
        None => &ARCHETYPES[rng.random(ARCHETYPES.len() as i32 - 1) as usize],
    };
    let exhaust = arch.name == "exhaust";
    let arch = if exhaust { &EXHAUST_BUILDS[rng.random(EXHAUST_BUILDS.len() as i32 - 1) as usize] } else { arch };
    let act = 1 + rng.random(2);
    let floor = match act {
        1 => rng.random_range(3, 15),
        2 => rng.random_range(18, 32),
        _ => rng.random_range(35, 49),
    };
    let mut cfg = RunConfig::new(seed);
    cfg.ascension = ascension;
    cfg.final_act = false;
    let mut run = Run::new(cfg);
    run.act = act;
    run.floor = floor;

    // The deck: the starter deck less some basics, the archetype's core and support, and a
    // few other red cards; more of each, and more upgrades, later in the run.
    let mut ids: Vec<CardId> = [StrikeR; 5].into_iter().chain([DefendR; 4]).chain([Bash]).collect();
    for _ in 0..count(rng, act, [(0, 1), (1, 3), (2, 5)]) {
        // Exhaust decks remove Strikes, not Defends (Defends still cycle for free under Corruption).
        let basics: Vec<usize> = (0..ids.len()).filter(|&i| if exhaust { ids[i] == StrikeR } else { matches!(ids[i], StrikeR | DefendR) }).collect();
        if let Some(&i) = basics.get(rng.random(basics.len() as i32 - 1).max(0) as usize) {
            ids.remove(i);
        }
    }
    let mut core: Vec<CardId> = arch.core.to_vec();
    if exhaust {
        // The build's defining card (Corruption, or Evolve) is always in.
        ids.push(core.remove(0));
    }
    if only == Some("block") && rng.random_float() < 0.75 {
        ids.push(Barricade);
        core.retain(|&c| c != Barricade);
    }
    for _ in 0..count(rng, act, [(1, 2), (2, 3), (3, 4)]) {
        if core.is_empty() {
            // Again, but not an exhaust build's defining card (a second Corruption does nothing).
            core = arch.core[if exhaust { 1 } else { 0 }..].to_vec();
        }
        ids.push(core.remove(rng.random(core.len() as i32 - 1) as usize));
    }
    for _ in 0..count(rng, act, [(2, 4), (4, 7), (6, 10)]) {
        ids.push(pick(rng, arch.support));
    }
    if exhaust {
        for _ in 0..count(rng, act, [(1, 2), (1, 3), (2, 3)]) {
            ids.push(pick(rng, ATTACKS));
        }
    }
    for _ in 0..count(rng, act, [(0, 2), (1, 3), (1, 4)]) {
        let id = pick(rng, &RED_LIBRARY_ORDER);
        if id.data().card_type != CardType::Curse {
            ids.push(id);
        }
    }
    let upgrade: f32 = [0.1, 0.3, 0.5][(act - 1) as usize];
    run.deck.clear();
    for id in ids {
        let mut c = run.new_card(id);
        // True Grit+ picks what it exhausts; Evolve+ draws more (the wiki's must-upgrades).
        let must = exhaust && matches!(id, TrueGrit | Evolve | Corruption | DarkEmbrace);
        if c.can_upgrade() && rng.random_float() < if must { upgrade.max(0.6) } else { upgrade } {
            c.upgrade();
        }
        run.deck.push(c);
    }

    let mut relics = RELICS.to_vec();
    for _ in 0..count(rng, act, [(0, 2), (1, 4), (3, 7)]) {
        let r = relics.remove(rng.random(relics.len() as i32 - 1) as usize);
        run.obtain_relic(r);
    }
    run.max_hp = 80 + rng.random(6 * act);
    run.hp = ((run.max_hp as f32) * rng.random_float_between(0.5, 1.0)).round() as i32;
    run.gold = 100;
    for slot in run.potions.iter_mut().take(count(rng, act, [(0, 1), (0, 2), (0, 2)]) as usize) {
        *slot = Some(random_potion(sts_core::character::Character::Ironclad, rng));
    }

    let (weak, strong, elites, bosses): (&[&str], &[&str], &[&str], &[&str]) = match act {
        1 => (&encounters::ACT1_WEAK, &encounters::ACT1_STRONG, &encounters::ACT1_ELITES, &encounters::ACT1_BOSSES),
        2 => (&encounters::ACT2_WEAK, &encounters::ACT2_STRONG, &encounters::ACT2_ELITES, &encounters::ACT2_BOSSES),
        _ => (&encounters::ACT3_WEAK, &encounters::ACT3_STRONG, &encounters::ACT3_ELITES, &encounters::ACT3_BOSSES),
    };
    run.boss = pick(rng, bosses).to_string();
    let boss_first = boss_share.is_some_and(|p| (rng.random_float() as f64) < p);
    for attempt in 0..5 {
        let roll = if boss_first && attempt == 0 { 1.0 } else { rng.random_float() };
        let (key, room) = if roll < 0.15 {
            (pick(rng, weak), RoomType::Monster)
        } else if roll < 0.55 {
            (pick(rng, strong), RoomType::Monster)
        } else if roll < 0.85 {
            (pick(rng, elites), RoomType::Elite)
        } else {
            let b = run.boss.clone();
            (bosses.iter().copied().find(|x| *x == b).unwrap_or(bosses[0]), RoomType::Boss)
        };
        run.start_fight(key, room);
        if matches!(run.screen, Screen::Combat) {
            return Ok((run, arch.name, key.to_string()));
        }
    }
    Err(format!("practice: no fight could start ({})", arch.name))
}

/// A random card of the character's (no basics, statuses or curses), by card-reward rarity, rarer
/// later in the run.
fn random_card(rng: &mut StsRandom, pools: &(Vec<CardId>, Vec<CardId>, Vec<CardId>, Vec<CardId>), act: i32) -> CardId {
    let (rare, uncommon) = [(0.03, 0.37), (0.06, 0.40), (0.10, 0.42), (0.10, 0.42)][(act - 1) as usize];
    let roll = rng.random_float();
    if roll < rare {
        pick(rng, &pools.2)
    } else if roll < rare + uncommon {
        pick(rng, &pools.1)
    } else {
        pick(rng, &pools.0)
    }
}

fn is_basic(id: CardId) -> bool {
    let d = id.data();
    d.starter_strike || d.starter_defend
}

/// A practice run of `character` in a fight, its deck of `kind` (`None`: a third each) sized for
/// `act` (`None`: Acts 1-3 three tenths each, Act 4 a tenth): (the run, its label, the encounter).
/// The label is the engine's name for an engine deck (`archetypes_for`), else the kind's; a
/// character without engines gets a grown deck instead. Counts come from `ACT_COUNTS`; relics from
/// the run's own pools, none that opens a screen on pickup (`Run::random_relic_no_screen`), one
/// boss relic after each of the Act 1 and 2 bosses. `boss_share`, if set, is the chance the fight
/// is the act's boss (otherwise 15%, Act 4's Heart half the time).
#[cfg_attr(not(test), allow(dead_code))] // (training calls `build_deck_with`)
pub fn build_deck(
    character: Character,
    seed: i64,
    ascension: i32,
    rng: &mut StsRandom,
    kind: Option<DeckKind>,
    act: Option<i32>,
    boss_share: Option<f64>,
    focus: Option<&Focus>,
) -> Result<(Run, &'static str, String), String> {
    build_deck_with(character, seed, ascension, rng, kind, act, boss_share, focus, None)
}

/// `build_deck`, with the fight's own RNG streams reseeded from `fight_seed` (if set) just before
/// it starts: the same deck, relics, potions, HP and encounter, other draws and monster rolls (a
/// replay of a lost fight, `Settings::retry_losses`).
#[allow(clippy::too_many_arguments)]
pub fn build_deck_with(
    character: Character,
    seed: i64,
    ascension: i32,
    rng: &mut StsRandom,
    kind: Option<DeckKind>,
    act: Option<i32>,
    boss_share: Option<f64>,
    focus: Option<&Focus>,
    fight_seed: Option<i64>,
) -> Result<(Run, &'static str, String), String> {
    let act = act.unwrap_or_else(|| {
        let roll = rng.random_float();
        if let Some(w) = focus.and_then(|f| f.act_weights.as_ref()) {
            return focus.unwrap().act(w, roll as f64);
        }
        if roll < 0.3 {
            1
        } else if roll < 0.6 {
            2
        } else if roll < 0.9 {
            3
        } else {
            4
        }
    });
    if !(1..=4).contains(&act) {
        return Err(format!("practice: no act {act}"));
    }
    let counts = &ACT_COUNTS[(act - 1) as usize];
    let mut kind = kind.unwrap_or_else(|| DeckKind::ALL[rng.random(2) as usize]);
    let archs = archetypes_for(character);
    if kind == DeckKind::Engine && archs.is_empty() {
        kind = DeckKind::Grown;
    }
    let mut label = kind.name();
    let pools = Pools::new_cards(character);
    let size = rng.random_range(counts.deck.0, counts.deck.1);
    let starter = character.starting_deck();
    let mut ids: Vec<CardId> = match kind {
        // Grown and random decks keep the act's count of Strikes and Defends, as engine decks do
        // (jack, 2026-10-30: "Sort out decks"): grown ones kept all of them, so an Act 3 grown deck
        // won 6-8% of its boss fights, and random ones had none, cleaner than any real deck.
        DeckKind::Grown => {
            let mut v = starter.clone();
            trim_basics(&mut v, rng, counts.basics);
            while (v.len() as i32) < size {
                v.push(random_card(rng, &pools, act));
            }
            v
        }
        DeckKind::Random => {
            let mut v: Vec<CardId> = starter.iter().copied().filter(|&c| is_basic(c)).collect();
            trim_basics(&mut v, rng, counts.basics);
            while (v.len() as i32) < size {
                v.push(random_card(rng, &pools, act));
            }
            v
        }
        DeckKind::Engine => {
            let arch = &archs[rng.random(archs.len() as i32 - 1) as usize];
            label = arch.name;
            let mut v = starter.clone();
            // A couple of basics out at least, more later in the run.
            trim_basics(&mut v, rng, counts.basics);
            let mut core = arch.core.to_vec();
            for _ in 0..rng.random_range([1, 2, 2, 3][(act - 1) as usize], [2, 3, 4, 4][(act - 1) as usize]) {
                if core.is_empty() {
                    core = arch.core.to_vec();
                }
                v.push(core.remove(rng.random(core.len() as i32 - 1) as usize));
            }
            let good = good_cards(character);
            if !good.is_empty() {
                for _ in 0..rng.random_range(1, 2) {
                    v.push(pick(rng, good));
                }
            }
            while (v.len() as i32) < size {
                v.push(pick(rng, arch.support));
            }
            v
        }
    };
    ids.retain(|id| id.data().card_type != CardType::Curse);

    let floor = match act {
        1 => rng.random_range(3, 15),
        2 => rng.random_range(18, 32),
        3 => rng.random_range(35, 49),
        _ => 54,
    };
    let mut cfg = RunConfig::new(seed);
    cfg.ascension = ascension;
    cfg.character = character;
    cfg.final_act = act == 4;
    let mut run = Run::new(cfg);
    run.act = act;
    run.floor = floor;
    if act == 4 {
        // Act 4 is reached with the three keys.
        run.keys.ruby = true;
        run.keys.emerald = true;
        run.keys.sapphire = true;
    }
    run.deck.clear();
    for id in ids {
        let c = run.new_card(id);
        run.deck.push(c);
    }
    for _ in 0..rng.random_range(counts.upgraded.0, counts.upgraded.1) {
        let can: Vec<usize> = (0..run.deck.len()).filter(|&i| run.deck[i].can_upgrade()).collect();
        if can.is_empty() {
            break;
        }
        let i = can[rng.random(can.len() as i32 - 1) as usize];
        run.deck[i].upgrade();
    }

    run.max_hp += rng.random(6 * act);
    // Drawn until the run holds that many: a boss relic that replaces the starting one (Frozen
    // Core, Ring of the Serpent...) adds none. A few tries more at most (an empty pool gives
    // Circlets, which stack).
    let relics = rng.random_range(counts.relics.0, counts.relics.1) as usize;
    let boss = counts.boss_relics.min(relics as i32 - 1);
    for k in 0..3 * relics as i32 {
        if run.relics.len() >= relics {
            break;
        }
        let tier = if k < boss {
            RelicTier::Boss
        } else {
            let roll = rng.random_float();
            if roll < 0.5 {
                RelicTier::Common
            } else if roll < 0.83 {
                RelicTier::Uncommon
            } else {
                RelicTier::Rare
            }
        };
        let r = run.random_relic_no_screen(tier);
        run.obtain_relic(r);
    }
    run.hp = ((run.max_hp as f32) * rng.random_float_between(0.5, 1.0)).round().max(1.0) as i32;
    run.gold = 100;
    let potions = rng.random_range(counts.potions.0, counts.potions.1) as usize;
    for slot in run.potions.iter_mut().take(potions) {
        *slot = Some(random_potion(character, rng));
    }

    let boss_first = boss_share.is_some_and(|p| (rng.random_float() as f64) < p);
    // (Drawn only with an elite share set, so fights without one come out as before.)
    let elite_first = !boss_first && focus.and_then(|f| f.elite_share).is_some_and(|p| (rng.random_float() as f64) < p);
    if act == 4 {
        let heart = boss_first || (!elite_first && rng.random_float() < 0.5);
        let (key, room) = if heart { ("The Heart", RoomType::Boss) } else { ("Shield and Spear", RoomType::Elite) };
        run.boss = "The Heart".to_string();
        run.floor = if heart { 55 } else { 54 };
        if let Some(f) = fight_seed {
            run.rng.reseed_fight(f);
        }
        run.start_fight(key, room);
        return if matches!(run.screen, Screen::Combat) { Ok((run, label, key.to_string())) } else { Err(format!("practice: {key} didn't start")) };
    }
    let (weak, strong, elites, bosses): (&[&str], &[&str], &[&str], &[&str]) = match act {
        1 => (&encounters::ACT1_WEAK, &encounters::ACT1_STRONG, &encounters::ACT1_ELITES, &encounters::ACT1_BOSSES),
        2 => (&encounters::ACT2_WEAK, &encounters::ACT2_STRONG, &encounters::ACT2_ELITES, &encounters::ACT2_BOSSES),
        _ => (&encounters::ACT3_WEAK, &encounters::ACT3_STRONG, &encounters::ACT3_ELITES, &encounters::ACT3_BOSSES),
    };
    run.boss = match focus.filter(|f| !f.boss_weights.is_empty()) {
        Some(f) => f.boss(rng, bosses),
        None => pick(rng, bosses),
    }
    .to_string();
    for attempt in 0..5 {
        let roll = if boss_first && attempt == 0 {
            1.0
        } else if elite_first && attempt == 0 {
            0.7
        } else {
            rng.random_float()
        };
        let (key, room) = if roll < 0.15 {
            (pick(rng, weak), RoomType::Monster)
        } else if roll < 0.55 {
            (pick(rng, strong), RoomType::Monster)
        } else if roll < 0.85 {
            (pick(rng, elites), RoomType::Elite)
        } else {
            run.floor = [16, 33, 50][(act - 1) as usize];
            let b = run.boss.clone();
            (bosses.iter().copied().find(|x| *x == b).unwrap_or(bosses[0]), RoomType::Boss)
        };
        if let Some(f) = fight_seed {
            run.rng.reseed_fight(f);
        }
        run.start_fight(key, room);
        if matches!(run.screen, Screen::Combat) {
            return Ok((run, label, key.to_string()));
        }
    }
    Err(format!("practice: no fight could start ({label})"))
}

/// The practice fight `practice` (a run in the fight `key`) set up as the modded game's
/// `harness` command sets one up, so the network's play of it can be watched in the game
/// (python/watch/watch_practice.py, tools/trace/playback_fights.py): a new run on the practice
/// run's seed, ascension and character, at Neow; its relics (the first harness command, which
/// runs each relic's onEquip on the starting deck); its deck, HP and potions (the second, which
/// replaces what those changed); then `key` in an ordinary monster room on floor 1, as the command
/// starts it. That fight differs from training's (floor 1's seeds, no elite or boss room for
/// Sling of Courage and the like), but the network still sees the practice run's act and floor.
/// Returns the run and the two harness commands. A0-10 only: the command resets the potion slots
/// to 3, and an A11+ game has 2.
pub fn harness_fight(practice: &Run, key: &str) -> Result<(Run, Vec<serde_json::Value>), String> {
    if practice.config.ascension > 10 {
        return Err(format!("harness fights: A0-10 only (A{})", practice.config.ascension));
    }
    let mut cfg = RunConfig::new(practice.config.seed);
    cfg.ascension = practice.config.ascension;
    cfg.character = practice.config.character;
    cfg.final_act = true; // as in the harness profile
    let mut run = Run::new(cfg);
    run.relics.clear();
    for r in &practice.relics {
        run.obtain_relic(r.id);
    }
    run.deck.clear();
    for c in &practice.deck {
        let mut card = run.new_card(c.id);
        for _ in 0..c.times_upgraded {
            card.upgrade();
        }
        run.deck.push(card);
    }
    run.max_hp = practice.max_hp;
    run.hp = practice.hp;
    let held: Vec<_> = practice.potions.iter().flatten().copied().collect();
    if held.len() > run.potions.len() {
        return Err(format!("harness fights: {} potions for {} slots", held.len(), run.potions.len()));
    }
    for (i, slot) in run.potions.iter_mut().enumerate() {
        *slot = held.get(i).copied();
    }
    run.start_harness_fight(key);
    if !matches!(run.screen, Screen::Combat) {
        return Err(format!("harness fights: {key} didn't start"));
    }
    run.act = practice.act;
    run.floor = practice.floor;
    let commands = vec![
        json!({"relics": practice.relics.iter().map(|r| r.id.java_id()).collect::<Vec<_>>()}),
        json!({"deck": practice.deck.iter().map(|c| json!([c.id.data().java_id, c.times_upgraded])).collect::<Vec<_>>(),
               "max_hp": practice.max_hp, "hp": practice.hp, "potions": held.iter().map(|p| p.java_id()).collect::<Vec<_>>(),
               "encounter": key}),
    ];
    Ok((run, commands))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every character's deck fights start, in their act, with the deck, relics and HP of their
    /// kind and act.
    #[test]
    fn deck_fights_start_for_every_character_kind_and_act() {
        for character in [Character::Ironclad, Character::Silent, Character::Defect, Character::Watcher] {
            for kind in DeckKind::ALL {
                for act in 1..=4 {
                    for seed in 0..12i64 {
                        let mut rng = StsRandom::new(seed * 31 + act as i64);
                        let (run, label, key) = build_deck(character, seed, 10, &mut rng, Some(kind), Some(act), None, None).unwrap();
                        let what = format!("{character:?} {kind:?} act {act} seed {seed}: {label} {key}");
                        assert!(matches!(run.screen, Screen::Combat), "{what}");
                        assert_eq!((run.act, run.config.character), (act, character), "{what}");
                        let c = &ACT_COUNTS[(act - 1) as usize];
                        let n = run.deck.len() as i32;
                        let starter = character.starting_deck().len() as i32;
                        assert!(n >= c.deck.0.min(starter - 2) && n <= c.deck.1 + 6, "{what}: {n} cards");
                        assert!(run.relics.len() as i32 >= c.relics.0 && run.relics.len() as i32 <= c.relics.1, "{what}: {} relics", run.relics.len());
                        assert!(run.hp >= 1 && run.hp <= run.max_hp, "{what}");
                        let basics = run.deck.iter().filter(|x| is_basic(x.id)).count();
                        let starter_basics = character.starting_deck().iter().filter(|&&s| is_basic(s)).count() as i32;
                        // The act's count of Strikes and Defends, two out at least (Pandora's Box turns them into other cards).
                        let act_basics = |b: i32| {
                            b <= c.basics.1.min(starter_basics - 2) && (run.has_relic(RelicId::PandorasBox) || b >= c.basics.0.min(starter_basics - 2))
                        };
                        match kind {
                            DeckKind::Grown => {
                                assert!(act_basics(basics as i32), "{what}: {basics} basics");
                                // The starting deck's other cards (Bash, Eruption and Vigilance...) stay.
                                assert!(
                                    run.has_relic(RelicId::PandorasBox)
                                        || character.starting_deck().iter().filter(|&&s| !is_basic(s)).all(|s| run.deck.iter().any(|x| x.id == *s)),
                                    "{what}: starting deck"
                                );
                            }
                            DeckKind::Random => assert!(act_basics(basics as i32), "{what}: {basics} basics in a random deck"),
                            DeckKind::Engine => {
                                let archs = archetypes_for(character);
                                if archs.is_empty() {
                                    assert_eq!(label, "grown", "{what}: no engines, a grown deck");
                                } else {
                                    let a = archs.iter().find(|a| a.name == label).expect(&what);
                                    assert!(run.deck.iter().any(|x| a.core.contains(&x.id)), "{what}: no core card");
                                    assert!(basics as i32 <= character.starting_deck().iter().filter(|&&s| is_basic(s)).count() as i32 - 2, "{what}: {basics} basics");
                                }
                            }
                        }
                        if act == 4 {
                            assert!(key == "Shield and Spear" || key == "The Heart", "{what}");
                            assert!(run.keys.ruby && run.keys.emerald && run.keys.sapphire, "{what}");
                        }
                    }
                }
            }
        }
    }

    /// The Defect's fights mix the three kinds a third each and Acts 1-3 three tenths each, and
    /// the Act 4 fights play on.
    #[test]
    fn defect_deck_fights_mix_kinds_and_acts() {
        let (mut kinds, mut acts) = (std::collections::HashMap::new(), [0; 4]);
        for seed in 0..600 {
            let mut rng = StsRandom::new(seed);
            let (run, label, _) = build_deck(Character::Defect, seed, 0, &mut rng, None, None, None, None).unwrap();
            let kind = if DEFECT_ARCHETYPES.iter().any(|a| a.name == label) { "engine" } else { label };
            *kinds.entry(kind).or_insert(0) += 1;
            acts[(run.act - 1) as usize] += 1;
        }
        for k in ["grown", "engine", "random"] {
            assert!((150..250).contains(&kinds[k]), "{kinds:?}");
        }
        assert!(acts[..3].iter().all(|&a| (140..220).contains(&a)) && (30..95).contains(&acts[3]), "{acts:?}");
        // An Act 4 fight runs: random legal moves for a while without an error.
        for seed in 0..6 {
            let mut rng = StsRandom::new(seed);
            let (mut run, _, key) = build_deck(Character::Defect, seed, 20, &mut rng, Some(DeckKind::Engine), Some(4), Some(if seed % 2 == 0 { 1.0 } else { 0.0 }), None).unwrap();
            assert!(key == "The Heart" || (seed % 2 == 1 && key == "Shield and Spear"), "{key}");
            for _ in 0..60 {
                let legal = run.legal_actions();
                if legal.is_empty() || !matches!(run.screen, Screen::Combat) {
                    break;
                }
                let a = legal[rng.random(legal.len() as i32 - 1) as usize].clone();
                run.step(&a).unwrap();
            }
        }
    }

    /// Every Watcher engine comes up, and its fights run (random legal moves through stances,
    /// Mantra, Scry and Retain), at A20 in any act.
    #[test]
    fn watcher_engine_decks_cover_every_engine() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0..80 {
            let mut rng = StsRandom::new(seed);
            let (mut run, label, _) = build_deck(Character::Watcher, seed, 20, &mut rng, Some(DeckKind::Engine), None, None, None).unwrap();
            seen.insert(label);
            for _ in 0..80 {
                let legal = run.legal_actions();
                if legal.is_empty() || !matches!(run.screen, Screen::Combat) {
                    break;
                }
                let a = legal[rng.random(legal.len() as i32 - 1) as usize].clone();
                run.step(&a).unwrap();
            }
        }
        assert_eq!(seen.len(), WATCHER_ARCHETYPES.len(), "{seen:?}");
    }

    #[test]
    fn practice_fights_start_with_their_archetype() {
        for seed in 0..200 {
            let mut rng = StsRandom::new(seed);
            let (run, arch, key) = build(seed, 0, &mut rng, None, None).unwrap();
            assert!(matches!(run.screen, Screen::Combat), "{arch} {key}");
            let a = ARCHETYPES.iter().find(|a| a.name == arch).unwrap();
            assert!(run.deck.iter().any(|c| a.core.contains(&c.id)), "{arch}: no core card");
            assert!(run.hp >= 1 && run.hp <= run.max_hp);
            assert!((1..=3).contains(&run.act));
        }
        let mut barricade = 0;
        for seed in 0..200 {
            let mut rng = StsRandom::new(seed);
            let (run, arch, _) = build(seed, 0, &mut rng, Some("block"), None).unwrap();
            assert_eq!(arch, "block");
            barricade += run.deck.iter().any(|c| c.id == Barricade) as usize;
        }
        assert!(barricade > 140, "{barricade} of 200 block decks hold Barricade");
        let bosses = (0..200)
            .filter(|&seed| {
                let mut rng = StsRandom::new(seed);
                let (run, _, key) = build(seed, 0, &mut rng, Some("block"), Some(0.6)).unwrap();
                [&encounters::ACT1_BOSSES[..], &encounters::ACT2_BOSSES[..], &encounters::ACT3_BOSSES[..]].iter().any(|b| b.contains(&key.as_str())) && run.act >= 1
            })
            .count();
        assert!((100..150).contains(&bosses), "{bosses} of 200 fights with boss_share 0.6 are bosses");
    }

    fn strip(v: &mut serde_json::Value, keys: &[&str]) {
        match v {
            serde_json::Value::Object(o) => {
                for k in keys {
                    o.remove(*k);
                }
                o.values_mut().for_each(|x| strip(x, keys));
            }
            serde_json::Value::Array(a) => a.iter_mut().for_each(|x| strip(x, keys)),
            _ => {}
        }
    }

    /// A harness fight is the fight `sts-cli replay` builds from the game's state after the same
    /// harness commands (`CombatConfig`, as `config_from_trace` reads a recording: the replay that
    /// matches the game's fights), for every character, deck kind and act 1-3, move by move
    /// under random play. Act 4 and A11+ are left out (not set up by the harness).
    #[test]
    fn harness_fights_are_the_fights_the_harness_sets_up() {
        use sts_core::relics::RelicState;
        use sts_core::{Combat, CombatConfig};
        let mut checked = 0;
        for character in [Character::Ironclad, Character::Silent, Character::Defect, Character::Watcher] {
            for kind in DeckKind::ALL {
                for act in 1..=3 {
                    for k in 0..4 {
                        let seed = 1_000 + 97 * k + 13 * act as i64;
                        let ascension = (k as i32 * 3 + act) % 11;
                        let mut rng = StsRandom::new(seed);
                        let (practice, _, key) = build_deck(character, seed, ascension, &mut rng, Some(kind), Some(act), None, None).unwrap();
                        let (run, commands) = harness_fight(&practice, &key).unwrap();
                        assert_eq!(commands.len(), 2);
                        assert_eq!((run.act, run.floor), (practice.act, practice.floor), "the network sees the practice fight's act and floor");
                        // What the replay reads from the game: the relics' onEquip counters (but
                        // these four, which the game's onEquip zeroes), gold and potion slots as
                        // the relics left them, the deck and HP as the second command set them.
                        let mut cfg = CombatConfig::new(character, seed, &key);
                        cfg.ascension = ascension;
                        cfg.deck = practice.deck.iter().map(|c| (c.id, c.times_upgraded)).collect();
                        cfg.player_hp = practice.hp;
                        cfg.player_max_hp = practice.max_hp;
                        cfg.gold = run.gold;
                        cfg.relics = practice.relics.iter().map(|r| r.id).collect();
                        cfg.relic_counters = Some(
                            cfg.relics
                                .iter()
                                .map(|&r| match r {
                                    RelicId::HappyFlower | RelicId::IncenseBurner | RelicId::Sundial | RelicId::TinyChest => 0,
                                    RelicId::DuVuDoll => cfg.deck.iter().filter(|(c, _)| c.data().card_type == CardType::Curse).count() as i32,
                                    _ => RelicState::new(r).counter,
                                })
                                .collect(),
                        );
                        cfg.potions = run.potions.clone();
                        let mut want = Combat::new(&cfg).unwrap();
                        let mut got = (**run.combat.as_ref().unwrap()).clone();
                        let same = |got: &Combat, want: &Combat| {
                            let (mut a, mut b) = (serde_json::to_value(got).unwrap(), serde_json::to_value(want).unwrap());
                            for v in [&mut a, &mut b] {
                                // A queued hand card is named by its instance: by its place in the hand instead (an end of turn
                                // with Burns in hand queues them), before the instances go.
                                let hand: Vec<_> = v["hand"].as_array().unwrap().iter().map(|c| c["instance"].clone()).collect();
                                for q in v["card_queue"].as_array_mut().unwrap() {
                                    if let Some(inst) = q["card"].get("InHand").cloned() {
                                        q["card"] = json!({"InHand": hand.iter().position(|x| *x == inst)});
                                    }
                                }
                                // Card ids count from different starts (the run made its starting
                                // deck first); only identity matters.
                                strip(v, &["uuid", "instance"]);
                                let o = v.as_object_mut().unwrap();
                                let types: Vec<_> = o["drawn_cards"].as_array().unwrap().iter().map(|p| p[1].clone()).collect();
                                o.insert("drawn_cards".into(), json!(types));
                                let growth: Vec<_> = o["master_deck_growth"].as_array().unwrap().iter().map(|p| p[1].clone()).collect();
                                o.insert("master_deck_growth".into(), json!(growth));
                                o.remove("next_uuid");
                                o.remove("next_instance");
                                // The harness's room is an ordinary MonsterRoom: no relic counts its
                                // fight as elite (eliteTrigger, which only MonsterRoomElite sets);
                                // CombatConfig marks the Act 1 elites elite, untested by any recording.
                                o.remove("elite");
                                // Drawn at the run's start (the map, the monster lists, the relic
                                // pools), in the game too, and never in a fight.
                                for k in ["map", "monster", "relic"] {
                                    o["rng"].as_object_mut().unwrap().remove(k);
                                }
                            }
                            if a != b {
                                let mut diff: Vec<_> = a.as_object().unwrap().iter().filter(|(k, v)| b.get(*k) != Some(v)).map(|(k, _)| k.clone()).collect();
                                for (k, v) in a["rng"].as_object().unwrap() {
                                    if b["rng"].get(k) != Some(v) {
                                        diff.push(format!("rng.{k}"));
                                    }
                                }
                                return Err(diff);
                            }
                            Ok(())
                        };
                        let mut roll = StsRandom::new(seed ^ 7);
                        for step in 0..400 {
                            if let Err(diff) = same(&got, &want) {
                                panic!("{character:?} {kind:?} act {act} A{ascension} {key}, step {step}: {diff:?} differ");
                            }
                            if got.is_over() {
                                break;
                            }
                            let legal = got.legal_actions();
                            let a = legal[roll.random(legal.len() as i32 - 1) as usize].clone();
                            got.step(&a).unwrap();
                            want.step(&a).unwrap();
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 4 * 3 * 3 * 4);
    }

    /// A focus on Act 3 and the Time Eater: every fight in Act 3, and with the act's boss every
    /// time, nearly all of them the Time Eater; without a focus the act roll is as before.
    #[test]
    fn focus_weights_acts_and_bosses() {
        let focus = Focus { act_weights: Some([0.0, 0.0, 1.0, 0.0]), boss_weights: vec![("Time Eater".to_string(), 100.0)], elite_share: None };
        let mut time_eater = 0;
        for seed in 0..60 {
            let mut rng = StsRandom::new(seed);
            let (run, _, key) = build_deck(Character::Watcher, seed, 20, &mut rng, Some(DeckKind::Grown), None, Some(1.0), Some(&focus)).unwrap();
            assert_eq!(run.act, 3);
            time_eater += (key == "Time Eater") as i32;
        }
        assert!(time_eater >= 55, "{time_eater} of 60");
        let (mut a, mut b) = (StsRandom::new(5), StsRandom::new(5));
        let plain = build_deck(Character::Watcher, 5, 20, &mut a, Some(DeckKind::Grown), None, None, None).unwrap();
        let empty = build_deck(Character::Watcher, 5, 20, &mut b, Some(DeckKind::Grown), None, None, Some(&Focus::default())).unwrap();
        assert_eq!((plain.0.act, plain.2), (empty.0.act, empty.2));
    }

    /// An elite share sends that share of the fights not sent to the boss to an elite of the act
    /// (Act 4's Shield and Spear); the boss share still comes first.
    #[test]
    fn elite_share_sends_fights_to_elites() {
        let all = Focus { elite_share: Some(1.0), ..Focus::default() };
        for act in 1..=4 {
            for seed in 0..20 {
                let mut rng = StsRandom::new(seed);
                let (run, _, key) = build_deck(Character::Ironclad, seed, 20, &mut rng, Some(DeckKind::Grown), Some(act), None, Some(&all)).unwrap();
                let elites: &[&str] = match act {
                    1 => &encounters::ACT1_ELITES,
                    2 => &encounters::ACT2_ELITES,
                    3 => &encounters::ACT3_ELITES,
                    _ => &["Shield and Spear"],
                };
                assert!(elites.contains(&key.as_str()), "act {act} seed {seed}: {key}");
                assert!(matches!(run.screen, Screen::Combat));
                let mut rng = StsRandom::new(seed);
                let (_, _, boss) = build_deck(Character::Ironclad, seed, 20, &mut rng, Some(DeckKind::Grown), Some(act), Some(1.0), Some(&all)).unwrap();
                assert!(!elites.contains(&boss.as_str()), "act {act} seed {seed}: the boss share first, not {boss}");
            }
        }
        let half = Focus { elite_share: Some(0.5), ..Focus::default() };
        let elite_fights = (0..400i64)
            .filter(|&seed| {
                let mut rng = StsRandom::new(seed);
                let (_, _, key) = build_deck(Character::Ironclad, seed, 20, &mut rng, Some(DeckKind::Engine), Some(2), None, Some(&half)).unwrap();
                encounters::ACT2_ELITES.contains(&key.as_str())
            })
            .count();
        // Half sent to an elite, and 30% of the other half rolls one: about 65%.
        assert!((220..300).contains(&elite_fights), "{elite_fights} of 400");
    }
}
