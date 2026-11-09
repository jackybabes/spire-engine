//! Card definitions and per-instance card state (`AbstractCard`).

mod data;

pub use data::{
    CardId, ALL_CARDS, BLUE_LIBRARY_ORDER, CARD_DATA, COLORLESS_LIBRARY_ORDER, CURSE_LIBRARY_ORDER, GREEN_LIBRARY_ORDER,
    PURPLE_LIBRARY_ORDER, RED_LIBRARY_ORDER,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardType {
    Attack,
    Skill,
    Power,
    Status,
    Curse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardColor {
    Red,
    Green,
    Blue,
    Purple,
    Colorless,
    Curse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardRarity {
    Basic,
    Special,
    Common,
    Uncommon,
    Rare,
    Curse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardTarget {
    Enemy,
    AllEnemy,
    SelfTarget,
    None,
    SelfAndEnemy,
    All,
}

/// Static, per-card-type data (the numbers passed to the `AbstractCard` constructor).
#[derive(Clone, Copy, Debug)]
pub struct CardData {
    pub id: CardId,
    pub java_id: &'static str,
    pub name: &'static str,
    pub cost: i32,
    pub card_type: CardType,
    pub color: CardColor,
    pub rarity: CardRarity,
    pub target: CardTarget,
    pub damage: i32,
    pub block: i32,
    pub magic: i32,
    pub exhaust: bool,
    pub ethereal: bool,
    pub innate: bool,
    /// `selfRetain` from the constructor (the Watcher's Retain cards).
    pub self_retain: bool,
    pub multi_damage: bool,
    pub strike: bool,
    pub healing: bool,
    pub starter_strike: bool,
    pub starter_defend: bool,
}

/// The other characters' cards whose logic is written so far (milestone 5; the port checklist,
/// `tools/extract/port_checklist.py`, reads this list).
pub const PORTED: &[CardId] = &[
    // The Silent.
    CardId::AThousandCuts,
    CardId::Accuracy,
    CardId::Acrobatics,
    CardId::Adrenaline,
    CardId::AfterImage,
    CardId::Alchemize,
    CardId::AllOutAttack,
    CardId::Backflip,
    CardId::Backstab,
    CardId::Bane,
    CardId::BladeDance,
    CardId::Blur,
    CardId::BouncingFlask,
    CardId::BulletTime,
    CardId::Burst,
    CardId::CalculatedGamble,
    CardId::Caltrops,
    CardId::Catalyst,
    CardId::Choke,
    CardId::CloakAndDagger,
    CardId::Concentrate,
    CardId::CorpseExplosion,
    CardId::CripplingPoison,
    CardId::DaggerSpray,
    CardId::DaggerThrow,
    CardId::Dash,
    CardId::DeadlyPoison,
    CardId::DefendG,
    CardId::Deflect,
    CardId::DieDieDie,
    CardId::Distraction,
    CardId::DodgeAndRoll,
    CardId::Doppelganger,
    CardId::EndlessAgony,
    CardId::Envenom,
    CardId::EscapePlan,
    CardId::Eviscerate,
    CardId::Expertise,
    CardId::Finisher,
    CardId::Flechettes,
    CardId::FlyingKnee,
    CardId::Footwork,
    CardId::GlassKnife,
    CardId::GrandFinale,
    CardId::HeelHook,
    CardId::InfiniteBlades,
    CardId::LegSweep,
    CardId::Malaise,
    CardId::MasterfulStab,
    CardId::Neutralize,
    CardId::Nightmare,
    CardId::NoxiousFumes,
    CardId::Outmaneuver,
    CardId::PhantasmalKiller,
    CardId::PiercingWail,
    CardId::PoisonedStab,
    CardId::Predator,
    CardId::Prepared,
    CardId::QuickSlash,
    CardId::Reflex,
    CardId::RiddleWithHoles,
    CardId::Setup,
    CardId::Skewer,
    CardId::Slice,
    CardId::SneakyStrike,
    CardId::StormOfSteel,
    CardId::StrikeG,
    CardId::SuckerPunch,
    CardId::Survivor,
    CardId::Tactician,
    CardId::Terror,
    CardId::ToolsOfTheTrade,
    CardId::Unload,
    CardId::WellLaidPlans,
    CardId::WraithForm,
    CardId::Shiv,
    // The Defect.
    CardId::Aggregate,
    CardId::AllForOne,
    CardId::Amplify,
    CardId::AutoShields,
    CardId::BallLightning,
    CardId::Barrage,
    CardId::BeamCell,
    CardId::BiasedCognition,
    CardId::Blizzard,
    CardId::BootSequence,
    CardId::Buffer,
    CardId::Capacitor,
    CardId::Chaos,
    CardId::Chill,
    CardId::Claw,
    CardId::ColdSnap,
    CardId::CompileDriver,
    CardId::ConserveBattery,
    CardId::Consume,
    CardId::Coolheaded,
    CardId::CoreSurge,
    CardId::CreativeAI,
    CardId::Darkness,
    CardId::DefendB,
    CardId::Defragment,
    CardId::DoomAndGloom,
    CardId::DoubleEnergy,
    CardId::Dualcast,
    CardId::EchoForm,
    CardId::Electrodynamics,
    CardId::Equilibrium,
    CardId::FTL,
    CardId::Fission,
    CardId::ForceField,
    CardId::Fusion,
    CardId::GeneticAlgorithm,
    CardId::Glacier,
    CardId::GoForTheEyes,
    CardId::Heatsinks,
    CardId::HelloWorld,
    CardId::Hologram,
    CardId::Hyperbeam,
    CardId::Leap,
    CardId::LockOn,
    CardId::Loop,
    CardId::MachineLearning,
    CardId::Melter,
    CardId::MeteorStrike,
    CardId::MultiCast,
    CardId::Overclock,
    CardId::Rainbow,
    CardId::Reboot,
    CardId::Rebound,
    CardId::Recursion,
    CardId::Recycle,
    CardId::ReinforcedBody,
    CardId::Reprogram,
    CardId::RipAndTear,
    CardId::Scrape,
    CardId::Seek,
    CardId::SelfRepair,
    CardId::Skim,
    CardId::Stack,
    CardId::StaticDischarge,
    CardId::SteamBarrier,
    CardId::Storm,
    CardId::Streamline,
    CardId::StrikeB,
    CardId::Sunder,
    CardId::SweepingBeam,
    CardId::Tempest,
    CardId::ThunderStrike,
    CardId::Turbo,
    CardId::WhiteNoise,
    CardId::Zap,
    // The Watcher (and her temporary and option cards).
    CardId::Alpha,
    CardId::BattleHymn,
    CardId::Blasphemy,
    CardId::BowlingBash,
    CardId::Brilliance,
    CardId::CarveReality,
    CardId::Collect,
    CardId::Conclude,
    CardId::ConjureBlade,
    CardId::Consecrate,
    CardId::Crescendo,
    CardId::CrushJoints,
    CardId::CutThroughFate,
    CardId::DeceiveReality,
    CardId::DefendP,
    CardId::DeusExMachina,
    CardId::DevaForm,
    CardId::Devotion,
    CardId::EmptyBody,
    CardId::EmptyFist,
    CardId::EmptyMind,
    CardId::Eruption,
    CardId::Establishment,
    CardId::Evaluate,
    CardId::Fasting,
    CardId::FearNoEvil,
    CardId::FlurryOfBlows,
    CardId::FlyingSleeves,
    CardId::FollowUp,
    CardId::ForeignInfluence,
    CardId::Foresight,
    CardId::Halt,
    CardId::Indignation,
    CardId::InnerPeace,
    CardId::Judgement,
    CardId::JustLucky,
    CardId::LessonLearned,
    CardId::LikeWater,
    CardId::MasterReality,
    CardId::Meditate,
    CardId::MentalFortress,
    CardId::Nirvana,
    CardId::Omniscience,
    CardId::Perseverance,
    CardId::Pray,
    CardId::PressurePoints,
    CardId::Prostrate,
    CardId::Protect,
    CardId::Ragnarok,
    CardId::ReachHeaven,
    CardId::Rushdown,
    CardId::Sanctity,
    CardId::SandsOfTime,
    CardId::SashWhip,
    CardId::Scrawl,
    CardId::SignatureMove,
    CardId::SimmeringFury,
    CardId::SpiritShield,
    CardId::StrikeP,
    CardId::Study,
    CardId::Swivel,
    CardId::TalkToTheHand,
    CardId::Tantrum,
    CardId::ThirdEye,
    CardId::Tranquility,
    CardId::Vault,
    CardId::Vigilance,
    CardId::Wallop,
    CardId::WaveOfTheHand,
    CardId::Weave,
    CardId::WheelKick,
    CardId::WindmillStrike,
    CardId::Wish,
    CardId::Worship,
    CardId::WreathOfFlame,
    CardId::Beta,
    CardId::Expunger,
    CardId::Insight,
    CardId::Miracle,
    CardId::Omega,
    CardId::Safety,
    CardId::Smite,
    CardId::ThroughViolence,
    CardId::BecomeAlmighty,
    CardId::ChooseCalm,
    CardId::ChooseWrath,
    CardId::FameAndFortune,
    CardId::LiveForever,
];

impl CardId {
    pub fn data(self) -> &'static CardData {
        &CARD_DATA[self as usize]
    }

    /// Whether the card's upgrade and `use()` are implemented: the Ironclad's cards, statuses,
    /// curses, colorless cards, and the other characters' cards in [`PORTED`].
    pub fn is_ported(self) -> bool {
        let d = self.data();
        d.color == CardColor::Red
            || d.card_type == CardType::Status
            || d.card_type == CardType::Curse
            || (d.color == CardColor::Colorless && d.rarity != CardRarity::Special)
            || matches!(self, CardId::JAX | CardId::Apparition | CardId::Bite | CardId::RitualDagger)
            || PORTED.contains(&self)
    }

    pub fn from_java_id(id: &str) -> Option<CardId> {
        CARD_DATA.iter().find(|d| d.java_id == id).map(|d| d.id)
    }

    pub fn from_name(name: &str) -> Option<CardId> {
        let n = name.to_ascii_lowercase();
        CARD_DATA
            .iter()
            .find(|d| d.java_id.to_ascii_lowercase() == n || d.name.to_ascii_lowercase() == n || format!("{:?}", d.id).to_ascii_lowercase() == n)
            .map(|d| d.id)
    }
}

/// One card instance. `uuid` is shared by `makeSameInstanceOf` copies; `instance` is unique
/// per object and stands in for Java reference identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Card {
    pub id: CardId,
    pub uuid: u32,
    pub instance: u32,
    pub upgraded: bool,
    pub times_upgraded: i32,
    pub cost: i32,
    pub cost_for_turn: i32,
    pub base_damage: i32,
    pub damage: i32,
    pub base_block: i32,
    pub block: i32,
    pub base_magic: i32,
    pub magic: i32,
    pub exhaust: bool,
    pub ethereal: bool,
    pub innate: bool,
    pub retain: bool,
    pub self_retain: bool,
    pub free_to_play_once: bool,
    pub exhaust_on_use_once: bool,
    pub dont_trigger_on_use: bool,
    pub purge_on_use: bool,
    pub in_autoplay: bool,
    pub energy_on_use: i32,
    pub ignore_energy_on_use: bool,
    pub misc: i32,
    /// Held by Bottled Flame/Lightning/Tornado: starts every combat in hand.
    #[serde(default)]
    pub bottled: bool,
    /// The card's pile- or state-counting base as its `applyPowers` set it when it was taken
    /// from the top of the draw pile (`PlayTopCardAction`): Stack's and Spirit Shield's block,
    /// Mind Blast's, Body Slam's and Blizzard's damage. The game counts only in `applyPowers`,
    /// not when the card is used, so a card Distilled Chaos queued plays with the count from
    /// before the cards queued ahead of it (and a reshuffle) changed it. Until the play ends
    /// (`exec_use_card`); the hand's cards count when they are played.
    #[serde(default)]
    pub applied_base: Option<i32>,
}

impl Card {
    pub fn new(id: CardId, uuid: u32, instance: u32) -> Card {
        let d = id.data();
        let mut c = Card {
            id,
            uuid,
            instance,
            upgraded: false,
            times_upgraded: 0,
            cost: d.cost,
            cost_for_turn: d.cost,
            base_damage: d.damage,
            damage: d.damage,
            base_block: d.block,
            block: d.block,
            base_magic: d.magic,
            magic: d.magic,
            exhaust: d.exhaust,
            ethereal: d.ethereal,
            innate: d.innate,
            retain: false,
            self_retain: d.self_retain,
            free_to_play_once: false,
            exhaust_on_use_once: false,
            dont_trigger_on_use: false,
            purge_on_use: false,
            in_autoplay: false,
            energy_on_use: 0,
            ignore_energy_on_use: false,
            misc: match id {
                CardId::RitualDagger => 15,
                CardId::GeneticAlgorithm => 1,
                _ => 0,
            },
            bottled: false,
            applied_base: None,
        };
        if id == CardId::RitualDagger {
            c.base_damage = 15;
            c.damage = 15;
        }
        if id == CardId::GeneticAlgorithm {
            c.base_block = 1;
            c.block = 1;
        }
        c
    }

    pub fn data(&self) -> &'static CardData {
        self.id.data()
    }

    pub fn card_type(&self) -> CardType {
        self.data().card_type
    }

    pub fn target(&self) -> CardTarget {
        // Trip+ and Blind+ hit every enemy.
        if self.upgraded && matches!(self.id, CardId::Trip | CardId::Blind) {
            return CardTarget::AllEnemy;
        }
        self.data().target
    }

    pub fn name(&self) -> String {
        let d = self.data();
        match (self.id, self.times_upgraded) {
            (_, 0) if !self.upgraded => d.name.to_string(),
            (CardId::SearingBlow, n) => format!("{}+{}", d.name, n),
            _ => format!("{}+", d.name),
        }
    }

    pub fn can_upgrade(&self) -> bool {
        match self.card_type() {
            CardType::Curse | CardType::Status => false,
            _ => self.id == CardId::SearingBlow || !self.upgraded,
        }
    }

    fn upgrade_name(&mut self) {
        self.times_upgraded += 1;
        self.upgraded = true;
    }

    fn upgrade_damage(&mut self, amount: i32) {
        self.base_damage += amount;
    }

    fn upgrade_block(&mut self, amount: i32) {
        self.base_block += amount;
    }

    fn upgrade_magic(&mut self, amount: i32) {
        self.base_magic += amount;
        self.magic = self.base_magic;
    }

    /// `upgradeBaseCost`.
    fn upgrade_base_cost(&mut self, new_base: i32) {
        let diff = self.cost_for_turn - self.cost;
        self.cost = new_base;
        if self.cost_for_turn > 0 {
            self.cost_for_turn = self.cost + diff;
        }
        if self.cost_for_turn < 0 {
            self.cost_for_turn = 0;
        }
    }

    /// Each card's `upgrade()`.
    pub fn upgrade(&mut self) {
        use CardId::*;
        if self.id == SearingBlow {
            self.upgrade_damage(4 + self.times_upgraded);
            self.times_upgraded += 1;
            self.upgraded = true;
            return;
        }
        if self.upgraded {
            return;
        }
        match self.id {
            Anger => self.upgrade_damage(2),
            Armaments => {}
            Barricade => self.upgrade_base_cost(2),
            Bash => {
                self.upgrade_damage(2);
                self.upgrade_magic(1);
            }
            BattleTrance => self.upgrade_magic(1),
            Berserk => self.upgrade_magic(-1),
            BloodForBlood => {
                if self.cost < 4 {
                    self.upgrade_base_cost(self.cost - 1);
                    if self.cost < 0 {
                        self.cost = 0;
                    }
                } else {
                    self.upgrade_base_cost(3);
                }
                self.upgrade_damage(4);
            }
            Bloodletting => self.upgrade_magic(1),
            Bludgeon => self.upgrade_damage(10),
            BodySlam => self.upgrade_base_cost(0),
            Brutality => self.innate = true,
            BurningPact => self.upgrade_magic(1),
            Carnage => self.upgrade_damage(8),
            Clash => self.upgrade_damage(4),
            Cleave => self.upgrade_damage(3),
            Clothesline => {
                self.upgrade_damage(2);
                self.upgrade_magic(1);
            }
            Combust => self.upgrade_magic(2),
            Corruption => self.upgrade_base_cost(2),
            DarkEmbrace => self.upgrade_base_cost(1),
            DefendR => self.upgrade_block(3),
            DemonForm => self.upgrade_magic(1),
            Disarm => self.upgrade_magic(1),
            DoubleTap => self.upgrade_magic(1),
            Dropkick => self.upgrade_damage(3),
            DualWield => self.upgrade_magic(1),
            Entrench => self.upgrade_base_cost(1),
            Evolve => self.upgrade_magic(1),
            Exhume => self.upgrade_base_cost(0),
            Feed => {
                self.upgrade_damage(2);
                self.upgrade_magic(1);
            }
            FeelNoPain => self.upgrade_magic(1),
            FiendFire => self.upgrade_damage(3),
            FireBreathing => self.upgrade_magic(4),
            FlameBarrier => {
                self.upgrade_block(4);
                self.upgrade_magic(2);
            }
            Flex => self.upgrade_magic(2),
            GhostlyArmor => self.upgrade_block(3),
            Havoc => self.upgrade_base_cost(0),
            Headbutt => self.upgrade_damage(3),
            HeavyBlade => self.upgrade_magic(2),
            Hemokinesis => self.upgrade_damage(5),
            Immolate => self.upgrade_damage(7),
            Impervious => self.upgrade_block(10),
            InfernalBlade => self.upgrade_base_cost(0),
            Inflame => self.upgrade_magic(1),
            Intimidate => self.upgrade_magic(1),
            IronWave => {
                self.upgrade_damage(2);
                self.upgrade_block(2);
            }
            Juggernaut => self.upgrade_magic(2),
            LimitBreak => self.exhaust = false,
            Metallicize => self.upgrade_magic(1),
            Offering => self.upgrade_magic(2),
            PerfectedStrike => self.upgrade_magic(1),
            PommelStrike => {
                self.upgrade_damage(1);
                self.upgrade_magic(1);
            }
            PowerThrough => self.upgrade_block(5),
            Pummel => self.upgrade_magic(1),
            Rage => self.upgrade_magic(2),
            Rampage => self.upgrade_magic(3),
            Reaper => self.upgrade_damage(1),
            RecklessCharge => self.upgrade_damage(3),
            Rupture => self.upgrade_magic(1),
            SearingBlow => unreachable!(),
            SecondWind => self.upgrade_block(2),
            SeeingRed => self.upgrade_base_cost(0),
            Sentinel => self.upgrade_block(3),
            SeverSoul => self.upgrade_damage(6),
            Shockwave => self.upgrade_magic(2),
            ShrugItOff => self.upgrade_block(3),
            SpotWeakness => self.upgrade_magic(1),
            StrikeR => self.upgrade_damage(3),
            SwordBoomerang => self.upgrade_magic(1),
            ThunderClap => self.upgrade_damage(3),
            TrueGrit => self.upgrade_block(2),
            TwinStrike => self.upgrade_damage(2),
            Uppercut => self.upgrade_magic(1),
            Warcry => self.upgrade_magic(1),
            Whirlwind => self.upgrade_damage(3),
            WildStrike => self.upgrade_damage(5),
            Burn => self.upgrade_magic(2),
            // Colorless.
            DarkShackles => self.upgrade_magic(6),
            SadisticNature | BandageUp => self.upgrade_magic(2),
            PanicButton => self.upgrade_block(10),
            Trip | Blind | Forethought | Enlightenment | Transmutation => {}
            DramaticEntrance => self.upgrade_damage(4),
            Impatience | DeepBreath | Violence | Panacea | MasterOfStrategy | JackOfAllTrades => self.upgrade_magic(1),
            TheBomb => self.upgrade_magic(10),
            SecretTechnique | SecretWeapon | Discovery | ThinkingAhead => self.exhaust = false,
            Panache => self.upgrade_magic(4),
            Apotheosis | Mayhem | Magnetism | MindBlast => self.upgrade_base_cost(1),
            HandOfGreed => {
                self.upgrade_damage(5);
                self.upgrade_magic(5);
            }
            FlashOfSteel | SwiftStrike => self.upgrade_damage(3),
            Purity | Chrysalis | Metamorphosis => self.upgrade_magic(2),
            Finesse => self.upgrade_block(2),
            GoodInstincts => self.upgrade_block(3),
            Madness => self.upgrade_base_cost(0),
            JAX => self.upgrade_magic(1),
            Apparition => self.ethereal = false,
            Bite => {
                self.upgrade_damage(1);
                self.upgrade_magic(1);
            }
            RitualDagger => self.upgrade_magic(2),
            // The Silent.
            StrikeG | Slice | DaggerThrow | FlyingKnee | Bane | HeelHook | Skewer => self.upgrade_damage(3),
            DefendG | Deflect | Backflip | Blur | Survivor => self.upgrade_block(3),
            DodgeAndRoll | EscapePlan => self.upgrade_block(2),
            Shiv | DaggerSpray | EndlessAgony | Eviscerate | Finisher | Flechettes => self.upgrade_damage(2),
            AllOutAttack | Backstab | DieDieDie | GlassKnife | MasterfulStab | QuickSlash | SneakyStrike | Unload => self.upgrade_damage(4),
            RiddleWithHoles => self.upgrade_damage(1),
            Predator => self.upgrade_damage(5),
            GrandFinale => self.upgrade_damage(10),
            Dash => {
                self.upgrade_damage(3);
                self.upgrade_block(3);
            }
            Neutralize => {
                self.upgrade_damage(1);
                self.upgrade_magic(1);
            }
            PoisonedStab => {
                self.upgrade_damage(2);
                self.upgrade_magic(1);
            }
            SuckerPunch => {
                self.upgrade_damage(2);
                self.upgrade_magic(1);
            }
            LegSweep => {
                self.upgrade_magic(1);
                self.upgrade_block(3);
            }
            AThousandCuts | Acrobatics | BladeDance | BouncingFlask | Burst | CloakAndDagger | Expertise | Footwork | NoxiousFumes | Prepared
            | Reflex | Tactician | WellLaidPlans | WraithForm => self.upgrade_magic(1),
            Accuracy | Caltrops | Choke | DeadlyPoison | PiercingWail => self.upgrade_magic(2),
            CorpseExplosion | CripplingPoison => self.upgrade_magic(3),
            Concentrate => self.upgrade_magic(-1),
            Alchemize | Distraction | PhantasmalKiller | Setup | Terror | ToolsOfTheTrade => self.upgrade_base_cost(0),
            Envenom => self.upgrade_base_cost(1),
            BulletTime | Nightmare => self.upgrade_base_cost(2),
            AfterImage | InfiniteBlades => self.innate = true,
            CalculatedGamble => self.exhaust = false,
            Adrenaline | Catalyst | Doppelganger | Malaise | Outmaneuver | StormOfSteel => {}
            // The Defect.
            StrikeB | BallLightning | ColdSnap | CompileDriver | Rebound => self.upgrade_damage(3),
            DefendB | BootSequence | ConserveBattery | Equilibrium | Glacier | Leap | Stack => self.upgrade_block(3),
            AllForOne | CoreSurge | DoomAndGloom | Melter => self.upgrade_damage(4),
            Barrage | Claw | RipAndTear => self.upgrade_damage(2),
            Hyperbeam | Sunder => self.upgrade_damage(8),
            MeteorStrike => self.upgrade_damage(6),
            Streamline => self.upgrade_damage(5),
            ThunderStrike => self.upgrade_damage(2),
            SweepingBeam => self.upgrade_damage(3),
            AutoShields | ForceField => self.upgrade_block(4),
            ReinforcedBody | SteamBarrier => self.upgrade_block(2),
            Hologram => {
                self.upgrade_block(2);
                self.exhaust = false;
            }
            BeamCell | FTL | GoForTheEyes => {
                self.upgrade_damage(1);
                self.upgrade_magic(1);
            }
            LockOn | Scrape => {
                self.upgrade_damage(3);
                self.upgrade_magic(1);
            }
            Amplify | BiasedCognition | Blizzard | Buffer | Capacitor | Chaos | Consume | Coolheaded | Defragment | Electrodynamics
            | GeneticAlgorithm | Heatsinks | Loop | Overclock | Reprogram | Seek | Skim | StaticDischarge | Turbo => self.upgrade_magic(1),
            Reboot => self.upgrade_magic(2),
            SelfRepair => self.upgrade_magic(3),
            Aggregate => self.upgrade_magic(-1),
            CreativeAI => self.upgrade_base_cost(2),
            Fusion => self.upgrade_base_cost(1),
            DoubleEnergy | Dualcast | Recursion | Recycle | WhiteNoise | Zap => self.upgrade_base_cost(0),
            Chill | HelloWorld | MachineLearning | Storm => self.innate = true,
            EchoForm => self.ethereal = false,
            Rainbow => self.exhaust = false,
            Darkness | Fission | MultiCast | Tempest => {}
            // The Watcher.
            StrikeP | FearNoEvil | LessonLearned | Wallop | BowlingBash | Consecrate => self.upgrade_damage(3),
            Brilliance | CarveReality | Conclude | FollowUp => self.upgrade_damage(4),
            EmptyFist | ReachHeaven | WheelKick => self.upgrade_damage(5),
            FlurryOfBlows | FlyingSleeves | Weave => self.upgrade_damage(2),
            SandsOfTime => self.upgrade_damage(6),
            SignatureMove => self.upgrade_damage(10),
            DefendP | DeceiveReality | EmptyBody | Sanctity | Swivel => self.upgrade_block(3),
            Evaluate | Protect | Vigilance => self.upgrade_block(4),
            CrushJoints | CutThroughFate | SashWhip | TalkToTheHand => {
                self.upgrade_damage(2);
                self.upgrade_magic(1);
            }
            JustLucky => {
                self.upgrade_damage(1);
                self.upgrade_block(1);
                self.upgrade_magic(1);
            }
            Ragnarok => {
                self.upgrade_damage(1);
                self.upgrade_magic(1);
            }
            WindmillStrike => {
                self.upgrade_damage(3);
                self.upgrade_magic(1);
            }
            Perseverance => {
                self.upgrade_block(2);
                self.upgrade_magic(1);
            }
            ThirdEye => {
                self.upgrade_block(2);
                self.upgrade_magic(2);
            }
            Wish => {
                self.upgrade_damage(1);
                self.upgrade_magic(5);
                self.upgrade_block(2);
            }
            // Halt: its Wrath number is its block plus 6, 10 once upgraded.
            Halt => {
                self.upgrade_block(1);
                self.base_magic = self.base_block + 10;
                self.magic = self.base_magic;
            }
            DeusExMachina | Devotion | EmptyMind | Fasting | Foresight | InnerPeace | Meditate | Nirvana | Pray | Prostrate | SimmeringFury
            | SpiritShield | Tantrum | WaveOfTheHand => self.upgrade_magic(1),
            Indignation | LikeWater | MentalFortress => self.upgrade_magic(2),
            PressurePoints | WreathOfFlame => self.upgrade_magic(3),
            Judgement => self.upgrade_magic(10),
            Crescendo | MasterReality | Rushdown | Scrawl | Tranquility => self.upgrade_base_cost(0),
            Eruption | Study => self.upgrade_base_cost(1),
            Vault => self.upgrade_base_cost(2),
            Omniscience => self.upgrade_base_cost(3),
            Alpha | BattleHymn | Establishment => self.innate = true,
            Blasphemy | Worship => self.self_retain = true,
            DevaForm => self.ethereal = false,
            Collect | ConjureBlade | ForeignInfluence => {}
            Beta => self.upgrade_base_cost(1),
            Expunger => self.upgrade_damage(6),
            Insight | BecomeAlmighty => self.upgrade_magic(1),
            Miracle => {}
            Omega => self.upgrade_magic(10),
            Safety => self.upgrade_block(4),
            Smite => self.upgrade_damage(4),
            ThroughViolence => self.upgrade_damage(10),
            FameAndFortune => self.upgrade_magic(5),
            LiveForever => self.upgrade_magic(2),
            // ChooseCalm and ChooseWrath's upgrade() does nothing, not even the name.
            ChooseCalm | ChooseWrath => return,
            // Statuses whose upgrade() does nothing.
            Dazed | Slimed | Void | Wound => return,
            // Colorless cards and curses are not ported yet (see `CardId::is_ported`).
            _ => return,
        }
        self.upgrade_name();
    }

    /// `resetAttributes` (end of turn).
    pub fn reset_attributes(&mut self) {
        self.block = self.base_block;
        self.damage = self.base_damage;
        self.magic = self.base_magic;
        self.cost_for_turn = self.cost;
    }

    /// `setCostForTurn`.
    pub fn set_cost_for_turn(&mut self, amount: i32) {
        if self.cost_for_turn >= 0 {
            self.cost_for_turn = amount.max(0);
        }
    }

    /// `modifyCostForCombat`.
    pub fn modify_cost_for_combat(&mut self, amount: i32) {
        if self.cost_for_turn > 0 {
            self.cost_for_turn = (self.cost_for_turn + amount).max(0);
            self.cost = self.cost_for_turn;
        } else if self.cost >= 0 {
            self.cost = (self.cost + amount).max(0);
            self.cost_for_turn = 0;
        }
    }

    /// `updateCost` (Blood for Blood, Slimed are the only statuses allowed).
    pub fn update_cost(&mut self, amount: i32) {
        let t = self.card_type();
        if t == CardType::Curse || (t == CardType::Status && self.id != CardId::Slimed) {
            return;
        }
        let diff = self.cost - self.cost_for_turn;
        let tmp = (self.cost + amount).max(0);
        if tmp != self.cost {
            self.cost = tmp;
            self.cost_for_turn = (self.cost - diff).max(0);
        }
    }
}
