//! Powers (`AbstractPower` subclasses). Hook behaviour lives in `combat::hooks`.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerId {
    // Shared.
    Strength,
    Dexterity,
    Vulnerable,
    Weakened,
    Frail,
    Artifact,
    Metallicize,
    Poison,
    Barricade,
    // Ironclad.
    Anger,
    Berserk,
    Brutality,
    Combust,
    Corruption,
    DarkEmbrace,
    DemonForm,
    DoubleTap,
    Entangled,
    Evolve,
    FeelNoPain,
    FireBreathing,
    FlameBarrier,
    Juggernaut,
    /// `LoseStrengthPower` (ID "Flex").
    Flex,
    NoDraw,
    Rage,
    Rupture,
    /// `GainStrengthPower` (ID "Shackled").
    Shackled,
    // From relics.
    Vigor,
    Thorns,
    PenNib,
    Buffer,
    IntangiblePlayer,
    PlatedArmor,
    NextTurnBlock,
    Confusion,
    // From potions.
    Regen,
    Duplication,
    /// `LoseDexterityPower` (ID "DexLoss").
    LoseDexterity,
    // From colorless cards.
    Sadistic,
    /// `NoBlockPower` (Panic Button).
    NoBlock,
    /// `TheBombPower`: each copy is its own power (`uid`), `hp_loss` holds the damage.
    TheBomb,
    /// `hp_loss` holds the damage; `amount` counts cards down from 5.
    Panache,
    Mayhem,
    Magnetism,
    // Act 1 monsters.
    Angry,
    CurlUp,
    ModeShift,
    Ritual,
    SharpHide,
    Split,
    SporeCloud,
    Thievery,
    Minion,
    // Act 2 monsters.
    Hex,
    /// `base` holds `storedAmount`.
    Flight,
    /// `base` holds `basePower`.
    Malleable,
    PainfulStabs,
    /// The card it holds lives in the Bronze Orb's AI state.
    Stasis,
    // Act 3 monsters.
    Explosive,
    Fading,
    Shifting,
    GenericStrengthUp,
    /// On the player; `base` holds the Spire Growth that applied it.
    Constricted,
    /// `ReactivePower` (ID "Compulsive").
    Reactive,
    Slow,
    /// The monsters' `IntangiblePower` (`just_applied` as in the game).
    Intangible,
    /// `RegrowPower` (ID "Life Link").
    Regrow,
    TimeWarp,
    DrawReduction,
    /// `RegenerateMonsterPower`.
    Regenerate,
    Curiosity,
    Unawakened,
    // Act 4.
    /// On the player while Spire Shield and Spear are both up.
    Surrounded,
    BackAttack,
    /// `base` holds `maxAmt`.
    Invincible,
    BeatOfDeath,
    // The Silent.
    /// `ThousandCutsPower`.
    ThousandCuts,
    Accuracy,
    AfterImage,
    Blur,
    Burst,
    /// `ChokePower` (ID "Choked").
    Choked,
    CorpseExplosion,
    Envenom,
    Energized,
    InfiniteBlades,
    NoxiousFumes,
    Phantasmal,
    DoubleDamage,
    /// `DrawCardNextTurnPower` (ID "Draw Card").
    DrawCardNextTurn,
    ToolsOfTheTrade,
    /// `RetainCardPower` (ID "Retain Cards").
    RetainCards,
    /// `WraithFormPower` (ID "Wraith Form v2").
    WraithForm,
    /// `NightmarePower` (ID "Night Terror"): each copy is its own power (`uid`); the card it
    /// makes lives in `Combat::night_terror_cards`.
    NightTerror,
    // The Defect.
    Focus,
    /// `LockOnPower` (ID "Lockon"): orb damage x1.5 on this monster.
    LockOn,
    /// `ElectroPower` (Electrodynamics): Lightning hits every enemy.
    Electro,
    Loop,
    Storm,
    /// `StaticDischargePower` (ID "StaticDischarge").
    StaticDischarge,
    /// `HeatsinkPower` (ID "Heatsink").
    Heatsink,
    /// `HelloPower` (ID "Hello").
    Hello,
    CreativeAI,
    /// `EchoPower` (ID "Echo Form"); `base` counts the cards doubled this turn.
    EchoForm,
    /// `BiasPower` (ID "Bias").
    Bias,
    /// `DrawPower` (ID "Draw", Machine Learning).
    Draw,
    /// `RepairPower` (ID "Repair", Self Repair).
    Repair,
    Amplify,
    Equilibrium,
    /// `ReboundPower`; `just_applied` is `justEvoked`.
    Rebound,
    EnergizedBlue,
    // The Watcher.
    /// `MantraPower`: at 10 the player enters Divinity.
    Mantra,
    BattleHymn,
    /// `BlockReturnPower` (Talk to the Hand), on a monster.
    BlockReturn,
    /// `MentalFortressPower` (ID "Controlled").
    MentalFortress,
    /// `DevaPower` (ID "DevaForm"); `base` holds `energyGainAmount`.
    DevaForm,
    /// `DevotionPower` (ID "DevotionPower").
    Devotion,
    /// `EndTurnDeathPower` (Blasphemy).
    EndTurnDeath,
    /// `EnergyDownPower` (Fasting).
    EnergyDown,
    /// `EstablishmentPower`.
    Establishment,
    /// `FreeAttackPower` (Swivel): the next attack costs nothing.
    FreeAttack,
    LikeWater,
    MasterReality,
    Nirvana,
    /// `OmegaPower`.
    Omega,
    /// `MarkPower` (ID "PathToVictoryPower", Pressure Points), on a monster.
    Mark,
    Study,
    WaveOfTheHand,
    /// `ForesightPower` (ID "WireheadingPower").
    Foresight,
    WrathNextTurn,
    /// `RushdownPower` (ID "Adaptation").
    Rushdown,
    /// `CollectPower` (ID "Collect").
    Collect,
}

impl PowerId {
    /// The last power, for the encodings' vocabulary size.
    pub const LAST: PowerId = PowerId::Collect;
}

impl PowerId {
    /// Every power, in declaration order (the order the encodings number them).
    pub fn all() -> impl Iterator<Item = PowerId> {
        // SAFETY: a field-less enum with contiguous discriminants 0..=LAST; transmute checks at
        // compile time that it is one byte.
        (0..=PowerId::LAST as u8).map(|i| unsafe { std::mem::transmute::<u8, PowerId>(i) })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerType {
    Buff,
    Debuff,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Power {
    pub id: PowerId,
    pub amount: i32,
    /// Vulnerable/Weak/Frail `justApplied`.
    pub just_applied: bool,
    /// Ritual `skipFirst`.
    pub skip_first: bool,
    /// Curl Up `triggered`.
    pub triggered: bool,
    /// Ritual `onPlayer`.
    pub on_player: bool,
    /// Combust `hpLoss`; The Bomb's and Panache's damage.
    pub hp_loss: i32,
    /// Distinguishes copies of a power that does not stack (The Bomb).
    #[serde(default)]
    pub uid: u32,
    /// The amount a power resets to (Flight's `storedAmount`, Malleable's `basePower`); Echo Form's
    /// cards doubled this turn.
    #[serde(default)]
    pub base: i32,
}

impl Power {
    pub fn new(id: PowerId, amount: i32) -> Power {
        let amount = match id {
            PowerId::Strength | PowerId::Dexterity | PowerId::Shackled | PowerId::Focus => amount.clamp(-999, 999),
            PowerId::Poison => amount.min(9999),
            PowerId::Energized | PowerId::EnergizedBlue | PowerId::Collect => amount.min(999),
            _ => amount,
        };
        // Echo Form's base counts the cards it doubled this turn (`cardsDoubledThisTurn`, 0 when
        // made); the others' base is the amount they reset to.
        let base = if id == PowerId::EchoForm { 0 } else { amount };
        Power { id, amount, just_applied: false, skip_first: true, triggered: false, on_player: false, hp_loss: 0, uid: 0, base }
    }

    /// Powers constructed without an amount (`amount = -1`).
    pub fn flag(id: PowerId) -> Power {
        Power::new(id, -1)
    }

    pub fn vulnerable(amount: i32, is_source_monster: bool, turn_has_ended: bool) -> Power {
        let mut p = Power::new(PowerId::Vulnerable, amount);
        p.just_applied = turn_has_ended && is_source_monster;
        p
    }

    pub fn weak(amount: i32, is_source_monster: bool) -> Power {
        let mut p = Power::new(PowerId::Weakened, amount);
        p.just_applied = is_source_monster;
        p
    }

    pub fn frail(amount: i32, is_source_monster: bool) -> Power {
        let mut p = Power::new(PowerId::Frail, amount);
        p.just_applied = is_source_monster;
        p
    }

    pub fn ritual(amount: i32, player_controlled: bool) -> Power {
        let mut p = Power::new(PowerId::Ritual, amount);
        p.on_player = player_controlled;
        p
    }

    pub fn intangible(amount: i32) -> Power {
        let mut p = Power::new(PowerId::Intangible, amount);
        p.just_applied = true;
        p
    }

    pub fn draw_reduction(amount: i32) -> Power {
        let mut p = Power::new(PowerId::DrawReduction, amount);
        p.just_applied = true;
        p
    }

    pub fn combust(hp_loss: i32, damage: i32) -> Power {
        let mut p = Power::new(PowerId::Combust, damage);
        p.hp_loss = hp_loss;
        p
    }

    pub fn priority(&self) -> i32 {
        match self.id {
            PowerId::Frail => 10,
            PowerId::Weakened => 99,
            PowerId::PenNib => 6,
            PowerId::IntangiblePlayer => 75,
            PowerId::Confusion => 0,
            PowerId::Flight => 50,
            PowerId::Reactive => 50,
            PowerId::Intangible => 75,
            PowerId::Constricted => 105,
            PowerId::Invincible => 99,
            PowerId::DoubleDamage => 6,
            PowerId::DrawCardNextTurn => 20,
            PowerId::ToolsOfTheTrade => 25,
            PowerId::Establishment => 25,
            _ => 5,
        }
    }

    pub fn power_type(&self) -> PowerType {
        use PowerId::*;
        match self.id {
            // `updateDescription` flips these by sign.
            Strength | Dexterity | Focus => {
                if self.amount > 0 {
                    PowerType::Buff
                } else {
                    PowerType::Debuff
                }
            }
            Vulnerable | Weakened | Frail | Poison | Entangled | NoDraw | Flex | Shackled | Confusion | LoseDexterity | NoBlock | Hex | Constricted | Slow | DrawReduction
            | Choked | CorpseExplosion | WraithForm | LockOn | Bias | BlockReturn | EnergyDown | Mark => PowerType::Debuff,
            Draw if self.amount < 0 => PowerType::Debuff,
            _ => PowerType::Buff,
        }
    }

    /// `stackPower`. Returns true when the power asks to be removed (`addToTop(RemoveSpecificPowerAction)`).
    pub fn stack(&mut self, amount: i32) -> bool {
        use PowerId::*;
        match self.id {
            Strength | Dexterity | Shackled | Focus => {
                self.amount += amount;
                let remove = self.amount == 0;
                self.amount = self.amount.clamp(-999, 999);
                remove
            }
            Combust => {
                self.amount += amount;
                self.hp_loss += 1;
                false
            }
            PlatedArmor => {
                self.amount = (self.amount + amount).min(999);
                false
            }
            Panache => {
                self.hp_loss += amount;
                false
            }
            Energized | EnergizedBlue | LikeWater | Collect => {
                self.amount = (self.amount + amount).min(999);
                false
            }
            // DevaPower.stackPower: one more energy each turn on top of the growth.
            DevaForm => {
                self.amount += amount;
                self.base += 1;
                false
            }
            // Wraith Form's -1 is an amount, not "no amount": it stacks.
            WraithForm => {
                self.amount += amount;
                false
            }
            Malleable => {
                self.amount += amount;
                self.base += amount;
                false
            }
            _ => {
                if self.amount != -1 {
                    self.amount += amount;
                }
                false
            }
        }
    }

    /// `reducePower`. Returns true when the power asks to be removed.
    pub fn reduce(&mut self, amount: i32) -> bool {
        use PowerId::*;
        match self.id {
            Strength | Dexterity | Shackled | Focus | Draw => {
                self.amount -= amount;
                let remove = self.amount == 0;
                self.amount = self.amount.clamp(-999, 999);
                remove
            }
            _ => {
                self.amount = (self.amount - amount).max(0);
                false
            }
        }
    }

    pub fn java_id(&self) -> &'static str {
        use PowerId::*;
        match self.id {
            Strength => "Strength",
            Dexterity => "Dexterity",
            Vulnerable => "Vulnerable",
            Weakened => "Weakened",
            Frail => "Frail",
            Artifact => "Artifact",
            Metallicize => "Metallicize",
            Poison => "Poison",
            Barricade => "Barricade",
            Anger => "Anger",
            Berserk => "Berserk",
            Brutality => "Brutality",
            Combust => "Combust",
            Corruption => "Corruption",
            DarkEmbrace => "Dark Embrace",
            DemonForm => "Demon Form",
            DoubleTap => "Double Tap",
            Entangled => "Entangled",
            Evolve => "Evolve",
            FeelNoPain => "Feel No Pain",
            FireBreathing => "Fire Breathing",
            FlameBarrier => "Flame Barrier",
            Juggernaut => "Juggernaut",
            Flex => "Flex",
            NoDraw => "No Draw",
            Rage => "Rage",
            Rupture => "Rupture",
            Shackled => "Shackled",
            Vigor => "Vigor",
            Thorns => "Thorns",
            PenNib => "Pen Nib",
            Buffer => "Buffer",
            IntangiblePlayer => "IntangiblePlayer",
            PlatedArmor => "Plated Armor",
            NextTurnBlock => "Next Turn Block",
            Confusion => "Confusion",
            Regen => "Regeneration",
            Duplication => "DuplicationPower",
            LoseDexterity => "DexLoss",
            Sadistic => "Sadistic",
            NoBlock => "NoBlockPower",
            TheBomb => "TheBomb",
            Panache => "Panache",
            Mayhem => "Mayhem",
            Magnetism => "Magnetism",
            Angry => "Angry",
            CurlUp => "Curl Up",
            ModeShift => "Mode Shift",
            Ritual => "Ritual",
            SharpHide => "Sharp Hide",
            Split => "Split",
            SporeCloud => "Spore Cloud",
            Thievery => "Thievery",
            Minion => "Minion",
            Hex => "Hex",
            Flight => "Flight",
            Malleable => "Malleable",
            PainfulStabs => "Painful Stabs",
            Stasis => "Stasis",
            Explosive => "Explosive",
            Fading => "Fading",
            Shifting => "Shifting",
            GenericStrengthUp => "Generic Strength Up Power",
            Constricted => "Constricted",
            Reactive => "Compulsive",
            Slow => "Slow",
            Intangible => "Intangible",
            Regrow => "Life Link",
            TimeWarp => "Time Warp",
            DrawReduction => "Draw Reduction",
            Regenerate => "Regenerate",
            Curiosity => "Curiosity",
            Unawakened => "Unawakened",
            Surrounded => "Surrounded",
            BackAttack => "BackAttack",
            Invincible => "Invincible",
            BeatOfDeath => "BeatOfDeath",
            ThousandCuts => "Thousand Cuts",
            Accuracy => "Accuracy",
            AfterImage => "After Image",
            Blur => "Blur",
            Burst => "Burst",
            Choked => "Choked",
            CorpseExplosion => "CorpseExplosionPower",
            Envenom => "Envenom",
            Energized => "Energized",
            InfiniteBlades => "Infinite Blades",
            NoxiousFumes => "Noxious Fumes",
            Phantasmal => "Phantasmal",
            DoubleDamage => "Double Damage",
            DrawCardNextTurn => "Draw Card",
            ToolsOfTheTrade => "Tools Of The Trade",
            RetainCards => "Retain Cards",
            WraithForm => "Wraith Form v2",
            NightTerror => "Night Terror",
            Focus => "Focus",
            LockOn => "Lockon",
            Electro => "Electro",
            Loop => "Loop",
            Storm => "Storm",
            StaticDischarge => "StaticDischarge",
            Heatsink => "Heatsink",
            Hello => "Hello",
            CreativeAI => "Creative AI",
            EchoForm => "Echo Form",
            Bias => "Bias",
            Draw => "Draw",
            Repair => "Repair",
            Amplify => "Amplify",
            Equilibrium => "Equilibrium",
            Rebound => "Rebound",
            EnergizedBlue => "EnergizedBlue",
            Mantra => "Mantra",
            BattleHymn => "BattleHymn",
            BlockReturn => "BlockReturnPower",
            MentalFortress => "Controlled",
            DevaForm => "DevaForm",
            Devotion => "DevotionPower",
            EndTurnDeath => "EndTurnDeath",
            EnergyDown => "EnergyDownPower",
            Establishment => "EstablishmentPower",
            FreeAttack => "FreeAttackPower",
            LikeWater => "LikeWaterPower",
            MasterReality => "MasterRealityPower",
            Nirvana => "Nirvana",
            Omega => "OmegaPower",
            Mark => "PathToVictoryPower",
            Study => "Study",
            WaveOfTheHand => "WaveOfTheHandPower",
            Foresight => "WireheadingPower",
            WrathNextTurn => "WrathNextTurnPower",
            Rushdown => "Adaptation",
            Collect => "Collect",
        }
    }
}
