//! Monster state (`AbstractMonster`). Per-monster AI lives in the act modules.

pub mod beyond;
pub mod city;
pub mod ending;
pub mod exordium;

use serde::{Deserialize, Serialize};

use crate::powers::{Power, PowerId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MonsterId {
    AcidSlimeL,
    AcidSlimeM,
    AcidSlimeS,
    Cultist,
    FungiBeast,
    GremlinFat,
    GremlinNob,
    GremlinThief,
    GremlinTsundere,
    GremlinWarrior,
    GremlinWizard,
    Hexaghost,
    JawWorm,
    Lagavulin,
    Looter,
    LouseDefensive,
    LouseNormal,
    Sentry,
    SlaverBlue,
    SlaverRed,
    SlimeBoss,
    SpikeSlimeL,
    SpikeSlimeM,
    SpikeSlimeS,
    TheGuardian,
    // Act 2.
    SphericGuardian,
    Chosen,
    ShelledParasite,
    Byrd,
    Mugger,
    SnakePlant,
    Snecko,
    Centurion,
    Healer,
    Taskmaster,
    GremlinLeader,
    BookOfStabbing,
    Champ,
    TheCollector,
    TorchHead,
    BronzeAutomaton,
    BronzeOrb,
    BanditBear,
    BanditLeader,
    BanditPointy,
    // Act 3.
    Darkling,
    OrbWalker,
    Spiker,
    Repulsor,
    Exploder,
    SpireGrowth,
    Transient,
    Maw,
    WrithingMass,
    GiantHead,
    Nemesis,
    Reptomancer,
    SnakeDagger,
    TimeEater,
    Donu,
    Deca,
    AwakenedOne,
    // Act 4.
    SpireShield,
    SpireSpear,
    CorruptHeart,
}

impl MonsterId {
    /// Every monster, in declaration order (the order the encodings number them).
    pub fn all() -> impl Iterator<Item = MonsterId> {
        // SAFETY: a field-less enum with contiguous discriminants 0..=CorruptHeart; transmute checks
        // at compile time that it is one byte.
        (0..=MonsterId::CorruptHeart as u8).map(|i| unsafe { std::mem::transmute::<u8, MonsterId>(i) })
    }
}

impl MonsterId {
    pub fn java_id(self) -> &'static str {
        use MonsterId::*;
        match self {
            AcidSlimeL => "AcidSlime_L",
            AcidSlimeM => "AcidSlime_M",
            AcidSlimeS => "AcidSlime_S",
            Cultist => "Cultist",
            FungiBeast => "FungiBeast",
            GremlinFat => "GremlinFat",
            GremlinNob => "GremlinNob",
            GremlinThief => "GremlinThief",
            GremlinTsundere => "GremlinTsundere",
            GremlinWarrior => "GremlinWarrior",
            GremlinWizard => "GremlinWizard",
            Hexaghost => "Hexaghost",
            JawWorm => "JawWorm",
            Lagavulin => "Lagavulin",
            Looter => "Looter",
            LouseDefensive => "FuzzyLouseDefensive",
            LouseNormal => "FuzzyLouseNormal",
            Sentry => "Sentry",
            SlaverBlue => "SlaverBlue",
            SlaverRed => "SlaverRed",
            SlimeBoss => "SlimeBoss",
            SpikeSlimeL => "SpikeSlime_L",
            SpikeSlimeM => "SpikeSlime_M",
            SpikeSlimeS => "SpikeSlime_S",
            TheGuardian => "TheGuardian",
            SphericGuardian => "SphericGuardian",
            Chosen => "Chosen",
            ShelledParasite => "Shelled Parasite",
            Byrd => "Byrd",
            Mugger => "Mugger",
            SnakePlant => "SnakePlant",
            Snecko => "Snecko",
            Centurion => "Centurion",
            Healer => "Healer",
            Taskmaster => "SlaverBoss",
            GremlinLeader => "GremlinLeader",
            BookOfStabbing => "BookOfStabbing",
            Champ => "Champ",
            TheCollector => "TheCollector",
            TorchHead => "TorchHead",
            BronzeAutomaton => "BronzeAutomaton",
            BronzeOrb => "BronzeOrb",
            BanditBear => "BanditBear",
            BanditLeader => "BanditLeader",
            BanditPointy => "BanditChild",
            Darkling => "Darkling",
            OrbWalker => "Orb Walker",
            Spiker => "Spiker",
            Repulsor => "Repulsor",
            Exploder => "Exploder",
            SpireGrowth => "Serpent",
            Transient => "Transient",
            Maw => "Maw",
            WrithingMass => "WrithingMass",
            GiantHead => "GiantHead",
            Nemesis => "Nemesis",
            Reptomancer => "Reptomancer",
            SnakeDagger => "Dagger",
            TimeEater => "TimeEater",
            Donu => "Donu",
            Deca => "Deca",
            AwakenedOne => "AwakenedOne",
            SpireShield => "SpireShield",
            SpireSpear => "SpireSpear",
            CorruptHeart => "CorruptHeart",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Intent {
    Attack,
    AttackBuff,
    AttackDebuff,
    AttackDefend,
    Buff,
    Debuff,
    StrongDebuff,
    Debug,
    Defend,
    DefendDebuff,
    DefendBuff,
    Escape,
    Magic,
    None,
    Sleep,
    Stun,
    Unknown,
}

impl Intent {
    pub fn is_attack(self) -> bool {
        matches!(self, Intent::Attack | Intent::AttackBuff | Intent::AttackDebuff | Intent::AttackDefend)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnemyType {
    Normal,
    Elite,
    Boss,
}

/// `EnemyMoveInfo`. `next_move == NO_MOVE` stands for the Java byte `-1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnemyMove {
    pub next_move: u8,
    pub intent: Intent,
    pub base_damage: i32,
    pub multiplier: i32,
    pub is_multi_damage: bool,
}

pub const NO_MOVE: u8 = u8::MAX;

impl EnemyMove {
    pub fn new(next_move: u8, intent: Intent, base_damage: i32, multiplier: i32, is_multi_damage: bool) -> EnemyMove {
        EnemyMove { next_move, intent, base_damage, multiplier, is_multi_damage }
    }

    pub fn simple(next_move: u8, intent: Intent) -> EnemyMove {
        EnemyMove::new(next_move, intent, -1, 0, false)
    }

    pub fn attack(next_move: u8, intent: Intent, base_damage: i32) -> EnemyMove {
        EnemyMove::new(next_move, intent, base_damage, 0, false)
    }
}

/// A monster's `DamageInfo` slot: the base, and the output last computed by `applyPowers`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DamageSlot {
    pub base: i32,
    pub output: i32,
}

impl DamageSlot {
    pub fn new(base: i32) -> DamageSlot {
        DamageSlot { base, output: base }
    }
}

/// Per-monster AI fields (the private fields of each `AbstractMonster` subclass).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Ai {
    AcidSlimeL { save_x: f32, split_triggered: bool },
    AcidSlimeM,
    AcidSlimeS,
    Cultist { first_move: bool, ritual: i32 },
    FungiBeast { str_amt: i32 },
    GremlinFat,
    GremlinNob { used_bellow: bool, can_vuln: bool },
    GremlinThief { thief_damage: i32 },
    GremlinTsundere { block_amt: i32 },
    GremlinWarrior,
    GremlinWizard { current_charge: i32 },
    Hexaghost {
        activated: bool,
        burn_upgraded: bool,
        orb_active_count: i32,
        str_amount: i32,
        sear_burn_count: i32,
        fire_tackle_count: i32,
        inferno_hits: i32,
        strengthen_block: i32,
    },
    JawWorm { first_move: bool, hard: bool, bellow_str: i32, bellow_block: i32, thrash_block: i32 },
    Lagavulin { asleep: bool, is_out: bool, is_out_triggered: bool, idle_count: i32, debuff_turn_count: i32, debuff: i32 },
    Looter { gold_amt: i32, slash_count: i32, stolen_gold: i32, escape_def: i32 },
    LouseDefensive { is_open: bool },
    LouseNormal { is_open: bool },
    Sentry { first_move: bool, dazed_amt: i32 },
    SlaverBlue { weak_amt: i32 },
    SlaverRed { used_entangle: bool, first_turn: bool, stab_dmg: i32 },
    SlimeBoss { first_turn: bool },
    SpikeSlimeL { save_x: f32, split_triggered: bool },
    SpikeSlimeM,
    SpikeSlimeS,
    TheGuardian {
        dmg_threshold: i32,
        dmg_threshold_increase: i32,
        dmg_taken: i32,
        is_open: bool,
        close_up_triggered: bool,
        thorns_damage: i32,
        vent_debuff: i32,
        defensive_block: i32,
        block_amount: i32,
        whirlwind_count: i32,
        whirlwind_damage: i32,
        twin_slam_damage: i32,
    },
    // Act 2.
    SphericGuardian { first_move: bool, second_move: bool },
    Chosen { first_turn: bool, used_hex: bool },
    ShelledParasite { first_move: bool },
    Byrd { first_move: bool, is_flying: bool, peck_count: i32, flight_amt: i32 },
    Mugger { gold_amt: i32, slash_count: i32, stolen_gold: i32, escape_def: i32 },
    SnakePlant,
    Snecko { first_turn: bool },
    Centurion { block_amount: i32, fury_hits: i32 },
    Healer { str_amt: i32, heal_amt: i32 },
    Taskmaster { wound_count: i32 },
    /// `gremlins` holds monster indices (the Java array of the three gremlin slots).
    GremlinLeader { str_amt: i32, block_amt: i32, gremlins: [Option<usize>; 3] },
    BookOfStabbing { stab_count: i32 },
    Champ {
        str_amt: i32,
        forge_amt: i32,
        block_amt: i32,
        num_turns: i32,
        forge_times: i32,
        forge_threshold: i32,
        threshold_reached: bool,
        first_turn: bool,
    },
    /// `slots` are `enemySlots` 1 and 2 (monster indices).
    TheCollector {
        str_amt: i32,
        block_amt: i32,
        mega_debuff_amt: i32,
        turns_taken: i32,
        ult_used: bool,
        initial_spawn: bool,
        slots: [Option<usize>; 2],
    },
    TorchHead,
    BronzeAutomaton { str_amt: i32, block_amt: i32, num_turns: i32, first_turn: bool },
    /// `stasis_card` is the card Stasis took (the Java power holds it).
    BronzeOrb { used_stasis: bool, count: i32, stasis_card: Option<crate::cards::Card> },
    BanditBear { con_reduction: i32 },
    BanditLeader { weak_amount: i32 },
    BanditPointy,
    // Act 3.
    Darkling { first_move: bool },
    OrbWalker,
    Spiker { starting_thorns: i32, thorns_count: i32 },
    Repulsor,
    Exploder { turn_count: i32 },
    SpireGrowth,
    Transient { count: i32, starting_death_dmg: i32 },
    Maw { roared: bool, turn_count: i32, str_up: i32, terrify_dur: i32 },
    WrithingMass { first_move: bool, used_mega_debuff: bool },
    GiantHead { count: i32, starting_death_dmg: i32 },
    Nemesis { scythe_cooldown: i32, first_move: bool, fire_dmg: i32 },
    /// `daggers` holds monster indices (the Java array of four dagger slots).
    Reptomancer { first_move: bool, daggers_per_spawn: i32, daggers: [Option<usize>; 4] },
    SnakeDagger { first_move: bool },
    TimeEater { used_haste: bool, first_turn: bool },
    Donu { is_attacking: bool },
    Deca { is_attacking: bool },
    AwakenedOne { form1: bool, first_turn: bool },
    // Act 4.
    SpireShield { move_count: i32 },
    SpireSpear { move_count: i32, skewer_count: i32 },
    CorruptHeart { is_first_move: bool, move_count: i32, buff_count: i32, blood_hit_count: i32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Monster {
    pub id: MonsterId,
    pub hp: i32,
    pub max_hp: i32,
    pub block: i32,
    pub powers: Vec<Power>,
    pub damage: Vec<DamageSlot>,
    pub mv: EnemyMove,
    /// `nextMove`: what `takeTurn` executes. Refreshed from `mv` by `createIntent`.
    pub next_move: u8,
    pub intent: Intent,
    pub intent_base_dmg: i32,
    pub intent_dmg: i32,
    pub intent_multi: i32,
    pub is_multi_dmg: bool,
    pub move_history: Vec<u8>,
    pub is_dying: bool,
    pub is_escaping: bool,
    pub half_dead: bool,
    pub escape_next: bool,
    pub cannot_escape: bool,
    pub gold: i32,
    pub enemy_type: EnemyType,
    /// The constructor's `offsetX`; only used to order spawned monsters.
    pub draw_x: f32,
    pub ai: Ai,
    /// `lastDamageTaken` (Wallop's block).
    #[serde(default)]
    pub last_damage_taken: i32,
}

impl Monster {
    pub fn new(id: MonsterId, max_hp: i32, draw_x: f32, ai: Ai) -> Monster {
        Monster {
            id,
            hp: max_hp,
            max_hp,
            block: 0,
            powers: Vec::new(),
            damage: Vec::new(),
            mv: EnemyMove::simple(NO_MOVE, Intent::Debug),
            next_move: NO_MOVE,
            intent: Intent::Debug,
            intent_base_dmg: -1,
            intent_dmg: -1,
            intent_multi: -1,
            is_multi_dmg: false,
            move_history: Vec::new(),
            is_dying: false,
            is_escaping: false,
            half_dead: false,
            escape_next: false,
            cannot_escape: false,
            gold: 0,
            enemy_type: EnemyType::Normal,
            draw_x,
            ai,
            last_damage_taken: 0,
        }
    }

    /// `setHp(min, max)`.
    pub fn set_hp(&mut self, rng: &mut sts_rng::StsRandom, min: i32, max: i32) {
        self.hp = rng.random_range(min, max);
        self.max_hp = self.hp;
    }

    pub fn power(&self, id: PowerId) -> Option<&Power> {
        self.powers.iter().find(|p| p.id == id)
    }

    pub fn power_amount(&self, id: PowerId) -> i32 {
        self.power(id).map_or(0, |p| p.amount)
    }

    pub fn has_power(&self, id: PowerId) -> bool {
        self.power(id).is_some()
    }

    pub fn is_dead_or_escaped(&self) -> bool {
        self.is_dying || self.half_dead || self.is_escaping
    }

    /// Alive for targeting (`!halfDead && !isDying && !isEscaping`).
    pub fn is_targetable(&self) -> bool {
        !self.half_dead && !self.is_dying && !self.is_escaping
    }

    pub fn set_move(&mut self, mv: EnemyMove) {
        if mv.next_move != NO_MOVE {
            self.move_history.push(mv.next_move);
        }
        self.mv = mv;
    }

    pub fn last_move(&self, mv: u8) -> bool {
        self.move_history.last() == Some(&mv)
    }

    pub fn last_move_before(&self, mv: u8) -> bool {
        let n = self.move_history.len();
        n >= 2 && self.move_history[n - 2] == mv
    }

    pub fn last_two_moves(&self, mv: u8) -> bool {
        let n = self.move_history.len();
        n >= 2 && self.move_history[n - 1] == mv && self.move_history[n - 2] == mv
    }
}
