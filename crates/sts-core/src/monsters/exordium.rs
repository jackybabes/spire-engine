//! Act 1 (Exordium) monsters: constructors, `getMove`, `takeTurn`, and their overrides.

use sts_rng::StsRandom;

use super::{Ai, DamageSlot, EnemyMove, EnemyType, Intent, Monster, MonsterId};
use crate::action::{Action, Cr, DamageInfo, DamageSource, MonsterState};
use crate::cards::CardId;
use crate::combat::Combat;
use crate::powers::{Power, PowerId};

// ---- Constructors (RNG use matches the Java constructors) ----

fn with_slots(mut m: Monster, slots: &[i32]) -> Monster {
    m.damage = slots.iter().map(|&b| DamageSlot::new(b)).collect();
    m
}

pub fn acid_slime_l(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = acid_slime_l_split(asc, x, 69);
    if asc >= 7 {
        m.set_hp(rng, 68, 72);
    } else {
        m.set_hp(rng, 65, 69);
    }
    m
}

pub fn acid_slime_l_split(asc: i32, x: f32, hp: i32) -> Monster {
    let m = Monster::new(MonsterId::AcidSlimeL, hp, x, Ai::AcidSlimeL { save_x: x, split_triggered: false });
    let mut m = if asc >= 2 { with_slots(m, &[12, 18]) } else { with_slots(m, &[11, 16]) };
    m.powers.push(Power::flag(PowerId::Split));
    m
}

pub fn acid_slime_m(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = acid_slime_m_split(asc, x, 32);
    if asc >= 7 {
        m.set_hp(rng, 29, 34);
    } else {
        m.set_hp(rng, 28, 32);
    }
    m
}

pub fn acid_slime_m_split(asc: i32, x: f32, hp: i32) -> Monster {
    let m = Monster::new(MonsterId::AcidSlimeM, hp, x, Ai::AcidSlimeM);
    if asc >= 2 {
        with_slots(m, &[8, 12])
    } else {
        with_slots(m, &[7, 10])
    }
}

pub fn acid_slime_s(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::AcidSlimeS, 12, x, Ai::AcidSlimeS);
    if asc >= 7 {
        m.set_hp(rng, 9, 13);
    } else {
        m.set_hp(rng, 8, 12);
    }
    with_slots(m, &[if asc >= 2 { 4 } else { 3 }])
}

pub fn spike_slime_l(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = spike_slime_l_split(asc, x, 70);
    if asc >= 7 {
        m.set_hp(rng, 67, 73);
    } else {
        m.set_hp(rng, 64, 70);
    }
    m
}

pub fn spike_slime_l_split(asc: i32, x: f32, hp: i32) -> Monster {
    let m = Monster::new(MonsterId::SpikeSlimeL, hp, x, Ai::SpikeSlimeL { save_x: x, split_triggered: false });
    let mut m = with_slots(m, &[if asc >= 2 { 18 } else { 16 }]);
    m.powers.push(Power::flag(PowerId::Split));
    m
}

pub fn spike_slime_m(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = spike_slime_m_split(asc, x, 32);
    if asc >= 7 {
        m.set_hp(rng, 29, 34);
    } else {
        m.set_hp(rng, 28, 32);
    }
    m
}

pub fn spike_slime_m_split(asc: i32, x: f32, hp: i32) -> Monster {
    let m = Monster::new(MonsterId::SpikeSlimeM, hp, x, Ai::SpikeSlimeM);
    with_slots(m, &[if asc >= 2 { 10 } else { 8 }])
}

pub fn spike_slime_s(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::SpikeSlimeS, 14, x, Ai::SpikeSlimeS);
    if asc >= 7 {
        m.set_hp(rng, 11, 15);
    } else {
        m.set_hp(rng, 10, 14);
    }
    with_slots(m, &[if asc >= 2 { 6 } else { 5 }])
}

pub fn cultist(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::Cultist, 54, x, Ai::Cultist { first_move: true, ritual: if asc >= 2 { 4 } else { 3 } });
    if asc >= 7 {
        m.set_hp(rng, 50, 56);
    } else {
        m.set_hp(rng, 48, 54);
    }
    with_slots(m, &[6])
}

pub fn fungi_beast(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::FungiBeast, 28, x, Ai::FungiBeast { str_amt: if asc >= 2 { 4 } else { 3 } });
    if asc >= 7 {
        m.set_hp(rng, 24, 28);
    } else {
        m.set_hp(rng, 22, 28);
    }
    with_slots(m, &[6])
}

pub fn gremlin_fat(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::GremlinFat, 17, x, Ai::GremlinFat);
    if asc >= 7 {
        m.set_hp(rng, 14, 18);
    } else {
        m.set_hp(rng, 13, 17);
    }
    with_slots(m, &[if asc >= 2 { 5 } else { 4 }])
}

pub fn gremlin_thief(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let dmg = if asc >= 2 { 10 } else { 9 };
    let mut m = Monster::new(MonsterId::GremlinThief, 14, x, Ai::GremlinThief { thief_damage: dmg });
    if asc >= 7 {
        m.set_hp(rng, 11, 15);
    } else {
        m.set_hp(rng, 10, 14);
    }
    with_slots(m, &[dmg])
}

pub fn gremlin_tsundere(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let block = if asc >= 17 {
        11
    } else if asc >= 7 {
        8
    } else {
        7
    };
    let mut m = Monster::new(MonsterId::GremlinTsundere, 15, x, Ai::GremlinTsundere { block_amt: block });
    if asc >= 7 {
        m.set_hp(rng, 13, 17);
    } else {
        m.set_hp(rng, 12, 15);
    }
    with_slots(m, &[if asc >= 2 { 8 } else { 6 }])
}

pub fn gremlin_warrior(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::GremlinWarrior, 24, x, Ai::GremlinWarrior);
    if asc >= 7 {
        m.set_hp(rng, 21, 25);
    } else {
        m.set_hp(rng, 20, 24);
    }
    with_slots(m, &[if asc >= 2 { 5 } else { 4 }])
}

pub fn gremlin_wizard(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::GremlinWizard, 25, x - 35.0, Ai::GremlinWizard { current_charge: 1 });
    if asc >= 7 {
        m.set_hp(rng, 22, 26);
    } else {
        m.set_hp(rng, 21, 25);
    }
    with_slots(m, &[if asc >= 2 { 30 } else { 25 }])
}

pub fn gremlin_nob(rng: &mut StsRandom, asc: i32, x: f32, can_vuln: bool) -> Monster {
    let mut m = Monster::new(MonsterId::GremlinNob, 86, x, Ai::GremlinNob { used_bellow: false, can_vuln });
    m.enemy_type = EnemyType::Elite;
    if asc >= 8 {
        m.set_hp(rng, 85, 90);
    } else {
        m.set_hp(rng, 82, 86);
    }
    if asc >= 3 {
        with_slots(m, &[16, 8])
    } else {
        with_slots(m, &[14, 6])
    }
}

pub fn hexaghost(rng: &mut StsRandom, asc: i32) -> Monster {
    let (str_amount, sear_burn_count, fire_tackle, inferno) = if asc >= 19 {
        (3, 2, 6, 3)
    } else if asc >= 4 {
        (2, 1, 6, 3)
    } else {
        (2, 1, 5, 2)
    };
    let ai = Ai::Hexaghost {
        activated: false,
        burn_upgraded: false,
        orb_active_count: 0,
        str_amount,
        sear_burn_count,
        fire_tackle_count: 2,
        inferno_hits: 6,
        strengthen_block: 12,
    };
    let mut m = Monster::new(MonsterId::Hexaghost, 250, 0.0, ai);
    m.enemy_type = EnemyType::Boss;
    let hp = if asc >= 9 { 264 } else { 250 };
    m.set_hp(rng, hp, hp);
    with_slots(m, &[fire_tackle, 6, -1, inferno])
}

pub fn jaw_worm(rng: &mut StsRandom, asc: i32, x: f32, hard: bool) -> Monster {
    let (bellow_str, bellow_block, chomp) = if asc >= 17 {
        (5, 9, 12)
    } else if asc >= 2 {
        (4, 6, 12)
    } else {
        (3, 6, 11)
    };
    let ai = Ai::JawWorm { first_move: !hard, hard, bellow_str, bellow_block, thrash_block: 5 };
    let mut m = Monster::new(MonsterId::JawWorm, 44, x, ai);
    if asc >= 7 {
        m.set_hp(rng, 42, 46);
    } else {
        m.set_hp(rng, 40, 44);
    }
    with_slots(m, &[chomp, 7])
}

pub fn lagavulin(rng: &mut StsRandom, asc: i32, asleep: bool) -> Monster {
    let ai = Ai::Lagavulin {
        asleep,
        is_out: !asleep,
        is_out_triggered: !asleep,
        idle_count: 0,
        debuff_turn_count: 0,
        debuff: if asc >= 18 { -2 } else { -1 },
    };
    let mut m = Monster::new(MonsterId::Lagavulin, 111, 0.0, ai);
    m.enemy_type = EnemyType::Elite;
    if asc >= 8 {
        m.set_hp(rng, 112, 115);
    } else {
        m.set_hp(rng, 109, 111);
    }
    with_slots(m, &[if asc >= 3 { 20 } else { 18 }])
}

pub fn looter(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let ai = Ai::Looter { gold_amt: if asc >= 17 { 20 } else { 15 }, slash_count: 0, stolen_gold: 0, escape_def: 6 };
    let mut m = Monster::new(MonsterId::Looter, 48, x, ai);
    if asc >= 7 {
        m.set_hp(rng, 46, 50);
    } else {
        m.set_hp(rng, 44, 48);
    }
    if asc >= 2 {
        with_slots(m, &[11, 14])
    } else {
        with_slots(m, &[10, 12])
    }
}

pub fn louse_normal(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::LouseNormal, 15, x, Ai::LouseNormal { is_open: true });
    if asc >= 7 {
        m.set_hp(rng, 11, 16);
    } else {
        m.set_hp(rng, 10, 15);
    }
    let bite = if asc >= 2 { rng.random_range(6, 8) } else { rng.random_range(5, 7) };
    with_slots(m, &[bite])
}

pub fn louse_defensive(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::LouseDefensive, 17, x, Ai::LouseDefensive { is_open: true });
    if asc >= 7 {
        m.set_hp(rng, 12, 18);
    } else {
        m.set_hp(rng, 11, 17);
    }
    let bite = if asc >= 2 { rng.random_range(6, 8) } else { rng.random_range(5, 7) };
    with_slots(m, &[bite])
}

pub fn sentry(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::Sentry, 42, x, Ai::Sentry { first_move: true, dazed_amt: if asc >= 18 { 3 } else { 2 } });
    m.enemy_type = EnemyType::Elite;
    if asc >= 8 {
        m.set_hp(rng, 39, 45);
    } else {
        m.set_hp(rng, 38, 42);
    }
    with_slots(m, &[if asc >= 3 { 10 } else { 9 }])
}

pub fn slaver_blue(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::SlaverBlue, 50, x, Ai::SlaverBlue { weak_amt: 1 });
    if asc >= 7 {
        m.set_hp(rng, 48, 52);
    } else {
        m.set_hp(rng, 46, 50);
    }
    if asc >= 2 {
        with_slots(m, &[13, 8])
    } else {
        with_slots(m, &[12, 7])
    }
}

pub fn slaver_red(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let (stab, scrape) = if asc >= 2 { (14, 9) } else { (13, 8) };
    let mut m = Monster::new(MonsterId::SlaverRed, 50, x, Ai::SlaverRed { used_entangle: false, first_turn: true, stab_dmg: stab });
    if asc >= 7 {
        m.set_hp(rng, 48, 52);
    } else {
        m.set_hp(rng, 46, 50);
    }
    with_slots(m, &[stab, scrape])
}

pub fn slime_boss(rng: &mut StsRandom, asc: i32) -> Monster {
    let mut m = Monster::new(MonsterId::SlimeBoss, 140, 0.0, Ai::SlimeBoss { first_turn: true });
    m.enemy_type = EnemyType::Boss;
    let hp = if asc >= 9 { 150 } else { 140 };
    m.set_hp(rng, hp, hp);
    let mut m = if asc >= 4 { with_slots(m, &[10, 38]) } else { with_slots(m, &[9, 35]) };
    m.powers.push(Power::flag(PowerId::Split));
    m
}

pub fn the_guardian(rng: &mut StsRandom, asc: i32) -> Monster {
    let threshold = if asc >= 19 {
        40
    } else if asc >= 9 {
        35
    } else {
        30
    };
    let ai = Ai::TheGuardian {
        dmg_threshold: threshold,
        dmg_threshold_increase: 10,
        dmg_taken: 0,
        is_open: true,
        close_up_triggered: false,
        thorns_damage: 3,
        vent_debuff: 2,
        defensive_block: 20,
        block_amount: 9,
        whirlwind_count: 4,
        whirlwind_damage: 5,
        twin_slam_damage: 8,
    };
    let mut m = Monster::new(MonsterId::TheGuardian, 240, -50.0, ai);
    m.enemy_type = EnemyType::Boss;
    let hp = if asc >= 9 { 250 } else { 240 };
    m.set_hp(rng, hp, hp);
    if asc >= 4 {
        with_slots(m, &[36, 10, 5, 8])
    } else {
        with_slots(m, &[32, 9, 5, 8])
    }
}

// ---- Behaviour ----

const ATTACK: Intent = Intent::Attack;

impl Combat {
    pub(super) fn mattack(&mut self, m: usize, slot: usize) {
        self.push_bot(Action::Damage { target: Cr::Player, source: DamageSource::MonsterSlot { monster: m, slot }, steal_gold: 0 });
    }

    pub(super) fn mroll(&mut self, m: usize) {
        self.push_bot(Action::RollMove { monster: m });
    }

    pub(super) fn mset(&mut self, m: usize, mv: EnemyMove) {
        self.monsters[m].set_move(mv);
    }

    pub(super) fn base(&self, m: usize, slot: usize) -> i32 {
        self.monsters[m].damage[slot].base
    }

    pub(super) fn rb(&mut self) -> bool {
        self.rng.ai.random_boolean()
    }

    pub(super) fn rbc(&mut self, chance: f32) -> bool {
        self.rng.ai.random_boolean_chance(chance)
    }

    pub(super) fn debuff_player(&mut self, m: usize, power: Power) {
        self.bot_power(Cr::Player, Some(Cr::Monster(m)), power);
    }

    pub(super) fn buff_self(&mut self, m: usize, power: Power) {
        self.bot_power(Cr::Monster(m), Some(Cr::Monster(m)), power);
    }

    pub(super) fn temp_card_discard(&mut self, id: CardId, amount: i32) {
        let c = self.new_card(id);
        self.push_bot(Action::MakeTempCardInDiscard { card: c, amount });
    }

    /// `AbstractMonster.rollMove()`.
    pub(crate) fn roll_move(&mut self, m: usize) {
        let num = self.rng.ai.random(99);
        self.get_move(m, num);
    }

    /// `usePreBattleAction()`.
    pub(crate) fn use_pre_battle_action(&mut self, m: usize) {
        let asc = self.ascension;
        match self.monsters[m].ai.clone() {
            Ai::FungiBeast { .. } => self.buff_self(m, Power::new(PowerId::SporeCloud, 2)),
            Ai::GremlinWarrior => self.buff_self(m, Power::new(PowerId::Angry, if asc >= 17 { 2 } else { 1 })),
            Ai::JawWorm { hard: true, bellow_str, bellow_block, .. } => {
                self.buff_self(m, Power::new(PowerId::Strength, bellow_str));
                self.push_bot(Action::GainBlock { target: Cr::Monster(m), amount: bellow_block });
            }
            Ai::Lagavulin { asleep, .. } => {
                if asleep {
                    self.push_bot(Action::GainBlock { target: Cr::Monster(m), amount: 8 });
                    self.buff_self(m, Power::new(PowerId::Metallicize, 8));
                } else {
                    self.mset(m, EnemyMove::simple(1, Intent::StrongDebuff));
                }
            }
            Ai::Looter { gold_amt, .. } => self.buff_self(m, Power::new(PowerId::Thievery, gold_amt)),
            Ai::LouseNormal { .. } | Ai::LouseDefensive { .. } => {
                let amount = if asc >= 17 {
                    self.rng.monster_hp.random_range(9, 12)
                } else if asc >= 7 {
                    self.rng.monster_hp.random_range(4, 8)
                } else {
                    self.rng.monster_hp.random_range(3, 7)
                };
                self.buff_self(m, Power::new(PowerId::CurlUp, amount));
            }
            Ai::Sentry { .. } => self.buff_self(m, Power::new(PowerId::Artifact, 1)),
            Ai::TheGuardian { dmg_threshold, .. } => {
                self.buff_self(m, Power::new(PowerId::ModeShift, dmg_threshold));
                self.push_bot(Action::ChangeState { monster: m, state: MonsterState::ResetThreshold });
            }
            _ => self.city_pre_battle(m),
        }
    }

    /// `getMove(num)`.
    pub(crate) fn get_move(&mut self, m: usize, num: i32) {
        let asc = self.ascension;
        let mon = &self.monsters[m];
        let d0 = mon.damage.first().map_or(-1, |s| s.base);
        let d1 = mon.damage.get(1).map_or(-1, |s| s.base);
        let mv = match mon.ai.clone() {
            Ai::AcidSlimeL { .. } => {
                let wound = EnemyMove::attack(1, Intent::AttackDebuff, d0);
                let tackle = EnemyMove::attack(2, ATTACK, d1);
                let lick = EnemyMove::simple(4, Intent::Debuff);
                if asc >= 17 {
                    if num < 40 {
                        if self.monsters[m].last_two_moves(1) {
                            if self.rbc(0.6) { tackle } else { lick }
                        } else {
                            wound
                        }
                    } else if num < 70 {
                        if self.monsters[m].last_two_moves(2) {
                            if self.rbc(0.6) { wound } else { lick }
                        } else {
                            tackle
                        }
                    } else if self.monsters[m].last_move(4) {
                        if self.rbc(0.4) { wound } else { tackle }
                    } else {
                        lick
                    }
                } else if num < 30 {
                    if self.monsters[m].last_two_moves(1) {
                        if self.rb() { tackle } else { lick }
                    } else {
                        wound
                    }
                } else if num < 70 {
                    if self.monsters[m].last_move(2) {
                        if self.rbc(0.4) { wound } else { lick }
                    } else {
                        tackle
                    }
                } else if self.monsters[m].last_two_moves(4) {
                    if self.rbc(0.4) { wound } else { tackle }
                } else {
                    lick
                }
            }
            Ai::AcidSlimeM => {
                let wound = EnemyMove::attack(1, Intent::AttackDebuff, d0);
                let tackle = EnemyMove::attack(2, ATTACK, d1);
                let lick = EnemyMove::simple(4, Intent::Debuff);
                if asc >= 17 {
                    if num < 40 {
                        if self.monsters[m].last_two_moves(1) {
                            if self.rb() { tackle } else { lick }
                        } else {
                            wound
                        }
                    } else if num < 80 {
                        if self.monsters[m].last_two_moves(2) {
                            if self.rbc(0.5) { wound } else { lick }
                        } else {
                            tackle
                        }
                    } else if self.monsters[m].last_move(4) {
                        if self.rbc(0.4) { wound } else { tackle }
                    } else {
                        lick
                    }
                } else if num < 30 {
                    if self.monsters[m].last_two_moves(1) {
                        if self.rb() { tackle } else { lick }
                    } else {
                        wound
                    }
                } else if num < 70 {
                    if self.monsters[m].last_move(2) {
                        if self.rbc(0.4) { wound } else { lick }
                    } else {
                        tackle
                    }
                } else if self.monsters[m].last_two_moves(4) {
                    if self.rbc(0.4) { wound } else { tackle }
                } else {
                    lick
                }
            }
            Ai::AcidSlimeS => {
                let tackle = EnemyMove::attack(1, ATTACK, d0);
                let lick = EnemyMove::simple(2, Intent::Debuff);
                if asc >= 17 {
                    if self.monsters[m].last_two_moves(1) { tackle } else { lick }
                } else if self.rb() {
                    tackle
                } else {
                    lick
                }
            }
            Ai::SpikeSlimeL { .. } | Ai::SpikeSlimeM => {
                let tackle = EnemyMove::attack(1, Intent::AttackDebuff, d0);
                let lick = EnemyMove::simple(4, Intent::Debuff);
                let mon = &self.monsters[m];
                if asc >= 17 {
                    if num < 30 {
                        if mon.last_two_moves(1) { lick } else { tackle }
                    } else if mon.last_move(4) {
                        tackle
                    } else {
                        lick
                    }
                } else if num < 30 {
                    if mon.last_two_moves(1) { lick } else { tackle }
                } else if mon.last_two_moves(4) {
                    tackle
                } else {
                    lick
                }
            }
            Ai::SpikeSlimeS => EnemyMove::attack(1, ATTACK, d0),
            Ai::Cultist { first_move, ritual } => {
                if first_move {
                    self.monsters[m].ai = Ai::Cultist { first_move: false, ritual };
                    EnemyMove::simple(3, Intent::Buff)
                } else {
                    EnemyMove::attack(1, ATTACK, d0)
                }
            }
            Ai::FungiBeast { .. } => {
                let bite = EnemyMove::attack(1, ATTACK, d0);
                let grow = EnemyMove::simple(2, Intent::Buff);
                let mon = &self.monsters[m];
                if num < 60 {
                    if mon.last_two_moves(1) { grow } else { bite }
                } else if mon.last_move(2) {
                    bite
                } else {
                    grow
                }
            }
            Ai::GremlinFat => EnemyMove::attack(2, Intent::AttackDebuff, d0),
            Ai::GremlinThief { .. } | Ai::GremlinWarrior => EnemyMove::attack(1, ATTACK, d0),
            Ai::GremlinTsundere { .. } => EnemyMove::simple(1, Intent::Defend),
            Ai::GremlinWizard { .. } => EnemyMove::simple(2, Intent::Unknown),
            Ai::GremlinNob { used_bellow, can_vuln } => {
                let rush = EnemyMove::attack(1, ATTACK, d0);
                let bash = if can_vuln { EnemyMove::attack(2, Intent::AttackDebuff, d1) } else { EnemyMove::attack(2, ATTACK, d1) };
                let mon = &self.monsters[m];
                if !used_bellow {
                    self.monsters[m].ai = Ai::GremlinNob { used_bellow: true, can_vuln };
                    EnemyMove::simple(3, Intent::Buff)
                } else if asc >= 18 {
                    if !mon.last_move(2) && !mon.last_move_before(2) {
                        bash
                    } else if mon.last_two_moves(1) {
                        bash
                    } else {
                        rush
                    }
                } else if num < 33 {
                    bash
                } else if mon.last_two_moves(1) {
                    bash
                } else {
                    rush
                }
            }
            Ai::Hexaghost { activated, orb_active_count, fire_tackle_count, inferno_hits, .. } => {
                if !activated {
                    if let Ai::Hexaghost { activated, .. } = &mut self.monsters[m].ai {
                        *activated = true;
                    }
                    EnemyMove::simple(5, Intent::Unknown)
                } else {
                    let d3 = self.monsters[m].damage[3].base;
                    match orb_active_count {
                        0 | 2 | 5 => EnemyMove::attack(4, Intent::AttackDebuff, d1),
                        1 | 4 => EnemyMove::new(2, ATTACK, d0, fire_tackle_count, true),
                        3 => EnemyMove::simple(3, Intent::DefendBuff),
                        6 => EnemyMove::new(6, Intent::AttackDebuff, d3, inferno_hits, true),
                        _ => return,
                    }
                }
            }
            Ai::JawWorm { first_move, hard, bellow_str, bellow_block, thrash_block } => {
                let chomp = EnemyMove::attack(1, ATTACK, d0);
                let bellow = EnemyMove::simple(2, Intent::DefendBuff);
                let thrash = EnemyMove::attack(3, Intent::AttackDefend, d1);
                if first_move {
                    self.monsters[m].ai = Ai::JawWorm { first_move: false, hard, bellow_str, bellow_block, thrash_block };
                    chomp
                } else if num < 25 {
                    if self.monsters[m].last_move(1) {
                        if self.rbc(0.5625) { bellow } else { thrash }
                    } else {
                        chomp
                    }
                } else if num < 55 {
                    if self.monsters[m].last_two_moves(3) {
                        if self.rbc(0.357) { chomp } else { bellow }
                    } else {
                        thrash
                    }
                } else if self.monsters[m].last_move(2) {
                    if self.rbc(0.416) { chomp } else { thrash }
                } else {
                    bellow
                }
            }
            Ai::Lagavulin { is_out, debuff_turn_count, .. } => {
                let debuff = EnemyMove::simple(1, Intent::StrongDebuff);
                let attack = EnemyMove::attack(3, ATTACK, d0);
                if is_out {
                    if debuff_turn_count < 2 {
                        if self.monsters[m].last_two_moves(3) { debuff } else { attack }
                    } else {
                        debuff
                    }
                } else {
                    EnemyMove::simple(5, Intent::Sleep)
                }
            }
            Ai::Looter { .. } => EnemyMove::attack(1, ATTACK, d0),
            Ai::LouseNormal { .. } | Ai::LouseDefensive { .. } => {
                let bite = EnemyMove::attack(3, ATTACK, d0);
                let other = if matches!(self.monsters[m].ai, Ai::LouseNormal { .. }) {
                    EnemyMove::simple(4, Intent::Buff)
                } else {
                    EnemyMove::simple(4, Intent::Debuff)
                };
                let mon = &self.monsters[m];
                if asc >= 17 {
                    if num < 25 {
                        if mon.last_move(4) { bite } else { other }
                    } else if mon.last_two_moves(3) {
                        other
                    } else {
                        bite
                    }
                } else if num < 25 {
                    if mon.last_two_moves(4) { bite } else { other }
                } else if mon.last_two_moves(3) {
                    other
                } else {
                    bite
                }
            }
            Ai::Sentry { first_move, dazed_amt } => {
                let bolt = EnemyMove::simple(3, Intent::Debuff);
                let beam = EnemyMove::attack(4, ATTACK, d0);
                if first_move {
                    // `monsters.lastIndexOf(this)`: -1 when not yet in the group.
                    let idx = self.order.iter().rposition(|&i| i == m).map_or(-1, |p| p as i32);
                    self.monsters[m].ai = Ai::Sentry { first_move: false, dazed_amt };
                    if idx % 2 == 0 { bolt } else { beam }
                } else if self.monsters[m].last_move(4) {
                    bolt
                } else {
                    beam
                }
            }
            Ai::SlaverBlue { .. } => {
                let stab = EnemyMove::attack(1, ATTACK, d0);
                let rake = EnemyMove::attack(4, Intent::AttackDebuff, d1);
                let mon = &self.monsters[m];
                if num >= 40 && !mon.last_two_moves(1) {
                    stab
                } else if asc >= 17 {
                    if !mon.last_move(4) { rake } else { stab }
                } else if !mon.last_two_moves(4) {
                    rake
                } else {
                    stab
                }
            }
            Ai::SlaverRed { used_entangle, first_turn, stab_dmg } => {
                let stab = EnemyMove::attack(1, ATTACK, d0);
                let entangle = EnemyMove::simple(2, Intent::StrongDebuff);
                let scrape = EnemyMove::attack(3, Intent::AttackDebuff, d1);
                let mon = &self.monsters[m];
                if first_turn {
                    self.monsters[m].ai = Ai::SlaverRed { used_entangle, first_turn: false, stab_dmg };
                    EnemyMove::attack(1, ATTACK, stab_dmg)
                } else if num >= 75 && !used_entangle {
                    entangle
                } else if num >= 55 && used_entangle && !mon.last_two_moves(1) {
                    stab
                } else if asc >= 17 {
                    if !mon.last_move(3) { scrape } else { stab }
                } else if !mon.last_two_moves(3) {
                    scrape
                } else {
                    stab
                }
            }
            Ai::SlimeBoss { first_turn } => {
                if first_turn {
                    self.monsters[m].ai = Ai::SlimeBoss { first_turn: false };
                    EnemyMove::simple(4, Intent::StrongDebuff)
                } else {
                    return;
                }
            }
            Ai::TheGuardian { is_open, .. } => {
                if is_open {
                    EnemyMove::simple(6, Intent::Defend)
                } else {
                    EnemyMove::attack(3, ATTACK, d1)
                }
            }            _ => return self.city_get_move(m, num),
        };
        self.mset(m, mv);
    }

    /// `takeTurn()`.
    pub(crate) fn take_turn(&mut self, m: usize) {
        let asc = self.ascension;
        let next = self.monsters[m].next_move;
        let me = Cr::Monster(m);
        match self.monsters[m].ai.clone() {
            Ai::AcidSlimeL { save_x, .. } => match next {
                1 => {
                    self.mattack(m, 0);
                    self.temp_card_discard(CardId::Slimed, 2);
                    self.mroll(m);
                }
                2 => {
                    self.mattack(m, 1);
                    self.mroll(m);
                }
                3 => {
                    let hp = self.monsters[m].hp;
                    self.split(m, [acid_slime_m_split(asc, save_x - 134.0, hp), acid_slime_m_split(asc, save_x + 134.0, hp)]);
                }
                4 => {
                    self.debuff_player(m, Power::weak(2, true));
                    self.mroll(m);
                }
                _ => {}
            },
            Ai::AcidSlimeM => match next {
                1 => {
                    self.mattack(m, 0);
                    self.temp_card_discard(CardId::Slimed, 1);
                    self.mroll(m);
                }
                2 => {
                    self.mattack(m, 1);
                    self.mroll(m);
                }
                4 => {
                    self.debuff_player(m, Power::weak(1, true));
                    self.mroll(m);
                }
                _ => {}
            },
            Ai::AcidSlimeS => match next {
                1 => {
                    self.mattack(m, 0);
                    self.mset(m, EnemyMove::simple(2, Intent::Debuff));
                }
                2 => {
                    self.debuff_player(m, Power::weak(1, true));
                    let d0 = self.base(m, 0);
                    self.mset(m, EnemyMove::attack(1, ATTACK, d0));
                }
                _ => {}
            },
            Ai::SpikeSlimeL { save_x, .. } => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        self.temp_card_discard(CardId::Slimed, 2);
                    }
                    3 => {
                        let hp = self.monsters[m].hp;
                        self.split(m, [spike_slime_m_split(asc, save_x - 134.0, hp), spike_slime_m_split(asc, save_x + 134.0, hp)]);
                    }
                    4 => self.debuff_player(m, Power::frail(if asc >= 17 { 3 } else { 2 }, true)),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::SpikeSlimeM => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        self.temp_card_discard(CardId::Slimed, 1);
                    }
                    4 => self.debuff_player(m, Power::frail(1, true)),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::SpikeSlimeS => {
                if next == 1 {
                    self.mattack(m, 0);
                    self.mroll(m);
                }
            }
            Ai::Cultist { ritual, .. } => {
                match next {
                    1 => self.mattack(m, 0),
                    3 => {
                        let amt = if asc >= 17 { ritual + 1 } else { ritual };
                        self.buff_self(m, Power::ritual(amt, false));
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::FungiBeast { str_amt } => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        let amt = if asc >= 17 { str_amt + 1 } else { str_amt };
                        self.buff_self(m, Power::new(PowerId::Strength, amt));
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::GremlinFat => match next {
                2 => {
                    self.mattack(m, 0);
                    self.debuff_player(m, Power::weak(1, true));
                    if asc >= 17 {
                        self.debuff_player(m, Power::frail(1, true));
                    }
                    self.escape_or(m, |c| c.mroll(m));
                }
                99 => self.do_escape(m),
                _ => {}
            },
            Ai::GremlinThief { thief_damage } => match next {
                1 => {
                    self.mattack(m, 0);
                    self.escape_or(m, |c| c.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(1, ATTACK, thief_damage) }));
                }
                99 => self.do_escape(m),
                _ => {}
            },
            Ai::GremlinTsundere { block_amt } => match next {
                1 => {
                    self.push_bot(Action::GainBlockRandomMonster { source: m, amount: block_amt });
                    let alive = self.order.iter().filter(|&&i| !self.monsters[i].is_dying && !self.monsters[i].is_escaping).count();
                    if self.monsters[m].escape_next {
                        self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(99, Intent::Escape) });
                    } else if alive > 1 {
                        self.mset(m, EnemyMove::simple(1, Intent::Defend));
                    } else {
                        let d0 = self.base(m, 0);
                        self.mset(m, EnemyMove::attack(2, ATTACK, d0));
                    }
                }
                2 => {
                    self.mattack(m, 0);
                    let d0 = self.base(m, 0);
                    self.escape_or(m, |c| c.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(2, ATTACK, d0) }));
                }
                99 => self.do_escape(m),
                _ => {}
            },
            Ai::GremlinWarrior => match next {
                1 => {
                    self.mattack(m, 0);
                    let d0 = self.base(m, 0);
                    self.escape_or(m, |c| c.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(1, ATTACK, d0) }));
                }
                99 => self.do_escape(m),
                _ => {}
            },
            Ai::GremlinWizard { current_charge } => match next {
                1 => {
                    self.monsters[m].ai = Ai::GremlinWizard { current_charge: 0 };
                    self.mattack(m, 0);
                    let d0 = self.base(m, 0);
                    if self.monsters[m].escape_next {
                        self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(99, Intent::Escape) });
                    } else if asc >= 17 {
                        self.mset(m, EnemyMove::attack(1, ATTACK, d0));
                    } else {
                        self.mset(m, EnemyMove::simple(2, Intent::Unknown));
                    }
                }
                2 => {
                    let charge = current_charge + 1;
                    self.monsters[m].ai = Ai::GremlinWizard { current_charge: charge };
                    let d0 = self.base(m, 0);
                    if self.monsters[m].escape_next {
                        self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(99, Intent::Escape) });
                    } else if charge == 3 {
                        self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(1, ATTACK, d0) });
                    } else {
                        self.mset(m, EnemyMove::simple(2, Intent::Unknown));
                    }
                }
                99 => self.do_escape(m),
                _ => {}
            },
            Ai::GremlinNob { can_vuln, .. } => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        self.mattack(m, 1);
                        if can_vuln {
                            let p = Power::vulnerable(2, true, self.turn_has_ended);
                            self.debuff_player(m, p);
                        }
                    }
                    3 => self.buff_self(m, Power::new(PowerId::Anger, if asc >= 18 { 3 } else { 2 })),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Hexaghost { str_amount, sear_burn_count, burn_upgraded, strengthen_block, inferno_hits, .. } => match next {
                1 => {
                    for _ in 0..6 {
                        self.mattack(m, 2);
                    }
                    self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Deactivate });
                    self.mroll(m);
                }
                2 => {
                    self.mattack(m, 0);
                    self.mattack(m, 0);
                    self.push_bot(Action::ChangeState { monster: m, state: MonsterState::ActivateOrb });
                    self.mroll(m);
                }
                3 => {
                    self.push_bot(Action::GainBlock { target: me, amount: strengthen_block });
                    self.buff_self(m, Power::new(PowerId::Strength, str_amount));
                    self.push_bot(Action::ChangeState { monster: m, state: MonsterState::ActivateOrb });
                    self.mroll(m);
                }
                4 => {
                    self.mattack(m, 1);
                    let mut burn = self.new_card(CardId::Burn);
                    if burn_upgraded {
                        burn.upgrade();
                    }
                    self.push_bot(Action::MakeTempCardInDiscard { card: burn, amount: sear_burn_count });
                    self.push_bot(Action::ChangeState { monster: m, state: MonsterState::ActivateOrb });
                    self.mroll(m);
                }
                5 => {
                    self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Activate });
                    let d = self.player.hp / 12 + 1;
                    self.monsters[m].damage[2].base = d;
                    self.monster_apply_powers(m);
                    self.mset(m, EnemyMove::new(1, ATTACK, d, 6, true));
                }
                6 => {
                    for _ in 0..inferno_hits {
                        self.mattack(m, 3);
                    }
                    self.push_bot(Action::BurnIncrease);
                    if let Ai::Hexaghost { burn_upgraded, .. } = &mut self.monsters[m].ai {
                        *burn_upgraded = true;
                    }
                    self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Deactivate });
                    self.mroll(m);
                }
                _ => {}
            },
            Ai::JawWorm { bellow_str, bellow_block, thrash_block, .. } => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        self.buff_self(m, Power::new(PowerId::Strength, bellow_str));
                        self.push_bot(Action::GainBlock { target: me, amount: bellow_block });
                    }
                    3 => {
                        self.mattack(m, 1);
                        self.push_bot(Action::GainBlock { target: me, amount: thrash_block });
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Lagavulin { debuff, idle_count, .. } => match next {
                1 => {
                    self.set_lagavulin(m, |_, _, _, dtc| *dtc = 0);
                    self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Debuff });
                    self.debuff_player(m, Power::new(PowerId::Dexterity, debuff));
                    self.debuff_player(m, Power::new(PowerId::Strength, debuff));
                    self.mroll(m);
                }
                3 => {
                    self.set_lagavulin(m, |_, _, _, dtc| *dtc += 1);
                    self.mattack(m, 0);
                    self.mroll(m);
                }
                4 => self.mroll(m),
                5 => {
                    let idle = idle_count + 1;
                    self.set_lagavulin(m, |_, _, ic, _| *ic = idle);
                    if idle >= 3 {
                        self.set_lagavulin(m, |_, trig, _, _| *trig = true);
                        self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Open });
                        let d0 = self.base(m, 0);
                        self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(3, ATTACK, d0) });
                    } else {
                        self.mset(m, EnemyMove::simple(5, Intent::Sleep));
                    }
                    if idle == 1 || idle == 2 {
                        self.mroll(m);
                    }
                }
                6 => {
                    self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Open });
                    let d0 = self.base(m, 0);
                    self.mset(m, EnemyMove::attack(3, ATTACK, d0));
                    self.create_intent(m);
                    self.set_lagavulin(m, |_, trig, _, _| *trig = true);
                    self.mroll(m);
                }
                _ => {}
            },
            Ai::Looter { gold_amt, slash_count, escape_def, .. } => match next {
                1 => {
                    if slash_count == 0 {
                        // The 60% roll only picks a line of dialogue, but consumes aiRng.
                        let _ = self.rbc(0.6);
                    }
                    self.push_bot(Action::LooterRecordGold { monster: m });
                    self.push_bot(Action::Damage { target: Cr::Player, source: DamageSource::MonsterSlot { monster: m, slot: 0 }, steal_gold: gold_amt });
                    let sc = slash_count + 1;
                    self.set_looter_slash(m, sc);
                    if sc == 2 {
                        if self.rbc(0.5) {
                            self.mset(m, EnemyMove::simple(2, Intent::Defend));
                        } else {
                            let d1 = self.base(m, 1);
                            self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(4, ATTACK, d1) });
                        }
                    } else {
                        let d0 = self.base(m, 0);
                        self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(1, ATTACK, d0) });
                    }
                }
                2 => {
                    self.push_bot(Action::GainBlock { target: me, amount: escape_def });
                    self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(3, Intent::Escape) });
                }
                3 => {
                    self.mugged = true;
                    self.push_bot(Action::Escape { monster: m });
                    self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(3, Intent::Escape) });
                }
                4 => {
                    self.set_looter_slash(m, slash_count + 1);
                    self.push_bot(Action::LooterRecordGold { monster: m });
                    self.push_bot(Action::Damage { target: Cr::Player, source: DamageSource::MonsterSlot { monster: m, slot: 1 }, steal_gold: gold_amt });
                    self.mset(m, EnemyMove::simple(2, Intent::Defend));
                }
                _ => {}
            },
            Ai::LouseNormal { is_open } | Ai::LouseDefensive { is_open } => {
                let normal = matches!(self.monsters[m].ai, Ai::LouseNormal { .. });
                match next {
                    3 => {
                        if !is_open {
                            self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Open });
                        }
                        self.mattack(m, 0);
                    }
                    4 => {
                        let state = if !is_open { MonsterState::Rear } else { MonsterState::RearIdle };
                        self.push_bot(Action::ChangeState { monster: m, state });
                        if normal {
                            self.buff_self(m, Power::new(PowerId::Strength, if asc >= 17 { 4 } else { 3 }));
                        } else {
                            self.debuff_player(m, Power::weak(2, true));
                        }
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Sentry { dazed_amt, .. } => {
                match next {
                    3 => self.temp_card_discard(CardId::Dazed, dazed_amt),
                    4 => self.mattack(m, 0),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::SlaverBlue { weak_amt } => {
                match next {
                    1 => self.mattack(m, 0),
                    4 => {
                        self.mattack(m, 1);
                        let amt = if asc >= 17 { weak_amt + 1 } else { weak_amt };
                        self.debuff_player(m, Power::weak(amt, true));
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::SlaverRed { first_turn, stab_dmg, .. } => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        self.push_bot(Action::ChangeState { monster: m, state: MonsterState::UseNet });
                        self.debuff_player(m, Power::new(PowerId::Entangled, 1));
                        self.monsters[m].ai = Ai::SlaverRed { used_entangle: true, first_turn, stab_dmg };
                    }
                    3 => {
                        self.mattack(m, 1);
                        let amt = if asc >= 17 { 2 } else { 1 };
                        let p = Power::vulnerable(amt, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::SlimeBoss { .. } => match next {
                1 => {
                    self.mattack(m, 1);
                    self.mset(m, EnemyMove::simple(4, Intent::StrongDebuff));
                }
                2 => {
                    let d1 = self.base(m, 1);
                    self.mset(m, EnemyMove::attack(1, ATTACK, d1));
                }
                3 => {
                    let hp = self.monsters[m].hp;
                    self.split(m, [spike_slime_l_split(asc, -385.0, hp), acid_slime_l_split(asc, 120.0, hp)]);
                }
                4 => {
                    self.temp_card_discard(CardId::Slimed, if asc >= 19 { 5 } else { 3 });
                    self.mset(m, EnemyMove::simple(2, Intent::Unknown));
                }
                _ => {}
            },
            Ai::TheGuardian { thorns_damage, vent_debuff, block_amount, whirlwind_count, whirlwind_damage, twin_slam_damage, .. } => {
                match next {
                    1 => {
                        let amt = if asc >= 19 { thorns_damage + 1 } else { thorns_damage };
                        self.buff_self(m, Power::new(PowerId::SharpHide, amt));
                        let d1 = self.base(m, 1);
                        self.mset(m, EnemyMove::attack(3, ATTACK, d1));
                    }
                    2 => {
                        self.mattack(m, 0);
                        self.mset(m, EnemyMove::simple(7, Intent::StrongDebuff));
                    }
                    3 => {
                        self.mattack(m, 1);
                        self.mset(m, EnemyMove::new(4, Intent::AttackBuff, twin_slam_damage, 2, true));
                    }
                    4 => {
                        self.push_bot(Action::ChangeState { monster: m, state: MonsterState::OffensiveMode });
                        self.mattack(m, 3);
                        self.mattack(m, 3);
                        self.push_bot(Action::RemovePower { target: me, id: PowerId::SharpHide });
                        self.mset(m, EnemyMove::new(5, ATTACK, whirlwind_damage, whirlwind_count, true));
                    }
                    5 => {
                        for _ in 0..whirlwind_count {
                            self.mattack(m, 2);
                        }
                        self.mset(m, EnemyMove::simple(6, Intent::Defend));
                    }
                    6 => {
                        self.push_bot(Action::GainBlock { target: me, amount: block_amount });
                        let d0 = self.base(m, 0);
                        self.mset(m, EnemyMove::attack(2, ATTACK, d0));
                    }
                    7 => {
                        self.debuff_player(m, Power::weak(vent_debuff, true));
                        let p = Power::vulnerable(vent_debuff, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                        self.mset(m, EnemyMove::new(5, ATTACK, whirlwind_damage, whirlwind_count, true));
                    }
                    _ => {}
                }
            }
            _ => self.city_take_turn(m),
        }
    }

    pub(super) fn escape_or(&mut self, m: usize, otherwise: impl FnOnce(&mut Combat)) {
        if self.monsters[m].escape_next {
            self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(99, Intent::Escape) });
        } else {
            otherwise(self);
        }
    }

    pub(super) fn do_escape(&mut self, m: usize) {
        self.push_bot(Action::Escape { monster: m });
        self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(99, Intent::Escape) });
    }

    /// Slime split: suicide, then spawn the two halves.
    fn split(&mut self, m: usize, halves: [Monster; 2]) {
        self.push_bot(Action::CannotLose);
        self.push_bot(Action::Suicide { monster: m, trigger_relics: false });
        let [a, b] = halves;
        self.push_bot(Action::SpawnMonster { monster: Box::new(a), minion: false });
        self.push_bot(Action::SpawnMonster { monster: Box::new(b), minion: false });
        self.push_bot(Action::CanLose);
        self.mset(m, EnemyMove::simple(3, Intent::Unknown));
    }

    fn set_lagavulin(&mut self, m: usize, f: impl FnOnce(&mut bool, &mut bool, &mut i32, &mut i32)) {
        if let Ai::Lagavulin { is_out, is_out_triggered, idle_count, debuff_turn_count, .. } = &mut self.monsters[m].ai {
            f(is_out, is_out_triggered, idle_count, debuff_turn_count);
        }
    }

    fn set_looter_slash(&mut self, m: usize, v: i32) {
        if let Ai::Looter { slash_count, .. } = &mut self.monsters[m].ai {
            *slash_count = v;
        }
    }

    /// `changeState(stateName)`.
    pub(crate) fn change_state(&mut self, m: usize, state: MonsterState) {
        let is_dying = self.monsters[m].is_dying;
        let d3 = self.monsters[m].damage.get(3).map_or(-1, |s| s.base);
        let mut inferno = None;
        match &mut self.monsters[m].ai {
            Ai::LouseNormal { is_open } | Ai::LouseDefensive { is_open } => *is_open = state != MonsterState::Closed,
            Ai::Hexaghost { orb_active_count, inferno_hits, .. } => match state {
                MonsterState::Activate => *orb_active_count = 6,
                MonsterState::ActivateOrb => {
                    *orb_active_count += 1;
                    if *orb_active_count == 6 {
                        inferno = Some(*inferno_hits);
                    }
                }
                MonsterState::Deactivate => *orb_active_count = 0,
                _ => {}
            },
            Ai::Lagavulin { is_out, .. } => {
                if state == MonsterState::Open && !is_dying {
                    *is_out = true;
                    self.push_bot(Action::ReducePower { target: Cr::Monster(m), id: PowerId::Metallicize, amount: 8 });
                }
            }
            Ai::TheGuardian { .. } => self.guardian_change_state(m, state),
            _ => {}
        }
        self.city_change_state(m, state);
        if let Some(hits) = inferno {
            self.mset(m, EnemyMove::new(6, Intent::AttackDebuff, d3, hits, true));
        }
    }

    fn guardian_change_state(&mut self, m: usize, state: MonsterState) {
        let me = Cr::Monster(m);
        let Ai::TheGuardian { dmg_threshold, dmg_threshold_increase, defensive_block, .. } = self.monsters[m].ai else { return };
        match state {
            MonsterState::DefensiveMode => {
                self.push_bot(Action::RemovePower { target: me, id: PowerId::ModeShift });
                self.push_bot(Action::GainBlock { target: me, amount: defensive_block });
                if let Ai::TheGuardian { dmg_threshold, is_open, .. } = &mut self.monsters[m].ai {
                    *dmg_threshold += dmg_threshold_increase;
                    *is_open = false;
                }
                self.mset(m, EnemyMove::simple(1, Intent::Buff));
                self.create_intent(m);
            }
            MonsterState::OffensiveMode => {
                self.buff_self(m, Power::new(PowerId::ModeShift, dmg_threshold));
                self.push_bot(Action::ChangeState { monster: m, state: MonsterState::ResetThreshold });
                let block = self.monsters[m].block;
                if block != 0 {
                    self.push_bot(Action::LoseBlock { target: me, amount: block });
                }
                if let Ai::TheGuardian { is_open, close_up_triggered, .. } = &mut self.monsters[m].ai {
                    *is_open = true;
                    *close_up_triggered = false;
                }
            }
            MonsterState::ResetThreshold => {
                if let Ai::TheGuardian { dmg_taken, .. } = &mut self.monsters[m].ai {
                    *dmg_taken = 0;
                }
            }
            _ => {}
        }
    }

    /// Subclass `damage()` overrides that run after `super.damage(info)`.
    pub(crate) fn after_monster_damaged(&mut self, m: usize, prev_hp: i32, info: &DamageInfo) {
        let _ = info;
        let mon = &self.monsters[m];
        let half = mon.hp as f32 <= mon.max_hp as f32 / 2.0;
        match mon.ai.clone() {
            Ai::AcidSlimeL { split_triggered: false, save_x } | Ai::SpikeSlimeL { split_triggered: false, save_x } => {
                if !mon.is_dying && half && mon.next_move != 3 {
                    self.interrupt_split(m);
                    match &mut self.monsters[m].ai {
                        Ai::AcidSlimeL { split_triggered, .. } | Ai::SpikeSlimeL { split_triggered, .. } => *split_triggered = true,
                        _ => {}
                    }
                    let _ = save_x;
                }
            }
            Ai::SlimeBoss { .. } => {
                if !mon.is_dying && half && mon.next_move != 3 {
                    self.interrupt_split(m);
                }
            }
            Ai::Lagavulin { is_out_triggered: false, .. } => {
                if mon.hp != prev_hp {
                    self.mset(m, EnemyMove::simple(4, Intent::Stun));
                    self.create_intent(m);
                    self.set_lagavulin(m, |_, trig, _, _| *trig = true);
                    self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Open });
                }
            }
            Ai::TheGuardian { is_open: true, close_up_triggered: false, dmg_threshold, .. } => {
                if prev_hp > mon.hp && !mon.is_dying {
                    let diff = prev_hp - mon.hp;
                    let mut trigger = false;
                    if let Ai::TheGuardian { dmg_taken, .. } = &mut self.monsters[m].ai {
                        *dmg_taken += diff;
                        trigger = *dmg_taken >= dmg_threshold;
                    }
                    if let Some(p) = self.monsters[m].powers.iter_mut().find(|p| p.id == PowerId::ModeShift) {
                        p.amount -= diff;
                    }
                    if trigger {
                        if let Ai::TheGuardian { dmg_taken, close_up_triggered, .. } = &mut self.monsters[m].ai {
                            *dmg_taken = 0;
                            *close_up_triggered = true;
                        }
                        self.push_bot(Action::ChangeState { monster: m, state: MonsterState::DefensiveMode });
                    }
                }
            }
            _ => self.beyond_after_damaged(m),
        }
    }

    fn interrupt_split(&mut self, m: usize) {
        let split = EnemyMove::simple(3, Intent::Unknown);
        self.mset(m, split);
        self.create_intent(m);
        self.push_bot(Action::SetMove { monster: m, mv: split });
    }

    /// Subclass `die()` overrides (only bookkeeping matters headlessly).
    pub(crate) fn after_monster_die(&mut self, m: usize) {
        if let Ai::Looter { stolen_gold, .. } = self.monsters[m].ai {
            let _ = stolen_gold;
        }
    }

}
