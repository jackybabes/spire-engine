//! Act 3 (The Beyond) monsters: constructors, `getMove`, `takeTurn`, and their overrides.

use sts_rng::StsRandom;

use super::{Ai, DamageSlot, EnemyMove, EnemyType, Intent, Monster, MonsterId};
use crate::action::{Action, Cr, DamageInfo, DamageType, MonsterState};
use crate::cards::CardId;
use crate::combat::Combat;
use crate::powers::{Power, PowerId, PowerType};

const ATTACK: Intent = Intent::Attack;

fn with_slots(mut m: Monster, slots: &[i32]) -> Monster {
    m.damage = slots.iter().map(|&b| DamageSlot::new(b)).collect();
    m
}

// ---- Constructors (RNG use matches the Java constructors) ----

pub fn darkling(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::Darkling, 56, x, Ai::Darkling { first_move: true });
    if asc >= 7 {
        m.set_hp(rng, 50, 59);
    } else {
        m.set_hp(rng, 48, 56);
    }
    let (chomp, nip) = if asc >= 2 { (9, rng.random_range(9, 13)) } else { (8, rng.random_range(7, 11)) };
    with_slots(m, &[chomp, nip])
}

pub fn orb_walker(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let first = rng.random_range(90, 96);
    let mut m = Monster::new(MonsterId::OrbWalker, first, x, Ai::OrbWalker);
    if asc >= 7 {
        m.set_hp(rng, 92, 102);
    } else {
        m.set_hp(rng, 90, 96);
    }
    if asc >= 2 {
        with_slots(m, &[11, 16])
    } else {
        with_slots(m, &[10, 15])
    }
}

pub fn spiker(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let (thorns, atk) = if asc >= 2 { (4, 9) } else { (3, 7) };
    let mut m = Monster::new(MonsterId::Spiker, 56, x, Ai::Spiker { starting_thorns: thorns, thorns_count: 0 });
    if asc >= 7 {
        m.set_hp(rng, 44, 60);
    } else {
        m.set_hp(rng, 42, 56);
    }
    with_slots(m, &[atk])
}

pub fn repulsor(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::Repulsor, 35, x, Ai::Repulsor);
    if asc >= 7 {
        m.set_hp(rng, 31, 38);
    } else {
        m.set_hp(rng, 29, 35);
    }
    with_slots(m, &[if asc >= 2 { 13 } else { 11 }])
}

pub fn exploder(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::Exploder, 30, x, Ai::Exploder { turn_count: 0 });
    if asc >= 7 {
        m.set_hp(rng, 30, 35);
    } else {
        m.set_hp(rng, 30, 30);
    }
    with_slots(m, &[if asc >= 2 { 11 } else { 9 }])
}

pub fn spire_growth(rng: &mut StsRandom, asc: i32) -> Monster {
    let mut m = Monster::new(MonsterId::SpireGrowth, 170, 0.0, Ai::SpireGrowth);
    let hp = if asc >= 7 { 190 } else { 170 };
    m.set_hp(rng, hp, hp);
    if asc >= 2 {
        with_slots(m, &[18, 25])
    } else {
        with_slots(m, &[16, 22])
    }
}

pub fn transient(asc: i32) -> Monster {
    let d = if asc >= 2 { 40 } else { 30 };
    let mut m = Monster::new(MonsterId::Transient, 999, 0.0, Ai::Transient { count: 0, starting_death_dmg: d });
    m.gold = 1;
    with_slots(m, &[d, d + 10, d + 20, d + 30, d + 40, d + 50, d + 60])
}

pub fn maw(asc: i32, x: f32) -> Monster {
    let (str_up, terrify) = if asc >= 17 { (5, 5) } else { (3, 3) };
    let m = Monster::new(MonsterId::Maw, 300, x, Ai::Maw { roared: false, turn_count: 1, str_up, terrify_dur: terrify });
    with_slots(m, &[if asc >= 2 { 30 } else { 25 }, 5])
}

pub fn writhing_mass(rng: &mut StsRandom, asc: i32) -> Monster {
    let mut m = Monster::new(MonsterId::WrithingMass, 160, 0.0, Ai::WrithingMass { first_move: true, used_mega_debuff: false });
    let hp = if asc >= 7 { 175 } else { 160 };
    m.set_hp(rng, hp, hp);
    if asc >= 2 {
        with_slots(m, &[38, 9, 16, 12])
    } else {
        with_slots(m, &[32, 7, 15, 10])
    }
}

pub fn giant_head(rng: &mut StsRandom, asc: i32) -> Monster {
    let d = if asc >= 3 { 40 } else { 30 };
    let mut m = Monster::new(MonsterId::GiantHead, 500, -70.0, Ai::GiantHead { count: 5, starting_death_dmg: d });
    m.enemy_type = EnemyType::Elite;
    let hp = if asc >= 8 { 520 } else { 500 };
    m.set_hp(rng, hp, hp);
    with_slots(m, &[13, d, d + 5, d + 10, d + 15, d + 20, d + 25, d + 30])
}

pub fn nemesis(rng: &mut StsRandom, asc: i32) -> Monster {
    let fire = if asc >= 3 { 7 } else { 6 };
    let mut m = Monster::new(MonsterId::Nemesis, 185, 0.0, Ai::Nemesis { scythe_cooldown: 0, first_move: true, fire_dmg: fire });
    m.enemy_type = EnemyType::Elite;
    let hp = if asc >= 8 { 200 } else { 185 };
    m.set_hp(rng, hp, hp);
    with_slots(m, &[45, fire])
}

/// `Reptomancer.POSX`.
pub const REPTOMANCER_X: [f32; 4] = [210.0, -220.0, 180.0, -250.0];

pub fn reptomancer(rng: &mut StsRandom, asc: i32) -> Monster {
    let first = rng.random_range(180, 190);
    let per_spawn = if asc >= 18 { 2 } else { 1 };
    let mut m = Monster::new(MonsterId::Reptomancer, first, -20.0, Ai::Reptomancer { first_move: true, daggers_per_spawn: per_spawn, daggers: [None; 4] });
    m.enemy_type = EnemyType::Elite;
    if asc >= 8 {
        m.set_hp(rng, 190, 200);
    } else {
        m.set_hp(rng, 180, 190);
    }
    if asc >= 3 {
        with_slots(m, &[16, 34])
    } else {
        with_slots(m, &[13, 30])
    }
}

pub fn snake_dagger(rng: &mut StsRandom, x: f32) -> Monster {
    let hp = rng.random_range(20, 25);
    let m = Monster::new(MonsterId::SnakeDagger, hp, x, Ai::SnakeDagger { first_move: true });
    with_slots(m, &[9, 25, 25])
}

pub fn time_eater(rng: &mut StsRandom, asc: i32) -> Monster {
    let mut m = Monster::new(MonsterId::TimeEater, 456, -50.0, Ai::TimeEater { used_haste: false, first_turn: true });
    let hp = if asc >= 9 { 480 } else { 456 };
    m.set_hp(rng, hp, hp);
    m.enemy_type = EnemyType::Boss;
    if asc >= 4 {
        with_slots(m, &[8, 32])
    } else {
        with_slots(m, &[7, 26])
    }
}

pub fn donu(rng: &mut StsRandom, asc: i32) -> Monster {
    let mut m = Monster::new(MonsterId::Donu, 250, 100.0, Ai::Donu { is_attacking: false });
    m.enemy_type = EnemyType::Boss;
    let hp = if asc >= 9 { 265 } else { 250 };
    m.set_hp(rng, hp, hp);
    with_slots(m, &[if asc >= 4 { 12 } else { 10 }])
}

pub fn deca(rng: &mut StsRandom, asc: i32) -> Monster {
    let mut m = Monster::new(MonsterId::Deca, 250, -350.0, Ai::Deca { is_attacking: true });
    m.enemy_type = EnemyType::Boss;
    let hp = if asc >= 9 { 265 } else { 250 };
    m.set_hp(rng, hp, hp);
    with_slots(m, &[if asc >= 4 { 12 } else { 10 }])
}

pub fn awakened_one(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::AwakenedOne, 300, x, Ai::AwakenedOne { form1: true, first_turn: true });
    let hp = if asc >= 9 { 320 } else { 300 };
    m.set_hp(rng, hp, hp);
    m.enemy_type = EnemyType::Boss;
    with_slots(m, &[20, 6, 40, 18, 10])
}

// ---- Behaviour ----

impl Combat {
    /// `usePreBattleAction()` for Act 3 monsters.
    pub(crate) fn beyond_pre_battle(&mut self, m: usize) {
        let asc = self.ascension;
        let me = Cr::Monster(m);
        match self.monsters[m].ai.clone() {
            Ai::Darkling { .. } => {
                self.cannot_lose = true;
                self.buff_self(m, Power::flag(PowerId::Regrow));
            }
            Ai::OrbWalker => self.buff_self(m, Power::new(PowerId::GenericStrengthUp, if asc >= 17 { 5 } else { 3 })),
            Ai::Spiker { starting_thorns, .. } => {
                let n = if asc >= 17 { starting_thorns + 3 } else { starting_thorns };
                self.buff_self(m, Power::new(PowerId::Thorns, n));
            }
            Ai::Exploder { .. } => self.buff_self(m, Power::new(PowerId::Explosive, 3)),
            Ai::Transient { .. } => {
                self.buff_self(m, Power::new(PowerId::Fading, if asc >= 17 { 6 } else { 5 }));
                self.buff_self(m, Power::flag(PowerId::Shifting));
            }
            Ai::WrithingMass { .. } => {
                self.buff_self(m, Power::flag(PowerId::Reactive));
                self.buff_self(m, Power::new(PowerId::Malleable, 3));
            }
            Ai::GiantHead { count, starting_death_dmg } => {
                self.buff_self(m, Power::new(PowerId::Slow, 0));
                if asc >= 18 {
                    self.monsters[m].ai = Ai::GiantHead { count: count - 1, starting_death_dmg };
                }
            }
            Ai::Reptomancer { first_move, daggers_per_spawn, mut daggers } => {
                let me_pos = self.order.iter().position(|&i| i == m).unwrap_or(0);
                for (pos, i) in self.order.clone().into_iter().enumerate() {
                    if self.monsters[i].id != MonsterId::Reptomancer {
                        self.bot_power(Cr::Monster(i), Some(Cr::Monster(i)), Power::flag(PowerId::Minion));
                    }
                    if self.monsters[i].id == MonsterId::SnakeDagger {
                        if pos > me_pos {
                            daggers[0] = Some(i);
                        } else {
                            daggers[1] = Some(i);
                        }
                    }
                }
                self.monsters[m].ai = Ai::Reptomancer { first_move, daggers_per_spawn, daggers };
            }
            Ai::TimeEater { .. } => self.buff_self(m, Power::new(PowerId::TimeWarp, 0)),
            Ai::Donu { .. } | Ai::Deca { .. } => self.buff_self(m, Power::new(PowerId::Artifact, if asc >= 19 { 3 } else { 2 })),
            Ai::AwakenedOne { .. } => {
                self.cannot_lose = true;
                if asc >= 19 {
                    self.buff_self(m, Power::new(PowerId::Regenerate, 15));
                    self.buff_self(m, Power::new(PowerId::Curiosity, 2));
                } else {
                    self.buff_self(m, Power::new(PowerId::Regenerate, 10));
                    self.buff_self(m, Power::new(PowerId::Curiosity, 1));
                }
                self.buff_self(m, Power::flag(PowerId::Unawakened));
                if asc >= 4 {
                    self.buff_self(m, Power::new(PowerId::Strength, 2));
                }
            }
            _ => self.ending_pre_battle(m),
        }
        let _ = me;
    }

    /// `getMove(num)` for Act 3 monsters.
    pub(crate) fn beyond_get_move(&mut self, m: usize, num: i32) {
        let asc = self.ascension;
        let mon = &self.monsters[m];
        let d0 = mon.damage.first().map_or(-1, |s| s.base);
        let d1 = mon.damage.get(1).map_or(-1, |s| s.base);
        let d2 = mon.damage.get(2).map_or(-1, |s| s.base);
        let d3 = mon.damage.get(3).map_or(-1, |s| s.base);
        let mv = match mon.ai.clone() {
            Ai::Darkling { first_move } => {
                let harden = EnemyMove::simple(2, if asc >= 17 { Intent::DefendBuff } else { Intent::Defend });
                let nip = EnemyMove::attack(3, ATTACK, d1);
                if self.monsters[m].half_dead {
                    EnemyMove::simple(5, Intent::Buff)
                } else if first_move {
                    self.monsters[m].ai = Ai::Darkling { first_move: false };
                    if num < 50 { harden } else { nip }
                } else if num < 40 {
                    let idx = self.order.iter().rposition(|&i| i == m).map_or(-1, |p| p as i32);
                    if !self.monsters[m].last_move(1) && idx % 2 == 0 {
                        EnemyMove::new(1, ATTACK, d0, 2, true)
                    } else {
                        let n = self.rng.ai.random_range(40, 99);
                        return self.beyond_get_move(m, n);
                    }
                } else if num < 70 {
                    if !self.monsters[m].last_move(2) { harden } else { nip }
                } else if !self.monsters[m].last_two_moves(3) {
                    nip
                } else {
                    let n = self.rng.ai.random_range(0, 99);
                    return self.beyond_get_move(m, n);
                }
            }
            Ai::OrbWalker => {
                let laser = EnemyMove::attack(1, Intent::AttackDebuff, d0);
                let claw = EnemyMove::attack(2, ATTACK, d1);
                let mon = &self.monsters[m];
                if num < 40 {
                    if !mon.last_two_moves(2) { claw } else { laser }
                } else if !mon.last_two_moves(1) {
                    laser
                } else {
                    claw
                }
            }
            Ai::Spiker { thorns_count, .. } => {
                let atk = EnemyMove::attack(1, ATTACK, d0);
                if thorns_count > 5 || (num < 50 && !self.monsters[m].last_move(1)) {
                    atk
                } else {
                    EnemyMove::simple(2, Intent::Buff)
                }
            }
            Ai::Repulsor => {
                if num < 20 && !self.monsters[m].last_move(2) {
                    EnemyMove::attack(2, ATTACK, d0)
                } else {
                    EnemyMove::simple(1, Intent::Debuff)
                }
            }
            Ai::Exploder { turn_count } => {
                if turn_count < 2 {
                    EnemyMove::attack(1, ATTACK, d0)
                } else {
                    EnemyMove::simple(2, Intent::Unknown)
                }
            }
            Ai::SpireGrowth => {
                let constricted = self.player.has_power(PowerId::Constricted);
                let mon = &self.monsters[m];
                let tackle = EnemyMove::attack(1, ATTACK, d0);
                if asc >= 17 && !constricted && !mon.last_move(2) {
                    EnemyMove::simple(2, Intent::StrongDebuff)
                } else if num < 50 && !mon.last_two_moves(1) {
                    tackle
                } else if !constricted && !mon.last_move(2) {
                    EnemyMove::simple(2, Intent::StrongDebuff)
                } else if !mon.last_two_moves(3) {
                    EnemyMove::attack(3, ATTACK, d1)
                } else {
                    tackle
                }
            }
            Ai::Transient { count, starting_death_dmg } => EnemyMove::attack(1, ATTACK, starting_death_dmg + count * 10),
            Ai::Maw { roared, turn_count, str_up, terrify_dur } => {
                let tc = turn_count + 1;
                self.monsters[m].ai = Ai::Maw { roared, turn_count: tc, str_up, terrify_dur };
                let mon = &self.monsters[m];
                if !roared {
                    EnemyMove::simple(2, Intent::StrongDebuff)
                } else if num < 50 && !mon.last_move(5) {
                    if tc / 2 <= 1 {
                        EnemyMove::attack(5, ATTACK, d1)
                    } else {
                        EnemyMove::new(5, ATTACK, d1, tc / 2, true)
                    }
                } else if !mon.last_move(3) && !mon.last_move(5) {
                    EnemyMove::attack(3, ATTACK, d0)
                } else {
                    EnemyMove::simple(4, Intent::Buff)
                }
            }
            Ai::WrithingMass { first_move, used_mega_debuff } => {
                let big = EnemyMove::attack(0, ATTACK, d0);
                let multi = EnemyMove::new(1, ATTACK, d1, 3, true);
                let block = EnemyMove::attack(2, Intent::AttackDefend, d2);
                let debuff = EnemyMove::attack(3, Intent::AttackDebuff, d3);
                let mega = EnemyMove::simple(4, Intent::StrongDebuff);
                if first_move {
                    self.monsters[m].ai = Ai::WrithingMass { first_move: false, used_mega_debuff };
                    if num < 33 {
                        multi
                    } else if num < 66 {
                        block
                    } else {
                        debuff
                    }
                } else {
                    let mon = &self.monsters[m];
                    let mv = if num < 10 {
                        if !mon.last_move(0) {
                            Some(big)
                        } else {
                            let n = self.rng.ai.random_range(10, 99);
                            self.beyond_get_move(m, n);
                            None
                        }
                    } else if num < 20 {
                        if !used_mega_debuff && !mon.last_move(4) {
                            Some(mega)
                        } else if self.rbc(0.1) {
                            Some(big)
                        } else {
                            let n = self.rng.ai.random_range(20, 99);
                            self.beyond_get_move(m, n);
                            None
                        }
                    } else if num < 40 {
                        if !mon.last_move(3) {
                            Some(debuff)
                        } else if self.rbc(0.4) {
                            let n = self.rng.ai.random(19);
                            self.beyond_get_move(m, n);
                            None
                        } else {
                            let n = self.rng.ai.random_range(40, 99);
                            self.beyond_get_move(m, n);
                            None
                        }
                    } else if num < 70 {
                        if !mon.last_move(1) {
                            Some(multi)
                        } else if self.rbc(0.3) {
                            Some(block)
                        } else {
                            let n = self.rng.ai.random(39);
                            self.beyond_get_move(m, n);
                            None
                        }
                    } else if !mon.last_move(2) {
                        Some(block)
                    } else {
                        let n = self.rng.ai.random(69);
                        self.beyond_get_move(m, n);
                        None
                    };
                    if let Some(mv) = mv {
                        self.mset(m, mv);
                    }
                    self.create_intent(m);
                    return;
                }
            }
            Ai::GiantHead { count, starting_death_dmg } => {
                let mut c = count;
                let mon = &self.monsters[m];
                let mv = if c <= 1 {
                    if c > -6 {
                        c -= 1;
                    }
                    EnemyMove::attack(2, ATTACK, starting_death_dmg - c * 5)
                } else {
                    c -= 1;
                    let glare = EnemyMove::simple(1, Intent::Debuff);
                    let count_attack = EnemyMove::attack(3, ATTACK, 13);
                    if num < 50 {
                        if !mon.last_two_moves(1) { glare } else { count_attack }
                    } else if !mon.last_two_moves(3) {
                        count_attack
                    } else {
                        glare
                    }
                };
                self.monsters[m].ai = Ai::GiantHead { count: c, starting_death_dmg };
                mv
            }
            Ai::Nemesis { scythe_cooldown, first_move, fire_dmg } => {
                let mut cd = scythe_cooldown - 1;
                let fire = EnemyMove::new(2, ATTACK, fire_dmg, 3, true);
                let scythe = EnemyMove::attack(3, ATTACK, 45);
                let burn = EnemyMove::simple(4, Intent::Debuff);
                let mv = if first_move {
                    if num < 50 { fire } else { burn }
                } else if num < 30 {
                    if !self.monsters[m].last_move(3) && cd <= 0 {
                        cd = 2;
                        scythe
                    } else if self.rb() {
                        if !self.monsters[m].last_two_moves(2) { fire } else { burn }
                    } else if !self.monsters[m].last_move(4) {
                        burn
                    } else {
                        fire
                    }
                } else if num < 65 {
                    if !self.monsters[m].last_two_moves(2) {
                        fire
                    } else if self.rb() {
                        if cd > 0 {
                            burn
                        } else {
                            cd = 2;
                            scythe
                        }
                    } else {
                        burn
                    }
                } else if !self.monsters[m].last_move(4) {
                    burn
                } else if self.rb() && cd <= 0 {
                    cd = 2;
                    scythe
                } else {
                    fire
                };
                self.monsters[m].ai = Ai::Nemesis { scythe_cooldown: cd, first_move: false, fire_dmg };
                mv
            }
            Ai::Reptomancer { first_move, daggers_per_spawn, daggers } => {
                let snake_strike = EnemyMove::new(1, Intent::AttackDebuff, d0, 2, true);
                if first_move {
                    self.monsters[m].ai = Ai::Reptomancer { first_move: false, daggers_per_spawn, daggers };
                    EnemyMove::simple(2, Intent::Unknown)
                } else if num < 33 {
                    if !self.monsters[m].last_move(1) {
                        snake_strike
                    } else {
                        let n = self.rng.ai.random_range(33, 99);
                        return self.beyond_get_move(m, n);
                    }
                } else if num < 66 {
                    let alive = self.order.iter().filter(|&&i| i != m && !self.monsters[i].is_dying).count();
                    if !self.monsters[m].last_two_moves(2) && alive <= 3 {
                        EnemyMove::simple(2, Intent::Unknown)
                    } else {
                        snake_strike
                    }
                } else if !self.monsters[m].last_move(3) {
                    EnemyMove::attack(3, ATTACK, d1)
                } else {
                    let n = self.rng.ai.random(65);
                    return self.beyond_get_move(m, n);
                }
            }
            Ai::SnakeDagger { first_move } => {
                if first_move {
                    self.monsters[m].ai = Ai::SnakeDagger { first_move: false };
                    EnemyMove::attack(1, Intent::AttackDebuff, 9)
                } else {
                    EnemyMove::attack(2, ATTACK, 25)
                }
            }
            Ai::TimeEater { used_haste, first_turn } => {
                let mon = &self.monsters[m];
                let reverb = EnemyMove::new(2, ATTACK, d0, 3, true);
                let ripple = EnemyMove::simple(3, Intent::DefendDebuff);
                if mon.hp < mon.max_hp / 2 && !used_haste {
                    self.monsters[m].ai = Ai::TimeEater { used_haste: true, first_turn };
                    EnemyMove::simple(5, Intent::Buff)
                } else if num < 45 {
                    if !mon.last_two_moves(2) {
                        reverb
                    } else {
                        let n = self.rng.ai.random_range(50, 99);
                        return self.beyond_get_move(m, n);
                    }
                } else if num < 80 {
                    if !mon.last_move(4) {
                        EnemyMove::attack(4, Intent::AttackDebuff, d1)
                    } else if self.rbc(0.66) {
                        reverb
                    } else {
                        ripple
                    }
                } else if !mon.last_move(3) {
                    ripple
                } else {
                    let n = self.rng.ai.random(74);
                    return self.beyond_get_move(m, n);
                }
            }
            Ai::Donu { is_attacking } => {
                if is_attacking {
                    EnemyMove::new(0, ATTACK, d0, 2, true)
                } else {
                    EnemyMove::simple(2, Intent::Buff)
                }
            }
            Ai::Deca { is_attacking } => {
                if is_attacking {
                    EnemyMove::new(0, Intent::AttackDebuff, d0, 2, true)
                } else if asc >= 19 {
                    EnemyMove::simple(2, Intent::DefendBuff)
                } else {
                    EnemyMove::simple(2, Intent::Defend)
                }
            }
            Ai::AwakenedOne { form1, first_turn } => {
                let mon = &self.monsters[m];
                if form1 {
                    let slash = EnemyMove::attack(1, ATTACK, 20);
                    let soul_strike = EnemyMove::new(2, ATTACK, 6, 4, true);
                    if first_turn {
                        self.monsters[m].ai = Ai::AwakenedOne { form1, first_turn: false };
                        slash
                    } else if num < 25 {
                        if !mon.last_move(2) { soul_strike } else { slash }
                    } else if !mon.last_two_moves(1) {
                        slash
                    } else {
                        soul_strike
                    }
                } else {
                    let sludge = EnemyMove::attack(6, Intent::AttackDebuff, 18);
                    let tackle = EnemyMove::new(8, ATTACK, 10, 3, true);
                    if first_turn {
                        EnemyMove::attack(5, ATTACK, 40)
                    } else if num < 50 {
                        if !mon.last_two_moves(6) { sludge } else { tackle }
                    } else if !mon.last_two_moves(8) {
                        tackle
                    } else {
                        sludge
                    }
                }
            }
            _ => return self.ending_get_move(m, num),
        };
        self.mset(m, mv);
    }

    /// `takeTurn()` for Act 3 monsters.
    pub(crate) fn beyond_take_turn(&mut self, m: usize) {
        let asc = self.ascension;
        let next = self.monsters[m].next_move;
        let me = Cr::Monster(m);
        match self.monsters[m].ai.clone() {
            Ai::Darkling { .. } => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        self.mattack(m, 0);
                    }
                    2 => {
                        self.push_bot(Action::GainBlock { target: me, amount: 12 });
                        if asc >= 17 {
                            self.buff_self(m, Power::new(PowerId::Strength, 2));
                        }
                    }
                    3 => self.mattack(m, 1),
                    5 => {
                        let half = self.monsters[m].max_hp / 2;
                        self.push_bot(Action::Heal { target: me, amount: half });
                        self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Revive });
                        // new RegrowPower(this) has amount -1; the stack amount 1 only matters
                        // if the power were already there.
                        let a = self.apply_power_action(me, Some(me), Power::flag(PowerId::Regrow), 1);
                        self.push_bot(a);
                        self.relics_on_spawn_monster(m);
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::OrbWalker => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        let c = self.new_card(CardId::Burn);
                        self.push_bot(Action::MakeTempCardInDiscardAndDeck { card: c });
                    }
                    2 => self.mattack(m, 1),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Spiker { starting_thorns, thorns_count } => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        self.monsters[m].ai = Ai::Spiker { starting_thorns, thorns_count: thorns_count + 1 };
                        self.buff_self(m, Power::new(PowerId::Thorns, 2));
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Repulsor => {
                match next {
                    1 => {
                        let card = self.new_card(CardId::Dazed);
                        self.push_bot(Action::MakeTempCardInDrawPile { card, amount: 2, random_spot: true, to_bottom: false });
                    }
                    2 => self.mattack(m, 0),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Exploder { turn_count } => {
                self.monsters[m].ai = Ai::Exploder { turn_count: turn_count + 1 };
                if next == 1 {
                    self.mattack(m, 0);
                }
                self.mroll(m);
            }
            Ai::SpireGrowth => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        let amt = if asc >= 17 { 12 } else { 10 };
                        let mut p = Power::new(PowerId::Constricted, amt);
                        p.base = m as i32;
                        self.debuff_player(m, p);
                    }
                    3 => self.mattack(m, 1),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Transient { count, starting_death_dmg } => {
                if next == 1 {
                    let slot = (count as usize).min(6);
                    self.mattack(m, slot);
                    let c = count + 1;
                    self.monsters[m].ai = Ai::Transient { count: c, starting_death_dmg };
                    self.mset(m, EnemyMove::attack(1, ATTACK, starting_death_dmg + c * 10));
                }
            }
            Ai::Maw { roared, turn_count, str_up, terrify_dur } => {
                match next {
                    2 => {
                        self.debuff_player(m, Power::weak(terrify_dur, true));
                        self.debuff_player(m, Power::frail(terrify_dur, true));
                        self.monsters[m].ai = Ai::Maw { roared: true, turn_count, str_up, terrify_dur };
                    }
                    3 => self.mattack(m, 0),
                    4 => self.buff_self(m, Power::new(PowerId::Strength, str_up)),
                    5 => {
                        for _ in 0..turn_count / 2 {
                            self.mattack(m, 1);
                        }
                    }
                    _ => {}
                }
                let _ = roared;
                self.mroll(m);
            }
            Ai::WrithingMass { first_move, .. } => {
                match next {
                    0 => self.mattack(m, 0),
                    1 => {
                        for _ in 0..3 {
                            self.mattack(m, 1);
                        }
                    }
                    2 => {
                        self.mattack(m, 2);
                        let b = self.base(m, 2);
                        self.push_bot(Action::GainBlock { target: me, amount: b });
                    }
                    3 => {
                        self.mattack(m, 3);
                        self.debuff_player(m, Power::weak(2, true));
                        let p = Power::vulnerable(2, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                    }
                    4 => {
                        self.monsters[m].ai = Ai::WrithingMass { first_move, used_mega_debuff: true };
                        self.push_bot(Action::AddCardToDeck { card: CardId::Parasite });
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::GiantHead { count, .. } => {
                match next {
                    1 => self.debuff_player(m, Power::weak(1, true)),
                    2 => {
                        let index = (1 - count).min(7).max(0) as usize;
                        self.mattack(m, index);
                    }
                    3 => self.mattack(m, 0),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Nemesis { .. } => {
                match next {
                    2 => {
                        for _ in 0..3 {
                            self.mattack(m, 1);
                        }
                    }
                    3 => self.mattack(m, 0),
                    4 => self.temp_card_discard(CardId::Burn, if asc >= 18 { 5 } else { 3 }),
                    _ => {}
                }
                if !self.monsters[m].has_power(PowerId::Intangible) {
                    self.buff_self(m, Power::intangible(1));
                }
                self.mroll(m);
            }
            Ai::Reptomancer { first_move, daggers_per_spawn, mut daggers } => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        self.mattack(m, 0);
                        self.debuff_player(m, Power::weak(1, true));
                    }
                    2 => {
                        let mut spawned = 0;
                        let mut i = 0;
                        while spawned < daggers_per_spawn && i < daggers.len() {
                            if daggers[i].map_or(true, |d| self.monsters[d].is_dead_or_escaped()) {
                                let dagger = snake_dagger(&mut self.rng.monster_hp, REPTOMANCER_X[i]);
                                let id = self.prepare_monster(dagger);
                                daggers[i] = Some(id);
                                self.push_bot(Action::SpawnPrepared { monster: id, minion: true });
                                spawned += 1;
                            }
                            i += 1;
                        }
                        self.monsters[m].ai = Ai::Reptomancer { first_move, daggers_per_spawn, daggers };
                    }
                    3 => self.mattack(m, 1),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::SnakeDagger { .. } => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        self.temp_card_discard(CardId::Wound, 1);
                    }
                    2 => {
                        self.mattack(m, 1);
                        let hp = self.monsters[m].hp;
                        self.push_bot(Action::LoseHp { target: me, source: Some(me), amount: hp });
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::TimeEater { used_haste, .. } => {
                self.monsters[m].ai = Ai::TimeEater { used_haste, first_turn: false };
                match next {
                    2 => {
                        for _ in 0..3 {
                            self.mattack(m, 0);
                        }
                    }
                    3 => {
                        self.push_bot(Action::GainBlock { target: me, amount: 20 });
                        let p = Power::vulnerable(1, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                        self.debuff_player(m, Power::weak(1, true));
                        if asc >= 19 {
                            self.debuff_player(m, Power::frail(1, true));
                        }
                    }
                    4 => {
                        self.mattack(m, 1);
                        self.debuff_player(m, Power::draw_reduction(1));
                        if asc >= 19 {
                            self.temp_card_discard(CardId::Slimed, 2);
                        }
                    }
                    5 => {
                        self.push_bot(Action::RemoveDebuffs { target: me });
                        self.push_bot(Action::RemovePower { target: me, id: PowerId::Shackled });
                        let heal = self.monsters[m].max_hp / 2 - self.monsters[m].hp;
                        self.push_bot(Action::Heal { target: me, amount: heal });
                        if asc >= 19 {
                            let b = self.base(m, 1);
                            self.push_bot(Action::GainBlock { target: me, amount: b });
                        }
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Donu { .. } => {
                match next {
                    0 => {
                        self.mattack(m, 0);
                        self.mattack(m, 0);
                        self.monsters[m].ai = Ai::Donu { is_attacking: false };
                    }
                    2 => {
                        for i in self.order.clone() {
                            self.bot_power(Cr::Monster(i), Some(me), Power::new(PowerId::Strength, 3));
                        }
                        self.monsters[m].ai = Ai::Donu { is_attacking: true };
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Deca { .. } => {
                match next {
                    0 => {
                        self.mattack(m, 0);
                        self.mattack(m, 0);
                        self.temp_card_discard(CardId::Dazed, 2);
                        self.monsters[m].ai = Ai::Deca { is_attacking: false };
                    }
                    2 => {
                        for i in self.order.clone() {
                            self.push_bot(Action::GainBlock { target: Cr::Monster(i), amount: 16 });
                            if asc >= 19 {
                                self.bot_power(Cr::Monster(i), Some(me), Power::new(PowerId::PlatedArmor, 3));
                            }
                        }
                        self.monsters[m].ai = Ai::Deca { is_attacking: true };
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::AwakenedOne { form1, .. } => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        for _ in 0..4 {
                            self.mattack(m, 1);
                        }
                    }
                    3 => self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Rebirth }),
                    5 => {
                        self.monsters[m].ai = Ai::AwakenedOne { form1, first_turn: false };
                        self.mattack(m, 2);
                    }
                    6 => {
                        self.mattack(m, 3);
                        let card = self.new_card(CardId::Void);
                        self.push_bot(Action::MakeTempCardInDrawPile { card, amount: 1, random_spot: true, to_bottom: false });
                    }
                    8 => {
                        for _ in 0..3 {
                            self.mattack(m, 4);
                        }
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            _ => self.ending_take_turn(m),
        }
    }

    /// `applyTurnPowers()` after a monster's turn: the powers' `duringTurn`.
    pub(crate) fn apply_turn_powers(&mut self, m: usize) {
        let powers: Vec<(PowerId, i32)> = self.monsters[m].powers.iter().map(|p| (p.id, p.amount)).collect();
        let me = Cr::Monster(m);
        for (id, amount) in powers {
            match id {
                PowerId::Explosive => {
                    if amount == 1 && !self.monsters[m].is_dying {
                        self.push_bot(Action::Suicide { monster: m, trigger_relics: true });
                        let info = DamageInfo::new(Some(me), 30, DamageType::Thorns);
                        self.push_bot(Action::Damage { target: Cr::Player, source: crate::action::DamageSource::Fixed(info), steal_gold: 0 });
                    } else {
                        self.push_bot(Action::ReducePower { target: me, id, amount: 1 });
                    }
                }
                PowerId::Fading => {
                    if amount == 1 && !self.monsters[m].is_dying {
                        self.push_bot(Action::Suicide { monster: m, trigger_relics: true });
                    } else {
                        self.push_bot(Action::ReducePower { target: me, id, amount: 1 });
                    }
                }
                _ => {}
            }
        }
    }

    /// `changeState` for Act 3 monsters.
    pub(crate) fn beyond_change_state(&mut self, m: usize, state: MonsterState) {
        match (self.monsters[m].ai.clone(), state) {
            (Ai::Darkling { .. }, MonsterState::Revive) => self.monsters[m].half_dead = false,
            (Ai::AwakenedOne { .. }, MonsterState::Rebirth) => {
                let max = if self.ascension >= 9 { 320 } else { 300 };
                self.monsters[m].max_hp = max;
                self.monsters[m].half_dead = false;
                self.push_bot(Action::Heal { target: Cr::Monster(m), amount: max });
                self.push_bot(Action::CanLose);
            }
            _ => {}
        }
    }

    /// The `die()` overrides that skip dying while the room cannot be lost.
    pub(crate) fn beyond_die_blocked(&self, m: usize) -> bool {
        matches!(self.monsters[m].ai, Ai::Darkling { .. } | Ai::AwakenedOne { .. } | Ai::TimeEater { .. } | Ai::CorruptHeart { .. }) && self.cannot_lose
    }

    /// Subclass `die()` overrides that run after `super.die()`.
    pub(crate) fn beyond_after_die(&mut self, m: usize) {
        match self.monsters[m].ai {
            Ai::Reptomancer { .. } => {
                for i in self.order.clone() {
                    if !self.monsters[i].is_dying {
                        self.push_top(Action::Suicide { monster: i, trigger_relics: true });
                    }
                }
            }
            Ai::AwakenedOne { .. } => {
                for i in self.order.clone() {
                    if !self.monsters[i].is_dying && self.monsters[i].id == MonsterId::Cultist {
                        self.push_bot(Action::Escape { monster: i });
                    }
                }
            }
            _ => {}
        }
    }

    /// Subclass `damage()` overrides that run after `super.damage(info)` (Darkling, Awakened One).
    pub(crate) fn beyond_after_damaged(&mut self, m: usize) {
        let mon = &self.monsters[m];
        if mon.hp > 0 || mon.half_dead {
            return;
        }
        match mon.ai.clone() {
            Ai::Darkling { .. } => {
                self.monsters[m].half_dead = true;
                self.on_death(m);
                self.relics_on_monster_death(m);
                self.monsters[m].powers.clear();
                let all_dead = !self.order.iter().any(|&i| self.monsters[i].id == MonsterId::Darkling && !self.monsters[i].half_dead);
                if !all_dead {
                    if self.monsters[m].next_move != 4 {
                        self.mset(m, EnemyMove::simple(4, Intent::Unknown));
                        self.create_intent(m);
                        self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(4, Intent::Unknown) });
                    }
                } else {
                    self.cannot_lose = false;
                    self.monsters[m].half_dead = false;
                    for i in self.order.clone() {
                        // m.die(): every Darkling dies now that the room can be lost.
                        self.monster_die(i, true);
                    }
                }
            }
            Ai::AwakenedOne { .. } => {
                if self.cannot_lose {
                    self.monsters[m].half_dead = true;
                }
                self.on_death(m);
                self.relics_on_monster_death(m);
                self.push_top(Action::ClearCardQueue);
                self.monsters[m].powers.retain(|p| {
                    !(p.power_type() == PowerType::Debuff || matches!(p.id, PowerId::Curiosity | PowerId::Unawakened | PowerId::Shackled))
                });
                self.mset(m, EnemyMove::simple(3, Intent::Unknown));
                self.create_intent(m);
                self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(3, Intent::Unknown) });
                self.monster_apply_powers(m);
                self.monsters[m].ai = Ai::AwakenedOne { form1: false, first_turn: true };
            }
            _ => {}
        }
    }
}
