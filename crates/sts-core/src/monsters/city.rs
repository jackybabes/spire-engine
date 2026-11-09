//! Act 2 (The City) monsters: constructors, `getMove`, `takeTurn`, and their overrides.

use sts_rng::StsRandom;

use super::{Ai, DamageSlot, EnemyMove, EnemyType, Intent, Monster, MonsterId};
use crate::action::{Action, Cr, DamageSource, MonsterState};
use crate::cards::{CardId, CardRarity};
use crate::combat::Combat;
use crate::powers::{Power, PowerId};

const ATTACK: Intent = Intent::Attack;

fn with_slots(mut m: Monster, slots: &[i32]) -> Monster {
    m.damage = slots.iter().map(|&b| DamageSlot::new(b)).collect();
    m
}

// ---- Constructors (RNG use matches the Java constructors) ----

pub fn spheric_guardian(asc: i32, x: f32) -> Monster {
    let m = Monster::new(MonsterId::SphericGuardian, 20, x, Ai::SphericGuardian { first_move: true, second_move: true });
    with_slots(m, &[if asc >= 2 { 11 } else { 10 }])
}

pub fn chosen(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::Chosen, 99, x, Ai::Chosen { first_turn: true, used_hex: false });
    if asc >= 7 {
        m.set_hp(rng, 98, 103);
    } else {
        m.set_hp(rng, 95, 99);
    }
    if asc >= 2 {
        with_slots(m, &[21, 12, 6])
    } else {
        with_slots(m, &[18, 10, 5])
    }
}

pub fn shelled_parasite(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::ShelledParasite, 72, x, Ai::ShelledParasite { first_move: true });
    if asc >= 7 {
        m.set_hp(rng, 70, 75);
    } else {
        m.set_hp(rng, 68, 72);
    }
    if asc >= 2 {
        with_slots(m, &[7, 21, 12])
    } else {
        with_slots(m, &[6, 18, 10])
    }
}

pub fn byrd(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let flight_amt = if asc >= 17 { 4 } else { 3 };
    let (peck_count, swoop) = if asc >= 2 { (6, 14) } else { (5, 12) };
    let mut m = Monster::new(MonsterId::Byrd, 31, x, Ai::Byrd { first_move: true, is_flying: true, peck_count, flight_amt });
    if asc >= 7 {
        m.set_hp(rng, 26, 33);
    } else {
        m.set_hp(rng, 25, 31);
    }
    with_slots(m, &[1, swoop, 3])
}

pub fn mugger(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let gold_amt = if asc >= 17 { 20 } else { 15 };
    let mut m = Monster::new(MonsterId::Mugger, 52, x, Ai::Mugger { gold_amt, slash_count: 0, stolen_gold: 0, escape_def: 11 });
    if asc >= 7 {
        m.set_hp(rng, 50, 54);
    } else {
        m.set_hp(rng, 48, 52);
    }
    if asc >= 2 {
        with_slots(m, &[11, 18])
    } else {
        with_slots(m, &[10, 16])
    }
}

pub fn snake_plant(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::SnakePlant, 79, x, Ai::SnakePlant);
    if asc >= 7 {
        m.set_hp(rng, 78, 82);
    } else {
        m.set_hp(rng, 75, 79);
    }
    with_slots(m, &[if asc >= 2 { 8 } else { 7 }])
}

pub fn snecko(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::Snecko, 120, x, Ai::Snecko { first_turn: true });
    if asc >= 7 {
        m.set_hp(rng, 120, 125);
    } else {
        m.set_hp(rng, 114, 120);
    }
    if asc >= 2 {
        with_slots(m, &[18, 10])
    } else {
        with_slots(m, &[15, 8])
    }
}

pub fn centurion(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let block_amount = if asc >= 17 { 20 } else { 15 };
    let mut m = Monster::new(MonsterId::Centurion, 80, x, Ai::Centurion { block_amount, fury_hits: 3 });
    if asc >= 7 {
        m.set_hp(rng, 78, 83);
    } else {
        m.set_hp(rng, 76, 80);
    }
    if asc >= 2 {
        with_slots(m, &[14, 7])
    } else {
        with_slots(m, &[12, 6])
    }
}

pub fn healer(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let (magic, str_amt, heal_amt) = if asc >= 17 {
        (9, 4, 20)
    } else if asc >= 2 {
        (9, 3, 16)
    } else {
        (8, 2, 16)
    };
    let mut m = Monster::new(MonsterId::Healer, 56, x, Ai::Healer { str_amt, heal_amt });
    if asc >= 7 {
        m.set_hp(rng, 50, 58);
    } else {
        m.set_hp(rng, 48, 56);
    }
    with_slots(m, &[magic])
}

pub fn taskmaster(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    // The super() call rolls a max HP of its own before setHp.
    let first = rng.random_range(54, 60);
    let wound_count = if asc >= 18 {
        3
    } else if asc >= 3 {
        2
    } else {
        1
    };
    let mut m = Monster::new(MonsterId::Taskmaster, first, x, Ai::Taskmaster { wound_count });
    m.enemy_type = EnemyType::Elite;
    if asc >= 8 {
        m.set_hp(rng, 57, 64);
    } else {
        m.set_hp(rng, 54, 60);
    }
    with_slots(m, &[4, 7])
}

pub fn gremlin_leader(rng: &mut StsRandom, asc: i32) -> Monster {
    let (str_amt, block_amt) = if asc >= 18 {
        (5, 10)
    } else if asc >= 3 {
        (4, 6)
    } else {
        (3, 6)
    };
    let mut m = Monster::new(MonsterId::GremlinLeader, 148, 35.0, Ai::GremlinLeader { str_amt, block_amt, gremlins: [None; 3] });
    m.enemy_type = EnemyType::Elite;
    if asc >= 8 {
        m.set_hp(rng, 145, 155);
    } else {
        m.set_hp(rng, 140, 148);
    }
    with_slots(m, &[6])
}

/// `GremlinLeader.POSX`.
pub const GREMLIN_LEADER_X: [f32; 3] = [-366.0, -170.0, -532.0];

pub fn book_of_stabbing(rng: &mut StsRandom, asc: i32) -> Monster {
    let mut m = Monster::new(MonsterId::BookOfStabbing, 164, 0.0, Ai::BookOfStabbing { stab_count: 1 });
    m.enemy_type = EnemyType::Elite;
    if asc >= 8 {
        m.set_hp(rng, 168, 172);
    } else {
        m.set_hp(rng, 160, 164);
    }
    if asc >= 3 {
        with_slots(m, &[7, 24])
    } else {
        with_slots(m, &[6, 21])
    }
}

pub fn champ(rng: &mut StsRandom, asc: i32) -> Monster {
    let (slash, execute, slap, str_amt, forge_amt, block_amt) = if asc >= 19 {
        (18, 10, 14, 4, 7, 20)
    } else if asc >= 9 {
        (18, 10, 14, 3, 6, 18)
    } else if asc >= 4 {
        (18, 10, 14, 3, 5, 15)
    } else {
        (16, 10, 12, 2, 5, 15)
    };
    let ai = Ai::Champ { str_amt, forge_amt, block_amt, num_turns: 0, forge_times: 0, forge_threshold: 2, threshold_reached: false, first_turn: true };
    let mut m = Monster::new(MonsterId::Champ, 420, -90.0, ai);
    m.enemy_type = EnemyType::Boss;
    let hp = if asc >= 9 { 440 } else { 420 };
    m.set_hp(rng, hp, hp);
    with_slots(m, &[slash, execute, slap])
}

pub fn the_collector(rng: &mut StsRandom, asc: i32) -> Monster {
    let (hp, block_amt) = if asc >= 9 { (300, 18) } else { (282, 15) };
    let (rake, str_amt, mega_debuff_amt) = if asc >= 19 {
        (21, 5, 5)
    } else if asc >= 4 {
        (21, 4, 3)
    } else {
        (18, 3, 3)
    };
    let ai = Ai::TheCollector { str_amt, block_amt, mega_debuff_amt, turns_taken: 0, ult_used: false, initial_spawn: true, slots: [None; 2] };
    let mut m = Monster::new(MonsterId::TheCollector, 282, 60.0, ai);
    m.enemy_type = EnemyType::Boss;
    m.set_hp(rng, hp, hp);
    with_slots(m, &[rake])
}

pub fn torch_head(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let first = rng.random_range(38, 40);
    let mut m = Monster::new(MonsterId::TorchHead, first, x, Ai::TorchHead);
    m.set_move(EnemyMove::attack(1, ATTACK, 7));
    if asc >= 9 {
        m.set_hp(rng, 40, 45);
    } else {
        m.set_hp(rng, 38, 40);
    }
    with_slots(m, &[7])
}

pub fn bronze_automaton(rng: &mut StsRandom, asc: i32) -> Monster {
    let (hp, block_amt) = if asc >= 9 { (320, 12) } else { (300, 9) };
    let (flail, beam, str_amt) = if asc >= 4 { (8, 50, 4) } else { (7, 45, 3) };
    let mut m = Monster::new(MonsterId::BronzeAutomaton, 300, -50.0, Ai::BronzeAutomaton { str_amt, block_amt, num_turns: 0, first_turn: true });
    m.enemy_type = EnemyType::Boss;
    m.set_hp(rng, hp, hp);
    with_slots(m, &[flail, beam])
}

pub fn bronze_orb(rng: &mut StsRandom, asc: i32, x: f32, count: i32) -> Monster {
    let first = rng.random_range(52, 58);
    let mut m = Monster::new(MonsterId::BronzeOrb, first, x, Ai::BronzeOrb { used_stasis: false, count, stasis_card: None });
    if asc >= 9 {
        m.set_hp(rng, 54, 60);
    } else {
        m.set_hp(rng, 52, 58);
    }
    with_slots(m, &[8])
}

pub fn bandit_bear(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let con_reduction = if asc >= 17 { -4 } else { -2 };
    let mut m = Monster::new(MonsterId::BanditBear, 42, x, Ai::BanditBear { con_reduction });
    if asc >= 7 {
        m.set_hp(rng, 40, 44);
    } else {
        m.set_hp(rng, 38, 42);
    }
    if asc >= 2 {
        with_slots(m, &[20, 10])
    } else {
        with_slots(m, &[18, 9])
    }
}

pub fn bandit_leader(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let weak_amount = if asc >= 17 { 3 } else { 2 };
    let mut m = Monster::new(MonsterId::BanditLeader, 39, x, Ai::BanditLeader { weak_amount });
    if asc >= 7 {
        m.set_hp(rng, 37, 41);
    } else {
        m.set_hp(rng, 35, 39);
    }
    if asc >= 2 {
        with_slots(m, &[17, 12])
    } else {
        with_slots(m, &[15, 10])
    }
}

pub fn bandit_pointy(rng: &mut StsRandom, asc: i32, x: f32) -> Monster {
    let mut m = Monster::new(MonsterId::BanditPointy, 30, x, Ai::BanditPointy);
    let hp = if asc >= 7 { 34 } else { 30 };
    m.set_hp(rng, hp, hp);
    with_slots(m, &[if asc >= 2 { 6 } else { 5 }])
}

// ---- Behaviour ----

impl Combat {
    fn alive_count(&self) -> usize {
        self.order.iter().filter(|&&i| !self.monsters[i].is_dying && !self.monsters[i].is_escaping).count()
    }

    /// `usePreBattleAction()` for Act 2 monsters.
    pub(crate) fn city_pre_battle(&mut self, m: usize) {
        let me = Cr::Monster(m);
        match self.monsters[m].ai.clone() {
            Ai::SphericGuardian { .. } => {
                self.buff_self(m, Power::flag(PowerId::Barricade));
                self.buff_self(m, Power::new(PowerId::Artifact, 3));
                self.push_bot(Action::GainBlock { target: me, amount: 40 });
            }
            Ai::ShelledParasite { .. } => {
                self.buff_self(m, Power::new(PowerId::PlatedArmor, 14));
                self.push_bot(Action::GainBlock { target: me, amount: 14 });
            }
            Ai::Byrd { flight_amt, .. } => self.buff_self(m, Power::new(PowerId::Flight, flight_amt)),
            Ai::Mugger { gold_amt, .. } => self.buff_self(m, Power::new(PowerId::Thievery, gold_amt)),
            Ai::SnakePlant => self.buff_self(m, Power::new(PowerId::Malleable, 3)),
            Ai::GremlinLeader { str_amt, block_amt, .. } => {
                // gremlins[0..2] are the group's first two monsters; the third slot starts empty.
                let g0 = self.order.first().copied();
                let g1 = self.order.get(1).copied();
                self.monsters[m].ai = Ai::GremlinLeader { str_amt, block_amt, gremlins: [g0, g1, None] };
                for g in [g0, g1].into_iter().flatten() {
                    self.bot_power(Cr::Monster(g), Some(Cr::Monster(g)), Power::flag(PowerId::Minion));
                }
            }
            Ai::BookOfStabbing { .. } => self.buff_self(m, Power::flag(PowerId::PainfulStabs)),
            Ai::BronzeAutomaton { .. } => self.buff_self(m, Power::new(PowerId::Artifact, 3)),
            _ => self.beyond_pre_battle(m),
        }
    }

    /// `getMove(num)` for Act 2 monsters.
    pub(crate) fn city_get_move(&mut self, m: usize, num: i32) {
        let asc = self.ascension;
        let mon = &self.monsters[m];
        let d0 = mon.damage.first().map_or(-1, |s| s.base);
        let d1 = mon.damage.get(1).map_or(-1, |s| s.base);
        let d2 = mon.damage.get(2).map_or(-1, |s| s.base);
        let mv = match mon.ai.clone() {
            Ai::SphericGuardian { first_move, second_move } => {
                if first_move {
                    self.monsters[m].ai = Ai::SphericGuardian { first_move: false, second_move };
                    EnemyMove::simple(2, Intent::Defend)
                } else if second_move {
                    self.monsters[m].ai = Ai::SphericGuardian { first_move, second_move: false };
                    EnemyMove::attack(4, Intent::AttackDebuff, d0)
                } else if self.monsters[m].last_move(1) {
                    EnemyMove::attack(3, Intent::AttackDefend, d0)
                } else {
                    EnemyMove::new(1, ATTACK, d0, 2, true)
                }
            }
            Ai::Chosen { first_turn, used_hex } => {
                let zap = EnemyMove::attack(1, ATTACK, d0);
                let drain = EnemyMove::simple(2, Intent::Debuff);
                let debilitate = EnemyMove::attack(3, Intent::AttackDebuff, d1);
                let hex = EnemyMove::simple(4, Intent::StrongDebuff);
                let poke = EnemyMove::new(5, ATTACK, d2, 2, true);
                let mon = &self.monsters[m];
                if asc < 17 && first_turn {
                    self.monsters[m].ai = Ai::Chosen { first_turn: false, used_hex };
                    poke
                } else if !used_hex {
                    self.monsters[m].ai = Ai::Chosen { first_turn, used_hex: true };
                    hex
                } else if !mon.last_move(3) && !mon.last_move(2) {
                    if num < 50 { debilitate } else { drain }
                } else if num < 40 {
                    zap
                } else {
                    poke
                }
            }
            Ai::ShelledParasite { first_move } => {
                let double = EnemyMove::new(2, ATTACK, d0, 2, true);
                let fell = EnemyMove::attack(1, Intent::AttackDebuff, d1);
                let suck = EnemyMove::attack(3, Intent::AttackBuff, d2);
                if first_move {
                    self.monsters[m].ai = Ai::ShelledParasite { first_move: false };
                    if asc >= 17 {
                        fell
                    } else if self.rb() {
                        double
                    } else {
                        suck
                    }
                } else if num < 20 {
                    if !self.monsters[m].last_move(1) {
                        fell
                    } else {
                        let n = self.rng.ai.random_range(20, 99);
                        return self.city_get_move(m, n);
                    }
                } else if num < 60 {
                    if !self.monsters[m].last_two_moves(2) { double } else { suck }
                } else if !self.monsters[m].last_two_moves(3) {
                    suck
                } else {
                    double
                }
            }
            Ai::Byrd { first_move, is_flying, peck_count, flight_amt } => {
                let peck = EnemyMove::new(1, ATTACK, d0, peck_count, true);
                let swoop = EnemyMove::attack(3, ATTACK, d1);
                let caw = EnemyMove::simple(6, Intent::Buff);
                if first_move {
                    self.monsters[m].ai = Ai::Byrd { first_move: false, is_flying, peck_count, flight_amt };
                    if self.rbc(0.375) { caw } else { peck }
                } else if is_flying {
                    if num < 50 {
                        if self.monsters[m].last_two_moves(1) {
                            if self.rbc(0.4) { swoop } else { caw }
                        } else {
                            peck
                        }
                    } else if num < 70 {
                        if self.monsters[m].last_move(3) {
                            if self.rbc(0.375) { caw } else { peck }
                        } else {
                            swoop
                        }
                    } else if self.monsters[m].last_move(6) {
                        if self.rbc(0.2857) { swoop } else { peck }
                    } else {
                        caw
                    }
                } else {
                    let d2 = self.monsters[m].damage[2].base;
                    EnemyMove::attack(5, ATTACK, d2)
                }
            }
            Ai::Mugger { .. } => EnemyMove::attack(1, ATTACK, d0),
            Ai::SnakePlant => {
                let chomp = EnemyMove::new(1, ATTACK, d0, 3, true);
                let spores = EnemyMove::simple(2, Intent::StrongDebuff);
                let mon = &self.monsters[m];
                if num < 65 {
                    if mon.last_two_moves(1) { spores } else { chomp }
                } else if asc >= 17 {
                    if !mon.last_move(2) && !mon.last_move_before(2) { spores } else { chomp }
                } else if mon.last_move(2) {
                    chomp
                } else {
                    spores
                }
            }
            Ai::Snecko { first_turn } => {
                if first_turn {
                    self.monsters[m].ai = Ai::Snecko { first_turn: false };
                    EnemyMove::simple(1, Intent::StrongDebuff)
                } else if num < 40 || self.monsters[m].last_two_moves(2) {
                    EnemyMove::attack(3, Intent::AttackDebuff, d1)
                } else {
                    EnemyMove::attack(2, ATTACK, d0)
                }
            }
            Ai::Centurion { fury_hits, .. } => {
                let protect_or_fury = |c: &Combat| {
                    if c.alive_count() > 1 {
                        EnemyMove::simple(2, Intent::Defend)
                    } else {
                        EnemyMove::new(3, ATTACK, d1, fury_hits, true)
                    }
                };
                let mon = &self.monsters[m];
                if num >= 65 && !mon.last_two_moves(2) && !mon.last_two_moves(3) {
                    protect_or_fury(self)
                } else if !mon.last_two_moves(1) {
                    EnemyMove::attack(1, ATTACK, d0)
                } else {
                    protect_or_fury(self)
                }
            }
            Ai::Healer { .. } => {
                let need: i32 = self
                    .order
                    .iter()
                    .map(|&i| &self.monsters[i])
                    .filter(|x| !x.is_dying && !x.is_escaping)
                    .map(|x| x.max_hp - x.hp)
                    .sum();
                let mon = &self.monsters[m];
                let attack = EnemyMove::attack(1, Intent::AttackDebuff, d0);
                let heal_threshold = if asc >= 17 { 20 } else { 15 };
                let can_attack = if asc >= 17 { !mon.last_move(1) } else { !mon.last_two_moves(1) };
                if need > heal_threshold && !mon.last_two_moves(2) {
                    EnemyMove::simple(2, Intent::Buff)
                } else if num >= 40 && can_attack {
                    attack
                } else if !mon.last_two_moves(3) {
                    EnemyMove::simple(3, Intent::Buff)
                } else {
                    attack
                }
            }
            Ai::Taskmaster { .. } => EnemyMove::attack(2, Intent::AttackDebuff, 7),
            Ai::GremlinLeader { .. } => {
                let alive = self.order.iter().filter(|&&i| i != m && !self.monsters[i].is_dying).count();
                let rally = EnemyMove::simple(2, Intent::Unknown);
                let encourage = EnemyMove::simple(3, Intent::DefendBuff);
                let stab = EnemyMove::new(4, ATTACK, 6, 3, true);
                let mon = &self.monsters[m];
                if alive == 0 {
                    if num < 75 {
                        if !mon.last_move(2) { rally } else { stab }
                    } else if !mon.last_move(4) {
                        stab
                    } else {
                        rally
                    }
                } else if alive < 2 {
                    if num < 50 {
                        if !mon.last_move(2) {
                            rally
                        } else {
                            let n = self.rng.ai.random_range(50, 99);
                            return self.city_get_move(m, n);
                        }
                    } else if num < 80 {
                        if !mon.last_move(3) { encourage } else { stab }
                    } else if !mon.last_move(4) {
                        stab
                    } else {
                        let n = self.rng.ai.random_range(0, 80);
                        return self.city_get_move(m, n);
                    }
                } else if num < 66 {
                    if !mon.last_move(3) { encourage } else { stab }
                } else if !mon.last_move(4) {
                    stab
                } else {
                    encourage
                }
            }
            Ai::BookOfStabbing { stab_count } => {
                let mut sc = stab_count;
                let mon = &self.monsters[m];
                let mv = if num < 15 {
                    if mon.last_move(2) {
                        sc += 1;
                        EnemyMove::new(1, ATTACK, d0, sc, true)
                    } else {
                        if asc >= 18 {
                            sc += 1;
                        }
                        EnemyMove::attack(2, ATTACK, d1)
                    }
                } else if mon.last_two_moves(1) {
                    if asc >= 18 {
                        sc += 1;
                    }
                    EnemyMove::attack(2, ATTACK, d1)
                } else {
                    sc += 1;
                    EnemyMove::new(1, ATTACK, d0, sc, true)
                };
                self.monsters[m].ai = Ai::BookOfStabbing { stab_count: sc };
                mv
            }
            Ai::Champ { str_amt, forge_amt, block_amt, num_turns, forge_times, forge_threshold, threshold_reached, first_turn } => {
                let mut nt = num_turns + 1;
                let mut ft = forge_times;
                let mut tr = threshold_reached;
                let mon = &self.monsters[m];
                let slap = EnemyMove::attack(4, Intent::AttackDebuff, d2);
                let forge_roll = if asc >= 19 { 30 } else { 15 };
                let mv = if mon.hp < mon.max_hp / 2 && !tr {
                    tr = true;
                    EnemyMove::simple(7, Intent::Buff)
                } else if !mon.last_move(3) && !mon.last_move_before(3) && tr {
                    EnemyMove::new(3, ATTACK, d1, 2, true)
                } else if nt == 4 && !tr {
                    nt = 0;
                    EnemyMove::simple(6, Intent::Debuff)
                } else if !mon.last_move(2) && ft < forge_threshold && num <= forge_roll {
                    ft += 1;
                    EnemyMove::simple(2, Intent::DefendBuff)
                } else if !mon.last_move(5) && !mon.last_move(2) && num <= 30 {
                    EnemyMove::simple(5, Intent::Buff)
                } else if !mon.last_move(4) && num <= 55 {
                    slap
                } else if !mon.last_move(1) {
                    EnemyMove::attack(1, ATTACK, d0)
                } else {
                    slap
                };
                self.monsters[m].ai =
                    Ai::Champ { str_amt, forge_amt, block_amt, num_turns: nt, forge_times: ft, forge_threshold, threshold_reached: tr, first_turn };
                mv
            }
            Ai::TheCollector { turns_taken, ult_used, initial_spawn, slots, .. } => {
                let minion_dead = slots.iter().flatten().any(|&i| self.monsters[i].is_dying);
                let mon = &self.monsters[m];
                if initial_spawn {
                    EnemyMove::simple(1, Intent::Unknown)
                } else if turns_taken >= 3 && !ult_used {
                    EnemyMove::simple(4, Intent::StrongDebuff)
                } else if num <= 25 && minion_dead && !mon.last_move(5) {
                    EnemyMove::simple(5, Intent::Unknown)
                } else if num <= 70 && !mon.last_two_moves(2) {
                    EnemyMove::attack(2, ATTACK, d0)
                } else if !mon.last_move(3) {
                    EnemyMove::simple(3, Intent::DefendBuff)
                } else {
                    EnemyMove::attack(2, ATTACK, d0)
                }
            }
            Ai::TorchHead => EnemyMove::attack(1, ATTACK, 7),
            Ai::BronzeAutomaton { str_amt, block_amt, num_turns, first_turn } => {
                let mon = &self.monsters[m];
                let mut nt = num_turns;
                let mv = if first_turn {
                    EnemyMove::simple(4, Intent::Unknown)
                } else if nt == 4 {
                    nt = 0;
                    EnemyMove::attack(2, ATTACK, d1)
                } else if mon.last_move(2) {
                    if asc >= 19 {
                        EnemyMove::simple(5, Intent::DefendBuff)
                    } else {
                        EnemyMove::simple(3, Intent::Stun)
                    }
                } else {
                    nt += 1;
                    if !mon.last_move(3) && !mon.last_move(5) && !mon.last_move(4) {
                        EnemyMove::simple(5, Intent::DefendBuff)
                    } else {
                        EnemyMove::new(1, ATTACK, d0, 2, true)
                    }
                };
                self.monsters[m].ai = Ai::BronzeAutomaton { str_amt, block_amt, num_turns: nt, first_turn: false };
                mv
            }
            Ai::BronzeOrb { used_stasis, count, stasis_card } => {
                let mon = &self.monsters[m];
                if !used_stasis && num >= 25 {
                    self.monsters[m].ai = Ai::BronzeOrb { used_stasis: true, count, stasis_card };
                    EnemyMove::simple(3, Intent::StrongDebuff)
                } else if num >= 70 && !mon.last_two_moves(2) {
                    EnemyMove::simple(2, Intent::Defend)
                } else if !mon.last_two_moves(1) {
                    EnemyMove::attack(1, ATTACK, 8)
                } else {
                    EnemyMove::simple(2, Intent::Defend)
                }
            }
            Ai::BanditBear { .. } => EnemyMove::simple(2, Intent::StrongDebuff),
            Ai::BanditLeader { .. } => EnemyMove::simple(2, Intent::Unknown),
            Ai::BanditPointy => EnemyMove::new(1, ATTACK, d0, 2, true),
            _ => return self.beyond_get_move(m, num),
        };
        self.mset(m, mv);
    }

    /// `takeTurn()` for Act 2 monsters.
    pub(crate) fn city_take_turn(&mut self, m: usize) {
        let asc = self.ascension;
        let next = self.monsters[m].next_move;
        let me = Cr::Monster(m);
        match self.monsters[m].ai.clone() {
            Ai::SphericGuardian { .. } => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        self.mattack(m, 0);
                    }
                    2 => self.push_bot(Action::GainBlock { target: me, amount: if asc >= 17 { 35 } else { 25 } }),
                    3 => {
                        self.push_bot(Action::GainBlock { target: me, amount: 15 });
                        self.mattack(m, 0);
                    }
                    4 => {
                        self.mattack(m, 0);
                        self.debuff_player(m, Power::frail(5, true));
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Chosen { .. } => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        self.debuff_player(m, Power::weak(3, true));
                        self.buff_self(m, Power::new(PowerId::Strength, 3));
                    }
                    3 => {
                        self.mattack(m, 1);
                        let p = Power::vulnerable(2, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                    }
                    4 => self.debuff_player(m, Power::new(PowerId::Hex, 1)),
                    5 => {
                        self.mattack(m, 2);
                        self.mattack(m, 2);
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::ShelledParasite { .. } => {
                match next {
                    1 => {
                        self.mattack(m, 1);
                        self.debuff_player(m, Power::frail(2, true));
                    }
                    2 => {
                        self.mattack(m, 0);
                        self.mattack(m, 0);
                    }
                    3 => self.push_bot(Action::VampireDamage { target: Cr::Player, source: DamageSource::MonsterSlot { monster: m, slot: 2 } }),
                    4 => {
                        let d1 = self.base(m, 1);
                        self.mset(m, EnemyMove::attack(1, Intent::AttackDebuff, d1));
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Byrd { first_move, peck_count, flight_amt, .. } => {
                match next {
                    1 => {
                        for _ in 0..peck_count {
                            self.mattack(m, 0);
                        }
                    }
                    2 => {
                        self.monsters[m].ai = Ai::Byrd { first_move, is_flying: true, peck_count, flight_amt };
                        self.buff_self(m, Power::new(PowerId::Flight, flight_amt));
                    }
                    3 => self.mattack(m, 1),
                    5 => {
                        self.mattack(m, 2);
                        self.mset(m, EnemyMove::simple(2, Intent::Unknown));
                        return;
                    }
                    6 => self.buff_self(m, Power::new(PowerId::Strength, 1)),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Mugger { gold_amt, slash_count, escape_def, .. } => match next {
                1 => {
                    // playSfx and the line of dialogue both roll aiRng.
                    self.rng.ai.random(2);
                    if slash_count == 1 {
                        let _ = self.rbc(0.6);
                    }
                    self.push_bot(Action::LooterRecordGold { monster: m });
                    self.push_bot(Action::Damage { target: Cr::Player, source: DamageSource::MonsterSlot { monster: m, slot: 0 }, steal_gold: gold_amt });
                    let sc = slash_count + 1;
                    self.set_mugger_slash(m, sc);
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
                    let amount = if asc >= 17 { escape_def + 6 } else { escape_def };
                    self.push_bot(Action::GainBlock { target: me, amount });
                    self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(3, Intent::Escape) });
                }
                3 => {
                    self.mugged = true;
                    self.push_bot(Action::Escape { monster: m });
                    self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::simple(3, Intent::Escape) });
                }
                4 => {
                    self.rng.ai.random(2);
                    self.set_mugger_slash(m, slash_count + 1);
                    self.push_bot(Action::LooterRecordGold { monster: m });
                    self.push_bot(Action::Damage { target: Cr::Player, source: DamageSource::MonsterSlot { monster: m, slot: 1 }, steal_gold: gold_amt });
                    self.mset(m, EnemyMove::simple(2, Intent::Defend));
                }
                _ => {}
            },
            Ai::SnakePlant => {
                match next {
                    1 => {
                        for _ in 0..3 {
                            self.mattack(m, 0);
                        }
                    }
                    2 => {
                        self.debuff_player(m, Power::frail(2, true));
                        self.debuff_player(m, Power::weak(2, true));
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Snecko { .. } => {
                match next {
                    1 => self.debuff_player(m, Power::flag(PowerId::Confusion)),
                    2 => self.mattack(m, 0),
                    3 => {
                        self.mattack(m, 1);
                        if asc >= 17 {
                            self.debuff_player(m, Power::weak(2, true));
                        }
                        let p = Power::vulnerable(2, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Centurion { block_amount, fury_hits } => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => self.push_bot(Action::GainBlockRandomMonster { source: m, amount: block_amount }),
                    3 => {
                        for _ in 0..fury_hits {
                            self.mattack(m, 1);
                        }
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Healer { str_amt, heal_amt } => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        self.debuff_player(m, Power::frail(2, true));
                    }
                    2 => {
                        for i in self.order.clone() {
                            if !self.monsters[i].is_dying && !self.monsters[i].is_escaping {
                                self.push_bot(Action::Heal { target: Cr::Monster(i), amount: heal_amt });
                            }
                        }
                    }
                    3 => {
                        for i in self.order.clone() {
                            if !self.monsters[i].is_dying && !self.monsters[i].is_escaping {
                                self.bot_power(Cr::Monster(i), Some(me), Power::new(PowerId::Strength, str_amt));
                            }
                        }
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Taskmaster { wound_count } => {
                if next == 2 {
                    self.mattack(m, 1);
                    self.temp_card_discard(CardId::Wound, wound_count);
                    if asc >= 18 {
                        self.buff_self(m, Power::new(PowerId::Strength, 1));
                    }
                }
                self.mroll(m);
            }
            Ai::GremlinLeader { str_amt, block_amt, .. } => {
                match next {
                    2 => {
                        self.summon_gremlin(m);
                        self.summon_gremlin(m);
                    }
                    3 => {
                        // getEncourageQuote picks its line with aiRng.
                        self.rng.ai.random_range(0, 2);
                        for i in self.order.clone() {
                            if i == m {
                                self.bot_power(me, Some(me), Power::new(PowerId::Strength, str_amt));
                            } else if !self.monsters[i].is_dying {
                                self.bot_power(Cr::Monster(i), Some(me), Power::new(PowerId::Strength, str_amt));
                                self.push_bot(Action::GainBlock { target: Cr::Monster(i), amount: block_amt });
                            }
                        }
                    }
                    4 => {
                        for _ in 0..3 {
                            self.mattack(m, 0);
                        }
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::BookOfStabbing { stab_count } => {
                match next {
                    1 => {
                        for _ in 0..stab_count {
                            self.mattack(m, 0);
                        }
                    }
                    2 => self.mattack(m, 1),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::Champ { str_amt, forge_amt, block_amt, .. } => {
                if let Ai::Champ { first_turn, .. } = &mut self.monsters[m].ai {
                    *first_turn = false;
                }
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        self.push_bot(Action::GainBlock { target: me, amount: block_amt });
                        self.buff_self(m, Power::new(PowerId::Metallicize, forge_amt));
                    }
                    3 => {
                        self.mattack(m, 1);
                        self.mattack(m, 1);
                    }
                    4 => {
                        self.mattack(m, 2);
                        self.debuff_player(m, Power::frail(2, true));
                        let p = Power::vulnerable(2, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                    }
                    5 => self.buff_self(m, Power::new(PowerId::Strength, str_amt)),
                    6 => {
                        self.debuff_player(m, Power::weak(2, true));
                        let p = Power::vulnerable(2, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                    }
                    7 => {
                        self.push_bot(Action::RemoveDebuffs { target: me });
                        self.push_bot(Action::RemovePower { target: me, id: PowerId::Shackled });
                        self.buff_self(m, Power::new(PowerId::Strength, str_amt * 3));
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::TheCollector { block_amt, str_amt, mega_debuff_amt, .. } => {
                match next {
                    1 => {
                        for i in 1..3 {
                            let x = -100.0 + -185.0 * i as f32;
                            let t = torch_head(&mut self.rng.monster_hp, asc, x);
                            let id = self.prepare_monster(t);
                            if let Ai::TheCollector { slots, .. } = &mut self.monsters[m].ai {
                                slots[i - 1] = Some(id);
                            }
                            self.push_bot(Action::SpawnPrepared { monster: id, minion: true });
                        }
                        if let Ai::TheCollector { initial_spawn, .. } = &mut self.monsters[m].ai {
                            *initial_spawn = false;
                        }
                    }
                    2 => self.mattack(m, 0),
                    3 => {
                        let amount = if asc >= 19 { block_amt + 5 } else { block_amt };
                        self.push_bot(Action::GainBlock { target: me, amount });
                        for i in self.order.clone() {
                            if !self.monsters[i].is_dying && !self.monsters[i].is_escaping {
                                self.bot_power(Cr::Monster(i), Some(me), Power::new(PowerId::Strength, str_amt));
                            }
                        }
                    }
                    4 => {
                        self.debuff_player(m, Power::weak(mega_debuff_amt, true));
                        let p = Power::vulnerable(mega_debuff_amt, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                        self.debuff_player(m, Power::frail(mega_debuff_amt, true));
                        if let Ai::TheCollector { ult_used, .. } = &mut self.monsters[m].ai {
                            *ult_used = true;
                        }
                    }
                    5 => {
                        for i in 1..3 {
                            let Ai::TheCollector { slots, .. } = &self.monsters[m].ai else { break };
                            let Some(old) = slots[i - 1] else { continue };
                            if self.monsters[old].is_dying {
                                let x = -100.0 + -185.0 * i as f32;
                                let t = torch_head(&mut self.rng.monster_hp, asc, x);
                                let id = self.prepare_monster(t);
                                if let Ai::TheCollector { slots, .. } = &mut self.monsters[m].ai {
                                    slots[i - 1] = Some(id);
                                }
                                self.push_bot(Action::SpawnPrepared { monster: id, minion: true });
                            }
                        }
                    }
                    _ => {}
                }
                if let Ai::TheCollector { turns_taken, .. } = &mut self.monsters[m].ai {
                    *turns_taken += 1;
                }
                self.mroll(m);
            }
            Ai::TorchHead => {
                if next == 1 {
                    self.mattack(m, 0);
                    self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(1, ATTACK, 7) });
                }
            }
            Ai::BronzeAutomaton { str_amt, block_amt, .. } => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        self.mattack(m, 0);
                    }
                    2 => self.mattack(m, 1),
                    4 => {
                        for (x, count) in [(-300.0, 0), (200.0, 1)] {
                            let orb = bronze_orb(&mut self.rng.monster_hp, asc, x, count);
                            let id = self.prepare_monster(orb);
                            self.push_bot(Action::SpawnPrepared { monster: id, minion: true });
                        }
                    }
                    5 => {
                        self.push_bot(Action::GainBlock { target: me, amount: block_amt });
                        self.buff_self(m, Power::new(PowerId::Strength, str_amt));
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::BronzeOrb { .. } => {
                match next {
                    1 => self.mattack(m, 0),
                    2 => {
                        if let Some(a) = self.order.iter().copied().find(|&i| self.monsters[i].id == MonsterId::BronzeAutomaton) {
                            self.push_bot(Action::GainBlock { target: Cr::Monster(a), amount: 12 });
                        }
                    }
                    3 => self.push_bot(Action::ApplyStasis { monster: m }),
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::BanditBear { con_reduction } => match next {
                1 => {
                    self.mattack(m, 0);
                    let d1 = self.base(m, 1);
                    self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(3, Intent::AttackDefend, d1) });
                }
                2 => {
                    self.debuff_player(m, Power::new(PowerId::Dexterity, con_reduction));
                    let d1 = self.base(m, 1);
                    self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(3, Intent::AttackDefend, d1) });
                }
                3 => {
                    self.mattack(m, 1);
                    self.push_bot(Action::GainBlock { target: me, amount: 9 });
                    let d0 = self.base(m, 0);
                    self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(1, ATTACK, d0) });
                }
                _ => {}
            },
            Ai::BanditLeader { weak_amount } => {
                let d0 = self.base(m, 0);
                let d1 = self.base(m, 1);
                match next {
                    1 => {
                        self.mattack(m, 0);
                        if asc >= 17 && !self.monsters[m].last_two_moves(1) {
                            self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(1, ATTACK, d0) });
                        } else {
                            self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(3, Intent::AttackDebuff, d1) });
                        }
                    }
                    2 => self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(3, Intent::AttackDebuff, d1) }),
                    3 => {
                        self.mattack(m, 1);
                        self.debuff_player(m, Power::weak(weak_amount, true));
                        self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::attack(1, ATTACK, d0) });
                    }
                    _ => {}
                }
            }
            Ai::BanditPointy => {
                self.mattack(m, 0);
                self.mattack(m, 0);
                let d0 = self.base(m, 0);
                self.push_bot(Action::SetMove { monster: m, mv: EnemyMove::new(1, ATTACK, d0, 2, true) });
            }
            _ => self.beyond_take_turn(m),
        }
    }

    fn set_mugger_slash(&mut self, m: usize, v: i32) {
        if let Ai::Mugger { slash_count, .. } = &mut self.monsters[m].ai {
            *slash_count = v;
        }
    }

    /// Adds a monster built during a turn to `monsters` (not yet to the group) so its summoner
    /// can refer to it before the spawn action runs.
    pub(super) fn prepare_monster(&mut self, mut mon: Monster) -> usize {
        mon.next_move = super::NO_MOVE;
        self.monsters.push(mon);
        self.monsters.len() - 1
    }

    /// The `SummonGremlinAction` constructor: picks the slot and the gremlin right away.
    fn summon_gremlin(&mut self, leader: usize) {
        let Ai::GremlinLeader { gremlins, .. } = self.monsters[leader].ai.clone() else { return };
        let Some(slot) = gremlins.iter().position(|g| g.map_or(true, |i| self.monsters[i].is_dying)) else { return };
        let pool = ["GremlinWarrior", "GremlinWarrior", "GremlinThief", "GremlinThief", "GremlinFat", "GremlinFat", "GremlinTsundere", "GremlinWizard"];
        let key = pool[self.rng.ai.random_range(0, pool.len() as i32 - 1) as usize];
        let asc = self.ascension;
        let g = crate::encounters::gremlin(&mut self.rng, asc, key, GREMLIN_LEADER_X[slot]);
        let id = self.prepare_monster(g);
        self.relics_on_spawn_monster(id);
        if let Ai::GremlinLeader { gremlins, .. } = &mut self.monsters[leader].ai {
            gremlins[slot] = Some(id);
        }
        self.push_bot(Action::SummonGremlin { monster: id });
    }

    /// `ApplyStasisAction.update`: takes the rarest card from the draw pile (or the discard pile).
    pub(crate) fn exec_apply_stasis(&mut self, m: usize) {
        if self.draw_pile.is_empty() && self.discard_pile.is_empty() {
            return;
        }
        let from_draw = !self.draw_pile.is_empty();
        let pile = if from_draw { &self.draw_pile } else { &self.discard_pile };
        let mut pick = None;
        for rarity in [CardRarity::Rare, CardRarity::Uncommon, CardRarity::Common] {
            // getRandomCard(rng, rarity) sorts the matches by card ID first.
            let mut tmp: Vec<usize> = (0..pile.len()).filter(|&i| pile[i].data().rarity == rarity).collect();
            if tmp.is_empty() {
                continue;
            }
            tmp.sort_by(|&a, &b| pile[a].data().java_id.cmp(pile[b].data().java_id));
            let k = self.rng.card_random.random(tmp.len() as i32 - 1) as usize;
            pick = Some(tmp[k]);
            break;
        }
        let idx = match pick {
            Some(i) => i,
            None => self.rng.card_random.random(pile.len() as i32 - 1) as usize,
        };
        let card = if from_draw { self.draw_pile.remove(idx) } else { self.discard_pile.remove(idx) };
        if let Ai::BronzeOrb { stasis_card, .. } = &mut self.monsters[m].ai {
            *stasis_card = Some(card);
        }
        self.top_power(Cr::Monster(m), Some(Cr::Monster(m)), Power::flag(PowerId::Stasis));
    }

    /// `changeState` for Act 2 monsters.
    pub(crate) fn city_change_state(&mut self, m: usize, state: MonsterState) {
        match (self.monsters[m].ai.clone(), state) {
            (Ai::ShelledParasite { .. }, MonsterState::ArmorBreak) => {
                self.mset(m, EnemyMove::simple(4, Intent::Stun));
                self.create_intent(m);
            }
            (Ai::Byrd { first_move, peck_count, flight_amt, .. }, MonsterState::Grounded) => {
                self.mset(m, EnemyMove::simple(4, Intent::Stun));
                self.create_intent(m);
                self.monsters[m].ai = Ai::Byrd { first_move, is_flying: false, peck_count, flight_amt };
            }
            _ => self.beyond_change_state(m, state),
        }
    }

    /// Subclass `die()` overrides that run before `super.die()`.
    pub(crate) fn city_before_die(&mut self, m: usize) {
        if let Ai::Mugger { .. } = self.monsters[m].ai {
            // playDeathSfx rolls aiRng.
            self.rng.ai.random(2);
        }
    }

    /// Subclass `die()` overrides that run after `super.die()`.
    pub(crate) fn city_after_die(&mut self, m: usize) {
        match self.monsters[m].ai {
            Ai::GremlinLeader { .. } => {
                for i in self.order.clone() {
                    if !self.monsters[i].is_dying {
                        self.push_bot(Action::Escape { monster: i });
                    }
                }
            }
            Ai::TheCollector { .. } | Ai::BronzeAutomaton { .. } => {
                for i in self.order.clone() {
                    if !self.monsters[i].is_dying {
                        self.push_top(Action::Suicide { monster: i, trigger_relics: true });
                    }
                }
            }
            _ => {}
        }
    }
}
