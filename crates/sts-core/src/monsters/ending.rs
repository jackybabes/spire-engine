//! Act 4 (The Ending) monsters: Spire Shield, Spire Spear and the Corrupt Heart.

use sts_rng::StsRandom;

use super::{Ai, DamageSlot, EnemyMove, EnemyType, Intent, Monster, MonsterId};
use crate::action::{Action, Cr};
use crate::cards::CardId;
use crate::combat::Combat;
use crate::powers::{Power, PowerId};

const ATTACK: Intent = Intent::Attack;

fn with_slots(mut m: Monster, slots: &[i32]) -> Monster {
    m.damage = slots.iter().map(|&b| DamageSlot::new(b)).collect();
    m
}

pub fn spire_shield(rng: &mut StsRandom, asc: i32) -> Monster {
    let mut m = Monster::new(MonsterId::SpireShield, 110, -1000.0, Ai::SpireShield { move_count: 0 });
    m.enemy_type = EnemyType::Elite;
    let hp = if asc >= 8 { 125 } else { 110 };
    m.set_hp(rng, hp, hp);
    if asc >= 3 {
        with_slots(m, &[14, 38])
    } else {
        with_slots(m, &[12, 34])
    }
}

pub fn spire_spear(rng: &mut StsRandom, asc: i32) -> Monster {
    let skewer_count = if asc >= 3 { 4 } else { 3 };
    let mut m = Monster::new(MonsterId::SpireSpear, 160, 70.0, Ai::SpireSpear { move_count: 0, skewer_count });
    m.enemy_type = EnemyType::Elite;
    let hp = if asc >= 8 { 180 } else { 160 };
    m.set_hp(rng, hp, hp);
    if asc >= 3 {
        with_slots(m, &[6, 10])
    } else {
        with_slots(m, &[5, 10])
    }
}

pub fn corrupt_heart(rng: &mut StsRandom, asc: i32) -> Monster {
    let blood_hit_count = if asc >= 4 { 15 } else { 12 };
    let ai = Ai::CorruptHeart { is_first_move: true, move_count: 0, buff_count: 0, blood_hit_count };
    let mut m = Monster::new(MonsterId::CorruptHeart, 750, -50.0, ai);
    m.enemy_type = EnemyType::Boss;
    let hp = if asc >= 9 { 800 } else { 750 };
    m.set_hp(rng, hp, hp);
    with_slots(m, &[if asc >= 4 { 45 } else { 40 }, 2])
}

impl Combat {
    /// `usePreBattleAction()` for Act 4 monsters.
    pub(crate) fn ending_pre_battle(&mut self, m: usize) {
        let asc = self.ascension;
        match self.monsters[m].ai {
            Ai::SpireShield { .. } => {
                self.bot_power(Cr::Player, Some(Cr::Monster(m)), Power::flag(PowerId::Surrounded));
                self.buff_self(m, Power::new(PowerId::Artifact, if asc >= 18 { 2 } else { 1 }));
            }
            Ai::SpireSpear { .. } => self.buff_self(m, Power::new(PowerId::Artifact, if asc >= 18 { 2 } else { 1 })),
            Ai::CorruptHeart { .. } => {
                let invincible = if asc >= 19 { 200 } else { 300 };
                let beat = if asc >= 19 { 2 } else { 1 };
                self.buff_self(m, Power::new(PowerId::Invincible, invincible));
                self.buff_self(m, Power::new(PowerId::BeatOfDeath, beat));
            }
            _ => {}
        }
    }

    /// `getMove(num)` for Act 4 monsters.
    pub(crate) fn ending_get_move(&mut self, m: usize, _num: i32) {
        let mon = &self.monsters[m];
        let d0 = mon.damage.first().map_or(-1, |s| s.base);
        let d1 = mon.damage.get(1).map_or(-1, |s| s.base);
        let mv = match mon.ai.clone() {
            Ai::SpireShield { move_count } => {
                let bash = EnemyMove::attack(1, Intent::AttackDebuff, d0);
                let fortify = EnemyMove::simple(2, Intent::Defend);
                let mv = match move_count % 3 {
                    0 => {
                        if self.rb() {
                            fortify
                        } else {
                            bash
                        }
                    }
                    1 => {
                        if !self.monsters[m].last_move(1) {
                            bash
                        } else {
                            fortify
                        }
                    }
                    _ => EnemyMove::attack(3, Intent::AttackDefend, d1),
                };
                self.monsters[m].ai = Ai::SpireShield { move_count: move_count + 1 };
                mv
            }
            Ai::SpireSpear { move_count, skewer_count } => {
                let burn_strike = EnemyMove::new(1, Intent::AttackDebuff, d0, 2, true);
                let piercer = EnemyMove::simple(2, Intent::Buff);
                let mv = match move_count % 3 {
                    0 => {
                        if !self.monsters[m].last_move(1) {
                            burn_strike
                        } else {
                            piercer
                        }
                    }
                    1 => EnemyMove::new(3, ATTACK, d1, skewer_count, true),
                    _ => {
                        if self.rb() {
                            piercer
                        } else {
                            burn_strike
                        }
                    }
                };
                self.monsters[m].ai = Ai::SpireSpear { move_count: move_count + 1, skewer_count };
                mv
            }
            Ai::CorruptHeart { is_first_move, move_count, buff_count, blood_hit_count } => {
                let blood_shots = EnemyMove::new(1, ATTACK, d1, blood_hit_count, true);
                let echo = EnemyMove::attack(2, ATTACK, d0);
                if is_first_move {
                    self.monsters[m].ai = Ai::CorruptHeart { is_first_move: false, move_count, buff_count, blood_hit_count };
                    EnemyMove::simple(3, Intent::StrongDebuff)
                } else {
                    let mv = match move_count % 3 {
                        0 => {
                            if self.rb() {
                                blood_shots
                            } else {
                                echo
                            }
                        }
                        1 => {
                            if !self.monsters[m].last_move(2) {
                                echo
                            } else {
                                blood_shots
                            }
                        }
                        _ => EnemyMove::simple(4, Intent::Buff),
                    };
                    self.monsters[m].ai = Ai::CorruptHeart { is_first_move, move_count: move_count + 1, buff_count, blood_hit_count };
                    mv
                }
            }
            _ => return,
        };
        self.mset(m, mv);
    }

    /// `takeTurn()` for Act 4 monsters.
    pub(crate) fn ending_take_turn(&mut self, m: usize) {
        let asc = self.ascension;
        let next = self.monsters[m].next_move;
        let me = Cr::Monster(m);
        match self.monsters[m].ai.clone() {
            Ai::SpireShield { .. } => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        // Focus when the player has orb slots (empty ones count: `orbs.isEmpty()`)
                        // and aiRng says so, else Strength; no roll without orb slots. (It took
                        // Strength always, from the Ironclad's port: a Defect run's Act 4 drifted
                        // from the game's at the first Bash, 2026-10-25.)
                        let focus = !self.player.orbs.is_empty() && self.rng.ai.random_boolean();
                        self.debuff_player(m, Power::new(if focus { PowerId::Focus } else { PowerId::Strength }, -1));
                    }
                    2 => {
                        for i in self.order.clone() {
                            self.push_bot(Action::GainBlock { target: Cr::Monster(i), amount: 30 });
                        }
                    }
                    3 => {
                        self.mattack(m, 1);
                        let amount = if asc >= 18 { 99 } else { self.monsters[m].damage[1].output };
                        self.push_bot(Action::GainBlock { target: me, amount });
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::SpireSpear { skewer_count, .. } => {
                match next {
                    1 => {
                        self.mattack(m, 0);
                        self.mattack(m, 0);
                        let card = self.new_card(CardId::Burn);
                        if asc >= 18 {
                            self.push_bot(Action::MakeTempCardInDrawPile { card, amount: 2, random_spot: false, to_bottom: false });
                        } else {
                            self.push_bot(Action::MakeTempCardInDiscard { card, amount: 2 });
                        }
                    }
                    2 => {
                        for i in self.order.clone() {
                            self.bot_power(Cr::Monster(i), Some(me), Power::new(PowerId::Strength, 2));
                        }
                    }
                    3 => {
                        for _ in 0..skewer_count {
                            self.mattack(m, 1);
                        }
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            Ai::CorruptHeart { is_first_move, move_count, buff_count, blood_hit_count } => {
                match next {
                    1 => {
                        for _ in 0..blood_hit_count {
                            self.mattack(m, 1);
                        }
                    }
                    2 => self.mattack(m, 0),
                    3 => {
                        let p = Power::vulnerable(2, true, self.turn_has_ended);
                        self.debuff_player(m, p);
                        self.debuff_player(m, Power::weak(2, true));
                        self.debuff_player(m, Power::frail(2, true));
                        for id in [CardId::Dazed, CardId::Slimed, CardId::Wound, CardId::Burn, CardId::Void] {
                            let card = self.new_card(id);
                            self.push_bot(Action::MakeTempCardInDrawPile { card, amount: 1, random_spot: true, to_bottom: false });
                        }
                    }
                    4 => {
                        let str_now = self.monsters[m].power_amount(PowerId::Strength);
                        let additional = if str_now < 0 { -str_now } else { 0 };
                        self.buff_self(m, Power::new(PowerId::Strength, additional + 2));
                        match buff_count {
                            0 => self.buff_self(m, Power::new(PowerId::Artifact, 2)),
                            1 => self.buff_self(m, Power::new(PowerId::BeatOfDeath, 1)),
                            2 => self.buff_self(m, Power::flag(PowerId::PainfulStabs)),
                            3 => self.buff_self(m, Power::new(PowerId::Strength, 10)),
                            _ => self.buff_self(m, Power::new(PowerId::Strength, 50)),
                        }
                        self.monsters[m].ai = Ai::CorruptHeart { is_first_move, move_count, buff_count: buff_count + 1, blood_hit_count };
                    }
                    _ => {}
                }
                self.mroll(m);
            }
            _ => {}
        }
    }

    /// Spire Shield and Spear `die()`: the survivor stops surrounding the player.
    pub(crate) fn ending_after_die(&mut self, m: usize) {
        if !matches!(self.monsters[m].ai, Ai::SpireShield { .. } | Ai::SpireSpear { .. }) {
            return;
        }
        for i in self.order.clone() {
            if self.monsters[i].is_dying {
                continue;
            }
            if self.player.has_power(PowerId::Surrounded) {
                self.player_flipped = self.monsters[i].draw_x < -960.0;
                self.push_bot(Action::RemovePower { target: Cr::Player, id: PowerId::Surrounded });
            }
            if self.monsters[i].has_power(PowerId::BackAttack) {
                self.push_bot(Action::RemovePower { target: Cr::Monster(i), id: PowerId::BackAttack });
            }
        }
    }

    /// The player turns to face `target` while Surrounded (`playCard`, `PotionPopUp`), then
    /// the hand refresh re-applies the monsters' powers (`refreshHandLayout`).
    pub(crate) fn face_target(&mut self, target: usize) {
        if !self.player.has_power(PowerId::Surrounded) {
            return;
        }
        self.player_flipped = self.monsters[target].draw_x < -960.0;
        for i in self.order.clone() {
            self.monster_apply_powers(i);
            if !self.back_attack(i) && self.monsters[i].has_power(PowerId::BackAttack) {
                self.push_top(Action::RemovePower { target: Cr::Monster(i), id: PowerId::BackAttack });
            }
        }
    }
}
