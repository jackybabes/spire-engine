//! Damage, block, healing and power application (`AbstractCreature`, `DamageInfo`,
//! `ApplyPowerAction` and friends).

use super::engine::java_floor;
use super::Combat;
use crate::action::{Action, Cr, DamageInfo, DamageType};
use crate::cards::{Card, CardId, CardType};
use crate::powers::{Power, PowerId, PowerType};

/// `atDamageGive` for one power. `strength_mult` implements Heavy Blade's temporary multiply;
/// `weak` is Weak's multiplier for this owner (Paper Crane makes the monsters' 0.6).
fn at_damage_give(p: &Power, dmg: f32, t: DamageType, strength_mult: i32, weak: f32) -> f32 {
    match (p.id, t) {
        (PowerId::Strength, DamageType::Normal) => dmg + (p.amount * strength_mult) as f32,
        (PowerId::Vigor, DamageType::Normal) => dmg + p.amount as f32,
        (PowerId::Weakened, DamageType::Normal) => dmg * weak,
        (PowerId::PenNib, DamageType::Normal) => dmg * 2.0,
        (PowerId::DoubleDamage, DamageType::Normal) => dmg * 2.0,
        _ => dmg,
    }
}

/// `vuln` is Vulnerable's multiplier for this target (Paper Frog / Odd Mushroom change it).
fn at_damage_receive(p: &Power, dmg: f32, t: DamageType, vuln: f32) -> f32 {
    match (p.id, t) {
        (PowerId::Vulnerable, DamageType::Normal) => dmg * vuln,
        (PowerId::Slow, DamageType::Normal) => dmg * (1.0 + p.amount as f32 * 0.1),
        _ => dmg,
    }
}

fn at_damage_final_give(_p: &Power, dmg: f32, _t: DamageType) -> f32 {
    dmg
}

fn at_damage_final_receive(p: &Power, dmg: f32, t: DamageType) -> f32 {
    match p.id {
        PowerId::IntangiblePlayer | PowerId::Intangible if dmg > 1.0 => 1.0,
        PowerId::Flight if t == DamageType::Normal => dmg / 2.0,
        _ => dmg,
    }
}

fn modify_block(p: &Power, block: f32) -> f32 {
    match p.id {
        PowerId::Dexterity => (block + p.amount as f32).max(0.0),
        PowerId::Frail => block * 0.75,
        _ => block,
    }
}

/// How one play of an attack hits (`Combat::card_hits`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hits {
    /// Hits per play (0 for a card that isn't an attack).
    pub count: i32,
    /// Each hit lands on every enemy (Cleave, Whirlwind, Dagger Spray...).
    pub all: bool,
    /// Each hit lands on a random enemy (Sword Boomerang, Ragnarok...).
    pub random: bool,
}

impl Combat {
    // ---- Damage numbers ----

    /// `applyPowersToBlock`.
    pub(crate) fn card_block(&self, card: &Card) -> i32 {
        let base = match card.id {
            // Stack.applyPowers: the discard pile's size (+3 upgraded), or as kept (`applied_base`).
            CardId::Stack => card.applied_base.unwrap_or_else(|| self.card_block_base(card)),
            // (Spirit Shield's `use` counts again.)
            CardId::SpiritShield => self.card_block_base(card),
            _ => card.base_block,
        };
        self.block_from_base(base)
    }

    /// `card.applyPowers()` for a card that plays without being counted again (`Card::applied_base`):
    /// Stack's discard pile and Mind Blast's draw pile as of now, kept for the play. For a card
    /// taken from the top of the draw pile (`PlayTopCardAction`) and a copy played from limbo
    /// (`play_copy`). Body Slam, Blizzard and Spirit Shield count again in their `use`.
    pub(crate) fn apply_powers_kept(&self, card: &mut Card) {
        card.applied_base = match card.id {
            CardId::Stack => Some(self.card_block_base(card)),
            CardId::MindBlast => Some(self.card_damage_base(card)),
            _ => None,
        };
    }

    fn card_block_base(&self, card: &Card) -> i32 {
        match card.id {
            // Stack.applyPowers: the discard pile's size (+3 upgraded).
            CardId::Stack => self.discard_pile.len() as i32 + if card.upgraded { 3 } else { 0 },
            // Spirit Shield.applyPowers: its magic number per other card in hand.
            CardId::SpiritShield => self.hand.iter().filter(|c| c.instance != card.instance).count() as i32 * card.magic,
            _ => card.base_block,
        }
    }

    fn card_damage_base(&self, card: &Card) -> i32 {
        match card.id {
            CardId::BodySlam => self.player.block,
            CardId::MindBlast => self.draw_pile.len() as i32,
            CardId::Blizzard => self.orbs_channeled_this_combat.iter().filter(|&&o| o == super::OrbId::Frost).count() as i32 * card.magic,
            _ => card.base_damage,
        }
    }

    /// Halt's magic number: its block with 6 more (10 upgraded), through the same powers.
    pub(crate) fn halt_wrath_block(&self, card: &Card) -> i32 {
        self.block_from_base(card.base_block + 6 + card.times_upgraded * 4)
    }

    fn block_from_base(&self, base: i32) -> i32 {
        let mut tmp = base as f32;
        for p in &self.player.powers {
            tmp = modify_block(p, tmp);
        }
        // modifyBlockLast: No Block (Panic Button).
        if self.player.has_power(PowerId::NoBlock) {
            tmp = 0.0;
        }
        java_floor(tmp.max(0.0))
    }

    fn count_strikes(&self) -> i32 {
        self.hand
            .iter()
            .chain(self.draw_pile.iter())
            .chain(self.discard_pile.iter())
            .filter(|c| c.data().strike)
            .count() as i32
    }

    /// The base damage `calculateCardDamage` starts from, including per-card overrides and
    /// relics' `atDamageModify`.
    fn card_effective_base(&self, card: &Card) -> f32 {
        let base = match card.id {
            // Body Slam's, Mind Blast's and Blizzard's counts as their `applyPowers` takes them
            // (`card_damage_base`), or Mind Blast's as kept (`applied_base`).
            CardId::MindBlast => card.applied_base.unwrap_or_else(|| self.card_damage_base(card)),
            CardId::BodySlam | CardId::Blizzard => self.card_damage_base(card),
            CardId::PerfectedStrike => card.base_damage + card.magic * self.count_strikes(),
            // Brilliance: the Mantra made this combat on top.
            CardId::Brilliance => card.base_damage + self.mantra_gained,
            _ => card.base_damage,
        };
        self.relics_at_damage_modify(base as f32, card)
    }

    /// Vulnerable's multiplier on monsters (Paper Frog) and on the player (Odd Mushroom).
    fn monster_vuln(&self) -> f32 {
        if self.has_relic(crate::relics::RelicId::PaperFrog) {
            1.75
        } else {
            1.5
        }
    }

    fn player_vuln(&self) -> f32 {
        if self.has_relic(crate::relics::RelicId::OddMushroom) {
            1.25
        } else {
            1.5
        }
    }

    /// `AbstractCard.calculateCardDamage(mo)`. Updates `damage`/`block` and returns the multi-damage
    /// array when the card hits every enemy (or has no target).
    pub(crate) fn calculate_card_damage(&self, card: &mut Card, target: Option<usize>) -> Vec<i32> {
        card.block = self.card_block(card);
        if card.id == CardId::BodySlam {
            card.base_damage = self.player.block;
        }
        let base = self.card_effective_base(card);
        let mult = if card.id == CardId::HeavyBlade { card.magic } else { 1 };
        let t = DamageType::Normal;
        let vuln = self.monster_vuln();
        let pp = &self.player.powers;
        let stance = self.player.stance;
        if !card.data().multi_damage {
            if let Some(m) = target {
                let mon = &self.monsters[m];
                let mut tmp = base;
                for p in pp {
                    tmp = at_damage_give(p, tmp, t, mult, 0.75);
                }
                tmp = stance.at_damage_give(tmp, t);
                for p in &mon.powers {
                    tmp = at_damage_receive(p, tmp, t, vuln);
                }
                for p in pp {
                    tmp = at_damage_final_give(p, tmp, t);
                }
                for p in &mon.powers {
                    tmp = at_damage_final_receive(p, tmp, t);
                }
                card.damage = java_floor(tmp.max(0.0));
                return Vec::new();
            }
        }
        let mut out = Vec::with_capacity(self.order.len());
        for mon in self.order.iter().map(|&i| &self.monsters[i]) {
            let mut tmp = base;
            for p in pp {
                tmp = at_damage_give(p, tmp, t, mult, 0.75);
            }
            tmp = stance.at_damage_give(tmp, t);
            let alive = !mon.is_dying && !mon.is_escaping;
            if alive {
                for p in &mon.powers {
                    tmp = at_damage_receive(p, tmp, t, vuln);
                }
            }
            for p in pp {
                tmp = at_damage_final_give(p, tmp, t);
            }
            if alive {
                for p in &mon.powers {
                    tmp = at_damage_final_receive(p, tmp, t);
                }
            }
            out.push(java_floor(tmp.max(0.0)));
        }
        if let Some(&first) = out.first() {
            card.damage = first;
        }
        out
    }

    /// What a card in hand shows in the game (`applyPowers`, no target): (damage, block), with the
    /// player's powers (Strength, Weak, Dexterity, Frail...), Body Slam's block and the like.
    pub fn card_display(&self, card: &Card) -> (i32, i32) {
        let mut c = card.clone();
        self.apply_powers_view(&mut c);
        (c.damage, c.block)
    }

    /// The damage an attack would deal to monster `m` (`calculateCardDamage(m)`, so the monster's
    /// powers count too: Vulnerable, Intangible...), or 0 for a card that isn't an attack.
    pub fn card_damage_against(&self, card: &Card, m: usize) -> i32 {
        if card.card_type() != CardType::Attack {
            return 0;
        }
        let mut c = card.clone();
        let out = self.calculate_card_damage(&mut c, Some(m));
        if out.is_empty() {
            return c.damage;
        }
        self.order.iter().position(|&i| i == m).and_then(|k| out.get(k).copied()).unwrap_or(c.damage)
    }

    /// The energy an X-cost card in hand would spend (`EnergyPanel.totalCount`, 2 more with
    /// Chemical X, as its action adds), or 0 for a card that doesn't cost X.
    pub fn x_energy(&self, card: &Card) -> i32 {
        if card.cost != -1 {
            return 0;
        }
        self.energy + if self.has_relic(crate::relics::RelicId::ChemicalX) { 2 } else { 0 }
    }

    /// How one play of `card` from the hand would hit, counted as its `use` and actions count
    /// (`cards_use.rs`): how many times, and whether each hit lands on every enemy or a random
    /// one. `target` is the monster it is played at (Bane hits a poisoned one twice). No hits for
    /// a card that isn't an attack.
    pub fn card_hits(&self, card: &Card, target: Option<usize>) -> Hits {
        use CardId::*;
        if card.card_type() != CardType::Attack {
            return Hits::default();
        }
        let d = card.data();
        let alive = |m: usize| !self.monsters[m].is_dead_or_escaped();
        let count = match card.id {
            TwinStrike | DaggerSpray | GlassKnife | FlyingSleeves => 2,
            Eviscerate => 3,
            RiddleWithHoles => 5,
            Pummel | SwordBoomerang | RipAndTear | Ragnarok | Tantrum | Expunger => card.magic,
            Whirlwind | Skewer => self.x_energy(card),
            // The attacks played before it this turn (DamagePerAttackPlayedAction).
            Finisher => self.attacks_played_this_turn,
            Flechettes => self.hand.iter().filter(|c| c.card_type() == CardType::Skill).count() as i32,
            Barrage => self.filled_orb_count() as i32,
            BowlingBash => self.order.iter().filter(|&&m| alive(m)).count() as i32,
            // Every other card in hand, each exhausted.
            FiendFire => self.hand.iter().filter(|c| c.instance != card.instance).count() as i32,
            ThunderStrike => self.orbs_channeled_this_combat.iter().filter(|&&o| o == super::OrbId::Lightning).count() as i32,
            Bane => 1 + target.map_or(0, |t| (self.monsters[t].has_power(PowerId::Poison) && self.monsters[t].hp > 0) as i32),
            _ => 1,
        };
        Hits {
            count,
            all: d.multi_damage,
            // Sword Boomerang, Rip and Tear, Thunder Strike, Ragnarok.
            random: !d.multi_damage && d.target == crate::cards::CardTarget::AllEnemy,
        }
    }

    /// `AbstractCard.applyPowers` (no target): what the hand displays.
    pub(crate) fn apply_powers_view(&self, card: &mut Card) {
        card.block = self.card_block(card);
        let base = self.card_effective_base(card);
        let mult = if card.id == CardId::HeavyBlade { card.magic } else { 1 };
        let mut tmp = base;
        for p in &self.player.powers {
            tmp = at_damage_give(p, tmp, DamageType::Normal, mult, 0.75);
        }
        tmp = self.player.stance.at_damage_give(tmp, DamageType::Normal);
        for p in &self.player.powers {
            tmp = at_damage_final_give(p, tmp, DamageType::Normal);
        }
        card.damage = java_floor(tmp.max(0.0));
    }

    /// `AbstractMonster.applyBackAttack()`: the player is Surrounded and faces away from `m`.
    pub(crate) fn back_attack(&self, m: usize) -> bool {
        // Monsters sit at 0.75 * width + offset; the player at 0.25 * width.
        let left_of_player = self.monsters[m].draw_x < -960.0;
        self.player.has_power(PowerId::Surrounded) && if self.player_flipped { !left_of_player } else { left_of_player }
    }

    /// `DamageInfo.applyPowers` for a monster-owned info targeting the player.
    fn monster_damage_output(&self, m: usize, base: i32) -> i32 {
        self.monster_damage_calc(m, base, false)
    }

    /// The monster damage chain; `intent` applies Back Attack before the final hooks like
    /// `calculateDamage` does.
    fn monster_damage_calc(&self, m: usize, base: i32, intent: bool) -> i32 {
        let t = DamageType::Normal;
        let mon = &self.monsters[m];
        let mut tmp = base as f32;
        let vuln = self.player_vuln();
        let weak = if self.has_relic(crate::relics::RelicId::PaperCrane) { 0.6 } else { 0.75 };
        for p in &mon.powers {
            tmp = at_damage_give(p, tmp, t, 1, weak);
        }
        for p in &self.player.powers {
            tmp = at_damage_receive(p, tmp, t, vuln);
        }
        tmp = self.player.stance.at_damage_receive(tmp, t);
        if intent && self.back_attack(m) {
            tmp = (tmp * 1.5) as i32 as f32;
        }
        for p in &mon.powers {
            tmp = at_damage_final_give(p, tmp, t);
        }
        for p in &self.player.powers {
            tmp = at_damage_final_receive(p, tmp, t);
        }
        java_floor(tmp).max(0)
    }

    /// `AbstractMonster.calculateDamage` (intent display).
    pub(crate) fn monster_intent_damage(&self, m: usize, base: i32) -> i32 {
        self.monster_damage_calc(m, base, true)
    }

    /// `AbstractMonster.applyPowers`.
    pub(crate) fn monster_apply_powers(&mut self, m: usize) {
        let back = self.back_attack(m);
        if back && !self.monsters[m].has_power(PowerId::BackAttack) {
            self.top_power(Cr::Monster(m), None, Power::flag(PowerId::BackAttack));
        }
        for s in 0..self.monsters[m].damage.len() {
            let base = self.monsters[m].damage[s].base;
            let mut out = self.monster_damage_output(m, base);
            if back {
                out = (out as f32 * 1.5) as i32;
            }
            self.monsters[m].damage[s].output = out;
        }
        let mv = self.monsters[m].mv;
        if mv.base_damage > -1 {
            self.monsters[m].intent_dmg = self.monster_intent_damage(m, mv.base_damage);
        }
    }

    /// `AbstractDungeon.onModifyPower`: refreshes every monster's damage numbers.
    pub(crate) fn on_modify_power(&mut self) {
        for m in 0..self.monsters.len() {
            self.monster_apply_powers(m);
        }
    }

    /// `DamageInfo.createDamageMatrix(base, isPureDamage = true)`.
    pub(crate) fn pure_damage_matrix(&self, base: i32) -> Vec<i32> {
        vec![base; self.order.len()]
    }

    // ---- Taking damage ----

    /// `AbstractCreature.decrementBlock`.
    fn decrement_block(&mut self, c: Cr, info: &DamageInfo, mut dmg: i32) -> i32 {
        let block = self.block(c);
        if info.damage_type != DamageType::HpLoss && block > 0 {
            if dmg >= block {
                dmg = if dmg > block { dmg - block } else { 0 };
                self.set_block(c, 0);
                // brokeBlock()
                if let Cr::Monster(m) = c {
                    self.relics_on_block_broken(m);
                }
            } else {
                self.set_block(c, block - dmg);
                dmg = 0;
            }
        }
        dmg
    }

    pub(crate) fn set_block(&mut self, c: Cr, v: i32) {
        match c {
            Cr::Player => self.player.block = v,
            Cr::Monster(i) => self.monsters[i].block = v,
        }
    }

    /// `creature.damage(info)`.
    pub(crate) fn damage(&mut self, target: Cr, info: DamageInfo) {
        match target {
            Cr::Player => self.damage_player(info),
            Cr::Monster(m) => self.damage_monster(m, info),
        }
    }

    /// `AbstractPlayer.damage`.
    fn damage_player(&mut self, info: DamageInfo) {
        let mut dmg = info.output.max(0);
        if dmg > 1 && self.player.has_power(PowerId::IntangiblePlayer) {
            dmg = 1;
        }
        dmg = self.decrement_block(Cr::Player, &info, dmg);
        if info.owner == Some(Cr::Player) {
            dmg = self.relics_on_attack_to_change_damage(&info, dmg);
        }
        dmg = self.on_attacked_to_change_damage(Cr::Player, dmg);
        if info.owner.is_some() {
            dmg = self.on_attacked(Cr::Player, &info, dmg);
            dmg = self.relics_on_attacked(&info, dmg);
        }
        dmg = self.relics_on_lose_hp_last(dmg);
        self.last_damage_taken = dmg.min(self.player.hp);
        if dmg > 0 {
            self.was_hp_lost(Cr::Player, &info, dmg);
            self.relics_was_hp_lost(dmg);
            // The attacker's onInflictDamage: Painful Stabs.
            if let Some(Cr::Monster(m)) = info.owner {
                if info.damage_type != DamageType::Thorns && self.monsters[m].has_power(PowerId::PainfulStabs) {
                    let c = self.new_card(CardId::Wound);
                    self.push_bot(Action::MakeTempCardInDiscard { card: c, amount: 1 });
                }
            }
            self.player.hp -= dmg;
            // updateCardsOnDamage: tookDamage.
            for c in self.hand.iter_mut().chain(self.discard_pile.iter_mut()).chain(self.draw_pile.iter_mut()) {
                match c.id {
                    CardId::BloodForBlood => c.update_cost(-1),
                    CardId::MasterfulStab => c.update_cost(1),
                    _ => {}
                }
            }
            self.player.damaged_this_combat += 1;
            if self.player.hp < 0 {
                self.player.hp = 0;
            }
            if self.player.hp as f32 <= self.player.max_hp as f32 / 2.0 && !self.player.is_bloodied {
                self.player.is_bloodied = true;
                self.relics_on_bloodied();
            }
            if self.player.hp < 1 {
                if !self.has_relic(crate::relics::RelicId::MarkOfTheBloom) && (self.fairy_revive() || self.relics_lizard_tail()) {
                    return;
                }
                self.player.is_dead = true;
                self.player.hp = 0;
                self.player.block = 0;
            }
        }
    }

    /// `AbstractMonster.damage` plus the subclass overrides.
    fn damage_monster(&mut self, m: usize, mut info: DamageInfo) {
        // Nemesis.damage: Intangible caps every hit at 1.
        if info.output > 0 && self.monsters[m].has_power(PowerId::Intangible) {
            info.output = 1;
        }
        let prev_hp = self.monsters[m].hp;
        self.damage_monster_base(m, &info);
        self.after_monster_damaged(m, prev_hp, &info);
    }

    fn damage_monster_base(&mut self, m: usize, info: &DamageInfo) {
        if self.monsters[m].is_dying || self.monsters[m].is_escaping {
            return;
        }
        let mut dmg = info.output.max(0);
        dmg = self.decrement_block(Cr::Monster(m), info, dmg);
        if info.owner == Some(Cr::Player) {
            dmg = self.relics_on_attack_to_change_damage(info, dmg);
        }
        dmg = self.on_attacked_to_change_damage(Cr::Monster(m), dmg);
        self.was_hp_lost(Cr::Monster(m), info, dmg);
        if info.owner == Some(Cr::Player) {
            self.player_on_attack(info, dmg, Cr::Monster(m));
        }
        dmg = self.on_attacked(Cr::Monster(m), info, dmg);
        self.monsters[m].last_damage_taken = dmg.min(self.monsters[m].hp);
        if dmg > 0 {
            let mon = &mut self.monsters[m];
            mon.hp = (mon.hp - dmg).max(0);
        }
        if self.monsters[m].hp <= 0 {
            // die(): the subclass override around super.die().
            self.city_before_die(m);
            if !self.beyond_die_blocked(m) {
                self.monster_die(m, true);
                self.city_after_die(m);
                self.beyond_after_die(m);
                self.ending_after_die(m);
            }
            if self.monsters_basically_dead() {
                self.clean_card_queue();
            }
            self.monsters[m].block = 0;
        }
    }

    /// `AbstractMonster.die(triggerRelics)`.
    pub(crate) fn monster_die(&mut self, m: usize, trigger_relics: bool) {
        if self.monsters[m].is_dying {
            return;
        }
        self.monsters[m].is_dying = true;
        if self.monsters[m].hp <= 0 && trigger_relics {
            self.on_death(m);
        }
        if trigger_relics {
            self.relics_on_monster_death(m);
        }
        if self.monsters_basically_dead() {
            self.limbo.clear();
        }
        if self.monsters[m].hp < 0 {
            self.monsters[m].hp = 0;
        }
        self.after_monster_die(m);
    }

    /// `GameActionManager.cleanCardQueue`.
    fn clean_card_queue(&mut self) {
        let hand: Vec<u32> = self.hand.iter().map(|c| c.instance).collect();
        self.card_queue.retain(|q| match q.card.instance() {
            Some(i) => !hand.contains(&i),
            None => true,
        });
    }

    /// `AbstractCreature.addBlock`.
    pub(crate) fn add_block(&mut self, c: Cr, amount: i32) {
        let mut tmp = amount as f32;
        if c == Cr::Player {
            tmp = self.relics_on_player_gained_block(tmp);
            if tmp > 0.0 {
                self.on_gained_block(tmp);
            }
        }
        let v = (self.block(c) + java_floor(tmp)).min(999);
        self.set_block(c, v);
    }

    /// `AbstractCreature.heal`.
    pub(crate) fn heal(&mut self, c: Cr, amount: i32) {
        match c {
            Cr::Player => {
                if self.player.is_dead {
                    return;
                }
                let amount = self.relics_on_player_heal(amount);
                self.player.hp = (self.player.hp + amount).min(self.player.max_hp);
                if self.player.hp as f32 > self.player.max_hp as f32 / 2.0 && self.player.is_bloodied {
                    self.player.is_bloodied = false;
                    self.relics_on_not_bloodied();
                }
            }
            Cr::Monster(m) => {
                let mon = &mut self.monsters[m];
                if mon.is_dying {
                    return;
                }
                mon.hp = (mon.hp + amount).min(mon.max_hp);
            }
        }
    }

    /// `AbstractCreature.increaseMaxHp`.
    pub(crate) fn increase_max_hp(&mut self, amount: i32) {
        self.player.max_hp += amount;
        self.heal(Cr::Player, amount);
    }

    // ---- Powers ----

    /// Builds an `ApplyPowerAction`, running its constructor side effects.
    pub(crate) fn apply_power_action(&mut self, target: Cr, source: Option<Cr>, mut power: Power, mut amount: i32) -> Action {
        if power.id == PowerId::Poison && source == Some(Cr::Player) && target != Cr::Player && self.has_relic(crate::relics::RelicId::SneckoSkull) {
            power.amount += 1;
            amount += 1;
        }
        if power.id == PowerId::Corruption {
            for c in self
                .hand
                .iter_mut()
                .chain(self.draw_pile.iter_mut())
                .chain(self.discard_pile.iter_mut())
                .chain(self.exhaust_pile.iter_mut())
            {
                if c.card_type() == CardType::Skill {
                    c.modify_cost_for_combat(-9);
                }
            }
        }
        Action::ApplyPower { target, source, power, amount }
    }

    /// `addToBot(new ApplyPowerAction(target, source, power, power.amount))`.
    pub(crate) fn bot_power(&mut self, target: Cr, source: Option<Cr>, power: Power) {
        let amount = power.amount;
        let a = self.apply_power_action(target, source, power, amount);
        self.push_bot(a);
    }

    pub(crate) fn top_power(&mut self, target: Cr, source: Option<Cr>, power: Power) {
        let amount = power.amount;
        let a = self.apply_power_action(target, source, power, amount);
        self.push_top(a);
    }

    /// `ApplyPowerAction.update`.
    pub(crate) fn exec_apply_power(&mut self, target: Cr, _source: Option<Cr>, power: Power, amount: i32) {
        if self.is_dead_or_escaped(target) {
            return;
        }
        if power.id == PowerId::NoDraw && self.has_power(target, PowerId::NoDraw) {
            return;
        }
        // The source's onApplyPower: Sadistic Nature.
        if _source == Some(Cr::Player) && target != Cr::Player && power.power_type() == PowerType::Debuff && power.id != PowerId::Shackled && !self.has_power(target, PowerId::Artifact) {
            if let Some(n) = self.player.powers.iter().find(|p| p.id == PowerId::Sadistic).map(|p| p.amount) {
                let info = DamageInfo::new(Some(Cr::Player), n, DamageType::Thorns);
                self.push_bot(Action::Damage { target, source: crate::action::DamageSource::Fixed(info), steal_gold: 0 });
            }
        }
        if self.has_relic(crate::relics::RelicId::ChampionsBelt)
            && _source == Some(Cr::Player)
            && target != Cr::Player
            && power.id == PowerId::Vulnerable
            && !self.has_power(target, PowerId::Artifact)
        {
            self.bot_power(target, Some(Cr::Player), Power::weak(1, false));
        }
        if target == Cr::Player
            && ((power.id == PowerId::Weakened && self.has_relic(crate::relics::RelicId::Ginger))
                || (power.id == PowerId::Frail && self.has_relic(crate::relics::RelicId::Turnip)))
        {
            return;
        }
        if power.power_type() == PowerType::Debuff && self.has_power(target, PowerId::Artifact) {
            // ArtifactPower.onSpecificTrigger
            if self.power_amount(target, PowerId::Artifact) <= 0 {
                self.push_top(Action::RemovePower { target, id: PowerId::Artifact });
            } else {
                self.push_top(Action::ReducePower { target, id: PowerId::Artifact, amount: 1 });
            }
            return;
        }
        if power.id == PowerId::TheBomb {
            // Every bomb is its own power.
            let mut b = power;
            self.bomb_counter += 1;
            b.uid = self.bomb_counter;
            self.powers_mut(target).push(b);
            self.on_modify_power();
            return;
        }
        if power.id == PowerId::NightTerror {
            // ApplyPowerAction never stacks Night Terror: each Nightmare is its own power.
            let powers = self.powers_mut(target);
            powers.push(power);
            powers.sort_by_key(|p| p.priority());
            self.on_modify_power();
            return;
        }
        let accuracy = power.id == PowerId::Accuracy;
        // DrawPower's constructor raises the hand size, stacked or not (Machine Learning).
        if power.id == PowerId::Draw && target == Cr::Player {
            self.player.game_hand_size += power.amount;
        }
        let powers = self.powers_mut(target);
        if let Some(p) = powers.iter_mut().find(|p| p.id == power.id) {
            let remove = p.stack(amount);
            // MantraPower.stackPower: at 10, Divinity, and the rest stays (gone at 0).
            let mut divinity = false;
            let mut mantra_spent = false;
            if p.id == PowerId::Mantra && p.amount >= 10 {
                p.amount -= 10;
                divinity = true;
                mantra_spent = p.amount <= 0;
            }
            if remove {
                self.push_top(Action::RemovePower { target, id: power.id });
            }
            if divinity {
                self.push_top(Action::ChangeStance { stance: super::Stance::Divinity });
                if mantra_spent {
                    self.push_top(Action::RemovePower { target, id: PowerId::Mantra });
                }
            }
            self.on_modify_power();
        } else {
            let id = power.id;
            powers.push(power);
            // Collections.sort is stable.
            powers.sort_by_key(|p| p.priority());
            // onInitialApplication
            if id == PowerId::DrawReduction && target == Cr::Player {
                self.player.game_hand_size -= 1;
            }
            self.on_modify_power();
        }
        if accuracy {
            self.update_existing_shivs();
        }
    }

    /// `AccuracyPower.updateExistingShivs`: every Shiv in the hand and the piles.
    fn update_existing_shivs(&mut self) {
        let n = self.power_amount(Cr::Player, PowerId::Accuracy);
        for c in self
            .hand
            .iter_mut()
            .chain(self.draw_pile.iter_mut())
            .chain(self.discard_pile.iter_mut())
            .chain(self.exhaust_pile.iter_mut())
            .filter(|c| c.id == CardId::Shiv)
        {
            c.base_damage = if c.upgraded { 6 } else { 4 } + n;
        }
    }

    /// `ReducePowerAction.update`.
    pub(crate) fn exec_reduce_power(&mut self, target: Cr, id: PowerId, amount: i32) {
        let powers = self.powers_mut(target);
        if let Some(p) = powers.iter_mut().find(|p| p.id == id) {
            if amount < p.amount {
                let remove = p.reduce(amount);
                if remove {
                    self.push_top(Action::RemovePower { target, id });
                }
                self.on_modify_power();
            } else {
                self.push_top(Action::RemovePower { target, id });
            }
        }
    }

    /// `AbstractCreature.addPower`: stack or append, without sorting.
    pub(crate) fn add_power_direct(&mut self, target: Cr, power: Power) {
        let powers = self.powers_mut(target);
        if let Some(p) = powers.iter_mut().find(|p| p.id == power.id) {
            p.stack(power.amount);
        } else {
            powers.push(power);
        }
    }

    /// `RemoveSpecificPowerAction.update`.
    pub(crate) fn exec_remove_power(&mut self, target: Cr, id: PowerId) {
        if self.is_dead_or_escaped(target) {
            return;
        }
        let powers = self.powers_mut(target);
        if let Some(pos) = powers.iter().position(|p| p.id == id) {
            // onRemove
            if let Cr::Monster(m) = target {
                match id {
                    PowerId::PlatedArmor => self.push_bot(Action::ChangeState { monster: m, state: crate::action::MonsterState::ArmorBreak }),
                    PowerId::Flight => self.push_bot(Action::ChangeState { monster: m, state: crate::action::MonsterState::Grounded }),
                    _ => {}
                }
            }
            if id == PowerId::DrawReduction && target == Cr::Player {
                self.player.game_hand_size += 1;
            }
            let removed = self.powers_mut(target).remove(pos);
            if id == PowerId::NightTerror {
                self.night_terror_cards.retain(|(uid, _)| *uid != removed.uid);
            }
            self.on_modify_power();
        }
    }
}
