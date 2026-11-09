//! Power hook dispatch. Each function mirrors one `AbstractPower` callback, iterating the
//! owner's powers in list order like the Java loops do.

use super::Combat;
use crate::action::{Action, Cr, DamageInfo, DamageSource, DamageType, MonsterState};
use crate::cards::{Card, CardType};
use crate::powers::{Power, PowerId};

impl Combat {
    fn power_ids(&self, c: Cr) -> Vec<(PowerId, i32)> {
        self.powers(c).iter().map(|p| (p.id, p.amount)).collect()
    }

    /// `atStartOfTurn` for every power of `c`.
    pub(crate) fn apply_start_of_turn_powers(&mut self, c: Cr) {
        for p in self.powers(c).clone() {
            let (id, amount) = (p.id, p.amount);
            match id {
                PowerId::NightTerror => {
                    if let Some(card) = self.night_terror_cards.iter().find(|(uid, _)| *uid == p.uid).map(|(_, card)| card.clone()) {
                        self.push_bot(Action::MakeTempCardInHand { card, amount });
                    }
                    self.push_bot(Action::RemovePower { target: c, id });
                }
                PowerId::Berserk => self.push_bot(Action::GainEnergy { amount }),
                PowerId::FlameBarrier => self.push_bot(Action::RemovePower { target: Cr::Player, id: PowerId::FlameBarrier }),
                PowerId::NextTurnBlock => {
                    self.push_bot(Action::GainBlock { target: c, amount });
                    self.push_bot(Action::RemovePower { target: c, id: PowerId::NextTurnBlock });
                }
                PowerId::Panache => {
                    if let Some(p) = self.powers_mut(c).iter_mut().find(|p| p.id == PowerId::Panache) {
                        p.amount = 5;
                    }
                }
                PowerId::Mayhem => {
                    for _ in 0..amount {
                        self.push_bot(Action::MayhemPlay);
                    }
                }
                PowerId::Magnetism => {
                    if !self.monsters_basically_dead() {
                        for _ in 0..amount {
                            let id = self.truly_random_colorless_in_combat();
                            let card = self.new_card(id);
                            self.push_bot(Action::MakeTempCardInHand { card, amount: 1 });
                        }
                    }
                }
                PowerId::Poison => {
                    if !self.monsters_basically_dead() {
                        self.push_bot(Action::PoisonLoseHp { target: c, amount });
                    }
                }
                PowerId::Choked => self.push_bot(Action::RemovePower { target: c, id }),
                PowerId::InfiniteBlades => {
                    if !self.monsters_basically_dead() {
                        let card = self.new_card(crate::cards::CardId::Shiv);
                        self.push_bot(Action::MakeTempCardInHand { card, amount });
                    }
                }
                PowerId::Loop => self.loop_front_orb(amount),
                PowerId::Hello => {
                    if !self.monsters_basically_dead() {
                        for _ in 0..amount {
                            // AbstractDungeon.getCard(COMMON, cardRandomRng): the run's common pool.
                            let pool: Vec<crate::cards::CardId> = self
                                .character
                                .library_order()
                                .iter()
                                .copied()
                                .filter(|c| c.data().rarity == crate::cards::CardRarity::Common)
                                .collect();
                            let id = pool[self.rng.card_random.random(pool.len() as i32 - 1) as usize];
                            let card = self.new_card(id);
                            self.push_bot(Action::MakeTempCardInHand { card, amount: 1 });
                        }
                    }
                }
                PowerId::CreativeAI => {
                    for _ in 0..amount {
                        let id = self.truly_random_card_in_combat(CardType::Power);
                        let card = self.new_card(id);
                        self.push_bot(Action::MakeTempCardInHand { card, amount: 1 });
                    }
                }
                PowerId::Bias => {
                    let a = self.apply_power_action(c, Some(c), Power::new(PowerId::Focus, -amount), -amount);
                    self.push_bot(a);
                }
                PowerId::EchoForm => {
                    if let Some(p) = self.powers_mut(c).iter_mut().find(|p| p.id == PowerId::EchoForm) {
                        p.base = 0;
                    }
                }
                PowerId::Phantasmal => {
                    let a = self.apply_power_action(c, Some(c), Power::new(PowerId::DoubleDamage, 1), amount);
                    self.push_bot(a);
                    self.push_bot(Action::ReducePower { target: c, id, amount: 1 });
                }
                PowerId::Flight | PowerId::Invincible => {
                    if let Some(p) = self.powers_mut(c).iter_mut().find(|p| p.id == id) {
                        p.amount = p.base;
                    }
                }
                PowerId::BattleHymn => {
                    if !self.monsters_basically_dead() {
                        let card = self.new_card(crate::cards::CardId::Smite);
                        self.push_bot(Action::MakeTempCardInHand { card, amount });
                    }
                }
                PowerId::EndTurnDeath => {
                    self.push_bot(Action::LoseHp { target: c, source: Some(c), amount: 99999 });
                    self.push_bot(Action::RemovePower { target: c, id });
                }
                PowerId::EnergyDown => self.push_bot(Action::LoseEnergy { amount }),
                PowerId::Foresight => {
                    if self.draw_pile.is_empty() {
                        // EmptyDeckShuffleAction's constructor runs the relics' onShuffle.
                        self.relics_on_shuffle();
                        self.push_top(Action::EmptyDeckShuffle);
                    }
                    let a = self.scry(amount);
                    self.push_bot(a);
                }
                PowerId::WrathNextTurn => {
                    self.push_bot(Action::ChangeStance { stance: super::Stance::Wrath });
                    self.push_bot(Action::RemovePower { target: c, id });
                }
                _ => {}
            }
        }
    }

    /// `atStartOfTurnPostDraw` for the player's powers.
    pub(crate) fn apply_start_of_turn_post_draw_powers(&mut self) {
        for (id, amount) in self.power_ids(Cr::Player) {
            match id {
                PowerId::Brutality => {
                    self.push_bot(Action::DrawCard { amount, follow_up: None, shuffle_check: false });
                    self.push_bot(Action::LoseHp { target: Cr::Player, source: Some(Cr::Player), amount });
                }
                PowerId::DemonForm => self.bot_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::Strength, amount)),
                PowerId::NoxiousFumes => {
                    if !self.monsters_basically_dead() {
                        for m in self.order.clone() {
                            // !isDead && !isDying (a dead monster is a dying one here).
                            if !self.monsters[m].is_dying {
                                self.bot_power(Cr::Monster(m), Some(Cr::Player), Power::new(PowerId::Poison, amount));
                            }
                        }
                    }
                }
                PowerId::DrawCardNextTurn => {
                    self.push_bot(Action::DrawCard { amount, follow_up: None, shuffle_check: false });
                    self.push_bot(Action::RemovePower { target: Cr::Player, id });
                }
                PowerId::ToolsOfTheTrade => {
                    self.push_bot(Action::DrawCard { amount, follow_up: None, shuffle_check: false });
                    self.push_bot(Action::Discard { amount, random: false, end_turn: false });
                }
                PowerId::Devotion => {
                    if !self.player.has_power(PowerId::Mantra) && amount >= 10 {
                        self.push_bot(Action::ChangeStance { stance: super::Stance::Divinity });
                    } else {
                        let m = self.mantra(amount);
                        self.bot_power(Cr::Player, Some(Cr::Player), m);
                    }
                }
                _ => {}
            }
        }
    }

    /// `atEndOfTurnPreEndTurnCards(isPlayer)`.
    pub(crate) fn at_end_of_turn_pre_end_turn_cards(&mut self, c: Cr) {
        for (id, amount) in self.power_ids(c) {
            if id == PowerId::Metallicize || id == PowerId::PlatedArmor {
                self.push_bot(Action::GainBlock { target: c, amount });
            }
            if id == PowerId::LikeWater && c == Cr::Player && self.player.stance == super::Stance::Calm {
                self.push_bot(Action::GainBlock { target: c, amount });
            }
        }
    }

    /// `AbstractCreature.applyEndOfTurnTriggers`: monsters also run the pre-end-turn hook.
    pub(crate) fn apply_end_of_turn_triggers(&mut self, c: Cr) {
        let is_player = c == Cr::Player;
        let snapshot: Vec<Power> = self.powers(c).clone();
        for p in snapshot {
            if !is_player && (p.id == PowerId::Metallicize || p.id == PowerId::PlatedArmor) {
                self.push_bot(Action::GainBlock { target: c, amount: p.amount });
            }
            self.at_end_of_turn(c, &p, is_player);
        }
    }

    /// `atEndOfTurn(isPlayer)`.
    fn at_end_of_turn(&mut self, owner: Cr, p: &Power, is_player: bool) {
        match p.id {
            PowerId::Combust => {
                if !self.monsters_basically_dead() {
                    self.push_bot(Action::LoseHp { target: owner, source: Some(owner), amount: p.hp_loss });
                    let damage = self.pure_damage_matrix(p.amount);
                    self.push_bot(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Thorns });
                }
            }
            PowerId::DoubleTap | PowerId::Entangled | PowerId::NoDraw => {
                if is_player {
                    self.push_bot(Action::RemovePower { target: owner, id: p.id });
                }
            }
            PowerId::Flex => {
                self.bot_power(owner, Some(owner), Power::new(PowerId::Strength, -p.amount));
                self.push_bot(Action::RemovePower { target: owner, id: PowerId::Flex });
            }
            PowerId::Shackled => {
                self.bot_power(owner, Some(owner), Power::new(PowerId::Strength, p.amount));
                self.push_bot(Action::RemovePower { target: owner, id: PowerId::Shackled });
            }
            PowerId::Rage => self.push_bot(Action::RemovePower { target: owner, id: PowerId::Rage }),
            PowerId::Burst | PowerId::Amplify | PowerId::Rebound => {
                if is_player {
                    self.push_bot(Action::RemovePower { target: owner, id: p.id });
                }
            }
            PowerId::Equilibrium => {
                if is_player {
                    for c in self.hand.iter_mut() {
                        if !c.ethereal {
                            c.retain = true;
                        }
                    }
                }
            }
            PowerId::WraithForm => self.bot_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::Dexterity, p.amount)),
            PowerId::RetainCards => {
                if is_player && !self.hand.is_empty() && !self.has_relic(crate::relics::RelicId::RunicPyramid) && !self.player.has_power(PowerId::Equilibrium) {
                    self.push_bot(Action::RetainCards { amount: p.amount });
                }
            }
            PowerId::TheBomb => {
                if !self.monsters_basically_dead() {
                    self.push_bot(Action::ReduceBomb { uid: p.uid });
                    if p.amount == 1 {
                        let damage = self.pure_damage_matrix(p.hp_loss);
                        self.push_bot(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Thorns });
                    }
                }
            }
            PowerId::Regen => self.push_top(Action::Regen { target: owner }),
            PowerId::LoseDexterity => {
                self.bot_power(owner, Some(owner), Power::new(PowerId::Dexterity, -p.amount));
                self.push_bot(Action::RemovePower { target: owner, id: PowerId::LoseDexterity });
            }
            PowerId::Ritual => {
                if is_player {
                    self.bot_power(owner, Some(owner), Power::new(PowerId::Strength, p.amount));
                }
            }
            PowerId::Intangible => {
                if let Some(q) = self.powers_mut(owner).iter_mut().find(|q| q.id == PowerId::Intangible) {
                    if q.just_applied {
                        q.just_applied = false;
                        return;
                    }
                }
                if p.amount == 0 {
                    self.push_bot(Action::RemovePower { target: owner, id: p.id });
                } else {
                    self.push_bot(Action::ReducePower { target: owner, id: p.id, amount: 1 });
                }
            }
            PowerId::Regenerate => {
                if let Cr::Monster(m) = owner {
                    let mon = &self.monsters[m];
                    if !mon.half_dead && !mon.is_dying {
                        self.push_bot(Action::Heal { target: owner, amount: p.amount });
                    }
                }
            }
            PowerId::Constricted => {
                let source = Some(Cr::Monster(p.base.max(0) as usize));
                let info = DamageInfo::new(source, p.amount, DamageType::Thorns);
                self.push_bot(Action::Damage { target: owner, source: DamageSource::Fixed(info), steal_gold: 0 });
            }
            PowerId::Malleable => {
                if !is_player {
                    if let Some(m) = self.powers_mut(owner).iter_mut().find(|q| q.id == PowerId::Malleable) {
                        m.amount = m.base;
                    }
                }
            }
            PowerId::Establishment => self.push_bot(Action::Establishment { amount: p.amount }),
            PowerId::Omega => {
                if is_player {
                    let damage = self.pure_damage_matrix(p.amount);
                    self.push_bot(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Thorns });
                }
            }
            PowerId::Study => {
                let card = self.new_card(crate::cards::CardId::Insight);
                self.push_bot(Action::MakeTempCardInDrawPile { card, amount: p.amount, random_spot: true, to_bottom: false });
            }
            _ => {}
        }
    }

    /// `atEndOfRound` for every power of `c`.
    pub(crate) fn at_end_of_round(&mut self, c: Cr) {
        let n = self.powers(c).len();
        for i in 0..n {
            let Some(p) = self.powers(c).get(i).cloned() else { break };
            match p.id {
                PowerId::Vulnerable | PowerId::Weakened | PowerId::Frail | PowerId::NoBlock => {
                    if p.just_applied {
                        self.powers_mut(c)[i].just_applied = false;
                    } else if p.amount == 0 {
                        self.push_bot(Action::RemovePower { target: c, id: p.id });
                    } else {
                        self.push_bot(Action::ReducePower { target: c, id: p.id, amount: 1 });
                    }
                }
                PowerId::Slow => self.powers_mut(c)[i].amount = 0,
                PowerId::LockOn | PowerId::Equilibrium => {
                    if p.amount == 0 {
                        self.push_bot(Action::RemovePower { target: c, id: p.id });
                    } else {
                        self.push_bot(Action::ReducePower { target: c, id: p.id, amount: 1 });
                    }
                }
                PowerId::Blur => {
                    if p.amount == 0 {
                        self.push_bot(Action::RemovePower { target: c, id: p.id });
                    } else {
                        self.push_bot(Action::ReducePower { target: c, id: p.id, amount: 1 });
                    }
                }
                PowerId::DoubleDamage => {
                    if p.just_applied {
                        self.powers_mut(c)[i].just_applied = false;
                    } else if p.amount == 0 {
                        self.push_bot(Action::RemovePower { target: c, id: p.id });
                    } else {
                        self.push_bot(Action::ReducePower { target: c, id: p.id, amount: 1 });
                    }
                }
                PowerId::GenericStrengthUp => self.bot_power(c, Some(c), Power::new(PowerId::Strength, p.amount)),
                PowerId::WaveOfTheHand => self.push_bot(Action::RemovePower { target: c, id: p.id }),
                PowerId::DrawReduction => {
                    if p.just_applied {
                        self.powers_mut(c)[i].just_applied = false;
                    } else {
                        self.push_bot(Action::ReducePower { target: c, id: p.id, amount: 1 });
                    }
                }
                PowerId::Duplication => {
                    if p.amount == 0 {
                        self.push_bot(Action::RemovePower { target: c, id: p.id });
                    } else {
                        self.push_bot(Action::ReducePower { target: c, id: p.id, amount: 1 });
                    }
                }
                PowerId::IntangiblePlayer => {
                    if p.amount == 0 {
                        self.push_bot(Action::RemovePower { target: c, id: p.id });
                    } else {
                        self.push_bot(Action::ReducePower { target: c, id: p.id, amount: 1 });
                    }
                }
                PowerId::Ritual => {
                    if !p.on_player {
                        if !p.skip_first {
                            self.bot_power(c, Some(c), Power::new(PowerId::Strength, p.amount));
                        } else {
                            self.powers_mut(c)[i].skip_first = false;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// `onUseCard` hooks run by the `UseCardAction` constructor (player powers, then monsters').
    pub(crate) fn on_use_card(&mut self, card: &Card, target: Option<usize>, exhaust_card: &mut bool) {
        let t = card.card_type();
        let n = self.player.powers.len();
        for i in 0..n {
            let Some(p) = self.player.powers.get(i).cloned() else { break };
            match p.id {
                PowerId::Corruption => {
                    if t == CardType::Skill {
                        *exhaust_card = true;
                    }
                }
                PowerId::DoubleTap | PowerId::Duplication => {
                    if !card.purge_on_use && (t == CardType::Attack || p.id == PowerId::Duplication) && p.amount > 0 {
                        let mut tmp = self.play_copy(card);
                        if let Some(m) = target {
                            self.calculate_card_damage(&mut tmp, Some(m));
                        }
                        tmp.purge_on_use = true;
                        let inst = tmp.instance;
                        self.limbo.insert(0, tmp);
                        let item = super::QueueItem {
                            card: super::QueuedCard::InLimbo(inst),
                            target,
                            energy_on_use: card.energy_on_use,
                            ignore_energy_total: true,
                            autoplay: true,
                            random_target: false,
                            end_turn_autoplay: false,
                        };
                        // addCardQueueItem(item, inFrontOfQueue = true)
                        if self.card_queue.is_empty() {
                            self.card_queue.push_back(item);
                        } else {
                            self.card_queue.insert(1, item);
                        }
                        self.player.powers[i].amount -= 1;
                        if self.player.powers[i].amount == 0 {
                            self.push_bot(Action::RemovePower { target: Cr::Player, id: p.id });
                        }
                    }
                }
                PowerId::Burst => {
                    if !card.purge_on_use && t == CardType::Skill && p.amount > 0 {
                        let mut tmp = self.play_copy(card);
                        if let Some(m) = target {
                            self.calculate_card_damage(&mut tmp, Some(m));
                        }
                        tmp.purge_on_use = true;
                        let inst = tmp.instance;
                        self.limbo.insert(0, tmp);
                        let item = super::QueueItem {
                            card: super::QueuedCard::InLimbo(inst),
                            target,
                            energy_on_use: card.energy_on_use,
                            ignore_energy_total: true,
                            autoplay: true,
                            random_target: false,
                            end_turn_autoplay: false,
                        };
                        if self.card_queue.is_empty() {
                            self.card_queue.push_back(item);
                        } else {
                            self.card_queue.insert(1, item);
                        }
                        self.player.powers[i].amount -= 1;
                        if self.player.powers[i].amount == 0 {
                            self.push_top(Action::RemovePower { target: Cr::Player, id: p.id });
                        }
                    }
                }
                PowerId::AfterImage => self.push_bot(Action::GainBlock { target: Cr::Player, amount: p.amount }),
                PowerId::Storm => {
                    if t == CardType::Power && p.amount > 0 {
                        for _ in 0..p.amount {
                            self.push_bot(Action::Channel { orb: super::Orb::new(super::OrbId::Lightning), auto_evoke: true });
                        }
                    }
                }
                PowerId::Heatsink => {
                    if t == CardType::Power {
                        self.push_top(Action::DrawCard { amount: p.amount, follow_up: None, shuffle_check: false });
                    }
                }
                PowerId::EchoForm => {
                    if !card.purge_on_use && p.amount > 0 && self.player.cards_played_this_turn - p.base <= p.amount {
                        self.player.powers[i].base += 1;
                        self.queue_copy(card, target);
                    }
                }
                PowerId::Amplify => {
                    if !card.purge_on_use && t == CardType::Power && p.amount > 0 {
                        self.queue_copy(card, target);
                        self.player.powers[i].amount -= 1;
                        if self.player.powers[i].amount == 0 {
                            self.push_bot(Action::RemovePower { target: Cr::Player, id: p.id });
                        }
                    }
                }
                PowerId::Rage => {
                    if t == CardType::Attack {
                        self.push_bot(Action::GainBlock { target: Cr::Player, amount: p.amount });
                    }
                }
                PowerId::Vigor | PowerId::PenNib => {
                    if t == CardType::Attack {
                        self.push_bot(Action::RemovePower { target: Cr::Player, id: p.id });
                    }
                }
                PowerId::FreeAttack => {
                    if t == CardType::Attack && !card.purge_on_use && p.amount > 0 {
                        self.player.powers[i].amount -= 1;
                        if self.player.powers[i].amount == 0 {
                            self.push_top(Action::RemovePower { target: Cr::Player, id: p.id });
                        }
                    }
                }
                PowerId::Hex => {
                    if t != CardType::Attack {
                        let card = self.new_card(crate::cards::CardId::Dazed);
                        self.push_bot(Action::MakeTempCardInDrawPile { card, amount: p.amount, random_spot: true, to_bottom: false });
                    }
                }
                PowerId::Panache => {
                    let pw = &mut self.player.powers[i];
                    pw.amount -= 1;
                    if pw.amount == 0 {
                        pw.amount = 5;
                        let damage = self.pure_damage_matrix(p.hp_loss);
                        self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage, damage_type: DamageType::Thorns });
                    }
                }
                _ => {}
            }
        }
        self.relics_on_use_card(card, target, exhaust_card);
        // The hand's, discard pile's and draw pile's triggerOnCardPlayed: Force Field.
        if t == CardType::Power {
            for c in self.hand.iter_mut().chain(self.discard_pile.iter_mut()).chain(self.draw_pile.iter_mut()) {
                if c.id == crate::cards::CardId::ForceField {
                    c.update_cost(-1);
                }
            }
        }
        for m in self.order.clone() {
            for (id, amount) in self.power_ids(Cr::Monster(m)) {
                match id {
                    PowerId::Anger => {
                        if t == CardType::Skill {
                            self.top_power(Cr::Monster(m), Some(Cr::Monster(m)), Power::new(PowerId::Strength, amount));
                        }
                    }
                    PowerId::Curiosity => {
                        if t == CardType::Power {
                            self.bot_power(Cr::Monster(m), Some(Cr::Monster(m)), Power::new(PowerId::Strength, amount));
                        }
                    }
                    PowerId::Choked => self.push_bot(Action::LoseHp { target: Cr::Monster(m), source: None, amount }),
                    PowerId::SharpHide => {
                        if t == CardType::Attack {
                            let info = DamageInfo::new(Some(Cr::Monster(m)), amount, DamageType::Thorns);
                            self.push_bot(Action::Damage { target: Cr::Player, source: DamageSource::Fixed(info), steal_gold: 0 });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// `onCardDraw` for the player's powers; `idx` is the drawn card's hand index.
    pub(crate) fn on_card_draw(&mut self, idx: usize) {
        let t = self.hand[idx].card_type();
        for (id, amount) in self.power_ids(Cr::Player) {
            match id {
                PowerId::Corruption => {
                    if t == CardType::Skill {
                        self.hand[idx].set_cost_for_turn(-9);
                    }
                }
                PowerId::Confusion => {
                    if self.hand[idx].cost >= 0 {
                        let new_cost = self.rng.card_random.random(3);
                        let c = &mut self.hand[idx];
                        if c.cost != new_cost {
                            c.cost = new_cost;
                            c.cost_for_turn = new_cost;
                        }
                        c.free_to_play_once = false;
                    }
                }
                PowerId::Evolve => {
                    if t == CardType::Status && !self.player.has_power(PowerId::NoDraw) {
                        self.push_bot(Action::DrawCard { amount, follow_up: None, shuffle_check: false });
                    }
                }
                PowerId::FireBreathing => {
                    if t == CardType::Status || t == CardType::Curse {
                        let damage = self.pure_damage_matrix(amount);
                        self.push_bot(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Thorns });
                    }
                }
                _ => {}
            }
        }
    }

    /// `onExhaust` for the player's powers.
    pub(crate) fn on_exhaust(&mut self) {
        for (id, amount) in self.power_ids(Cr::Player) {
            match id {
                PowerId::DarkEmbrace => {
                    if !self.monsters_basically_dead() {
                        self.push_bot(Action::DrawCard { amount, follow_up: None, shuffle_check: false });
                    }
                }
                PowerId::FeelNoPain => self.push_bot(Action::GainBlock { target: Cr::Player, amount }),
                _ => {}
            }
        }
    }

    /// `onGainedBlock` for the player's powers.
    pub(crate) fn on_gained_block(&mut self, block: f32) {
        for (id, amount) in self.power_ids(Cr::Player) {
            if id == PowerId::Juggernaut && block > 0.0 {
                let info = DamageInfo::new(Some(Cr::Player), amount, DamageType::Thorns);
                self.push_bot(Action::DamageRandomEnemy { info });
            }
            if id == PowerId::WaveOfTheHand && block > 0.0 {
                // Every monster in the group, dead ones included (their ApplyPowerAction fizzles).
                for m in self.order.clone() {
                    self.bot_power(Cr::Monster(m), Some(Cr::Player), Power::weak(amount, false));
                }
            }
        }
    }

    /// `onAttackedToChangeDamage` for the target's powers (Buffer).
    pub(crate) fn on_attacked_to_change_damage(&mut self, target: Cr, mut dmg: i32) -> i32 {
        for (id, _) in self.power_ids(target) {
            if id == PowerId::Buffer {
                if dmg > 0 {
                    self.push_top(Action::ReducePower { target, id, amount: 1 });
                }
                dmg = 0;
            }
            if id == PowerId::Invincible {
                if let Some(p) = self.powers_mut(target).iter_mut().find(|p| p.id == PowerId::Invincible) {
                    if dmg > p.amount {
                        dmg = p.amount;
                    }
                    p.amount = (p.amount - dmg).max(0);
                }
            }
        }
        dmg
    }

    /// `onAttacked` for the target's powers.
    pub(crate) fn on_attacked(&mut self, target: Cr, info: &DamageInfo, dmg: i32) -> i32 {
        let n = self.powers(target).len();
        for i in 0..n {
            let Some(p) = self.powers(target).get(i).cloned() else { break };
            match p.id {
                PowerId::Angry => {
                    if info.owner.is_some() && dmg > 0 && info.damage_type != DamageType::HpLoss && info.damage_type != DamageType::Thorns {
                        self.top_power(target, Some(target), Power::new(PowerId::Strength, p.amount));
                    }
                }
                PowerId::CurlUp => {
                    if !p.triggered && dmg < self.hp(target) && dmg > 0 && info.owner.is_some() && info.damage_type == DamageType::Normal {
                        self.powers_mut(target)[i].triggered = true;
                        if let Cr::Monster(m) = target {
                            self.push_bot(Action::ChangeState { monster: m, state: MonsterState::Closed });
                        }
                        self.push_bot(Action::GainBlock { target, amount: p.amount });
                        self.push_bot(Action::RemovePower { target, id: PowerId::CurlUp });
                    }
                }
                PowerId::StaticDischarge => {
                    if info.damage_type != DamageType::Thorns && info.damage_type != DamageType::HpLoss && info.owner.is_some_and(|o| o != target) && dmg > 0 {
                        for _ in 0..p.amount {
                            self.push_top(Action::Channel { orb: super::Orb::new(super::OrbId::Lightning), auto_evoke: true });
                        }
                    }
                }
                PowerId::Thorns => {
                    if let Some(owner) = info.owner {
                        if info.damage_type != DamageType::Thorns && info.damage_type != DamageType::HpLoss && owner != target {
                            let thorns = DamageInfo::new(Some(target), p.amount, DamageType::Thorns);
                            self.push_top(Action::Damage { target: owner, source: DamageSource::Fixed(thorns), steal_gold: 0 });
                        }
                    }
                }
                PowerId::FlameBarrier => {
                    if let Some(owner) = info.owner {
                        if info.damage_type != DamageType::Thorns && info.damage_type != DamageType::HpLoss && owner != target {
                            let thorns = DamageInfo::new(Some(target), p.amount, DamageType::Thorns);
                            self.push_top(Action::Damage { target: owner, source: DamageSource::Fixed(thorns), steal_gold: 0 });
                        }
                    }
                }
                // Talk to the Hand: the player gains block for each attack on this monster.
                PowerId::BlockReturn => {
                    if info.damage_type != DamageType::Thorns && info.damage_type != DamageType::HpLoss && info.owner.is_some_and(|o| o != target) {
                        self.push_top(Action::GainBlock { target: Cr::Player, amount: p.amount });
                    }
                }
                PowerId::Flight => {
                    let hit = info.damage_type != DamageType::HpLoss && info.damage_type != DamageType::Thorns;
                    let taken = if hit { dmg as f32 / 2.0 } else { dmg as f32 };
                    if info.owner.is_some() && hit && dmg > 0 && taken < self.hp(target) as f32 {
                        self.push_bot(Action::ReducePower { target, id: PowerId::Flight, amount: 1 });
                    }
                }
                PowerId::Shifting => {
                    if dmg > 0 {
                        self.top_power(target, Some(target), Power::new(PowerId::Strength, -dmg));
                        if !self.has_power(target, PowerId::Artifact) {
                            self.top_power(target, Some(target), Power::new(PowerId::Shackled, dmg));
                        }
                    }
                }
                PowerId::Reactive => {
                    if info.owner.is_some()
                        && info.damage_type != DamageType::HpLoss
                        && info.damage_type != DamageType::Thorns
                        && dmg > 0
                        && dmg < self.hp(target)
                    {
                        if let Cr::Monster(m) = target {
                            self.push_bot(Action::RollMove { monster: m });
                        }
                    }
                }
                PowerId::Malleable => {
                    if dmg < self.hp(target) && dmg > 0 && info.owner.is_some() && info.damage_type == DamageType::Normal {
                        if target == Cr::Player {
                            self.push_top(Action::GainBlock { target, amount: p.amount });
                        } else {
                            self.push_bot(Action::GainBlock { target, amount: p.amount });
                        }
                        self.powers_mut(target)[i].amount += 1;
                    }
                }
                _ => {}
            }
        }
        dmg
    }

    /// `wasHPLost` for the target's powers.
    pub(crate) fn was_hp_lost(&mut self, target: Cr, info: &DamageInfo, dmg: i32) {
        for (id, amount) in self.power_ids(target) {
            if id == PowerId::Rupture && dmg > 0 && info.owner == Some(target) {
                self.top_power(target, Some(target), Power::new(PowerId::Strength, amount));
            }
            if id == PowerId::PlatedArmor
                && info.owner.is_some_and(|o| o != target)
                && info.damage_type != DamageType::HpLoss
                && info.damage_type != DamageType::Thorns
                && dmg > 0
            {
                self.push_bot(Action::ReducePower { target, id, amount: 1 });
            }
        }
    }

    /// `onDeath` for a dying monster's powers.
    pub(crate) fn on_death(&mut self, m: usize) {
        for (id, amount) in self.power_ids(Cr::Monster(m)) {
            if id == PowerId::SporeCloud && !self.monsters_basically_dead() {
                let turn_has_ended = self.turn_has_ended;
                self.top_power(Cr::Player, None, Power::vulnerable(amount, true, turn_has_ended));
            }
            if id == PowerId::CorpseExplosion && !self.monsters_basically_dead() && self.monsters[m].hp <= 0 {
                let damage = self.pure_damage_matrix(self.monsters[m].max_hp * amount);
                self.push_bot(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Thorns });
            }
            if id == PowerId::Stasis {
                if let crate::monsters::Ai::BronzeOrb { stasis_card: Some(card), .. } = &self.monsters[m].ai {
                    let card = card.clone();
                    if self.hand.len() != 10 {
                        self.push_bot(Action::MakeTempCardInHandSame { card });
                    } else {
                        self.push_bot(Action::MakeTempCardInDiscardSame { card });
                    }
                }
            }
        }
    }

    /// Plays a copy of `card` next (`makeSameInstanceOf`, purged on use, queued at the front): Burst,
    /// Echo Form, Amplify.
    fn queue_copy(&mut self, card: &Card, target: Option<usize>) {
        let mut tmp = self.play_copy(card);
        if let Some(m) = target {
            self.calculate_card_damage(&mut tmp, Some(m));
        }
        tmp.purge_on_use = true;
        let inst = tmp.instance;
        self.limbo.insert(0, tmp);
        let item = super::QueueItem {
            card: super::QueuedCard::InLimbo(inst),
            target,
            energy_on_use: card.energy_on_use,
            ignore_energy_total: true,
            autoplay: true,
            random_target: false,
            end_turn_autoplay: false,
        };
        if self.card_queue.is_empty() {
            self.card_queue.push_back(item);
        } else {
            self.card_queue.insert(1, item);
        }
    }

    /// `onAfterUseCard` for the player's powers (Rebound). Returns whether the card goes on top of
    /// the draw pile (`UseCardAction.reboundCard`).
    pub(crate) fn player_on_after_use_card(&mut self, card: &Card) -> bool {
        let mut rebound = false;
        if let Some(i) = self.player.powers.iter().position(|p| p.id == PowerId::Rebound) {
            if self.player.powers[i].just_applied {
                self.player.powers[i].just_applied = false;
            } else {
                rebound = card.card_type() != CardType::Power;
                self.push_bot(Action::ReducePower { target: Cr::Player, id: PowerId::Rebound, amount: 1 });
            }
        }
        rebound
    }

    /// `onAfterCardPlayed` for the player's powers (A Thousand Cuts).
    pub(crate) fn on_after_card_played(&mut self) {
        for (id, amount) in self.power_ids(Cr::Player) {
            if id == PowerId::ThousandCuts {
                let damage = self.pure_damage_matrix(amount);
                self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage, damage_type: DamageType::Thorns });
            }
        }
    }

    /// `onAttack` for the attacker's powers, here the player's (Envenom): `dmg` is what got
    /// through block.
    pub(crate) fn player_on_attack(&mut self, info: &DamageInfo, dmg: i32, target: Cr) {
        for (id, amount) in self.power_ids(Cr::Player) {
            if id == PowerId::Envenom && dmg > 0 && target != Cr::Player && info.damage_type == DamageType::Normal {
                let a = self.apply_power_action(target, Some(Cr::Player), Power::new(PowerId::Poison, amount), amount);
                self.push_top(a);
            }
        }
    }

    /// `onAfterUseCard` for the monsters' powers (Slow, Time Warp).
    pub(crate) fn on_after_use_card(&mut self) {
        for m in self.order.clone() {
            for (id, _) in self.power_ids(Cr::Monster(m)) {
                match id {
                    PowerId::Slow => self.bot_power(Cr::Monster(m), Some(Cr::Monster(m)), Power::new(PowerId::Slow, 1)),
                    PowerId::BeatOfDeath => {
                        let amount = self.monsters[m].power_amount(PowerId::BeatOfDeath);
                        let info = DamageInfo::new(Some(Cr::Monster(m)), amount, DamageType::Thorns);
                        self.push_bot(Action::Damage { target: Cr::Player, source: DamageSource::Fixed(info), steal_gold: 0 });
                    }
                    PowerId::TimeWarp => {
                        let mut ends = false;
                        if let Some(p) = self.monsters[m].powers.iter_mut().find(|p| p.id == PowerId::TimeWarp) {
                            p.amount += 1;
                            if p.amount == 12 {
                                p.amount = 0;
                                ends = true;
                            }
                        }
                        if ends {
                            self.end_turn_early();
                            for i in self.order.clone() {
                                self.bot_power(Cr::Monster(i), Some(Cr::Monster(i)), Power::new(PowerId::Strength, 2));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
