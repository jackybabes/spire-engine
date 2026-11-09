//! `AbstractGameAction.update()` for every action, collapsed to one step, plus the
//! continuations of actions that open a selection screen.

use super::{ChoiceKind, ChoicePile, Combat, Orb, OrbId, PendingChoice, QueueItem, QueuedCard};
use crate::action::{Action, Cr, DamageInfo, DamageSource, DamageType};
use crate::cards::{Card, CardId, CardType};
use crate::monsters::{Intent, NO_MOVE};
use crate::powers::{Power, PowerId};
use sts_rng::{java_shuffle, JavaRandom};

impl Combat {
    pub(crate) fn execute(&mut self, action: Action) {
        match action {
            Action::Damage { target, source, steal_gold } => self.exec_damage(target, source, steal_gold),
            Action::DamageAllEnemies { source, damage, damage_type } => {
                for (pos, i) in self.order.clone().into_iter().enumerate() {
                    if !self.monsters[i].is_dead_or_escaped() {
                        let out = damage.get(pos).copied().unwrap_or(0);
                        self.damage(Cr::Monster(i), DamageInfo::new(source, out, damage_type));
                    }
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::VampireDamageAllEnemies { damage } => {
                let mut heal = 0;
                for (pos, i) in self.order.clone().into_iter().enumerate() {
                    let m = &self.monsters[i];
                    if !m.is_dying && m.hp > 0 && !m.is_escaping {
                        let hp_before = self.monsters[i].hp;
                        let out = damage.get(pos).copied().unwrap_or(0);
                        self.damage(Cr::Monster(i), DamageInfo::new(Some(Cr::Player), out, DamageType::Normal));
                        // lastDamageTaken = min(damage after block, currentHealth before)
                        heal += (hp_before - self.monsters[i].hp).max(0);
                    }
                }
                if heal > 0 {
                    self.push_bot(Action::Heal { target: Cr::Player, amount: heal });
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::DamageRandomEnemy { info } => {
                if let Some(t) = self.get_random_monster(None, true, true) {
                    self.push_top(Action::Damage { target: Cr::Monster(t), source: DamageSource::Fixed(info), steal_gold: 0 });
                }
            }
            Action::PummelDamage { target, info } => {
                if self.monsters[target].hp > 0 {
                    if info.damage_type != DamageType::Thorns && info.owner.map_or(false, |o| self.is_dying(o)) {
                        return;
                    }
                    self.damage(Cr::Monster(target), info);
                    if self.monsters_basically_dead() {
                        self.clear_post_combat_actions();
                    }
                }
            }
            Action::Feed { target, info, max_hp } => {
                self.damage(Cr::Monster(target), info);
                let m = &self.monsters[target];
                if (m.is_dying || m.hp <= 0) && !m.half_dead && !m.has_power(PowerId::Minion) {
                    self.increase_max_hp(max_hp);
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::RitualDagger { target, info, amount, uuid } => {
                self.damage(Cr::Monster(target), info);
                let m = &self.monsters[target];
                if (m.is_dying || m.hp <= 0) && !m.half_dead && !m.has_power(PowerId::Minion) {
                    self.master_deck_growth.push((uuid, amount));
                    // GetAllInBattleInstances: the card in use too.
                    self.for_each_battle_instance(uuid, |c| {
                        c.misc += amount;
                        c.base_damage = c.misc;
                    });
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::LoseHp { target, source, amount } => {
                self.damage(target, DamageInfo::new(source, amount, DamageType::HpLoss));
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::Heal { target, amount } => self.heal(target, amount),
            Action::IncreaseMaxHp { target, amount } => {
                // AbstractCreature.increaseMaxHp: max HP up, then heal by the same amount.
                self.monsters[target].max_hp += amount;
                self.heal(Cr::Monster(target), amount);
            }
            Action::GainBlock { target, amount } => {
                if !self.is_dying(target) {
                    self.add_block(target, amount);
                }
            }
            Action::LoseBlock { target, amount } => {
                let b = self.block(target);
                if b != 0 {
                    self.set_block(target, (b - amount).max(0));
                }
            }
            Action::DoubleYourBlock { target } => {
                let b = self.block(target);
                if b > 0 {
                    self.add_block(target, b);
                }
            }
            Action::GainBlockRandomMonster { source, amount } => {
                let valid: Vec<usize> = self
                    .order
                    .iter()
                    .copied()
                    .filter(|&i| i != source && self.monsters[i].intent != Intent::Escape && !self.monsters[i].is_dying)
                    .collect();
                let t = if valid.is_empty() {
                    source
                } else {
                    let k = self.rng.ai.random(valid.len() as i32 - 1);
                    valid[k as usize]
                };
                self.add_block(Cr::Monster(t), amount);
            }
            Action::ApplyPower { target, source, power, amount } => self.exec_apply_power(target, source, power, amount),
            Action::ReducePower { target, id, amount } => self.exec_reduce_power(target, id, amount),
            Action::RemovePower { target, id } => self.exec_remove_power(target, id),
            Action::GainEnergy { amount } => self.energy = (self.energy + amount).min(999),
            Action::LoseEnergy { amount } => self.energy = (self.energy - amount).max(0),
            Action::GainEnergyAndEnableControls { amount } => {
                self.energy = (self.energy + amount).min(999);
                self.powers_on_energy_recharge();
                self.turn_has_ended = false;
            }
            Action::DrawCard { amount, follow_up, shuffle_check } => self.exec_draw(amount, follow_up, shuffle_check, true),
            Action::DrawCardKeepHistory { amount, follow_up } => self.exec_draw(amount, follow_up, false, false),
            Action::EmptyDeckShuffle => {
                let seed = self.rng.shuffle.random_long();
                java_shuffle(&mut self.discard_pile, &mut JavaRandom::new(seed));
                // Each card lands in the draw pile through a Soul, which resets it there
                // (`clearPowers`: its cost for the turn, as in every move to the draw pile).
                let cards: Vec<Card> = self.discard_pile.drain(..).map(|mut c| {
                    c.reset_attributes();
                    c
                }).collect();
                self.draw_pile.extend(cards);
            }
            Action::Exhaust { amount, random, any_number, can_pick_zero } => self.exec_exhaust(amount, random, any_number, can_pick_zero),
            Action::ExhaustSpecificInHand { instance } => {
                if let Some(pos) = self.hand.iter().position(|c| c.instance == instance) {
                    let mut c = self.hand.remove(pos);
                    c.exhaust_on_use_once = false;
                    c.free_to_play_once = false;
                    self.move_to_exhaust_pile(c);
                }
            }
            Action::ExhaustAllEthereal => {
                let ids: Vec<u32> = self.hand.iter().filter(|c| c.ethereal).map(|c| c.instance).collect();
                for inst in ids {
                    self.push_top(Action::ExhaustSpecificInHand { instance: inst });
                }
            }
            Action::ExhaustAllNonAttack => {
                let ids: Vec<u32> = self.hand.iter().filter(|c| c.card_type() != CardType::Attack).map(|c| c.instance).collect();
                for inst in ids {
                    self.push_top(Action::ExhaustSpecificInHand { instance: inst });
                }
            }
            Action::Discard { amount, random, end_turn } => self.exec_discard(amount, random, end_turn),
            Action::DiscardAtEndOfTurn => self.exec_discard_at_end_of_turn(),
            Action::RestoreRetainedCards => {
                let mut keep = Vec::new();
                for mut c in std::mem::take(&mut self.limbo) {
                    if c.retain || c.self_retain {
                        // onRetained.
                        match c.id {
                            CardId::Perseverance => c.base_block += c.magic,
                            CardId::WindmillStrike => c.base_damage += c.magic,
                            CardId::SandsOfTime => self.push_bot(Action::ReduceCostCard { instance: c.instance }),
                            _ => {}
                        }
                        c.retain = false;
                        self.hand.push(c);
                    } else {
                        keep.push(c);
                    }
                }
                self.limbo = keep;
            }
            Action::ClearCardQueue => {
                let queued: Vec<u32> = self.card_queue.iter().filter_map(|q| q.card.instance()).collect();
                self.limbo.retain(|c| !queued.contains(&c.instance));
                self.card_queue.clear();
            }
            Action::MakeTempCardInDiscard { card, amount } => {
                if amount < 6 {
                    for _ in 0..amount {
                        let mut c = self.stat_copy(&card);
                        self.master_reality(&mut c);
                        self.discard_pile.push(c);
                    }
                }
            }
            Action::MakeTempCardInHand { card, amount } => {
                if amount == 0 {
                    return;
                }
                let mut to_discard = 0;
                let mut to_hand = amount;
                if amount + self.hand.len() as i32 > 10 {
                    to_discard = amount + self.hand.len() as i32 - 10;
                    to_hand -= to_discard;
                }
                for _ in 0..to_hand.max(0) {
                    let mut c = self.stat_copy(&card);
                    self.master_reality(&mut c);
                    let skill = c.card_type() == CardType::Skill;
                    if self.player.has_power(PowerId::Corruption) && skill {
                        c.set_cost_for_turn(-9);
                    }
                    self.hand.push(c);
                    self.on_card_draw_or_discard();
                }
                for _ in 0..to_discard {
                    let mut c = self.stat_copy(&card);
                    self.master_reality(&mut c);
                    self.discard_pile.push(c);
                }
            }
            Action::MakeTempCardInDrawPile { card, amount, random_spot, to_bottom } => {
                for _ in 0..amount {
                    let first = self.stat_copy(&card);
                    let mut c = self.stat_copy(&first);
                    self.master_reality(&mut c);
                    if to_bottom {
                        self.draw_pile.insert(0, c);
                    } else if random_spot {
                        self.add_to_random_spot(c);
                    } else {
                        self.draw_pile.push(c);
                    }
                }
            }
            Action::UseCard { card, target, exhaust_card } => self.exec_use_card(card, target, exhaust_card),
            Action::QueueEndTurn => {
                if !self.card_queue.iter().any(|q| matches!(q.card, QueuedCard::EndTurn)) {
                    self.card_queue.push_back(QueueItem {
                        card: QueuedCard::EndTurn,
                        target: None,
                        energy_on_use: 0,
                        ignore_energy_total: false,
                        autoplay: false,
                        random_target: false,
                        end_turn_autoplay: false,
                    });
                }
            }
            Action::QueueCard { card, target, immediate, autoplay, random_target } => {
                let inst = card.instance;
                if !self.card_queue.iter().any(|q| q.card.instance() == Some(inst)) {
                    let item = QueueItem {
                        card: QueuedCard::Owned(Box::new(card)),
                        target,
                        energy_on_use: self.energy,
                        ignore_energy_total: false,
                        autoplay,
                        random_target,
                        end_turn_autoplay: false,
                    };
                    if immediate && !self.card_queue.is_empty() {
                        self.card_queue.insert(1, item);
                    } else {
                        self.card_queue.push_back(item);
                    }
                }
            }
            Action::PlayTopCard { target, exhaust } => {
                if self.draw_pile.len() + self.discard_pile.len() == 0 {
                    return;
                }
                if self.draw_pile.is_empty() {
                    self.push_top(Action::PlayTopCard { target, exhaust });
                    self.relics_on_shuffle();
                self.push_top(Action::EmptyDeckShuffle);
                    return;
                }
                let mut card = self.draw_pile.pop().unwrap();
                card.exhaust_on_use_once = exhaust;
                // `card.applyPowers()` once it has left the draw pile.
                self.apply_powers_kept(&mut card);
                self.push_top(Action::QueueCard { card, target, immediate: false, autoplay: true, random_target: false });
            }
            Action::EndTurnSequence => {
                self.push_bot(Action::EndTurn);
                if !self.skip_monster_turn {
                    self.push_bot(Action::MonsterStartTurn);
                }
                self.monster_attacks_queued = false;
            }
            Action::EndTurn => self.turn_has_ended = true,
            Action::MonsterStartTurn => {
                // MonsterGroup.applyPreTurnLogic
                for i in self.order.clone() {
                    if !self.monsters[i].is_dying && !self.monsters[i].is_escaping {
                        if !self.monsters[i].has_power(PowerId::Barricade) {
                            self.monsters[i].block = 0;
                        }
                        self.apply_start_of_turn_powers(Cr::Monster(i));
                    }
                }
            }
            Action::RollMove { monster } => self.roll_move(monster),
            Action::SetMove { monster, mv } => self.monsters[monster].set_move(mv),
            Action::ChangeState { monster, state } => self.change_state(monster, state),
            Action::Escape { monster } => self.monsters[monster].is_escaping = true,
            Action::Suicide { monster, trigger_relics } => {
                self.monsters[monster].gold = 0;
                self.monsters[monster].hp = 0;
                self.monster_die(monster, trigger_relics);
            }
            Action::SpawnMonster { monster, minion } => {
                let mut m = *monster;
                let pos = self.order.iter().filter(|&&i| m.draw_x > self.monsters[i].draw_x).count();
                m.next_move = NO_MOVE;
                let id = self.monsters.len();
                self.monsters.push(m);
                self.relics_on_spawn_monster(id);
                // init() runs before the monster joins the group.
                self.roll_move(id);
                self.monster_apply_powers(id);
                self.order.insert(pos, id);
                if minion {
                    self.top_power(Cr::Monster(id), Some(Cr::Monster(id)), Power::flag(PowerId::Minion));
                }
            }
            Action::CannotLose => self.cannot_lose = true,
            Action::CanLose => self.cannot_lose = false,
            Action::LooterRecordGold { monster } => {
                if let crate::monsters::Ai::Looter { gold_amt, stolen_gold, .. } | crate::monsters::Ai::Mugger { gold_amt, stolen_gold, .. } =
                    &mut self.monsters[monster].ai
                {
                    *stolen_gold += (*gold_amt).min(self.player.gold);
                }
            }
            Action::Whirlwind { damage, free_to_play_once, energy_on_use } => {
                let mut effect = if energy_on_use != -1 { energy_on_use } else { self.energy };
                if self.has_relic(crate::relics::RelicId::ChemicalX) {
                    effect += 2;
                }
                if effect > 0 {
                    for _ in 0..effect {
                        self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: damage.clone(), damage_type: DamageType::Normal });
                    }
                    if !free_to_play_once {
                        self.energy = 0;
                    }
                }
            }
            Action::SpotWeakness { amount, target } => {
                if let Some(m) = target {
                    if self.monsters[m].intent_base_dmg >= 0 {
                        self.bot_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::Strength, amount));
                    }
                }
            }
            Action::PutOnDeck { amount, random } => self.exec_put_on_deck(amount, random),
            Action::ModifyDamage { uuid, amount } => {
                let bump = |c: &mut Card| {
                    if c.uuid == uuid {
                        c.base_damage = (c.base_damage + amount).max(0);
                    }
                };
                self.draw_pile.iter_mut().for_each(bump);
                self.discard_pile.iter_mut().for_each(bump);
                self.exhaust_pile.iter_mut().for_each(bump);
                self.limbo.iter_mut().for_each(bump);
                self.hand.iter_mut().for_each(bump);
                // The card currently being used lives in its pending UseCardAction.
                for a in self.actions.iter_mut() {
                    if let Action::UseCard { card, .. } = a {
                        bump(card);
                    }
                }
            }
            Action::LimitBreak => {
                if self.player.has_power(PowerId::Strength) {
                    let s = self.power_amount(Cr::Player, PowerId::Strength);
                    self.top_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::Strength, s));
                }
            }
            Action::FiendFire { target, info } => {
                let count = self.hand.len();
                for _ in 0..count {
                    self.push_top(Action::Damage { target: Cr::Monster(target), source: DamageSource::Fixed(info), steal_gold: 0 });
                }
                for _ in 0..count {
                    self.push_top(Action::Exhaust { amount: 1, random: true, any_number: true, can_pick_zero: false });
                }
            }
            Action::Exhume { upgrade } => self.exec_exhume(upgrade),
            Action::DualWield { amount } => self.exec_dual_wield(amount),
            Action::Dropkick { target, info } => {
                if self.monsters[target].has_power(PowerId::Vulnerable) {
                    self.push_top(Action::DrawCard { amount: 1, follow_up: None, shuffle_check: false });
                    self.push_top(Action::GainEnergy { amount: 1 });
                }
                self.push_top(Action::Damage { target: Cr::Monster(target), source: DamageSource::Fixed(info), steal_gold: 0 });
            }
            Action::DiscardPileToTopOfDeck => self.exec_discard_to_top(),
            Action::BurnIncrease => {
                for c in self.discard_pile.iter_mut().chain(self.draw_pile.iter_mut()) {
                    if c.id == CardId::Burn {
                        c.upgrade();
                    }
                }
                for _ in 0..3 {
                    let mut b = self.new_card(CardId::Burn);
                    b.upgrade();
                    self.discard_pile.push(b);
                }
            }
            Action::BlockPerNonAttack { amount } => {
                let ids: Vec<u32> = self.hand.iter().filter(|c| c.card_type() != CardType::Attack).map(|c| c.instance).collect();
                for _ in &ids {
                    self.push_top(Action::GainBlock { target: Cr::Player, amount });
                }
                for inst in ids {
                    self.push_top(Action::ExhaustSpecificInHand { instance: inst });
                }
            }
            Action::AttackDamageRandomEnemy { mut card } => {
                if let Some(t) = self.get_random_monster(None, true, true) {
                    self.calculate_card_damage(&mut card, Some(t));
                    let info = DamageInfo::new(Some(Cr::Player), card.damage, DamageType::Normal);
                    self.push_top(Action::Damage { target: Cr::Monster(t), source: DamageSource::Fixed(info), steal_gold: 0 });
                }
            }
            Action::Armaments { upgraded } => self.exec_armaments(upgraded),
            Action::ChooseOneColorless => {
                let mut offer: Vec<Card> = Vec::new();
                while offer.len() != 3 {
                    let id = self.truly_random_colorless_in_combat();
                    if !offer.iter().any(|c| c.id == id) {
                        let c = self.new_card(id);
                        offer.push(c);
                    }
                }
                self.pending = Some(PendingChoice {
                    kind: ChoiceKind::Discover,
                    pile: ChoicePile::Offer,
                    count: 1,
                    any_number: false,
                    can_pick_zero: false,
                    options: (0..offer.len()).collect(),
                    offer,
                });
            }
            Action::GamblingChip => self.open_choice(ChoiceKind::GamblingChip, ChoicePile::Hand, 99, true, true),
            Action::RedSkullCheck => {
                if let Some(i) = self.player.relics.iter().position(|r| r.id == crate::relics::RelicId::RedSkull) {
                    if !self.player.relics[i].flag(crate::relics::FLAG_A) && self.player.is_bloodied {
                        self.add_power_direct(Cr::Player, Power::new(PowerId::Strength, 3));
                        self.player.relics[i].set_flag(crate::relics::FLAG_A, true);
                        self.on_modify_power();
                    }
                }
            }
            Action::Discovery { card_type, colorless, amount } => self.exec_discovery(card_type, colorless, amount),
            Action::UpgradeRandomCard => {
                let mut cands: Vec<usize> = (0..self.hand.len()).filter(|&i| self.hand[i].can_upgrade() && self.hand[i].card_type() != CardType::Status).collect();
                if !cands.is_empty() {
                    let seed = self.rng.shuffle.random_long();
                    java_shuffle(&mut cands, &mut JavaRandom::new(seed));
                    self.hand[cands[0]].upgrade();
                }
            }
            Action::Codex => {
                if !self.monsters_basically_dead() {
                    let mut offer: Vec<Card> = Vec::new();
                    while offer.len() != 3 {
                        let id = self.truly_random_any_card_in_combat();
                        if !offer.iter().any(|c| c.id == id) {
                            let c = self.new_card(id);
                            offer.push(c);
                        }
                    }
                    self.pending = Some(PendingChoice {
                        kind: ChoiceKind::Codex,
                        pile: ChoicePile::Offer,
                        count: 1,
                        any_number: false,
                        can_pick_zero: true,
                        options: (0..offer.len()).collect(),
                        offer,
                    });
                }
            }
            Action::DiscardToHand { amount, set_cost } => {
                if self.discard_pile.is_empty() || amount <= 0 {
                    return;
                }
                if self.discard_pile.len() as i32 <= amount {
                    for mut c in std::mem::take(&mut self.discard_pile) {
                        if self.hand.len() < 10 {
                            if set_cost {
                                c.set_cost_for_turn(0);
                            }
                            self.hand.push(c);
                        } else {
                            self.discard_pile.push(c);
                        }
                    }
                    return;
                }
                self.open_choice(ChoiceKind::DiscardToHand { set_cost }, ChoicePile::Discard, amount as usize, false, false);
            }
            Action::RandomizeHandCost => {
                for i in 0..self.hand.len() {
                    if self.hand[i].cost >= 0 {
                        let n = self.rng.card_random.random(3);
                        let c = &mut self.hand[i];
                        if c.cost != n {
                            c.cost = n;
                            c.cost_for_turn = n;
                        }
                    }
                }
            }
            Action::ObtainRandomPotion => {
                if !self.has_relic(crate::relics::RelicId::Sozu) {
                    let p = crate::run::pools::return_random_potion_limited(self.character, &mut self.rng.potion, true);
                    if let Some(s) = self.potions.iter_mut().find(|s| s.is_none()) {
                        *s = Some(p);
                    }
                }
            }
            Action::Regen { target } => {
                if self.hp(target) > 0 {
                    let amount = self.power_amount(target, PowerId::Regen);
                    self.heal(target, amount);
                }
                if target == Cr::Player {
                    if let Some(pos) = self.player.powers.iter().position(|p| p.id == PowerId::Regen) {
                        self.player.powers[pos].amount -= 1;
                        if self.player.powers[pos].amount == 0 {
                            self.player.powers.remove(pos);
                        }
                    }
                }
            }
            Action::ConditionalDraw { amount } => {
                if !self.hand.iter().any(|c| c.card_type() == CardType::Attack) {
                    self.push_top(Action::DrawCard { amount, follow_up: None, shuffle_check: false });
                }
            }
            Action::FromDeckToHand { card_type } => {
                // CardGroup.addToRandomSpot builds the list in a random order.
                let mut tmp: Vec<usize> = Vec::new();
                for i in 0..self.draw_pile.len() {
                    if self.draw_pile[i].card_type() == card_type {
                        if tmp.is_empty() {
                            tmp.push(i);
                        } else {
                            let k = self.rng.card_random.random(tmp.len() as i32 - 1) as usize;
                            tmp.insert(k, i);
                        }
                    }
                }
                match tmp.len() {
                    0 => {}
                    1 => {
                        let c = self.draw_pile.remove(tmp[0]);
                        if self.hand.len() == 10 {
                            self.discard_pile.push(c);
                        } else {
                            self.hand.push(c);
                        }
                    }
                    _ => {
                        self.pending = Some(PendingChoice {
                            kind: ChoiceKind::FromDeckToHand,
                            pile: ChoicePile::Draw,
                            count: 1,
                            any_number: false,
                            can_pick_zero: false,
                            options: tmp,
                            offer: Vec::new(),
                        });
                    }
                }
            }
            Action::ShuffleDrawPile => {
                let seed = self.rng.shuffle.random_long();
                java_shuffle(&mut self.draw_pile, &mut JavaRandom::new(seed));
            }
            Action::DrawPileToHand { amount, card_type } => {
                let mut tmp: Vec<u32> = Vec::new();
                for c in &self.draw_pile {
                    if c.card_type() == card_type {
                        if tmp.is_empty() {
                            tmp.push(c.instance);
                        } else {
                            let k = self.rng.card_random.random(tmp.len() as i32 - 1) as usize;
                            tmp.insert(k, c.instance);
                        }
                    }
                }
                for _ in 0..amount {
                    if tmp.is_empty() {
                        break;
                    }
                    let seed = self.rng.shuffle.random_long();
                    java_shuffle(&mut tmp, &mut JavaRandom::new(seed));
                    let inst = tmp.remove(0);
                    if let Some(pos) = self.draw_pile.iter().position(|c| c.instance == inst) {
                        let c = self.draw_pile.remove(pos);
                        if self.hand.len() == 10 {
                            self.discard_pile.push(c);
                        } else {
                            self.hand.push(c);
                        }
                    }
                }
            }
            Action::Apotheosis => {
                for c in self.hand.iter_mut().chain(self.draw_pile.iter_mut()).chain(self.discard_pile.iter_mut()).chain(self.exhaust_pile.iter_mut()) {
                    if c.can_upgrade() {
                        c.upgrade();
                    }
                }
            }
            Action::Greed { target, info, gold } => {
                self.damage(Cr::Monster(target), info);
                let mon = &self.monsters[target];
                if (mon.is_dying || mon.hp <= 0) && !mon.half_dead && !mon.powers.iter().any(|p| p.id == PowerId::Minion) {
                    // gainGold: Ectoplasm blocks it, Bloody Idol heals.
                    if !self.has_relic(crate::relics::RelicId::Ectoplasm) {
                        self.player.gold += gold;
                        if self.has_relic(crate::relics::RelicId::BloodyIdol) {
                            self.heal(Cr::Player, 5);
                        }
                    }
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::Forethought { any } => {
                if self.hand.is_empty() {
                    return;
                }
                if self.hand.len() == 1 && !any {
                    let mut c = self.hand.pop().unwrap();
                    if c.cost > 0 {
                        c.free_to_play_once = true;
                    }
                    c.reset_attributes();
                    self.draw_pile.insert(0, c);
                    return;
                }
                if any {
                    self.open_choice(ChoiceKind::Forethought, ChoicePile::Hand, 99, true, true);
                } else {
                    self.open_choice(ChoiceKind::Forethought, ChoicePile::Hand, 1, false, false);
                }
            }
            Action::Enlightenment { combat } => {
                for c in self.hand.iter_mut() {
                    if c.cost_for_turn > 1 {
                        c.cost_for_turn = 1;
                    }
                    if combat && c.cost > 1 {
                        c.cost = 1;
                    }
                }
            }
            Action::Transmutation { upgraded, free, energy_on_use } => {
                let mut effect = if energy_on_use != -1 { energy_on_use } else { self.energy };
                if self.has_relic(crate::relics::RelicId::ChemicalX) {
                    effect += 2;
                }
                if effect > 0 {
                    for _ in 0..effect {
                        let id = self.truly_random_colorless_in_combat();
                        let mut c = self.new_card(id);
                        if upgraded {
                            c.upgrade();
                        }
                        c.set_cost_for_turn(0);
                        self.push_bot(Action::MakeTempCardInHand { card: c, amount: 1 });
                    }
                    if !free {
                        self.energy = 0;
                    }
                }
            }
            Action::Madness => {
                let better = self.hand.iter().any(|c| c.cost_for_turn > 0);
                let possible = self.hand.iter().any(|c| c.cost_for_turn <= 0 && c.cost > 0);
                if better || possible {
                    loop {
                        let k = self.rng.card_random.random(self.hand.len() as i32 - 1) as usize;
                        let c = &mut self.hand[k];
                        if (better && c.cost_for_turn > 0) || (!better && c.cost > 0) {
                            c.cost = 0;
                            c.cost_for_turn = 0;
                            break;
                        }
                    }
                }
            }
            Action::ReduceBomb { uid } => {
                if let Some(pos) = self.player.powers.iter().position(|p| p.id == PowerId::TheBomb && p.uid == uid) {
                    if self.player.powers[pos].amount <= 1 {
                        self.player.powers.remove(pos);
                    } else {
                        self.player.powers[pos].amount -= 1;
                    }
                    self.on_modify_power();
                }
            }
            Action::MayhemPlay => {
                let t = self.get_random_monster(None, true, true);
                self.push_bot(Action::PlayTopCard { target: t, exhaust: false });
            }
            Action::RemoveDebuffs { target } => {
                let debuffs: Vec<PowerId> =
                    self.powers(target).iter().filter(|p| p.power_type() == crate::powers::PowerType::Debuff).map(|p| p.id).collect();
                for id in debuffs {
                    self.push_top(Action::RemovePower { target, id });
                }
            }
            Action::VampireDamage { target, source } => {
                let (info, attacker) = match source {
                    DamageSource::Fixed(info) => (info, info.owner),
                    DamageSource::MonsterSlot { monster, slot } => {
                        (DamageInfo::new(Some(Cr::Monster(monster)), self.monsters[monster].damage[slot].output, DamageType::Normal), Some(Cr::Monster(monster)))
                    }
                };
                self.damage(target, info);
                if self.last_damage_taken > 0 {
                    if let Some(a) = attacker {
                        self.push_top(Action::Heal { target: a, amount: self.last_damage_taken });
                    }
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::SpawnPrepared { monster, minion } => {
                // onSpawnMonster (no relic uses it), init(), applyPowers(), then addMonster at
                // the position counting every monster drawn further left.
                let x = self.monsters[monster].draw_x;
                let pos = self.order.iter().filter(|&&i| x > self.monsters[i].draw_x).count();
                self.relics_on_spawn_monster(monster);
                self.roll_move(monster);
                self.monster_apply_powers(monster);
                self.order.insert(pos, monster);
                if minion {
                    self.top_power(Cr::Monster(monster), Some(Cr::Monster(monster)), Power::flag(PowerId::Minion));
                }
            }
            Action::SummonGremlin { monster } => {
                // init(), applyPowers(), addMonster(getSmartPosition()), Minion, then
                // usePreBattleAction once the slide-in finishes.
                self.roll_move(monster);
                self.monster_apply_powers(monster);
                let x = self.monsters[monster].draw_x;
                let pos = self.order.iter().take_while(|&&i| x > self.monsters[i].draw_x).count();
                self.order.insert(pos, monster);
                self.bot_power(Cr::Monster(monster), Some(Cr::Monster(monster)), Power::flag(PowerId::Minion));
                self.use_pre_battle_action(monster);
            }
            Action::ApplyStasis { monster } => self.exec_apply_stasis(monster),
            Action::MakeTempCardInHandSame { card } => {
                let mut c = self.same_instance_copy(&card);
                self.master_reality(&mut c);
                if self.hand.len() < 10 {
                    if self.player.has_power(PowerId::Corruption) && c.card_type() == CardType::Skill {
                        c.set_cost_for_turn(-9);
                    }
                    self.hand.push(c);
                    self.on_card_draw_or_discard();
                } else {
                    self.discard_pile.push(c);
                }
            }
            Action::MakeTempCardInDiscardAndDeck { card } => {
                let mut a = self.stat_copy(&card);
                self.master_reality(&mut a);
                self.add_to_random_spot(a);
                let mut b = self.stat_copy(&card);
                self.master_reality(&mut b);
                self.discard_pile.push(b);
            }
            Action::AddCardToDeck { card } => self.obtain_card_in_combat(card),
            Action::MakeTempCardInDiscardSame { card } => {
                let mut c = self.same_instance_copy(&card);
                self.master_reality(&mut c);
                self.discard_pile.push(c);
            }
            Action::PoisonLoseHp { target, amount } => {
                if self.hp(target) > 0 {
                    // Poison only comes from the player.
                    let source = if target == Cr::Player { None } else { Some(Cr::Player) };
                    self.damage(target, DamageInfo::new(source, amount, DamageType::HpLoss));
                }
                // The poison counts down (removed straight from the list at 0).
                let powers = self.powers_mut(target);
                if let Some(pos) = powers.iter().position(|p| p.id == PowerId::Poison) {
                    powers[pos].amount -= 1;
                    if powers[pos].amount == 0 {
                        powers.remove(pos);
                    }
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::DiscardSpecific { instance } => {
                if let Some(pos) = self.hand.iter().position(|c| c.instance == instance) {
                    let c = self.hand.remove(pos);
                    self.discard_from_hand(c, true, false, false);
                }
            }
            Action::Bane { target, info } => {
                if self.monsters[target].has_power(PowerId::Poison) && self.monsters[target].hp > 0 {
                    if info.damage_type != DamageType::Thorns && info.owner.map_or(false, |o| self.is_dying(o)) {
                        return;
                    }
                    self.damage(Cr::Monster(target), info);
                    if self.monsters_basically_dead() {
                        self.clear_post_combat_actions();
                    }
                }
            }
            Action::BouncingFlask { target, amount, times } => {
                let Some(t) = target else { return };
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                    return;
                }
                if times > 1 {
                    let next = self.get_random_monster(None, true, true);
                    self.push_top(Action::BouncingFlask { target: next, amount, times: times - 1 });
                }
                if self.monsters[t].hp > 0 {
                    self.top_power(Cr::Monster(t), Some(Cr::Player), Power::new(PowerId::Poison, amount));
                }
            }
            Action::BulletTime => {
                for c in self.hand.iter_mut() {
                    c.set_cost_for_turn(-9);
                }
            }
            Action::CalculatedGamble => {
                // Always built with upgraded = false: the upgrade only stops the exhaust.
                let count = self.hand.len() as i32;
                if count != 0 {
                    self.push_top(Action::DrawCard { amount: count, follow_up: None, shuffle_check: false });
                    self.push_top(Action::Discard { amount: count, random: true, end_turn: false });
                }
            }
            Action::MultiplyPoison { target, extra } => {
                if let Some(n) = self.monsters[target].powers.iter().find(|p| p.id == PowerId::Poison).map(|p| p.amount) {
                    let a = self.apply_power_action(Cr::Monster(target), Some(Cr::Player), Power::new(PowerId::Poison, n * extra), n * extra);
                    self.push_top(a);
                }
            }
            Action::Doppelganger { upgraded, free, energy_on_use } => {
                let effect = self.x_effect(energy_on_use) + upgraded as i32;
                if effect > 0 {
                    self.bot_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::Energized, effect));
                    self.bot_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::DrawCardNextTurn, effect));
                    if !free {
                        self.energy = 0;
                    }
                }
            }
            Action::EscapePlan { block } => {
                if self.drawn_cards.iter().any(|&(_, t)| t == CardType::Skill) {
                    self.push_top(Action::GainBlock { target: Cr::Player, amount: block });
                }
            }
            Action::Expertise { amount } => {
                let to_draw = amount - self.hand.len() as i32;
                if to_draw > 0 {
                    self.push_top(Action::DrawCard { amount: to_draw, follow_up: None, shuffle_check: false });
                }
            }
            Action::DamagePerAttackPlayed { target, info } => {
                if self.monsters[target].hp > 0 {
                    // The Finisher itself is one of the attacks played.
                    for _ in 0..self.attacks_played_this_turn - 1 {
                        self.push_top(Action::Damage { target: Cr::Monster(target), source: DamageSource::Fixed(info), steal_gold: 0 });
                    }
                }
            }
            Action::Flechette { target, info } => {
                let skills = self.hand.iter().filter(|c| c.card_type() == CardType::Skill).count();
                for _ in 0..skills {
                    self.push_top(Action::Damage { target: Cr::Monster(target), source: DamageSource::Fixed(info), steal_gold: 0 });
                }
            }
            Action::HeelHook { target, info } => {
                if self.monsters[target].has_power(PowerId::Weakened) {
                    self.push_top(Action::DrawCard { amount: 1, follow_up: None, shuffle_check: false });
                    self.push_top(Action::GainEnergy { amount: 1 });
                }
                self.push_top(Action::Damage { target: Cr::Monster(target), source: DamageSource::Fixed(info), steal_gold: 0 });
            }
            Action::Malaise { target, upgraded, free, energy_on_use } => {
                let effect = self.x_effect(energy_on_use) + upgraded as i32;
                if effect > 0 {
                    self.bot_power(Cr::Monster(target), Some(Cr::Player), Power::new(PowerId::Strength, -effect));
                    self.bot_power(Cr::Monster(target), Some(Cr::Player), Power::weak(effect, false));
                    if !free {
                        self.energy = 0;
                    }
                }
            }
            Action::Nightmare { amount } => match self.hand.len() {
                0 => {}
                1 => {
                    let card = self.hand[0].clone();
                    let a = self.night_terror(&card, amount);
                    self.push_top(a);
                }
                _ => self.open_choice(ChoiceKind::Nightmare { amount }, ChoicePile::Hand, 1, false, false),
            },
            Action::Setup => match self.hand.len() {
                0 => {}
                1 => {
                    let c = self.hand.pop().unwrap();
                    self.setup_card(c);
                }
                _ => self.open_choice(ChoiceKind::Setup, ChoicePile::Hand, 1, false, false),
            },
            Action::Skewer { target, damage, free, energy_on_use } => {
                let effect = self.x_effect(energy_on_use);
                if effect > 0 {
                    for _ in 0..effect {
                        let info = DamageInfo::new(Some(Cr::Player), damage, DamageType::Normal);
                        self.push_bot(Action::Damage { target: Cr::Monster(target), source: DamageSource::Fixed(info), steal_gold: 0 });
                    }
                    if !free {
                        self.energy = 0;
                    }
                }
            }
            Action::GainEnergyIfDiscard { amount } => {
                if self.total_discarded_this_turn > 0 {
                    self.energy = (self.energy + amount).min(999);
                }
            }
            Action::BladeFury { upgraded } => {
                let n = self.hand.len() as i32;
                let mut shiv = self.new_card(CardId::Shiv);
                if upgraded {
                    shiv.upgrade();
                }
                self.push_top(Action::MakeTempCardInHand { card: shiv, amount: n });
                self.push_top(Action::Discard { amount: n, random: false, end_turn: false });
            }
            Action::Unload => {
                let ids: Vec<u32> = self.hand.iter().filter(|c| c.card_type() != CardType::Attack).map(|c| c.instance).collect();
                for instance in ids {
                    self.push_top(Action::DiscardSpecific { instance });
                }
            }
            Action::PoisonRandomEnemy { amount } => {
                if let Some(t) = self.get_random_monster(None, true, true) {
                    let a = self.apply_power_action(Cr::Monster(t), Some(Cr::Player), Power::new(PowerId::Poison, amount), amount);
                    self.push_top(a);
                }
            }
            Action::Channel { orb, auto_evoke } => {
                if auto_evoke || self.has_empty_orb() {
                    self.channel_orb(orb);
                }
            }
            Action::EvokeOrb { count } => {
                for _ in 0..count {
                    self.evoke_orb();
                }
            }
            Action::EvokeAllOrbs => {
                for _ in 0..self.player.orbs.len() {
                    self.push_top(Action::EvokeOrb { count: 1 });
                }
            }
            Action::EvokeWithoutRemovingOrb { count } => {
                for _ in 0..count {
                    self.evoke_without_losing_orb();
                }
            }
            Action::RemoveAllOrbs => {
                // Orbs fill the slots from the front, so this empties every slot.
                while self.has_orb() {
                    self.remove_next_orb();
                }
            }
            Action::IncreaseMaxOrb { amount } => {
                for _ in 0..amount {
                    self.increase_max_orb_slots(1);
                }
            }
            Action::DecreaseMaxOrb { amount } => {
                for _ in 0..amount {
                    self.decrease_max_orb_slots();
                }
            }
            Action::TriggerEndOfTurnOrbs => self.trigger_end_of_turn_orbs(),
            Action::LightningOrbDamage { amount, hit_all } => self.lightning_orb_damage(amount, hit_all),
            Action::DarkOrbEvoke { target, amount } => {
                let Some(t) = target else { return };
                let out = self.lock_on(t, amount);
                self.damage(Cr::Monster(t), DamageInfo::new(Some(Cr::Player), out, DamageType::Thorns));
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::Impulse { dark_only } => self.impulse(dark_only),
            Action::EssenceOfDarkness { amount } => {
                for _ in 0..self.player.orbs.len() {
                    for _ in 0..amount {
                        self.channel_orb(Orb::new(OrbId::Dark));
                    }
                }
            }
            Action::AggregateEnergy { divide } => self.energy = (self.energy + self.draw_pile.len() as i32 / divide).min(999),
            Action::AllCostToHand => {
                let ids: Vec<u32> = self.discard_pile.iter().filter(|c| c.cost == 0 || c.free_to_play_once).map(|c| c.instance).collect();
                for instance in ids {
                    self.push_bot(Action::DiscardToHandCard { instance });
                }
            }
            Action::DiscardToHandCard { instance } => {
                if self.hand.len() < 10 {
                    if let Some(pos) = self.discard_pile.iter().position(|c| c.instance == instance) {
                        let c = self.discard_pile.remove(pos);
                        self.hand.push(c);
                    }
                }
            }
            Action::Barrage { target, info } => {
                for _ in 0..self.filled_orb_count() {
                    self.push_top(Action::Damage { target: Cr::Monster(target), source: DamageSource::Fixed(info), steal_gold: 0 });
                }
            }
            Action::CompileDriver { amount } => {
                let mut kinds: Vec<OrbId> = Vec::new();
                for o in self.player.orbs.iter().flatten() {
                    if !kinds.contains(&o.id) {
                        kinds.push(o.id);
                    }
                }
                let to_draw = kinds.len() as i32 * amount;
                if to_draw > 0 {
                    self.push_top(Action::DrawCard { amount: to_draw, follow_up: None, shuffle_check: false });
                }
            }
            Action::DoubleEnergy => self.energy = (self.energy * 2).min(999),
            Action::Ftl { target, info, count } => {
                self.push_bot(Action::Damage { target: Cr::Monster(target), source: DamageSource::Fixed(info), steal_gold: 0 });
                if self.player.cards_played_this_turn - 1 < count {
                    self.push_top(Action::DrawCard { amount: 1, follow_up: None, shuffle_check: false });
                }
            }
            Action::Fission { upgraded } => {
                let n = self.filled_orb_count() as i32;
                self.push_top(Action::DrawCard { amount: n, follow_up: None, shuffle_check: false });
                self.push_top(Action::GainEnergy { amount: n });
                self.push_top(if upgraded { Action::EvokeAllOrbs } else { Action::RemoveAllOrbs });
            }
            Action::ForTheEyes { target, amount } => {
                if self.monsters[target].intent_base_dmg >= 0 {
                    self.top_power(Cr::Monster(target), Some(Cr::Player), Power::weak(amount, false));
                }
            }
            Action::Gash { amount } => {
                // The Claw in play (its UseCardAction) and every Claw in the hand and piles.
                for a in self.actions.iter_mut() {
                    if let Action::UseCard { card, .. } = a {
                        if card.id == CardId::Claw {
                            card.base_damage += amount;
                        }
                    }
                }
                for c in self.discard_pile.iter_mut().chain(self.draw_pile.iter_mut()).chain(self.hand.iter_mut()) {
                    if c.id == CardId::Claw {
                        c.base_damage += amount;
                    }
                }
            }
            Action::IncreaseMisc { uuid, amount } => {
                self.master_deck_growth.push((uuid, amount));
                self.for_each_battle_instance(uuid, |c| {
                    c.misc += amount;
                    c.base_block = c.misc;
                });
            }
            Action::Multicast { upgraded, free, energy_on_use } => {
                if self.has_orb() {
                    let effect = self.x_effect(energy_on_use) + upgraded as i32;
                    if effect > 0 {
                        for _ in 0..effect - 1 {
                            self.push_bot(Action::EvokeWithoutRemovingOrb { count: 1 });
                        }
                        self.push_bot(Action::EvokeOrb { count: 1 });
                        if !free {
                            self.energy = 0;
                        }
                    }
                }
            }
            Action::Recycle => match self.hand.len() {
                0 => {}
                1 => {
                    let c = self.hand.remove(0);
                    self.recycle(c);
                }
                _ => self.open_choice(ChoiceKind::Recycle, ChoicePile::Hand, 1, false, false),
            },
            Action::Redo => {
                if let Some(Some(o)) = self.player.orbs.first().copied() {
                    self.push_top(Action::Channel { orb: o, auto_evoke: false });
                    self.push_top(Action::EvokeOrb { count: 1 });
                }
            }
            Action::ReinforcedBody { block, free, energy_on_use } => {
                let effect = self.x_effect(energy_on_use);
                if effect > 0 {
                    for _ in 0..effect {
                        self.push_bot(Action::GainBlock { target: Cr::Player, amount: block });
                    }
                    if !free {
                        self.energy = 0;
                    }
                }
            }
            Action::ScrapeFollowUp => {
                for (instance, _) in self.drawn_cards.clone() {
                    let Some(pos) = self.hand.iter().position(|c| c.instance == instance) else { continue };
                    if self.hand[pos].cost_for_turn != 0 && !self.hand[pos].free_to_play_once {
                        let c = self.hand.remove(pos);
                        self.discard_from_hand(c, true, false, true);
                    }
                }
            }
            Action::ShuffleAll => {
                // The hand goes on top of the draw pile at random once the discard pile is in.
                self.push_top(Action::PutOnDeck { amount: 99, random: true });
                let seed = self.rng.shuffle.random_long();
                java_shuffle(&mut self.discard_pile, &mut JavaRandom::new(seed));
                // Each card lands in the draw pile through a Soul, which resets it there
                // (`clearPowers`: its cost for the turn, as in every move to the draw pile).
                let cards: Vec<Card> = self.discard_pile.drain(..).map(|mut c| {
                    c.reset_attributes();
                    c
                }).collect();
                self.draw_pile.extend(cards);
            }
            Action::Sunder { target, info } => {
                self.damage(Cr::Monster(target), info);
                if self.monsters[target].is_dying || self.monsters[target].hp <= 0 {
                    self.push_bot(Action::GainEnergy { amount: 3 });
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::Tempest { upgraded, free, energy_on_use } => {
                let effect = self.x_effect(energy_on_use) + upgraded as i32;
                if effect > 0 {
                    for _ in 0..effect {
                        self.push_bot(Action::Channel { orb: Orb::new(OrbId::Lightning), auto_evoke: true });
                    }
                    if !free {
                        self.energy = 0;
                    }
                }
            }
            Action::DrawPileToHandChoice { amount } => {
                if self.draw_pile.is_empty() || amount <= 0 {
                    return;
                }
                if self.draw_pile.len() as i32 <= amount {
                    for c in std::mem::take(&mut self.draw_pile) {
                        if self.hand.len() == 10 {
                            self.move_to_discard_pile(c);
                        } else {
                            self.hand.push(c);
                        }
                    }
                    return;
                }
                // The grid: by name, then rarity (highest first), statuses last; stable sorts.
                let mut options: Vec<usize> = (0..self.draw_pile.len()).collect();
                options.sort_by(|&a, &b| self.draw_pile[a].name().cmp(&self.draw_pile[b].name()));
                options.sort_by(|&a, &b| (self.draw_pile[b].data().rarity as u8).cmp(&(self.draw_pile[a].data().rarity as u8)));
                options.sort_by_key(|&i| self.draw_pile[i].card_type() == CardType::Status);
                self.pending = Some(PendingChoice {
                    kind: ChoiceKind::DrawPileToHand,
                    pile: ChoicePile::Draw,
                    count: amount as usize,
                    any_number: false,
                    can_pick_zero: false,
                    options,
                    offer: Vec::new(),
                });
            }
            Action::RemoveAllBlock { target } => {
                if !self.is_dying(target) && self.block(target) > 0 {
                    self.set_block(target, 0);
                }
            }
            Action::ModifyBlock { uuid, amount } => self.for_each_battle_instance(uuid, |c| c.base_block += amount),
            Action::ReduceCost { uuid } => self.for_each_battle_instance(uuid, |c| c.modify_cost_for_combat(-1)),
            Action::RetainCards { amount } => {
                if !self.hand.is_empty() {
                    self.open_choice(ChoiceKind::Retain, ChoicePile::Hand, amount as usize, true, true);
                }
            }
            // The Watcher.
            Action::ChangeStance { stance } => self.change_stance(stance),
            Action::Scry { amount } => self.exec_scry(amount),
            Action::PressEndTurnButton => self.end_turn_early(),
            Action::SkipEnemiesTurn => self.skip_monster_turn = true,
            Action::ChooseOne { options } => {
                self.pending = Some(PendingChoice {
                    kind: ChoiceKind::ChooseOne,
                    pile: ChoicePile::Offer,
                    count: 1,
                    any_number: false,
                    can_pick_zero: false,
                    options: (0..options.len()).collect(),
                    offer: options,
                });
            }
            Action::ForeignInfluence { upgraded } => self.exec_foreign_influence(upgraded),
            Action::Meditate { amount } => {
                if self.discard_pile.is_empty() || amount <= 0 {
                    return;
                }
                if self.discard_pile.len() as i32 <= amount {
                    // Every card is marked retained; those that don't fit in the hand stay put.
                    let mut rest = Vec::new();
                    for mut c in std::mem::take(&mut self.discard_pile) {
                        c.retain = true;
                        if self.hand.len() < 10 {
                            self.hand.push(c);
                        } else {
                            rest.push(c);
                        }
                    }
                    self.discard_pile = rest;
                    return;
                }
                self.open_choice(ChoiceKind::Meditate, ChoicePile::Discard, amount as usize, false, false);
            }
            Action::Omniscience { amount } => {
                if self.draw_pile.is_empty() {
                    return;
                }
                let options = self.draw_pile_grid_order();
                self.pending = Some(PendingChoice {
                    kind: ChoiceKind::Omniscience { amount },
                    pile: ChoicePile::Draw,
                    count: 1,
                    any_number: false,
                    can_pick_zero: false,
                    options,
                    offer: Vec::new(),
                });
            }
            Action::Collect { upgraded, free, energy_on_use } => {
                let effect = self.x_effect(energy_on_use) + upgraded as i32;
                if effect > 0 {
                    self.bot_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::Collect, effect));
                    if !free {
                        self.energy = 0;
                    }
                }
            }
            Action::ConjureBlade { free, energy_on_use } => {
                let effect = self.x_effect(energy_on_use);
                // Expunger.setX: its hits.
                let mut card = self.new_card(CardId::Expunger);
                card.magic = effect;
                card.base_magic = effect;
                self.push_bot(Action::MakeTempCardInDrawPile { card, amount: 1, random_spot: true, to_bottom: false });
                if !free {
                    self.energy = 0;
                }
            }
            Action::PreviousCardCheck { card_type, then } => {
                let n = self.played_types.len();
                if n >= 2 && self.played_types[n - 2] == card_type {
                    self.push_top(*then);
                }
            }
            Action::FearNoEvil { target, info } => {
                if self.monsters[target].intent.is_attack() {
                    self.push_top(Action::ChangeStance { stance: super::Stance::Calm });
                }
                self.push_top(Action::Damage { target: Cr::Monster(target), source: DamageSource::Fixed(info), steal_gold: 0 });
            }
            Action::Halt { block, extra } => {
                let amount = if self.player.stance == super::Stance::Wrath { block + extra } else { block };
                self.push_top(Action::GainBlock { target: Cr::Player, amount });
            }
            Action::Indignation { amount } => {
                if self.player.stance == super::Stance::Wrath {
                    for m in self.order.clone() {
                        self.bot_power(Cr::Monster(m), Some(Cr::Player), Power::vulnerable(amount, false, self.turn_has_ended));
                    }
                } else {
                    self.push_bot(Action::ChangeStance { stance: super::Stance::Wrath });
                }
            }
            Action::InnerPeace { amount } => {
                if self.player.stance == super::Stance::Calm {
                    self.push_top(Action::DrawCard { amount, follow_up: None, shuffle_check: false });
                } else {
                    self.push_top(Action::ChangeStance { stance: super::Stance::Calm });
                }
            }
            Action::Judgement { target, cutoff } => {
                if self.monsters[target].hp <= cutoff {
                    // InstantKillAction: no HP left, then a 0 HP_LOSS hit to run the death.
                    self.monsters[target].hp = 0;
                    self.damage(Cr::Monster(target), DamageInfo::new(None, 0, DamageType::HpLoss));
                }
            }
            Action::LessonLearned { target, info } => {
                self.damage(Cr::Monster(target), info);
                let m = &self.monsters[target];
                if (m.is_dying || m.hp <= 0) && !m.half_dead && !m.has_power(PowerId::Minion) {
                    let cands: Vec<usize> = (0..self.master_deck.len()).filter(|&i| self.master_deck[i].can_upgrade()).collect();
                    if !cands.is_empty() {
                        let k = self.rng.misc.random_range(0, cands.len() as i32 - 1) as usize;
                        let c = &mut self.master_deck[cands[k]];
                        c.upgrade();
                        self.master_deck_upgrades.push(c.uuid);
                    }
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::Wallop { target, info } => {
                let source_dying = info.owner.map_or(false, |o| self.is_dying(o));
                if source_dying || self.monsters[target].is_dead_or_escaped() {
                    return;
                }
                self.damage(Cr::Monster(target), info);
                let taken = self.monsters[target].last_damage_taken;
                if taken > 0 {
                    self.push_top(Action::GainBlock { target: Cr::Player, amount: taken });
                }
                if self.monsters_basically_dead() {
                    self.clear_post_combat_actions();
                }
            }
            Action::TriggerMarks => {
                for m in self.order.clone() {
                    for (id, amount) in self.monsters[m].powers.iter().map(|p| (p.id, p.amount)).collect::<Vec<_>>() {
                        if id == PowerId::Mark {
                            self.push_bot(Action::LoseHp { target: Cr::Monster(m), source: None, amount });
                        }
                    }
                }
            }
            Action::Establishment { amount } => {
                for c in self.hand.iter_mut() {
                    if c.self_retain || c.retain {
                        c.modify_cost_for_combat(-amount);
                    }
                }
            }
            Action::GainGold { amount } => self.gain_gold(amount),
            Action::ReduceCostCard { instance } => {
                for c in self
                    .hand
                    .iter_mut()
                    .chain(self.draw_pile.iter_mut())
                    .chain(self.discard_pile.iter_mut())
                    .chain(self.exhaust_pile.iter_mut())
                    .chain(self.limbo.iter_mut())
                {
                    if c.instance == instance {
                        c.modify_cost_for_combat(-1);
                    }
                }
            }
        }
    }

    /// An option card's `onChoseThisOption` (Wish, Stance Potion).
    fn choose_option(&mut self, c: &Card) {
        let p = Some(Cr::Player);
        match c.id {
            CardId::BecomeAlmighty => self.bot_power(Cr::Player, p, Power::new(PowerId::Strength, c.magic)),
            CardId::FameAndFortune => self.push_bot(Action::GainGold { amount: c.magic }),
            CardId::LiveForever => self.bot_power(Cr::Player, p, Power::new(PowerId::PlatedArmor, c.magic)),
            CardId::ChooseCalm => self.push_bot(Action::ChangeStance { stance: super::Stance::Calm }),
            CardId::ChooseWrath => self.push_bot(Action::ChangeStance { stance: super::Stance::Wrath }),
            _ => {}
        }
    }

    /// `AbstractPlayer.gainGold` in a fight: Ectoplasm blocks it, Bloody Idol heals.
    pub(crate) fn gain_gold(&mut self, amount: i32) {
        if !self.has_relic(crate::relics::RelicId::Ectoplasm) {
            self.player.gold += amount;
            if self.has_relic(crate::relics::RelicId::BloodyIdol) {
                self.heal(Cr::Player, 5);
            }
        }
    }

    /// A card made during the fight comes upgraded under Master Reality (not curses or statuses).
    pub(crate) fn master_reality(&self, c: &mut Card) {
        if self.player.has_power(PowerId::MasterReality) && !matches!(c.card_type(), CardType::Curse | CardType::Status) {
            c.upgrade();
        }
    }

    /// `new ScryAction(n)`: Golden Eye adds 2 when the action is made.
    pub(crate) fn scry(&self, amount: i32) -> Action {
        let extra = if self.has_relic(crate::relics::RelicId::GoldenEye) { 2 } else { 0 };
        Action::Scry { amount: amount + extra }
    }

    /// `new MantraPower(p, n)`: the constructor counts it in `mantraGained`.
    pub(crate) fn mantra(&mut self, amount: i32) -> Power {
        self.mantra_gained += amount;
        Power::new(PowerId::Mantra, amount)
    }

    /// The draw pile as a grid shows it sorted (Seek, Omniscience): by name, then rarity (highest
    /// first), statuses last; stable sorts.
    fn draw_pile_grid_order(&self) -> Vec<usize> {
        let mut options: Vec<usize> = (0..self.draw_pile.len()).collect();
        options.sort_by(|&a, &b| self.draw_pile[a].name().cmp(&self.draw_pile[b].name()));
        options.sort_by(|&a, &b| (self.draw_pile[b].data().rarity as u8).cmp(&(self.draw_pile[a].data().rarity as u8)));
        options.sort_by_key(|&i| self.draw_pile[i].card_type() == CardType::Status);
        options
    }

    /// `ScryAction.update`: the powers' onScry (Nirvana), then a grid of the top cards; the picks
    /// are discarded, and every Weave in the discard pile returns to the hand.
    fn exec_scry(&mut self, amount: i32) {
        if self.monsters_basically_dead() {
            return;
        }
        for (id, n) in self.player.powers.iter().map(|p| (p.id, p.amount)).collect::<Vec<_>>() {
            if id == PowerId::Nirvana {
                self.push_bot(Action::GainBlock { target: Cr::Player, amount: n });
            }
        }
        if self.draw_pile.is_empty() {
            return;
        }
        // The grid lists the top card first.
        let len = self.draw_pile.len();
        let options: Vec<usize> = (0..(amount.max(0) as usize).min(len)).map(|i| len - 1 - i).collect();
        let count = options.len();
        self.pending = Some(PendingChoice {
            kind: ChoiceKind::Scry,
            pile: ChoicePile::Draw,
            count,
            any_number: true,
            can_pick_zero: true,
            options,
            offer: Vec::new(),
        });
    }

    /// The discard pile's `triggerOnScry` after a scry: each Weave there comes back.
    fn trigger_on_scry(&mut self) {
        let weaves: Vec<u32> = self.discard_pile.iter().filter(|c| c.id == CardId::Weave).map(|c| c.instance).collect();
        for instance in weaves {
            self.push_bot(Action::DiscardToHandCard { instance });
        }
    }

    /// `ForeignInfluenceAction`: three different attacks of any color, each of a rolled rarity
    /// (`getAnyColorCard`: cardRandomRng shuffles the list, which is then sorted by id, and
    /// cardRng picks).
    fn exec_foreign_influence(&mut self, upgraded: bool) {
        let mut offer: Vec<Card> = Vec::new();
        while offer.len() != 3 {
            let roll = self.rng.card_random.random(99);
            let rarity = if roll < 55 {
                crate::cards::CardRarity::Common
            } else if roll < 85 {
                crate::cards::CardRarity::Uncommon
            } else {
                crate::cards::CardRarity::Rare
            };
            let mut pool: Vec<CardId> = crate::cards::ALL_CARDS
                .iter()
                .copied()
                .filter(|id| {
                    let d = id.data();
                    d.rarity == rarity && !d.healing && d.card_type == CardType::Attack
                })
                .collect();
            let _ = self.rng.card_random.random_long();
            pool.sort_by(|a, b| a.data().java_id.cmp(b.data().java_id));
            let id = pool[self.rng.card.random(pool.len() as i32 - 1) as usize];
            if !offer.iter().any(|c| c.id == id) {
                let c = self.new_card(id);
                offer.push(c);
            }
        }
        self.pending = Some(PendingChoice {
            kind: ChoiceKind::ForeignInfluence { upgraded },
            pile: ChoicePile::Offer,
            count: 1,
            any_number: false,
            can_pick_zero: true,
            options: (0..offer.len()).collect(),
            offer,
        });
    }

    /// `GetAllInBattleInstances`: the card in use and every copy in the hand and piles.
    fn for_each_battle_instance(&mut self, uuid: u32, mut f: impl FnMut(&mut Card)) {
        for a in self.actions.iter_mut() {
            if let Action::UseCard { card, .. } = a {
                if card.uuid == uuid {
                    f(card);
                }
            }
        }
        for c in self
            .draw_pile
            .iter_mut()
            .chain(self.discard_pile.iter_mut())
            .chain(self.exhaust_pile.iter_mut())
            .chain(self.limbo.iter_mut())
            .chain(self.hand.iter_mut())
        {
            if c.uuid == uuid {
                f(c);
            }
        }
    }

    /// `RecycleAction` for the chosen card: energy for what it costs this turn (an X card gives
    /// the current energy), then it is exhausted.
    fn recycle(&mut self, c: Card) {
        if c.cost_for_turn == -1 {
            self.push_top(Action::GainEnergy { amount: self.energy });
        } else if c.cost_for_turn > 0 {
            self.push_top(Action::GainEnergy { amount: c.cost_for_turn });
        }
        self.move_to_exhaust_pile(c);
    }

    /// An X-cost card's effect: the energy it was played with, plus 2 with Chemical X.
    fn x_effect(&self, energy_on_use: i32) -> i32 {
        let effect = if energy_on_use != -1 { energy_on_use } else { self.energy };
        effect + if self.has_relic(crate::relics::RelicId::ChemicalX) { 2 } else { 0 }
    }

    /// `new NightmarePower(p, amount, card)` in an `ApplyPowerAction`: a copy of the card, reset,
    /// kept for the power to make next turn.
    fn night_terror(&mut self, card: &Card, amount: i32) -> Action {
        let mut copy = self.stat_copy(card);
        copy.reset_attributes();
        self.bomb_counter += 1;
        let uid = self.bomb_counter;
        self.night_terror_cards.push((uid, copy));
        let mut p = Power::new(PowerId::NightTerror, amount);
        p.uid = uid;
        self.apply_power_action(Cr::Player, Some(Cr::Player), p, amount)
    }

    /// Setup's `hand.moveToDeck(c, false)`, free once if it costs anything.
    fn setup_card(&mut self, mut c: Card) {
        if c.cost > 0 {
            c.free_to_play_once = true;
        }
        self.card_queue.retain(|q| q.card.instance() != Some(c.instance));
        c.reset_attributes();
        self.draw_pile.push(c);
    }

    /// A hand card discarded by an action: `moveToDiscardPile`, then `triggerOnManualDiscard`
    /// (unless `trigger` is false: the end-of-turn discard) and `incrementDiscard`, in that order,
    /// or the count first (`trigger_first` false: Gambling Chip, `DiscardSpecificCardAction`).
    pub(crate) fn discard_from_hand(&mut self, c: Card, trigger: bool, end_turn: bool, trigger_first: bool) {
        let (id, magic) = (c.id, c.magic);
        self.move_to_discard_pile(c);
        if trigger && trigger_first {
            self.trigger_on_manual_discard(id, magic);
        }
        self.increment_discard(end_turn);
        if trigger && !trigger_first {
            self.trigger_on_manual_discard(id, magic);
        }
    }

    /// `AbstractCard.triggerOnManualDiscard`.
    fn trigger_on_manual_discard(&mut self, id: CardId, magic: i32) {
        match id {
            CardId::Reflex => self.push_bot(Action::DrawCard { amount: magic, follow_up: None, shuffle_check: false }),
            CardId::Tactician => self.push_top(Action::GainEnergy { amount: magic }),
            _ => {}
        }
    }

    /// `GameActionManager.incrementDiscard`.
    fn increment_discard(&mut self, end_turn: bool) {
        self.total_discarded_this_turn += 1;
        if !self.turn_has_ended && !end_turn {
            // updateCardsOnDiscard: didDiscard.
            for c in self.hand.iter_mut().chain(self.discard_pile.iter_mut()).chain(self.draw_pile.iter_mut()) {
                if c.id == CardId::Eviscerate {
                    c.set_cost_for_turn(c.cost_for_turn - 1);
                }
            }
            self.relics_on_manual_discard();
        }
    }

    /// `AddCardToDeckAction` (Writhing Mass's Implant): a `ShowCardAndObtainEffect` during the
    /// fight. Omamori may cancel a curse at once; otherwise the relics' `onObtainCard` reactions
    /// that matter in a fight (Darkstone Periapt's +6 max HP, Ceramic Fish's gold) happen now. In
    /// the game they land when the effect ends, moments later, while the fight goes on. The card
    /// itself joins the master deck when the fight ends (`Run::add_obtained_card`), where the eggs
    /// upgrade it.
    fn obtain_card_in_combat(&mut self, card: CardId) {
        use crate::cards::CardColor;
        use crate::relics::RelicId;
        let curse = card.data().color == CardColor::Curse;
        if curse {
            if let Some(o) = self.player.relics.iter_mut().find(|r| r.id == RelicId::Omamori && r.counter != 0) {
                o.counter -= 1;
                if o.counter == 0 {
                    o.used_up = true;
                }
                return;
            }
        }
        let ids: Vec<RelicId> = self.player.relics.iter().map(|r| r.id).collect();
        for id in ids {
            match id {
                // gainGold: Ectoplasm blocks it, Bloody Idol heals.
                RelicId::CeramicFish if !self.has_relic(RelicId::Ectoplasm) => {
                    self.player.gold += 9;
                    if self.has_relic(RelicId::BloodyIdol) {
                        self.heal(Cr::Player, 5);
                    }
                }
                RelicId::DarkstonePeriapt if curse => self.increase_max_hp(6),
                _ => {}
            }
        }
        self.master_deck_add.push(card);
    }

    /// `DamageAction.update`.
    fn exec_damage(&mut self, target: Cr, source: DamageSource, steal_gold: i32) {
        let info = match source {
            DamageSource::Fixed(info) => info,
            DamageSource::MonsterSlot { monster, slot } => {
                DamageInfo::new(Some(Cr::Monster(monster)), self.monsters[monster].damage[slot].output, DamageType::Normal)
            }
        };
        let source_dying = info.owner.map_or(false, |o| self.is_dying(o));
        let cancel = source_dying || self.is_dead_or_escaped(target);
        if cancel && info.damage_type != DamageType::Thorns {
            return;
        }
        if info.damage_type != DamageType::Thorns {
            if let Some(Cr::Monster(o)) = info.owner {
                if self.monsters[o].is_dying || self.monsters[o].half_dead {
                    return;
                }
            }
        }
        if steal_gold != 0 && target == Cr::Player {
            let stolen = steal_gold.min(self.player.gold);
            self.player.gold -= stolen;
            self.stolen_gold += stolen;
        }
        self.damage(target, info);
        if self.monsters_basically_dead() {
            self.clear_post_combat_actions();
        }
    }

    /// `GameActionManager.clearPostCombatActions`.
    pub(crate) fn clear_post_combat_actions(&mut self) {
        self.actions.retain(|a| a.survives_combat_end());
    }

    fn add_to_random_spot(&mut self, c: Card) {
        if self.draw_pile.is_empty() {
            self.draw_pile.push(c);
        } else {
            let k = self.rng.card_random.random(self.draw_pile.len() as i32 - 1);
            self.draw_pile.insert(k as usize, c);
        }
    }

    /// `DrawCardAction.update`.
    fn exec_draw(&mut self, mut amount: i32, follow_up: Option<Box<Action>>, shuffle_check: bool, clear_history: bool) {
        if clear_history {
            self.drawn_cards.clear();
        }
        let finish = |c: &mut Combat, follow_up: Option<Box<Action>>| {
            if let Some(f) = follow_up {
                c.push_top(*f);
            }
        };
        if self.player.has_power(PowerId::NoDraw) || amount <= 0 {
            return finish(self, follow_up);
        }
        let deck = self.draw_pile.len() as i32;
        let discard = self.discard_pile.len() as i32;
        if deck + discard == 0 || self.hand.len() == 10 {
            return finish(self, follow_up);
        }
        if !shuffle_check {
            let hand = self.hand.len() as i32;
            if amount + hand > 10 {
                amount += 10 - (amount + hand);
            }
            if amount > deck {
                let tmp = amount - deck;
                self.push_top(Action::DrawCardKeepHistory { amount: tmp, follow_up });
                self.relics_on_shuffle();
                self.push_top(Action::EmptyDeckShuffle);
                if deck != 0 {
                    self.push_top(Action::DrawCardKeepHistory { amount: deck, follow_up: None });
                }
                return;
            }
        }
        for _ in 0..amount {
            if self.draw_pile.is_empty() {
                break;
            }
            self.draw_one();
        }
        finish(self, follow_up);
    }

    /// `CardGroup.moveToExhaustPile` (the card has already left its pile). `ExhaustCardEffect`
    /// resets the card's attributes when its animation ends, before the next decision.
    pub(crate) fn move_to_exhaust_pile(&mut self, mut card: Card) {
        card.reset_attributes();
        self.card_queue.retain(|q| q.card.instance() != Some(card.instance));
        self.relics_on_exhaust();
        self.on_exhaust();
        if card.id == CardId::Sentinel {
            self.push_top(Action::GainEnergy { amount: if card.upgraded { 3 } else { 2 } });
        }
        if card.id == CardId::Necronomicurse {
            let copy = self.new_card(CardId::Necronomicurse);
            self.push_bot(Action::MakeTempCardInHand { card: copy, amount: 1 });
        }
        self.exhaust_pile.push(card);
        self.on_card_draw_or_discard();
    }

    /// `CardGroup.moveToDiscardPile`.
    pub(crate) fn move_to_discard_pile(&mut self, mut card: Card) {
        self.card_queue.retain(|q| q.card.instance() != Some(card.instance));
        // The card's Soul calls clearPowers() (resetAttributes) when it lands on the pile.
        card.reset_attributes();
        self.discard_pile.push(card);
        self.on_card_draw_or_discard();
    }

    /// `ExhaustAction.update`.
    fn exec_exhaust(&mut self, amount: i32, random: bool, any_number: bool, can_pick_zero: bool) {
        if self.hand.is_empty() {
            return;
        }
        if !any_number && self.hand.len() as i32 <= amount {
            while let Some(c) = self.hand.pop() {
                self.move_to_exhaust_pile(c);
            }
            return;
        }
        if !random {
            self.open_choice(ChoiceKind::Exhaust, ChoicePile::Hand, amount as usize, any_number, can_pick_zero);
            return;
        }
        for _ in 0..amount {
            if self.hand.is_empty() {
                break;
            }
            let k = self.rng.card_random.random(self.hand.len() as i32 - 1) as usize;
            let c = self.hand.remove(k);
            self.move_to_exhaust_pile(c);
        }
    }

    /// `DiscardAction.update`.
    fn exec_discard(&mut self, mut amount: i32, random: bool, end_turn: bool) {
        if self.monsters_basically_dead() {
            return;
        }
        if self.hand.len() as i32 <= amount {
            amount = self.hand.len() as i32;
            for _ in 0..amount {
                let c = self.hand.pop().unwrap();
                self.discard_from_hand(c, !end_turn, end_turn, true);
            }
            return;
        }
        if !random {
            if amount < 0 {
                self.open_choice(ChoiceKind::Discard { end_turn }, ChoicePile::Hand, 99, true, true);
            } else {
                self.open_choice(ChoiceKind::Discard { end_turn }, ChoicePile::Hand, amount as usize, false, false);
            }
            return;
        }
        for _ in 0..amount {
            let k = self.rng.card_random.random(self.hand.len() as i32 - 1) as usize;
            let c = self.hand.remove(k);
            self.discard_from_hand(c, true, end_turn, true);
        }
    }

    /// `DiscardAtEndOfTurnAction.update`.
    fn exec_discard_at_end_of_turn(&mut self) {
        let mut kept = Vec::new();
        for c in std::mem::take(&mut self.hand) {
            if c.retain || c.self_retain {
                self.limbo.push(c);
            } else {
                kept.push(c);
            }
        }
        self.hand = kept;
        self.push_top(Action::RestoreRetainedCards);
        if !self.has_relic(crate::relics::RelicId::RunicPyramid) {
            let n = self.hand.len() as i32;
            for _ in 0..n {
                self.push_top(Action::Discard { amount: self.hand.len() as i32, random: true, end_turn: true });
            }
        }
        // triggerOnEndOfPlayerTurn in an (unseeded) shuffled order.
        let mut order: Vec<(u32, CardId, bool)> = self.hand.iter().map(|c| (c.instance, c.id, c.ethereal)).collect();
        self.unseeded_shuffle(&mut order);
        for (inst, id, ethereal) in order {
            match id {
                CardId::Entrench | CardId::GhostlyArmor => self.push_top(Action::ExhaustAllEthereal),
                _ => {
                    if ethereal {
                        self.push_top(Action::ExhaustSpecificInHand { instance: inst });
                    }
                }
            }
        }
    }

    /// `UseCardAction.update`.
    fn exec_use_card(&mut self, mut card: Card, _target: Option<usize>, exhaust_card: bool) {
        let mut rebound = false;
        if !card.dont_trigger_on_use {
            rebound = self.player_on_after_use_card(&card);
            self.on_after_use_card();
        }
        card.free_to_play_once = false;
        card.in_autoplay = false;
        card.applied_base = None;
        if card.purge_on_use {
            return;
        }
        if card.card_type() == CardType::Power {
            return;
        }
        let dont_trigger = card.dont_trigger_on_use;
        card.exhaust_on_use_once = false;
        card.dont_trigger_on_use = false;
        let _ = dont_trigger;
        let spoon = exhaust_card && self.has_relic(crate::relics::RelicId::StrangeSpoon) && self.rng.card_random.random_boolean();
        if exhaust_card && !spoon {
            self.move_to_exhaust_pile(card);
        } else if rebound {
            // moveToDeck(card, false): on top of the draw pile.
            self.card_queue.retain(|q| q.card.instance() != Some(card.instance));
            card.reset_attributes();
            self.draw_pile.push(card);
        } else if card.id == CardId::Tantrum {
            // shuffleBackIntoDrawPile: moveToDeck(card, true), at a random spot. The move resets the
            // card, as every move into the draw pile does: a Tantrum made free for the turn (Attack
            // Potion) costs 1 again when drawn back (run H0ZR of the 2026-10-28 Watcher soak).
            self.card_queue.retain(|q| q.card.instance() != Some(card.instance));
            card.reset_attributes();
            self.add_to_random_spot(card);
        } else {
            self.move_to_discard_pile(card);
        }
    }

    /// `PutOnDeckAction.update` (Warcry).
    fn exec_put_on_deck(&mut self, mut amount: i32, random: bool) {
        if (self.hand.len() as i32) < amount {
            amount = self.hand.len() as i32;
        }
        if random {
            for _ in 0..amount {
                let k = self.rng.card_random.random(self.hand.len() as i32 - 1) as usize;
                let mut c = self.hand.remove(k);
                c.reset_attributes();
                self.draw_pile.push(c);
            }
            return;
        }
        if self.hand.len() as i32 > amount {
            self.open_choice(ChoiceKind::PutOnDeck, ChoicePile::Hand, amount as usize, false, false);
            return;
        }
        // `for (i = 0; i < hand.size(); i++)` re-evaluates the shrinking size.
        let mut i = 0;
        while i < self.hand.len() {
            let k = self.rng.card_random.random(self.hand.len() as i32 - 1) as usize;
            let mut c = self.hand.remove(k);
            c.reset_attributes();
            self.draw_pile.push(c);
            i += 1;
        }
    }

    /// `ExhumeAction.update`.
    fn exec_exhume(&mut self, upgrade: bool) {
        if self.hand.len() == 10 || self.exhaust_pile.is_empty() {
            return;
        }
        if self.exhaust_pile.len() == 1 {
            if self.exhaust_pile[0].id == CardId::Exhume {
                return;
            }
            let mut c = self.exhaust_pile.pop().unwrap();
            if self.player.has_power(PowerId::Corruption) && c.card_type() == CardType::Skill {
                c.set_cost_for_turn(-9);
            }
            if upgrade && c.can_upgrade() {
                c.upgrade();
            }
            self.hand.push(c);
            return;
        }
        let mut exhumes = Vec::new();
        let mut rest = Vec::new();
        for c in std::mem::take(&mut self.exhaust_pile) {
            if c.id == CardId::Exhume {
                exhumes.push(c);
            } else {
                rest.push(c);
            }
        }
        self.exhaust_pile = rest;
        if self.exhaust_pile.is_empty() {
            self.exhaust_pile.extend(exhumes);
            return;
        }
        self.open_choice(ChoiceKind::Exhume { upgrade, exhumes }, ChoicePile::Exhaust, 1, false, false);
    }

    /// `DualWieldAction.update`.
    fn exec_dual_wield(&mut self, amount: i32) {
        let wieldable = |c: &Card| matches!(c.card_type(), CardType::Attack | CardType::Power);
        let n_ok = self.hand.iter().filter(|c| wieldable(c)).count();
        if n_ok == 0 {
            return;
        }
        if n_ok == 1 {
            let c = self.hand.iter().find(|c| wieldable(c)).cloned().unwrap();
            for _ in 0..amount {
                let copy = self.stat_copy(&c);
                self.push_top(Action::MakeTempCardInHand { card: copy, amount: 1 });
            }
            return;
        }
        let mut cannot = Vec::new();
        let mut ok = Vec::new();
        for c in std::mem::take(&mut self.hand) {
            if wieldable(&c) {
                ok.push(c);
            } else {
                cannot.push(c);
            }
        }
        self.hand = ok;
        self.open_choice(ChoiceKind::DualWield { amount, cannot_duplicate: cannot }, ChoicePile::Hand, 1, false, false);
    }

    /// `ArmamentsAction.update`.
    fn exec_armaments(&mut self, upgraded: bool) {
        if upgraded {
            for c in self.hand.iter_mut() {
                if c.can_upgrade() {
                    c.upgrade();
                }
            }
            return;
        }
        let n_ok = self.hand.iter().filter(|c| c.can_upgrade()).count();
        if n_ok == 0 {
            return;
        }
        if n_ok == 1 {
            if let Some(c) = self.hand.iter_mut().find(|c| c.can_upgrade()) {
                c.upgrade();
            }
            return;
        }
        let mut cannot = Vec::new();
        let mut ok = Vec::new();
        for c in std::mem::take(&mut self.hand) {
            if c.can_upgrade() {
                ok.push(c);
            } else {
                cannot.push(c);
            }
        }
        self.hand = ok;
        self.open_choice(ChoiceKind::Armaments { cannot_upgrade: cannot }, ChoicePile::Hand, 1, false, false);
    }

    /// `DiscardPileToTopOfDeckAction.update` (Headbutt).
    fn exec_discard_to_top(&mut self) {
        if self.monsters_basically_dead() || self.discard_pile.is_empty() {
            return;
        }
        if self.discard_pile.len() == 1 {
            let mut c = self.discard_pile.pop().unwrap();
            c.reset_attributes();
            self.draw_pile.push(c);
        }
        if !self.discard_pile.is_empty() {
            self.open_choice(ChoiceKind::DiscardToTopOfDeck, ChoicePile::Discard, 1, false, false);
        }
    }

    fn open_choice(&mut self, kind: ChoiceKind, pile: ChoicePile, count: usize, any_number: bool, can_pick_zero: bool) {
        let len = match pile {
            ChoicePile::Hand => self.hand.len(),
            ChoicePile::Discard => self.discard_pile.len(),
            ChoicePile::Exhaust => self.exhaust_pile.len(),
            ChoicePile::Offer => 0,
            ChoicePile::Draw => self.draw_pile.len(),
        };
        let count = count.min(len);
        self.pending = Some(PendingChoice { kind, pile, count, any_number, can_pick_zero, options: (0..len).collect(), offer: Vec::new() });
    }

    /// Finishes the action that opened the pending selection screen.
    pub(crate) fn resolve_choice(&mut self, indices: &[usize]) -> Result<(), String> {
        let p = self.pending.as_ref().ok_or("no selection pending")?;
        let mut idx: Vec<usize> = indices.to_vec();
        idx.sort_unstable();
        idx.dedup();
        if idx.len() != indices.len() {
            return Err("duplicate indices".into());
        }
        if idx.iter().any(|i| !p.options.contains(i)) {
            return Err("index not selectable".into());
        }
        let ok_count = if p.any_number {
            idx.len() <= p.count && (p.can_pick_zero || !idx.is_empty() || p.count == 0)
        } else {
            idx.len() == p.count
        };
        if !ok_count {
            return Err(format!("select exactly {} card(s)", p.count));
        }
        let p = self.pending.take().unwrap();
        // Selected cards leave their pile in selection order (the screen's group). Meditate's leave
        // only while the hand has room: the rest stay where they are in the discard pile
        // (MeditateAction; run H141 of the 2026-10-28 Watcher soak, where the engine put the one
        // that didn't fit back on top, and the next shuffle came out in another order).
        let leaving = if matches!(p.kind, ChoiceKind::Meditate) { indices.len().min(10usize.saturating_sub(self.hand.len())) } else { indices.len() };
        let mut offer = p.offer.clone();
        let selected: Vec<Card> = {
            let pile = match p.pile {
                ChoicePile::Hand => &mut self.hand,
                ChoicePile::Discard => &mut self.discard_pile,
                ChoicePile::Exhaust => &mut self.exhaust_pile,
                ChoicePile::Offer => &mut offer,
                ChoicePile::Draw => &mut self.draw_pile,
            };
            let picked: Vec<Card> = indices[..leaving].iter().map(|&i| pile[i].clone()).collect();
            let ids: Vec<u32> = picked.iter().map(|c| c.instance).collect();
            pile.retain(|c| !ids.contains(&c.instance));
            picked
        };
        match p.kind {
            ChoiceKind::Exhaust => {
                for c in selected {
                    self.move_to_exhaust_pile(c);
                }
            }
            ChoiceKind::Discard { end_turn } => {
                for c in selected {
                    self.discard_from_hand(c, true, end_turn, true);
                }
            }
            ChoiceKind::PutOnDeck => {
                for mut c in selected {
                    c.reset_attributes();
                    self.draw_pile.push(c);
                }
            }
            ChoiceKind::Armaments { cannot_upgrade } => {
                for mut c in selected {
                    c.upgrade();
                    self.hand.push(c);
                }
                self.hand.extend(cannot_upgrade);
            }
            ChoiceKind::DualWield { amount, cannot_duplicate } => {
                // Queued in this order with addToTop, so the last pushed resolves first.
                for c in &selected {
                    let first = self.stat_copy(c);
                    self.push_top(Action::MakeTempCardInHand { card: first, amount: 1 });
                    for _ in 0..amount {
                        let copy = self.stat_copy(c);
                        self.push_top(Action::MakeTempCardInHand { card: copy, amount: 1 });
                    }
                }
                self.hand.extend(cannot_duplicate);
            }
            ChoiceKind::DiscardToTopOfDeck => {
                for mut c in selected {
                    c.reset_attributes();
                    self.draw_pile.push(c);
                }
            }
            ChoiceKind::Exhume { upgrade, exhumes } => {
                for mut c in selected {
                    if self.player.has_power(PowerId::Corruption) && c.card_type() == CardType::Skill {
                        c.set_cost_for_turn(-9);
                    }
                    if upgrade && c.can_upgrade() {
                        c.upgrade();
                    }
                    self.hand.push(c);
                }
                self.exhaust_pile.extend(exhumes);
            }
            ChoiceKind::GamblingChip => {
                // GamblingChipAction: draw as many as were discarded, after the discards.
                if !selected.is_empty() {
                    self.push_top(Action::DrawCard { amount: selected.len() as i32, follow_up: None, shuffle_check: false });
                }
                for c in selected {
                    self.discard_from_hand(c, true, false, false);
                }
            }
            ChoiceKind::Discovery { amount } => {
                for c in &selected {
                    for _ in 0..amount.max(1) {
                        let mut copy = self.stat_copy(c);
                        self.master_reality(&mut copy);
                        copy.set_cost_for_turn(0);
                        if self.hand.len() < 10 {
                            self.hand.push(copy);
                        } else {
                            self.discard_pile.push(copy);
                        }
                    }
                }
            }
            ChoiceKind::Codex => {
                for c in &selected {
                    let copy = self.stat_copy(c);
                    self.add_to_random_spot(copy);
                }
            }
            ChoiceKind::DiscardToHand { set_cost } => {
                for mut c in selected {
                    if self.hand.len() < 10 {
                        if set_cost {
                            c.set_cost_for_turn(0);
                        }
                        self.hand.push(c);
                    } else {
                        self.discard_pile.push(c);
                    }
                }
            }
            ChoiceKind::FromDeckToHand => {
                for c in selected {
                    if self.hand.len() == 10 {
                        self.discard_pile.push(c);
                    } else {
                        self.hand.push(c);
                    }
                }
            }
            ChoiceKind::Forethought => {
                for mut c in selected {
                    if c.cost > 0 {
                        c.free_to_play_once = true;
                    }
                    c.reset_attributes();
                    self.draw_pile.insert(0, c);
                }
            }
            ChoiceKind::Discover => {
                for c in selected {
                    let mut copy = self.stat_copy(&c);
                    self.master_reality(&mut copy);
                    if self.hand.len() < 10 {
                        self.hand.push(copy);
                    } else {
                        self.discard_pile.push(copy);
                    }
                }
            }
            ChoiceKind::Nightmare { amount } => {
                for c in selected {
                    let a = self.night_terror(&c, amount);
                    self.push_top(a);
                    self.hand.push(c);
                }
            }
            ChoiceKind::Setup => {
                for c in selected {
                    self.setup_card(c);
                }
            }
            ChoiceKind::Recycle => {
                for c in selected {
                    self.recycle(c);
                }
            }
            ChoiceKind::DrawPileToHand => {
                for c in selected {
                    if self.hand.len() == 10 {
                        self.move_to_discard_pile(c);
                    } else {
                        self.hand.push(c);
                    }
                }
            }
            ChoiceKind::Retain => {
                for mut c in selected {
                    if !c.ethereal {
                        c.retain = true;
                    }
                    self.hand.push(c);
                }
            }
            ChoiceKind::Scry => {
                // The grid's picks go to the discard pile in the order picked.
                for c in selected {
                    self.move_to_discard_pile(c);
                }
                self.trigger_on_scry();
            }
            ChoiceKind::ChooseOne => {
                for c in selected {
                    self.choose_option(&c);
                }
            }
            ChoiceKind::ForeignInfluence { upgraded } => {
                for c in selected {
                    let mut copy = self.stat_copy(&c);
                    if upgraded {
                        copy.set_cost_for_turn(0);
                    }
                    self.master_reality(&mut copy);
                    if self.hand.len() < 10 {
                        self.hand.push(copy);
                        self.on_card_draw_or_discard();
                    } else {
                        self.discard_pile.push(copy);
                    }
                }
            }
            ChoiceKind::Meditate => {
                for mut c in selected {
                    c.retain = true;
                    self.hand.push(c);
                }
            }
            ChoiceKind::Omniscience { amount } => {
                for mut c in selected {
                    c.exhaust = true;
                    let original = c.clone();
                    self.push_bot(Action::QueueCard { card: c, target: None, immediate: false, autoplay: true, random_target: true });
                    for _ in 0..amount - 1 {
                        let mut tmp = self.stat_copy(&original);
                        tmp.purge_on_use = true;
                        self.push_bot(Action::QueueCard { card: tmp, target: None, immediate: false, autoplay: true, random_target: true });
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combat::CombatConfig;
    use crate::relics::RelicId;

    fn implant(relics: &[RelicId], counters: Option<Vec<i32>>) -> Combat {
        let mut cfg = CombatConfig::ironclad(1, "Writhing Mass");
        cfg.relics.extend_from_slice(relics);
        cfg.relic_counters = counters;
        cfg.player_hp = 50;
        let mut c = Combat::new(&cfg).unwrap();
        c.push_bot(Action::AddCardToDeck { card: CardId::Parasite });
        c.run();
        c
    }

    #[test]
    fn implant_with_darkstone_raises_max_hp_during_the_fight() {
        let c = implant(&[RelicId::DarkstonePeriapt], None);
        assert_eq!((c.player.hp, c.player.max_hp), (56, 86));
        assert_eq!(c.master_deck_add, [CardId::Parasite]);
    }

    #[test]
    fn implant_with_omamori_is_cancelled() {
        let c = implant(&[RelicId::Omamori, RelicId::DarkstonePeriapt], Some(vec![-1, 2, -1]));
        assert_eq!((c.player.hp, c.player.max_hp), (50, 80));
        assert!(c.master_deck_add.is_empty());
        assert_eq!(c.player.relics.iter().find(|r| r.id == RelicId::Omamori).unwrap().counter, 1);
    }

    #[test]
    fn implant_with_ceramic_fish_gains_gold() {
        let c = implant(&[], None);
        let fish = implant(&[RelicId::CeramicFish], None);
        assert_eq!(fish.player.gold, c.player.gold + 9);
    }
}
