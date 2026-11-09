//! The `GameActionManager` loop, turn sequencing and card playing.

use sts_rng::{java_shuffle, JavaRandom};

use super::{Combat, Outcome, PlayerAction, QueueItem, QueuedCard};
use crate::action::{Action, Cr};
use crate::cards::{Card, CardId, CardTarget, CardType};
use crate::powers::PowerId;
use crate::relics::RelicId;

/// `MathUtils.floor(float)`.
pub fn java_floor(v: f32) -> i32 {
    ((v as f64 + 16384.0) as i32) - 16384
}

impl Combat {
    /// Room entry through the first player turn (`onPlayerEntry`, `preBattlePrep`, battle start).
    pub(crate) fn start(&mut self, master_deck: Vec<Card>) {
        // MonsterGroup.init(): every monster rolls its first move.
        for i in self.order.clone() {
            self.roll_move(i);
        }

        // AbstractPlayer.preBattlePrep()
        self.actions.clear();
        self.pre_turn_actions.clear();
        self.card_queue.clear();
        self.turn_has_ended = false;
        self.turn = 1;
        self.player.damaged_this_combat = 0;
        self.player.cards_played_this_turn = 0;
        self.total_discarded_this_turn = 0;
        self.attacks_played_this_turn = 0;
        self.player.is_bloodied = self.player.hp <= self.player.max_hp / 2;
        self.player.game_hand_size = self.player.master_hand_size;
        // GameActionManager.clear(): the combat's history starts afresh.
        self.mantra_gained = 0;
        self.played_types.clear();
        self.master_deck = master_deck.clone();
        self.initialize_deck(master_deck);
        // maxOrbs = 0, then increaseMaxOrbSlots(masterMaxOrbs): the Defect's 3, or the 1 that
        // Prismatic Shard's onEquip gives anyone else.
        let slots = if self.character.orb_slots() == 0 && self.has_relic(RelicId::PrismaticShard) { 1 } else { self.character.orb_slots() };
        self.player.orbs = vec![None; slots];
        self.hand.clear();
        self.discard_pile.clear();
        self.exhaust_pile.clear();
        // SlaversCollar.beforeEnergyPrep, then energy.prep().
        if self.has_relic(RelicId::SlaversCollar) && (self.elite || self.order.iter().any(|&m| self.monsters[m].enemy_type == crate::monsters::EnemyType::Boss)) {
            self.player.energy_master += 1;
        }
        self.energy = 0;
        self.player.powers.clear();
        for i in self.order.clone() {
            self.use_pre_battle_action(i);
        }
        if let Some(act) = self.emerald_act {
            self.apply_emerald_elite_buff(act);
        }
        self.relics_at_pre_battle();

        // AbstractRoom.update() while waitTimer > 0: drain the pre-battle actions.
        self.run();

        // Battle start.
        self.turn_has_ended = true;
        self.push_bot(Action::GainEnergyAndEnableControls { amount: self.player.energy_master });
        self.relics_at_battle_start_pre_draw();
        self.push_bot(Action::DrawCard { amount: self.player.game_hand_size, follow_up: None, shuffle_check: false });
        self.relics_at_battle_start();
        self.skip_monster_turn = false;
        self.apply_start_of_turn_relics();
        self.relics_at_turn_start_post_draw();
        self.apply_start_of_turn_cards();
        self.apply_start_of_turn_powers(Cr::Player);
        self.apply_start_of_turn_orbs();
        self.show_intent();
        self.run();
    }

    /// `CardGroup.initializeDeck`.
    fn initialize_deck(&mut self, master_deck: Vec<Card>) {
        let mut copy: Vec<Card> = master_deck.iter().map(|c| self.same_instance_copy(c)).collect();
        let seed = self.rng.shuffle.random_long();
        java_shuffle(&mut copy, &mut JavaRandom::new(seed));
        let mut on_top = Vec::new();
        self.draw_pile.clear();
        for c in copy {
            if c.innate || c.bottled {
                on_top.push(c);
            } else {
                self.draw_pile.push(c);
            }
        }
        let extra = on_top.len() as i32 - self.player.master_hand_size;
        self.draw_pile.extend(on_top);
        if extra > 0 {
            self.pre_turn_actions.push_front(Action::DrawCard { amount: extra, follow_up: None, shuffle_check: false });
        }
    }

    /// Runs `GameActionManager.getNextAction` until the game waits on the player.
    pub(crate) fn run(&mut self) {
        loop {
            if self.outcome.is_some() || self.pending.is_some() {
                return;
            }
            if self.player.is_dead {
                self.outcome = Some(Outcome::Defeat);
                return;
            }
            if let Some(a) = self.actions.pop_front() {
                self.execute(a);
                continue;
            }
            if let Some(a) = self.pre_turn_actions.pop_front() {
                self.execute(a);
                continue;
            }
            if !self.card_queue.is_empty() {
                self.process_card_queue();
                continue;
            }
            if !self.monster_attacks_queued {
                self.monster_attacks_queued = true;
                if !self.skip_monster_turn {
                    for i in self.order.clone() {
                        let m = &self.monsters[i];
                        if !m.is_dead_or_escaped() || m.half_dead {
                            self.monster_queue.push_back(i);
                        }
                    }
                }
                continue;
            }
            if let Some(&m) = self.monster_queue.front() {
                let mon = &self.monsters[m];
                if !mon.is_dead_or_escaped() || mon.half_dead {
                    self.take_turn(m);
                    self.apply_turn_powers(m);
                }
                self.monster_queue.pop_front();
                continue;
            }
            if self.turn_has_ended && !self.monsters_basically_dead() {
                self.start_player_turn();
                continue;
            }
            if self.monsters_basically_dead() {
                self.victory();
                return;
            }
            if self.smoked {
                // The player ran away: endBattle() once the queue is empty.
                self.victory();
                return;
            }
            if self.unceasing_top_draw() {
                continue;
            }
            if self.end_turn_queued {
                self.end_turn_queued = false;
                self.room_end_turn();
                continue;
            }
            // Death animations finish before the player can act again; the game then clears
            // the dead monster's powers (`updateDeathAnimation`).
            for m in self.monsters.iter_mut() {
                if m.is_dying {
                    m.powers.clear();
                }
            }
            return;
        }
    }

    fn victory(&mut self) {
        // AbstractRoom.endBattle() -> AbstractPlayer.onVictory()
        if !self.player.is_dead {
            self.relics_on_victory();
            // The powers' onVictory: Self Repair.
            let repair: i32 = self.player.powers.iter().filter(|p| p.id == PowerId::Repair).map(|p| p.amount).sum();
            if repair > 0 && self.player.hp > 0 {
                self.heal(Cr::Player, repair);
            }
        }
        self.player.damaged_this_combat = 0;
        self.outcome = Some(Outcome::Victory);
    }

    /// The `turnHasEnded` branch of `getNextAction`: end of round, then the next player turn.
    fn start_player_turn(&mut self) {
        if !self.skip_monster_turn {
            // MonsterGroup.applyEndOfTurnPowers()
            for i in self.order.clone() {
                if !self.monsters[i].is_dying && !self.monsters[i].is_escaping {
                    self.apply_end_of_turn_triggers(Cr::Monster(i));
                }
            }
            self.at_end_of_round(Cr::Player);
            for i in self.order.clone() {
                if !self.monsters[i].is_dying && !self.monsters[i].is_escaping {
                    self.at_end_of_round(Cr::Monster(i));
                }
            }
        }
        self.player.cards_played_this_turn = 0;
        self.orbs_channeled_this_turn.clear();
        self.apply_start_of_turn_relics();
        self.apply_start_of_turn_cards();
        self.apply_start_of_turn_powers(Cr::Player);
        self.apply_start_of_turn_orbs();
        self.turn += 1;
        self.skip_monster_turn = false;
        self.turn_has_ended = false;
        self.total_discarded_this_turn = 0;
        self.attacks_played_this_turn = 0;
        if !self.player.has_power(PowerId::Barricade) && !self.player.has_power(PowerId::Blur) {
            if self.has_relic(RelicId::Calipers) {
                self.player.block = (self.player.block - 15).max(0);
            } else {
                self.player.block = 0;
            }
        }
        // DrawCardAction(..., endTurnDraw = true) constructs a PlayerTurnEffect in its constructor,
        // before the action is queued: the powers' onEnergyRecharge (Collect's Miracle) come first.
        self.player_turn_effect();
        self.push_bot(Action::DrawCard { amount: self.player.game_hand_size, follow_up: None, shuffle_check: false });
        self.relics_at_turn_start_post_draw();
        self.apply_start_of_turn_post_draw_powers();
    }

    /// `PlayerTurnEffect` constructor: energy recharge and intent refresh.
    fn player_turn_effect(&mut self) {
        // EnergyManager.recharge(): Ice Cream keeps unspent energy.
        if self.has_relic(RelicId::IceCream) {
            self.energy = (self.energy + self.player.energy_master).min(999);
        } else {
            self.energy = self.player.energy_master;
        }
        self.powers_on_energy_recharge();
        self.show_intent();
    }

    /// `onEnergyRecharge` for the player's powers (Energized, Deva Form, Collect).
    pub(crate) fn powers_on_energy_recharge(&mut self) {
        for (id, amount) in self.player.powers.iter().map(|p| (p.id, p.amount)).collect::<Vec<_>>() {
            match id {
                PowerId::Energized | PowerId::EnergizedBlue => {
                    // AbstractPlayer.gainEnergy
                    self.energy = (self.energy + amount).min(999);
                    self.push_bot(Action::RemovePower { target: Cr::Player, id });
                }
                PowerId::DevaForm => {
                    if let Some(p) = self.player.powers.iter_mut().find(|p| p.id == id) {
                        let gain = p.base;
                        p.base += p.amount;
                        self.energy = (self.energy + gain).min(999);
                    }
                }
                PowerId::Collect => {
                    let mut miracle = self.new_card(CardId::Miracle);
                    miracle.upgrade();
                    self.push_bot(Action::MakeTempCardInHand { card: miracle, amount: 1 });
                    if amount <= 1 {
                        self.push_bot(Action::RemovePower { target: Cr::Player, id });
                    } else {
                        self.push_bot(Action::ReducePower { target: Cr::Player, id, amount: 1 });
                    }
                }
                _ => {}
            }
        }
    }

    /// `GameActionManager.callEndTurnEarlySequence()` (Time Warp): autoplayed queued cards are
    /// used without their effects, the queue and limbo are cleared, and the turn ends.
    pub(crate) fn end_turn_early(&mut self) {
        let items: Vec<QueueItem> = self.card_queue.drain(..).collect();
        for item in items {
            if !item.autoplay {
                continue;
            }
            let card = match &item.card {
                QueuedCard::InHand(inst) => self.hand.iter().position(|c| c.instance == *inst).map(|k| self.hand.remove(k)),
                QueuedCard::InLimbo(inst) => self.limbo.iter().position(|c| c.instance == *inst).map(|k| self.limbo.remove(k)),
                QueuedCard::Owned(c) => Some((**c).clone()),
                QueuedCard::EndTurn => None,
            };
            if let Some(mut c) = card {
                c.dont_trigger_on_use = true;
                self.push_bot(Action::UseCard { card: c, target: None, exhaust_card: false });
            }
        }
        self.limbo.clear();
        // EndTurnButton.disable(true)
        self.push_bot(Action::QueueEndTurn);
        self.end_turn_queued = true;
    }

    /// `MonsterRoomElite.applyEmeraldEliteBuff`: one of four buffs, picked with mapRng.
    fn apply_emerald_elite_buff(&mut self, act: i32) {
        let which = self.rng.map.random_range(0, 3);
        for m in self.order.clone() {
            let me = Some(Cr::Monster(m));
            match which {
                0 => self.bot_power(Cr::Monster(m), me, crate::powers::Power::new(PowerId::Strength, act + 1)),
                1 => {
                    // IncreaseMaxHpAction(m, 0.25, true)
                    let amt = crate::run::map::java_round(self.monsters[m].max_hp as f32 * 0.25);
                    self.push_bot(Action::IncreaseMaxHp { target: m, amount: amt });
                }
                2 => self.bot_power(Cr::Monster(m), me, crate::powers::Power::new(PowerId::Metallicize, act * 2 + 2)),
                _ => self.bot_power(Cr::Monster(m), me, crate::powers::Power::new(PowerId::Regenerate, 1 + act * 2)),
            }
        }
    }

    /// `MonsterGroup.showIntent()`.
    pub(crate) fn show_intent(&mut self) {
        for i in self.order.clone() {
            self.create_intent(i);
        }
    }

    /// `AbstractMonster.createIntent()`.
    pub(crate) fn create_intent(&mut self, i: usize) {
        let mv = self.monsters[i].mv;
        let dmg = if mv.base_damage > -1 { self.monster_intent_damage(i, mv.base_damage) } else { -1 };
        let m = &mut self.monsters[i];
        m.intent = mv.intent;
        m.next_move = mv.next_move;
        m.intent_base_dmg = mv.base_damage;
        if mv.base_damage > -1 {
            m.intent_dmg = dmg;
            if mv.is_multi_damage {
                m.intent_multi = mv.multiplier;
                m.is_multi_dmg = true;
            } else {
                m.intent_multi = -1;
                m.is_multi_dmg = false;
            }
        }
    }

    /// `applyStartOfTurnRelics`: the stance's `atStartOfTurn`, then the relics' `atTurnStart`.
    fn apply_start_of_turn_relics(&mut self) {
        self.stance_at_start_of_turn();
        self.relics_at_turn_start();
    }

    /// `applyStartOfTurnCards`: `atTurnStart` for the draw pile, hand and discard pile.
    fn apply_start_of_turn_cards(&mut self) {
        for c in self.draw_pile.iter_mut().chain(self.hand.iter_mut()).chain(self.discard_pile.iter_mut()) {
            if c.id == CardId::Eviscerate {
                c.reset_attributes();
            }
        }
    }

    /// `AbstractRoom.endTurn()`, run once the end-of-turn queue item has resolved.
    fn room_end_turn(&mut self) {
        self.apply_end_of_turn_triggers(Cr::Player);
        self.push_bot(Action::ClearCardQueue);
        self.push_bot(Action::DiscardAtEndOfTurn);
        for c in self.draw_pile.iter_mut().chain(self.discard_pile.iter_mut()).chain(self.hand.iter_mut()) {
            c.reset_attributes();
        }
        self.push_bot(Action::EndTurnSequence);
    }

    /// `GameActionManager.callEndOfTurnActions()`.
    fn call_end_of_turn_actions(&mut self) {
        self.relics_on_player_end_turn();
        self.at_end_of_turn_pre_end_turn_cards(Cr::Player);
        self.push_bot(Action::TriggerEndOfTurnOrbs);
        // Cards in hand with an end-of-turn effect (Burn, Decay, Doubt, Regret, Shame) queue
        // themselves (`triggerOnEndOfTurnForPlayingCard`); Pride puts a copy on the draw pile.
        let prides: Vec<Card> = self.hand.iter().filter(|c| c.id == CardId::Pride).cloned().collect();
        for p in prides {
            let card = self.stat_copy(&p);
            self.push_bot(Action::MakeTempCardInDrawPile { card, amount: 1, random_spot: false, to_bottom: false });
        }
        let hand_size = self.hand.len() as i32;
        let burns: Vec<u32> = self
            .hand
            .iter()
            .filter(|c| matches!(c.id, CardId::Burn | CardId::Decay | CardId::Doubt | CardId::Regret | CardId::Shame))
            .map(|c| c.instance)
            .collect();
        for inst in burns {
            if let Some(c) = self.hand.iter_mut().find(|c| c.instance == inst) {
                c.dont_trigger_on_use = true;
                if c.id == CardId::Regret {
                    c.magic = hand_size;
                    c.base_magic = hand_size;
                }
            }
            self.card_queue.push_back(QueueItem {
                card: QueuedCard::InHand(inst),
                target: None,
                energy_on_use: self.energy,
                ignore_energy_total: false,
                autoplay: false,
                random_target: false,
                end_turn_autoplay: true,
            });
        }
    }

    /// The `cardQueue` branch of `getNextAction`.
    fn process_card_queue(&mut self) {
        let item = self.card_queue.front().cloned().expect("queue non-empty");
        if matches!(item.card, QueuedCard::EndTurn) {
            self.call_end_of_turn_actions();
            self.card_queue.pop_front();
            return;
        }
        if self.card_queue.len() == 1 && item.end_turn_autoplay {
            if let Some(r) = self.player.relics.iter_mut().find(|r| r.id == RelicId::UnceasingTop) {
                r.set_flag(crate::relics::FLAG_B, true);
            }
        }

        // Take a working copy of the card; it is removed from its pile when actually used.
        let mut card = match &item.card {
            QueuedCard::InHand(inst) => self.hand.iter().find(|c| c.instance == *inst).cloned(),
            QueuedCard::InLimbo(inst) => self.limbo.iter().find(|c| c.instance == *inst).cloned(),
            QueuedCard::Owned(c) => Some((**c).clone()),
            QueuedCard::EndTurn => None,
        };
        let Some(c) = card.as_mut() else {
            self.card_queue.pop_front();
            return;
        };
        c.in_autoplay = item.autoplay;
        let mut target = item.target;
        if item.random_target {
            target = self.get_random_monster(None, true, true);
        }

        let can_play;
        if !self.can_use(c, target) && !c.dont_trigger_on_use {
            // The card fizzles; a limbo copy is removed.
            self.remove_from_limbo(c.instance);
            can_play = false;
        } else {
            can_play = true;
            if self.free_to_play(c) {
                c.free_to_play_once = true;
            }
            c.energy_on_use = item.energy_on_use;
            c.ignore_energy_on_use = if c.in_autoplay { true } else { item.ignore_energy_total };
            if !c.dont_trigger_on_use {
                self.relics_on_play_card();
                self.player.cards_played_this_turn += 1;
                self.cards_played_this_combat += 1;
                self.played_types.push(c.card_type());
                match c.card_type() {
                    CardType::Attack => self.attacks_played_this_turn += 1,
                    CardType::Power => self.powers_played_this_combat += 1,
                    _ => {}
                }
            }
            let needs_target = c.target() == CardTarget::Enemy;
            let target_gone = target.map_or(true, |t| self.monsters[t].is_dead_or_escaped());
            if needs_target && target_gone {
                // Fades out of limbo; a hand card stays in hand.
                self.remove_from_limbo(c.instance);
            } else {
                self.use_card(c.clone(), target, &item.card);
            }
        }
        self.card_queue.pop_front();
        if !can_play && c.in_autoplay {
            let mut owned = c.clone();
            owned.dont_trigger_on_use = true;
            if let QueuedCard::InHand(inst) = item.card {
                self.hand.retain(|h| h.instance != inst);
            }
            let exhaust_card = owned.exhaust_on_use_once || owned.exhaust;
            self.push_bot(Action::UseCard { card: owned, target: None, exhaust_card });
        }
    }

    fn remove_from_limbo(&mut self, instance: u32) {
        self.limbo.retain(|c| c.instance != instance);
    }

    /// Removes the queued card from its pile (`hand.removeCard` / leaving limbo).
    fn take_card_for_use(&mut self, from: &QueuedCard) {
        match from {
            QueuedCard::InHand(inst) => {
                if let Some(pos) = self.hand.iter().position(|c| c.instance == *inst) {
                    self.hand.remove(pos);
                }
            }
            QueuedCard::InLimbo(inst) => self.remove_from_limbo(*inst),
            _ => {}
        }
    }

    /// `AbstractPlayer.useCard`. `card` is a working copy; the original stays in its pile until
    /// after `use()` and the `UseCardAction` hooks, as in the game.
    fn use_card(&mut self, mut card: Card, target: Option<usize>, from: &QueuedCard) {
        let multi = self.calculate_card_damage(&mut card, target);
        if card.cost == -1 && self.energy < card.energy_on_use && !card.ignore_energy_on_use {
            card.energy_on_use = self.energy;
        }
        if card.cost == -1 && card.in_autoplay {
            card.free_to_play_once = true;
        }
        self.card_use(&mut card, target, multi);

        // UseCardAction constructor.
        let mut exhaust_card = card.exhaust_on_use_once || card.exhaust;
        if !card.dont_trigger_on_use {
            self.on_use_card(&card, target, &mut exhaust_card);
            // hand.triggerOnOtherCardPlayed: Pain.
            let pains = self.hand.iter().filter(|c| c.id == CardId::Pain && c.instance != card.instance).count();
            for _ in 0..pains {
                self.push_top(Action::LoseHp { target: Cr::Player, source: Some(Cr::Player), amount: 1 });
            }
        }
        self.take_card_for_use(from);
        let cost_for_turn = card.cost_for_turn;
        let free = self.free_to_play(&card);
        let autoplay = card.in_autoplay;
        let is_skill = card.card_type() == CardType::Skill;
        let dont_trigger = card.dont_trigger_on_use;
        self.push_bot(Action::UseCard { card, target, exhaust_card });
        if !dont_trigger {
            // hand.triggerOnOtherCardPlayed: the player's powers' onAfterCardPlayed (A Thousand Cuts).
            self.on_after_card_played();
        }

        if cost_for_turn > 0 && !free && !autoplay && !(self.player.has_power(PowerId::Corruption) && is_skill) {
            self.energy = (self.energy - cost_for_turn).max(0);
        }
    }

    /// `AbstractCard.canUse`.
    pub fn can_use(&self, card: &Card, target: Option<usize>) -> bool {
        let t = card.card_type();
        if t == CardType::Status && card.cost_for_turn < -1 && !self.has_relic(RelicId::MedicalKit) {
            return false;
        }
        if t == CardType::Curse && card.cost_for_turn < -1 && !self.has_relic(RelicId::BlueCandle) {
            return false;
        }
        if !self.card_playable(card, target) || !self.has_enough_energy(card) {
            return false;
        }
        match card.id {
            CardId::Clash => self.hand.iter().all(|c| c.card_type() == CardType::Attack),
            CardId::GrandFinale => self.draw_pile.is_empty(),
            CardId::Reflex | CardId::Tactician | CardId::DeusExMachina => false,
            // Secret Technique and Secret Weapon need a skill (an attack) in the draw pile.
            CardId::SecretTechnique => self.draw_pile.iter().any(|c| c.card_type() == CardType::Skill),
            CardId::SecretWeapon => self.draw_pile.iter().any(|c| c.card_type() == CardType::Attack),
            // Signature Move: no other attack in hand.
            CardId::SignatureMove => !self.hand.iter().any(|c| c.card_type() == CardType::Attack && c.instance != card.instance),
            _ => true,
        }
    }

    fn card_playable(&self, card: &Card, target: Option<usize>) -> bool {
        let t = card.target();
        let target_ok = (t != CardTarget::Enemy && t != CardTarget::SelfAndEnemy) || target.map_or(true, |m| !self.monsters[m].is_dying);
        target_ok && !self.monsters_basically_dead()
    }

    fn has_enough_energy(&self, card: &Card) -> bool {
        if self.turn_has_ended {
            return false;
        }
        if self.player.has_power(PowerId::Entangled) && card.card_type() == CardType::Attack {
            return false;
        }
        if !self.relics_can_play() {
            return false;
        }
        // Normality: no more than three cards a turn.
        if self.player.cards_played_this_turn >= 3 && self.hand.iter().any(|c| c.id == CardId::Normality) {
            return false;
        }
        !(self.energy < card.cost_for_turn && !self.free_to_play(card) && !card.in_autoplay)
    }

    /// `freeToPlay`: free once, or an attack while Swivel's Free Attack lasts.
    pub fn free_to_play(&self, card: &Card) -> bool {
        card.free_to_play_once || (self.player.has_power(PowerId::FreeAttack) && card.card_type() == CardType::Attack)
    }

    /// `AbstractPlayer.draw()` for one card.
    pub(crate) fn draw_one(&mut self) {
        if self.hand.len() >= 10 || self.draw_pile.is_empty() {
            return;
        }
        let mut card = self.draw_pile.pop().unwrap();
        self.drawn_cards.push((card.instance, card.card_type()));
        // triggerWhenDrawn.
        match card.id {
            CardId::Void => self.push_bot(Action::LoseEnergy { amount: 1 }),
            CardId::EndlessAgony => {
                let copy = self.stat_copy(&card);
                self.push_top(Action::MakeTempCardInHand { card: copy, amount: 1 });
            }
            CardId::Eviscerate => card.set_cost_for_turn(card.cost - self.total_discarded_this_turn),
            CardId::DeusExMachina => {
                // addToTop twice: the card exhausts first, then the Miracles arrive.
                let miracle = self.new_card(CardId::Miracle);
                self.push_top(Action::MakeTempCardInHand { card: miracle, amount: card.magic });
                self.push_top(Action::ExhaustSpecificInHand { instance: card.instance });
            }
            _ => {}
        }
        self.hand.push(card);
        let idx = self.hand.len() - 1;
        self.on_card_draw(idx);
        self.on_card_draw_or_discard();
    }

    /// `AbstractPlayer.onCardDrawOrDiscard`.
    pub(crate) fn on_card_draw_or_discard(&mut self) {
        // The powers' onDrawOrDiscard: Accuracy refreshes the Shivs in hand.
        if let Some(n) = self.player.power(PowerId::Accuracy).map(|p| p.amount) {
            for c in self.hand.iter_mut().filter(|c| c.id == CardId::Shiv) {
                c.base_damage = if c.upgraded { 6 } else { 4 } + n;
            }
        }
        if self.player.has_power(PowerId::Corruption) {
            for c in self.hand.iter_mut() {
                if c.card_type() == CardType::Skill && c.cost_for_turn != 0 {
                    c.modify_cost_for_combat(-9);
                }
            }
        }
    }

    /// `MonsterGroup.getRandomMonster(null, aliveOnly, rng)` using `cardRandomRng` or `aiRng`.
    pub(crate) fn get_random_monster(&mut self, exception: Option<usize>, alive_only: bool, card_rng: bool) -> Option<usize> {
        if self.monsters_basically_dead() {
            return None;
        }
        let candidates: Vec<usize> = self
            .order
            .iter()
            .copied()
            .filter(|&i| exception != Some(i))
            .filter(|&i| !alive_only || self.monsters[i].is_targetable())
            .collect();
        if exception.is_some() && self.order.len() == 1 {
            return Some(self.order[0]);
        }
        if candidates.is_empty() {
            return None;
        }
        let rng = if card_rng { &mut self.rng.card_random } else { &mut self.rng.ai };
        let k = rng.random_range(0, candidates.len() as i32 - 1);
        Some(candidates[k as usize])
    }

    /// Shuffles a list with an unseeded-in-game `Collections.shuffle(list)`.
    pub(crate) fn unseeded_shuffle<T>(&mut self, items: &mut [T]) {
        let seed = self.rng.unseeded.random_long();
        java_shuffle(items, &mut JavaRandom::new(seed));
    }

    // ---- Agent interface ----

    /// Decisions available right now.
    pub fn legal_actions(&self) -> Vec<PlayerAction> {
        if self.outcome.is_some() {
            return Vec::new();
        }
        if let Some(p) = &self.pending {
            return super::observe::choice_actions(p);
        }
        let mut out = Vec::new();
        for (i, c) in self.hand.iter().enumerate() {
            match c.target() {
                CardTarget::Enemy | CardTarget::SelfAndEnemy => {
                    for &m in &self.order {
                        let mon = &self.monsters[m];
                        if mon.is_targetable() && self.can_use(c, Some(m)) {
                            out.push(PlayerAction::PlayCard { hand_index: i, target: Some(m) });
                        }
                    }
                }
                _ => {
                    if self.can_use(c, None) {
                        out.push(PlayerAction::PlayCard { hand_index: i, target: None });
                    }
                }
            }
        }
        for (slot, p) in self.potions.iter().enumerate() {
            let Some(p) = *p else { continue };
            if self.can_use_potion(p) {
                if p.data().needs_target {
                    for &m in &self.order {
                        if self.monsters[m].is_targetable() {
                            out.push(PlayerAction::UsePotion { slot, target: Some(m) });
                        }
                    }
                } else {
                    out.push(PlayerAction::UsePotion { slot, target: None });
                }
            }
        }
        out.push(PlayerAction::EndTurn);
        out
    }

    /// Applies a decision and runs the game until the next decision.
    pub fn step(&mut self, action: &PlayerAction) -> Result<(), String> {
        if self.outcome.is_some() {
            return Err("combat is over".into());
        }
        if self.pending.is_some() {
            let PlayerAction::Choose { indices } = action else {
                return Err("a card selection is pending; send a choose action".into());
            };
            self.resolve_choice(indices)?;
            self.run();
            return Ok(());
        }
        match action {
            PlayerAction::PlayCard { hand_index, target } => {
                let card = self.hand.get(*hand_index).ok_or("hand index out of range")?;
                let needs_target = matches!(card.target(), CardTarget::Enemy | CardTarget::SelfAndEnemy);
                if needs_target {
                    let t = target.ok_or("this card needs a target")?;
                    if t >= self.monsters.len() || !self.monsters[t].is_targetable() {
                        return Err("invalid target".into());
                    }
                }
                let target = if needs_target { *target } else { None };
                if !self.can_use(card, target) {
                    return Err(format!("{} cannot be played now", card.name()));
                }
                let inst = card.instance;
                if let Some(t) = target {
                    self.face_target(t);
                }
                self.card_queue.push_back(QueueItem {
                    card: QueuedCard::InHand(inst),
                    target,
                    energy_on_use: self.energy,
                    ignore_energy_total: false,
                    autoplay: false,
                    random_target: false,
                    end_turn_autoplay: false,
                });
            }
            PlayerAction::EndTurn => {
                // EndTurnButton.disable(true)
                self.push_bot(Action::QueueEndTurn);
                self.end_turn_queued = true;
            }
            PlayerAction::Choose { .. } => return Err("no selection is pending".into()),
            PlayerAction::UsePotion { slot, target } => {
                let p = self.potions.get(*slot).copied().flatten().ok_or("no potion in that slot")?;
                if !self.can_use_potion(p) {
                    return Err("that potion cannot be used now".into());
                }
                let target = if p.data().needs_target {
                    let t = target.ok_or("this potion needs a target")?;
                    if t >= self.monsters.len() || !self.monsters[t].is_targetable() {
                        return Err("invalid target".into());
                    }
                    Some(t)
                } else {
                    None
                };
                if let Some(t) = target {
                    self.face_target(t);
                }
                self.use_potion(*slot, target);
            }
            PlayerAction::DiscardPotion { slot } => {
                let s = self.potions.get_mut(*slot).ok_or("no such potion slot")?;
                if s.is_none() {
                    return Err("that slot is empty".into());
                }
                *s = None;
                return Ok(());
            }
        }
        self.run();
        Ok(())
    }
}
