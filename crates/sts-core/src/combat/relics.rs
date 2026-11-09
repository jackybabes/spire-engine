//! Relic hooks during combat. Each function mirrors one `AbstractRelic` callback and walks the
//! player's relics in order, as the game's loops do.

use super::Combat;
use crate::action::{Action, Cr, DamageInfo, DamageType};
use crate::cards::{Card, CardId, CardRarity, CardType};
use crate::monsters::EnemyType;
use crate::powers::{Power, PowerId};
use crate::relics::{RelicId, FLAG_A, FLAG_B, FLAG_C};

impl Combat {
    pub fn has_relic(&self, id: RelicId) -> bool {
        self.player.relics.iter().any(|r| r.id == id)
    }

    fn relic_ids(&self) -> Vec<RelicId> {
        self.player.relics.iter().map(|r| r.id).collect()
    }

    /// `AbstractDungeon.returnTrulyRandomCardInCombat()`: any non-healing card of the src pools.
    pub(crate) fn truly_random_any_card_in_combat(&mut self) -> CardId {
        let mut list = Vec::new();
        for rarity in [CardRarity::Common, CardRarity::Uncommon, CardRarity::Rare] {
            for &id in self.character.library_order().iter().rev() {
                let d = id.data();
                if d.rarity == rarity && !d.healing {
                    list.push(id);
                }
            }
        }
        list[self.rng.card_random.random(list.len() as i32 - 1) as usize]
    }

    /// `AbstractDungeon.returnTrulyRandomColorlessCardInCombat()`.
    pub(crate) fn truly_random_colorless_in_combat(&mut self) -> CardId {
        let list: Vec<CardId> = crate::cards::COLORLESS_LIBRARY_ORDER.iter().rev().copied().filter(|c| !c.data().healing).collect();
        list[self.rng.card_random.random(list.len() as i32 - 1) as usize]
    }

    fn has_boss(&self) -> bool {
        self.order.iter().any(|&m| self.monsters[m].enemy_type == EnemyType::Boss)
    }

    /// `applyPreCombatLogic`: `atPreBattle`.
    pub(crate) fn relics_at_pre_battle(&mut self) {
        for i in 0..self.player.relics.len() {
            let r = &mut self.player.relics[i];
            match r.id {
                RelicId::AncientTeaSet | RelicId::Lantern => r.set_flag(FLAG_A, true),
                RelicId::ArtOfWar => {
                    r.set_flag(FLAG_A, true);
                    r.set_flag(FLAG_B, true);
                }
                RelicId::CentennialPuzzle | RelicId::UnceasingTop => r.set_flag(FLAG_A, false),
                // firstTurn
                RelicId::RunicCapacitor => r.set_flag(FLAG_A, true),
                RelicId::CrackedCore => self.channel_orb(super::Orb::new(super::OrbId::Lightning)),
                RelicId::SymbioticVirus => self.channel_orb(super::Orb::new(super::OrbId::Dark)),
                RelicId::NuclearBattery => self.channel_orb(super::Orb::new(super::OrbId::Plasma)),
                RelicId::SneckoEye => self.bot_power(Cr::Player, Some(Cr::Player), Power::flag(PowerId::Confusion)),
                RelicId::Enchiridion => {
                    let id = self.truly_random_card_in_combat(CardType::Power);
                    let mut c = self.new_card(id);
                    if c.cost != -1 {
                        c.set_cost_for_turn(0);
                    }
                    self.push_bot(Action::MakeTempCardInHand { card: c, amount: 1 });
                }
                _ => {}
            }
        }
    }

    /// `applyStartOfCombatPreDrawLogic`: `atBattleStartPreDraw`.
    pub(crate) fn relics_at_battle_start_pre_draw(&mut self) {
        for i in 0..self.player.relics.len() {
            match self.player.relics[i].id {
                RelicId::GamblingChip => self.player.relics[i].set_flag(FLAG_A, false),
                RelicId::Toolbox => self.push_bot(Action::ChooseOneColorless),
                RelicId::NinjaScroll => {
                    let card = self.new_card(CardId::Shiv);
                    self.push_bot(Action::MakeTempCardInHand { card, amount: 3 });
                }
                RelicId::PureWater | RelicId::HolyWater => {
                    let card = self.new_card(CardId::Miracle);
                    let amount = if self.player.relics[i].id == RelicId::PureWater { 1 } else { 3 };
                    self.push_bot(Action::MakeTempCardInHand { card, amount });
                }
                _ => {}
            }
        }
    }

    /// `applyStartOfCombatLogic`: `atBattleStart`.
    pub(crate) fn relics_at_battle_start(&mut self) {
        let p = Some(Cr::Player);
        for i in 0..self.player.relics.len() {
            let id = self.player.relics[i].id;
            match id {
                RelicId::Akabeko => self.top_power(Cr::Player, p, Power::new(PowerId::Vigor, 8)),
                RelicId::TeardropLocket => self.push_top(Action::ChangeStance { stance: super::Stance::Calm }),
                RelicId::Anchor => self.push_bot(Action::GainBlock { target: Cr::Player, amount: 10 }),
                RelicId::BagOfMarbles => {
                    for m in self.order.clone() {
                        self.bot_power(Cr::Monster(m), p, Power::vulnerable(1, false, false));
                    }
                }
                RelicId::BagOfPreparation | RelicId::SnakeRing => self.push_bot(Action::DrawCard { amount: 2, follow_up: None, shuffle_check: false }),
                RelicId::DataDisk => self.top_power(Cr::Player, p, Power::new(PowerId::Focus, 1)),
                RelicId::TwistedFunnel => {
                    for m in self.order.clone() {
                        if !self.monsters[m].is_dying {
                            self.bot_power(Cr::Monster(m), p, Power::new(PowerId::Poison, 4));
                        }
                    }
                }
                RelicId::BloodVial => self.push_top(Action::Heal { target: Cr::Player, amount: 2 }),
                RelicId::BronzeScales => self.top_power(Cr::Player, p, Power::new(PowerId::Thorns, 3)),
                RelicId::CaptainsWheel | RelicId::HornCleat => {
                    let r = &mut self.player.relics[i];
                    r.counter = 0;
                    r.used_up = false;
                }
                RelicId::StoneCalendar | RelicId::VelvetChoker => self.player.relics[i].counter = 0,
                RelicId::ClockworkSouvenir => self.top_power(Cr::Player, p, Power::new(PowerId::Artifact, 1)),
                RelicId::DuVuDoll | RelicId::Girya => {
                    let n = self.player.relics[i].counter;
                    if (id == RelicId::DuVuDoll && n > 0) || (id == RelicId::Girya && n != 0) {
                        self.top_power(Cr::Player, p, Power::new(PowerId::Strength, n));
                    }
                }
                RelicId::FossilizedHelix => self.bot_power(Cr::Player, p, Power::new(PowerId::Buffer, 1)),
                RelicId::GremlinMask => self.bot_power(Cr::Player, p, Power::weak(1, false)),
                RelicId::MarkOfPain => {
                    let wound = self.new_card(CardId::Wound);
                    self.push_bot(Action::MakeTempCardInDrawPile { card: wound, amount: 2, random_spot: true, to_bottom: false });
                }
                RelicId::MutagenicStrength => {
                    self.top_power(Cr::Player, p, Power::new(PowerId::Strength, 3));
                    self.top_power(Cr::Player, p, Power::new(PowerId::Flex, 3));
                }
                RelicId::NeowsLament => {
                    let r = &mut self.player.relics[i];
                    if r.counter > 0 {
                        r.counter -= 1;
                        if r.counter == 0 {
                            r.counter = -2;
                            r.used_up = true;
                        }
                        for m in self.order.clone() {
                            self.monsters[m].hp = 1;
                        }
                    }
                }
                RelicId::OddlySmoothStone => self.top_power(Cr::Player, p, Power::new(PowerId::Dexterity, 1)),
                RelicId::Pantograph => {
                    if self.has_boss() {
                        self.push_top(Action::Heal { target: Cr::Player, amount: 25 });
                    }
                }
                RelicId::PenNib => {
                    if self.player.relics[i].counter == 9 {
                        self.bot_power(Cr::Player, p, Power::new(PowerId::PenNib, 1));
                    }
                }
                RelicId::PhilosopherStone => {
                    for m in self.order.clone() {
                        self.add_power_direct(Cr::Monster(m), Power::new(PowerId::Strength, 1));
                    }
                    self.on_modify_power();
                }
                RelicId::Pocketwatch => {
                    let r = &mut self.player.relics[i];
                    r.counter = 0;
                    r.set_flag(FLAG_A, true);
                }
                RelicId::PreservedInsect => {
                    if self.elite {
                        for m in self.order.clone() {
                            let mon = &mut self.monsters[m];
                            let cap = (mon.max_hp as f32 * (1.0 - 0.25)) as i32;
                            if mon.hp > cap {
                                mon.hp = cap;
                            }
                        }
                    }
                }
                RelicId::RedMask => {
                    for m in self.order.clone() {
                        self.bot_power(Cr::Monster(m), p, Power::weak(1, false));
                    }
                }
                RelicId::RedSkull => {
                    self.player.relics[i].set_flag(FLAG_A, false);
                    self.push_bot(Action::RedSkullCheck);
                }
                RelicId::Sling => {
                    if self.elite {
                        self.top_power(Cr::Player, p, Power::new(PowerId::Strength, 2));
                    }
                }
                RelicId::ThreadAndNeedle => self.top_power(Cr::Player, p, Power::new(PowerId::PlatedArmor, 4)),
                RelicId::Vajra => self.top_power(Cr::Player, p, Power::new(PowerId::Strength, 1)),
                _ => {}
            }
        }
    }

    /// `applyStartOfTurnRelics`: `atTurnStart`.
    pub(crate) fn relics_at_turn_start(&mut self) {
        let p = Some(Cr::Player);
        for i in 0..self.player.relics.len() {
            let id = self.player.relics[i].id;
            match id {
                // triggeredThisTurn
                RelicId::HoveringKite => self.player.relics[i].set_flag(FLAG_A, false),
                RelicId::Damaru => {
                    let m = self.mantra(1);
                    self.bot_power(Cr::Player, None, m);
                }
                // pulse: HP was lost since the last turn started.
                RelicId::EmotionChip => {
                    if self.player.relics[i].flag(FLAG_A) {
                        self.player.relics[i].set_flag(FLAG_A, false);
                        self.push_bot(Action::Impulse { dark_only: false });
                    }
                }
                RelicId::Inserter => {
                    let r = &mut self.player.relics[i];
                    r.counter = if r.counter == -1 { r.counter + 2 } else { r.counter + 1 };
                    if r.counter == 2 {
                        r.counter = 0;
                        self.push_bot(Action::IncreaseMaxOrb { amount: 1 });
                    }
                }
                RelicId::RunicCapacitor => {
                    if self.player.relics[i].flag(FLAG_A) {
                        self.push_bot(Action::IncreaseMaxOrb { amount: 3 });
                        self.player.relics[i].set_flag(FLAG_A, false);
                    }
                }
                RelicId::AncientTeaSet => {
                    let r = &mut self.player.relics[i];
                    if r.flag(FLAG_A) {
                        if r.counter == -2 {
                            r.counter = -1;
                            self.push_top(Action::GainEnergy { amount: 2 });
                        }
                        self.player.relics[i].set_flag(FLAG_A, false);
                    }
                }
                RelicId::ArtOfWar => {
                    let r = &mut self.player.relics[i];
                    let gain = r.flag(FLAG_B) && !r.flag(FLAG_A);
                    r.set_flag(FLAG_A, false);
                    r.set_flag(FLAG_B, true);
                    if gain {
                        self.push_bot(Action::GainEnergy { amount: 1 });
                    }
                }
                RelicId::Brimstone => {
                    self.top_power(Cr::Player, p, Power::new(PowerId::Strength, 2));
                    for m in self.order.clone() {
                        self.top_power(Cr::Monster(m), Some(Cr::Monster(m)), Power::new(PowerId::Strength, 1));
                    }
                }
                RelicId::CaptainsWheel | RelicId::HornCleat => {
                    let (at, block) = if id == RelicId::CaptainsWheel { (3, 18) } else { (2, 14) };
                    let r = &mut self.player.relics[i];
                    if !r.used_up {
                        r.counter += 1;
                    }
                    if r.counter == at {
                        r.counter = -1;
                        r.used_up = true;
                        self.push_bot(Action::GainBlock { target: Cr::Player, amount: block });
                    }
                }
                RelicId::HappyFlower | RelicId::IncenseBurner => {
                    let at = if id == RelicId::HappyFlower { 3 } else { 6 };
                    let r = &mut self.player.relics[i];
                    if r.counter == -1 {
                        r.counter += 2;
                    } else {
                        r.counter += 1;
                    }
                    if r.counter == at {
                        r.counter = 0;
                        if id == RelicId::HappyFlower {
                            self.push_bot(Action::GainEnergy { amount: 1 });
                        } else {
                            self.bot_power(Cr::Player, None, Power::new(PowerId::IntangiblePlayer, 1));
                        }
                    }
                }
                RelicId::Kunai | RelicId::LetterOpener | RelicId::OrnamentalFan | RelicId::Shuriken | RelicId::VelvetChoker => {
                    self.player.relics[i].counter = 0
                }
                RelicId::Lantern => {
                    if self.player.relics[i].flag(FLAG_A) {
                        self.push_top(Action::GainEnergy { amount: 1 });
                        self.player.relics[i].set_flag(FLAG_A, false);
                    }
                }
                RelicId::MercuryHourglass => {
                    let damage = self.pure_damage_matrix(3);
                    self.push_bot(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Thorns });
                }
                RelicId::Necronomicon | RelicId::UnceasingTop => {
                    let r = &mut self.player.relics[i];
                    r.set_flag(FLAG_A, true);
                    r.set_flag(FLAG_B, false);
                }
                RelicId::OrangePellets => self.player.relics[i].flags = 0,
                RelicId::Orichalcum => self.player.relics[i].set_flag(FLAG_A, false),
                RelicId::StoneCalendar => self.player.relics[i].counter += 1,
                _ => {}
            }
        }
    }

    /// `applyStartOfTurnPostDrawRelics`: `atTurnStartPostDraw`.
    pub(crate) fn relics_at_turn_start_post_draw(&mut self) {
        for i in 0..self.player.relics.len() {
            match self.player.relics[i].id {
                RelicId::GamblingChip => {
                    if !self.player.relics[i].flag(FLAG_A) {
                        self.player.relics[i].set_flag(FLAG_A, true);
                        self.push_bot(Action::GamblingChip);
                    }
                }
                RelicId::WarpedTongs => self.push_bot(Action::UpgradeRandomCard),
                RelicId::Pocketwatch => {
                    let r = &mut self.player.relics[i];
                    if r.counter <= 3 && !r.flag(FLAG_A) {
                        self.push_bot(Action::DrawCard { amount: 3, follow_up: None, shuffle_check: false });
                    } else {
                        self.player.relics[i].set_flag(FLAG_A, false);
                    }
                    self.player.relics[i].counter = 0;
                }
                _ => {}
            }
        }
    }

    /// `applyEndOfTurnRelics`: `onPlayerEndTurn`.
    pub(crate) fn relics_on_player_end_turn(&mut self) {
        for i in 0..self.player.relics.len() {
            match self.player.relics[i].id {
                RelicId::Orichalcum => {
                    if self.player.block == 0 || self.player.relics[i].flag(FLAG_A) {
                        self.player.relics[i].set_flag(FLAG_A, false);
                        self.push_top(Action::GainBlock { target: Cr::Player, amount: 6 });
                    }
                }
                RelicId::NilrysCodex => self.push_bot(Action::Codex),
                RelicId::CloakClasp => {
                    if !self.hand.is_empty() {
                        self.push_bot(Action::GainBlock { target: Cr::Player, amount: self.hand.len() as i32 });
                    }
                }
                RelicId::FrozenCore => {
                    if self.has_empty_orb() {
                        self.channel_orb(super::Orb::new(super::OrbId::Frost));
                    }
                }
                RelicId::StoneCalendar => {
                    if self.player.relics[i].counter == 7 {
                        let damage = self.pure_damage_matrix(52);
                        self.push_bot(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Thorns });
                    }
                }
                _ => {}
            }
        }
    }

    /// `onPlayCard` (from the card queue, before the card is used).
    pub(crate) fn relics_on_play_card(&mut self) {
        for r in self.player.relics.iter_mut() {
            match r.id {
                RelicId::Pocketwatch => r.counter += 1,
                RelicId::VelvetChoker => {
                    if r.counter < 6 {
                        r.counter += 1;
                    }
                }
                _ => {}
            }
        }
    }

    /// `onUseCard` (the `UseCardAction` constructor).
    pub(crate) fn relics_on_use_card(&mut self, card: &Card, target: Option<usize>, exhaust_card: &mut bool) {
        let t = card.card_type();
        let p = Some(Cr::Player);
        for i in 0..self.player.relics.len() {
            let id = self.player.relics[i].id;
            match id {
                RelicId::ArtOfWar => {
                    if t == CardType::Attack {
                        self.player.relics[i].set_flag(FLAG_B, false);
                    }
                }
                RelicId::BirdFacedUrn => {
                    if t == CardType::Power {
                        self.push_top(Action::Heal { target: Cr::Player, amount: 2 });
                    }
                }
                RelicId::BlueCandle => {
                    if t == CardType::Curse {
                        self.push_bot(Action::LoseHp { target: Cr::Player, source: p, amount: 1 });
                        *exhaust_card = true;
                    }
                }
                RelicId::MedicalKit => {
                    if t == CardType::Status {
                        *exhaust_card = true;
                    }
                }
                RelicId::InkBottle => {
                    let r = &mut self.player.relics[i];
                    r.counter += 1;
                    if r.counter == 10 {
                        r.counter = 0;
                        self.push_bot(Action::DrawCard { amount: 1, follow_up: None, shuffle_check: false });
                    }
                }
                RelicId::Kunai | RelicId::Shuriken | RelicId::OrnamentalFan | RelicId::LetterOpener => {
                    let want = if id == RelicId::LetterOpener { CardType::Skill } else { CardType::Attack };
                    if t == want {
                        let r = &mut self.player.relics[i];
                        r.counter += 1;
                        if r.counter % 3 == 0 {
                            r.counter = 0;
                            match id {
                                RelicId::Kunai => self.bot_power(Cr::Player, p, Power::new(PowerId::Dexterity, 1)),
                                RelicId::Shuriken => self.bot_power(Cr::Player, p, Power::new(PowerId::Strength, 1)),
                                RelicId::OrnamentalFan => self.push_bot(Action::GainBlock { target: Cr::Player, amount: 4 }),
                                _ => {
                                    let damage = self.pure_damage_matrix(5);
                                    self.push_bot(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Thorns });
                                }
                            }
                        }
                    }
                }
                RelicId::MummifiedHand => {
                    if t == CardType::Power {
                        let queued: Vec<u32> = self.card_queue.iter().filter_map(|q| q.card.instance()).collect();
                        let cands: Vec<usize> = (0..self.hand.len())
                            .filter(|&h| {
                                let c = &self.hand[h];
                                c.cost > 0 && c.cost_for_turn > 0 && !c.free_to_play_once && !queued.contains(&c.instance)
                            })
                            .collect();
                        if !cands.is_empty() {
                            let k = self.rng.card_random.random_range(0, cands.len() as i32 - 1) as usize;
                            self.hand[cands[k]].set_cost_for_turn(0);
                        }
                    }
                }
                RelicId::Necronomicon => {
                    let big = (card.cost_for_turn >= 2 && !card.free_to_play_once) || (card.cost == -1 && card.energy_on_use >= 2);
                    if t == CardType::Attack && big && self.player.relics[i].flag(FLAG_A) {
                        self.player.relics[i].set_flag(FLAG_A, false);
                        let mut tmp = self.same_instance_copy(card);
                        // (`tmp.applyPowers()`: counted now, kept for its play)
                        self.apply_powers_kept(&mut tmp);
                        self.apply_powers_view(&mut tmp);
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
                }
                RelicId::Nunchaku => {
                    if t == CardType::Attack {
                        let r = &mut self.player.relics[i];
                        r.counter += 1;
                        if r.counter % 10 == 0 {
                            r.counter = 0;
                            self.push_bot(Action::GainEnergy { amount: 1 });
                        }
                    }
                }
                RelicId::OrangePellets => {
                    let r = &mut self.player.relics[i];
                    match t {
                        CardType::Attack => r.set_flag(FLAG_A, true),
                        CardType::Skill => r.set_flag(FLAG_B, true),
                        CardType::Power => r.set_flag(FLAG_C, true),
                        _ => {}
                    }
                    if r.flags & (FLAG_A | FLAG_B | FLAG_C) == FLAG_A | FLAG_B | FLAG_C {
                        r.flags = 0;
                        self.push_bot(Action::RemoveDebuffs { target: Cr::Player });
                    }
                }
                RelicId::PenNib => {
                    if t == CardType::Attack {
                        let r = &mut self.player.relics[i];
                        r.counter += 1;
                        if r.counter == 10 {
                            r.counter = 0;
                        } else if r.counter == 9 {
                            self.bot_power(Cr::Player, p, Power::new(PowerId::PenNib, 1));
                        }
                    }
                }
                // Duality: an attack gives 1 Dexterity for the turn.
                RelicId::Duality => {
                    if t == CardType::Attack {
                        self.bot_power(Cr::Player, p, Power::new(PowerId::Dexterity, 1));
                        self.bot_power(Cr::Player, p, Power::new(PowerId::LoseDexterity, 1));
                    }
                }
                _ => {}
            }
        }
    }

    /// `onExhaust` (before the player's powers).
    pub(crate) fn relics_on_exhaust(&mut self) {
        for id in self.relic_ids() {
            match id {
                RelicId::DeadBranch => {
                    if !self.monsters_basically_dead() {
                        let cid = self.truly_random_any_card_in_combat();
                        let c = self.new_card(cid);
                        self.push_bot(Action::MakeTempCardInHand { card: c, amount: 1 });
                    }
                }
                RelicId::CharonsAshes => {
                    let damage = self.pure_damage_matrix(3);
                    self.push_top(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Thorns });
                }
                _ => {}
            }
        }
    }

    /// `onShuffle` (the `EmptyDeckShuffleAction` constructor).
    pub(crate) fn relics_on_shuffle(&mut self) {
        for i in 0..self.player.relics.len() {
            match self.player.relics[i].id {
                RelicId::Abacus => self.push_bot(Action::GainBlock { target: Cr::Player, amount: 6 }),
                RelicId::Melange => {
                    let a = self.scry(3);
                    self.push_bot(a);
                }
                RelicId::Sundial => {
                    let r = &mut self.player.relics[i];
                    r.counter += 1;
                    if r.counter == 3 {
                        r.counter = 0;
                        self.push_bot(Action::GainEnergy { amount: 2 });
                    }
                }
                _ => {}
            }
        }
    }

    /// `wasHPLost` for the player's relics.
    pub(crate) fn relics_was_hp_lost(&mut self, dmg: i32) {
        if dmg <= 0 {
            return;
        }
        for i in 0..self.player.relics.len() {
            match self.player.relics[i].id {
                RelicId::CentennialPuzzle => {
                    if !self.player.relics[i].flag(FLAG_A) {
                        self.player.relics[i].set_flag(FLAG_A, true);
                        self.push_top(Action::DrawCard { amount: 3, follow_up: None, shuffle_check: false });
                    }
                }
                RelicId::RunicCube => self.push_top(Action::DrawCard { amount: 1, follow_up: None, shuffle_check: false }),
                RelicId::EmotionChip => self.player.relics[i].set_flag(FLAG_A, true),
                RelicId::SelfFormingClay => self.top_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::NextTurnBlock, 3)),
                _ => {}
            }
        }
    }

    pub(crate) fn relics_on_bloodied(&mut self) {
        for i in 0..self.player.relics.len() {
            if self.player.relics[i].id == RelicId::RedSkull && !self.player.relics[i].flag(FLAG_A) {
                self.top_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::Strength, 3));
                self.player.relics[i].set_flag(FLAG_A, true);
            }
        }
    }

    pub(crate) fn relics_on_not_bloodied(&mut self) {
        for i in 0..self.player.relics.len() {
            if self.player.relics[i].id == RelicId::RedSkull {
                if self.player.relics[i].flag(FLAG_A) {
                    self.top_power(Cr::Player, Some(Cr::Player), Power::new(PowerId::Strength, -3));
                }
                self.player.relics[i].set_flag(FLAG_A, false);
            }
        }
    }

    /// `onPlayerHeal` (in combat).
    pub(crate) fn relics_on_player_heal(&self, mut amount: i32) -> i32 {
        for r in &self.player.relics {
            match r.id {
                RelicId::MagicFlower => amount = crate::run::map::java_round(amount as f32 * 1.5),
                RelicId::MarkOfTheBloom => amount = 0,
                _ => {}
            }
        }
        amount
    }

    /// `onPlayerGainedBlock`.
    pub(crate) fn relics_on_player_gained_block(&self, mut tmp: f32) -> f32 {
        for r in &self.player.relics {
            if r.id == RelicId::Orichalcum {
                tmp = super::engine::java_floor(tmp) as f32;
            }
        }
        tmp
    }

    /// `onBlockBroken` for a monster whose block was broken.
    pub(crate) fn relics_on_block_broken(&mut self, m: usize) {
        if self.has_relic(RelicId::HandDrill) {
            self.bot_power(Cr::Monster(m), Some(Cr::Player), Power::vulnerable(2, false, false));
        }
    }

    /// `onMonsterDeath`.
    /// `onSpawnMonster`: Philosopher's Stone gives the newcomer 1 Strength.
    pub(crate) fn relics_on_spawn_monster(&mut self, m: usize) {
        if self.has_relic(RelicId::PhilosopherStone) {
            self.add_power_direct(Cr::Monster(m), Power::new(PowerId::Strength, 1));
            self.on_modify_power();
        }
    }

    pub(crate) fn relics_on_monster_death(&mut self, m: usize) {
        for id in self.relic_ids() {
            match id {
                RelicId::GremlinHorn => {
                    if self.monsters[m].hp == 0 && !self.monsters_basically_dead() {
                        self.push_bot(Action::GainEnergy { amount: 1 });
                        self.push_bot(Action::DrawCard { amount: 1, follow_up: None, shuffle_check: false });
                    }
                }
                RelicId::TheSpecimen => {
                    let poison = self.monsters[m].power_amount(PowerId::Poison);
                    if self.monsters[m].has_power(PowerId::Poison) && !self.monsters_basically_dead() {
                        self.push_bot(Action::PoisonRandomEnemy { amount: poison });
                    }
                }
                _ => {}
            }
        }
    }

    /// `onManualDiscard` (from `GameActionManager.incrementDiscard`).
    pub(crate) fn relics_on_manual_discard(&mut self) {
        for i in 0..self.player.relics.len() {
            match self.player.relics[i].id {
                RelicId::Tingsha => {
                    let info = DamageInfo::new(Some(Cr::Player), 3, DamageType::Thorns);
                    self.push_bot(Action::DamageRandomEnemy { info });
                }
                RelicId::ToughBandages => self.push_bot(Action::GainBlock { target: Cr::Player, amount: 3 }),
                RelicId::HoveringKite => {
                    if !self.player.relics[i].flag(FLAG_A) {
                        self.player.relics[i].set_flag(FLAG_A, true);
                        self.push_bot(Action::GainEnergy { amount: 1 });
                    }
                }
                _ => {}
            }
        }
    }

    /// `onAttackToChangeDamage` when the player attacks (Boot).
    pub(crate) fn relics_on_attack_to_change_damage(&self, info: &DamageInfo, dmg: i32) -> i32 {
        if self.has_relic(RelicId::Boot)
            && info.owner.is_some()
            && info.damage_type != DamageType::HpLoss
            && info.damage_type != DamageType::Thorns
            && dmg > 0
            && dmg < 5
        {
            return 5;
        }
        dmg
    }

    /// `onAttacked` for the player's relics (Torii).
    pub(crate) fn relics_on_attacked(&self, info: &DamageInfo, dmg: i32) -> i32 {
        if self.has_relic(RelicId::Torii)
            && info.owner.is_some()
            && info.damage_type != DamageType::HpLoss
            && info.damage_type != DamageType::Thorns
            && dmg > 1
            && dmg <= 5
        {
            return 1;
        }
        dmg
    }

    /// `onLoseHpLast` (Tungsten Rod).
    pub(crate) fn relics_on_lose_hp_last(&self, dmg: i32) -> i32 {
        if self.has_relic(RelicId::TungstenRod) && dmg > 0 {
            return dmg - 1;
        }
        dmg
    }

    /// `atDamageModify` (Strike Dummy), applied to a card's base damage before powers.
    pub(crate) fn relics_at_damage_modify(&self, mut dmg: f32, card: &Card) -> f32 {
        if self.has_relic(RelicId::StrikeDummy) && card.data().strike {
            dmg += 3.0;
        }
        // Wrist Blade: cards that cost 0 this turn (or are free, unless X-cost).
        if self.has_relic(RelicId::WristBlade) && (card.cost_for_turn == 0 || (card.free_to_play_once && card.cost != -1)) {
            dmg += 4.0;
        }
        dmg
    }

    /// `canPlay` (Velvet Choker).
    pub(crate) fn relics_can_play(&self) -> bool {
        !self.player.relics.iter().any(|r| r.id == RelicId::VelvetChoker && r.counter >= 6)
    }

    /// `onRefreshHand` (Unceasing Top), checked when the game goes idle.
    pub(crate) fn unceasing_top_draw(&mut self) -> bool {
        let Some(r) = self.player.relics.iter().find(|r| r.id == RelicId::UnceasingTop) else { return false };
        if self.actions.is_empty()
            && self.hand.is_empty()
            && !self.turn_has_ended
            && r.flag(FLAG_A)
            && !r.flag(FLAG_B)
            && !self.player.has_power(PowerId::NoDraw)
            && self.pending.is_none()
            && !self.monsters_basically_dead()
            && (!self.discard_pile.is_empty() || !self.draw_pile.is_empty())
        {
            self.push_bot(Action::DrawCard { amount: 1, follow_up: None, shuffle_check: false });
            return true;
        }
        false
    }

    /// `AbstractRoom.endBattle()`: Meat on the Bone, then `AbstractPlayer.onVictory` relics.
    pub(crate) fn relics_on_victory(&mut self) {
        if self.has_relic(RelicId::MeatOnTheBone) && self.player.hp as f32 <= self.player.max_hp as f32 / 2.0 && self.player.hp > 0 {
            self.heal(Cr::Player, 12);
        }
        for i in 0..self.player.relics.len() {
            let id = self.player.relics[i].id;
            match id {
                RelicId::BurningBlood | RelicId::BlackBlood => {
                    if self.player.hp > 0 {
                        self.heal(Cr::Player, if id == RelicId::BurningBlood { 6 } else { 12 });
                    }
                }
                RelicId::FaceOfCleric => self.increase_max_hp(1),
                RelicId::EmotionChip => self.player.relics[i].set_flag(FLAG_A, false),
                RelicId::CaptainsWheel | RelicId::HornCleat => {
                    let r = &mut self.player.relics[i];
                    r.counter = -1;
                    r.used_up = false;
                }
                RelicId::Kunai
                | RelicId::LetterOpener
                | RelicId::OrnamentalFan
                | RelicId::Shuriken
                | RelicId::StoneCalendar
                | RelicId::Pocketwatch
                | RelicId::VelvetChoker => self.player.relics[i].counter = -1,
                RelicId::RedSkull => self.player.relics[i].set_flag(FLAG_A, false),
                _ => {}
            }
        }
    }

    /// Lizard Tail when the player would die. Returns true if it saved them.
    pub(crate) fn relics_lizard_tail(&mut self) -> bool {
        let Some(i) = self.player.relics.iter().position(|r| r.id == RelicId::LizardTail && r.counter == -1) else { return false };
        self.player.hp = 0;
        let heal = (self.player.max_hp / 2).max(1);
        self.heal(Cr::Player, heal);
        let r = &mut self.player.relics[i];
        r.counter = -2;
        r.used_up = true;
        true
    }
}
