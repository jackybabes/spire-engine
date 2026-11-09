//! Potions in combat (`AbstractPotion.use` for the Ironclad pool) and Fairy in a Bottle.

use super::{ChoiceKind, ChoicePile, Combat, PendingChoice};
use crate::action::{Action, Cr, DamageInfo, DamageSource, DamageType};
use crate::cards::{Card, CardType};
use crate::monsters::EnemyType;
use crate::potions::PotionId;
use crate::powers::{Power, PowerId};
use crate::relics::RelicId;

impl Combat {
    /// Potency with Sacred Bark's doubling.
    pub(crate) fn potency(&self, p: PotionId) -> i32 {
        let base = p.data().potency;
        if self.has_relic(RelicId::SacredBark) {
            base * 2
        } else {
            base
        }
    }

    /// `AbstractPotion.canUse()` in combat.
    pub fn can_use_potion(&self, p: PotionId) -> bool {
        if self.monsters_basically_dead() || self.turn_has_ended || self.pending.is_some() || self.outcome.is_some() {
            return false;
        }
        match p {
            PotionId::FairyPotion => false,
            // Not from a boss, nor while any monster has Back Attack (Spire Shield and Spear: the
            // game's SmokeBomb.canUse; the engine missed it, so a network could escape the Act 4
            // elite, which the game refuses: soak 2026-10-27).
            PotionId::SmokeBomb => !self
                .order
                .iter()
                .any(|&m| self.monsters[m].enemy_type == EnemyType::Boss || self.monsters[m].has_power(PowerId::BackAttack)),
            _ => true,
        }
    }

    /// Uses the potion in `slot` (PotionPopUp): `use`, relics' `onUsePotion`, then the slot empties.
    pub(crate) fn use_potion(&mut self, slot: usize, target: Option<usize>) {
        let Some(p) = self.potions.get(slot).copied().flatten() else { return };
        let n = self.potency(p);
        let me = Some(Cr::Player);
        let pl = Cr::Player;
        use PotionId::*;
        match p {
            BloodPotion => {
                let amount = (self.player.max_hp as f32 * (n as f32 / 100.0)) as i32;
                self.push_bot(Action::Heal { target: pl, amount });
            }
            Elixir => self.push_bot(Action::Exhaust { amount: 99, random: false, any_number: true, can_pick_zero: true }),
            HeartOfIron => self.bot_power(pl, me, Power::new(PowerId::Metallicize, n)),
            BlockPotion => self.push_bot(Action::GainBlock { target: pl, amount: n }),
            DexterityPotion => self.bot_power(pl, me, Power::new(PowerId::Dexterity, n)),
            EnergyPotion => self.push_bot(Action::GainEnergy { amount: n }),
            ExplosivePotion => {
                let damage = self.pure_damage_matrix(n);
                self.push_bot(Action::DamageAllEnemies { source: None, damage, damage_type: DamageType::Normal });
            }
            FirePotion => {
                if let Some(m) = target {
                    // applyEnemyPowersOnly: THORNS damage, so only final-receive hooks apply.
                    let info = DamageInfo::new(me, n, DamageType::Thorns);
                    self.push_bot(Action::Damage { target: Cr::Monster(m), source: DamageSource::Fixed(info), steal_gold: 0 });
                }
            }
            StrengthPotion => self.bot_power(pl, me, Power::new(PowerId::Strength, n)),
            SwiftPotion => self.push_bot(Action::DrawCard { amount: n, follow_up: None, shuffle_check: false }),
            WeakenPotion => {
                if let Some(m) = target {
                    self.bot_power(Cr::Monster(m), me, Power::weak(n, false));
                }
            }
            FearPotion => {
                if let Some(m) = target {
                    self.bot_power(Cr::Monster(m), me, Power::vulnerable(n, false, false));
                }
            }
            AttackPotion => self.push_bot(Action::Discovery { card_type: Some(CardType::Attack), colorless: false, amount: n }),
            SkillPotion => self.push_bot(Action::Discovery { card_type: Some(CardType::Skill), colorless: false, amount: n }),
            PowerPotion => self.push_bot(Action::Discovery { card_type: Some(CardType::Power), colorless: false, amount: n }),
            ColorlessPotion => self.push_bot(Action::Discovery { card_type: None, colorless: true, amount: n }),
            SteroidPotion => {
                self.bot_power(pl, me, Power::new(PowerId::Strength, n));
                self.bot_power(pl, me, Power::new(PowerId::Flex, n));
            }
            SpeedPotion => {
                self.bot_power(pl, me, Power::new(PowerId::Dexterity, n));
                self.bot_power(pl, me, Power::new(PowerId::LoseDexterity, n));
            }
            BlessingOfTheForge => self.push_bot(Action::Armaments { upgraded: true }),
            RegenPotion => self.bot_power(pl, me, Power::new(PowerId::Regen, n)),
            AncientPotion => self.bot_power(pl, me, Power::new(PowerId::Artifact, n)),
            LiquidBronze => self.bot_power(pl, me, Power::new(PowerId::Thorns, n)),
            GamblersBrew => {
                if !self.hand.is_empty() {
                    self.push_bot(Action::GamblingChip);
                }
            }
            EssenceOfSteel => self.bot_power(pl, me, Power::new(PowerId::PlatedArmor, n)),
            DuplicationPotion => self.bot_power(pl, me, Power::new(PowerId::Duplication, n)),
            DistilledChaosPotion => {
                for _ in 0..n {
                    let t = self.get_random_monster(None, true, true);
                    self.push_bot(Action::PlayTopCard { target: t, exhaust: false });
                }
            }
            LiquidMemories => self.push_bot(Action::DiscardToHand { amount: n, set_cost: true }),
            CultistPotion => self.bot_power(pl, me, Power::ritual(n, true)),
            FruitJuice => self.increase_max_hp(n),
            SneckoOil => {
                self.push_bot(Action::DrawCard { amount: n, follow_up: None, shuffle_check: false });
                self.push_bot(Action::RandomizeHandCost);
            }
            SmokeBomb => self.smoked = true,
            EntropicBrew => {
                for _ in 0..self.potions.len() {
                    self.push_bot(Action::ObtainRandomPotion);
                }
            }
            // The Silent's.
            PoisonPotion => {
                if let Some(m) = target {
                    self.bot_power(Cr::Monster(m), me, Power::new(PowerId::Poison, n));
                }
            }
            CunningPotion => {
                let mut shiv = self.new_card(crate::cards::CardId::Shiv);
                shiv.upgrade();
                self.push_bot(Action::MakeTempCardInHand { card: shiv, amount: n });
            }
            GhostInAJar => self.bot_power(pl, me, Power::new(PowerId::IntangiblePlayer, n)),
            // The Defect's.
            FocusPotion => self.bot_power(pl, me, Power::new(PowerId::Focus, n)),
            PotionOfCapacity => self.push_bot(Action::IncreaseMaxOrb { amount: n }),
            EssenceOfDarkness => self.push_bot(Action::EssenceOfDarkness { amount: n }),
            // The Watcher's.
            Ambrosia => self.push_bot(Action::ChangeStance { stance: super::Stance::Divinity }),
            BottledMiracle => {
                let card = self.new_card(crate::cards::CardId::Miracle);
                self.push_bot(Action::MakeTempCardInHand { card, amount: n });
            }
            StancePotion => {
                let wrath = self.new_card(crate::cards::CardId::ChooseWrath);
                let calm = self.new_card(crate::cards::CardId::ChooseCalm);
                self.push_bot(Action::ChooseOne { options: vec![wrath, calm] });
            }
            _ => {}
        }
        if self.has_relic(RelicId::ToyOrnithopter) {
            self.push_bot(Action::Heal { target: pl, amount: 5 });
        }
        self.potions[slot] = None;
    }

    /// Fairy in a Bottle when the player would die: heal and keep going.
    pub(crate) fn fairy_revive(&mut self) -> bool {
        let Some(slot) = self.potions.iter().position(|p| *p == Some(PotionId::FairyPotion)) else { return false };
        self.player.hp = 0;
        let pct = self.potency(PotionId::FairyPotion) as f32 / 100.0;
        let amt = ((self.player.max_hp as f32 * pct) as i32).max(1);
        self.heal(Cr::Player, amt);
        self.potions[slot] = None;
        true
    }

    /// `DiscoveryAction`: three distinct random cards of a type; the pick costs 0 this turn.
    /// The game regenerates the choices on every frame the action updates, so its
    /// cardRandomRng use depends on frame timing; this generates them once.
    pub(crate) fn exec_discovery(&mut self, card_type: Option<CardType>, colorless: bool, amount: i32) {
        let mut offer: Vec<Card> = Vec::new();
        while offer.len() != 3 {
            let id = if colorless {
                self.truly_random_colorless_in_combat()
            } else {
                match card_type {
                    Some(t) => self.truly_random_card_in_combat(t),
                    None => self.truly_random_any_card_in_combat(),
                }
            };
            if !offer.iter().any(|c| c.id == id) {
                let c = self.new_card(id);
                offer.push(c);
            }
        }
        self.pending = Some(PendingChoice {
            kind: ChoiceKind::Discovery { amount },
            pile: ChoicePile::Offer,
            count: 1,
            any_number: false,
            can_pick_zero: false,
            options: (0..offer.len()).collect(),
            offer,
        });
    }
}
