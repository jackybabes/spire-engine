//! `AbstractCard.use()` for every implemented card: each pushes its actions.

use super::Combat;
use crate::action::{Action, Cr, DamageInfo, DamageSource, DamageType};
use crate::cards::{Card, CardId, CardRarity, CardType};
use super::{OrbId, Stance};
use crate::character::Character;
use crate::powers::{Power, PowerId};

/// The cards `returnTrulyRandomCardInCombat(type)` picks from, in the game's order (the
/// character's all-unlocked pool, no healing cards).
pub(crate) fn truly_random_in_combat_list(character: Character, t: CardType) -> Vec<CardId> {
    let mut list = Vec::new();
    for rarity in [CardRarity::Common, CardRarity::Uncommon, CardRarity::Rare] {
        // srcXCardPool is the rarity pool in reverse library order.
        for &id in character.library_order().iter().rev() {
            let d = id.data();
            if d.rarity == rarity && d.card_type == t && !d.healing {
                list.push(id);
            }
        }
    }
    list
}

impl Combat {
    fn hit(&mut self, target: Option<usize>, damage: i32) {
        if let Some(m) = target {
            let info = DamageInfo::new(Some(Cr::Player), damage, DamageType::Normal);
            self.push_bot(Action::Damage { target: Cr::Monster(m), source: DamageSource::Fixed(info), steal_gold: 0 });
        }
    }

    fn block_self(&mut self, amount: i32) {
        self.push_bot(Action::GainBlock { target: Cr::Player, amount });
    }

    fn draw(&mut self, amount: i32) {
        self.push_bot(Action::DrawCard { amount, follow_up: None, shuffle_check: false });
    }

    /// `DiscardAction(p, p, amount, random)`.
    fn discard(&mut self, amount: i32, random: bool) {
        self.push_bot(Action::Discard { amount, random, end_turn: false });
    }

    /// `ChannelAction(new orb)`.
    fn channel(&mut self, id: super::OrbId) {
        self.push_bot(Action::Channel { orb: super::Orb::new(id), auto_evoke: true });
    }

    /// `AbstractOrb.getRandomOrb(true)`.
    fn random_orb(&mut self) -> super::OrbId {
        let k = self.rng.card_random.random(super::orbs::RANDOM_ORBS.len() as i32 - 1);
        super::orbs::RANDOM_ORBS[k as usize]
    }

    /// `MakeTempCardInHandAction(new Shiv(), n)`.
    fn shivs(&mut self, n: i32) {
        let card = self.new_card(CardId::Shiv);
        self.push_bot(Action::MakeTempCardInHand { card, amount: n });
    }

    fn info(dmg: i32) -> DamageInfo {
        DamageInfo::new(Some(Cr::Player), dmg, DamageType::Normal)
    }

    fn self_power(&mut self, power: Power) {
        self.bot_power(Cr::Player, Some(Cr::Player), power);
    }

    fn enemy_power(&mut self, target: Option<usize>, power: Power) {
        if let Some(m) = target {
            self.bot_power(Cr::Monster(m), Some(Cr::Player), power);
        }
    }

    /// `AbstractDungeon.returnTrulyRandomCardInCombat(type)` with the character's all-unlocked pool.
    pub(crate) fn truly_random_card_in_combat(&mut self, t: CardType) -> CardId {
        let list = truly_random_in_combat_list(self.character, t);
        let k = self.rng.card_random.random(list.len() as i32 - 1);
        list[k as usize]
    }

    pub(crate) fn card_use(&mut self, card: &mut Card, m: Option<usize>, multi: Vec<i32>) {
        use CardId::*;
        let dmg = card.damage;
        let blk = card.block;
        let magic = card.magic;
        let turn_has_ended = self.turn_has_ended;
        match card.id {
            Anger => {
                self.hit(m, dmg);
                let copy = self.stat_copy(card);
                self.push_bot(Action::MakeTempCardInDiscard { card: copy, amount: 1 });
            }
            Armaments => {
                self.block_self(blk);
                self.push_bot(Action::Armaments { upgraded: card.upgraded });
            }
            Barricade => {
                if !self.player.has_power(PowerId::Barricade) {
                    self.self_power(Power::flag(PowerId::Barricade));
                }
            }
            Bash => {
                self.hit(m, dmg);
                self.enemy_power(m, Power::vulnerable(magic, false, turn_has_ended));
            }
            BattleTrance => {
                self.draw(magic);
                self.self_power(Power::flag(PowerId::NoDraw));
            }
            Berserk => {
                self.self_power(Power::vulnerable(magic, false, turn_has_ended));
                self.self_power(Power::new(PowerId::Berserk, 1));
            }
            BloodForBlood | Bludgeon | BodySlam | Carnage | Clash | HeavyBlade | PerfectedStrike | SearingBlow | StrikeR => {
                self.hit(m, dmg)
            }
            Bloodletting => {
                self.push_bot(Action::LoseHp { target: Cr::Player, source: Some(Cr::Player), amount: 3 });
                self.push_bot(Action::GainEnergy { amount: magic });
            }
            Brutality => self.self_power(Power::new(PowerId::Brutality, 1)),
            BurningPact => {
                self.push_bot(Action::Exhaust { amount: 1, random: false, any_number: false, can_pick_zero: false });
                self.draw(magic);
            }
            Cleave => self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal }),
            Clothesline => {
                self.hit(m, dmg);
                self.enemy_power(m, Power::weak(magic, false));
            }
            Combust => {
                let a = self.apply_power_action(Cr::Player, Some(Cr::Player), Power::combust(1, magic), magic);
                self.push_bot(a);
            }
            Corruption => {
                if !self.player.has_power(PowerId::Corruption) {
                    self.self_power(Power::flag(PowerId::Corruption));
                }
            }
            DarkEmbrace => self.self_power(Power::new(PowerId::DarkEmbrace, 1)),
            DefendR | GhostlyArmor | Impervious | Sentinel => self.block_self(blk),
            DemonForm => self.self_power(Power::new(PowerId::DemonForm, magic)),
            Disarm => self.enemy_power(m, Power::new(PowerId::Strength, -magic)),
            DoubleTap => self.self_power(Power::new(PowerId::DoubleTap, magic)),
            Dropkick => {
                if let Some(t) = m {
                    let info = DamageInfo::new(Some(Cr::Player), dmg, DamageType::Normal);
                    self.push_bot(Action::Dropkick { target: t, info });
                }
            }
            DualWield => self.push_bot(Action::DualWield { amount: magic }),
            Entrench => self.push_bot(Action::DoubleYourBlock { target: Cr::Player }),
            Evolve => self.self_power(Power::new(PowerId::Evolve, magic)),
            Exhume => self.push_bot(Action::Exhume { upgrade: false }),
            Feed => {
                if let Some(t) = m {
                    let info = DamageInfo::new(Some(Cr::Player), dmg, DamageType::Normal);
                    self.push_bot(Action::Feed { target: t, info, max_hp: magic });
                }
            }
            FeelNoPain => self.self_power(Power::new(PowerId::FeelNoPain, magic)),
            // Event cards.
            JAX => {
                self.push_bot(Action::LoseHp { target: Cr::Player, source: Some(Cr::Player), amount: 3 });
                self.self_power(Power::new(PowerId::Strength, magic));
            }
            Apparition => self.self_power(Power::new(PowerId::IntangiblePlayer, 1)),
            Bite => {
                if let Some(t) = m {
                    let info = DamageInfo::new(Some(Cr::Player), dmg, DamageType::Normal);
                    self.push_bot(Action::Damage { target: Cr::Monster(t), source: crate::action::DamageSource::Fixed(info), steal_gold: 0 });
                }
                self.push_bot(Action::Heal { target: Cr::Player, amount: magic });
            }
            RitualDagger => {
                if let Some(t) = m {
                    let info = DamageInfo::new(Some(Cr::Player), dmg, DamageType::Normal);
                    self.push_bot(Action::RitualDagger { target: t, info, amount: magic, uuid: card.uuid });
                }
            }
            FiendFire => {
                if let Some(t) = m {
                    let info = DamageInfo::new(Some(Cr::Player), dmg, DamageType::Normal);
                    self.push_bot(Action::FiendFire { target: t, info });
                }
            }
            FireBreathing => self.self_power(Power::new(PowerId::FireBreathing, magic)),
            FlameBarrier => {
                self.block_self(blk);
                self.self_power(Power::new(PowerId::FlameBarrier, magic));
            }
            Flex => {
                self.self_power(Power::new(PowerId::Strength, magic));
                self.self_power(Power::new(PowerId::Flex, magic));
            }
            Havoc => {
                let target = self.get_random_monster(None, true, true);
                self.push_bot(Action::PlayTopCard { target, exhaust: true });
            }
            Headbutt => {
                self.hit(m, dmg);
                self.push_bot(Action::DiscardPileToTopOfDeck);
            }
            Hemokinesis => {
                self.push_bot(Action::LoseHp { target: Cr::Player, source: Some(Cr::Player), amount: magic });
                self.hit(m, dmg);
            }
            Immolate => {
                self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal });
                let burn = self.new_card(Burn);
                self.push_bot(Action::MakeTempCardInDiscard { card: burn, amount: 1 });
            }
            InfernalBlade => {
                let id = self.truly_random_card_in_combat(CardType::Attack);
                let mut c = self.new_card(id);
                c.set_cost_for_turn(0);
                self.push_bot(Action::MakeTempCardInHand { card: c, amount: 1 });
            }
            Inflame => self.self_power(Power::new(PowerId::Strength, magic)),
            Intimidate => {
                for mo in self.order.clone() {
                    self.bot_power(Cr::Monster(mo), Some(Cr::Player), Power::weak(magic, false));
                }
            }
            IronWave => {
                self.block_self(blk);
                self.hit(m, dmg);
            }
            Juggernaut => self.self_power(Power::new(PowerId::Juggernaut, magic)),
            LimitBreak => self.push_bot(Action::LimitBreak),
            Metallicize => self.self_power(Power::new(PowerId::Metallicize, magic)),
            Offering => {
                self.push_bot(Action::LoseHp { target: Cr::Player, source: Some(Cr::Player), amount: 6 });
                self.push_bot(Action::GainEnergy { amount: 2 });
                self.draw(magic);
            }
            PommelStrike => {
                self.hit(m, dmg);
                self.draw(magic);
            }
            PowerThrough => {
                let wound = self.new_card(Wound);
                self.push_bot(Action::MakeTempCardInHand { card: wound, amount: 2 });
                self.block_self(blk);
            }
            Pummel => {
                if let Some(t) = m {
                    let info = DamageInfo::new(Some(Cr::Player), dmg, DamageType::Normal);
                    for _ in 1..magic {
                        self.push_bot(Action::PummelDamage { target: t, info });
                    }
                }
                self.hit(m, dmg);
            }
            Rage => self.self_power(Power::new(PowerId::Rage, magic)),
            Rampage => {
                self.hit(m, dmg);
                self.push_bot(Action::ModifyDamage { uuid: card.uuid, amount: magic });
            }
            Reaper => self.push_bot(Action::VampireDamageAllEnemies { damage: multi }),
            RecklessCharge => {
                self.hit(m, dmg);
                let dazed = self.new_card(Dazed);
                self.push_bot(Action::MakeTempCardInDrawPile { card: dazed, amount: 1, random_spot: true, to_bottom: false });
            }
            Rupture => self.self_power(Power::new(PowerId::Rupture, magic)),
            SecondWind => self.push_bot(Action::BlockPerNonAttack { amount: blk }),
            SeeingRed => self.push_bot(Action::GainEnergy { amount: 2 }),
            SeverSoul => {
                self.push_bot(Action::ExhaustAllNonAttack);
                self.hit(m, dmg);
            }
            Shockwave => {
                for mo in self.order.clone() {
                    self.bot_power(Cr::Monster(mo), Some(Cr::Player), Power::weak(magic, false));
                    self.bot_power(Cr::Monster(mo), Some(Cr::Player), Power::vulnerable(magic, false, turn_has_ended));
                }
            }
            ShrugItOff => {
                self.block_self(blk);
                self.draw(1);
            }
            SpotWeakness => self.push_bot(Action::SpotWeakness { amount: magic, target: m }),
            SwordBoomerang => {
                for _ in 0..magic {
                    self.push_bot(Action::AttackDamageRandomEnemy { card: card.clone() });
                }
            }
            ThunderClap => {
                self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal });
                for mo in self.order.clone() {
                    self.bot_power(Cr::Monster(mo), Some(Cr::Player), Power::vulnerable(1, false, turn_has_ended));
                }
            }
            TrueGrit => {
                self.block_self(blk);
                if card.upgraded {
                    self.push_bot(Action::Exhaust { amount: 1, random: false, any_number: false, can_pick_zero: false });
                } else {
                    self.push_bot(Action::Exhaust { amount: 1, random: true, any_number: false, can_pick_zero: false });
                }
            }
            TwinStrike => {
                self.hit(m, dmg);
                self.hit(m, dmg);
            }
            Uppercut => {
                self.hit(m, dmg);
                self.enemy_power(m, Power::weak(magic, false));
                self.enemy_power(m, Power::vulnerable(magic, false, turn_has_ended));
            }
            Warcry => {
                self.draw(magic);
                self.push_bot(Action::PutOnDeck { amount: 1, random: false });
            }
            Whirlwind => self.push_bot(Action::Whirlwind {
                damage: multi,
                free_to_play_once: card.free_to_play_once,
                energy_on_use: card.energy_on_use,
            }),
            WildStrike => {
                self.hit(m, dmg);
                let wound = self.new_card(Wound);
                self.push_bot(Action::MakeTempCardInDrawPile { card: wound, amount: 1, random_spot: true, to_bottom: false });
            }
            Burn => {
                if card.dont_trigger_on_use {
                    let info = DamageInfo::new(Some(Cr::Player), magic, DamageType::Thorns);
                    self.push_bot(Action::Damage { target: Cr::Player, source: DamageSource::Fixed(info), steal_gold: 0 });
                }
            }
            Dazed | Slimed | Void | Wound => {}
            // Curses that act when played at the end of the turn.
            Decay => {
                if card.dont_trigger_on_use {
                    let info = DamageInfo::new(Some(Cr::Player), 2, DamageType::Thorns);
                    self.push_top(Action::Damage { target: Cr::Player, source: DamageSource::Fixed(info), steal_gold: 0 });
                }
            }
            Doubt | Shame => {
                if card.dont_trigger_on_use {
                    let p = if card.id == Doubt { Power::weak(1, true) } else { Power::frail(1, true) };
                    self.top_power(Cr::Player, Some(Cr::Player), p);
                }
            }
            Regret => {
                if card.dont_trigger_on_use {
                    self.push_top(Action::LoseHp { target: Cr::Player, source: Some(Cr::Player), amount: magic });
                }
            }
            // Colorless.
            DarkShackles => {
                self.enemy_power(m, Power::new(PowerId::Strength, -magic));
                if let Some(t) = m {
                    if !self.monsters[t].powers.iter().any(|p| p.id == PowerId::Artifact) {
                        self.enemy_power(m, Power::new(PowerId::Shackled, magic));
                    }
                }
            }
            SadisticNature => self.self_power(Power::new(PowerId::Sadistic, magic)),
            PanicButton => {
                self.block_self(blk);
                self.self_power(Power::new(PowerId::NoBlock, magic));
            }
            Trip | Blind => {
                let mk = |n: i32| if card.id == Trip { Power::vulnerable(n, false, turn_has_ended) } else { Power::weak(n, false) };
                if !card.upgraded {
                    self.enemy_power(m, mk(magic));
                } else {
                    for mo in self.order.clone() {
                        self.bot_power(Cr::Monster(mo), Some(Cr::Player), mk(magic));
                    }
                }
            }
            DramaticEntrance => self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal }),
            Impatience => self.push_bot(Action::ConditionalDraw { amount: magic }),
            TheBomb => {
                let mut b = Power::new(PowerId::TheBomb, 3);
                b.hp_loss = magic;
                let a = self.apply_power_action(Cr::Player, Some(Cr::Player), b, 3);
                self.push_bot(a);
            }
            BandageUp => self.push_bot(Action::Heal { target: Cr::Player, amount: magic }),
            SecretTechnique => self.push_bot(Action::FromDeckToHand { card_type: CardType::Skill }),
            SecretWeapon => self.push_bot(Action::FromDeckToHand { card_type: CardType::Attack }),
            DeepBreath => {
                if !self.discard_pile.is_empty() {
                    self.relics_on_shuffle();
                    self.push_bot(Action::EmptyDeckShuffle);
                    self.push_bot(Action::ShuffleDrawPile);
                }
                self.draw(magic);
            }
            Violence => self.push_bot(Action::DrawPileToHand { amount: magic, card_type: CardType::Attack }),
            Panache => {
                let mut pw = Power::new(PowerId::Panache, 5);
                pw.hp_loss = magic;
                let a = self.apply_power_action(Cr::Player, Some(Cr::Player), pw, magic);
                self.push_bot(a);
            }
            Apotheosis => self.push_bot(Action::Apotheosis),
            Mayhem => self.self_power(Power::new(PowerId::Mayhem, magic.max(1))),
            HandOfGreed => {
                if let Some(t) = m {
                    let info = DamageInfo::new(Some(Cr::Player), dmg, DamageType::Normal);
                    self.push_bot(Action::Greed { target: t, info, gold: magic });
                }
            }
            FlashOfSteel => {
                self.hit(m, dmg);
                self.draw(1);
            }
            Forethought => self.push_bot(Action::Forethought { any: card.upgraded }),
            Enlightenment => self.push_bot(Action::Enlightenment { combat: card.upgraded }),
            Purity => self.push_bot(Action::Exhaust { amount: magic, random: false, any_number: true, can_pick_zero: true }),
            Panacea => self.self_power(Power::new(PowerId::Artifact, magic)),
            Transmutation => {
                let energy_on_use = if card.energy_on_use < self.energy { self.energy } else { card.energy_on_use };
                self.push_bot(Action::Transmutation { upgraded: card.upgraded, free: card.free_to_play_once, energy_on_use });
            }
            Chrysalis | Metamorphosis => {
                let t = if card.id == Chrysalis { CardType::Skill } else { CardType::Attack };
                for _ in 0..magic {
                    let id = self.truly_random_card_in_combat(t);
                    let mut c = self.new_card(id);
                    if c.cost > 0 {
                        c.cost = 0;
                        c.cost_for_turn = 0;
                    }
                    self.push_bot(Action::MakeTempCardInDrawPile { card: c, amount: 1, random_spot: true, to_bottom: false });
                }
            }
            Discovery => self.push_bot(Action::Discovery { card_type: None, colorless: false, amount: 1 }),
            Finesse => {
                self.block_self(blk);
                self.draw(1);
            }
            Magnetism => self.self_power(Power::new(PowerId::Magnetism, magic.max(1))),
            MasterOfStrategy => self.draw(magic),
            GoodInstincts => self.block_self(blk),
            SwiftStrike | MindBlast => self.hit(m, dmg),
            JackOfAllTrades => {
                let id = self.truly_random_colorless_in_combat();
                let c = self.new_card(id);
                self.push_bot(Action::MakeTempCardInHand { card: c, amount: 1 });
                if card.upgraded {
                    let id = self.truly_random_colorless_in_combat();
                    let c = self.new_card(id);
                    self.push_bot(Action::MakeTempCardInHand { card: c, amount: 1 });
                }
            }
            ThinkingAhead => {
                self.draw(2);
                if !self.hand.is_empty() {
                    self.push_bot(Action::PutOnDeck { amount: 1, random: false });
                }
            }
            Madness => self.push_bot(Action::Madness),
            // The Silent.
            StrikeG | Slice | Backstab | EndlessAgony | MasterfulStab | Shiv => self.hit(m, dmg),
            DefendG | Deflect => self.block_self(blk),
            Neutralize | SuckerPunch => {
                self.hit(m, dmg);
                self.enemy_power(m, Power::weak(magic, false));
            }
            Survivor => {
                self.block_self(blk);
                self.discard(1, false);
            }
            AThousandCuts => self.self_power(Power::new(PowerId::ThousandCuts, magic)),
            Accuracy => self.self_power(Power::new(PowerId::Accuracy, magic)),
            Acrobatics => {
                self.draw(magic);
                self.discard(1, false);
            }
            Adrenaline => {
                self.push_bot(Action::GainEnergy { amount: if card.upgraded { 2 } else { 1 } });
                self.draw(2);
            }
            AfterImage => self.self_power(Power::new(PowerId::AfterImage, 1)),
            Alchemize => self.push_bot(Action::ObtainRandomPotion),
            AllOutAttack => {
                self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal });
                self.discard(1, true);
            }
            Backflip => {
                self.block_self(blk);
                self.draw(2);
            }
            Bane => {
                self.hit(m, dmg);
                if let Some(t) = m {
                    self.push_bot(Action::Bane { target: t, info: Self::info(dmg) });
                }
            }
            BladeDance => self.shivs(magic),
            Blur => {
                self.block_self(blk);
                self.self_power(Power::new(PowerId::Blur, 1));
            }
            BouncingFlask => {
                let target = self.get_random_monster(None, true, true);
                self.push_bot(Action::BouncingFlask { target, amount: 3, times: magic });
            }
            BulletTime => {
                self.self_power(Power::flag(PowerId::NoDraw));
                self.push_bot(Action::BulletTime);
            }
            Burst => self.self_power(Power::new(PowerId::Burst, magic)),
            CalculatedGamble => self.push_bot(Action::CalculatedGamble),
            Caltrops => self.self_power(Power::new(PowerId::Thorns, magic)),
            Catalyst => {
                if let Some(t) = m {
                    self.push_bot(Action::MultiplyPoison { target: t, extra: if card.upgraded { 2 } else { 1 } });
                }
            }
            Choke => {
                self.hit(m, dmg);
                self.enemy_power(m, Power::new(PowerId::Choked, magic));
            }
            CloakAndDagger => {
                self.block_self(blk);
                self.shivs(magic);
            }
            Concentrate => {
                self.discard(magic, false);
                self.push_bot(Action::GainEnergy { amount: 2 });
            }
            CorpseExplosion => {
                self.enemy_power(m, Power::new(PowerId::Poison, magic));
                self.enemy_power(m, Power::new(PowerId::CorpseExplosion, 1));
            }
            CripplingPoison => {
                if !self.monsters_basically_dead() {
                    for mo in self.order.clone() {
                        if !self.monsters[mo].is_dying {
                            self.bot_power(Cr::Monster(mo), Some(Cr::Player), Power::new(PowerId::Poison, magic));
                            self.bot_power(Cr::Monster(mo), Some(Cr::Player), Power::weak(2, false));
                        }
                    }
                }
            }
            DaggerSpray => {
                self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi.clone(), damage_type: DamageType::Normal });
                self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal });
            }
            DaggerThrow => {
                self.hit(m, dmg);
                self.draw(1);
                self.discard(1, false);
            }
            Dash => {
                self.block_self(blk);
                self.hit(m, dmg);
            }
            DeadlyPoison => self.enemy_power(m, Power::new(PowerId::Poison, magic)),
            DieDieDie | GrandFinale => {
                self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal })
            }
            Distraction => {
                let id = self.truly_random_card_in_combat(CardType::Skill);
                let mut c = self.new_card(id);
                c.set_cost_for_turn(-99);
                self.push_bot(Action::MakeTempCardInHand { card: c, amount: 1 });
            }
            DodgeAndRoll => {
                self.block_self(blk);
                self.self_power(Power::new(PowerId::NextTurnBlock, blk));
            }
            Doppelganger => self.push_bot(Action::Doppelganger { upgraded: card.upgraded, free: card.free_to_play_once, energy_on_use: card.energy_on_use }),
            Envenom => self.self_power(Power::new(PowerId::Envenom, 1)),
            EscapePlan => {
                let follow_up = Some(Box::new(Action::EscapePlan { block: blk }));
                self.push_bot(Action::DrawCard { amount: 1, follow_up, shuffle_check: false });
            }
            Eviscerate => {
                for _ in 0..3 {
                    self.hit(m, dmg);
                }
            }
            Expertise => self.push_bot(Action::Expertise { amount: magic }),
            Finisher => {
                if let Some(t) = m {
                    self.push_bot(Action::DamagePerAttackPlayed { target: t, info: Self::info(dmg) });
                }
            }
            Flechettes => {
                if let Some(t) = m {
                    self.push_bot(Action::Flechette { target: t, info: Self::info(dmg) });
                }
            }
            FlyingKnee => {
                self.hit(m, dmg);
                self.self_power(Power::new(PowerId::Energized, 1));
            }
            Footwork => self.self_power(Power::new(PowerId::Dexterity, magic)),
            GlassKnife => {
                self.hit(m, dmg);
                self.hit(m, dmg);
                self.push_bot(Action::ModifyDamage { uuid: card.uuid, amount: -2 });
            }
            HeelHook => {
                if let Some(t) = m {
                    self.push_bot(Action::HeelHook { target: t, info: Self::info(dmg) });
                }
            }
            InfiniteBlades => self.self_power(Power::new(PowerId::InfiniteBlades, 1)),
            LegSweep => {
                self.enemy_power(m, Power::weak(magic, false));
                self.block_self(blk);
            }
            Malaise => {
                if let Some(t) = m {
                    self.push_bot(Action::Malaise { target: t, upgraded: card.upgraded, free: card.free_to_play_once, energy_on_use: card.energy_on_use });
                }
            }
            Nightmare => self.push_bot(Action::Nightmare { amount: magic }),
            NoxiousFumes => self.self_power(Power::new(PowerId::NoxiousFumes, magic)),
            Outmaneuver => self.self_power(Power::new(PowerId::Energized, if card.upgraded { 3 } else { 2 })),
            PhantasmalKiller => self.self_power(Power::new(PowerId::Phantasmal, 1)),
            PiercingWail => {
                for mo in self.order.clone() {
                    self.bot_power(Cr::Monster(mo), Some(Cr::Player), Power::new(PowerId::Strength, -magic));
                }
                for mo in self.order.clone() {
                    if !self.monsters[mo].has_power(PowerId::Artifact) {
                        self.bot_power(Cr::Monster(mo), Some(Cr::Player), Power::new(PowerId::Shackled, magic));
                    }
                }
            }
            PoisonedStab => {
                self.hit(m, dmg);
                self.enemy_power(m, Power::new(PowerId::Poison, magic));
            }
            Predator => {
                self.hit(m, dmg);
                self.self_power(Power::new(PowerId::DrawCardNextTurn, 2));
            }
            Prepared => {
                self.draw(magic);
                self.discard(magic, false);
            }
            QuickSlash => {
                self.hit(m, dmg);
                self.draw(1);
            }
            // Reflex and Tactician can't be played (`canUse`); Reflex's use() would draw.
            Reflex => self.draw(magic),
            Tactician => {}
            RiddleWithHoles => {
                for _ in 0..5 {
                    self.hit(m, dmg);
                }
            }
            Setup => self.push_bot(Action::Setup),
            Skewer => {
                if let Some(t) = m {
                    self.push_bot(Action::Skewer { target: t, damage: dmg, free: card.free_to_play_once, energy_on_use: card.energy_on_use });
                }
            }
            SneakyStrike => {
                self.hit(m, dmg);
                self.push_bot(Action::GainEnergyIfDiscard { amount: 2 });
            }
            StormOfSteel => self.push_bot(Action::BladeFury { upgraded: card.upgraded }),
            Terror => self.enemy_power(m, Power::vulnerable(99, false, turn_has_ended)),
            ToolsOfTheTrade => self.self_power(Power::new(PowerId::ToolsOfTheTrade, 1)),
            Unload => {
                self.hit(m, dmg);
                self.push_bot(Action::Unload);
            }
            WellLaidPlans => self.self_power(Power::new(PowerId::RetainCards, magic)),
            WraithForm => {
                self.self_power(Power::new(PowerId::IntangiblePlayer, magic));
                self.self_power(Power::new(PowerId::WraithForm, -1));
            }
            // The Defect.
            StrikeB => self.hit(m, dmg),
            DefendB | Leap | BootSequence | ForceField | Stack => self.block_self(blk),
            Zap => {
                for _ in 0..magic {
                    self.channel(OrbId::Lightning);
                }
            }
            Dualcast => {
                self.push_bot(Action::EvokeWithoutRemovingOrb { count: 1 });
                self.push_bot(Action::EvokeOrb { count: 1 });
            }
            Aggregate => self.push_bot(Action::AggregateEnergy { divide: magic }),
            AllForOne => {
                self.hit(m, dmg);
                self.push_bot(Action::AllCostToHand);
            }
            Amplify => self.self_power(Power::new(PowerId::Amplify, magic)),
            AutoShields => {
                if self.player.block == 0 {
                    self.block_self(blk);
                }
            }
            BallLightning | ColdSnap | MeteorStrike => {
                self.hit(m, dmg);
                let orb = match card.id {
                    BallLightning => OrbId::Lightning,
                    ColdSnap => OrbId::Frost,
                    _ => OrbId::Plasma,
                };
                for _ in 0..magic {
                    self.channel(orb);
                }
            }
            Barrage => {
                if let Some(t) = m {
                    self.push_bot(Action::Barrage { target: t, info: Self::info(dmg) });
                }
            }
            BeamCell => {
                self.hit(m, dmg);
                self.enemy_power(m, Power::vulnerable(magic, false, turn_has_ended));
            }
            BiasedCognition => {
                self.self_power(Power::new(PowerId::Focus, magic));
                self.self_power(Power::new(PowerId::Bias, 1));
            }
            Blizzard | Hyperbeam | DoomAndGloom | SweepingBeam => {
                self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal });
                match card.id {
                    Hyperbeam => self.self_power(Power::new(PowerId::Focus, -magic)),
                    DoomAndGloom => self.channel(OrbId::Dark),
                    SweepingBeam => self.draw(magic),
                    _ => {}
                }
            }
            Buffer => self.self_power(Power::new(PowerId::Buffer, magic)),
            Capacitor => self.push_bot(Action::IncreaseMaxOrb { amount: magic }),
            Chaos => {
                if card.upgraded {
                    let o = self.random_orb();
                    self.channel(o);
                }
                let o = self.random_orb();
                self.channel(o);
            }
            Chill => {
                let alive = self.order.iter().filter(|&&mo| !self.monsters[mo].is_dead_or_escaped()).count() as i32;
                for _ in 0..alive * magic {
                    self.channel(OrbId::Frost);
                }
            }
            Claw => {
                self.hit(m, dmg);
                self.push_bot(Action::Gash { amount: magic });
            }
            CompileDriver => {
                self.hit(m, dmg);
                self.push_bot(Action::CompileDriver { amount: magic });
            }
            ConserveBattery => {
                self.block_self(blk);
                self.self_power(Power::new(PowerId::EnergizedBlue, 1));
            }
            Consume => {
                self.self_power(Power::new(PowerId::Focus, magic));
                self.push_bot(Action::DecreaseMaxOrb { amount: 1 });
            }
            Coolheaded => {
                self.channel(OrbId::Frost);
                self.draw(magic);
            }
            CoreSurge => {
                self.hit(m, dmg);
                self.self_power(Power::new(PowerId::Artifact, magic));
            }
            CreativeAI => self.self_power(Power::new(PowerId::CreativeAI, magic)),
            Darkness => {
                self.channel(OrbId::Dark);
                if card.upgraded {
                    self.push_bot(Action::Impulse { dark_only: true });
                }
            }
            Defragment => self.self_power(Power::new(PowerId::Focus, magic)),
            DoubleEnergy => self.push_bot(Action::DoubleEnergy),
            EchoForm => self.self_power(Power::new(PowerId::EchoForm, 1)),
            Electrodynamics => {
                if !self.player.has_power(PowerId::Electro) {
                    self.self_power(Power::flag(PowerId::Electro));
                }
                for _ in 0..magic {
                    self.channel(OrbId::Lightning);
                }
            }
            Equilibrium => {
                self.block_self(blk);
                self.self_power(Power::new(PowerId::Equilibrium, magic));
            }
            FTL => {
                if let Some(t) = m {
                    self.push_bot(Action::Ftl { target: t, info: Self::info(dmg), count: magic });
                }
            }
            Fission => self.push_bot(Action::Fission { upgraded: card.upgraded }),
            Fusion => {
                for _ in 0..magic {
                    self.channel(OrbId::Plasma);
                }
            }
            GeneticAlgorithm => {
                self.push_bot(Action::IncreaseMisc { uuid: card.uuid, amount: magic });
                self.block_self(blk);
            }
            Glacier => {
                self.block_self(blk);
                for _ in 0..magic {
                    self.channel(OrbId::Frost);
                }
            }
            GoForTheEyes => {
                self.hit(m, dmg);
                if let Some(t) = m {
                    self.push_bot(Action::ForTheEyes { target: t, amount: magic });
                }
            }
            Heatsinks => self.self_power(Power::new(PowerId::Heatsink, magic)),
            HelloWorld => self.self_power(Power::new(PowerId::Hello, 1)),
            Hologram => {
                self.block_self(blk);
                self.push_bot(Action::DiscardToHand { amount: 1, set_cost: false });
            }
            LockOn => {
                self.hit(m, dmg);
                self.enemy_power(m, Power::new(PowerId::LockOn, magic));
            }
            Loop => self.self_power(Power::new(PowerId::Loop, magic)),
            MachineLearning => self.self_power(Power::new(PowerId::Draw, magic)),
            Melter => {
                if let Some(t) = m {
                    self.push_bot(Action::RemoveAllBlock { target: Cr::Monster(t) });
                }
                self.hit(m, dmg);
            }
            MultiCast => self.push_bot(Action::Multicast { upgraded: card.upgraded, free: card.free_to_play_once, energy_on_use: card.energy_on_use }),
            Overclock => {
                self.draw(magic);
                let burn = self.new_card(Burn);
                self.push_bot(Action::MakeTempCardInDiscard { card: burn, amount: 1 });
            }
            Rainbow => {
                self.channel(OrbId::Lightning);
                self.channel(OrbId::Frost);
                self.channel(OrbId::Dark);
            }
            Reboot => {
                // ShuffleAllAction's constructor runs the relics' onShuffle.
                self.relics_on_shuffle();
                self.push_bot(Action::ShuffleAll);
                self.push_bot(Action::ShuffleDrawPile);
                self.draw(magic);
            }
            Rebound => {
                self.hit(m, dmg);
                let mut p = Power::new(PowerId::Rebound, 1);
                p.just_applied = true;
                self.self_power(p);
            }
            Recursion => self.push_bot(Action::Redo),
            Recycle => self.push_bot(Action::Recycle),
            ReinforcedBody => self.push_bot(Action::ReinforcedBody { block: blk, free: card.free_to_play_once, energy_on_use: card.energy_on_use }),
            Reprogram => {
                self.self_power(Power::new(PowerId::Focus, -magic));
                self.self_power(Power::new(PowerId::Strength, magic));
                self.self_power(Power::new(PowerId::Dexterity, magic));
            }
            RipAndTear => {
                for _ in 0..magic {
                    self.push_bot(Action::AttackDamageRandomEnemy { card: card.clone() });
                }
            }
            Scrape => {
                self.hit(m, dmg);
                let follow_up = Some(Box::new(Action::ScrapeFollowUp));
                self.push_bot(Action::DrawCard { amount: magic, follow_up, shuffle_check: false });
            }
            Seek => self.push_bot(Action::DrawPileToHandChoice { amount: magic }),
            SelfRepair => self.self_power(Power::new(PowerId::Repair, magic)),
            Skim => self.draw(magic),
            StaticDischarge => self.self_power(Power::new(PowerId::StaticDischarge, magic)),
            SteamBarrier => {
                self.block_self(blk);
                self.push_bot(Action::ModifyBlock { uuid: card.uuid, amount: -1 });
            }
            Storm => self.self_power(Power::new(PowerId::Storm, magic)),
            Streamline => {
                self.hit(m, dmg);
                self.push_bot(Action::ReduceCost { uuid: card.uuid });
            }
            Sunder => {
                if let Some(t) = m {
                    self.push_bot(Action::Sunder { target: t, info: Self::info(dmg) });
                }
            }
            Tempest => self.push_bot(Action::Tempest { upgraded: card.upgraded, free: card.free_to_play_once, energy_on_use: card.energy_on_use }),
            ThunderStrike => {
                let n = self.orbs_channeled_this_combat.iter().filter(|&&o| o == OrbId::Lightning).count();
                card.magic = n as i32;
                card.base_magic = n as i32;
                for _ in 0..n {
                    self.push_bot(Action::AttackDamageRandomEnemy { card: card.clone() });
                }
            }
            Turbo => {
                self.push_bot(Action::GainEnergy { amount: magic });
                let void = self.new_card(Void);
                self.push_bot(Action::MakeTempCardInDiscard { card: void, amount: 1 });
            }
            WhiteNoise => {
                let id = self.truly_random_card_in_combat(CardType::Power);
                let mut c = self.new_card(id);
                c.set_cost_for_turn(0);
                self.push_bot(Action::MakeTempCardInHand { card: c, amount: 1 });
            }
            // The Watcher.
            StrikeP | Smite | ThroughViolence | SandsOfTime | SignatureMove | WindmillStrike | Weave | FlurryOfBlows | Brilliance => {
                self.hit(m, dmg)
            }
            DefendP | Protect | Safety | Perseverance | SpiritShield => self.block_self(blk),
            Eruption | EmptyFist => {
                self.hit(m, dmg);
                let stance = if card.id == Eruption { Stance::Wrath } else { Stance::Neutral };
                self.stance(stance);
            }
            Vigilance | EmptyBody => {
                self.block_self(blk);
                let stance = if card.id == Vigilance { Stance::Calm } else { Stance::Neutral };
                self.stance(stance);
            }
            EmptyMind => {
                self.draw(magic);
                self.stance(Stance::Neutral);
            }
            Crescendo => self.stance(Stance::Wrath),
            Tranquility => self.stance(Stance::Calm),
            Alpha | Evaluate | Pray | ReachHeaven | Beta => {
                match card.id {
                    Evaluate => self.block_self(blk),
                    Pray => {
                        let p = self.mantra(magic);
                        self.self_power(p);
                    }
                    ReachHeaven => self.hit(m, dmg),
                    _ => {}
                }
                let made = match card.id {
                    Alpha => Beta,
                    Beta => Omega,
                    ReachHeaven => ThroughViolence,
                    _ => Insight,
                };
                let card = self.new_card(made);
                self.push_bot(Action::MakeTempCardInDrawPile { card, amount: 1, random_spot: true, to_bottom: false });
            }
            CarveReality | DeceiveReality => {
                if card.id == CarveReality {
                    self.hit(m, dmg);
                } else {
                    self.block_self(blk);
                }
                let made = if card.id == CarveReality { Smite } else { Safety };
                let card = self.new_card(made);
                self.push_bot(Action::MakeTempCardInHand { card, amount: 1 });
            }
            BattleHymn => self.self_power(Power::new(PowerId::BattleHymn, magic)),
            Blasphemy => {
                self.stance(Stance::Divinity);
                self.self_power(Power::flag(PowerId::EndTurnDeath));
            }
            BowlingBash => {
                let alive = self.order.iter().filter(|&&mo| !self.monsters[mo].is_dead_or_escaped()).count();
                for _ in 0..alive {
                    self.hit(m, dmg);
                }
            }
            Collect => self.push_bot(Action::Collect { upgraded: card.upgraded, free: card.free_to_play_once, energy_on_use: card.energy_on_use }),
            Conclude => {
                self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal });
                self.push_bot(Action::PressEndTurnButton);
            }
            ConjureBlade => {
                let energy_on_use = if card.upgraded { card.energy_on_use + 1 } else { card.energy_on_use };
                self.push_bot(Action::ConjureBlade { free: card.free_to_play_once, energy_on_use });
            }
            Consecrate => self.push_bot(Action::DamageAllEnemies { source: Some(Cr::Player), damage: multi, damage_type: DamageType::Normal }),
            CrushJoints | SashWhip => {
                self.hit(m, dmg);
                if let Some(t) = m {
                    let (card_type, power) = if card.id == CrushJoints {
                        (CardType::Skill, Power::vulnerable(magic, false, turn_has_ended))
                    } else {
                        (CardType::Attack, Power::weak(magic, false))
                    };
                    let then = Box::new(self.apply_power_action(Cr::Monster(t), Some(Cr::Player), power, magic));
                    self.push_bot(Action::PreviousCardCheck { card_type, then });
                }
            }
            FollowUp => {
                self.hit(m, dmg);
                self.push_bot(Action::PreviousCardCheck { card_type: CardType::Attack, then: Box::new(Action::GainEnergy { amount: 1 }) });
            }
            Sanctity => {
                self.block_self(blk);
                let then = Box::new(Action::DrawCard { amount: magic, follow_up: None, shuffle_check: false });
                self.push_bot(Action::PreviousCardCheck { card_type: CardType::Skill, then });
            }
            CutThroughFate => {
                self.hit(m, dmg);
                let a = self.scry(magic);
                self.push_bot(a);
                self.draw(1);
            }
            JustLucky => {
                let a = self.scry(magic);
                self.push_bot(a);
                self.block_self(blk);
                self.hit(m, dmg);
            }
            ThirdEye => {
                self.block_self(blk);
                let a = self.scry(magic);
                self.push_bot(a);
            }
            DeusExMachina => {}
            DevaForm => self.self_power(Power::new(PowerId::DevaForm, 1)),
            Devotion => self.self_power(Power::new(PowerId::Devotion, magic)),
            Establishment => self.self_power(Power::new(PowerId::Establishment, magic)),
            Fasting => {
                self.self_power(Power::new(PowerId::Strength, magic));
                self.self_power(Power::new(PowerId::Dexterity, magic));
                self.self_power(Power::new(PowerId::EnergyDown, 1));
            }
            FearNoEvil => {
                if let Some(t) = m {
                    self.push_bot(Action::FearNoEvil { target: t, info: Self::info(dmg) });
                }
            }
            FlyingSleeves => {
                self.hit(m, dmg);
                self.hit(m, dmg);
            }
            ForeignInfluence => self.push_bot(Action::ForeignInfluence { upgraded: card.upgraded }),
            Foresight => self.self_power(Power::new(PowerId::Foresight, magic)),
            Halt => {
                let extra = self.halt_wrath_block(card);
                self.push_bot(Action::Halt { block: blk, extra });
            }
            Indignation => self.push_bot(Action::Indignation { amount: magic }),
            InnerPeace => self.push_bot(Action::InnerPeace { amount: magic }),
            Judgement => {
                if let Some(t) = m {
                    self.push_bot(Action::Judgement { target: t, cutoff: magic });
                }
            }
            LessonLearned | Wallop => {
                if let Some(t) = m {
                    let info = Self::info(dmg);
                    self.push_bot(if card.id == Wallop { Action::Wallop { target: t, info } } else { Action::LessonLearned { target: t, info } });
                }
            }
            LikeWater => self.self_power(Power::new(PowerId::LikeWater, magic)),
            MasterReality => self.self_power(Power::flag(PowerId::MasterReality)),
            MentalFortress => self.self_power(Power::new(PowerId::MentalFortress, magic)),
            Nirvana => self.self_power(Power::new(PowerId::Nirvana, magic)),
            Rushdown => self.self_power(Power::new(PowerId::Rushdown, magic)),
            Study => self.self_power(Power::new(PowerId::Study, magic)),
            WaveOfTheHand => self.self_power(Power::new(PowerId::WaveOfTheHand, magic)),
            Omega => self.self_power(Power::new(PowerId::Omega, magic)),
            Meditate => {
                self.push_bot(Action::Meditate { amount: magic });
                self.stance(Stance::Calm);
                self.push_bot(Action::PressEndTurnButton);
            }
            Omniscience => self.push_bot(Action::Omniscience { amount: magic }),
            PressurePoints => {
                self.enemy_power(m, Power::new(PowerId::Mark, magic));
                self.push_bot(Action::TriggerMarks);
            }
            Prostrate => {
                let p = self.mantra(magic);
                self.self_power(p);
                self.block_self(blk);
            }
            Worship => {
                let p = self.mantra(magic);
                self.self_power(p);
            }
            Ragnarok => {
                for _ in 0..magic {
                    self.push_bot(Action::AttackDamageRandomEnemy { card: card.clone() });
                }
            }
            Scrawl => self.push_bot(Action::Expertise { amount: 10 }),
            SimmeringFury => {
                self.self_power(Power::flag(PowerId::WrathNextTurn));
                self.self_power(Power::new(PowerId::DrawCardNextTurn, magic));
            }
            Swivel => {
                self.block_self(blk);
                self.self_power(Power::new(PowerId::FreeAttack, 1));
            }
            TalkToTheHand => {
                self.hit(m, dmg);
                self.enemy_power(m, Power::new(PowerId::BlockReturn, magic));
            }
            Tantrum => {
                for _ in 0..magic {
                    self.hit(m, dmg);
                }
                self.stance(Stance::Wrath);
            }
            Vault => {
                self.push_bot(Action::SkipEnemiesTurn);
                self.push_bot(Action::PressEndTurnButton);
            }
            WheelKick => {
                self.hit(m, dmg);
                self.draw(magic);
            }
            Wish => {
                let mut options = Vec::new();
                for id in [BecomeAlmighty, FameAndFortune, LiveForever] {
                    let mut c = self.new_card(id);
                    if card.upgraded {
                        c.upgrade();
                    }
                    options.push(c);
                }
                self.push_bot(Action::ChooseOne { options });
            }
            WreathOfFlame => self.self_power(Power::new(PowerId::Vigor, magic)),
            Expunger => {
                for _ in 0..magic {
                    self.hit(m, dmg);
                }
            }
            Insight => self.draw(magic),
            Miracle => self.push_bot(Action::GainEnergy { amount: if card.upgraded { 2 } else { 1 } }),
            // Option cards are chosen, not played; their use() would do what choosing does.
            BecomeAlmighty | FameAndFortune | LiveForever | ChooseCalm | ChooseWrath => {}
            _ => {}
        }
    }

    /// `addToBot(new ChangeStanceAction(...))`.
    fn stance(&mut self, stance: Stance) {
        self.push_bot(Action::ChangeStance { stance });
    }
}
