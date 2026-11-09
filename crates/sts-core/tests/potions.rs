//! The shared potions and the Ironclad's, one test each, checked against the decompiled game
//! (`potions/` and the actions they use). Combat use goes through `Combat::step`; the three that
//! work outside a fight through the run.

use sts_core::action::Cr;
use sts_core::cards::{Card, CardColor, CardId, CardType};
use sts_core::character::Character;
use sts_core::combat::{Combat, CombatConfig, PlayerAction};
use sts_core::potions::PotionId;
use sts_core::powers::PowerId;
use sts_core::relics::{RelicId, RelicState};
use sts_core::run::{Run, RunAction, RunConfig};

fn fight(seed: i64, encounter: &str, deck: &[(CardId, i32)]) -> Combat {
    let mut cfg = CombatConfig::new(Character::Ironclad, seed, encounter);
    cfg.deck = deck.to_vec();
    cfg.relics.clear();
    Combat::new(&cfg).unwrap()
}

/// A fight whose hand is exactly `hand` and whose draw pile is `draw` (top last), with `potion`
/// in the first slot.
fn setup(encounter: &str, potion: PotionId, hand: &[CardId], draw: &[CardId]) -> Combat {
    let mut c = fight(7, encounter, &[(CardId::DefendR, 0); 5]);
    c.hand.clear();
    c.draw_pile.clear();
    c.discard_pile.clear();
    for &id in hand {
        let card = c.new_card(id);
        c.hand.push(card);
    }
    for &id in draw {
        let card = c.new_card(id);
        c.draw_pile.push(card);
    }
    c.potions = vec![Some(potion), None, None];
    c
}

fn drink(c: &mut Combat, target: Option<usize>) {
    c.step(&PlayerAction::UsePotion { slot: 0, target }).unwrap();
}

fn ids(cards: &[Card]) -> Vec<CardId> {
    cards.iter().map(|c| c.id).collect()
}

fn choose(c: &mut Combat, indices: &[usize]) {
    c.step(&PlayerAction::Choose { indices: indices.to_vec() }).unwrap();
}

fn end_turn(c: &mut Combat) {
    c.step(&PlayerAction::EndTurn).unwrap();
}

fn m0(c: &Combat) -> usize {
    c.order[0]
}

fn power(c: &Combat, who: Cr, id: PowerId) -> i32 {
    c.power_amount(who, id)
}

#[test]
fn self_buff_potions() {
    use PotionId::*;
    // (potion, power, amount)
    let cases = [
        (AncientPotion, PowerId::Artifact, 1),
        (DexterityPotion, PowerId::Dexterity, 2),
        (StrengthPotion, PowerId::Strength, 2),
        (EssenceOfSteel, PowerId::PlatedArmor, 4),
        (HeartOfIron, PowerId::Metallicize, 6),
        (LiquidBronze, PowerId::Thorns, 3),
        (RegenPotion, PowerId::Regen, 5),
        (DuplicationPotion, PowerId::Duplication, 1),
        (CultistPotion, PowerId::Ritual, 1),
        (SteroidPotion, PowerId::Strength, 5),
        (SpeedPotion, PowerId::Dexterity, 5),
    ];
    for (p, id, amount) in cases {
        let mut c = setup("Jaw Worm", p, &[], &[]);
        drink(&mut c, None);
        assert_eq!(power(&c, Cr::Player, id), amount, "{p:?}");
        assert!(c.potions[0].is_none(), "{p:?}");
    }
}

#[test]
fn steroid_and_speed_wear_off_and_cultist_grows() {
    for (p, id) in [(PotionId::SteroidPotion, PowerId::Strength), (PotionId::SpeedPotion, PowerId::Dexterity)] {
        let mut c = setup("Cultist", p, &[], &[CardId::DefendR; 10]);
        drink(&mut c, None);
        end_turn(&mut c);
        assert!(!c.player.has_power(id), "{p:?}");
    }
    let mut c = setup("Cultist", PotionId::CultistPotion, &[], &[CardId::DefendR; 10]);
    drink(&mut c, None);
    end_turn(&mut c);
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 1);
}

#[test]
fn enemy_debuff_potions() {
    let mut c = setup("Jaw Worm", PotionId::WeakenPotion, &[], &[]);
    let m = m0(&c);
    drink(&mut c, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 3);
    let mut c = setup("Jaw Worm", PotionId::FearPotion, &[], &[]);
    drink(&mut c, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Vulnerable), 3);
}

#[test]
fn damage_potions() {
    // Fire Potion: 20 THORNS damage, only the target's powers apply (Vulnerable doesn't).
    let mut c = setup("Jaw Worm", PotionId::FirePotion, &[CardId::Bash], &[]);
    let m = m0(&c);
    c.monsters[m].hp = 100;
    c.step(&PlayerAction::PlayCard { hand_index: 0, target: Some(m) }).unwrap();
    drink(&mut c, Some(m));
    assert_eq!(c.monsters[m].hp, 100 - 8 - 20);
    // Explosive Potion: 10 to every enemy.
    let mut c = setup("3 Sentries", PotionId::ExplosivePotion, &[], &[]);
    let before: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    drink(&mut c, None);
    assert!(c.order.iter().zip(&before).all(|(&m, &b)| c.monsters[m].hp == b - 10));
}

#[test]
fn block_energy_draw_and_heal_potions() {
    let mut c = setup("Jaw Worm", PotionId::BlockPotion, &[], &[]);
    drink(&mut c, None);
    assert_eq!(c.player.block, 12);
    let mut c = setup("Jaw Worm", PotionId::EnergyPotion, &[], &[]);
    drink(&mut c, None);
    assert_eq!(c.energy, 5);
    let mut c = setup("Jaw Worm", PotionId::SwiftPotion, &[], &[CardId::DefendR; 5]);
    drink(&mut c, None);
    assert_eq!(c.hand.len(), 3);
    let mut c = setup("Jaw Worm", PotionId::BloodPotion, &[], &[]);
    c.player.hp = 40;
    drink(&mut c, None);
    // 20% of 80.
    assert_eq!(c.player.hp, 56);
    let mut c = setup("Jaw Worm", PotionId::FruitJuice, &[], &[]);
    drink(&mut c, None);
    assert_eq!((c.player.hp, c.player.max_hp), (85, 85));
}

#[test]
fn discovery_potions_offer_three_of_their_type() {
    use PotionId::*;
    for (p, check) in [(AttackPotion, 0), (SkillPotion, 1), (PowerPotion, 2), (ColorlessPotion, 3)] {
        let mut c = setup("Jaw Worm", p, &[], &[]);
        drink(&mut c, None);
        let offer: Vec<Card> = c.pending.as_ref().expect("a discovery").offer.clone();
        assert_eq!(offer.len(), 3, "{p:?}");
        for card in &offer {
            match check {
                0 => assert_eq!(card.card_type(), CardType::Attack),
                1 => assert_eq!(card.card_type(), CardType::Skill),
                2 => assert_eq!(card.card_type(), CardType::Power),
                _ => assert_eq!(card.data().color, CardColor::Colorless),
            }
        }
        choose(&mut c, &[0]);
        // The pick arrives free this turn.
        assert_eq!(c.hand.len(), 1);
        assert_eq!(c.hand[0].id, offer[0].id);
        assert!(c.hand[0].cost_for_turn <= 0);
    }
}

#[test]
fn blessing_of_the_forge_upgrades_the_hand() {
    let mut c = setup("Jaw Worm", PotionId::BlessingOfTheForge, &[CardId::StrikeR, CardId::DefendR, CardId::Wound], &[]);
    drink(&mut c, None);
    assert!(c.hand.iter().filter(|x| x.id != CardId::Wound).all(|x| x.upgraded));
}

#[test]
fn elixir_and_gamblers_brew_choose_from_the_hand() {
    let mut c = setup("Jaw Worm", PotionId::Elixir, &[CardId::StrikeR, CardId::DefendR, CardId::Wound], &[]);
    drink(&mut c, None);
    let p = c.pending.as_ref().unwrap();
    assert!(p.any_number && p.can_pick_zero);
    choose(&mut c, &[1, 2]);
    assert_eq!(ids(&c.hand), [CardId::StrikeR]);
    assert_eq!(c.exhaust_pile.len(), 2);

    let mut c = setup("Jaw Worm", PotionId::GamblersBrew, &[CardId::Wound, CardId::Wound, CardId::StrikeR], &[CardId::DefendR; 3]);
    drink(&mut c, None);
    choose(&mut c, &[0, 1]);
    assert_eq!(ids(&c.hand), [CardId::StrikeR, CardId::DefendR, CardId::DefendR]);
}

#[test]
fn distilled_chaos_plays_the_top_three_cards() {
    let mut c = setup("Jaw Worm", PotionId::DistilledChaosPotion, &[], &[CardId::DefendR, CardId::DefendR, CardId::DefendR, CardId::StrikeR]);
    drink(&mut c, None);
    // Strike (top), then two Defends.
    assert_eq!(c.player.block, 10);
    assert_eq!(c.draw_pile.len(), 1);
    assert_eq!(c.energy, 3);
}

#[test]
fn duplication_plays_the_next_card_twice() {
    let mut c = setup("Jaw Worm", PotionId::DuplicationPotion, &[CardId::DefendR], &[]);
    drink(&mut c, None);
    c.step(&PlayerAction::PlayCard { hand_index: 0, target: None }).unwrap();
    assert_eq!(c.player.block, 10);
}

#[test]
fn liquid_memories_returns_a_discarded_card_for_free() {
    let mut c = setup("Jaw Worm", PotionId::LiquidMemories, &[], &[]);
    let b = c.new_card(CardId::Bludgeon);
    c.discard_pile.push(b);
    drink(&mut c, None);
    assert_eq!(ids(&c.hand), [CardId::Bludgeon]);
    assert_eq!(c.hand[0].cost_for_turn, 0);
}

#[test]
fn regen_heals_less_each_turn() {
    let mut c = setup("Cultist", PotionId::RegenPotion, &[], &[CardId::DefendR; 20]);
    c.player.hp = 50;
    drink(&mut c, None);
    end_turn(&mut c);
    assert_eq!((c.player.hp, power(&c, Cr::Player, PowerId::Regen)), (55, 4));
}

#[test]
fn snecko_oil_draws_and_randomizes_costs() {
    let mut c = setup("Jaw Worm", PotionId::SneckoOil, &[], &[CardId::Bludgeon; 6]);
    drink(&mut c, None);
    assert_eq!(c.hand.len(), 5);
    assert!(c.hand.iter().all(|x| (0..=3).contains(&x.cost)));
}

#[test]
fn smoke_bomb_escapes_except_from_bosses() {
    let mut c = setup("Jaw Worm", PotionId::SmokeBomb, &[], &[]);
    drink(&mut c, None);
    assert!(c.smoked);
    assert!(c.is_over());
    let c = setup("Hexaghost", PotionId::SmokeBomb, &[], &[]);
    assert!(!c.can_use_potion(PotionId::SmokeBomb));
    // Nor while a monster has Back Attack (the Act 4 elite, an elite, not a boss).
    let c = setup("Shield and Spear", PotionId::SmokeBomb, &[], &[]);
    assert!(c.order.iter().any(|&m| c.monsters[m].has_power(PowerId::BackAttack)));
    assert!(!c.can_use_potion(PotionId::SmokeBomb));
}

#[test]
fn distilled_chaos_plays_stack_with_the_count_from_when_it_was_taken() {
    // Stack under a Defend, 6 Strikes in the discard pile: Distilled Chaos takes the Defend and
    // Stack, then reshuffles the discard pile for its third card. Stack blocks the 6 the discard
    // pile held when it was taken (PlayTopCardAction's applyPowers), not the 1 (the Defend) left
    // after the reshuffle by the time it is played (soak 2026-10-27, a Defect run at the Bronze
    // Automaton: 34 block in the game, 18 in the engine).
    let mut c = setup("Jaw Worm", PotionId::DistilledChaosPotion, &[], &[CardId::Stack, CardId::DefendR]);
    for _ in 0..6 {
        let strike = c.new_card(CardId::StrikeR);
        c.discard_pile.push(strike);
    }
    drink(&mut c, None);
    assert_eq!(c.player.block, 5 + 6);
    // Played from the hand afterwards, Stack counts the discard pile then.
    assert!(c.discard_pile.iter().chain(c.draw_pile.iter()).chain(c.hand.iter()).all(|x| x.applied_base.is_none()));
}

#[test]
fn fairy_in_a_bottle_saves_the_player_and_cant_be_drunk() {
    let mut c = setup("Jaw Worm", PotionId::FairyPotion, &[], &[CardId::DefendR; 10]);
    assert!(!c.can_use_potion(PotionId::FairyPotion));
    c.player.hp = 5;
    end_turn(&mut c);
    // Chomp's 11 would kill: back at 30% of 80.
    assert_eq!(c.player.hp, 24);
    assert!(c.potions[0].is_none());
}

#[test]
fn entropic_brew_fills_every_slot() {
    let mut c = setup("Jaw Worm", PotionId::EntropicBrew, &[], &[]);
    drink(&mut c, None);
    assert!(c.potions.iter().all(|p| p.is_some()));
}

#[test]
fn sacred_bark_doubles_and_toy_ornithopter_heals() {
    let mut c = setup("Jaw Worm", PotionId::BlockPotion, &[], &[]);
    c.player.relics.push(RelicState::new(RelicId::SacredBark));
    drink(&mut c, None);
    assert_eq!(c.player.block, 24);
    let mut c = setup("Jaw Worm", PotionId::BlockPotion, &[], &[]);
    c.player.relics.push(RelicState::new(RelicId::ToyOrnithopter));
    c.player.hp = 50;
    drink(&mut c, None);
    assert_eq!(c.player.hp, 55);
}

#[test]
fn blood_fruit_juice_and_entropic_brew_work_outside_a_fight() {
    let mut run = Run::new(RunConfig::new(4));
    run.step(&RunAction::Neow { option: 1 }).unwrap();
    run.hp = 40;
    run.potions = vec![Some(PotionId::BloodPotion), Some(PotionId::FruitJuice), Some(PotionId::EntropicBrew)];
    let max = run.max_hp;
    run.step(&RunAction::UsePotion { slot: 0 }).unwrap();
    assert_eq!(run.hp, 40 + max / 5);
    run.step(&RunAction::UsePotion { slot: 1 }).unwrap();
    assert_eq!(run.max_hp, max + 5);
    run.step(&RunAction::UsePotion { slot: 2 }).unwrap();
    assert!(run.potions.iter().all(|p| p.is_some()));
    // Other potions can only be discarded there.
    run.potions = vec![Some(PotionId::BlockPotion), None, None];
    assert!(!run.legal_actions().contains(&RunAction::UsePotion { slot: 0 }));
    assert!(run.legal_actions().contains(&RunAction::DiscardPotion { slot: 0 }));
}
