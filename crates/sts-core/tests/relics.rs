//! The shared relics and the Ironclad's in combat, one test per behaviour, checked against the
//! decompiled game (`relics/` and the actions they queue). Expected numbers come from reading
//! those classes; the recordings and soaks checked the same relics in whole fights. Relic effects
//! outside combat are tested in the run module (`run/relic_tests.rs`).

use sts_core::action::{Action, Cr};
use sts_core::cards::{Card, CardColor, CardId, CardType};
use sts_core::character::Character;
use sts_core::combat::{ChoiceKind, Combat, CombatConfig, Outcome, PlayerAction};
use sts_core::monsters::Intent;
use sts_core::potions::PotionId;
use sts_core::powers::{Power, PowerId};
use sts_core::relics::{RelicId, RelicState};

/// An Ironclad fight with `relics` from the start (their battle-start hooks fire) and ten Defends.
fn fight_with(encounter: &str, relics: &[RelicId]) -> Combat {
    fight_deck(encounter, relics, &[(CardId::DefendR, 0); 10], 80)
}

fn fight_deck(encounter: &str, relics: &[RelicId], deck: &[(CardId, i32)], hp: i32) -> Combat {
    let mut cfg = CombatConfig::new(Character::Ironclad, 7, encounter);
    cfg.deck = deck.to_vec();
    cfg.relics = relics.to_vec();
    cfg.player_hp = hp;
    Combat::new(&cfg).unwrap()
}

/// A fight with one relic whose counter starts at `counter` (as a run would carry it).
fn fight_counter(encounter: &str, relic: RelicId, counter: i32) -> Combat {
    let mut cfg = CombatConfig::new(Character::Ironclad, 7, encounter);
    cfg.deck = vec![(CardId::DefendR, 0); 10];
    cfg.relics = vec![relic];
    cfg.relic_counters = Some(vec![counter]);
    Combat::new(&cfg).unwrap()
}

/// Replaces the hand and draw pile (top last) and empties the discard pile.
fn set_piles(c: &mut Combat, hand: &[CardId], draw: &[CardId]) {
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
}

/// A fight with `relics` from the start and exactly `hand` and `draw`.
fn setup(encounter: &str, relics: &[RelicId], hand: &[CardId], draw: &[CardId]) -> Combat {
    let mut c = fight_with(encounter, relics);
    set_piles(&mut c, hand, draw);
    c
}

fn idx(c: &Combat, id: CardId) -> usize {
    c.hand.iter().position(|card| card.id == id).unwrap_or_else(|| panic!("{id:?} not in hand: {:?}", ids(&c.hand)))
}

fn ids(cards: &[Card]) -> Vec<CardId> {
    cards.iter().map(|c| c.id).collect()
}

fn play(c: &mut Combat, id: CardId, target: Option<usize>) {
    let i = idx(c, id);
    c.step(&PlayerAction::PlayCard { hand_index: i, target }).unwrap();
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

fn relic(c: &Combat, id: RelicId) -> &RelicState {
    c.player.relics.iter().find(|r| r.id == id).unwrap()
}

fn count(cards: &[Card], id: CardId) -> usize {
    cards.iter().filter(|c| c.id == id).count()
}

fn can_play(c: &Combat, hand_index: usize) -> bool {
    c.legal_actions().iter().any(|a| matches!(a, PlayerAction::PlayCard { hand_index: h, .. } if *h == hand_index))
}

/// Queues `action` and lets the engine run it (playing a 0-cost Flex, a Skill giving 2 Strength
/// this turn, to drive the queue).
fn run_action(c: &mut Combat, action: Action) {
    c.push_bot(action);
    let card = c.new_card(CardId::Flex);
    c.hand.push(card);
    play(c, CardId::Flex, None);
}

// ---- Battle start ----

#[test]
fn battle_start_relics_give_their_powers() {
    use RelicId::*;
    // (relic, power on the player, amount): `atBattleStart` / `atPreBattle`.
    let cases = [
        (Akabeko, PowerId::Vigor, 8),
        (BronzeScales, PowerId::Thorns, 3),
        (ClockworkSouvenir, PowerId::Artifact, 1),
        (FossilizedHelix, PowerId::Buffer, 1),
        (GremlinMask, PowerId::Weakened, 1),
        (OddlySmoothStone, PowerId::Dexterity, 1),
        (ThreadAndNeedle, PowerId::PlatedArmor, 4),
        (Vajra, PowerId::Strength, 1),
        (MutagenicStrength, PowerId::Strength, 3),
        (MutagenicStrength, PowerId::Flex, 3),
    ];
    for (r, p, amount) in cases {
        let c = fight_with("Jaw Worm", &[r]);
        assert_eq!(power(&c, Cr::Player, p), amount, "{r:?}");
    }
    // Ginger keeps Gremlin Mask's Weak off; Turnip keeps Frail off.
    let c = fight_with("Jaw Worm", &[GremlinMask, Ginger]);
    assert!(!c.player.has_power(PowerId::Weakened));
    let mut c = setup("Jaw Worm", &[Turnip], &[], &[CardId::DefendR; 5]);
    let m = m0(&c);
    run_action(&mut c, Action::ApplyPower { target: Cr::Player, source: Some(Cr::Monster(m)), power: Power::frail(2, true), amount: 2 });
    assert!(!c.player.has_power(PowerId::Frail));
    let mut c = setup("Jaw Worm", &[], &[], &[CardId::DefendR; 5]);
    run_action(&mut c, Action::ApplyPower { target: Cr::Player, source: Some(Cr::Monster(m)), power: Power::frail(2, true), amount: 2 });
    assert_eq!(power(&c, Cr::Player, PowerId::Frail), 2);
}

#[test]
fn battle_start_block_draw_heal_and_cards() {
    use RelicId::*;
    assert_eq!(fight_with("Jaw Worm", &[Anchor]).player.block, 10);
    assert_eq!(fight_with("Jaw Worm", &[BagOfPreparation]).hand.len(), 7);
    // Blood Vial heals 2; Pantograph 25, only against a boss.
    let deck = [(CardId::DefendR, 0); 10];
    assert_eq!(fight_deck("Jaw Worm", &[BloodVial], &deck, 50).player.hp, 52);
    assert_eq!(fight_deck("Hexaghost", &[Pantograph], &deck, 40).player.hp, 65);
    assert_eq!(fight_deck("Jaw Worm", &[Pantograph], &deck, 40).player.hp, 40);
    // Mark of Pain: two Wounds shuffled into the draw pile, and an energy.
    let c = fight_with("Jaw Worm", &[MarkOfPain]);
    assert_eq!(count(&c.draw_pile, CardId::Wound), 2);
    assert_eq!(c.energy, 4);
    // Enchiridion: a random Power, free this turn, before the opening draw.
    let c = fight_with("Jaw Worm", &[Enchiridion]);
    assert_eq!(c.hand.len(), 6);
    let powers: Vec<&Card> = c.hand.iter().filter(|card| card.card_type() == CardType::Power).collect();
    assert_eq!(powers.len(), 1);
    assert_eq!(powers[0].cost_for_turn, 0);
    assert_eq!(powers[0].data().color, CardColor::Red);
}

#[test]
fn battle_start_relics_on_monsters() {
    use RelicId::*;
    let c = fight_with("2 Louse", &[BagOfMarbles, RedMask]);
    for &m in &c.order {
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Vulnerable), 1);
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 1);
    }
    let c = fight_with("2 Louse", &[PhilosopherStone]);
    for &m in &c.order {
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), 1);
    }
    assert_eq!(c.energy, 4);
    // ... and to monsters spawned later (`onSpawnMonster`): a Large Slime's split.
    let mut c = setup("Large Slime", &[PhilosopherStone], &[CardId::Bludgeon], &[CardId::DefendR; 10]);
    c.player.hp = 300;
    let m = m0(&c);
    c.monsters[m].hp = 35;
    play(&mut c, CardId::Bludgeon, Some(m));
    end_turn(&mut c);
    assert_eq!(c.monsters.len(), 3);
    for spawned in 1..3 {
        assert_eq!(power(&c, Cr::Monster(spawned), PowerId::Strength), 1);
    }
    // Neow's Lament: three fights with every enemy at 1 HP.
    let mut cfg = CombatConfig::new(Character::Ironclad, 7, "2 Louse");
    cfg.relics = vec![NeowsLament];
    let c = Combat::new(&cfg).unwrap();
    assert!(c.order.iter().all(|&m| c.monsters[m].hp == 1));
    assert_eq!(relic(&c, NeowsLament).counter, 2);
    cfg.relic_counters = Some(vec![1]);
    let c = Combat::new(&cfg).unwrap();
    assert_eq!((relic(&c, NeowsLament).counter, relic(&c, NeowsLament).used_up), (-2, true));
    cfg.relic_counters = Some(vec![-2]);
    let c = Combat::new(&cfg).unwrap();
    assert!(c.order.iter().all(|&m| c.monsters[m].hp > 1));
}

#[test]
fn elite_and_boss_relics() {
    use RelicId::*;
    // Sling: 2 Strength against elites.
    assert_eq!(power(&fight_with("Gremlin Nob", &[Sling]), Cr::Player, PowerId::Strength), 2);
    assert_eq!(power(&fight_with("Jaw Worm", &[Sling]), Cr::Player, PowerId::Strength), 0);
    // Preserved Insect: elites start at 75% of their max HP (rounded down).
    let c = fight_with("Gremlin Nob", &[PreservedInsect]);
    let nob = &c.monsters[m0(&c)];
    assert_eq!(nob.hp, (nob.max_hp as f32 * 0.75) as i32);
    let c = fight_with("Jaw Worm", &[PreservedInsect]);
    assert_eq!(c.monsters[m0(&c)].hp, c.monsters[m0(&c)].max_hp);
    // Slaver's Collar: an energy in elite and boss fights.
    assert_eq!(fight_with("Gremlin Nob", &[SlaversCollar]).energy, 4);
    assert_eq!(fight_with("Hexaghost", &[SlaversCollar]).energy, 4);
    assert_eq!(fight_with("Jaw Worm", &[SlaversCollar]).energy, 3);
}

#[test]
fn boss_relics_add_an_energy() {
    use RelicId::*;
    for r in [BustedCrown, CoffeeDripper, CursedKey, Ectoplasm, FusionHammer, MarkOfPain, PhilosopherStone, RunicDome, Sozu, VelvetChoker] {
        let mut c = fight_with("Jaw Worm", &[r]);
        assert_eq!(c.energy, 4, "{r:?}");
        end_turn(&mut c);
        assert_eq!(c.energy, 4, "{r:?} turn 2");
    }
}

#[test]
fn snecko_eye_draws_two_more_and_confuses() {
    let c = fight_with("Jaw Worm", &[RelicId::SneckoEye]);
    assert_eq!(c.hand.len(), 7);
    assert!(c.player.has_power(PowerId::Confusion));
}

#[test]
fn counter_relics_at_battle_start() {
    // Du-Vu Doll: 1 Strength per curse (its counter); Girya: 1 per lift.
    assert_eq!(power(&fight_counter("Jaw Worm", RelicId::DuVuDoll, 2), Cr::Player, PowerId::Strength), 2);
    assert!(!fight_counter("Jaw Worm", RelicId::DuVuDoll, 0).player.has_power(PowerId::Strength));
    assert_eq!(power(&fight_counter("Jaw Worm", RelicId::Girya, 3), Cr::Player, PowerId::Strength), 3);
    // Pen Nib at 9 carries its Pen Nib power into the fight.
    let c = fight_counter("Jaw Worm", RelicId::PenNib, 9);
    assert!(c.player.has_power(PowerId::PenNib));
    assert!(!fight_counter("Jaw Worm", RelicId::PenNib, 8).player.has_power(PowerId::PenNib));
    // Incense Burner's 6th turn: Intangible.
    let c = fight_counter("Jaw Worm", RelicId::IncenseBurner, 5);
    assert_eq!(power(&c, Cr::Player, PowerId::IntangiblePlayer), 1);
    assert_eq!(relic(&c, RelicId::IncenseBurner).counter, 0);
}

#[test]
fn toolbox_and_gambling_chip_open_choices() {
    // Toolbox: one of three colorless cards, before the opening draw.
    let mut c = fight_with("Jaw Worm", &[RelicId::Toolbox]);
    let p = c.pending.as_ref().expect("Toolbox's choice");
    assert!(matches!(p.kind, ChoiceKind::Discover));
    assert_eq!(p.offer.len(), 3);
    assert!(p.offer.iter().all(|card| card.data().color == CardColor::Colorless));
    let pick = p.offer[1].id;
    choose(&mut c, &[1]);
    assert_eq!(c.hand.len(), 6);
    assert_eq!(c.hand[0].id, pick);

    // Gambling Chip: discard any cards after the opening draw and draw as many.
    let mut c = fight_deck("Jaw Worm", &[RelicId::GamblingChip], &[(CardId::DefendR, 0), (CardId::StrikeR, 0)].repeat(6), 80);
    let p = c.pending.as_ref().expect("Gambling Chip's choice");
    assert!(matches!(p.kind, ChoiceKind::GamblingChip) && p.any_number);
    let draw_top: Vec<CardId> = ids(&c.draw_pile).into_iter().rev().take(2).collect();
    choose(&mut c, &[0, 1]);
    assert_eq!(c.hand.len(), 5);
    assert_eq!(c.discard_pile.len(), 2);
    assert_eq!(ids(&c.hand[3..]), draw_top);
    // Only once a fight.
    end_turn(&mut c);
    assert!(c.pending.is_none());
}

// ---- Turn start ----

#[test]
fn energy_relics_by_turn() {
    use RelicId::*;
    // Lantern: an energy on the first turn.
    let mut c = fight_with("Jaw Worm", &[Lantern]);
    assert_eq!(c.energy, 4);
    end_turn(&mut c);
    assert_eq!(c.energy, 3);
    // Ancient Tea Set after a rest site (counter -2): 2 energy on the first turn.
    let c = fight_counter("Jaw Worm", AncientTeaSet, -2);
    assert_eq!(c.energy, 5);
    assert_eq!(relic(&c, AncientTeaSet).counter, -1);
    assert_eq!(fight_counter("Jaw Worm", AncientTeaSet, -1).energy, 3);
    // Happy Flower: an energy every 3 turns.
    let mut c = fight_counter("Jaw Worm", HappyFlower, 0);
    let mut energy = vec![c.energy];
    for _ in 0..3 {
        end_turn(&mut c);
        energy.push(c.energy);
    }
    assert_eq!(energy, [3, 3, 4, 3]);
    // Art of War: an energy after a turn without attacks, not on the first turn.
    let deck = [(CardId::StrikeR, 0); 15];
    let mut c = fight_deck("Gremlin Nob", &[ArtOfWar], &deck, 80);
    assert_eq!(c.energy, 3);
    end_turn(&mut c);
    assert_eq!(c.energy, 4);
    play(&mut c, CardId::StrikeR, Some(0));
    end_turn(&mut c);
    assert_eq!(c.energy, 3);
}

#[test]
fn block_relics_by_turn() {
    // Horn Cleat: 14 Block at the start of turn 2; Captain's Wheel: 18 at turn 3. Once a fight.
    let deck = [(CardId::DefendR, 0); 10];
    let mut c = fight_deck("Jaw Worm", &[RelicId::HornCleat], &deck, 300);
    assert_eq!(c.player.block, 0);
    end_turn(&mut c);
    assert_eq!(c.player.block, 14);
    assert!(relic(&c, RelicId::HornCleat).used_up);
    end_turn(&mut c);
    assert_eq!(c.player.block, 0);
    let mut c = fight_deck("Jaw Worm", &[RelicId::CaptainsWheel], &deck, 300);
    end_turn(&mut c);
    assert_eq!(c.player.block, 0);
    end_turn(&mut c);
    assert_eq!(c.player.block, 18);
    end_turn(&mut c);
    assert_eq!(c.player.block, 0);
}

#[test]
fn damage_relics_by_turn() {
    // Mercury Hourglass: 3 to every enemy at each turn start.
    let c = fight_with("2 Louse", &[RelicId::MercuryHourglass]);
    for &m in &c.order {
        assert_eq!(c.monsters[m].hp, c.monsters[m].max_hp - 3);
    }
    // Brimstone: 2 Strength for the player, 1 for each enemy, every turn.
    let mut c = fight_with("Gremlin Nob", &[RelicId::Brimstone]);
    let m = m0(&c);
    assert_eq!((power(&c, Cr::Player, PowerId::Strength), power(&c, Cr::Monster(m), PowerId::Strength)), (2, 1));
    end_turn(&mut c);
    assert_eq!((power(&c, Cr::Player, PowerId::Strength), power(&c, Cr::Monster(m), PowerId::Strength)), (4, 2));
    // Stone Calendar: 52 to every enemy at the end of turn 7.
    let mut c = fight_deck("Gremlin Nob", &[RelicId::StoneCalendar], &[(CardId::DefendR, 0); 10], 500);
    let m = m0(&c);
    for _ in 0..6 {
        end_turn(&mut c);
    }
    assert_eq!(relic(&c, RelicId::StoneCalendar).counter, 7);
    let hp = c.monsters[m].hp;
    end_turn(&mut c);
    assert_eq!(c.monsters[m].hp, hp - 52);
}

#[test]
fn orichalcum_blocks_when_the_turn_ends_without_block() {
    // Jaw Worm opens with Chomp, 11.
    let mut c = setup("Jaw Worm", &[RelicId::Orichalcum], &[CardId::DefendR], &[CardId::DefendR; 10]);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 80 - (11 - 6));
    let mut c = setup("Jaw Worm", &[RelicId::Orichalcum], &[CardId::DefendR], &[CardId::DefendR; 10]);
    play(&mut c, CardId::DefendR, None);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 80 - (11 - 5));
}

#[test]
fn calipers_ice_cream_and_runic_pyramid_keep_things() {
    // Calipers: lose only 15 Block at turn start.
    let mut c = setup("Jaw Worm", &[RelicId::Calipers], &[], &[CardId::DefendR; 10]);
    c.player.block = 40;
    end_turn(&mut c);
    assert_eq!(c.player.block, 40 - 11 - 15);
    // Ice Cream: unspent energy carries over.
    let mut c = setup("Jaw Worm", &[RelicId::IceCream], &[CardId::DefendR], &[CardId::DefendR; 10]);
    play(&mut c, CardId::DefendR, None);
    end_turn(&mut c);
    assert_eq!(c.energy, 2 + 3);
    // Runic Pyramid: the hand stays.
    let mut c = setup("Jaw Worm", &[RelicId::RunicPyramid], &[CardId::StrikeR, CardId::Bash], &[CardId::DefendR; 10]);
    end_turn(&mut c);
    assert_eq!(c.hand.len(), 7);
    assert_eq!(ids(&c.hand[..2]), [CardId::StrikeR, CardId::Bash]);
}

#[test]
fn pocketwatch_draws_three_after_a_short_turn() {
    let mut c = fight_with("Jaw Worm", &[RelicId::Pocketwatch]);
    c.player.hp = 300;
    assert_eq!(c.hand.len(), 5);
    end_turn(&mut c);
    assert_eq!(c.hand.len(), 8);
    // Four cards played: no extra draw.
    c.energy = 10;
    for _ in 0..4 {
        play(&mut c, CardId::DefendR, None);
    }
    end_turn(&mut c);
    assert_eq!(c.hand.len(), 5);
}

#[test]
fn warped_tongs_upgrades_a_card_in_hand_each_turn() {
    let mut c = fight_with("Jaw Worm", &[RelicId::WarpedTongs]);
    assert_eq!(c.hand.iter().filter(|card| card.upgraded).count(), 1);
    end_turn(&mut c);
    assert_eq!(c.hand.iter().filter(|card| card.upgraded).count(), 1);
}

#[test]
fn unceasing_top_draws_when_the_hand_empties() {
    let mut c = setup("Jaw Worm", &[RelicId::UnceasingTop], &[CardId::DefendR], &[CardId::StrikeR, CardId::Bash]);
    play(&mut c, CardId::DefendR, None);
    assert_eq!(ids(&c.hand), [CardId::Bash]);
    let m = m0(&c);
    play(&mut c, CardId::Bash, Some(m));
    assert_eq!(ids(&c.hand), [CardId::StrikeR]);
}

#[test]
fn nilrys_codex_offers_a_card_at_the_end_of_the_turn() {
    let mut c = setup("Jaw Worm", &[RelicId::NilrysCodex], &[], &[CardId::DefendR; 10]);
    end_turn(&mut c);
    let p = c.pending.as_ref().expect("the Codex's choice");
    assert!(matches!(p.kind, ChoiceKind::Codex) && p.can_pick_zero);
    assert_eq!(p.offer.len(), 3);
    let pick = p.offer[2].id;
    let draw = c.draw_pile.len();
    choose(&mut c, &[2]);
    // Shuffled into the draw pile, then the next turn's draw.
    assert_eq!(c.draw_pile.len() + c.hand.len(), draw + 1);
    assert!(c.draw_pile.iter().chain(&c.hand).any(|card| card.id == pick));
}

// ---- Playing cards ----

#[test]
fn every_third_card_relics() {
    use RelicId::*;
    let a = CardId::Anger;
    // (relic, three 0-cost cards of the counted type)
    let mut c = setup("Gremlin Nob", &[Kunai, Shuriken, OrnamentalFan], &[a, a, a], &[]);
    let m = m0(&c);
    for _ in 0..2 {
        play(&mut c, a, Some(m));
    }
    assert_eq!(c.player.block, 0);
    play(&mut c, a, Some(m));
    assert_eq!(power(&c, Cr::Player, PowerId::Dexterity), 1);
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 1);
    // A relic's GainBlockAction is a flat amount: no Dexterity.
    assert_eq!(c.player.block, 4);
    assert_eq!(relic(&c, Kunai).counter, 0);
    // The count starts again each turn.
    let mut c = setup("Gremlin Nob", &[Kunai], &[a, a], &[a, a, a, a, a]);
    play(&mut c, a, Some(m));
    play(&mut c, a, Some(m));
    end_turn(&mut c);
    play(&mut c, a, Some(m));
    assert_eq!(power(&c, Cr::Player, PowerId::Dexterity), 0);
    // Letter Opener: every third Skill deals 5 to every enemy (Thorns: no Strength).
    let mut c = setup("2 Louse", &[LetterOpener], &[CardId::Flex; 3], &[]);
    let hp: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    for _ in 0..3 {
        play(&mut c, CardId::Flex, None);
    }
    let after: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp + c.monsters[m].block).collect();
    assert_eq!(after, hp.iter().map(|h| h - 5).collect::<Vec<_>>());
}

#[test]
fn long_count_relics() {
    // Ink Bottle: every 10th card draws 1.
    let mut c = setup("Jaw Worm", &[], &[CardId::DefendR], &[CardId::Bash]);
    let mut r = RelicState::new(RelicId::InkBottle);
    r.counter = 9;
    c.player.relics.push(r);
    play(&mut c, CardId::DefendR, None);
    assert_eq!(ids(&c.hand), [CardId::Bash]);
    assert_eq!(relic(&c, RelicId::InkBottle).counter, 0);
    // Nunchaku: every 10th Attack gives an energy.
    let mut c = setup("Jaw Worm", &[], &[CardId::Anger], &[]);
    let mut r = RelicState::new(RelicId::Nunchaku);
    r.counter = 9;
    c.player.relics.push(r);
    let m = m0(&c);
    play(&mut c, CardId::Anger, Some(m));
    assert_eq!(c.energy, 4);
    // Pen Nib: the 10th Attack deals double.
    let mut c = setup("Jaw Worm", &[], &[CardId::Anger, CardId::Anger], &[]);
    let mut r = RelicState::new(RelicId::PenNib);
    r.counter = 8;
    c.player.relics.push(r);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::Anger, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 6);
    assert!(c.player.has_power(PowerId::PenNib));
    play(&mut c, CardId::Anger, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 6 - 12);
    assert!(!c.player.has_power(PowerId::PenNib));
    assert_eq!(relic(&c, RelicId::PenNib).counter, 0);
}

#[test]
fn power_card_relics() {
    // Bird-Faced Urn: playing a Power heals 2.
    let mut c = setup("Jaw Worm", &[RelicId::BirdFacedUrn], &[CardId::Inflame], &[]);
    c.player.hp = 50;
    play(&mut c, CardId::Inflame, None);
    assert_eq!(c.player.hp, 52);
    // Mummified Hand: a random card in hand that costs energy costs 0 this turn.
    let mut c = setup("Jaw Worm", &[RelicId::MummifiedHand], &[CardId::Inflame, CardId::Anger, CardId::DefendR, CardId::Bash], &[]);
    play(&mut c, CardId::Inflame, None);
    let free: Vec<CardId> = c.hand.iter().filter(|card| card.cost_for_turn == 0).map(|card| card.id).collect();
    assert_eq!(free.len(), 2, "{free:?}");
    assert!(free.contains(&CardId::Anger));
    // Orange Pellets: an Attack, a Skill and a Power in a turn remove the player's debuffs.
    let mut c = setup("Jaw Worm", &[], &[CardId::Anger, CardId::DefendR, CardId::Inflame], &[]);
    let m = m0(&c);
    run_action(&mut c, Action::ApplyPower { target: Cr::Player, source: Some(Cr::Monster(m)), power: Power::weak(2, true), amount: 2 });
    c.player.relics.push(RelicState::new(RelicId::OrangePellets));
    play(&mut c, CardId::Anger, Some(m));
    play(&mut c, CardId::DefendR, None);
    assert!(c.player.has_power(PowerId::Weakened));
    play(&mut c, CardId::Inflame, None);
    assert!(!c.player.has_power(PowerId::Weakened));
}

#[test]
fn blue_candle_and_medical_kit_make_curses_and_statuses_playable() {
    let mut c = setup("Jaw Worm", &[], &[CardId::Injury, CardId::Wound], &[]);
    assert!(!can_play(&c, 0) && !can_play(&c, 1));
    c.player.relics.push(RelicState::new(RelicId::BlueCandle));
    c.player.relics.push(RelicState::new(RelicId::MedicalKit));
    assert!(can_play(&c, 0) && can_play(&c, 1));
    play(&mut c, CardId::Injury, None);
    assert_eq!(c.player.hp, 79);
    play(&mut c, CardId::Wound, None);
    assert_eq!(c.player.hp, 79);
    assert_eq!(ids(&c.exhaust_pile), [CardId::Injury, CardId::Wound]);
    assert_eq!(c.energy, 3);
}

#[test]
fn necronomicon_plays_a_big_attack_twice_once_a_turn() {
    let mut c = setup("Gremlin Nob", &[RelicId::Necronomicon], &[CardId::StrikeR, CardId::Carnage, CardId::Carnage], &[]);
    c.energy = 10;
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 6);
    play(&mut c, CardId::Carnage, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 6 - 40);
    play(&mut c, CardId::Carnage, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 6 - 40 - 20);
    // The copy is purged: three cards in the discard pile, and energy for the originals only.
    assert_eq!(c.discard_pile.len(), 3);
    assert_eq!(c.energy, 10 - 1 - 2 - 2);
}

#[test]
fn velvet_choker_stops_the_seventh_card() {
    let mut c = fight_with("Jaw Worm", &[RelicId::VelvetChoker]);
    set_piles(&mut c, &[CardId::DefendR; 7], &[CardId::DefendR; 10]);
    c.energy = 10;
    for _ in 0..6 {
        play(&mut c, CardId::DefendR, None);
    }
    assert!(!can_play(&c, 0));
    end_turn(&mut c);
    assert!(can_play(&c, 0));
}

#[test]
fn chemical_x_adds_two_to_x_costs() {
    let mut c = setup("Jaw Worm", &[RelicId::ChemicalX], &[CardId::Whirlwind], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::Whirlwind, None);
    assert_eq!(c.monsters[m].hp, hp - 5 * 5);
    assert_eq!(c.energy, 0);
}

#[test]
fn strange_spoon_saves_exhausted_cards_on_a_coin_flip() {
    let mut c = setup("2 Louse", &[RelicId::StrangeSpoon], &[CardId::Intimidate; 6], &[]);
    let mut rng = c.rng.card_random.clone();
    let saved = (0..6).filter(|_| rng.random_boolean()).count();
    for _ in 0..6 {
        play(&mut c, CardId::Intimidate, None);
    }
    assert_eq!(c.discard_pile.len(), saved);
    assert_eq!(c.exhaust_pile.len(), 6 - saved);
    assert!(saved > 0 && saved < 6);
}

#[test]
fn exhaust_relics() {
    // Dead Branch: a random card of the character's pools for each exhaust.
    let mut c = setup("Jaw Worm", &[RelicId::DeadBranch], &[CardId::Intimidate], &[]);
    play(&mut c, CardId::Intimidate, None);
    assert_eq!(c.hand.len(), 1);
    let d = c.hand[0].data();
    assert!(d.color == CardColor::Red && !d.healing, "{:?}", c.hand[0].id);
    // Charon's Ashes: 3 to every enemy for each exhaust.
    let mut c = setup("2 Louse", &[RelicId::CharonsAshes], &[CardId::Intimidate], &[]);
    let hp: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    play(&mut c, CardId::Intimidate, None);
    let after: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp + c.monsters[m].block).collect();
    assert_eq!(after, hp.iter().map(|h| h - 3).collect::<Vec<_>>());
}

#[test]
fn shuffle_relics() {
    // The Abacus: 6 Block on each shuffle; Sundial: 2 energy every third shuffle.
    let mut c = setup("Jaw Worm", &[RelicId::Abacus], &[CardId::ShrugItOff], &[]);
    let card = c.new_card(CardId::DefendR);
    c.discard_pile.push(card);
    play(&mut c, CardId::ShrugItOff, None);
    assert_eq!(c.player.block, 8 + 6);
    assert_eq!(ids(&c.hand), [CardId::DefendR]);

    let mut c = setup("Jaw Worm", &[], &[CardId::ShrugItOff], &[]);
    let mut r = RelicState::new(RelicId::Sundial);
    r.counter = 2;
    c.player.relics.push(r);
    let card = c.new_card(CardId::DefendR);
    c.discard_pile.push(card);
    play(&mut c, CardId::ShrugItOff, None);
    assert_eq!(c.energy, 3 - 1 + 2);
    assert_eq!(relic(&c, RelicId::Sundial).counter, 0);
}

// ---- Losing HP, healing, dying ----

#[test]
fn hp_loss_relics() {
    let draw = [CardId::StrikeR; 8];
    // Centennial Puzzle: the first HP loss of the fight draws 3.
    let mut c = setup("Jaw Worm", &[RelicId::CentennialPuzzle], &[CardId::Bloodletting, CardId::Bloodletting], &draw);
    play(&mut c, CardId::Bloodletting, None);
    assert_eq!(c.hand.len(), 4);
    play(&mut c, CardId::Bloodletting, None);
    assert_eq!(c.hand.len(), 3);
    // Runic Cube: every HP loss draws 1.
    let mut c = setup("Jaw Worm", &[RelicId::RunicCube], &[CardId::Bloodletting, CardId::Bloodletting], &draw);
    play(&mut c, CardId::Bloodletting, None);
    play(&mut c, CardId::Bloodletting, None);
    assert_eq!(c.hand.len(), 2);
    // Self-Forming Clay: 3 Block next turn.
    let mut c = setup("Jaw Worm", &[RelicId::SelfFormingClay], &[CardId::Bloodletting], &draw);
    play(&mut c, CardId::Bloodletting, None);
    assert_eq!(power(&c, Cr::Player, PowerId::NextTurnBlock), 3);
    // Tungsten Rod: every HP loss is 1 less.
    let mut c = setup("Jaw Worm", &[RelicId::TungstenRod], &[CardId::Bloodletting], &draw);
    play(&mut c, CardId::Bloodletting, None);
    assert_eq!(c.player.hp, 78);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 78 - 10);
    // Torii: an attack that would do 2-5 (after Block) does 1.
    let mut c = setup("Jaw Worm", &[RelicId::Torii], &[], &draw);
    c.player.block = 7;
    end_turn(&mut c);
    assert_eq!(c.player.hp, 79);
    let mut c = setup("Jaw Worm", &[RelicId::Torii], &[], &draw);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 69);
}

#[test]
fn red_skull_gives_strength_while_bloodied() {
    let deck = [(CardId::DefendR, 0); 10];
    assert_eq!(power(&fight_deck("Jaw Worm", &[RelicId::RedSkull], &deck, 40), Cr::Player, PowerId::Strength), 3);
    let mut c = fight_deck("Jaw Worm", &[RelicId::RedSkull], &deck, 42);
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 0);
    set_piles(&mut c, &[CardId::Bloodletting], &[]);
    play(&mut c, CardId::Bloodletting, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 3);
    c.potions = vec![Some(PotionId::BloodPotion), None, None];
    c.step(&PlayerAction::UsePotion { slot: 0, target: None }).unwrap();
    assert_eq!(c.player.hp, 39 + 16);
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 0);
}

#[test]
fn healing_relics() {
    let deck = [(CardId::DefendR, 0); 10];
    // Magic Flower: healing in combat is 50% more (rounded); Mark of the Bloom: none.
    assert_eq!(fight_deck("Jaw Worm", &[RelicId::BloodVial, RelicId::MagicFlower], &deck, 50).player.hp, 53);
    assert_eq!(fight_deck("Jaw Worm", &[RelicId::BloodVial, RelicId::MarkOfTheBloom], &deck, 50).player.hp, 50);
}

#[test]
fn victory_relics() {
    let win = |relics: &[RelicId], hp: i32| {
        let mut c = fight_deck("Jaw Worm", relics, &[(CardId::StrikeR, 0); 10], hp);
        let m = m0(&c);
        c.monsters[m].hp = 1;
        play(&mut c, CardId::StrikeR, Some(m));
        assert_eq!(c.outcome, Some(Outcome::Victory));
        (c.player.hp, c.player.max_hp)
    };
    assert_eq!(win(&[RelicId::BurningBlood], 50), (56, 80));
    assert_eq!(win(&[RelicId::BlackBlood], 50), (62, 80));
    // Meat on the Bone: 12 at 50% or less, before the others.
    assert_eq!(win(&[RelicId::BurningBlood, RelicId::MeatOnTheBone], 40), (58, 80));
    assert_eq!(win(&[RelicId::BurningBlood, RelicId::MeatOnTheBone], 41), (47, 80));
    // Face of Cleric: 1 max HP.
    assert_eq!(win(&[RelicId::FaceOfCleric], 50), (51, 81));
}

#[test]
fn lizard_tail_saves_the_player_once() {
    let mut c = setup("Jaw Worm", &[RelicId::LizardTail], &[], &[CardId::DefendR; 10]);
    c.player.hp = 5;
    end_turn(&mut c);
    assert_eq!(c.player.hp, 40);
    assert!(relic(&c, RelicId::LizardTail).used_up);
    // Jaw Worm's next attack kills.
    for _ in 0..3 {
        if c.outcome.is_none() {
            c.player.hp = 1;
            end_turn(&mut c);
        }
    }
    assert_eq!(c.outcome, Some(Outcome::Defeat));
    // Not with Mark of the Bloom.
    let mut c = setup("Jaw Worm", &[RelicId::LizardTail, RelicId::MarkOfTheBloom], &[], &[CardId::DefendR; 10]);
    c.player.hp = 5;
    end_turn(&mut c);
    assert_eq!(c.outcome, Some(Outcome::Defeat));
}

// ---- Damage ----

#[test]
fn damage_relics() {
    // Boot: an attack doing 1-4 does 5 (here a Weakened Strike, 4).
    let mut c = setup("Jaw Worm", &[RelicId::Boot, RelicId::GremlinMask], &[CardId::StrikeR], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 5);
    // Strike Dummy: +3 on Strikes.
    let mut c = setup("Gremlin Nob", &[RelicId::StrikeDummy], &[CardId::StrikeR, CardId::TwinStrike, CardId::Anger], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::StrikeR, Some(m));
    play(&mut c, CardId::TwinStrike, Some(m));
    play(&mut c, CardId::Anger, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 9 - 16 - 6);
    // Paper Phrog: Vulnerable enemies take 75% more.
    let mut c = setup("Gremlin Nob", &[RelicId::PaperFrog], &[CardId::Bash, CardId::StrikeR], &[]);
    let m = m0(&c);
    play(&mut c, CardId::Bash, Some(m));
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 10);
    // Odd Mushroom: a Vulnerable player takes 25% more (Chomp 11 -> 13).
    let mut c = setup("Jaw Worm", &[RelicId::OddMushroom], &[], &[CardId::DefendR; 10]);
    let m = m0(&c);
    run_action(&mut c, Action::ApplyPower { target: Cr::Player, source: Some(Cr::Monster(m)), power: Power::vulnerable(2, true, false), amount: 2 });
    end_turn(&mut c);
    assert_eq!(c.player.hp, 80 - 13);
    // Champion Belt: applying Vulnerable also applies 1 Weak.
    let mut c = setup("Jaw Worm", &[RelicId::ChampionsBelt], &[CardId::Bash], &[]);
    let m = m0(&c);
    play(&mut c, CardId::Bash, Some(m));
    assert_eq!((power(&c, Cr::Monster(m), PowerId::Vulnerable), power(&c, Cr::Monster(m), PowerId::Weakened)), (2, 1));
    // Hand Drill: breaking an enemy's Block applies 2 Vulnerable.
    let mut c = setup("Jaw Worm", &[RelicId::HandDrill], &[CardId::StrikeR], &[]);
    let m = m0(&c);
    c.monsters[m].block = 3;
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Vulnerable), 2);
}

#[test]
fn gremlin_horn_on_a_kill_that_does_not_end_the_fight() {
    let mut c = setup("2 Louse", &[RelicId::GremlinHorn], &[CardId::StrikeR, CardId::StrikeR], &[CardId::Bash, CardId::Bash]);
    let (a, b) = (c.order[0], c.order[1]);
    c.monsters[a].hp = 1;
    c.monsters[a].powers.clear();
    play(&mut c, CardId::StrikeR, Some(a));
    assert_eq!(c.energy, 3);
    assert_eq!(ids(&c.hand), [CardId::StrikeR, CardId::Bash]);
    c.monsters[b].hp = 1;
    c.monsters[b].powers.clear();
    play(&mut c, CardId::StrikeR, Some(b));
    assert_eq!(c.outcome, Some(Outcome::Victory));
    assert_eq!(c.energy, 2);
}

// ---- Potions, gold, cards obtained ----

#[test]
fn potion_relics_in_combat() {
    // Sacred Bark doubles potency; Toy Ornithopter heals 5 per potion.
    let mut c = setup("Jaw Worm", &[RelicId::SacredBark, RelicId::ToyOrnithopter], &[], &[]);
    c.player.hp = 50;
    c.potions = vec![Some(PotionId::BlockPotion), None, None];
    c.step(&PlayerAction::UsePotion { slot: 0, target: None }).unwrap();
    assert_eq!(c.player.block, 24);
    assert_eq!(c.player.hp, 55);
    // Sozu: no potions obtained.
    let mut c = setup("Jaw Worm", &[RelicId::Sozu], &[], &[]);
    c.potions = vec![Some(PotionId::EntropicBrew), None, None];
    c.step(&PlayerAction::UsePotion { slot: 0, target: None }).unwrap();
    assert!(c.potions.iter().all(|p| p.is_none()));
}

#[test]
fn gold_relics_in_combat() {
    let mut c = setup("Jaw Worm", &[RelicId::BloodyIdol], &[], &[]);
    c.player.hp = 50;
    let gold = c.player.gold;
    run_action(&mut c, Action::GainGold { amount: 20 });
    assert_eq!((c.player.gold, c.player.hp), (gold + 20, 55));
    let mut c = setup("Jaw Worm", &[RelicId::Ectoplasm, RelicId::BloodyIdol], &[], &[]);
    c.player.hp = 50;
    run_action(&mut c, Action::GainGold { amount: 20 });
    assert_eq!((c.player.gold, c.player.hp), (gold, 50));
}

#[test]
fn cards_obtained_in_combat() {
    // Ceramic Fish: 9 gold; Darkstone Periapt: 6 max HP for a curse; Omamori cancels two curses.
    let mut c = setup("Jaw Worm", &[RelicId::CeramicFish, RelicId::DarkstonePeriapt], &[], &[]);
    let gold = c.player.gold;
    run_action(&mut c, Action::AddCardToDeck { card: CardId::Parasite });
    assert_eq!((c.player.gold, c.player.max_hp), (gold + 9, 86));
    assert_eq!(c.master_deck_add, [CardId::Parasite]);
    let mut c = setup("Jaw Worm", &[RelicId::Omamori, RelicId::DarkstonePeriapt], &[], &[]);
    for _ in 0..3 {
        run_action(&mut c, Action::AddCardToDeck { card: CardId::Parasite });
    }
    assert_eq!(c.master_deck_add, [CardId::Parasite]);
    assert_eq!(c.player.max_hp, 86);
    assert!(relic(&c, RelicId::Omamori).used_up);
}

#[test]
fn runic_dome_hides_intents() {
    let o = fight_with("Jaw Worm", &[]).observe();
    assert_eq!(o.monsters[0].intent_damage, 11);
    let o = fight_with("Jaw Worm", &[RelicId::RunicDome]).observe();
    assert_eq!(o.monsters[0].intent, Intent::Unknown);
    assert_eq!(o.monsters[0].intent_damage, -1);
}

/// Relics that do nothing in a fight leave it exactly as it would be without them.
#[test]
fn relics_without_combat_hooks_change_nothing() {
    use RelicId::*;
    let base = serde_json::to_string(&fight_with("Jaw Worm", &[])).unwrap();
    for r in [
        Astrolabe, BlackStar, BottledFlame, CallingBell, Cauldron, Circlet, CultistMask, DollysMirror, DreamCatcher, EmptyCage, EternalFeather,
        FrozenEgg2, FrozenEye, GoldenIdol, JuzuBracelet, Waffle, Mango, Matryoshka, MawBank, MealTicket, MembershipCard, MoltenEgg2, NlothsGift,
        NlothsMask, OldCoin, Orrery, PandorasBox, PeacePipe, Pear, PotionBelt, PrayerWheel, QuestionCard, RegalPillow, Shovel, SingingBowl,
        SmilingMask, SpiritPoop, SsserpentHead, Strawberry, Courier, TinyChest, TinyHouse, ToxicEgg2, WarPaint, Whetstone, WhiteBeast, WingBoots,
    ] {
        let mut c = fight_with("Jaw Worm", &[r]);
        c.player.relics.clear();
        assert_eq!(serde_json::to_string(&c).unwrap(), base, "{r:?}");
    }
}
