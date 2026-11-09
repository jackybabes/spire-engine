//! The Watcher's stances, Mantra, Scry, Retain, cards, powers, relics and potions, checked
//! against the decompiled game (`stances/`, `cards/purple`, `powers/watcher`, the actions they
//! use). Expected numbers come from reading those classes.

use sts_core::action::Cr;
use sts_core::cards::{Card, CardId, PURPLE_LIBRARY_ORDER};
use sts_core::character::Character;
use sts_core::combat::{ChoiceKind, ChoicePile, Combat, CombatConfig, PlayerAction, Stance};
use sts_core::encounters;
use sts_core::potions::PotionId;
use sts_core::powers::PowerId;
use sts_core::relics::{RelicClass, RelicId, RelicState};
use sts_core::rng::StsRandom;
use sts_core::run::{Run, RunAction, RunConfig, Screen};

/// A Watcher fight with `deck` and no relics (so Pure Water adds no Miracle).
fn fight(seed: i64, encounter: &str, deck: &[(CardId, i32)]) -> Combat {
    let mut cfg = CombatConfig::new(Character::Watcher, seed, encounter);
    cfg.deck = deck.to_vec();
    cfg.relics.clear();
    Combat::new(&cfg).unwrap()
}

/// A fight whose hand is exactly `hand` and whose draw pile is `draw` (top last).
fn setup(encounter: &str, hand: &[CardId], draw: &[CardId]) -> Combat {
    let mut c = fight(7, encounter, &[(CardId::DefendP, 0); 5]);
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
    c
}

fn upgraded(c: &mut Combat, id: CardId) -> Card {
    let mut card = c.new_card(id);
    card.upgrade();
    card
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

fn hp(c: &Combat, m: usize) -> i32 {
    c.monsters[m].hp
}

// ---- Fuzz ----

/// Random Watcher decks (every card, some upgraded, her relics now and then) against every
/// encounter, with random legal play: no panics, no illegal states, every playable card played.
#[test]
fn random_watcher_fights_stay_consistent() {
    let mut all: Vec<&str> = encounters::all_act1();
    all.extend(encounters::all_act2());
    all.extend(encounters::all_act3());
    let relics = [
        RelicId::PureWater,
        RelicId::HolyWater,
        RelicId::Damaru,
        RelicId::GoldenEye,
        RelicId::Melange,
        RelicId::TeardropLocket,
        RelicId::VioletLotus,
        RelicId::Duality,
        RelicId::CloakClasp,
    ];
    let mut pick = StsRandom::new(11);
    let mut played = std::collections::HashSet::new();
    let mut stances = std::collections::HashSet::new();
    for seed in 0..600i64 {
        let encounter = all[seed as usize % all.len()];
        let n = 10 + pick.random(10);
        let deck: Vec<(CardId, i32)> =
            (0..n).map(|_| (PURPLE_LIBRARY_ORDER[pick.random(PURPLE_LIBRARY_ORDER.len() as i32 - 1) as usize], pick.random(1))).collect();
        let mut cfg = CombatConfig::new(Character::Watcher, seed, encounter);
        cfg.deck = deck.clone();
        cfg.player_hp = 300;
        cfg.player_max_hp = 300;
        cfg.relics.push(relics[seed as usize % relics.len()]);
        cfg.potions = vec![Some(PotionId::Ambrosia), Some(PotionId::BottledMiracle), Some(PotionId::StancePotion)];
        let mut c = Combat::new(&cfg).unwrap();
        let mut steps = 0;
        while !c.is_over() && steps < 2000 {
            let legal = c.legal_actions();
            assert!(!legal.is_empty(), "seed {seed} ({encounter}): no legal actions");
            let a = legal[pick.random(legal.len() as i32 - 1) as usize].clone();
            if let PlayerAction::PlayCard { hand_index, .. } = a {
                played.insert(c.hand[hand_index].id);
            }
            c.step(&a).unwrap_or_else(|e| panic!("seed {seed} ({encounter}) {deck:?}: {a:?}: {e}"));
            assert!(c.hand.len() <= 10 && c.energy >= 0 && c.player.hp >= 0, "seed {seed}");
            stances.insert(c.player.stance);
            steps += 1;
        }
    }
    // Deus Ex Machina can't be played: it exhausts itself when drawn.
    let missing: Vec<CardId> = PURPLE_LIBRARY_ORDER.iter().copied().filter(|&id| id != CardId::DeusExMachina && !played.contains(&id)).collect();
    assert!(missing.is_empty(), "never played: {missing:?}");
    assert_eq!(stances.len(), 4, "{stances:?}");
}

// ---- Stances ----

#[test]
fn a_watcher_fight_starts_neutral_with_pure_waters_miracle() {
    let c = Combat::new(&CombatConfig::watcher(3, "Jaw Worm")).unwrap();
    assert_eq!((c.player.hp, c.player.max_hp), (72, 72));
    assert_eq!(c.player.stance, Stance::Neutral);
    // Pure Water's Miracle comes before the opening draw: six cards.
    assert_eq!(c.hand.len(), 6);
    assert!(c.hand.iter().any(|card| card.id == CardId::Miracle));
}

#[test]
fn wrath_doubles_damage_both_ways() {
    let mut c = setup("Jaw Worm", &[CardId::Eruption, CardId::StrikeP], &[]);
    let m = m0(&c);
    let intent = c.monsters[m].intent_dmg;
    let before = hp(&c, m);
    // Eruption hits for its 9 before Wrath begins.
    play(&mut c, CardId::Eruption, Some(m));
    assert_eq!(c.player.stance, Stance::Wrath);
    assert_eq!(hp(&c, m), before - 9);
    play(&mut c, CardId::StrikeP, Some(m));
    assert_eq!(hp(&c, m), before - 9 - 12);
    // The monster's intent and damage double too (onModifyPower).
    assert_eq!(c.monsters[m].intent_dmg, intent * 2);
}

#[test]
fn leaving_calm_gives_two_energy_and_violet_lotus_one_more() {
    let mut c = setup("Jaw Worm", &[CardId::Vigilance, CardId::Crescendo], &[]);
    play(&mut c, CardId::Vigilance, None);
    assert_eq!((c.player.stance, c.player.block, c.energy), (Stance::Calm, 8, 1));
    play(&mut c, CardId::Crescendo, None);
    assert_eq!((c.player.stance, c.energy), (Stance::Wrath, 2));

    let mut c = setup("Jaw Worm", &[CardId::Vigilance, CardId::Crescendo], &[]);
    c.player.relics.push(RelicState::new(RelicId::VioletLotus));
    play(&mut c, CardId::Vigilance, None);
    play(&mut c, CardId::Crescendo, None);
    assert_eq!(c.energy, 3);
}

#[test]
fn divinity_triples_damage_gives_energy_and_ends_next_turn() {
    let mut c = setup("Cultist", &[CardId::Blasphemy, CardId::StrikeP], &[CardId::DefendP; 6]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::Blasphemy, None);
    assert_eq!((c.player.stance, c.energy), (Stance::Divinity, 5));
    play(&mut c, CardId::StrikeP, Some(m));
    assert_eq!(hp(&c, m), before - 18);
    // Blasphemy: the next turn starts with death.
    end_turn(&mut c);
    assert!(c.player.is_dead);
}

#[test]
fn divinity_leaves_at_the_start_of_the_next_turn() {
    let mut c = setup("Cultist", &[CardId::Worship, CardId::Worship], &[CardId::DefendP; 10]);
    c.energy = 4;
    play(&mut c, CardId::Worship, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Mantra), 5);
    play(&mut c, CardId::Worship, None);
    // Mantra reached 10: Divinity, and the power goes (10 - 10 = 0).
    assert_eq!(c.player.stance, Stance::Divinity);
    assert!(!c.player.has_power(PowerId::Mantra));
    assert_eq!(c.mantra_gained, 10);
    end_turn(&mut c);
    assert_eq!(c.player.stance, Stance::Neutral);
}

#[test]
fn mantra_over_ten_keeps_the_rest() {
    let mut c = setup("Cultist", &[CardId::Worship, CardId::Pray, CardId::Prostrate, CardId::Prostrate], &[CardId::DefendP; 10]);
    c.energy = 10;
    play(&mut c, CardId::Worship, None);
    play(&mut c, CardId::Pray, None);
    play(&mut c, CardId::Prostrate, None);
    assert_eq!((c.player.stance, power(&c, Cr::Player, PowerId::Mantra)), (Stance::Divinity, 0));
    // 5 + 3 = 8, + 2 = 10: Divinity with nothing left; then 2 more start again.
    play(&mut c, CardId::Prostrate, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Mantra), 2);
    // Pray shuffles an Insight into the draw pile.
    assert_eq!(c.draw_pile.iter().filter(|card| card.id == CardId::Insight).count(), 1);
}

#[test]
fn mental_fortress_rushdown_and_flurry_react_to_stance_changes() {
    let mut c = setup("Jaw Worm", &[CardId::MentalFortress, CardId::Rushdown, CardId::Crescendo, CardId::EmptyBody], &[CardId::DefendP; 6]);
    c.energy = 10;
    let flurry = c.new_card(CardId::FlurryOfBlows);
    c.discard_pile.push(flurry);
    play(&mut c, CardId::MentalFortress, None);
    play(&mut c, CardId::Rushdown, None);
    play(&mut c, CardId::Crescendo, None);
    // Wrath: 4 block, 2 cards, and Flurry of Blows back in hand.
    assert_eq!(c.player.block, 4);
    assert_eq!(c.hand.iter().filter(|card| card.id == CardId::DefendP).count(), 2);
    assert!(c.hand.iter().any(|card| card.id == CardId::FlurryOfBlows));
    // Back to Neutral: Mental Fortress again, Rushdown only on Wrath.
    play(&mut c, CardId::EmptyBody, None);
    assert_eq!(c.player.block, 4 + 7 + 4);
    assert_eq!(c.hand.iter().filter(|card| card.id == CardId::DefendP).count(), 2);
}

#[test]
fn changing_to_the_same_stance_does_nothing() {
    let mut c = setup("Jaw Worm", &[CardId::MentalFortress, CardId::Crescendo, CardId::Crescendo], &[]);
    c.energy = 10;
    play(&mut c, CardId::MentalFortress, None);
    play(&mut c, CardId::Crescendo, None);
    play(&mut c, CardId::Crescendo, None);
    assert_eq!(c.player.block, 4);
}

#[test]
fn brilliance_adds_the_mantra_made_this_combat() {
    let mut c = setup("Jaw Worm", &[CardId::Prostrate, CardId::Brilliance], &[]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::Prostrate, None);
    play(&mut c, CardId::Brilliance, Some(m));
    assert_eq!(hp(&c, m), before - 14);
}

#[test]
fn devotion_and_damaru_make_mantra_each_turn() {
    let mut c = setup("Cultist", &[CardId::Devotion], &[CardId::DefendP; 10]);
    c.player.relics.push(RelicState::new(RelicId::Damaru));
    play(&mut c, CardId::Devotion, None);
    end_turn(&mut c);
    // Damaru's 1 at the turn start, Devotion's 2 after the draw.
    assert_eq!(power(&c, Cr::Player, PowerId::Mantra), 3);
}

// ---- Scry ----

#[test]
fn scry_shows_the_top_cards_and_discards_the_picks() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[ThirdEye], &[StrikeP, DefendP, Weave, Eruption]);
    play(&mut c, ThirdEye, None);
    assert_eq!(c.player.block, 7);
    let p = c.pending.as_ref().expect("scry grid");
    assert!(matches!(p.kind, ChoiceKind::Scry) && p.pile == ChoicePile::Draw && p.any_number && p.can_pick_zero);
    // Top card first: Eruption, Weave, Defend.
    assert_eq!(p.options.iter().map(|&i| c.draw_pile[i].id).collect::<Vec<_>>(), [Eruption, Weave, DefendP]);
    let (eruption, weave) = (p.options[0], p.options[1]);
    choose(&mut c, &[eruption, weave]);
    assert_eq!(ids(&c.draw_pile), [StrikeP, DefendP]);
    // Weave comes back from the discard pile after the scry; Third Eye follows Eruption there.
    assert_eq!(ids(&c.discard_pile), [Eruption, ThirdEye]);
    assert!(c.hand.iter().any(|card| card.id == Weave));
}

#[test]
fn scrying_nothing_is_allowed_and_nirvana_blocks_every_scry() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Nirvana, JustLucky, JustLucky], &[StrikeP, DefendP]);
    let m = m0(&c);
    play(&mut c, Nirvana, None);
    play(&mut c, JustLucky, Some(m));
    choose(&mut c, &[]);
    assert_eq!(c.player.block, 3 + 2);
    assert_eq!(c.draw_pile.len(), 2);
    // With an empty draw pile there is no grid, but Nirvana still blocks.
    c.draw_pile.clear();
    play(&mut c, JustLucky, Some(m));
    assert!(c.pending.is_none());
    assert_eq!(c.player.block, 2 * (3 + 2));
}

#[test]
fn golden_eye_scries_two_more_and_melange_scries_on_shuffle() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[CutThroughFate], &[DefendP; 8]);
    c.player.relics.push(RelicState::new(RelicId::GoldenEye));
    let m = m0(&c);
    play(&mut c, CutThroughFate, Some(m));
    assert_eq!(c.pending.as_ref().unwrap().options.len(), 4);
    choose(&mut c, &[]);

    let mut c = setup("Jaw Worm", &[WheelKick], &[]);
    c.player.relics.push(RelicState::new(RelicId::Melange));
    for _ in 0..5 {
        let d = c.new_card(DefendP);
        c.discard_pile.push(d);
    }
    let m = m0(&c);
    play(&mut c, WheelKick, Some(m));
    // The shuffle for the draw runs Melange's scry 3, after the draw.
    assert_eq!(c.hand.len(), 2);
    assert_eq!(c.pending.as_ref().unwrap().options.len(), 3);
}

#[test]
fn foresight_scries_at_the_start_of_each_turn() {
    use CardId::*;
    let mut c = setup("Cultist", &[Foresight], &[DefendP; 12]);
    play(&mut c, Foresight, None);
    end_turn(&mut c);
    let p = c.pending.as_ref().expect("Foresight's scry");
    assert_eq!(p.options.len(), 3);
}

// ---- Retain ----

#[test]
fn retained_cards_stay_and_grow() {
    use CardId::*;
    let mut c = setup("Cultist", &[Protect, Perseverance, WindmillStrike, SandsOfTime, Miracle, StrikeP], &[DefendP; 10]);
    end_turn(&mut c);
    let held: Vec<CardId> = ids(&c.hand).into_iter().filter(|&id| id != DefendP).collect();
    assert_eq!(held, [Protect, Perseverance, WindmillStrike, SandsOfTime, Miracle]);
    let card = |c: &Combat, id: CardId| c.hand.iter().find(|card| card.id == id).unwrap().clone();
    assert_eq!(card(&c, Perseverance).base_block, 5 + 2);
    assert_eq!(card(&c, WindmillStrike).base_damage, 7 + 4);
    assert_eq!(card(&c, SandsOfTime).cost, 3);
    assert_eq!(c.discard_pile.iter().map(|card| card.id).collect::<Vec<_>>(), [StrikeP]);
}

#[test]
fn establishment_cheapens_retained_cards() {
    use CardId::*;
    let mut c = setup("Cultist", &[Establishment, Protect], &[DefendP; 10]);
    play(&mut c, Establishment, None);
    end_turn(&mut c);
    assert_eq!(c.hand.iter().find(|card| card.id == Protect).unwrap().cost, 1);
}

#[test]
fn meditate_returns_cards_retained_enters_calm_and_ends_the_turn() {
    use CardId::*;
    let mut c = setup("Cultist", &[Meditate], &[DefendP; 10]);
    for id in [StrikeP, Eruption, Vigilance] {
        let card = c.new_card(id);
        c.discard_pile.push(card);
    }
    play(&mut c, Meditate, None);
    let p = c.pending.as_ref().expect("Meditate's grid");
    assert_eq!((p.pile, p.count), (ChoicePile::Discard, 1));
    choose(&mut c, &[1]);
    // The turn ended: Eruption was retained, and the next turn starts in Calm.
    assert_eq!(c.turn, 2);
    assert_eq!(c.player.stance, Stance::Calm);
    assert!(c.hand.iter().any(|card| card.id == Eruption));
    assert!(!c.hand.iter().any(|card| card.id == StrikeP));
}

/// Meditate's picks that don't fit in the hand stay where they are in the discard pile: the game
/// moves a pick only while the hand has room (run H141 of the 2026-10-28 Watcher soak).
#[test]
fn meditate_leaves_a_pick_that_does_not_fit_in_place() {
    use CardId::*;
    let mut c = setup("Cultist", &[Meditate, Miracle, Miracle, Miracle, Miracle, Miracle, Miracle, Miracle, Miracle, Miracle], &[DefendP; 10]);
    c.hand[0].upgrade();
    for id in [StrikeP, Eruption, Vigilance, Halt] {
        let card = c.new_card(id);
        c.discard_pile.push(card);
    }
    play(&mut c, Meditate, None);
    assert_eq!(c.pending.as_ref().expect("Meditate's grid").count, 2);
    choose(&mut c, &[1, 2]);
    assert!(c.hand.iter().any(|card| card.id == Eruption));
    let at = |id: CardId| c.discard_pile.iter().position(|card| card.id == id).unwrap();
    assert!(at(StrikeP) < at(Vigilance) && at(Vigilance) < at(Halt), "{:?}", ids(&c.discard_pile));
}

// ---- Cards that look at the stance or the previous card ----

#[test]
fn halt_blocks_more_in_wrath() {
    let mut c = setup("Jaw Worm", &[CardId::Halt, CardId::Halt], &[]);
    play(&mut c, CardId::Halt, None);
    assert_eq!(c.player.block, 3);
    c.player.stance = Stance::Wrath;
    play(&mut c, CardId::Halt, None);
    assert_eq!(c.player.block, 3 + 3 + 9);
    let mut c = setup("Jaw Worm", &[], &[]);
    let h = upgraded(&mut c, CardId::Halt);
    assert_eq!((h.base_block, h.base_magic), (4, 14));
}

#[test]
fn indignation_inner_peace_and_fear_no_evil() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Indignation, Indignation, InnerPeace, InnerPeace, FearNoEvil], &[DefendP; 5]);
    c.energy = 10;
    let m = m0(&c);
    play(&mut c, Indignation, None);
    assert_eq!(c.player.stance, Stance::Wrath);
    play(&mut c, Indignation, None);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Vulnerable), 3);
    play(&mut c, InnerPeace, None);
    assert_eq!(c.player.stance, Stance::Calm);
    play(&mut c, InnerPeace, None);
    assert_eq!(c.hand.iter().filter(|card| card.id == DefendP).count(), 3);
    // Jaw Worm attacks first: Fear No Evil enters Calm (already Calm here: no change).
    c.player.stance = Stance::Neutral;
    play(&mut c, FearNoEvil, Some(m));
    assert_eq!(c.player.stance, Stance::Calm);
}

#[test]
fn follow_up_crush_joints_sash_whip_and_sanctity_check_the_card_before() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[StrikeP, FollowUp, SashWhip, DefendP, CrushJoints, Sanctity, Sanctity], &[DefendP; 5]);
    c.energy = 10;
    let m = m0(&c);
    play(&mut c, StrikeP, Some(m));
    play(&mut c, FollowUp, Some(m));
    assert_eq!(c.energy, 10 - 1 - 1 + 1);
    play(&mut c, SashWhip, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 1);
    play(&mut c, DefendP, None);
    play(&mut c, CrushJoints, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Vulnerable), 1);
    // After an attack Sanctity only blocks; after a skill it draws 2.
    play(&mut c, Sanctity, None);
    let hand = c.hand.len();
    play(&mut c, Sanctity, None);
    assert_eq!(c.hand.len(), hand - 1 + 2);
}

#[test]
fn signature_move_needs_no_other_attack() {
    let c = setup("Jaw Worm", &[CardId::SignatureMove, CardId::StrikeP], &[]);
    let m = m0(&c);
    assert!(!c.can_use(&c.hand[0], Some(m)));
    let c = setup("Jaw Worm", &[CardId::SignatureMove, CardId::DefendP], &[]);
    assert!(c.can_use(&c.hand[0], Some(m)));
}

#[test]
fn spirit_shield_counts_the_other_cards_in_hand() {
    let mut c = setup("Jaw Worm", &[CardId::SpiritShield, CardId::DefendP, CardId::DefendP, CardId::StrikeP], &[]);
    play(&mut c, CardId::SpiritShield, None);
    assert_eq!(c.player.block, 9);
}

#[test]
fn bowling_bash_hits_once_per_enemy() {
    let mut c = setup("3 Sentries", &[CardId::BowlingBash], &[]);
    let m = c.order[1];
    let before = hp(&c, m);
    play(&mut c, CardId::BowlingBash, Some(m));
    assert_eq!(hp(&c, m), before - 21);
}

// ---- Other cards ----

#[test]
fn judgement_kills_at_or_below_its_number() {
    let mut c = setup("Cultist", &[CardId::Judgement, CardId::Judgement], &[]);
    let m = m0(&c);
    c.monsters[m].hp = 31;
    play(&mut c, CardId::Judgement, Some(m));
    assert_eq!(hp(&c, m), 31);
    c.monsters[m].hp = 30;
    play(&mut c, CardId::Judgement, Some(m));
    assert!(c.is_over());
}

#[test]
fn lesson_learned_upgrades_a_master_deck_card_on_a_kill() {
    let mut c = fight(5, "Cultist", &[(CardId::LessonLearned, 0), (CardId::StrikeP, 0), (CardId::StrikeP, 0), (CardId::DefendP, 0), (CardId::DefendP, 0)]);
    let m = m0(&c);
    c.monsters[m].hp = 5;
    play(&mut c, CardId::LessonLearned, Some(m));
    assert!(c.is_over());
    assert_eq!(c.master_deck_upgrades.len(), 1);
    let up = c.master_deck_upgrades[0];
    assert!(c.master_deck.iter().find(|card| card.uuid == up).unwrap().upgraded);
}

#[test]
fn wallop_blocks_the_unblocked_damage() {
    let mut c = setup("Jaw Worm", &[CardId::Wallop, CardId::Wallop], &[]);
    c.energy = 4;
    let m = m0(&c);
    play(&mut c, CardId::Wallop, Some(m));
    assert_eq!(c.player.block, 9);
    c.monsters[m].block = 5;
    play(&mut c, CardId::Wallop, Some(m));
    assert_eq!(c.player.block, 9 + 4);
}

#[test]
fn talk_to_the_hand_and_pressure_points_mark_their_target() {
    use CardId::*;
    let mut c = setup("Cultist", &[TalkToTheHand, StrikeP, PressurePoints, PressurePoints], &[]);
    c.energy = 10;
    let m = m0(&c);
    play(&mut c, TalkToTheHand, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::BlockReturn), 2);
    play(&mut c, StrikeP, Some(m));
    assert_eq!(c.player.block, 2);
    let before = hp(&c, m);
    play(&mut c, PressurePoints, Some(m));
    play(&mut c, PressurePoints, Some(m));
    assert_eq!(hp(&c, m), before - 8 - 16);
}

#[test]
fn tantrum_goes_back_into_the_draw_pile() {
    let mut c = setup("Jaw Worm", &[CardId::Tantrum], &[CardId::DefendP; 4]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::Tantrum, Some(m));
    assert_eq!(hp(&c, m), before - 9);
    assert_eq!(c.player.stance, Stance::Wrath);
    assert!(c.draw_pile.iter().any(|card| card.id == CardId::Tantrum));
    assert!(c.discard_pile.is_empty());
}

/// A Tantrum free for the turn (Attack Potion's pick) costs 1 again once shuffled back: every move
/// into the draw pile resets the card.
#[test]
fn tantrum_shuffled_back_costs_one_again() {
    let mut c = setup("Jaw Worm", &[CardId::Tantrum], &[CardId::DefendP; 4]);
    c.hand[0].set_cost_for_turn(0);
    let m = m0(&c);
    play(&mut c, CardId::Tantrum, Some(m));
    let t = c.draw_pile.iter().find(|card| card.id == CardId::Tantrum).unwrap();
    assert_eq!(t.cost_for_turn, 1);
}

#[test]
fn conclude_ends_the_turn_and_vault_skips_the_monsters() {
    let mut c = setup("Jaw Worm", &[CardId::Conclude, CardId::DefendP], &[CardId::DefendP; 10]);
    play(&mut c, CardId::Conclude, None);
    assert_eq!(c.turn, 2);
    assert!(c.player.hp < 72);

    let mut c = setup("Jaw Worm", &[CardId::Vault], &[CardId::DefendP; 10]);
    let m = m0(&c);
    let history = c.monsters[m].move_history.len();
    play(&mut c, CardId::Vault, None);
    assert_eq!((c.turn, c.player.hp), (2, 72));
    assert_eq!(c.monsters[m].move_history.len(), history);
}

#[test]
fn omniscience_plays_a_draw_pile_card_twice_and_exhausts_it() {
    let mut c = setup("Jaw Worm", &[CardId::Omniscience], &[CardId::DefendP, CardId::Vigilance]);
    c.energy = 4;
    play(&mut c, CardId::Omniscience, None);
    let p = c.pending.as_ref().unwrap();
    assert!(matches!(p.kind, ChoiceKind::Omniscience { amount: 2 }));
    let k = *p.options.iter().find(|&&i| c.draw_pile[i].id == CardId::Vigilance).unwrap();
    choose(&mut c, &[k]);
    assert_eq!(c.player.block, 16);
    assert_eq!(c.player.stance, Stance::Calm);
    assert_eq!(c.energy, 0);
    assert!(c.exhaust_pile.iter().any(|card| card.id == CardId::Vigilance));
}

#[test]
fn wish_and_stance_potion_choose_one_option() {
    let mut c = setup("Jaw Worm", &[CardId::Wish, CardId::Wish, CardId::Wish], &[]);
    c.energy = 9;
    for (pick, check) in [(0usize, 0usize), (1, 1), (2, 2)] {
        play(&mut c, CardId::Wish, None);
        let p = c.pending.as_ref().unwrap();
        assert_eq!(ids(&p.offer), [CardId::BecomeAlmighty, CardId::FameAndFortune, CardId::LiveForever]);
        let gold = c.player.gold;
        choose(&mut c, &[pick]);
        match check {
            0 => assert_eq!(power(&c, Cr::Player, PowerId::Strength), 3),
            1 => assert_eq!(c.player.gold, gold + 25),
            _ => assert_eq!(power(&c, Cr::Player, PowerId::PlatedArmor), 6),
        }
    }
    let mut c = setup("Jaw Worm", &[], &[]);
    c.potions = vec![Some(PotionId::StancePotion), Some(PotionId::Ambrosia), Some(PotionId::BottledMiracle)];
    c.step(&PlayerAction::UsePotion { slot: 0, target: None }).unwrap();
    assert_eq!(ids(&c.pending.as_ref().unwrap().offer), [CardId::ChooseWrath, CardId::ChooseCalm]);
    choose(&mut c, &[1]);
    assert_eq!(c.player.stance, Stance::Calm);
    c.step(&PlayerAction::UsePotion { slot: 1, target: None }).unwrap();
    assert_eq!((c.player.stance, c.energy), (Stance::Divinity, 3 + 2 + 3));
    c.step(&PlayerAction::UsePotion { slot: 2, target: None }).unwrap();
    assert_eq!(c.hand.iter().filter(|card| card.id == CardId::Miracle).count(), 2);
}

#[test]
fn foreign_influence_offers_three_different_attacks_of_any_color() {
    let mut c = setup("Jaw Worm", &[], &[]);
    let fi = upgraded(&mut c, CardId::ForeignInfluence);
    c.hand.push(fi);
    let card_rng = c.rng.card.counter;
    play(&mut c, CardId::ForeignInfluence, None);
    let p = c.pending.as_ref().unwrap();
    assert!(p.can_pick_zero);
    let offer = ids(&p.offer);
    assert_eq!(offer.len(), 3);
    assert!(offer.iter().all(|id| id.data().card_type == sts_core::cards::CardType::Attack));
    assert!(c.rng.card.counter >= card_rng + 3);
    choose(&mut c, &[0]);
    // Foreign Influence+: free this turn.
    assert_eq!(c.hand[0].id, offer[0]);
    assert_eq!(c.hand[0].cost_for_turn, 0);
}

#[test]
fn collect_and_conjure_blade_spend_all_energy() {
    let mut c = setup("Cultist", &[CardId::Collect], &[CardId::DefendP; 10]);
    play(&mut c, CardId::Collect, None);
    assert_eq!((c.energy, power(&c, Cr::Player, PowerId::Collect)), (0, 3));
    end_turn(&mut c);
    let miracles: Vec<&Card> = c.hand.iter().filter(|card| card.id == CardId::Miracle).collect();
    assert_eq!(miracles.len(), 1);
    assert!(miracles[0].upgraded);
    assert_eq!(power(&c, Cr::Player, PowerId::Collect), 2);

    let mut c = setup("Cultist", &[], &[CardId::DefendP; 3]);
    let cb = upgraded(&mut c, CardId::ConjureBlade);
    c.hand.push(cb);
    play(&mut c, CardId::ConjureBlade, None);
    let e = c.draw_pile.iter().find(|card| card.id == CardId::Expunger).expect("an Expunger");
    assert_eq!((e.magic, c.energy), (4, 0));
}

#[test]
fn expunger_hits_its_x_times() {
    let mut c = setup("Jaw Worm", &[], &[]);
    let mut e = c.new_card(CardId::Expunger);
    e.magic = 3;
    e.base_magic = 3;
    // A copy keeps the X (Expunger's makeStatEquivalentCopy).
    let copy = c.stat_copy(&e);
    assert_eq!(copy.magic, 3);
    c.hand.push(e);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::Expunger, Some(m));
    assert_eq!(hp(&c, m), before - 27);
}

#[test]
fn deva_form_gives_more_energy_each_turn() {
    let mut c = setup("Cultist", &[CardId::DevaForm], &[CardId::DefendP; 15]);
    play(&mut c, CardId::DevaForm, None);
    end_turn(&mut c);
    assert_eq!(c.energy, 4);
    end_turn(&mut c);
    assert_eq!(c.energy, 5);
}

#[test]
fn fasting_trades_an_energy_for_strength_and_dexterity() {
    let mut c = setup("Cultist", &[CardId::Fasting], &[CardId::DefendP; 15]);
    play(&mut c, CardId::Fasting, None);
    assert_eq!((power(&c, Cr::Player, PowerId::Strength), power(&c, Cr::Player, PowerId::Dexterity)), (3, 3));
    end_turn(&mut c);
    assert_eq!(c.energy, 2);
}

#[test]
fn swivel_makes_the_next_attack_free() {
    let mut c = setup("Jaw Worm", &[CardId::Swivel, CardId::SandsOfTime], &[]);
    let m = m0(&c);
    play(&mut c, CardId::Swivel, None);
    assert_eq!((c.player.block, c.energy), (8, 1));
    play(&mut c, CardId::SandsOfTime, Some(m));
    assert_eq!(c.energy, 1);
    assert!(!c.player.has_power(PowerId::FreeAttack));
}

#[test]
fn wave_of_the_hand_weakens_on_block_and_like_water_blocks_in_calm() {
    let mut c = setup("Cultist", &[CardId::WaveOfTheHand, CardId::DefendP, CardId::LikeWater, CardId::Tranquility], &[CardId::DefendP; 10]);
    c.energy = 10;
    let m = m0(&c);
    play(&mut c, CardId::WaveOfTheHand, None);
    play(&mut c, CardId::DefendP, None);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 1);
    end_turn(&mut c);
    assert!(!c.player.has_power(PowerId::WaveOfTheHand));

    // Like Water: 5 block at the end of a turn in Calm, against Chomp's 11.
    let mut c = setup("Jaw Worm", &[CardId::LikeWater, CardId::Tranquility], &[CardId::DefendP; 10]);
    play(&mut c, CardId::LikeWater, None);
    play(&mut c, CardId::Tranquility, None);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 72 - 6);
}

#[test]
fn beta_evaluate_and_reach_heaven_shuffle_their_cards_in() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Beta, Evaluate, ReachHeaven], &[DefendP; 4]);
    c.energy = 10;
    let m = m0(&c);
    play(&mut c, Beta, None);
    play(&mut c, Evaluate, None);
    play(&mut c, ReachHeaven, Some(m));
    let made: Vec<CardId> = ids(&c.draw_pile).into_iter().filter(|&id| id != DefendP).collect();
    let mut sorted = made.clone();
    sorted.sort();
    assert_eq!(sorted, [Insight, Omega, ThroughViolence]);
}

#[test]
fn study_alpha_battle_hymn_and_simmering_fury_make_cards_later() {
    use CardId::*;
    let mut c = setup("Cultist", &[Study, Alpha, BattleHymn, SimmeringFury], &[DefendP; 15]);
    c.energy = 10;
    play(&mut c, Study, None);
    play(&mut c, Alpha, None);
    assert!(c.draw_pile.iter().any(|card| card.id == Beta));
    play(&mut c, BattleHymn, None);
    play(&mut c, SimmeringFury, None);
    end_turn(&mut c);
    assert!(c.draw_pile.iter().any(|card| card.id == Insight) || c.hand.iter().any(|card| card.id == Insight));
    assert!(c.hand.iter().any(|card| card.id == Smite));
    assert_eq!(c.player.stance, Stance::Wrath);
    // 5 + Simmering Fury's 2.
    assert_eq!(c.hand.iter().filter(|card| card.id != Smite).count(), 7);
}

#[test]
fn omega_burns_every_enemy_at_the_end_of_the_turn() {
    let mut c = setup("Cultist", &[], &[CardId::DefendP; 10]);
    let omega = c.new_card(CardId::Omega);
    c.hand.push(omega);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::Omega, None);
    end_turn(&mut c);
    assert_eq!(hp(&c, m), (before - 50).max(0));
}

#[test]
fn master_reality_upgrades_cards_made_in_combat() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[MasterReality, CarveReality, DeceiveReality], &[]);
    c.energy = 10;
    let m = m0(&c);
    play(&mut c, MasterReality, None);
    play(&mut c, CarveReality, Some(m));
    play(&mut c, DeceiveReality, None);
    assert!(c.hand.iter().filter(|card| matches!(card.id, Smite | Safety)).all(|card| card.upgraded));
    assert_eq!(c.hand.len(), 2);
}

#[test]
fn deus_ex_machina_exhausts_itself_for_miracles_when_drawn() {
    let mut c = setup("Jaw Worm", &[CardId::EmptyMind], &[CardId::DefendP, CardId::DeusExMachina]);
    play(&mut c, CardId::EmptyMind, None);
    assert_eq!(c.hand.iter().filter(|card| card.id == CardId::Miracle).count(), 2);
    assert!(c.exhaust_pile.iter().any(|card| card.id == CardId::DeusExMachina));
}

#[test]
fn scrawl_draws_to_ten_and_ragnarok_hits_five_times() {
    let mut c = setup("Jaw Worm", &[CardId::Scrawl, CardId::Ragnarok], &[CardId::DefendP; 12]);
    c.energy = 5;
    play(&mut c, CardId::Scrawl, None);
    assert_eq!(c.hand.len(), 10);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::Ragnarok, None);
    assert_eq!(hp(&c, m), before - 25);
}

#[test]
fn simple_watcher_cards() {
    use CardId::*;
    // (card, damage dealt to Jaw Worm, block gained, cards drawn from a pile of Defends)
    let cases = [
        (StrikeP, 6, 0, 0),
        (DefendP, 0, 5, 0),
        (Consecrate, 5, 0, 0),
        (FlyingSleeves, 8, 0, 0),
        (WheelKick, 15, 0, 2),
        (Smite, 12, 0, 0),
        (ThroughViolence, 20, 0, 0),
        (Safety, 0, 12, 0),
        (Protect, 0, 12, 0),
        (Insight, 0, 0, 2),
        (EmptyFist, 9, 0, 0),
        (SashWhip, 8, 0, 0),
        (CrushJoints, 8, 0, 0),
        (FollowUp, 7, 0, 0),
        (FlurryOfBlows, 4, 0, 0),
        (Weave, 4, 0, 0),
        (SignatureMove, 30, 0, 0),
        (ReachHeaven, 10, 0, 0),
        (CarveReality, 6, 0, 0),
        (Evaluate, 0, 6, 0),
        (DeceiveReality, 0, 4, 0),
        (Perseverance, 0, 5, 0),
        (Prostrate, 0, 4, 0),
        (Sanctity, 0, 6, 0),
    ];
    for (id, dmg, blk, drawn) in cases {
        let mut c = setup("Jaw Worm", &[id], &[DefendP; 6]);
        c.energy = 5;
        let m = m0(&c);
        let before = hp(&c, m);
        play(&mut c, id, Some(m));
        if let Some(p) = c.pending.clone() {
            let _ = p;
            choose(&mut c, &[]);
        }
        assert_eq!(before - hp(&c, m), dmg, "{id:?} damage");
        assert_eq!(c.player.block, blk, "{id:?} block");
        assert_eq!(c.hand.iter().filter(|card| card.id == DefendP).count() as i32, drawn, "{id:?} draw");
    }
    // Miracle: +1 energy (2 upgraded).
    let mut c = setup("Jaw Worm", &[Miracle], &[]);
    play(&mut c, Miracle, None);
    assert_eq!(c.energy, 4);
    // Wreath of Flame: Vigor for the next attack.
    let mut c = setup("Jaw Worm", &[WreathOfFlame, StrikeP], &[]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, WreathOfFlame, None);
    play(&mut c, StrikeP, Some(m));
    assert_eq!(hp(&c, m), before - 11);
}

// ---- Relics ----

fn fight_with(encounter: &str, relics: &[RelicId], deck: &[(CardId, i32)]) -> Combat {
    let mut cfg = CombatConfig::new(Character::Watcher, 7, encounter);
    cfg.deck = deck.to_vec();
    cfg.relics = relics.to_vec();
    Combat::new(&cfg).unwrap()
}

#[test]
fn holy_water_teardrop_locket_cloak_clasp_and_duality() {
    let deck = [(CardId::DefendP, 0); 10];
    let c = fight_with("Jaw Worm", &[RelicId::HolyWater], &deck);
    assert_eq!(c.hand.iter().filter(|card| card.id == CardId::Miracle).count(), 3);
    assert_eq!(c.hand.len(), 8);

    let c = fight_with("Jaw Worm", &[RelicId::TeardropLocket], &deck);
    assert_eq!(c.player.stance, Stance::Calm);

    // Cloak Clasp: 1 block per card in hand at the end of the turn, against Chomp's 11.
    let mut c = fight_with("Jaw Worm", &[RelicId::CloakClasp], &deck);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 72 - (11 - 5));

    let mut c = setup("Jaw Worm", &[CardId::StrikeP, CardId::DefendP], &[]);
    c.player.relics.push(RelicState::new(RelicId::Duality));
    let m = m0(&c);
    play(&mut c, CardId::StrikeP, Some(m));
    assert_eq!(power(&c, Cr::Player, PowerId::Dexterity), 1);
    play(&mut c, CardId::DefendP, None);
    assert_eq!(c.player.block, 6);
    assert!(c.player.has_power(PowerId::LoseDexterity));
}

// ---- Upgrades ----

#[test]
fn upgrades_match_each_cards_upgrade_method() {
    use CardId::*;
    // (card, base damage, base block, base magic, cost) after one upgrade.
    let table = [
        (Alpha, 0, 0, 0, 1),
        (BattleHymn, 0, 0, 1, 1),
        (Blasphemy, 0, 0, 0, 1),
        (BowlingBash, 10, 0, 0, 1),
        (Brilliance, 16, 0, 0, 1),
        (CarveReality, 10, 0, 0, 1),
        (Collect, 0, 0, 0, -1),
        (Conclude, 16, 0, 0, 1),
        (ConjureBlade, 0, 0, 0, -1),
        (Consecrate, 8, 0, 0, 0),
        (Crescendo, 0, 0, 0, 0),
        (CrushJoints, 10, 0, 2, 1),
        (CutThroughFate, 9, 0, 3, 1),
        (DeceiveReality, 0, 7, 0, 1),
        (DefendP, 0, 8, 0, 1),
        (DeusExMachina, 0, 0, 3, -2),
        (DevaForm, 0, 0, 1, 3),
        (Devotion, 0, 0, 3, 1),
        (EmptyBody, 0, 10, 0, 1),
        (EmptyFist, 14, 0, 0, 1),
        (EmptyMind, 0, 0, 3, 1),
        (Eruption, 9, 0, 0, 1),
        (Establishment, 0, 0, 1, 1),
        (Evaluate, 0, 10, 0, 1),
        (Fasting, 0, 0, 4, 2),
        (FearNoEvil, 11, 0, 0, 1),
        (FlurryOfBlows, 6, 0, 0, 0),
        (FlyingSleeves, 6, 0, 0, 1),
        (FollowUp, 11, 0, 0, 1),
        (ForeignInfluence, 0, 0, 0, 0),
        (Foresight, 0, 0, 4, 1),
        (Halt, 0, 4, 14, 0),
        (Indignation, 0, 0, 5, 1),
        (InnerPeace, 0, 0, 4, 1),
        (Judgement, 0, 0, 40, 1),
        (JustLucky, 4, 3, 2, 0),
        (LessonLearned, 13, 0, 0, 2),
        (LikeWater, 0, 0, 7, 1),
        (MasterReality, 0, 0, 0, 0),
        (Meditate, 0, 0, 2, 1),
        (MentalFortress, 0, 0, 6, 1),
        (Nirvana, 0, 0, 4, 1),
        (Omniscience, 0, 0, 2, 3),
        (Perseverance, 0, 7, 3, 1),
        (Pray, 0, 0, 4, 1),
        (PressurePoints, 0, 0, 11, 1),
        (Prostrate, 0, 4, 3, 0),
        (Protect, 0, 16, 0, 2),
        (Ragnarok, 6, 0, 6, 3),
        (ReachHeaven, 15, 0, 0, 2),
        (Rushdown, 0, 0, 2, 0),
        (Sanctity, 0, 9, 2, 1),
        (SandsOfTime, 26, 0, 0, 4),
        (SashWhip, 10, 0, 2, 1),
        (Scrawl, 0, 0, 0, 0),
        (SignatureMove, 40, 0, 0, 2),
        (SimmeringFury, 0, 0, 3, 1),
        (SpiritShield, 0, 0, 4, 2),
        (StrikeP, 9, 0, 0, 1),
        (Study, 0, 0, 1, 1),
        (Swivel, 0, 11, 0, 2),
        (TalkToTheHand, 7, 0, 3, 1),
        (Tantrum, 3, 0, 4, 1),
        (ThirdEye, 0, 9, 5, 1),
        (Tranquility, 0, 0, 0, 0),
        (Vault, 0, 0, 0, 2),
        (Vigilance, 0, 12, 0, 2),
        (Wallop, 12, 0, 0, 2),
        (WaveOfTheHand, 0, 0, 2, 1),
        (Weave, 6, 0, 0, 0),
        (WheelKick, 20, 0, 2, 2),
        (WindmillStrike, 10, 0, 5, 2),
        (Wish, 4, 8, 30, 3),
        (Worship, 0, 0, 5, 2),
        (WreathOfFlame, 0, 0, 8, 1),
        (Beta, 0, 0, 0, 1),
        (Expunger, 15, 0, 0, 1),
        (Insight, 0, 0, 3, 0),
        (Miracle, 0, 0, 0, 0),
        (Omega, 0, 0, 60, 3),
        (Safety, 0, 16, 0, 1),
        (Smite, 16, 0, 0, 1),
        (ThroughViolence, 30, 0, 0, 0),
        (BecomeAlmighty, 0, 0, 4, -2),
        (FameAndFortune, 0, 0, 30, -2),
        (LiveForever, 0, 0, 8, -2),
    ];
    let mut c = setup("Jaw Worm", &[], &[]);
    for (id, dmg, blk, magic, cost) in table {
        let card = upgraded(&mut c, id);
        assert!(card.upgraded, "{id:?}");
        assert_eq!((card.base_damage, card.base_block, card.base_magic, card.cost), (dmg, blk, magic, cost), "{id:?}");
    }
    assert_eq!(table.len(), 75 + 8 + 3);
    for id in [Alpha, BattleHymn, Establishment] {
        assert!(upgraded(&mut c, id).innate, "{id:?}");
    }
    for id in [Blasphemy, Worship] {
        assert!(!c.new_card(id).self_retain && upgraded(&mut c, id).self_retain, "{id:?}");
    }
    assert!(c.new_card(DevaForm).ethereal && !upgraded(&mut c, DevaForm).ethereal);
    // The stance choices don't change at all.
    assert!(!upgraded(&mut c, ChooseCalm).upgraded);
}

// ---- Runs ----

fn watcher_run(seed: i64, ascension: i32) -> Run {
    let mut cfg = RunConfig::new(seed);
    cfg.character = Character::Watcher;
    cfg.ascension = ascension;
    Run::new(cfg)
}

#[test]
fn a_watcher_run_starts_with_her_deck_relic_and_pools() {
    use sts_core::cards::CardColor;
    let r = watcher_run(5, 0);
    assert_eq!((r.hp, r.max_hp), (72, 72));
    assert_eq!(r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), [RelicId::PureWater]);
    let mut deck = ids(&r.deck);
    deck.sort();
    let mut want = Character::Watcher.starting_deck();
    want.sort();
    assert_eq!(deck, want);
    for pool in [&r.pools.common, &r.pools.uncommon, &r.pools.rare] {
        assert!(!pool.is_empty() && pool.iter().all(|c| c.data().color == CardColor::Purple));
    }
    let relics: Vec<RelicId> = [&r.pools.common_relics, &r.pools.uncommon_relics, &r.pools.rare_relics, &r.pools.shop_relics, &r.pools.boss_relics]
        .into_iter()
        .flatten()
        .copied()
        .collect();
    assert!(relics.iter().all(|x| matches!(x.data().class, RelicClass::Shared | RelicClass::Purple)));
    assert!(relics.contains(&RelicId::HolyWater) && relics.contains(&RelicId::Damaru));
    assert_eq!((watcher_run(5, 14).max_hp, watcher_run(5, 14).hp), (68, 61));
    let mut r = watcher_run(5, 0);
    r.obtain_relic(RelicId::HolyWater);
    assert_eq!(r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), [RelicId::HolyWater]);
}

#[test]
fn lesson_learned_upgrades_the_run_deck() {
    for seed in 0..40i64 {
        let mut r = watcher_run(seed, 0);
        r.step(&RunAction::Neow { option: 1 }).unwrap();
        while !matches!(r.screen, Screen::Combat) {
            let a = r.legal_actions()[0].clone();
            r.step(&a).unwrap();
        }
        let c = r.combat.as_mut().unwrap();
        if c.order.len() != 1 {
            continue;
        }
        let m = c.order[0];
        c.monsters[m].hp = 1;
        let ll = c.new_card(CardId::LessonLearned);
        c.hand.push(ll);
        c.energy = 3;
        let i = c.hand.len() - 1;
        let upgraded_before = r.deck.iter().filter(|card| card.upgraded).count();
        r.step(&RunAction::Combat { action: PlayerAction::PlayCard { hand_index: i, target: Some(m) } }).unwrap();
        assert!(r.combat.is_none());
        assert_eq!(r.deck.iter().filter(|card| card.upgraded).count(), upgraded_before + 1);
        return;
    }
    panic!("no single-monster first fight");
}

/// A random policy that plays cards more often than it ends turns and rests when hurt.
fn random_policy(run: &Run, state: &mut u64) -> RunAction {
    let acts = run.legal_actions();
    assert!(!acts.is_empty(), "no legal actions on {:?}", run.screen);
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r = (*state >> 33) as usize;
    let plays: Vec<&RunAction> = acts.iter().filter(|a| !matches!(a, RunAction::Combat { action: PlayerAction::EndTurn })).collect();
    if matches!(run.screen, Screen::Combat) && !plays.is_empty() && r % 10 != 0 {
        plays[r / 10 % plays.len()].clone()
    } else if matches!(run.screen, Screen::Rest) && run.hp * 2 < run.max_hp && acts.contains(&RunAction::Rest) {
        RunAction::Rest
    } else {
        acts[r % acts.len()].clone()
    }
}

#[test]
fn random_watcher_runs_play_through_without_errors() {
    let mut later = 0;
    for seed in 0..60i64 {
        let mut run = watcher_run(seed, if seed % 3 == 0 { 20 } else { 0 });
        run.step(&RunAction::Neow { option: 1 }).unwrap();
        run.max_hp = 600;
        run.hp = 600;
        let mut state = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
        let mut steps = 0;
        while !run.is_over() && steps < 60_000 {
            let a = random_policy(&run, &mut state);
            run.step(&a).unwrap_or_else(|e| panic!("seed {seed}: {a:?} failed: {e}"));
            steps += 1;
        }
        assert!(run.is_over(), "seed {seed} stuck on {:?}", run.screen);
        assert!(run.deck.iter().all(|c| c.id.is_ported()), "seed {seed}");
        later += (run.act > 1) as i32;
    }
    assert!(later > 20, "{later}");
}

// ---- Plays and upgrades the tests above leave out ----

#[test]
fn windmill_strike_hits_harder_for_each_retain() {
    use CardId::*;
    // 7 (10 upgraded); each turn it is retained, 4 more (5 upgraded).
    for (plus, dmg, grow) in [(false, 7, 4), (true, 10, 5)] {
        let mut c = setup("Cultist", &[], &[DefendP; 10]);
        let w = if plus { upgraded(&mut c, WindmillStrike) } else { c.new_card(WindmillStrike) };
        c.hand.push(w);
        let m = m0(&c);
        let before = hp(&c, m);
        end_turn(&mut c);
        play(&mut c, WindmillStrike, Some(m));
        assert_eq!(before - hp(&c, m), dmg + grow, "upgraded {plus}");
    }
}

/// The upgrades that change what a card does rather than its numbers.
#[test]
fn wish_collect_miracle_and_expunger_upgraded() {
    use CardId::*;
    // Wish+: the options are upgraded: 4 Strength, 30 gold, 8 Plated Armor.
    for (pick, want) in [(0usize, 4), (1, 30), (2, 8)] {
        let mut c = setup("Jaw Worm", &[], &[]);
        let w = upgraded(&mut c, Wish);
        c.hand.push(w);
        let gold = c.player.gold;
        play(&mut c, Wish, None);
        assert!(c.pending.as_ref().unwrap().offer.iter().all(|o| o.upgraded));
        choose(&mut c, &[pick]);
        let got = match pick {
            0 => power(&c, Cr::Player, PowerId::Strength),
            1 => c.player.gold - gold,
            _ => power(&c, Cr::Player, PowerId::PlatedArmor),
        };
        assert_eq!(got, want);
    }
    // Collect+: X + 1 Miracles+.
    let mut c = setup("Cultist", &[], &[DefendP; 10]);
    let col = upgraded(&mut c, Collect);
    c.hand.push(col);
    play(&mut c, Collect, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Collect), 4);
    // Miracle+: 2 energy, exhausted.
    let mut c = setup("Jaw Worm", &[], &[]);
    let mir = upgraded(&mut c, Miracle);
    c.hand.push(mir);
    play(&mut c, Miracle, None);
    assert_eq!((c.energy, c.exhaust_pile.len()), (5, 1));
    // Under Master Reality, Conjure Blade's Expunger is upgraded (15 damage) after setX, so it keeps
    // X hits (setX's +1 for an upgraded Expunger never applies).
    let mut c = setup("Cultist", &[ConjureBlade], &[DefendP; 3]);
    c.player.powers.push(sts_core::powers::Power::flag(PowerId::MasterReality));
    play(&mut c, ConjureBlade, None);
    let e = c.draw_pile.iter().find(|card| card.id == Expunger).expect("an Expunger").clone();
    assert!(e.upgraded);
    assert_eq!((e.base_damage, e.magic), (15, 3));
    c.hand.push(e);
    c.draw_pile.retain(|card| card.id != Expunger);
    c.energy = 1;
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, Expunger, Some(m));
    assert_eq!(before - hp(&c, m), 45);
}
