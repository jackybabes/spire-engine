//! The Silent's cards and powers, checked against the decompiled game (`cards/green`, the powers
//! and actions they use). Expected numbers come from reading those classes.

use sts_core::action::Cr;
use sts_core::cards::{Card, CardId, GREEN_LIBRARY_ORDER};
use sts_core::combat::{Combat, CombatConfig, PlayerAction};
use sts_core::encounters;
use sts_core::character::Character;
use sts_core::potions::PotionId;
use sts_core::powers::PowerId;
use sts_core::relics::{RelicClass, RelicId, RelicState};
use sts_core::rng::StsRandom;
use sts_core::run::{Run, RunAction, RunConfig, Screen};

/// A Silent fight with `deck` and no relics (so the opening hand is the first five cards drawn).
fn fight(seed: i64, encounter: &str, deck: &[(CardId, i32)]) -> Combat {
    let mut cfg = CombatConfig::silent(seed, encounter);
    cfg.deck = deck.to_vec();
    cfg.relics.clear();
    Combat::new(&cfg).unwrap()
}

/// A fight whose hand is exactly `hand` (fresh cards, upgraded when the flag is set) and whose draw
/// pile is `draw` (top last), so tests don't depend on the shuffle.
fn setup(encounter: &str, hand: &[CardId], draw: &[CardId]) -> Combat {
    let mut c = fight(7, encounter, &[(CardId::DefendG, 0); 5]);
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

fn choose(c: &mut Combat, hand_ids: &[CardId]) {
    let mut taken = Vec::new();
    for &id in hand_ids {
        let i = c.hand.iter().enumerate().position(|(k, card)| card.id == id && !taken.contains(&k)).unwrap();
        taken.push(i);
    }
    c.step(&PlayerAction::Choose { indices: taken }).unwrap();
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

// ---- Fuzz ----

/// Random Silent decks (every card, some upgraded) against every encounter, with random legal
/// play: no panics, no illegal states.
#[test]
fn random_silent_fights_stay_consistent() {
    let mut all: Vec<&str> = encounters::all_act1();
    all.extend(encounters::all_act2());
    all.extend(encounters::all_act3());
    let mut pick = StsRandom::new(5);
    let mut played = std::collections::HashSet::new();
    let mut steps_total = 0;
    for seed in 0..400i64 {
        let encounter = all[seed as usize % all.len()];
        let n = 10 + pick.random(10);
        let deck: Vec<(CardId, i32)> = (0..n)
            .map(|_| {
                let id = GREEN_LIBRARY_ORDER[pick.random(GREEN_LIBRARY_ORDER.len() as i32 - 1) as usize];
                (id, pick.random(1))
            })
            .collect();
        let mut cfg = CombatConfig::silent(seed, encounter);
        cfg.deck = deck.clone();
        cfg.player_hp = 300;
        cfg.player_max_hp = 300;
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
            assert!(c.hand.len() <= 10, "seed {seed}: hand of {}", c.hand.len());
            assert!(c.energy >= 0 && c.player.hp >= 0);
            for &m in &c.order {
                assert!(c.monsters[m].hp >= 0);
            }
            steps += 1;
        }
        steps_total += steps;
    }
    // Every playable Silent card got played (Reflex and Tactician can't be).
    let missing: Vec<CardId> = GREEN_LIBRARY_ORDER.iter().copied().filter(|id| !played.contains(id) && !matches!(id, CardId::Reflex | CardId::Tactician)).collect();
    assert!(missing.is_empty(), "never played: {missing:?} ({steps_total} steps)");
    assert!(played.contains(&CardId::Shiv));
}

// ---- Starting out ----

#[test]
fn a_silent_fight_starts_with_seventy_hp_and_seven_cards() {
    let c = Combat::new(&CombatConfig::silent(3, "Jaw Worm")).unwrap();
    assert_eq!((c.player.hp, c.player.max_hp), (70, 70));
    // Five cards, then Ring of the Snake's two.
    assert_eq!(c.hand.len(), 7);
    assert_eq!(c.draw_pile.len(), 5);
    assert_eq!(c.energy, 3);
}

#[test]
fn upgrades_match_each_cards_upgrade_method() {
    use CardId::*;
    // (card, base damage, base block, base magic, cost) after one upgrade.
    let table = [
        (StrikeG, 9, 0, 0, 1),
        (DefendG, 0, 8, 0, 1),
        (Neutralize, 4, 0, 2, 0),
        (Survivor, 0, 11, 0, 1),
        (AThousandCuts, 0, 0, 2, 2),
        (Accuracy, 0, 0, 6, 1),
        (Acrobatics, 0, 0, 4, 1),
        (Alchemize, 0, 0, 0, 0),
        (AllOutAttack, 14, 0, 0, 1),
        (Backflip, 0, 8, 0, 1),
        (Backstab, 15, 0, 0, 0),
        (Bane, 10, 0, 0, 1),
        (BladeDance, 0, 0, 4, 1),
        (Blur, 0, 8, 0, 1),
        (BouncingFlask, 0, 0, 4, 2),
        (BulletTime, 0, 0, 0, 2),
        (Burst, 0, 0, 2, 1),
        (Caltrops, 0, 0, 5, 1),
        (Choke, 12, 0, 5, 2),
        (CloakAndDagger, 0, 6, 2, 1),
        (Concentrate, 0, 0, 2, 0),
        (CorpseExplosion, 0, 0, 9, 2),
        (CripplingPoison, 0, 0, 7, 2),
        (DaggerSpray, 6, 0, 0, 1),
        (DaggerThrow, 12, 0, 0, 1),
        (Dash, 13, 13, 0, 2),
        (DeadlyPoison, 0, 0, 7, 1),
        (Deflect, 0, 7, 0, 0),
        (DieDieDie, 17, 0, 0, 1),
        (Distraction, 0, 0, 0, 0),
        (DodgeAndRoll, 0, 6, 0, 1),
        (EndlessAgony, 6, 0, 0, 0),
        (Envenom, 0, 0, 0, 1),
        (EscapePlan, 0, 5, 0, 0),
        (Eviscerate, 9, 0, 0, 3),
        (Expertise, 0, 0, 7, 1),
        (Finisher, 8, 0, 0, 1),
        (Flechettes, 6, 0, 0, 1),
        (FlyingKnee, 11, 0, 0, 1),
        (Footwork, 0, 0, 3, 1),
        (GlassKnife, 12, 0, 0, 1),
        (GrandFinale, 60, 0, 0, 0),
        (HeelHook, 8, 0, 0, 1),
        (LegSweep, 0, 14, 3, 2),
        (MasterfulStab, 16, 0, 0, 0),
        (Nightmare, 0, 0, 3, 2),
        (NoxiousFumes, 0, 0, 3, 1),
        (PhantasmalKiller, 0, 0, 0, 0),
        (PiercingWail, 0, 0, 8, 1),
        (PoisonedStab, 8, 0, 4, 1),
        (Predator, 20, 0, 0, 2),
        (Prepared, 0, 0, 2, 0),
        (QuickSlash, 12, 0, 0, 1),
        (Reflex, 0, 0, 3, -2),
        (RiddleWithHoles, 4, 0, 0, 2),
        (Setup, 0, 0, 0, 0),
        (Skewer, 10, 0, 0, -1),
        (Slice, 9, 0, 0, 0),
        (SneakyStrike, 16, 0, 0, 2),
        (SuckerPunch, 9, 0, 2, 1),
        (Tactician, 0, 0, 2, -2),
        (Terror, 0, 0, 0, 0),
        (ToolsOfTheTrade, 0, 0, 0, 0),
        (Unload, 18, 0, 0, 1),
        (WellLaidPlans, 0, 0, 2, 1),
        (WraithForm, 0, 0, 3, 3),
        (Shiv, 6, 0, 0, 0),
    ];
    let mut c = setup("Jaw Worm", &[], &[]);
    for (id, damage, block, magic, cost) in table {
        let card = upgraded(&mut c, id);
        assert_eq!((card.base_damage, card.base_block, card.base_magic, card.cost), (damage, block, magic, cost), "{id:?}+");
        assert!(card.upgraded && card.name().ends_with('+'), "{id:?}");
    }
    assert!(upgraded(&mut c, AfterImage).innate && upgraded(&mut c, InfiniteBlades).innate);
    assert!(!upgraded(&mut c, CalculatedGamble).exhaust);
}

#[test]
fn starter_cards_hit_block_and_weaken() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[StrikeG, DefendG, Neutralize, Survivor, Slice], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Neutralize, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 3);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 1);
    play(&mut c, StrikeG, Some(m));
    play(&mut c, Slice, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 15);
    play(&mut c, DefendG, None);
    play(&mut c, Survivor, None);
    assert_eq!(c.player.block, 13);
    assert_eq!(c.energy, 0);
}

// ---- Poison ----

#[test]
fn poison_hits_at_the_monsters_turn_and_counts_down() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[DeadlyPoison, PoisonedStab], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, DeadlyPoison, Some(m));
    play(&mut c, PoisonedStab, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 6);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 8);
    end_turn(&mut c);
    assert_eq!(c.monsters[m].hp, hp - 6 - 8);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 7);
}

#[test]
fn catalyst_doubles_poison_and_triples_it_upgraded() {
    use CardId::*;
    for (up, want) in [(false, 10), (true, 15)] {
        let mut c = setup("Jaw Worm", &[DeadlyPoison], &[]);
        let cat = if up { upgraded(&mut c, Catalyst) } else { c.new_card(Catalyst) };
        c.hand.push(cat);
        let m = m0(&c);
        play(&mut c, DeadlyPoison, Some(m));
        play(&mut c, Catalyst, Some(m));
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), want);
        assert_eq!(c.exhaust_pile.len(), 1);
    }
}

#[test]
fn bouncing_flask_spreads_three_poison_three_times() {
    let mut c = setup("Small Slimes", &[CardId::BouncingFlask], &[]);
    play(&mut c, CardId::BouncingFlask, None);
    let total: i32 = c.order.iter().map(|&m| power(&c, Cr::Monster(m), PowerId::Poison)).sum();
    assert_eq!(total, 9);
}

#[test]
fn crippling_poison_and_noxious_fumes_poison_every_monster() {
    use CardId::*;
    let mut c = setup("Small Slimes", &[CripplingPoison, NoxiousFumes], &[]);
    play(&mut c, CripplingPoison, None);
    for &m in &c.order {
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 4);
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 2);
    }
    play(&mut c, NoxiousFumes, None);
    end_turn(&mut c);
    // The monsters' turn ticked 4 down to 3, then the Fumes added 2 after the player's draw.
    for &m in &c.order {
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 5);
    }
}

#[test]
fn corpse_explosion_hits_the_others_for_the_max_hp_of_the_dead() {
    use CardId::*;
    let mut c = setup("Small Slimes", &[CorpseExplosion, StrikeG], &[]);
    let (a, b) = if c.monsters[c.order[0]].max_hp < c.monsters[c.order[1]].max_hp { (c.order[0], c.order[1]) } else { (c.order[1], c.order[0]) };
    play(&mut c, CorpseExplosion, Some(a));
    assert_eq!(power(&c, Cr::Monster(a), PowerId::Poison), 6);
    c.monsters[a].hp = 1;
    let hp_b = c.monsters[b].hp;
    play(&mut c, StrikeG, Some(a));
    assert!(c.monsters[a].is_dying);
    assert_eq!(c.monsters[b].hp, hp_b - c.monsters[a].max_hp);
}

#[test]
fn envenom_poisons_on_unblocked_attack_damage() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Envenom, Shiv, Shiv], &[]);
    let m = m0(&c);
    play(&mut c, Envenom, None);
    c.monsters[m].block = 10;
    play(&mut c, Shiv, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 0);
    c.monsters[m].block = 0;
    play(&mut c, Shiv, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 1);
}

#[test]
fn bane_hits_twice_on_a_poisoned_target() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Bane, DeadlyPoison, Bane], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Bane, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 7);
    play(&mut c, DeadlyPoison, Some(m));
    play(&mut c, Bane, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 21);
}

// ---- Shivs ----

#[test]
fn shivs_read_accuracy_whenever_they_are_made() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[BladeDance, Accuracy, CloakAndDagger], &[]);
    let m = m0(&c);
    play(&mut c, BladeDance, None);
    assert_eq!(c.hand.iter().filter(|h| h.id == Shiv && h.base_damage == 4).count(), 3);
    play(&mut c, Accuracy, None);
    play(&mut c, CloakAndDagger, None);
    assert_eq!(c.player.block, 6);
    let shivs: Vec<i32> = c.hand.iter().filter(|h| h.id == Shiv).map(|h| h.base_damage).collect();
    assert_eq!(shivs, [8, 8, 8, 8]);
    let hp = c.monsters[m].hp;
    play(&mut c, Shiv, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 8);
    assert_eq!(c.exhaust_pile.last().unwrap().id, Shiv);
}

#[test]
fn storm_of_steel_discards_the_hand_for_shivs() {
    use CardId::*;
    for up in [false, true] {
        let mut c = setup("Jaw Worm", &[StrikeG, DefendG, Reflex], &[]);
        let storm = if up { upgraded(&mut c, StormOfSteel) } else { c.new_card(StormOfSteel) };
        c.hand.insert(0, storm);
        play(&mut c, StormOfSteel, None);
        let shivs: Vec<&Card> = c.hand.iter().filter(|h| h.id == Shiv).collect();
        assert_eq!(shivs.len(), 3);
        assert!(shivs.iter().all(|s| s.upgraded == up && s.base_damage == if up { 6 } else { 4 }));
        assert_eq!(c.total_discarded_this_turn, 3);
        // Reflex was discarded by a card: it drew 2 (after a shuffle of the discard pile).
        assert_eq!(c.hand.len(), 5);
    }
}

#[test]
fn infinite_blades_adds_a_shiv_each_turn() {
    let mut c = setup("Jaw Worm", &[CardId::InfiniteBlades], &[]);
    play(&mut c, CardId::InfiniteBlades, None);
    end_turn(&mut c);
    assert_eq!(ids(&c.hand), [CardId::Shiv]);
}

// ---- Discards ----

#[test]
fn reflex_and_tactician_trigger_when_a_card_discards_them() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Survivor, Reflex, Tactician, StrikeG], &[DefendG, DefendG, DefendG, DefendG]);
    play(&mut c, Survivor, None);
    choose(&mut c, &[Reflex]);
    assert_eq!(c.hand.len(), 4);
    assert_eq!(c.draw_pile.len(), 2);
    let mut c = setup("Jaw Worm", &[Survivor, Tactician, StrikeG], &[]);
    play(&mut c, Survivor, None);
    choose(&mut c, &[Tactician]);
    assert_eq!(c.energy, 3);
    // Neither can be played.
    let c = setup("Jaw Worm", &[Reflex, Tactician], &[]);
    assert_eq!(c.legal_actions(), [PlayerAction::EndTurn]);
}

#[test]
fn the_end_of_turn_discard_triggers_nothing() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Reflex, Tactician], &[DefendG, DefendG, DefendG, DefendG, DefendG, DefendG, DefendG]);
    end_turn(&mut c);
    assert_eq!(c.energy, 3);
    assert_eq!(c.hand.len(), 5);
    assert_eq!(c.total_discarded_this_turn, 0);
}

#[test]
fn sneaky_strike_refunds_and_eviscerate_gets_cheaper_after_a_discard() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[SneakyStrike, Eviscerate, Survivor, StrikeG], &[]);
    let m = m0(&c);
    play(&mut c, Survivor, None);
    choose(&mut c, &[StrikeG]);
    assert_eq!(c.hand[idx(&c, Eviscerate)].cost_for_turn, 2);
    let hp = c.monsters[m].hp;
    play(&mut c, SneakyStrike, Some(m));
    assert_eq!(c.energy, 2);
    play(&mut c, Eviscerate, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 12 - 21);
    assert_eq!(c.energy, 0);
}

#[test]
fn eviscerate_drawn_after_discards_costs_less() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Survivor, StrikeG, QuickSlash], &[Eviscerate]);
    let m = m0(&c);
    play(&mut c, Survivor, None);
    choose(&mut c, &[StrikeG]);
    play(&mut c, QuickSlash, Some(m));
    assert_eq!(c.hand[idx(&c, Eviscerate)].cost_for_turn, 2);
}

#[test]
fn calculated_gamble_discards_the_hand_and_draws_as_many() {
    use CardId::*;
    for up in [false, true] {
        let mut c = setup("Jaw Worm", &[StrikeG, DefendG, Reflex], &[Deflect, Deflect, Deflect, Deflect, Deflect]);
        let gamble = if up { upgraded(&mut c, CalculatedGamble) } else { c.new_card(CalculatedGamble) };
        c.hand.insert(0, gamble);
        play(&mut c, CalculatedGamble, None);
        // Three drawn for the three discarded, two more for Reflex.
        assert_eq!(ids(&c.hand), [Deflect; 5]);
        assert_eq!(c.total_discarded_this_turn, 3);
        assert_eq!(c.exhaust_pile.len(), if up { 0 } else { 1 });
    }
}

#[test]
fn unload_discards_every_non_attack() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Unload, DefendG, StrikeG, Reflex, Deflect], &[Slice, Slice]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Unload, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 14);
    let mut hand = ids(&c.hand);
    hand.sort();
    assert_eq!(hand, [Slice, Slice, StrikeG]);
    assert_eq!(c.total_discarded_this_turn, 3);
}

#[test]
fn concentrate_acrobatics_prepared_and_dagger_throw_draw_and_discard() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Concentrate, StrikeG, DefendG, Deflect, Slice], &[]);
    play(&mut c, Concentrate, None);
    choose(&mut c, &[StrikeG, DefendG, Deflect]);
    assert_eq!(c.energy, 5);
    assert_eq!(ids(&c.hand), [Slice]);

    let mut c = setup("Jaw Worm", &[Acrobatics, Prepared], &[Slice, Slice, Slice, Deflect]);
    play(&mut c, Acrobatics, None);
    assert_eq!(c.hand.len(), 4);
    choose(&mut c, &[Slice]);
    play(&mut c, Prepared, None);
    choose(&mut c, &[Slice]);
    assert_eq!(c.hand.len(), 2);
    assert_eq!(c.total_discarded_this_turn, 2);

    let mut c = setup("Jaw Worm", &[DaggerThrow, DefendG], &[Slice]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, DaggerThrow, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 9);
    choose(&mut c, &[DefendG]);
    assert_eq!(ids(&c.hand), [Slice]);
}

#[test]
fn all_out_attack_hits_everything_and_discards_at_random() {
    use CardId::*;
    let mut c = setup("Small Slimes", &[AllOutAttack, DefendG, Deflect], &[]);
    let hps: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    play(&mut c, AllOutAttack, None);
    for (k, &m) in c.order.iter().enumerate() {
        assert_eq!(c.monsters[m].hp, hps[k] - 10);
    }
    assert_eq!(c.hand.len(), 1);
    assert_eq!(c.total_discarded_this_turn, 1);
}

#[test]
fn tools_of_the_trade_draws_and_discards_after_the_draw() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[ToolsOfTheTrade], &[DefendG; 8]);
    play(&mut c, ToolsOfTheTrade, None);
    end_turn(&mut c);
    assert_eq!(c.hand.len(), 6);
    choose(&mut c, &[DefendG]);
    assert_eq!(c.hand.len(), 5);
    assert_eq!(c.total_discarded_this_turn, 1);
}

// ---- Powers ----

#[test]
fn after_image_blocks_and_a_thousand_cuts_hits_after_each_card() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[AfterImage, AThousandCuts, Shiv, Shiv], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, AfterImage, None);
    assert_eq!(c.player.block, 0);
    play(&mut c, AThousandCuts, None);
    assert_eq!(c.player.block, 1);
    assert_eq!(c.monsters[m].hp, hp);
    play(&mut c, Shiv, Some(m));
    play(&mut c, Shiv, Some(m));
    assert_eq!(c.player.block, 3);
    assert_eq!(c.monsters[m].hp, hp - 10);
}

#[test]
fn burst_plays_the_next_skill_twice() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Burst, DefendG, StrikeG], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Burst, None);
    play(&mut c, DefendG, None);
    assert_eq!(c.player.block, 10);
    assert!(!c.player.has_power(PowerId::Burst));
    play(&mut c, StrikeG, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 6);
}

#[test]
fn blur_keeps_block_into_the_next_turn() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Blur, DefendG, Deflect], &[]);
    play(&mut c, Blur, None);
    play(&mut c, DefendG, None);
    play(&mut c, Deflect, None);
    end_turn(&mut c);
    // Jaw Worm's Chomp (11) took 11 of the 14.
    assert_eq!(c.player.hp, 70);
    assert_eq!(c.player.block, 3);
    assert!(!c.player.has_power(PowerId::Blur));
}

#[test]
fn wraith_form_is_intangible_and_loses_dexterity_each_turn() {
    let mut c = setup("Jaw Worm", &[CardId::WraithForm], &[]);
    play(&mut c, CardId::WraithForm, None);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 69);
    assert_eq!(power(&c, Cr::Player, PowerId::Dexterity), -1);
    assert_eq!(power(&c, Cr::Player, PowerId::IntangiblePlayer), 1);
    assert_eq!(power(&c, Cr::Player, PowerId::WraithForm), -1);
}

#[test]
fn footwork_and_dodge_and_roll_block_now_and_next_turn() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Footwork, DodgeAndRoll], &[]);
    play(&mut c, Footwork, None);
    play(&mut c, DodgeAndRoll, None);
    assert_eq!(c.player.block, 6);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 65);
    assert_eq!(c.player.block, 6);
}

#[test]
fn caltrops_hits_back() {
    let mut c = setup("Jaw Worm", &[CardId::Caltrops], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::Caltrops, None);
    end_turn(&mut c);
    assert_eq!(c.monsters[m].hp, hp - 3);
}

#[test]
fn phantasmal_killer_doubles_attack_damage_next_turn() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[PhantasmalKiller], &[StrikeG; 5]);
    let m = m0(&c);
    play(&mut c, PhantasmalKiller, None);
    end_turn(&mut c);
    assert_eq!(power(&c, Cr::Player, PowerId::DoubleDamage), 1);
    let hp = c.monsters[m].hp;
    play(&mut c, StrikeG, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 12);
}

#[test]
fn choke_costs_the_target_hp_for_every_card_this_turn() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Choke, Shiv], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Choke, Some(m));
    play(&mut c, Shiv, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 12 - 4 - 3);
    end_turn(&mut c);
    assert!(!c.monsters[m].has_power(PowerId::Choked));
}

#[test]
fn debuffs_terror_piercing_wail_leg_sweep_sucker_punch() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Terror, StrikeG], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Terror, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Vulnerable), 99);
    play(&mut c, StrikeG, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 9);

    let mut c = setup("Small Slimes", &[PiercingWail], &[]);
    play(&mut c, PiercingWail, None);
    for &m in &c.order {
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), -6);
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Shackled), 6);
    }
    end_turn(&mut c);
    for &m in &c.order {
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), 0);
    }

    let mut c = setup("Jaw Worm", &[LegSweep, SuckerPunch], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, LegSweep, Some(m));
    play(&mut c, SuckerPunch, Some(m));
    assert_eq!(c.player.block, 11);
    assert_eq!(c.monsters[m].hp, hp - 7);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 3);
}

#[test]
fn energized_and_draw_next_turn() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Outmaneuver, Predator], &[DefendG; 10]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Outmaneuver, None);
    play(&mut c, Predator, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 15);
    end_turn(&mut c);
    assert_eq!(c.energy, 5);
    assert_eq!(c.hand.len(), 7);
    assert!(!c.player.has_power(PowerId::Energized) && !c.player.has_power(PowerId::DrawCardNextTurn));

    let mut c = setup("Jaw Worm", &[FlyingKnee], &[]);
    let m = m0(&c);
    play(&mut c, FlyingKnee, Some(m));
    end_turn(&mut c);
    assert_eq!(c.energy, 4);
}

#[test]
fn x_cost_cards_use_all_the_energy() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Doppelganger], &[DefendG; 10]);
    play(&mut c, Doppelganger, None);
    assert_eq!(c.energy, 0);
    end_turn(&mut c);
    assert_eq!(c.energy, 6);
    assert_eq!(c.hand.len(), 8);

    let mut c = setup("Jaw Worm", &[], &[]);
    let malaise = upgraded(&mut c, Malaise);
    c.hand.push(malaise);
    let m = m0(&c);
    play(&mut c, Malaise, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), -4);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 4);
    assert_eq!(c.energy, 0);

    let mut c = setup("Jaw Worm", &[Skewer], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Skewer, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 21);
    assert_eq!(c.energy, 0);
}

// ---- Attacks that count things ----

#[test]
fn finisher_flechettes_heel_hook_and_glass_knife() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Shiv, Shiv, Finisher], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Shiv, Some(m));
    play(&mut c, Shiv, Some(m));
    play(&mut c, Finisher, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 4 - 4 - 12);

    let mut c = setup("Jaw Worm", &[Flechettes, DefendG, Deflect, StrikeG], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Flechettes, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 8);

    let mut c = setup("Jaw Worm", &[HeelHook, Neutralize, HeelHook], &[DefendG]);
    let m = m0(&c);
    play(&mut c, HeelHook, Some(m));
    assert_eq!((c.energy, c.hand.len()), (2, 2));
    play(&mut c, Neutralize, Some(m));
    play(&mut c, HeelHook, Some(m));
    assert_eq!((c.energy, ids(&c.hand)), (2, vec![DefendG]));

    let mut c = setup("Jaw Worm", &[GlassKnife], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, GlassKnife, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 16);
    assert_eq!(c.discard_pile[0].base_damage, 6);
}

#[test]
fn bane_riddle_dash_backflip_quick_slash_numbers() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[RiddleWithHoles, Dash], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, RiddleWithHoles, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 15);
    c.energy = 2;
    play(&mut c, Dash, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 25);
    assert_eq!(c.player.block, 10);

    let mut c = setup("Jaw Worm", &[Backflip, QuickSlash], &[Slice, Slice, Slice]);
    let m = m0(&c);
    play(&mut c, Backflip, None);
    assert_eq!((c.player.block, c.hand.len()), (5, 3));
    play(&mut c, QuickSlash, Some(m));
    assert_eq!(c.hand.len(), 3);
}

#[test]
fn masterful_stab_costs_more_after_each_hit_taken() {
    let mut c = setup("Jaw Worm", &[CardId::MasterfulStab], &[]);
    end_turn(&mut c);
    assert!(c.player.hp < 70);
    let stab = c.hand.iter().chain(&c.discard_pile).chain(&c.draw_pile).find(|h| h.id == CardId::MasterfulStab).unwrap();
    assert_eq!(stab.cost, 1);
}

#[test]
fn endless_agony_copies_itself_when_drawn() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[QuickSlash], &[EndlessAgony]);
    let m = m0(&c);
    play(&mut c, QuickSlash, Some(m));
    assert_eq!(ids(&c.hand), [EndlessAgony, EndlessAgony]);
}

#[test]
fn grand_finale_needs_an_empty_draw_pile() {
    use CardId::*;
    let c = setup("Small Slimes", &[GrandFinale], &[DefendG]);
    assert_eq!(c.legal_actions(), [PlayerAction::EndTurn]);
    let mut c = setup("Small Slimes", &[GrandFinale, DieDieDie], &[]);
    let hps: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    c.monsters[c.order[0]].hp = 60;
    c.monsters[c.order[0]].max_hp = 60;
    play(&mut c, DieDieDie, None);
    assert_eq!(c.monsters[c.order[0]].hp, 47);
    assert!(c.monsters[c.order[1]].hp < hps[1]);
    play(&mut c, GrandFinale, None);
    assert!(c.is_over());
}

#[test]
fn dagger_spray_hits_everything_twice() {
    let mut c = setup("Small Slimes", &[CardId::DaggerSpray], &[]);
    let hps: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    play(&mut c, CardId::DaggerSpray, None);
    for (k, &m) in c.order.iter().enumerate() {
        assert_eq!(c.monsters[m].hp, hps[k] - 8);
    }
}

// ---- Card draw and selection ----

#[test]
fn expertise_draws_up_to_six() {
    let mut c = setup("Jaw Worm", &[CardId::Expertise, CardId::Slice, CardId::Slice], &[CardId::DefendG; 10]);
    play(&mut c, CardId::Expertise, None);
    assert_eq!(c.hand.len(), 6);
}

#[test]
fn escape_plan_blocks_if_it_drew_a_skill_even_across_a_shuffle() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[EscapePlan], &[DefendG]);
    play(&mut c, EscapePlan, None);
    assert_eq!(c.player.block, 3);
    let mut c = setup("Jaw Worm", &[EscapePlan], &[StrikeG]);
    play(&mut c, EscapePlan, None);
    assert_eq!(c.player.block, 0);
    // An empty draw pile: the draw shuffles the discard pile first.
    let mut c = setup("Jaw Worm", &[EscapePlan], &[]);
    let d = c.new_card(Deflect);
    c.discard_pile.push(d);
    play(&mut c, EscapePlan, None);
    assert_eq!(ids(&c.hand), [Deflect]);
    assert_eq!(c.player.block, 3);
}

#[test]
fn adrenaline_and_bullet_time() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Adrenaline], &[Slice, Slice]);
    play(&mut c, Adrenaline, None);
    assert_eq!((c.energy, c.hand.len(), c.exhaust_pile.len()), (4, 2, 1));

    let mut c = setup("Jaw Worm", &[BulletTime, StrikeG, Dash], &[Slice]);
    play(&mut c, BulletTime, None);
    assert_eq!(c.energy, 0);
    assert!(c.hand.iter().all(|h| h.cost_for_turn == 0));
    assert!(c.player.has_power(PowerId::NoDraw));
    let m = m0(&c);
    play(&mut c, Dash, Some(m));
    assert_eq!(c.player.block, 10);
}

#[test]
fn setup_puts_a_card_on_top_free_once() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Setup, Dash], &[]);
    play(&mut c, Setup, None);
    let top = c.draw_pile.last().unwrap();
    assert_eq!(top.id, Dash);
    assert!(top.free_to_play_once);
    end_turn(&mut c);
    let m = m0(&c);
    play(&mut c, Dash, Some(m));
    assert_eq!(c.energy, 3);
    // With a choice to make.
    let mut c = setup("Jaw Worm", &[Setup, Dash, Slice], &[]);
    play(&mut c, Setup, None);
    choose(&mut c, &[Slice]);
    assert_eq!(c.draw_pile.last().unwrap().id, Slice);
    assert!(!c.draw_pile.last().unwrap().free_to_play_once);
}

#[test]
fn nightmare_makes_three_copies_next_turn_and_never_stacks() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Nightmare, Nightmare, StrikeG, DefendG], &[]);
    c.energy = 6;
    play(&mut c, Nightmare, None);
    choose(&mut c, &[StrikeG]);
    play(&mut c, Nightmare, None);
    choose(&mut c, &[DefendG]);
    assert_eq!(c.player.powers.iter().filter(|p| p.id == PowerId::NightTerror).count(), 2);
    end_turn(&mut c);
    let count = |id| c.hand.iter().filter(|h| h.id == id).count();
    assert_eq!((count(StrikeG), count(DefendG)), (4, 4));
    assert!(!c.player.has_power(PowerId::NightTerror));
    assert!(c.night_terror_cards.is_empty());
}

#[test]
fn well_laid_plans_retains_the_chosen_card() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[WellLaidPlans, DefendG, StrikeG], &[Deflect; 5]);
    play(&mut c, WellLaidPlans, None);
    end_turn(&mut c);
    choose(&mut c, &[DefendG]);
    let mut hand = ids(&c.hand);
    hand.sort();
    assert_eq!(hand, [DefendG, Deflect, Deflect, Deflect, Deflect, Deflect]);
    assert!(c.hand.iter().all(|h| !h.retain));
    assert_eq!(c.discard_pile.iter().filter(|h| h.id == StrikeG).count(), 1);
}

#[test]
fn distraction_makes_a_free_silent_skill() {
    use sts_core::cards::{CardColor, CardType};
    for seed in 0..20 {
        let mut c = setup("Jaw Worm", &[CardId::Distraction], &[]);
        c.rng.card_random = StsRandom::new(seed);
        play(&mut c, CardId::Distraction, None);
        let made = &c.hand[0];
        assert_eq!((made.data().color, made.card_type()), (CardColor::Green, CardType::Skill), "{:?}", made.id);
        assert!(!made.data().healing);
        // setCostForTurn(-99) leaves unplayable (-2) and X (-1) costs alone.
        assert_eq!(made.cost_for_turn, made.cost.min(0), "{:?}", made.id);
    }
}

#[test]
fn alchemize_brews_a_potion() {
    let mut c = setup("Jaw Worm", &[CardId::Alchemize], &[]);
    play(&mut c, CardId::Alchemize, None);
    assert!(c.potions[0].is_some());
}

#[test]
fn backstab_and_upgraded_after_image_start_in_hand() {
    use CardId::*;
    let mut deck = vec![(DefendG, 0); 9];
    deck.push((Backstab, 0));
    deck.push((AfterImage, 1));
    let c = fight(11, "Jaw Worm", &deck);
    assert!(c.hand.iter().any(|h| h.id == Backstab) && c.hand.iter().any(|h| h.id == AfterImage));
}

// ---- Relics ----

fn with_relic(c: &mut Combat, r: RelicId) {
    c.player.relics.push(RelicState::new(r));
}

/// A Silent fight with these relics from the start (battle-start relics fire).
fn fight_with(encounter: &str, relics: &[RelicId], deck: &[(CardId, i32)]) -> Combat {
    let mut cfg = CombatConfig::silent(7, encounter);
    cfg.deck = deck.to_vec();
    cfg.relics = relics.to_vec();
    Combat::new(&cfg).unwrap()
}

#[test]
fn snecko_skull_adds_one_to_every_poison_the_player_applies() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[DeadlyPoison, Catalyst], &[]);
    with_relic(&mut c, RelicId::SneckoSkull);
    let m = m0(&c);
    play(&mut c, DeadlyPoison, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 6);
    // Catalyst's DoublePoisonAction builds an ApplyPowerAction too: 6 + (6 + 1).
    play(&mut c, Catalyst, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 13);
}

#[test]
fn paper_crane_makes_weak_monsters_hit_for_sixty_percent() {
    for (crane, want) in [(false, 8), (true, 6)] {
        let mut c = setup("Jaw Worm", &[CardId::Neutralize], &[]);
        if crane {
            with_relic(&mut c, RelicId::PaperCrane);
        }
        let m = m0(&c);
        assert_eq!(c.monsters[m].intent_dmg, 11);
        play(&mut c, CardId::Neutralize, Some(m));
        assert_eq!(c.monsters[m].intent_dmg, want);
        end_turn(&mut c);
        assert_eq!(c.player.hp, 70 - want);
    }
}

#[test]
fn wrist_blade_adds_four_to_zero_cost_attacks() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Shiv, StrikeG], &[]);
    with_relic(&mut c, RelicId::WristBlade);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Shiv, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 8);
    play(&mut c, StrikeG, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 14);
}

#[test]
fn battle_start_relics_funnel_scroll_and_the_serpent() {
    let deck = [(CardId::DefendG, 0); 10];
    let c = fight_with("Small Slimes", &[RelicId::TwistedFunnel], &deck);
    for &m in &c.order {
        assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 4);
    }
    let c = fight_with("Jaw Worm", &[RelicId::NinjaScroll], &deck);
    assert_eq!(c.hand.iter().filter(|h| h.id == CardId::Shiv).count(), 3);
    assert_eq!(c.hand.len(), 8);
    let mut c = fight_with("Jaw Worm", &[RelicId::RingOfTheSerpent], &deck);
    assert_eq!(c.hand.len(), 6);
    end_turn(&mut c);
    assert_eq!(c.hand.len(), 6);
}

#[test]
fn the_specimen_passes_poison_on_when_a_poisoned_monster_dies() {
    use CardId::*;
    let mut c = setup("Small Slimes", &[DeadlyPoison, StrikeG], &[]);
    with_relic(&mut c, RelicId::TheSpecimen);
    let (a, b) = (c.order[0], c.order[1]);
    play(&mut c, DeadlyPoison, Some(a));
    c.monsters[a].hp = 1;
    c.monsters[b].hp = 50;
    play(&mut c, StrikeG, Some(a));
    assert!(c.monsters[a].is_dying);
    assert_eq!(power(&c, Cr::Monster(b), PowerId::Poison), 5);
}

#[test]
fn discard_relics_tingsha_tough_bandages_and_hovering_kite() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Survivor, StrikeG, Prepared, DefendG], &[Deflect]);
    with_relic(&mut c, RelicId::Tingsha);
    with_relic(&mut c, RelicId::ToughBandages);
    with_relic(&mut c, RelicId::HoveringKite);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Survivor, None);
    choose(&mut c, &[StrikeG]);
    assert_eq!(c.monsters[m].hp, hp - 3);
    assert_eq!(c.player.block, 11);
    assert_eq!(c.energy, 3);
    play(&mut c, Prepared, None);
    choose(&mut c, &[Deflect]);
    // Tingsha and Bandages again; the Kite only once a turn.
    assert_eq!(c.monsters[m].hp, hp - 6);
    assert_eq!(c.player.block, 14);
    assert_eq!(c.energy, 3);
}

// ---- Potions ----

fn drink(c: &mut Combat, p: PotionId, target: Option<usize>) {
    c.potions[0] = Some(p);
    c.step(&PlayerAction::UsePotion { slot: 0, target }).unwrap();
}

#[test]
fn the_silents_potions() {
    let mut c = setup("Jaw Worm", &[], &[]);
    let m = m0(&c);
    drink(&mut c, PotionId::PoisonPotion, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 6);
    with_relic(&mut c, RelicId::SacredBark);
    drink(&mut c, PotionId::PoisonPotion, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Poison), 18);

    let mut c = setup("Jaw Worm", &[], &[]);
    drink(&mut c, PotionId::CunningPotion, None);
    assert_eq!(c.hand.len(), 3);
    assert!(c.hand.iter().all(|h| h.id == CardId::Shiv && h.upgraded && h.base_damage == 6));

    let mut c = setup("Jaw Worm", &[], &[]);
    drink(&mut c, PotionId::GhostInAJar, None);
    assert_eq!(power(&c, Cr::Player, PowerId::IntangiblePlayer), 1);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 69);
}

// ---- Runs ----

fn silent_run(seed: i64, ascension: i32) -> Run {
    let mut cfg = RunConfig::new(seed);
    cfg.character = Character::Silent;
    cfg.ascension = ascension;
    Run::new(cfg)
}

#[test]
fn a_silent_run_starts_with_the_silents_deck_relic_and_pools() {
    use sts_core::cards::CardColor;
    let r = silent_run(5, 0);
    assert_eq!((r.hp, r.max_hp), (70, 70));
    assert_eq!(r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), [RelicId::SnakeRing]);
    let mut deck = ids(&r.deck);
    deck.sort();
    let mut want = Character::Silent.starting_deck();
    want.sort();
    assert_eq!(deck, want);
    for pool in [&r.pools.common, &r.pools.uncommon, &r.pools.rare] {
        assert!(!pool.is_empty() && pool.iter().all(|c| c.data().color == CardColor::Green));
    }
    assert_eq!(r.pools.common.len() + r.pools.uncommon.len() + r.pools.rare.len(), 71);
    let relics: Vec<RelicId> = [&r.pools.common_relics, &r.pools.uncommon_relics, &r.pools.rare_relics, &r.pools.shop_relics, &r.pools.boss_relics]
        .into_iter()
        .flatten()
        .copied()
        .collect();
    assert!(relics.iter().all(|x| matches!(x.data().class, RelicClass::Shared | RelicClass::Green)), "{relics:?}");
    for x in [RelicId::SneckoSkull, RelicId::Tingsha, RelicId::WristBlade, RelicId::TwistedFunnel, RelicId::RingOfTheSerpent] {
        assert!(relics.contains(&x), "{x:?}");
    }
    // A14 takes 4 max HP from the Silent (5 from the Ironclad); A6 starts at 90%.
    let r = silent_run(5, 14);
    assert_eq!((r.hp, r.max_hp), (59, 66));
}

#[test]
fn the_silents_potion_rewards_come_from_her_pool() {
    let mut rng = StsRandom::new(3);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..2000 {
        seen.insert(sts_core::run::pools::return_random_potion(Character::Silent, &mut rng));
    }
    for p in [PotionId::PoisonPotion, PotionId::CunningPotion, PotionId::GhostInAJar] {
        assert!(seen.contains(&p), "{p:?}");
    }
    for p in [PotionId::BloodPotion, PotionId::Elixir, PotionId::HeartOfIron] {
        assert!(!seen.contains(&p), "{p:?}");
    }
}

#[test]
fn ring_of_the_serpent_replaces_ring_of_the_snake() {
    let mut r = silent_run(5, 0);
    r.obtain_relic(RelicId::RingOfTheSerpent);
    assert_eq!(r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), [RelicId::RingOfTheSerpent]);
}

/// The random policy from tests/run_random.rs: plays cards more often than it ends turns and
/// rests when hurt.
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
fn random_silent_runs_play_through_without_errors() {
    let mut later = 0;
    for seed in 0..60i64 {
        let mut run = silent_run(seed, if seed % 3 == 0 { 20 } else { 0 });
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
    // Random play gets the Silent past Act 1 less often than the Ironclad (139 against 161 of
    // 200 runs at A0 with this policy): her starting deck hits softer and random discards hurt.
    assert!(later > 20, "{later}");
}


// ---- Plays and upgrades the tests above leave out ----

/// Backstab, Endless Agony and Masterful Stab played (the tests above only draw them), base and
/// upgraded; the first two exhaust.
#[test]
fn backstab_endless_agony_and_masterful_stab_hit() {
    use CardId::*;
    for (id, dmg, up) in [(Backstab, 11, 15), (EndlessAgony, 4, 6), (MasterfulStab, 12, 16)] {
        for plus in [false, true] {
            let mut c = setup("Jaw Worm", &[], &[]);
            let card = if plus { upgraded(&mut c, id) } else { c.new_card(id) };
            c.hand.push(card);
            let m = m0(&c);
            let hp = c.monsters[m].hp;
            play(&mut c, id, Some(m));
            assert_eq!(c.monsters[m].hp, hp - if plus { up } else { dmg }, "{id:?} upgraded {plus}");
            assert_eq!((c.exhaust_pile.len(), c.energy), ((id != MasterfulStab) as usize, 3), "{id:?}");
        }
    }
    // Endless Agony+'s copy is upgraded too (makeStatEquivalentCopy).
    let mut c = setup("Jaw Worm", &[QuickSlash], &[]);
    let agony = upgraded(&mut c, EndlessAgony);
    c.draw_pile.push(agony);
    let m = m0(&c);
    play(&mut c, QuickSlash, Some(m));
    assert_eq!(ids(&c.hand), [EndlessAgony, EndlessAgony]);
    assert!(c.hand.iter().all(|h| h.upgraded && h.base_damage == 6));
    // Masterful Stab after a hit costs 1.
    let mut c = setup("Jaw Worm", &[MasterfulStab], &[]);
    end_turn(&mut c);
    let m = m0(&c);
    play(&mut c, MasterfulStab, Some(m));
    assert_eq!(c.energy, 2);
}

/// The upgrades that change what a card does rather than its numbers.
#[test]
fn adrenaline_doppelganger_and_outmaneuver_upgraded() {
    use CardId::*;
    // Adrenaline+: 2 energy instead of 1.
    let mut c = setup("Jaw Worm", &[], &[Slice, Slice]);
    let a = upgraded(&mut c, Adrenaline);
    c.hand.push(a);
    play(&mut c, Adrenaline, None);
    assert_eq!((c.energy, c.hand.len(), c.exhaust_pile.len()), (5, 2, 1));
    // Doppelganger+: X + 1 energy and cards next turn.
    let mut c = setup("Jaw Worm", &[], &[DefendG; 12]);
    let d = upgraded(&mut c, Doppelganger);
    c.hand.push(d);
    play(&mut c, Doppelganger, None);
    end_turn(&mut c);
    assert_eq!((c.energy, c.hand.len()), (3 + 4, 5 + 4));
    // Outmaneuver+: 3 energy next turn instead of 2.
    let mut c = setup("Jaw Worm", &[], &[DefendG; 10]);
    let o = upgraded(&mut c, Outmaneuver);
    c.hand.push(o);
    play(&mut c, Outmaneuver, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Energized), 3);
    end_turn(&mut c);
    assert_eq!(c.energy, 6);
}
