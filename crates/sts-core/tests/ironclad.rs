//! The Ironclad's 75 cards and the powers they make, one test per behaviour, checked against the
//! decompiled game (`cards/red`, the powers and actions they use). Expected numbers come from
//! reading those classes; the recordings and soaks checked the same cards in whole fights.

use sts_core::action::Cr;
use sts_core::cards::{Card, CardId, RED_LIBRARY_ORDER};
use sts_core::character::Character;
use sts_core::combat::{ChoicePile, Combat, CombatConfig, PlayerAction};
use sts_core::encounters;
use sts_core::powers::PowerId;
use sts_core::rng::StsRandom;

/// An Ironclad fight with `deck` and no relics (no Burning Blood heal to account for).
fn fight(seed: i64, encounter: &str, deck: &[(CardId, i32)]) -> Combat {
    let mut cfg = CombatConfig::new(Character::Ironclad, seed, encounter);
    cfg.deck = deck.to_vec();
    cfg.relics.clear();
    Combat::new(&cfg).unwrap()
}

/// A fight whose hand is exactly `hand` and whose draw pile is `draw` (top last).
fn setup(encounter: &str, hand: &[CardId], draw: &[CardId]) -> Combat {
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

fn count(cards: &[Card], id: CardId) -> usize {
    cards.iter().filter(|c| c.id == id).count()
}

// ---- Fuzz ----

/// Random Ironclad decks (every card, some upgraded) against every encounter, with random legal
/// play: no panics, no illegal states, every card played.
#[test]
fn random_ironclad_fights_stay_consistent() {
    let mut all: Vec<&str> = encounters::all_act1();
    all.extend(encounters::all_act2());
    all.extend(encounters::all_act3());
    let mut pick = StsRandom::new(13);
    let mut played = std::collections::HashSet::new();
    for seed in 0..500i64 {
        let encounter = all[seed as usize % all.len()];
        let n = 10 + pick.random(10);
        let deck: Vec<(CardId, i32)> =
            (0..n).map(|_| (RED_LIBRARY_ORDER[pick.random(RED_LIBRARY_ORDER.len() as i32 - 1) as usize], pick.random(1))).collect();
        let mut cfg = CombatConfig::new(Character::Ironclad, seed, encounter);
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
            assert!(c.hand.len() <= 10 && c.energy >= 0 && c.player.hp >= 0, "seed {seed}");
            steps += 1;
        }
    }
    let missing: Vec<CardId> = RED_LIBRARY_ORDER.iter().copied().filter(|id| !played.contains(id)).collect();
    assert!(missing.is_empty(), "never played: {missing:?}");
}

// ---- Attacks ----

#[test]
fn simple_attacks_deal_their_damage() {
    use CardId::*;
    // (card, damage to Jaw Worm, extra energy needed)
    let cases = [
        (StrikeR, 6),
        (Anger, 6),
        (Bludgeon, 32),
        (Carnage, 20),
        (Headbutt, 9),
        (HeavyBlade, 14),
        (IronWave, 5),
        (PommelStrike, 9),
        (TwinStrike, 10),
        (WildStrike, 12),
        (RecklessCharge, 7),
        (Clothesline, 12),
        (Uppercut, 13),
        (Hemokinesis, 15),
        (SearingBlow, 12),
        (SeverSoul, 16),
        (Rampage, 8),
        (Dropkick, 5),
        (Pummel, 8),
        (SwordBoomerang, 9),
        (Bash, 8),
    ];
    for (id, dmg) in cases {
        let mut c = setup("Jaw Worm", &[id], &[CardId::DefendR; 4]);
        c.energy = 3;
        let m = m0(&c);
        let before = hp(&c, m);
        play(&mut c, id, Some(m));
        assert_eq!(before - hp(&c, m), dmg, "{id:?}");
    }
}

#[test]
fn bash_uppercut_and_clothesline_debuff() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Bash, StrikeR, Uppercut, Clothesline], &[]);
    c.energy = 10;
    let m = m0(&c);
    c.monsters[m].hp = 200;
    play(&mut c, Bash, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Vulnerable), 2);
    let before = hp(&c, m);
    play(&mut c, StrikeR, Some(m));
    // Vulnerable: 6 * 1.5.
    assert_eq!(before - hp(&c, m), 9);
    play(&mut c, Uppercut, Some(m));
    assert_eq!((power(&c, Cr::Monster(m), PowerId::Vulnerable), power(&c, Cr::Monster(m), PowerId::Weakened)), (3, 1));
    play(&mut c, Clothesline, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 3);
}

#[test]
fn anger_copies_itself_into_the_discard_pile() {
    let mut c = setup("Jaw Worm", &[CardId::Anger], &[]);
    let m = m0(&c);
    play(&mut c, CardId::Anger, Some(m));
    assert_eq!(count(&c.discard_pile, CardId::Anger), 2);
}

#[test]
fn clash_needs_a_hand_of_attacks() {
    let c = setup("Jaw Worm", &[CardId::Clash, CardId::DefendR], &[]);
    let m = m0(&c);
    assert!(!c.can_use(&c.hand[0], Some(m)));
    let mut c = setup("Jaw Worm", &[CardId::Clash, CardId::StrikeR], &[]);
    let before = hp(&c, m);
    play(&mut c, CardId::Clash, Some(m));
    assert_eq!(before - hp(&c, m), 14);
}

#[test]
fn heavy_blade_and_perfected_strike_scale() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Inflame, HeavyBlade], &[]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, Inflame, None);
    play(&mut c, HeavyBlade, Some(m));
    // 14 + 2 Strength x3.
    assert_eq!(before - hp(&c, m), 20);

    let mut c = setup("Jaw Worm", &[PerfectedStrike, StrikeR], &[StrikeR, TwinStrike, DefendR]);
    let before = hp(&c, m);
    play(&mut c, PerfectedStrike, Some(m));
    // 6 + 2 per Strike card (Perfected Strike itself, Strike, Strike, Twin Strike).
    assert_eq!(before - hp(&c, m), 6 + 2 * 4);
}

#[test]
fn body_slam_hits_for_the_block() {
    let mut c = setup("Jaw Worm", &[CardId::DefendR, CardId::BodySlam], &[]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::DefendR, None);
    play(&mut c, CardId::BodySlam, Some(m));
    assert_eq!(before - hp(&c, m), 5);
}

#[test]
fn rampage_grows_and_searing_blow_upgrades_without_end() {
    let mut c = setup("Jaw Worm", &[CardId::Rampage], &[]);
    let m = m0(&c);
    play(&mut c, CardId::Rampage, Some(m));
    assert_eq!(c.discard_pile[0].base_damage, 13);

    let mut c = setup("Jaw Worm", &[], &[]);
    let mut sb = c.new_card(CardId::SearingBlow);
    let mut damage = vec![sb.base_damage];
    for _ in 0..4 {
        assert!(sb.can_upgrade());
        sb.upgrade();
        damage.push(sb.base_damage);
    }
    // +4, +5, +6, +7 (4 + times upgraded).
    assert_eq!(damage, [12, 16, 21, 27, 34]);
    assert_eq!(sb.name(), "Searing Blow+4");
}

#[test]
fn blood_for_blood_costs_less_each_time_hp_is_lost() {
    let mut c = setup("Jaw Worm", &[CardId::BloodForBlood, CardId::Hemokinesis], &[CardId::DefendR; 6]);
    let m = m0(&c);
    play(&mut c, CardId::Hemokinesis, Some(m));
    assert_eq!(c.player.hp, 80 - 2);
    assert_eq!(c.hand[idx(&c, CardId::BloodForBlood)].cost_for_turn, 3);
    end_turn(&mut c);
    // Chomp's 11 lands: one more.
    assert_eq!(c.hand.iter().chain(c.draw_pile.iter()).chain(c.discard_pile.iter()).find(|x| x.id == CardId::BloodForBlood).unwrap().cost, 2);
}

#[test]
fn dropkick_refunds_and_draws_against_vulnerable() {
    let mut c = setup("Jaw Worm", &[CardId::Dropkick, CardId::Dropkick, CardId::Bash], &[CardId::DefendR; 3]);
    c.energy = 4;
    let m = m0(&c);
    play(&mut c, CardId::Dropkick, Some(m));
    assert_eq!((c.energy, c.hand.len()), (3, 2));
    play(&mut c, CardId::Bash, Some(m));
    play(&mut c, CardId::Dropkick, Some(m));
    assert_eq!((c.energy, c.hand.len()), (1, 1));
}

#[test]
fn twin_strike_pummel_and_sword_boomerang_hit_several_times() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[TwinStrike, Pummel, SwordBoomerang], &[]);
    c.energy = 10;
    c.player.powers.push(sts_core::powers::Power::new(PowerId::Strength, 1));
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, TwinStrike, Some(m));
    assert_eq!(before - hp(&c, m), 2 * 6);
    let before = hp(&c, m);
    play(&mut c, Pummel, Some(m));
    assert_eq!(before - hp(&c, m), 4 * 3);
    assert!(c.exhaust_pile.iter().any(|x| x.id == Pummel));
    let before = hp(&c, m);
    play(&mut c, SwordBoomerang, None);
    assert_eq!(before - hp(&c, m), 3 * 4);
}

#[test]
fn hitting_every_enemy() {
    use CardId::*;
    let mut c = setup("3 Sentries", &[Cleave, ThunderClap, Immolate], &[]);
    c.energy = 10;
    let before: Vec<i32> = c.order.iter().map(|&m| hp(&c, m)).collect();
    play(&mut c, Cleave, None);
    play(&mut c, ThunderClap, None);
    play(&mut c, Immolate, None);
    // Sentries' Artifact stops Thunderclap's Vulnerable, so 8 + 4 + 21 each.
    let after: Vec<i32> = c.order.iter().map(|&m| hp(&c, m)).collect();
    for (b, a) in before.iter().zip(after) {
        assert_eq!(b - a, 33);
    }
    assert_eq!(count(&c.discard_pile, Burn), 1);
}

#[test]
fn whirlwind_hits_once_per_energy() {
    let mut c = setup("Jaw Worm", &[CardId::Whirlwind], &[]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::Whirlwind, None);
    assert_eq!((before - hp(&c, m), c.energy), (15, 0));
}

#[test]
fn reaper_heals_the_unblocked_damage() {
    let mut c = setup("3 Sentries", &[CardId::Reaper], &[]);
    c.player.hp = 50;
    play(&mut c, CardId::Reaper, None);
    assert_eq!(c.player.hp, 50 + 3 * 4);
    assert!(c.exhaust_pile.iter().any(|x| x.id == CardId::Reaper));
}

#[test]
fn feed_raises_max_hp_on_a_kill() {
    let mut c = setup("Cultist", &[CardId::Feed], &[]);
    let m = m0(&c);
    c.monsters[m].hp = 10;
    play(&mut c, CardId::Feed, Some(m));
    assert_eq!((c.player.hp, c.player.max_hp), (83, 83));
}

#[test]
fn fiend_fire_exhausts_the_hand_and_hits_per_card() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[FiendFire, DefendR, DefendR, StrikeR], &[]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, FiendFire, Some(m));
    assert_eq!(before - hp(&c, m), 3 * 7);
    assert!(c.hand.is_empty());
    assert_eq!(c.exhaust_pile.len(), 4);
}

#[test]
fn sever_soul_exhausts_non_attacks() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[SeverSoul, DefendR, StrikeR, ShrugItOff], &[]);
    let m = m0(&c);
    play(&mut c, SeverSoul, Some(m));
    assert_eq!(ids(&c.hand), [StrikeR]);
    assert_eq!(count(&c.exhaust_pile, DefendR) + count(&c.exhaust_pile, ShrugItOff), 2);
}

#[test]
fn wild_strike_and_reckless_charge_shuffle_statuses_in() {
    let mut c = setup("Jaw Worm", &[CardId::WildStrike, CardId::RecklessCharge], &[CardId::DefendR; 3]);
    let m = m0(&c);
    play(&mut c, CardId::WildStrike, Some(m));
    play(&mut c, CardId::RecklessCharge, Some(m));
    assert_eq!((count(&c.draw_pile, CardId::Wound), count(&c.draw_pile, CardId::Dazed)), (1, 1));
}

#[test]
fn headbutt_puts_a_discarded_card_on_top() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Headbutt], &[DefendR]);
    for id in [StrikeR, Bash] {
        let card = c.new_card(id);
        c.discard_pile.push(card);
    }
    let m = m0(&c);
    play(&mut c, Headbutt, Some(m));
    let p = c.pending.as_ref().expect("a choice of two");
    assert_eq!(p.pile, ChoicePile::Discard);
    choose(&mut c, &[1]);
    assert_eq!(c.draw_pile.last().unwrap().id, Bash);
}

// ---- Skills ----

#[test]
fn simple_block_skills() {
    use CardId::*;
    // (card, block gained, cards drawn from a pile of Strikes)
    for (id, blk, drawn) in [(DefendR, 5, 0), (Impervious, 30, 0), (GhostlyArmor, 10, 0), (ShrugItOff, 8, 1), (Sentinel, 5, 0), (FlameBarrier, 12, 0), (IronWave, 5, 0)] {
        let mut c = setup("Jaw Worm", &[id], &[StrikeR; 4]);
        let m = m0(&c);
        play(&mut c, id, Some(m));
        assert_eq!(c.player.block, blk, "{id:?}");
        assert_eq!(count(&c.hand, StrikeR), drawn, "{id:?}");
    }
}

#[test]
fn armaments_upgrades_one_card_or_all() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Armaments, StrikeR, DefendR], &[]);
    play(&mut c, Armaments, None);
    assert_eq!(c.player.block, 5);
    // Two upgradable cards: a choice.
    let k = c.hand.iter().position(|x| x.id == DefendR).unwrap();
    choose(&mut c, &[k]);
    assert!(c.hand.iter().find(|x| x.id == DefendR).unwrap().upgraded);
    assert!(!c.hand.iter().find(|x| x.id == StrikeR).unwrap().upgraded);

    let mut c = setup("Jaw Worm", &[StrikeR, DefendR], &[]);
    let a = upgraded(&mut c, Armaments);
    c.hand.push(a);
    play(&mut c, Armaments, None);
    assert!(c.hand.iter().all(|x| x.upgraded));
}

#[test]
fn battle_trance_draws_then_stops_drawing() {
    let mut c = setup("Jaw Worm", &[CardId::BattleTrance, CardId::ShrugItOff], &[CardId::StrikeR; 6]);
    play(&mut c, CardId::BattleTrance, None);
    assert_eq!(c.hand.len(), 4);
    play(&mut c, CardId::ShrugItOff, None);
    assert_eq!(c.hand.len(), 3);
    assert!(c.player.has_power(PowerId::NoDraw));
}

#[test]
fn bloodletting_offering_and_seeing_red_buy_energy() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Bloodletting, Offering, SeeingRed], &[StrikeR; 6]);
    play(&mut c, Bloodletting, None);
    assert_eq!((c.player.hp, c.energy), (77, 5));
    play(&mut c, Offering, None);
    assert_eq!((c.player.hp, c.energy), (71, 7));
    assert_eq!(count(&c.hand, StrikeR), 3);
    play(&mut c, SeeingRed, None);
    assert_eq!(c.energy, 8);
    assert_eq!(count(&c.exhaust_pile, Offering) + count(&c.exhaust_pile, SeeingRed), 2);
}

#[test]
fn burning_pact_true_grit_and_second_wind_exhaust() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[BurningPact, StrikeR, DefendR], &[Bash; 3]);
    play(&mut c, BurningPact, None);
    let k = c.hand.iter().position(|x| x.id == DefendR).unwrap();
    choose(&mut c, &[k]);
    assert_eq!(ids(&c.exhaust_pile), [DefendR]);
    assert_eq!(count(&c.hand, Bash), 2);

    // True Grit exhausts a random card; True Grit+ asks.
    let mut c = setup("Jaw Worm", &[TrueGrit, StrikeR], &[]);
    play(&mut c, TrueGrit, None);
    assert_eq!((c.player.block, ids(&c.exhaust_pile)), (7, vec![StrikeR]));
    let mut c = setup("Jaw Worm", &[StrikeR, DefendR], &[]);
    let tg = upgraded(&mut c, TrueGrit);
    c.hand.push(tg);
    play(&mut c, TrueGrit, None);
    assert!(c.pending.is_some());
    choose(&mut c, &[1]);
    assert_eq!((c.player.block, ids(&c.exhaust_pile)), (9, vec![DefendR]));

    let mut c = setup("Jaw Worm", &[SecondWind, StrikeR, DefendR, Bash, ShrugItOff], &[]);
    play(&mut c, SecondWind, None);
    assert_eq!(c.player.block, 2 * 5);
    assert_eq!(count(&c.hand, StrikeR) + count(&c.hand, Bash), 2);
}

#[test]
fn disarm_spot_weakness_and_intimidate() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Disarm, SpotWeakness, Intimidate], &[]);
    let m = m0(&c);
    play(&mut c, Disarm, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), -2);
    // Jaw Worm attacks first: Spot Weakness gives 3 Strength.
    play(&mut c, SpotWeakness, Some(m));
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 3);
    play(&mut c, Intimidate, None);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 1);

    // The Cultist's first move is a buff: no Strength.
    let mut c = setup("Cultist", &[SpotWeakness], &[]);
    let m = m0(&c);
    play(&mut c, SpotWeakness, Some(m));
    assert!(!c.player.has_power(PowerId::Strength));
}

#[test]
fn shockwave_weakens_and_exposes_every_enemy() {
    let mut c = setup("Gremlin Gang", &[CardId::Shockwave], &[]);
    play(&mut c, CardId::Shockwave, None);
    for &m in &c.order {
        assert_eq!((power(&c, Cr::Monster(m), PowerId::Weakened), power(&c, Cr::Monster(m), PowerId::Vulnerable)), (3, 3));
    }
}

#[test]
fn double_tap_plays_the_next_attack_twice() {
    let mut c = setup("Jaw Worm", &[CardId::DoubleTap, CardId::StrikeR], &[]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::DoubleTap, None);
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(before - hp(&c, m), 12);
    assert!(!c.player.has_power(PowerId::DoubleTap));
}

#[test]
fn dual_wield_copies_an_attack_or_power() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[DualWield, StrikeR, DefendR], &[]);
    play(&mut c, DualWield, None);
    // Only one card can be copied: no choice.
    assert_eq!(count(&c.hand, StrikeR), 2);
    let mut c = setup("Jaw Worm", &[StrikeR, Inflame], &[]);
    let dw = upgraded(&mut c, DualWield);
    c.hand.push(dw);
    play(&mut c, DualWield, None);
    let k = c.hand.iter().position(|x| x.id == Inflame).unwrap();
    choose(&mut c, &[k]);
    assert_eq!(count(&c.hand, Inflame), 3);
}

#[test]
fn entrench_doubles_block() {
    let mut c = setup("Jaw Worm", &[CardId::DefendR, CardId::Entrench], &[]);
    play(&mut c, CardId::DefendR, None);
    play(&mut c, CardId::Entrench, None);
    assert_eq!(c.player.block, 10);
}

#[test]
fn exhume_returns_an_exhausted_card() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Exhume], &[]);
    for id in [Bash, Impervious] {
        let card = c.new_card(id);
        c.exhaust_pile.push(card);
    }
    play(&mut c, Exhume, None);
    choose(&mut c, &[1]);
    assert!(c.hand.iter().any(|x| x.id == Impervious));
    // Exhume itself exhausts; it can't fetch itself.
    assert_eq!(ids(&c.exhaust_pile), [Bash, Exhume]);
}

#[test]
fn flame_barrier_burns_attackers_for_one_turn() {
    let mut c = setup("Jaw Worm", &[CardId::FlameBarrier], &[CardId::DefendR; 10]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::FlameBarrier, None);
    end_turn(&mut c);
    // Chomp (11) into 12 block; 4 back.
    assert_eq!((hp(&c, m), c.player.hp), (before - 4, 80));
    assert!(!c.player.has_power(PowerId::FlameBarrier));
}

#[test]
fn flex_lasts_until_the_end_of_the_turn() {
    let mut c = setup("Cultist", &[CardId::Flex], &[CardId::DefendR; 10]);
    play(&mut c, CardId::Flex, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 2);
    end_turn(&mut c);
    assert!(!c.player.has_power(PowerId::Strength));
}

#[test]
fn ghostly_armor_and_carnage_are_ethereal() {
    let mut c = setup("Cultist", &[CardId::GhostlyArmor, CardId::Carnage, CardId::StrikeR], &[CardId::DefendR; 10]);
    end_turn(&mut c);
    assert_eq!(ids(&c.exhaust_pile).len(), 2);
    assert_eq!(ids(&c.discard_pile), [CardId::StrikeR]);
}

#[test]
fn havoc_plays_and_exhausts_the_top_card() {
    let mut c = setup("Jaw Worm", &[CardId::Havoc], &[CardId::DefendR, CardId::ShrugItOff]);
    play(&mut c, CardId::Havoc, None);
    assert_eq!(c.player.block, 8);
    assert!(c.exhaust_pile.iter().any(|x| x.id == CardId::ShrugItOff));
    assert_eq!(c.energy, 2);
}

#[test]
fn infernal_blade_adds_a_free_attack() {
    let mut c = setup("Jaw Worm", &[CardId::InfernalBlade], &[]);
    play(&mut c, CardId::InfernalBlade, None);
    assert_eq!(c.hand.len(), 1);
    let card = &c.hand[0];
    assert_eq!(card.card_type(), sts_core::cards::CardType::Attack);
    assert_eq!(card.data().color, sts_core::cards::CardColor::Red);
    assert_eq!(card.cost_for_turn, 0);
}

#[test]
fn limit_break_doubles_strength() {
    let mut c = setup("Jaw Worm", &[CardId::Inflame, CardId::LimitBreak], &[]);
    play(&mut c, CardId::Inflame, None);
    play(&mut c, CardId::LimitBreak, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 4);
    assert!(c.exhaust_pile.iter().any(|x| x.id == CardId::LimitBreak));
    let mut c = setup("Jaw Worm", &[], &[]);
    assert!(!upgraded(&mut c, CardId::LimitBreak).exhaust);
}

#[test]
fn power_through_adds_wounds() {
    let mut c = setup("Jaw Worm", &[CardId::PowerThrough], &[]);
    play(&mut c, CardId::PowerThrough, None);
    assert_eq!((c.player.block, count(&c.hand, CardId::Wound)), (15, 2));
}

#[test]
fn rage_blocks_per_attack_this_turn() {
    let mut c = setup("Cultist", &[CardId::Rage, CardId::StrikeR, CardId::StrikeR], &[CardId::DefendR; 10]);
    let m = m0(&c);
    play(&mut c, CardId::Rage, None);
    play(&mut c, CardId::StrikeR, Some(m));
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.player.block, 6);
    end_turn(&mut c);
    assert!(!c.player.has_power(PowerId::Rage));
}

#[test]
fn sentinel_gives_energy_when_exhausted() {
    let mut c = setup("Jaw Worm", &[CardId::BurningPact, CardId::Sentinel], &[CardId::DefendR; 4]);
    play(&mut c, CardId::BurningPact, None);
    assert_eq!(c.energy, 2 + 2);
    let mut c = setup("Jaw Worm", &[CardId::BurningPact], &[CardId::DefendR; 4]);
    let s = upgraded(&mut c, CardId::Sentinel);
    c.hand.push(s);
    play(&mut c, CardId::BurningPact, None);
    assert_eq!(c.energy, 2 + 3);
}

#[test]
fn warcry_draws_then_puts_a_card_on_top() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Warcry, StrikeR], &[Bash]);
    play(&mut c, Warcry, None);
    let k = c.hand.iter().position(|x| x.id == StrikeR).unwrap();
    choose(&mut c, &[k]);
    assert_eq!(ids(&c.hand), [Bash]);
    assert_eq!(c.draw_pile.last().unwrap().id, StrikeR);
}

// ---- Powers ----

#[test]
fn barricade_keeps_block_and_metallicize_adds_it() {
    let mut c = setup("Cultist", &[CardId::Barricade, CardId::DefendR], &[CardId::DefendR; 10]);
    c.energy = 4;
    play(&mut c, CardId::Barricade, None);
    play(&mut c, CardId::DefendR, None);
    end_turn(&mut c);
    assert_eq!(c.player.block, 5);
    // A second Barricade adds nothing.
    let mut c = setup("Cultist", &[CardId::Metallicize], &[CardId::DefendR; 10]);
    play(&mut c, CardId::Metallicize, None);
    end_turn(&mut c);
    // Gained at the end of the turn, lost at the next start.
    assert_eq!((c.player.block, c.player.hp), (0, 80));
}

#[test]
fn berserk_brutality_and_demon_form_act_each_turn() {
    use CardId::*;
    let mut c = setup("Cultist", &[Berserk, Brutality, DemonForm], &[DefendR; 15]);
    c.energy = 3;
    play(&mut c, Berserk, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Vulnerable), 2);
    play(&mut c, Brutality, None);
    play(&mut c, DemonForm, None);
    end_turn(&mut c);
    // Berserk's energy, Brutality's card and HP, Demon Form's Strength.
    assert_eq!(c.energy, 4);
    assert_eq!(c.hand.len(), 6);
    assert_eq!(c.player.hp, 79);
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 2);
}

#[test]
fn combust_burns_everyone_at_the_end_of_the_turn() {
    let mut c = setup("3 Sentries", &[CardId::Combust, CardId::Combust], &[CardId::DefendR; 10]);
    let before: Vec<i32> = c.order.iter().map(|&m| hp(&c, m)).collect();
    play(&mut c, CardId::Combust, None);
    play(&mut c, CardId::Combust, None);
    end_turn(&mut c);
    // Two copies: 10 damage each, and 2 HP lost (hpLoss grows by one per stack); the middle
    // Sentry's Beam hits for 9.
    let after: Vec<i32> = c.order.iter().map(|&m| hp(&c, m)).collect();
    assert!(before.iter().zip(&after).all(|(b, a)| b - a == 10));
    assert_eq!(c.player.hp, 80 - 2 - 9);
}

#[test]
fn corruption_makes_skills_free_and_exhausts_them() {
    let mut c = setup("Jaw Worm", &[CardId::Corruption, CardId::Impervious, CardId::DefendR], &[]);
    play(&mut c, CardId::Corruption, None);
    assert!(c.hand.iter().all(|x| x.cost_for_turn == 0));
    play(&mut c, CardId::Impervious, None);
    play(&mut c, CardId::DefendR, None);
    assert_eq!((c.energy, c.player.block), (0, 35));
    assert_eq!(c.exhaust_pile.len(), 2);
}

#[test]
fn dark_embrace_and_feel_no_pain_react_to_exhaust() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[DarkEmbrace, FeelNoPain, TrueGrit, StrikeR], &[Bash; 3]);
    c.energy = 10;
    play(&mut c, DarkEmbrace, None);
    play(&mut c, FeelNoPain, None);
    play(&mut c, TrueGrit, None);
    // True Grit's 7, Feel No Pain's 3; Dark Embrace drew a Bash.
    assert_eq!(c.player.block, 10);
    assert_eq!(ids(&c.hand), [Bash]);
}

#[test]
fn evolve_and_fire_breathing_react_to_drawn_statuses() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Evolve, FireBreathing, ShrugItOff], &[DefendR, DefendR, Wound]);
    c.energy = 10;
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, Evolve, None);
    play(&mut c, FireBreathing, None);
    play(&mut c, ShrugItOff, None);
    // The Wound: 6 to every enemy and one more card.
    assert_eq!(before - hp(&c, m), 6);
    assert_eq!(count(&c.hand, DefendR), 1);
}

#[test]
fn juggernaut_hits_when_block_is_gained() {
    let mut c = setup("Jaw Worm", &[CardId::Juggernaut, CardId::DefendR], &[]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::Juggernaut, None);
    play(&mut c, CardId::DefendR, None);
    assert_eq!(before - hp(&c, m), 5);
}

#[test]
fn rupture_turns_self_damage_into_strength() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Rupture, Hemokinesis, Bloodletting], &[]);
    let m = m0(&c);
    play(&mut c, Rupture, None);
    play(&mut c, Bloodletting, None);
    play(&mut c, Hemokinesis, Some(m));
    assert_eq!(power(&c, Cr::Player, PowerId::Strength), 2);
}

// ---- Upgrades ----

#[test]
fn upgrades_match_each_cards_upgrade_method() {
    use CardId::*;
    // (card, base damage, base block, base magic, cost) after one upgrade.
    let table = [
        (Anger, 8, 0, 0, 0),
        (Armaments, 0, 5, 0, 1),
        (Barricade, 0, 0, 0, 2),
        (Bash, 10, 0, 3, 2),
        (BattleTrance, 0, 0, 4, 0),
        (Berserk, 0, 0, 1, 0),
        (BloodForBlood, 22, 0, 0, 3),
        (Bloodletting, 0, 0, 3, 0),
        (Bludgeon, 42, 0, 0, 3),
        (BodySlam, 0, 0, 0, 0),
        (Brutality, 0, 0, 0, 0),
        (BurningPact, 0, 0, 3, 1),
        (Carnage, 28, 0, 0, 2),
        (Clash, 18, 0, 0, 0),
        (Cleave, 11, 0, 0, 1),
        (Clothesline, 14, 0, 3, 2),
        (Combust, 0, 0, 7, 1),
        (Corruption, 0, 0, 3, 2),
        (DarkEmbrace, 0, 0, 0, 1),
        (DefendR, 0, 8, 0, 1),
        (DemonForm, 0, 0, 3, 3),
        (Disarm, 0, 0, 3, 1),
        (DoubleTap, 0, 0, 2, 1),
        (Dropkick, 8, 0, 0, 1),
        (DualWield, 0, 0, 2, 1),
        (Entrench, 0, 0, 0, 1),
        (Evolve, 0, 0, 2, 1),
        (Exhume, 0, 0, 0, 0),
        (Feed, 12, 0, 4, 1),
        (FeelNoPain, 0, 0, 4, 1),
        (FiendFire, 10, 0, 0, 2),
        (FireBreathing, 0, 0, 10, 1),
        (FlameBarrier, 0, 16, 6, 2),
        (Flex, 0, 0, 4, 0),
        (GhostlyArmor, 0, 13, 0, 1),
        (Havoc, 0, 0, 0, 0),
        (Headbutt, 12, 0, 0, 1),
        (HeavyBlade, 14, 0, 5, 2),
        (Hemokinesis, 20, 0, 2, 1),
        (Immolate, 28, 0, 0, 2),
        (Impervious, 0, 40, 0, 2),
        (InfernalBlade, 0, 0, 0, 0),
        (Inflame, 0, 0, 3, 1),
        (Intimidate, 0, 0, 2, 0),
        (IronWave, 7, 7, 0, 1),
        (Juggernaut, 0, 0, 7, 2),
        (LimitBreak, 0, 0, 0, 1),
        (Metallicize, 0, 0, 4, 1),
        (Offering, 0, 0, 5, 0),
        (PerfectedStrike, 6, 0, 3, 2),
        (PommelStrike, 10, 0, 2, 1),
        (PowerThrough, 0, 20, 0, 1),
        (Pummel, 2, 0, 5, 1),
        (Rage, 0, 0, 5, 0),
        (Rampage, 8, 0, 8, 1),
        (Reaper, 5, 0, 0, 2),
        (RecklessCharge, 10, 0, 0, 0),
        (Rupture, 0, 0, 2, 1),
        (SearingBlow, 16, 0, 0, 2),
        (SecondWind, 0, 7, 0, 1),
        (SeeingRed, 0, 0, 0, 0),
        (Sentinel, 0, 8, 0, 1),
        (SeverSoul, 22, 0, 0, 2),
        (Shockwave, 0, 0, 5, 2),
        (ShrugItOff, 0, 11, 0, 1),
        (SpotWeakness, 0, 0, 4, 1),
        (StrikeR, 9, 0, 0, 1),
        (SwordBoomerang, 3, 0, 4, 1),
        (ThunderClap, 7, 0, 0, 1),
        (TrueGrit, 0, 9, 0, 1),
        (TwinStrike, 7, 0, 0, 1),
        (Uppercut, 13, 0, 2, 2),
        (Warcry, 0, 0, 2, 0),
        (Whirlwind, 8, 0, 0, -1),
        (WildStrike, 17, 0, 0, 1),
    ];
    assert_eq!(table.len(), 75);
    let mut c = setup("Jaw Worm", &[], &[]);
    for (id, dmg, blk, magic, cost) in table {
        let card = upgraded(&mut c, id);
        assert!(card.upgraded, "{id:?}");
        assert_eq!((card.base_damage, card.base_block, card.base_magic, card.cost), (dmg, blk, magic, cost), "{id:?}");
    }
    assert!(upgraded(&mut c, Brutality).innate);
    assert!(!upgraded(&mut c, LimitBreak).exhaust);
}
