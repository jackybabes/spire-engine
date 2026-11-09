//! Colorless cards, curses and statuses, one test per behaviour, checked against the decompiled
//! game (`cards/colorless`, `cards/curses`, `cards/status` and the actions and powers they use).

use sts_core::action::Cr;
use sts_core::cards::{Card, CardColor, CardId, CardType, COLORLESS_LIBRARY_ORDER};
use sts_core::character::Character;
use sts_core::combat::{Combat, CombatConfig, PlayerAction};
use sts_core::powers::PowerId;
use sts_core::rng::StsRandom;

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

/// Random decks of colorless cards, curses and statuses beside the Ironclad's basics: no panics,
/// every playable colorless card played.
#[test]
fn random_colorless_fights_stay_consistent() {
    let mut pool: Vec<CardId> = COLORLESS_LIBRARY_ORDER.to_vec();
    pool.extend([CardId::Bite, CardId::JAX, CardId::RitualDagger, CardId::Apparition]);
    let extra = [CardId::Burn, CardId::Wound, CardId::Dazed, CardId::Slimed, CardId::Void, CardId::Decay, CardId::Doubt, CardId::Regret, CardId::Shame, CardId::Pain];
    let mut pick = StsRandom::new(17);
    let mut played = std::collections::HashSet::new();
    let all = sts_core::encounters::all_act1();
    for seed in 0..400i64 {
        let encounter = all[seed as usize % all.len()];
        let mut deck: Vec<(CardId, i32)> = (0..8).map(|_| (pool[pick.random(pool.len() as i32 - 1) as usize], pick.random(1))).collect();
        deck.extend((0..2).map(|_| (extra[pick.random(extra.len() as i32 - 1) as usize], 0)));
        deck.extend([(CardId::StrikeR, 0), (CardId::DefendR, 0)]);
        let mut cfg = CombatConfig::new(Character::Ironclad, seed, encounter);
        cfg.deck = deck.clone();
        cfg.player_hp = 300;
        cfg.player_max_hp = 300;
        let mut c = Combat::new(&cfg).unwrap();
        let mut steps = 0;
        while !c.is_over() && steps < 2000 {
            let legal = c.legal_actions();
            let a = legal[pick.random(legal.len() as i32 - 1) as usize].clone();
            if let PlayerAction::PlayCard { hand_index, .. } = a {
                played.insert(c.hand[hand_index].id);
            }
            c.step(&a).unwrap_or_else(|e| panic!("seed {seed} {deck:?}: {a:?}: {e}"));
            assert!(c.hand.len() <= 10 && c.energy >= 0, "seed {seed}");
            steps += 1;
        }
    }
    let missing: Vec<CardId> = pool.iter().copied().filter(|id| !played.contains(id)).collect();
    assert!(missing.is_empty(), "never played: {missing:?}");
}

// ---- Colorless cards ----

#[test]
fn simple_colorless_cards() {
    use CardId::*;
    // (card, damage to Jaw Worm, block, cards drawn from a pile of Strikes)
    let cases = [
        (SwiftStrike, 7, 0, 0),
        (FlashOfSteel, 3, 0, 1),
        (Finesse, 0, 2, 1),
        (GoodInstincts, 0, 6, 0),
        (MasterOfStrategy, 0, 0, 3),
        (Bite, 7, 0, 0),
        (HandOfGreed, 20, 0, 0),
        (DramaticEntrance, 8, 0, 0),
        (RitualDagger, 15, 0, 0),
        (PanicButton, 0, 30, 0),
    ];
    for (id, dmg, blk, drawn) in cases {
        let mut c = setup("Jaw Worm", &[id], &[CardId::StrikeR; 5]);
        let m = m0(&c);
        let before = hp(&c, m);
        play(&mut c, id, Some(m));
        assert_eq!(before - hp(&c, m), dmg, "{id:?}");
        assert_eq!(c.player.block, blk, "{id:?}");
        assert_eq!(count(&c.hand, CardId::StrikeR), drawn, "{id:?}");
    }
}

#[test]
fn healing_and_strength_cards() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[BandageUp, Bite, JAX], &[]);
    c.player.hp = 50;
    let m = m0(&c);
    play(&mut c, BandageUp, None);
    assert_eq!(c.player.hp, 54);
    play(&mut c, Bite, Some(m));
    assert_eq!(c.player.hp, 56);
    play(&mut c, JAX, None);
    assert_eq!((c.player.hp, power(&c, Cr::Player, PowerId::Strength)), (53, 2));
}

#[test]
fn blind_trip_and_dark_shackles_debuff() {
    use CardId::*;
    let mut c = setup("Gremlin Gang", &[Blind, Trip, DarkShackles], &[DefendR; 10]);
    let m = m0(&c);
    play(&mut c, Blind, Some(m));
    play(&mut c, Trip, Some(m));
    play(&mut c, DarkShackles, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 2);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Vulnerable), 2);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), -9);
    // The upgraded Blind and Trip hit every enemy.
    let mut c = setup("Gremlin Gang", &[], &[]);
    for id in [Blind, Trip] {
        let card = upgraded(&mut c, id);
        c.hand.push(card);
        let i = c.hand.len() - 1;
        c.step(&PlayerAction::PlayCard { hand_index: i, target: None }).unwrap();
    }
    for &m in &c.order {
        assert_eq!((power(&c, Cr::Monster(m), PowerId::Weakened), power(&c, Cr::Monster(m), PowerId::Vulnerable)), (2, 2));
    }
}

#[test]
fn dark_shackles_strength_comes_back_at_the_end_of_the_turn() {
    let mut c = setup("Jaw Worm", &[CardId::DarkShackles], &[CardId::DefendR; 10]);
    let m = m0(&c);
    play(&mut c, CardId::DarkShackles, Some(m));
    end_turn(&mut c);
    // Chomp hits for 11 - 9; the Strength is back once the monster's turn ends.
    assert_eq!(c.player.hp, 80 - 2);
    assert!(!c.monsters[m].has_power(PowerId::Shackled));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Strength), 0);
}

#[test]
fn apotheosis_upgrades_everything() {
    let mut c = setup("Jaw Worm", &[CardId::Apotheosis, CardId::StrikeR], &[CardId::DefendR; 3]);
    let d = c.new_card(CardId::Bash);
    c.discard_pile.push(d);
    play(&mut c, CardId::Apotheosis, None);
    assert!(c.hand.iter().chain(c.draw_pile.iter()).chain(c.discard_pile.iter()).all(|x| x.upgraded));
}

#[test]
fn apparition_caps_damage_at_one() {
    let mut c = setup("Jaw Worm", &[CardId::Apparition], &[CardId::DefendR; 10]);
    play(&mut c, CardId::Apparition, None);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 79);
    // Ethereal and exhausting.
    assert!(c.new_card(CardId::Apparition).ethereal && c.new_card(CardId::Apparition).exhaust);
}

#[test]
fn chrysalis_metamorphosis_and_jack_of_all_trades_make_free_cards() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Chrysalis, Metamorphosis], &[]);
    c.energy = 4;
    play(&mut c, Chrysalis, None);
    play(&mut c, Metamorphosis, None);
    let made: Vec<&Card> = c.draw_pile.iter().collect();
    assert_eq!(made.len(), 6);
    assert_eq!(made.iter().filter(|x| x.card_type() == CardType::Skill).count(), 3);
    assert_eq!(made.iter().filter(|x| x.card_type() == CardType::Attack).count(), 3);
    assert!(made.iter().all(|x| x.cost <= 0 && x.data().color == CardColor::Red));

    let mut c = setup("Jaw Worm", &[], &[]);
    let j = upgraded(&mut c, JackOfAllTrades);
    c.hand.push(j);
    play(&mut c, JackOfAllTrades, None);
    assert_eq!(c.hand.len(), 2);
    assert!(c.hand.iter().all(|x| x.data().color == CardColor::Colorless));
}

#[test]
fn discovery_offers_three_and_the_pick_is_free() {
    let mut c = setup("Jaw Worm", &[CardId::Discovery], &[]);
    play(&mut c, CardId::Discovery, None);
    let offer = ids(&c.pending.as_ref().unwrap().offer);
    assert_eq!(offer.len(), 3);
    choose(&mut c, &[2]);
    assert_eq!(c.hand[0].id, offer[2]);
    assert!(c.hand[0].cost_for_turn <= 0);
}

#[test]
fn deep_breath_reshuffles_the_discard_pile() {
    let mut c = setup("Jaw Worm", &[CardId::DeepBreath], &[]);
    for _ in 0..3 {
        let d = c.new_card(CardId::Bash);
        c.discard_pile.push(d);
    }
    play(&mut c, CardId::DeepBreath, None);
    assert_eq!((c.hand.len(), c.draw_pile.len(), c.discard_pile.len()), (1, 2, 1));
}

#[test]
fn enlightenment_lowers_costs_to_one() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Enlightenment, Bludgeon, Bash, StrikeR], &[]);
    play(&mut c, Enlightenment, None);
    let costs: Vec<(i32, i32)> = c.hand.iter().map(|x| (x.cost, x.cost_for_turn)).collect();
    // This turn only; the upgraded card makes it last the combat.
    assert_eq!(costs, [(3, 1), (2, 1), (1, 1)]);
    let mut c = setup("Jaw Worm", &[Bludgeon], &[]);
    let e = upgraded(&mut c, Enlightenment);
    c.hand.push(e);
    play(&mut c, Enlightenment, None);
    assert_eq!((c.hand[0].cost, c.hand[0].cost_for_turn), (1, 1));
}

#[test]
fn forethought_puts_a_card_under_the_pile_for_free() {
    let mut c = setup("Jaw Worm", &[CardId::Forethought, CardId::Bludgeon], &[CardId::DefendR]);
    play(&mut c, CardId::Forethought, None);
    assert_eq!(c.draw_pile[0].id, CardId::Bludgeon);
    assert!(c.draw_pile[0].free_to_play_once);
}

#[test]
fn hand_of_greed_gives_gold_on_a_kill() {
    let mut c = setup("Cultist", &[CardId::HandOfGreed], &[]);
    let m = m0(&c);
    c.monsters[m].hp = 5;
    let gold = c.player.gold;
    play(&mut c, CardId::HandOfGreed, Some(m));
    assert_eq!(c.player.gold, gold + 20);
}

#[test]
fn impatience_draws_only_without_attacks() {
    let mut c = setup("Jaw Worm", &[CardId::Impatience, CardId::StrikeR], &[CardId::DefendR; 4]);
    play(&mut c, CardId::Impatience, None);
    assert_eq!(c.hand.len(), 1);
    let mut c = setup("Jaw Worm", &[CardId::Impatience, CardId::DefendR], &[CardId::DefendR; 4]);
    play(&mut c, CardId::Impatience, None);
    assert_eq!(c.hand.len(), 3);
}

#[test]
fn madness_makes_a_card_free_for_the_combat() {
    let mut c = setup("Jaw Worm", &[CardId::Madness, CardId::Bludgeon], &[]);
    play(&mut c, CardId::Madness, None);
    assert_eq!((c.hand[0].cost, c.hand[0].cost_for_turn), (0, 0));
}

#[test]
fn magnetism_and_mayhem_act_at_the_start_of_each_turn() {
    let mut c = setup("Cultist", &[CardId::Magnetism], &[CardId::DefendR; 10]);
    play(&mut c, CardId::Magnetism, None);
    end_turn(&mut c);
    assert_eq!(c.hand.iter().filter(|x| x.data().color == CardColor::Colorless).count(), 1);

    // Mayhem's play is queued behind the turn's draw: the card on top after the draw (Shrug It
    // Off) is played; its own draw finds the piles empty.
    let mut c = setup("Cultist", &[CardId::Mayhem], &[CardId::ShrugItOff, CardId::DefendR, CardId::DefendR, CardId::DefendR, CardId::DefendR, CardId::DefendR]);
    play(&mut c, CardId::Mayhem, None);
    end_turn(&mut c);
    assert_eq!(c.player.block, 8);
    assert_eq!(c.hand.len(), 5);
}

#[test]
fn mind_blast_hits_for_the_draw_pile() {
    let mut c = setup("Jaw Worm", &[CardId::MindBlast], &[CardId::DefendR; 7]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::MindBlast, Some(m));
    assert_eq!(before - hp(&c, m), 7);
    assert!(c.new_card(CardId::MindBlast).innate);
}

#[test]
fn panacea_gives_artifact() {
    let mut c = setup("Jaw Worm", &[CardId::Panacea], &[]);
    play(&mut c, CardId::Panacea, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Artifact), 1);
}

#[test]
fn panache_hits_everyone_every_fifth_card() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Panache, DefendR, DefendR, DefendR, DefendR, Finesse], &[StrikeR]);
    c.energy = 10;
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, Panache, None);
    for _ in 0..4 {
        play(&mut c, DefendR, None);
    }
    assert_eq!(hp(&c, m), before);
    play(&mut c, Finesse, None);
    assert_eq!(before - hp(&c, m), 10);
}

#[test]
fn panic_button_stops_block_from_cards() {
    let mut c = setup("Cultist", &[CardId::PanicButton, CardId::DefendR], &[CardId::DefendR; 10]);
    play(&mut c, CardId::PanicButton, None);
    play(&mut c, CardId::DefendR, None);
    assert_eq!(c.player.block, 30);
    assert_eq!(power(&c, Cr::Player, PowerId::NoBlock), 2);
}

#[test]
fn purity_exhausts_up_to_three() {
    let mut c = setup("Jaw Worm", &[CardId::Purity, CardId::Wound, CardId::Wound, CardId::Dazed, CardId::StrikeR], &[]);
    play(&mut c, CardId::Purity, None);
    let p = c.pending.as_ref().unwrap();
    assert!(p.any_number && p.can_pick_zero && p.count == 3);
    choose(&mut c, &[0, 1, 2]);
    assert_eq!(ids(&c.hand), [CardId::StrikeR]);
}

#[test]
fn ritual_dagger_grows_on_a_kill() {
    let mut c = setup("Cultist", &[CardId::RitualDagger], &[]);
    let m = m0(&c);
    c.monsters[m].hp = 5;
    let uuid = c.hand[0].uuid;
    play(&mut c, CardId::RitualDagger, Some(m));
    assert_eq!(c.master_deck_growth, [(uuid, 3)]);
    assert_eq!(c.exhaust_pile[0].base_damage, 18);
}

#[test]
fn sadistic_nature_hurts_on_each_debuff() {
    let mut c = setup("Jaw Worm", &[CardId::SadisticNature, CardId::Bash], &[]);
    let m = m0(&c);
    let before = hp(&c, m);
    play(&mut c, CardId::SadisticNature, None);
    play(&mut c, CardId::Bash, Some(m));
    assert_eq!(before - hp(&c, m), 8 + 5);
}

#[test]
fn secret_technique_and_weapon_fetch_from_the_draw_pile() {
    use CardId::*;
    let c = setup("Jaw Worm", &[SecretTechnique], &[StrikeR]);
    assert!(!c.can_use(&c.hand[0], None));
    let mut c = setup("Jaw Worm", &[SecretTechnique, SecretWeapon], &[StrikeR, DefendR, Bash]);
    play(&mut c, SecretTechnique, None);
    assert!(c.hand.iter().any(|x| x.id == DefendR));
    play(&mut c, SecretWeapon, None);
    // Two attacks: a choice.
    let p = c.pending.as_ref().unwrap();
    let k = *p.options.iter().find(|&&i| c.draw_pile[i].id == Bash).unwrap();
    choose(&mut c, &[k]);
    assert!(c.hand.iter().any(|x| x.id == Bash));
}

#[test]
fn the_bomb_goes_off_after_three_turns() {
    let mut c = setup("3 Sentries", &[CardId::TheBomb], &[CardId::DefendR; 30]);
    c.player.hp = 300;
    c.player.max_hp = 300;
    let before: Vec<i32> = c.order.iter().map(|&m| hp(&c, m)).collect();
    play(&mut c, CardId::TheBomb, None);
    end_turn(&mut c);
    end_turn(&mut c);
    assert!(c.order.iter().zip(&before).all(|(&m, &b)| hp(&c, m) == b));
    end_turn(&mut c);
    assert!(c.order.iter().zip(&before).all(|(&m, &b)| c.monsters[m].is_dying || hp(&c, m) == b - 40));
}

#[test]
fn thinking_ahead_draws_two_and_puts_one_back() {
    let mut c = setup("Jaw Worm", &[CardId::ThinkingAhead], &[CardId::StrikeR, CardId::Bash]);
    play(&mut c, CardId::ThinkingAhead, None);
    choose(&mut c, &[1]);
    assert_eq!(c.hand.len(), 1);
    assert_eq!(c.draw_pile.len(), 1);
}

#[test]
fn transmutation_makes_x_free_colorless_cards() {
    let mut c = setup("Jaw Worm", &[CardId::Transmutation], &[]);
    play(&mut c, CardId::Transmutation, None);
    assert_eq!(c.hand.len(), 3);
    assert!(c.hand.iter().all(|x| x.data().color == CardColor::Colorless && x.cost_for_turn <= 0));
    assert_eq!(c.energy, 0);
}

#[test]
fn violence_fetches_attacks() {
    let mut c = setup("Jaw Worm", &[CardId::Violence], &[CardId::StrikeR, CardId::DefendR, CardId::Bash, CardId::Cleave, CardId::Anger]);
    play(&mut c, CardId::Violence, None);
    assert_eq!(c.hand.len(), 3);
    assert!(c.hand.iter().all(|x| x.card_type() == CardType::Attack));
}

// ---- Curses and statuses ----

#[test]
fn end_of_turn_curses_and_burns() {
    use CardId::*;
    let mut c = setup("Cultist", &[Burn, Decay, Regret, DefendR], &[DefendR; 10]);
    end_turn(&mut c);
    // Burn 2, Decay 2, Regret = cards in hand (4).
    assert_eq!(c.player.hp, 80 - 2 - 2 - 4);
    let mut c = setup("Cultist", &[], &[DefendR; 10]);
    let b = upgraded(&mut c, Burn);
    c.hand.push(b);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 76);

    let mut c = setup("Cultist", &[Doubt, Shame], &[DefendR; 10]);
    end_turn(&mut c);
    assert_eq!((power(&c, Cr::Player, PowerId::Weakened), power(&c, Cr::Player, PowerId::Frail)), (1, 1));
}

#[test]
fn burn_is_blocked_by_block() {
    let mut c = setup("Cultist", &[CardId::Burn, CardId::DefendR], &[CardId::DefendR; 10]);
    play(&mut c, CardId::DefendR, None);
    end_turn(&mut c);
    assert_eq!(c.player.hp, 80);
}

#[test]
fn pain_and_normality_restrict_play() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Pain, DefendR, DefendR], &[]);
    play(&mut c, DefendR, None);
    play(&mut c, DefendR, None);
    assert_eq!(c.player.hp, 78);
    let mut c = setup("Jaw Worm", &[Normality, Finesse, Finesse, Finesse, Finesse], &[DefendR; 5]);
    for _ in 0..3 {
        play(&mut c, Finesse, None);
    }
    let k = idx(&c, Finesse);
    assert!(!c.can_use(&c.hand[k], None));
}

#[test]
fn ethereal_cards_exhaust_and_pride_copies_itself() {
    use CardId::*;
    let mut c = setup("Cultist", &[Clumsy, Dazed, AscendersBane, Void, Wound, Injury], &[DefendR; 10]);
    end_turn(&mut c);
    assert_eq!(c.exhaust_pile.len(), 4);
    assert_eq!(c.discard_pile.len(), 2);

    let mut c = setup("Cultist", &[Pride], &[DefendR; 10]);
    play(&mut c, Pride, None);
    end_turn(&mut c);
    assert!(c.exhaust_pile.iter().any(|x| x.id == Pride));
    assert!(c.new_card(Pride).innate && c.new_card(Writhe).innate);

    let mut c = setup("Cultist", &[Pride], &[DefendR; 10]);
    end_turn(&mut c);
    // Unplayed at the end of the turn: a copy goes on top of the draw pile, then the draw.
    assert_eq!(count(&c.hand, Pride) + count(&c.draw_pile, Pride) + count(&c.discard_pile, Pride), 2);
}

#[test]
fn unplayable_curses_and_statuses() {
    use CardId::*;
    let c = setup("Jaw Worm", &[Wound, Injury, Writhe, Parasite, CurseOfTheBell, Clumsy, Decay, Necronomicurse, Dazed, Burn], &[]);
    for card in &c.hand {
        assert!(!c.can_use(card, None), "{:?}", card.id);
    }
    let mut c = setup("Jaw Worm", &[Slimed], &[]);
    play(&mut c, Slimed, None);
    assert_eq!((c.energy, ids(&c.exhaust_pile)), (2, vec![Slimed]));
}

#[test]
fn void_takes_energy_when_drawn() {
    let mut c = setup("Jaw Worm", &[CardId::ShrugItOff], &[CardId::Void]);
    play(&mut c, CardId::ShrugItOff, None);
    assert_eq!(c.energy, 1);
}

#[test]
fn necronomicurse_comes_back_when_exhausted() {
    let mut c = setup("Jaw Worm", &[CardId::TrueGrit, CardId::Necronomicurse], &[]);
    play(&mut c, CardId::TrueGrit, None);
    assert_eq!(ids(&c.hand), [CardId::Necronomicurse]);
    assert!(c.exhaust_pile.iter().any(|x| x.id == CardId::Necronomicurse));
}

#[test]
fn colorless_upgrades_match_each_cards_upgrade_method() {
    use CardId::*;
    // (card, base damage, base block, base magic, cost) after one upgrade.
    let table = [
        (Apotheosis, 0, 0, 0, 1),
        (Apparition, 0, 0, 0, 1),
        (BandageUp, 0, 0, 6, 0),
        (Bite, 8, 0, 3, 1),
        (Blind, 0, 0, 2, 0),
        (Chrysalis, 0, 0, 5, 2),
        (DarkShackles, 0, 0, 15, 0),
        (DeepBreath, 0, 0, 2, 0),
        (Discovery, 0, 0, 0, 1),
        (DramaticEntrance, 12, 0, 0, 0),
        (Enlightenment, 0, 0, 0, 0),
        (Finesse, 0, 4, 0, 0),
        (FlashOfSteel, 6, 0, 0, 0),
        (Forethought, 0, 0, 0, 0),
        (GoodInstincts, 0, 9, 0, 0),
        (HandOfGreed, 25, 0, 25, 2),
        (Impatience, 0, 0, 3, 0),
        (JAX, 0, 0, 3, 0),
        (JackOfAllTrades, 0, 0, 2, 0),
        (Madness, 0, 0, 0, 0),
        (Magnetism, 0, 0, 1, 1),
        (MasterOfStrategy, 0, 0, 4, 0),
        (Mayhem, 0, 0, 1, 1),
        (Metamorphosis, 0, 0, 5, 2),
        (MindBlast, 0, 0, 0, 1),
        (Panacea, 0, 0, 2, 0),
        (Panache, 0, 0, 14, 0),
        (PanicButton, 0, 40, 2, 0),
        (Purity, 0, 0, 5, 0),
        (RitualDagger, 15, 0, 5, 1),
        (SadisticNature, 0, 0, 7, 0),
        (SecretTechnique, 0, 0, 0, 0),
        (SecretWeapon, 0, 0, 0, 0),
        (SwiftStrike, 10, 0, 0, 0),
        (TheBomb, 0, 0, 50, 2),
        (ThinkingAhead, 0, 0, 0, 0),
        (Transmutation, 0, 0, 0, -1),
        (Trip, 0, 0, 2, 0),
        (Violence, 0, 0, 4, 0),
        (Burn, 0, 0, 4, -2),
    ];
    let mut c = setup("Jaw Worm", &[], &[]);
    for (id, dmg, blk, magic, cost) in table {
        let card = upgraded(&mut c, id);
        assert!(card.upgraded, "{id:?}");
        assert_eq!((card.base_damage, card.base_block, card.base_magic, card.cost), (dmg, blk, magic, cost), "{id:?}");
    }
    assert!(!upgraded(&mut c, Apparition).ethereal);
    for id in [Discovery, SecretTechnique, SecretWeapon, ThinkingAhead] {
        assert!(!upgraded(&mut c, id).exhaust, "{id:?}");
    }
    // Curses and the other statuses don't upgrade.
    for id in [Wound, Dazed, Slimed, Void, Injury, Regret] {
        assert!(!c.new_card(id).can_upgrade(), "{id:?}");
    }
}

// ---- Upgrades the tests above leave out ----

/// The upgrades that change what a card does rather than its numbers.
#[test]
fn forethought_and_transmutation_upgraded() {
    use CardId::*;
    // Forethought+: any number of cards to the bottom of the draw pile, those that cost energy
    // free once (`if (c.cost > 0)`).
    let mut c = setup("Jaw Worm", &[Bludgeon, Anger, DefendR], &[StrikeR]);
    let f = upgraded(&mut c, Forethought);
    c.hand.push(f);
    play(&mut c, Forethought, None);
    let p = c.pending.as_ref().expect("Forethought+'s choice");
    assert!(p.any_number && p.can_pick_zero);
    choose(&mut c, &[0, 1]);
    assert_eq!(ids(&c.draw_pile[..2]), [Anger, Bludgeon]);
    assert_eq!((c.draw_pile[0].free_to_play_once, c.draw_pile[1].free_to_play_once), (false, true));
    assert_eq!(ids(&c.hand), [DefendR]);
    // Transmutation+: the cards come upgraded.
    let mut c = setup("Jaw Worm", &[], &[]);
    let t = upgraded(&mut c, Transmutation);
    c.hand.push(t);
    play(&mut c, Transmutation, None);
    assert_eq!(c.hand.len(), 3);
    assert!(c.hand.iter().all(|x| x.upgraded || !x.can_upgrade()));
}
