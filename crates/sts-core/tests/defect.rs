//! The Defect's orbs, cards, powers, relics and potions, checked against the decompiled game
//! (`orbs/`, `cards/blue`, the powers and actions they use). Expected numbers come from reading
//! those classes.

use sts_core::action::Cr;
use sts_core::cards::{Card, CardId, BLUE_LIBRARY_ORDER};
use sts_core::character::Character;
use sts_core::combat::{Combat, CombatConfig, OrbId, PlayerAction};
use sts_core::encounters;
use sts_core::potions::PotionId;
use sts_core::powers::PowerId;
use sts_core::relics::{RelicClass, RelicId, RelicState};
use sts_core::rng::StsRandom;
use sts_core::run::{Run, RunAction, RunConfig, Screen};

/// A Defect fight with `deck` and no relics (so no orb is channeled at the start).
fn fight(seed: i64, encounter: &str, deck: &[(CardId, i32)]) -> Combat {
    let mut cfg = CombatConfig::new(Character::Defect, seed, encounter);
    cfg.deck = deck.to_vec();
    cfg.relics.clear();
    Combat::new(&cfg).unwrap()
}

/// A fight whose hand is exactly `hand` and whose draw pile is `draw` (top last).
fn setup(encounter: &str, hand: &[CardId], draw: &[CardId]) -> Combat {
    let mut c = fight(7, encounter, &[(CardId::DefendB, 0); 5]);
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

fn end_turn(c: &mut Combat) {
    c.step(&PlayerAction::EndTurn).unwrap();
}

fn m0(c: &Combat) -> usize {
    c.order[0]
}

fn power(c: &Combat, who: Cr, id: PowerId) -> i32 {
    c.power_amount(who, id)
}

/// The orb slots as (id, passive, evoke), `None` for empty.
fn orbs(c: &Combat) -> Vec<Option<(OrbId, i32, i32)>> {
    c.player.orbs.iter().map(|o| o.map(|o| (o.id, c.orb_passive(&o), c.orb_evoke(&o)))).collect()
}

fn orb_ids(c: &Combat) -> Vec<Option<OrbId>> {
    c.player.orbs.iter().map(|o| o.map(|o| o.id)).collect()
}

// ---- Fuzz ----

/// Random Defect decks (every card, some upgraded, her relics now and then) against every
/// encounter, with random legal play: no panics, no illegal states, every playable card played.
#[test]
fn random_defect_fights_stay_consistent() {
    let mut all: Vec<&str> = encounters::all_act1();
    all.extend(encounters::all_act2());
    all.extend(encounters::all_act3());
    let relics = [RelicId::CrackedCore, RelicId::DataDisk, RelicId::EmotionChip, RelicId::FrozenCore, RelicId::GoldPlatedCables, RelicId::Inserter, RelicId::NuclearBattery, RelicId::RunicCapacitor, RelicId::SymbioticVirus];
    let mut pick = StsRandom::new(9);
    let mut played = std::collections::HashSet::new();
    for seed in 0..500i64 {
        let encounter = all[seed as usize % all.len()];
        let n = 10 + pick.random(10);
        let deck: Vec<(CardId, i32)> =
            (0..n).map(|_| (BLUE_LIBRARY_ORDER[pick.random(BLUE_LIBRARY_ORDER.len() as i32 - 1) as usize], pick.random(1))).collect();
        let mut cfg = CombatConfig::new(Character::Defect, seed, encounter);
        cfg.deck = deck.clone();
        cfg.player_hp = 300;
        cfg.player_max_hp = 300;
        cfg.relics.push(relics[seed as usize % relics.len()]);
        cfg.potions = vec![Some(PotionId::FocusPotion), Some(PotionId::PotionOfCapacity), Some(PotionId::EssenceOfDarkness)];
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
            assert!(c.player.orbs.len() <= 10, "seed {seed}: {} slots", c.player.orbs.len());
            // Orbs fill the slots from the front.
            let filled = c.player.orbs.iter().take_while(|o| o.is_some()).count();
            assert!(c.player.orbs[filled..].iter().all(|o| o.is_none()), "seed {seed}: {:?}", c.player.orbs);
            steps += 1;
        }
    }
    let missing: Vec<CardId> = BLUE_LIBRARY_ORDER.iter().copied().filter(|id| !played.contains(id)).collect();
    assert!(missing.is_empty(), "never played: {missing:?}");
}

// ---- Orbs ----

#[test]
fn a_defect_fight_starts_with_three_slots_and_cracked_core_lightning() {
    let c = Combat::new(&CombatConfig::new(Character::Defect, 3, "Jaw Worm")).unwrap();
    assert_eq!((c.player.hp, c.player.max_hp), (75, 75));
    assert_eq!(orbs(&c), [Some((OrbId::Lightning, 3, 8)), None, None]);
    assert_eq!(c.hand.len(), 5);
}

#[test]
fn orbs_act_at_the_end_of_the_turn() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Zap, Coolheaded, Darkness], &[DefendB; 5]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Zap, None);
    play(&mut c, Coolheaded, None);
    play(&mut c, Darkness, None);
    assert_eq!(orbs(&c), [Some((OrbId::Lightning, 3, 8)), Some((OrbId::Frost, 2, 5)), Some((OrbId::Dark, 6, 6))]);
    end_turn(&mut c);
    // Lightning hit for 3, Frost blocked 2 of Chomp's 11, Dark's evoke grew by its passive.
    assert_eq!(c.monsters[m].hp, hp - 3);
    assert_eq!(c.player.hp, 75 - 9);
    assert_eq!(orbs(&c)[2], Some((OrbId::Dark, 6, 12)));
}

#[test]
fn channeling_into_full_slots_evokes_the_front_orb() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Zap, Zap, Zap, Zap], &[]);
    c.energy = 4;
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    for _ in 0..4 {
        play(&mut c, Zap, None);
    }
    assert_eq!(c.monsters[m].hp, hp - 8);
    assert_eq!(orb_ids(&c), [Some(OrbId::Lightning); 3]);
}

#[test]
fn focus_raises_every_orb_but_plasma_and_dark_evoke() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Defragment, Zap, Coolheaded, Darkness], &[DefendB; 5]);
    c.energy = 4;
    play(&mut c, Defragment, None);
    play(&mut c, Zap, None);
    play(&mut c, Coolheaded, None);
    play(&mut c, Darkness, None);
    assert_eq!(orbs(&c), [Some((OrbId::Lightning, 4, 9)), Some((OrbId::Frost, 3, 6)), Some((OrbId::Dark, 7, 6))]);
    let mut c = setup("Jaw Worm", &[Defragment, Fusion], &[]);
    play(&mut c, Defragment, None);
    play(&mut c, Fusion, None);
    assert_eq!(orbs(&c)[0], Some((OrbId::Plasma, 1, 2)));
    end_turn(&mut c);
    assert_eq!(c.energy, 4);
}

#[test]
fn dualcast_and_lock_on() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Zap, Dualcast], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Zap, None);
    play(&mut c, Dualcast, None);
    assert_eq!(c.monsters[m].hp, hp - 16);
    assert_eq!(orb_ids(&c), [None, None, None]);

    let mut c = setup("Jaw Worm", &[LockOn, Zap, Dualcast], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, LockOn, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::LockOn), 2);
    play(&mut c, Zap, None);
    play(&mut c, Dualcast, None);
    assert_eq!(c.monsters[m].hp, hp - 8 - 24);
}

#[test]
fn electrodynamics_lightning_hits_everything() {
    let mut c = setup("Small Slimes", &[CardId::Electrodynamics], &[]);
    let hps: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    play(&mut c, CardId::Electrodynamics, None);
    end_turn(&mut c);
    for (k, &m) in c.order.iter().enumerate() {
        assert_eq!(c.monsters[m].hp, hps[k] - 6);
    }
}

#[test]
fn dark_evoke_hits_the_monster_with_the_least_hp() {
    use CardId::*;
    let mut c = setup("Small Slimes", &[Darkness, Dualcast], &[]);
    let (weak, strong) = if c.monsters[c.order[0]].hp < c.monsters[c.order[1]].hp { (c.order[0], c.order[1]) } else { (c.order[1], c.order[0]) };
    c.monsters[weak].hp = 30;
    c.monsters[strong].hp = 40;
    play(&mut c, Darkness, None);
    play(&mut c, Dualcast, None);
    assert_eq!((c.monsters[weak].hp, c.monsters[strong].hp), (18, 40));
}

#[test]
fn multicast_recursion_fission_barrage_compile_driver() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Zap, MultiCast], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Zap, None);
    play(&mut c, MultiCast, None);
    assert_eq!(c.monsters[m].hp, hp - 16);
    assert_eq!((c.energy, c.has_orb()), (0, false));
    // With no orb, Multi-Cast does nothing and keeps the energy.
    let mut c = setup("Jaw Worm", &[MultiCast], &[]);
    play(&mut c, MultiCast, None);
    assert_eq!(c.energy, 3);

    let mut c = setup("Jaw Worm", &[Darkness, Recursion], &[]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Darkness, None);
    play(&mut c, Recursion, None);
    assert_eq!(c.monsters[m].hp, hp - 6);
    assert_eq!(orbs(&c), [Some((OrbId::Dark, 6, 6)), None, None]);

    for up in [false, true] {
        let mut c = setup("Jaw Worm", &[Zap, Coolheaded], &[DefendB; 5]);
        let fission = if up { upgraded(&mut c, Fission) } else { c.new_card(Fission) };
        c.hand.push(fission);
        let m = m0(&c);
        let hp = c.monsters[m].hp;
        play(&mut c, Zap, None);
        play(&mut c, Coolheaded, None);
        play(&mut c, Fission, None);
        assert_eq!((c.energy, c.hand.len(), c.filled_orb_count()), (3, 3, 0));
        // Fission+ evokes them: Lightning's 8 damage and Frost's 5 block.
        assert_eq!((c.monsters[m].hp, c.player.block), if up { (hp - 8, 5) } else { (hp, 0) });
    }

    let mut c = setup("Jaw Worm", &[Zap, Coolheaded, Barrage, CompileDriver], &[DefendB; 5]);
    c.energy = 4;
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Zap, None);
    play(&mut c, Coolheaded, None);
    play(&mut c, Barrage, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 8);
    let before = c.hand.len();
    play(&mut c, CompileDriver, Some(m));
    assert_eq!(c.hand.len(), before - 1 + 2);
}

#[test]
fn blizzard_and_thunder_strike_count_the_orbs_channeled_this_combat() {
    use CardId::*;
    let mut c = setup("Small Slimes", &[Coolheaded, Coolheaded, Blizzard], &[DefendB; 5]);
    let hps: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    play(&mut c, Coolheaded, None);
    play(&mut c, Coolheaded, None);
    play(&mut c, Blizzard, None);
    for (k, &m) in c.order.iter().enumerate() {
        assert_eq!(c.monsters[m].hp, hps[k] - 4);
    }
    let mut c = setup("Jaw Worm", &[Zap, Zap, ThunderStrike], &[]);
    c.energy = 5;
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Zap, None);
    play(&mut c, Zap, None);
    play(&mut c, ThunderStrike, None);
    assert_eq!(c.monsters[m].hp, hp - 14);
}

#[test]
fn orb_slots_capacitor_consume_and_the_cap_of_ten() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Capacitor, Consume], &[]);
    play(&mut c, Capacitor, None);
    assert_eq!(c.player.orbs.len(), 5);
    play(&mut c, Consume, None);
    assert_eq!(c.player.orbs.len(), 4);
    assert_eq!(power(&c, Cr::Player, PowerId::Focus), 2);
    let mut c = setup("Jaw Worm", &[Capacitor, Capacitor, Capacitor, Capacitor], &[]);
    c.energy = 4;
    for _ in 0..4 {
        play(&mut c, Capacitor, None);
    }
    assert_eq!(c.player.orbs.len(), 10);
}

#[test]
fn loop_repeats_the_front_orbs_passive_at_the_start_of_the_turn() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Coolheaded, Loop], &[DefendB; 6]);
    play(&mut c, Coolheaded, None);
    play(&mut c, Loop, None);
    end_turn(&mut c);
    assert_eq!(c.player.block, 2);
}

// ---- Cards that change cards ----

#[test]
fn claw_steam_barrier_streamline_and_genetic_algorithm_change_themselves() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Claw, Claw], &[Claw]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Claw, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 3);
    assert_eq!((c.hand[0].base_damage, c.draw_pile[0].base_damage, c.discard_pile[0].base_damage), (5, 5, 5));

    let mut c = setup("Jaw Worm", &[SteamBarrier, Streamline], &[]);
    let m = m0(&c);
    play(&mut c, SteamBarrier, None);
    play(&mut c, Streamline, Some(m));
    assert_eq!(c.player.block, 6);
    assert_eq!((c.discard_pile[0].base_block, c.discard_pile[1].cost), (5, 1));

    let mut c = setup("Jaw Worm", &[GeneticAlgorithm], &[]);
    let uuid = c.hand[0].uuid;
    play(&mut c, GeneticAlgorithm, None);
    assert_eq!(c.player.block, 1);
    assert_eq!((c.exhaust_pile[0].misc, c.exhaust_pile[0].base_block), (3, 3));
    assert_eq!(c.master_deck_growth, [(uuid, 2)]);
}

#[test]
fn force_field_costs_less_for_each_power_played() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Defragment, ForceField], &[]);
    play(&mut c, Defragment, None);
    assert_eq!(c.hand[idx(&c, ForceField)].cost, 3);
    // A Force Field made later in the fight starts cheaper too.
    let made = c.new_card(ForceField);
    assert_eq!(made.cost, 3);
}

#[test]
fn rebound_puts_the_next_card_on_top_of_the_draw_pile() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Rebound, Zap], &[DefendB]);
    let m = m0(&c);
    play(&mut c, Rebound, Some(m));
    assert!(c.player.has_power(PowerId::Rebound));
    play(&mut c, Zap, None);
    assert_eq!(c.draw_pile.last().unwrap().id, Zap);
    assert!(!c.player.has_power(PowerId::Rebound));
}

#[test]
fn seek_offers_the_draw_pile_in_the_games_grid_order() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Seek], &[Zap, DefendB, Leap, Hyperbeam, Dazed]);
    play(&mut c, Seek, None);
    let p = c.pending.as_ref().unwrap();
    let order: Vec<CardId> = p.options.iter().map(|&i| c.draw_pile[i].id).collect();
    // Name order, then rarity (rare first, a stable sort), statuses last.
    assert_eq!(order, [Hyperbeam, Leap, DefendB, Zap, Dazed]);
    let k = p.options[0];
    c.step(&PlayerAction::Choose { indices: vec![k] }).unwrap();
    assert_eq!(ids(&c.hand), [Hyperbeam]);
}

#[test]
fn recycle_hologram_all_for_one() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Recycle, Hyperbeam], &[]);
    play(&mut c, Recycle, None);
    assert_eq!(c.energy, 4);
    assert_eq!(c.exhaust_pile[0].id, Hyperbeam);

    let mut c = setup("Jaw Worm", &[Hologram], &[]);
    let h = c.new_card(Hyperbeam);
    c.discard_pile.push(h);
    play(&mut c, Hologram, None);
    assert_eq!((c.hand[0].id, c.hand[0].cost_for_turn), (Hyperbeam, 2));
    assert_eq!((c.player.block, c.exhaust_pile.len()), (3, 1));

    let mut c = setup("Jaw Worm", &[AllForOne], &[]);
    let m = m0(&c);
    for id in [Claw, Zap, FTL] {
        let card = c.new_card(id);
        c.discard_pile.push(card);
    }
    play(&mut c, AllForOne, Some(m));
    assert_eq!(ids(&c.hand), [Claw, FTL]);
}

#[test]
fn reboot_scrape_skim_and_draw_counting_cards() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Reboot, Zap, DefendB], &[StrikeB, StrikeB, StrikeB]);
    for _ in 0..2 {
        let l = c.new_card(Leap);
        c.discard_pile.push(l);
    }
    play(&mut c, Reboot, None);
    assert_eq!((c.hand.len(), c.draw_pile.len(), c.discard_pile.len()), (4, 3, 0));
    assert_eq!(c.exhaust_pile[0].id, Reboot);

    // Scrape draws 4 (the top last), then discards the ones that cost something.
    let mut c = setup("Jaw Worm", &[Scrape], &[Zap, Claw, FTL, Leap]);
    let m = m0(&c);
    play(&mut c, Scrape, Some(m));
    assert_eq!(ids(&c.hand), [FTL, Claw]);
    assert_eq!(c.total_discarded_this_turn, 2);

    let mut c = setup("Jaw Worm", &[FTL, FTL, FTL, FTL], &[DefendB; 6]);
    let m = m0(&c);
    for _ in 0..4 {
        play(&mut c, FTL, Some(m));
    }
    // The first three draw (fewer than 3 cards played before each).
    assert_eq!(c.hand.len(), 3);

    let mut c = setup("Jaw Worm", &[Aggregate, DoubleEnergy], &[DefendB; 9]);
    play(&mut c, Aggregate, None);
    assert_eq!(c.energy, 4);
    play(&mut c, DoubleEnergy, None);
    assert_eq!(c.energy, 6);
}

#[test]
fn attacks_go_for_the_eyes_sunder_melter_hyperbeam() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[GoForTheEyes, Sunder], &[]);
    let m = m0(&c);
    play(&mut c, GoForTheEyes, Some(m));
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Weakened), 1);
    c.monsters[m].hp = 20;
    play(&mut c, Sunder, Some(m));
    assert!(c.is_over());

    let mut c = setup("Jaw Worm", &[Melter], &[]);
    let m = m0(&c);
    c.monsters[m].block = 10;
    let hp = c.monsters[m].hp;
    play(&mut c, Melter, Some(m));
    assert_eq!((c.monsters[m].block, c.monsters[m].hp), (0, hp - 10));

    let mut c = setup("Small Slimes", &[Hyperbeam], &[]);
    let hps: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    play(&mut c, Hyperbeam, None);
    for (k, &m) in c.order.iter().enumerate() {
        assert_eq!(c.monsters[m].hp, (hps[k] - 26).max(0));
    }
    assert_eq!(power(&c, Cr::Player, PowerId::Focus), -3);
}

#[test]
fn x_costs_and_orb_makers() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Tempest], &[]);
    play(&mut c, Tempest, None);
    assert_eq!(orb_ids(&c), [Some(OrbId::Lightning); 3]);
    let mut c = setup("Jaw Worm", &[ReinforcedBody], &[]);
    play(&mut c, ReinforcedBody, None);
    assert_eq!(c.player.block, 21);
    let mut c = setup("Jaw Worm", &[Rainbow], &[]);
    play(&mut c, Rainbow, None);
    assert_eq!(orb_ids(&c), [Some(OrbId::Lightning), Some(OrbId::Frost), Some(OrbId::Dark)]);
    let mut c = setup("Small Slimes", &[Chill], &[]);
    play(&mut c, Chill, None);
    assert_eq!(orb_ids(&c), [Some(OrbId::Frost), Some(OrbId::Frost), None]);
    let mut c = setup("Jaw Worm", &[], &[]);
    let d = upgraded(&mut c, Darkness);
    c.hand.push(d);
    play(&mut c, Darkness, None);
    assert_eq!(orbs(&c)[0], Some((OrbId::Dark, 6, 12)));
    let mut c = setup("Jaw Worm", &[], &[]);
    let chaos = upgraded(&mut c, Chaos);
    c.hand.push(chaos);
    play(&mut c, Chaos, None);
    assert_eq!(c.filled_orb_count(), 2);
    let mut c = setup("Jaw Worm", &[MeteorStrike], &[]);
    c.energy = 5;
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, MeteorStrike, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 24);
    assert_eq!(orb_ids(&c), [Some(OrbId::Plasma); 3]);
}

#[test]
fn block_cards_auto_shields_stack_equilibrium() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[AutoShields, AutoShields], &[]);
    play(&mut c, AutoShields, None);
    play(&mut c, AutoShields, None);
    assert_eq!(c.player.block, 11);
    let mut c = setup("Jaw Worm", &[Stack], &[]);
    for _ in 0..4 {
        let d = c.new_card(DefendB);
        c.discard_pile.push(d);
    }
    play(&mut c, Stack, None);
    assert_eq!(c.player.block, 4);
    let mut c = setup("Jaw Worm", &[Equilibrium, Zap, DefendB], &[StrikeB; 5]);
    play(&mut c, Equilibrium, None);
    assert_eq!(c.player.block, 13);
    end_turn(&mut c);
    let mut hand = ids(&c.hand);
    hand.sort();
    assert_eq!(hand, [DefendB, StrikeB, StrikeB, StrikeB, StrikeB, StrikeB, Zap]);
}

// ---- Powers ----

#[test]
fn echo_form_doubles_the_first_card_each_turn() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[EchoForm], &[Zap, Zap, DefendB, DefendB, DefendB]);
    play(&mut c, EchoForm, None);
    end_turn(&mut c);
    play(&mut c, Zap, None);
    assert_eq!(c.filled_orb_count(), 2);
    play(&mut c, Zap, None);
    assert_eq!(c.filled_orb_count(), 3);

    // Echo Form itself is the turn's first card: the card after it isn't doubled (recorded
    // fight, 2026-10-19; a new EchoPower has doubled nothing yet).
    let mut c = setup("Jaw Worm", &[EchoForm, Zap], &[]);
    c.energy = 4;
    play(&mut c, EchoForm, None);
    play(&mut c, Zap, None);
    assert_eq!(c.filled_orb_count(), 1);

    // The copy plays with the original's count (makeSameInstanceOf keeps Stack's base block, and
    // nothing counts again): 2 + 2 with 2 cards in the discard pile, not 2 + 3 with the first Stack
    // in it by then (soak 2026-10-27, a Defect run at the Giant Head: 6 block in the game, 7 in the
    // engine).
    let mut c = setup("Jaw Worm", &[EchoForm], &[Stack, DefendB, DefendB, DefendB, DefendB]);
    play(&mut c, EchoForm, None);
    end_turn(&mut c);
    for _ in 0..2 {
        let s = c.new_card(StrikeB);
        c.discard_pile.push(s);
    }
    play(&mut c, Stack, None);
    assert_eq!(c.player.block, 4);
}

#[test]
fn amplify_storm_heatsinks_static_discharge() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[Amplify, Defragment], &[]);
    play(&mut c, Amplify, None);
    play(&mut c, Defragment, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Focus), 2);

    let mut c = setup("Jaw Worm", &[Storm, Heatsinks, Defragment], &[DefendB]);
    play(&mut c, Storm, None);
    play(&mut c, Heatsinks, None);
    // Heatsinks itself is a Power: Storm channels for it.
    assert_eq!(c.filled_orb_count(), 1);
    play(&mut c, Defragment, None);
    assert_eq!(c.filled_orb_count(), 2);
    assert_eq!(ids(&c.hand), [DefendB]);

    let mut c = setup("Jaw Worm", &[StaticDischarge], &[]);
    play(&mut c, StaticDischarge, None);
    end_turn(&mut c);
    assert_eq!(orb_ids(&c)[0], Some(OrbId::Lightning));
}

#[test]
fn biased_cognition_machine_learning_hello_world_creative_ai() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[BiasedCognition, MachineLearning, HelloWorld], &[DefendB; 10]);
    c.energy = 4;
    play(&mut c, BiasedCognition, None);
    play(&mut c, MachineLearning, None);
    play(&mut c, HelloWorld, None);
    assert_eq!(power(&c, Cr::Player, PowerId::Focus), 4);
    end_turn(&mut c);
    assert_eq!(power(&c, Cr::Player, PowerId::Focus), 3);
    // Six drawn, and Hello World's common card.
    assert_eq!(c.hand.len(), 7);
    let made: Vec<&Card> = c.hand.iter().filter(|h| h.id != DefendB).collect();
    assert_eq!(made.len(), 1);
    assert_eq!(made[0].data().rarity, sts_core::cards::CardRarity::Common);

    let mut c = setup("Jaw Worm", &[CreativeAI], &[]);
    play(&mut c, CreativeAI, None);
    end_turn(&mut c);
    assert_eq!(c.hand.len(), 1);
    assert_eq!(c.hand[0].card_type(), sts_core::cards::CardType::Power);
}

#[test]
fn self_repair_heals_at_the_end_of_the_fight() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[SelfRepair, StrikeB], &[]);
    c.player.hp = 50;
    let m = m0(&c);
    play(&mut c, SelfRepair, None);
    c.monsters[m].hp = 1;
    play(&mut c, StrikeB, Some(m));
    assert!(c.is_over());
    assert_eq!(c.player.hp, 57);
}

#[test]
fn conserve_battery_core_surge_beam_cell_overclock_turbo_white_noise() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[ConserveBattery], &[]);
    play(&mut c, ConserveBattery, None);
    end_turn(&mut c);
    assert_eq!(c.energy, 4);
    let mut c = setup("Jaw Worm", &[CoreSurge, BeamCell], &[]);
    let m = m0(&c);
    play(&mut c, CoreSurge, Some(m));
    play(&mut c, BeamCell, Some(m));
    assert_eq!(power(&c, Cr::Player, PowerId::Artifact), 1);
    assert_eq!(power(&c, Cr::Monster(m), PowerId::Vulnerable), 1);
    let mut c = setup("Jaw Worm", &[Overclock, Turbo], &[DefendB, DefendB]);
    play(&mut c, Overclock, None);
    play(&mut c, Turbo, None);
    assert_eq!(c.energy, 5);
    let made: Vec<CardId> = c.discard_pile.iter().map(|d| d.id).filter(|id| matches!(id, Burn | Void)).collect();
    assert_eq!(made, [Burn, Void]);
    let mut c = setup("Jaw Worm", &[WhiteNoise], &[]);
    play(&mut c, WhiteNoise, None);
    assert_eq!(c.hand[0].card_type(), sts_core::cards::CardType::Power);
    assert!(c.hand[0].cost_for_turn <= 0 || c.hand[0].cost < 0);
}

// ---- Relics and potions ----

fn with_relic(c: &mut Combat, r: RelicId) {
    c.player.relics.push(RelicState::new(r));
}

fn fight_with(encounter: &str, relics: &[RelicId], deck: &[(CardId, i32)]) -> Combat {
    let mut cfg = CombatConfig::new(Character::Defect, 7, encounter);
    cfg.deck = deck.to_vec();
    cfg.relics = relics.to_vec();
    Combat::new(&cfg).unwrap()
}

#[test]
fn start_of_fight_relics_channel_orbs_and_focus() {
    let deck = [(CardId::DefendB, 0); 10];
    let c = fight_with("Jaw Worm", &[RelicId::CrackedCore, RelicId::SymbioticVirus, RelicId::NuclearBattery, RelicId::DataDisk], &deck);
    assert_eq!(orb_ids(&c), [Some(OrbId::Lightning), Some(OrbId::Dark), Some(OrbId::Plasma)]);
    assert_eq!(power(&c, Cr::Player, PowerId::Focus), 1);
    // Plasma's start-of-turn energy already came in on turn 1.
    assert_eq!(c.energy, 4);
}

#[test]
fn frozen_core_gold_plated_cables_inserter_runic_capacitor() {
    let deck = [(CardId::DefendB, 0); 10];
    let mut c = fight_with("Jaw Worm", &[RelicId::FrozenCore], &deck);
    end_turn(&mut c);
    assert_eq!(orb_ids(&c), [Some(OrbId::Frost), None, None]);

    let mut c = setup("Jaw Worm", &[CardId::Coolheaded], &[CardId::DefendB; 6]);
    with_relic(&mut c, RelicId::GoldPlatedCables);
    play(&mut c, CardId::Coolheaded, None);
    end_turn(&mut c);
    // Frost's passive twice (4 block) against Chomp's 11.
    assert_eq!(c.player.hp, 75 - 7);

    let mut c = fight_with("Jaw Worm", &[RelicId::Inserter], &deck);
    assert_eq!(c.player.orbs.len(), 3);
    end_turn(&mut c);
    assert_eq!(c.player.orbs.len(), 4);

    let c = fight_with("Jaw Worm", &[RelicId::RunicCapacitor], &deck);
    assert_eq!(c.player.orbs.len(), 6);
}

#[test]
fn emotion_chip_triggers_every_orb_after_losing_hp() {
    let mut c = setup("Jaw Worm", &[CardId::Coolheaded], &[CardId::DefendB; 6]);
    with_relic(&mut c, RelicId::EmotionChip);
    play(&mut c, CardId::Coolheaded, None);
    end_turn(&mut c);
    // Chomp got through Frost's 2 block, so the turn starts with another 2.
    assert_eq!(c.player.block, 2);
}

#[test]
fn the_defects_potions() {
    let mut c = setup("Jaw Worm", &[], &[]);
    c.potions = vec![Some(PotionId::FocusPotion), Some(PotionId::PotionOfCapacity), Some(PotionId::EssenceOfDarkness)];
    c.step(&PlayerAction::UsePotion { slot: 0, target: None }).unwrap();
    assert_eq!(power(&c, Cr::Player, PowerId::Focus), 2);
    c.step(&PlayerAction::UsePotion { slot: 1, target: None }).unwrap();
    assert_eq!(c.player.orbs.len(), 5);
    c.step(&PlayerAction::UsePotion { slot: 2, target: None }).unwrap();
    assert_eq!(orb_ids(&c), [Some(OrbId::Dark); 5]);
}

// ---- Runs ----

fn defect_run(seed: i64, ascension: i32) -> Run {
    let mut cfg = RunConfig::new(seed);
    cfg.character = Character::Defect;
    cfg.ascension = ascension;
    Run::new(cfg)
}

#[test]
fn a_defect_run_starts_with_her_deck_relic_and_pools() {
    use sts_core::cards::CardColor;
    let r = defect_run(5, 0);
    assert_eq!((r.hp, r.max_hp), (75, 75));
    assert_eq!(r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), [RelicId::CrackedCore]);
    let mut deck = ids(&r.deck);
    deck.sort();
    let mut want = Character::Defect.starting_deck();
    want.sort();
    assert_eq!(deck, want);
    for pool in [&r.pools.common, &r.pools.uncommon, &r.pools.rare] {
        assert!(!pool.is_empty() && pool.iter().all(|c| c.data().color == CardColor::Blue));
    }
    let relics: Vec<RelicId> = [&r.pools.common_relics, &r.pools.uncommon_relics, &r.pools.rare_relics, &r.pools.shop_relics, &r.pools.boss_relics]
        .into_iter()
        .flatten()
        .copied()
        .collect();
    assert!(relics.iter().all(|x| matches!(x.data().class, RelicClass::Shared | RelicClass::Blue)));
    assert!(relics.contains(&RelicId::FrozenCore) && relics.contains(&RelicId::GoldPlatedCables));
    assert_eq!((defect_run(5, 14).max_hp, defect_run(5, 14).hp), (71, 64));
    let mut r = defect_run(5, 0);
    r.obtain_relic(RelicId::FrozenCore);
    assert_eq!(r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), [RelicId::FrozenCore]);
}

#[test]
fn a_defect_runs_fights_start_with_her_orbs() {
    let mut r = defect_run(3, 0);
    r.step(&RunAction::Neow { option: 1 }).unwrap();
    while !matches!(r.screen, Screen::Combat) {
        let a = r.legal_actions()[0].clone();
        r.step(&a).unwrap();
    }
    let c = r.combat.as_ref().unwrap();
    assert_eq!(c.character, Character::Defect);
    assert_eq!(orb_ids(c), [Some(OrbId::Lightning), None, None]);
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
fn random_defect_runs_play_through_without_errors() {
    let mut later = 0;
    for seed in 0..60i64 {
        let mut run = defect_run(seed, if seed % 3 == 0 { 20 } else { 0 });
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

#[test]
fn upgrades_match_each_cards_upgrade_method() {
    use CardId::*;
    // (card, base damage, base block, base magic, cost) after one upgrade.
    let table = [
        (StrikeB, 9, 0, 0, 1),
        (DefendB, 0, 8, 0, 1),
        (Zap, 0, 0, 1, 0),
        (Dualcast, 0, 0, 0, 0),
        (Aggregate, 0, 0, 3, 1),
        (AllForOne, 14, 0, 0, 2),
        (Amplify, 0, 0, 2, 1),
        (AutoShields, 0, 15, 0, 1),
        (BallLightning, 10, 0, 1, 1),
        (Barrage, 6, 0, 0, 1),
        (BeamCell, 4, 0, 2, 0),
        (BiasedCognition, 0, 0, 5, 1),
        (Blizzard, 0, 0, 3, 1),
        (BootSequence, 0, 13, 0, 0),
        (Buffer, 0, 0, 2, 2),
        (Capacitor, 0, 0, 3, 1),
        (Chaos, 0, 0, 2, 1),
        (Claw, 5, 0, 2, 0),
        (ColdSnap, 9, 0, 1, 1),
        (CompileDriver, 10, 0, 1, 1),
        (ConserveBattery, 0, 10, 0, 1),
        (Consume, 0, 0, 3, 2),
        (Coolheaded, 0, 0, 2, 1),
        (CoreSurge, 15, 0, 1, 1),
        (CreativeAI, 0, 0, 1, 2),
        (Defragment, 0, 0, 2, 1),
        (DoomAndGloom, 14, 0, 1, 2),
        (DoubleEnergy, 0, 0, 0, 0),
        (Electrodynamics, 0, 0, 3, 2),
        (Equilibrium, 0, 16, 1, 2),
        (FTL, 6, 0, 4, 0),
        (ForceField, 0, 16, 0, 4),
        (Fusion, 0, 0, 1, 1),
        (GeneticAlgorithm, 0, 1, 3, 1),
        (Glacier, 0, 10, 2, 2),
        (GoForTheEyes, 4, 0, 2, 0),
        (Heatsinks, 0, 0, 2, 1),
        (Hologram, 0, 5, 0, 1),
        (Hyperbeam, 34, 0, 3, 2),
        (Leap, 0, 12, 0, 1),
        (LockOn, 11, 0, 3, 1),
        (Loop, 0, 0, 2, 1),
        (Melter, 14, 0, 0, 1),
        (MeteorStrike, 30, 0, 3, 5),
        (Overclock, 0, 0, 3, 0),
        (Reboot, 0, 0, 6, 0),
        (Rebound, 12, 0, 0, 1),
        (Recursion, 0, 0, 0, 0),
        (Recycle, 0, 0, 0, 0),
        (ReinforcedBody, 0, 9, 0, -1),
        (Reprogram, 0, 0, 2, 1),
        (RipAndTear, 9, 0, 2, 1),
        (Scrape, 10, 0, 5, 1),
        (Seek, 0, 0, 2, 0),
        (SelfRepair, 0, 0, 10, 1),
        (Skim, 0, 0, 4, 1),
        (StaticDischarge, 0, 0, 2, 1),
        (SteamBarrier, 0, 8, 0, 0),
        (Streamline, 20, 0, 1, 2),
        (Sunder, 32, 0, 0, 3),
        (SweepingBeam, 9, 0, 1, 1),
        (ThunderStrike, 9, 0, 0, 3),
        (Turbo, 0, 0, 3, 0),
        (WhiteNoise, 0, 0, 0, 0),
    ];
    let mut c = setup("Jaw Worm", &[], &[]);
    for (id, damage, block, magic, cost) in table {
        let card = upgraded(&mut c, id);
        assert_eq!((card.base_damage, card.base_block, card.base_magic, card.cost), (damage, block, magic, cost), "{id:?}+");
    }
    for id in [Chill, HelloWorld, MachineLearning, Storm] {
        assert!(upgraded(&mut c, id).innate, "{id:?}");
    }
    assert!(!upgraded(&mut c, EchoForm).ethereal);
    assert!(!upgraded(&mut c, Rainbow).exhaust);
    for id in [Darkness, Fission, MultiCast, Tempest] {
        assert!(upgraded(&mut c, id).upgraded);
    }
}

#[test]
fn simple_cards_hit_block_channel_and_draw() {
    use CardId::*;
    let mut c = setup("Jaw Worm", &[BallLightning, ColdSnap, Glacier], &[]);
    c.energy = 4;
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, BallLightning, Some(m));
    play(&mut c, ColdSnap, Some(m));
    play(&mut c, Glacier, None);
    // 7 and 6, and Glacier's second Frost evokes the front Lightning (8).
    assert_eq!(c.monsters[m].hp, hp - 21);
    assert_eq!(c.player.block, 7);
    assert_eq!(orb_ids(&c), [Some(OrbId::Frost); 3]);

    let mut c = setup("Jaw Worm", &[Leap, BootSequence, Buffer], &[]);
    play(&mut c, Leap, None);
    play(&mut c, BootSequence, None);
    play(&mut c, Buffer, None);
    assert_eq!(c.player.block, 19);
    assert_eq!(power(&c, Cr::Player, PowerId::Buffer), 1);

    let mut c = setup("Small Slimes", &[DoomAndGloom, SweepingBeam], &[DefendB]);
    let hps: Vec<i32> = c.order.iter().map(|&m| c.monsters[m].hp).collect();
    play(&mut c, DoomAndGloom, None);
    play(&mut c, SweepingBeam, None);
    for (k, &m) in c.order.iter().enumerate() {
        assert_eq!(c.monsters[m].hp, (hps[k] - 16).max(0));
    }
    assert_eq!(orb_ids(&c)[0], Some(OrbId::Dark));

    let mut c = setup("Jaw Worm", &[Reprogram, RipAndTear, Skim], &[DefendB; 3]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Reprogram, None);
    assert_eq!((power(&c, Cr::Player, PowerId::Focus), power(&c, Cr::Player, PowerId::Strength), power(&c, Cr::Player, PowerId::Dexterity)), (-1, 1, 1));
    play(&mut c, RipAndTear, None);
    assert_eq!(c.monsters[m].hp, hp - 16);
    play(&mut c, Skim, None);
    assert_eq!(c.hand.len(), 3);
}

// ---- Plays and upgrades the tests above leave out ----

#[test]
fn defend_and_force_field_block() {
    use CardId::*;
    for (plus, blk) in [(false, 5), (true, 8)] {
        let mut c = setup("Jaw Worm", &[], &[]);
        let d = if plus { upgraded(&mut c, DefendB) } else { c.new_card(DefendB) };
        c.hand.push(d);
        play(&mut c, DefendB, None);
        assert_eq!(c.player.block, blk);
    }
    // Force Field: 12 Block (16 upgraded); after one Power it costs 3.
    for (plus, blk) in [(false, 12), (true, 16)] {
        let mut c = setup("Jaw Worm", &[Defragment], &[]);
        c.energy = 4;
        let f = if plus { upgraded(&mut c, ForceField) } else { c.new_card(ForceField) };
        c.hand.push(f);
        play(&mut c, Defragment, None);
        play(&mut c, ForceField, None);
        assert_eq!((c.player.block, c.energy), (blk, 0));
    }
}

/// The upgrades that change what a card does rather than its numbers.
#[test]
fn multicast_tempest_stack_and_hologram_upgraded() {
    use CardId::*;
    // Multi-Cast+: evokes X + 1 times (a Lightning orb: 8 each).
    let mut c = setup("Jaw Worm", &[Zap], &[]);
    let mc = upgraded(&mut c, MultiCast);
    c.hand.push(mc);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Zap, None);
    play(&mut c, MultiCast, None);
    assert_eq!(c.monsters[m].hp, hp - 8 * 3);
    // Tempest+: X + 1 Lightning; the fourth pushes the first out (evoked: 8).
    let mut c = setup("Jaw Worm", &[], &[]);
    let t = upgraded(&mut c, Tempest);
    c.hand.push(t);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, Tempest, None);
    assert_eq!(orb_ids(&c), [Some(OrbId::Lightning); 3]);
    assert_eq!(c.monsters[m].hp, hp - 8);
    // Stack+: Block equal to the discard pile, plus 3.
    let mut c = setup("Jaw Worm", &[], &[]);
    let s = upgraded(&mut c, Stack);
    c.hand.push(s);
    for _ in 0..4 {
        let d = c.new_card(DefendB);
        c.discard_pile.push(d);
    }
    play(&mut c, Stack, None);
    assert_eq!(c.player.block, 4 + 3);
    // Hologram+: 5 Block and no exhaust.
    let mut c = setup("Jaw Worm", &[], &[]);
    let h = upgraded(&mut c, Hologram);
    c.hand.push(h);
    let z = c.new_card(Zap);
    c.discard_pile.push(z);
    play(&mut c, Hologram, None);
    assert_eq!((ids(&c.hand), c.player.block), (vec![Zap], 5));
    assert_eq!((ids(&c.discard_pile), c.exhaust_pile.len()), (vec![Hologram], 0));
}

/// Spire Shield's Bash takes 1 Focus or 1 Strength by aiRng when the player has orb slots
/// (`!player.orbs.isEmpty() && aiRng.randomBoolean()`), so a Defect loses either; it took Strength
/// always (the Ironclad's port), and a Defect run's Act 4 drifted from the game's (2026-10-25).
#[test]
fn spire_shields_bash_takes_focus_or_strength() {
    let (mut focus, mut strength) = (0, 0);
    for seed in 0..16 {
        let mut c = fight(seed, "Shield and Spear", &[(CardId::DefendB, 0); 10]);
        c.player.hp = 500;
        c.player.max_hp = 500;
        let shield = c.monsters.iter().position(|m| m.id == sts_core::monsters::MonsterId::SpireShield).unwrap();
        c.monsters[shield].next_move = 1;
        let before = (power(&c, Cr::Player, PowerId::Focus), power(&c, Cr::Player, PowerId::Strength));
        end_turn(&mut c);
        match (before.0 - power(&c, Cr::Player, PowerId::Focus), before.1 - power(&c, Cr::Player, PowerId::Strength)) {
            (1, 0) => focus += 1,
            (0, 1) => strength += 1,
            d => panic!("seed {seed}: Focus and Strength lost {d:?}"),
        }
    }
    assert!(focus > 0 && strength > 0, "Focus {focus}, Strength {strength} of 16");
}

/// A card made free for the turn (Mummified Hand, Enlightenment...) costs its cost again once it
/// goes back into the draw pile: the game's Soul resets it as it lands (`clearPowers`). Reboot puts
/// the hand back, so the redrawn Strike costs 1 (soak 2026-10-27, run GJPD: the engine kept it at 0
/// and played it with no energy).
#[test]
fn a_card_put_back_into_the_draw_pile_loses_its_cost_for_the_turn() {
    let mut c = setup("Jaw Worm", &[CardId::StrikeB, CardId::Reboot], &[]);
    let s = idx(&c, CardId::StrikeB);
    c.hand[s].set_cost_for_turn(0);
    play(&mut c, CardId::Reboot, None);
    let strike = c.hand.iter().find(|x| x.id == CardId::StrikeB).expect("Reboot draws the Strike back");
    assert_eq!(strike.cost_for_turn, 1);
}
