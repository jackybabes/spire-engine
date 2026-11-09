//! Behaviour tests. Expected numbers come from reading the decompiled game code
//! (`reference/src`), e.g. Vulnerable is x1.5 floored, Weak is x0.75 floored.

use sts_core::action::Cr;
use sts_core::cards::CardId;
use sts_core::combat::{Combat, CombatConfig, Outcome, PlayerAction};
use sts_core::monsters::{Intent, MonsterId};
use sts_core::powers::PowerId;
use sts_core::rng::StsRandom;
use sts_core::rngs::Rngs;

fn combat(seed: i64, encounter: &str, deck: &[(CardId, i32)]) -> Combat {
    let mut cfg = CombatConfig::ironclad(seed, encounter);
    cfg.deck = deck.to_vec();
    Combat::new(&cfg).unwrap()
}

fn hand_index(c: &Combat, id: CardId) -> usize {
    c.hand.iter().position(|card| card.id == id).unwrap_or_else(|| panic!("{id:?} not in hand: {:?}", c.hand.iter().map(|c| c.id).collect::<Vec<_>>()))
}

fn play(c: &mut Combat, id: CardId, target: Option<usize>) {
    let i = hand_index(c, id);
    c.step(&PlayerAction::PlayCard { hand_index: i, target }).unwrap();
}

fn end_turn(c: &mut Combat) {
    c.step(&PlayerAction::EndTurn).unwrap();
}

fn m0(c: &Combat) -> usize {
    c.order[0]
}

#[test]
fn seeded_start_matches_rng_streams() {
    for seed in [0i64, 1, 42, -7, 123456789] {
        let c = combat(seed, "Jaw Worm", &CombatConfig::ironclad(0, "").deck);
        let mut r = Rngs::new(seed);
        r.enter_floor(1);
        let hp = r.monster_hp.random_range(40, 44);
        assert_eq!(c.monsters[0].hp, hp, "seed {seed}");
        // Jaw Worm always opens with Chomp; the roll still consumes aiRng.
        assert_eq!(c.monsters[0].next_move, 1);
        assert_eq!(c.monsters[0].intent, Intent::Attack);
        assert_eq!(c.hand.len(), 5);
        assert_eq!(c.draw_pile.len(), 5);
        assert_eq!(c.energy, 3);
    }
}

#[test]
fn same_seed_same_game() {
    let run = |seed: i64| {
        let mut c = combat(seed, "Gremlin Gang", &CombatConfig::ironclad(0, "").deck);
        let mut pick = StsRandom::new(99);
        let mut log = Vec::new();
        while !c.is_over() && c.turn < 50 {
            let legal = c.legal_actions();
            let a = legal[pick.random(legal.len() as i32 - 1) as usize].clone();
            c.step(&a).unwrap();
            log.push(serde_json::to_string(&c.observe()).unwrap());
        }
        log
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
}

#[test]
fn bash_then_strike_uses_vulnerable() {
    let mut c = combat(3, "Jaw Worm", &[(CardId::Bash, 0), (CardId::StrikeR, 0), (CardId::DefendR, 0), (CardId::DefendR, 0), (CardId::DefendR, 0)]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::Bash, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 8);
    assert_eq!(c.monsters[m].power_amount(PowerId::Vulnerable), 2);
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 8 - 9); // floor(6 * 1.5)
    assert_eq!(c.energy, 0);
}

#[test]
fn block_absorbs_jaw_worm_chomp_and_resets() {
    let mut c = combat(3, "Jaw Worm", &[(CardId::DefendR, 0); 5]);
    play(&mut c, CardId::DefendR, None);
    play(&mut c, CardId::DefendR, None);
    assert_eq!(c.player.block, 10);
    end_turn(&mut c);
    // Chomp deals 11: 10 blocked, 1 through. Block resets at the start of turn 2.
    assert_eq!(c.player.hp, 79);
    assert_eq!(c.player.block, 0);
    assert_eq!(c.turn, 2);
    assert_eq!(c.energy, 3);
    assert_eq!(c.hand.len(), 5);
}

#[test]
fn weak_reduces_monster_attack() {
    let mut c = combat(3, "Jaw Worm", &[(CardId::Clothesline, 0); 5]);
    let m = m0(&c);
    play(&mut c, CardId::Clothesline, Some(m));
    assert_eq!(c.monsters[m].damage[0].output, 8); // floor(11 * 0.75)
    end_turn(&mut c);
    assert_eq!(c.player.hp, 80 - 8);
    // Weak 2 applied by the player ticks down at end of round.
    assert_eq!(c.monsters[m].power_amount(PowerId::Weakened), 1);
}

#[test]
fn heavy_blade_and_perfected_strike_scaling() {
    let mut c = combat(4, "Jaw Worm", &[(CardId::Flex, 0), (CardId::HeavyBlade, 0), (CardId::StrikeR, 0), (CardId::StrikeR, 0), (CardId::StrikeR, 0)]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::Flex, None);
    play(&mut c, CardId::HeavyBlade, Some(m));
    assert_eq!(c.monsters[m].hp, hp - (14 + 2 * 3));

    // Perfected Strike counts itself (it is still in hand when damage is calculated).
    let mut c = combat(4, "Jaw Worm", &[(CardId::PerfectedStrike, 0), (CardId::StrikeR, 0), (CardId::StrikeR, 0), (CardId::DefendR, 0), (CardId::DefendR, 0)]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::PerfectedStrike, Some(m));
    assert_eq!(c.monsters[m].hp, hp - (6 + 2 * 3));
}

#[test]
fn body_slam_uses_current_block() {
    let mut c = combat(4, "Jaw Worm", &[(CardId::BodySlam, 0), (CardId::DefendR, 0), (CardId::DefendR, 0), (CardId::DefendR, 0), (CardId::DefendR, 0)]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::DefendR, None);
    play(&mut c, CardId::DefendR, None);
    play(&mut c, CardId::BodySlam, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 10);
}

#[test]
fn rampage_grows_each_play() {
    let mut c = combat(4, "Cultist", &[(CardId::Rampage, 0)]);
    let m = m0(&c);
    let hp0 = c.monsters[m].hp;
    play(&mut c, CardId::Rampage, Some(m));
    assert_eq!(c.monsters[m].hp, hp0 - 8);
    assert_eq!(c.discard_pile[0].base_damage, 13);
}

#[test]
fn double_tap_replays_attack() {
    let mut c = combat(4, "Cultist", &[(CardId::DoubleTap, 0), (CardId::StrikeR, 0)]);
    let m = m0(&c);
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::DoubleTap, None);
    play(&mut c, CardId::StrikeR, Some(m));
    assert_eq!(c.monsters[m].hp, hp - 12);
    assert!(!c.player.has_power(PowerId::DoubleTap));
    assert_eq!(c.discard_pile.len(), 2); // Double Tap + Strike; the copy is purged
}

#[test]
fn corruption_makes_skills_free_and_exhaust() {
    let mut c = combat(4, "Cultist", &[(CardId::Corruption, 0), (CardId::DefendR, 0), (CardId::ShrugItOff, 0), (CardId::DefendR, 0), (CardId::DefendR, 0)]);
    play(&mut c, CardId::Corruption, None);
    assert_eq!(c.energy, 0);
    assert!(c.hand.iter().all(|card| card.cost_for_turn == 0));
    play(&mut c, CardId::DefendR, None);
    assert_eq!(c.exhaust_pile.len(), 1);
    assert_eq!(c.player.block, 5);
}

#[test]
fn feel_no_pain_blocks_on_exhaust_choice() {
    let mut c = combat(9, "Cultist", &[(CardId::FeelNoPain, 0), (CardId::TrueGrit, 1), (CardId::DefendR, 0), (CardId::DefendR, 0), (CardId::Sentinel, 0)]);
    play(&mut c, CardId::FeelNoPain, None);
    play(&mut c, CardId::TrueGrit, None);
    // True Grit+ asks which card to exhaust: pick Sentinel (refunds 2 energy).
    let p = c.pending.as_ref().expect("selection pending");
    assert_eq!(p.count, 1);
    let sentinel = hand_index(&c, CardId::Sentinel);
    c.step(&PlayerAction::Choose { indices: vec![sentinel] }).unwrap();
    assert_eq!(c.player.block, 9 + 3);
    assert_eq!(c.energy, 3 - 1 - 1 + 2);
    assert_eq!(c.exhaust_pile.len(), 1);
}

#[test]
fn slime_splits_at_half_hp() {
    let mut c = combat(2, "Large Slime", &[(CardId::Bludgeon, 1)]);
    let m = m0(&c);
    c.energy = 3;
    let max = c.monsters[m].max_hp;
    play(&mut c, CardId::Bludgeon, Some(m));
    assert!(c.monsters[m].hp <= max / 2);
    assert_eq!(c.monsters[m].next_move, 3, "split interrupts the intent");
    end_turn(&mut c);
    let alive: Vec<_> = c.order.iter().filter(|&&i| !c.monsters[i].is_dead_or_escaped()).map(|&i| c.monsters[i].id).collect();
    assert_eq!(alive.len(), 2);
    assert!(alive.iter().all(|id| matches!(id, MonsterId::AcidSlimeM | MonsterId::SpikeSlimeM)));
    let hp_left = max - 42;
    for &i in &c.order {
        if !c.monsters[i].is_dead_or_escaped() {
            assert_eq!(c.monsters[i].hp, hp_left);
        }
    }
}

#[test]
fn lagavulin_wakes_when_damaged() {
    let mut c = combat(2, "Lagavulin", &[(CardId::Bludgeon, 0)]);
    let m = m0(&c);
    assert_eq!(c.monsters[m].block, 8);
    assert_eq!(c.monsters[m].power_amount(PowerId::Metallicize), 8);
    play(&mut c, CardId::Bludgeon, Some(m));
    assert_eq!(c.monsters[m].intent, Intent::Stun);
    assert!(!c.monsters[m].has_power(PowerId::Metallicize));
}

#[test]
fn sentries_alternate_first_moves() {
    let c = combat(2, "3 Sentries", &CombatConfig::ironclad(0, "").deck);
    let moves: Vec<u8> = c.order.iter().map(|&i| c.monsters[i].next_move).collect();
    assert_eq!(moves, vec![3, 4, 3]);
    assert!(c.order.iter().all(|&i| c.monsters[i].power_amount(PowerId::Artifact) == 1));
}

#[test]
fn guardian_shifts_to_defensive_mode() {
    let mut c = combat(2, "The Guardian", &[(CardId::Bludgeon, 0)]);
    let m = m0(&c);
    assert_eq!(c.monsters[m].power_amount(PowerId::ModeShift), 30);
    play(&mut c, CardId::Bludgeon, Some(m));
    // 32 damage >= 30 threshold: Defensive Mode adds 20 block and Close Up next.
    assert_eq!(c.monsters[m].block, 20);
    assert_eq!(c.monsters[m].next_move, 1);
    assert!(!c.monsters[m].has_power(PowerId::ModeShift));
}

#[test]
fn gremlin_nob_enrages_on_skills() {
    let mut c = combat(2, "Gremlin Nob", &[(CardId::DefendR, 0); 5]);
    let m = m0(&c);
    end_turn(&mut c); // Nob bellows
    assert_eq!(c.monsters[m].power_amount(PowerId::Anger), 2);
    play(&mut c, CardId::DefendR, None);
    assert_eq!(c.monsters[m].power_amount(PowerId::Strength), 2);
}

#[test]
fn louse_curls_up_once() {
    let mut c = combat(8, "2 Louse", &[(CardId::StrikeR, 0); 5]);
    let m = m0(&c);
    let curl = c.monsters[m].power_amount(PowerId::CurlUp);
    assert!((3..=7).contains(&curl));
    let hp = c.monsters[m].hp;
    play(&mut c, CardId::StrikeR, Some(m));
    if hp > 6 {
        assert_eq!(c.monsters[m].block, curl);
        assert!(!c.monsters[m].has_power(PowerId::CurlUp));
    }
}

#[test]
fn burn_hurts_at_end_of_turn() {
    let mut c = combat(5, "Cultist", &[(CardId::Burn, 0); 5]);
    end_turn(&mut c);
    // 5 Burns x 2 damage, then the Cultist buffs on turn 1.
    assert_eq!(c.player.hp, 80 - 10);
}

#[test]
fn victory_heals_with_burning_blood() {
    let mut c = combat(5, "Cultist", &[(CardId::Bludgeon, 2); 5]);
    c.player.hp = 50;
    c.energy = 9;
    let m = m0(&c);
    while !c.is_over() {
        if c.hand.iter().any(|h| h.id == CardId::Bludgeon) && c.energy >= 3 {
            play(&mut c, CardId::Bludgeon, Some(m));
        } else {
            end_turn(&mut c);
        }
    }
    assert_eq!(c.outcome, Some(Outcome::Victory));
    assert_eq!(c.player.hp, 56);
    let _ = Cr::Player;
}
