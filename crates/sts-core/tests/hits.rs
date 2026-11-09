//! `Combat::card_hits`: how one play of an attack hits, for the network's inputs. Each count is
//! checked twice: against the number expected from the card's `use` (`cards_use.rs`, the
//! decompiled card), and by playing the card at monsters with 999 HP and dividing the HP they
//! lost by one hit's damage.

use sts_core::cards::{CardId, CardTarget};
use sts_core::character::Character;
use sts_core::combat::{Combat, CombatConfig, Hits, Orb, OrbId, PlayerAction};
use sts_core::powers::{Power, PowerId};
use sts_core::relics::{RelicId, RelicState};

/// A fight against `encounter` whose hand is exactly `hand`, with no relics, 10 energy and every
/// monster at 999 HP with no block.
fn fight(ch: Character, encounter: &str, hand: &[CardId]) -> Combat {
    let mut cfg = CombatConfig::new(ch, 3, encounter);
    cfg.relics.clear();
    let mut c = Combat::new(&cfg).unwrap();
    c.hand.clear();
    for &id in hand {
        let card = c.new_card(id);
        c.hand.push(card);
    }
    for m in c.order.clone() {
        c.monsters[m].hp = 999;
        c.monsters[m].max_hp = 999;
        c.monsters[m].block = 0;
    }
    c.energy = 10;
    c
}

fn alive(c: &Combat) -> Vec<usize> {
    c.order.iter().copied().filter(|&m| !c.monsters[m].is_dead_or_escaped()).collect()
}

/// Plays hand card `h` (at the first monster if it takes a target) and returns how many hits
/// landed: the HP the monsters lost over one hit's damage (on each of them, for a card that hits
/// every enemy).
fn hits_landed(c: &mut Combat, h: usize) -> i32 {
    let card = c.hand[h].clone();
    let first = alive(c)[0];
    let target = (card.target() == CardTarget::Enemy).then_some(first);
    let per_hit = c.card_damage_against(&card, first);
    let spread = if card.data().multi_damage { alive(c).len() as i32 } else { 1 };
    let hp = |c: &Combat| c.order.iter().map(|&m| c.monsters[m].hp).sum::<i32>();
    let before = hp(c);
    c.step(&PlayerAction::PlayCard { hand_index: h, target }).unwrap();
    let lost = before - hp(c);
    assert_eq!(lost % (per_hit * spread), 0, "{:?}: lost {lost}, {per_hit} a hit", card.id);
    lost / (per_hit * spread)
}

/// `card_hits` of the hand's first card at the first monster, then the hits it lands when played.
fn check(ch: Character, encounter: &str, hand: &[CardId], setup: impl Fn(&mut Combat), want: Hits) {
    let mut c = fight(ch, encounter, hand);
    setup(&mut c);
    let first = alive(&c)[0];
    let got = c.card_hits(&c.hand[0].clone(), Some(first));
    assert_eq!(got, want, "{:?}", hand[0]);
    assert_eq!(hits_landed(&mut c, 0), want.count, "{:?} played", hand[0]);
}

const ONE: Hits = Hits { count: 1, all: false, random: false };
const fn hits(count: i32) -> Hits {
    Hits { count, all: false, random: false }
}
const fn all(count: i32) -> Hits {
    Hits { count, all: true, random: false }
}
const fn random(count: i32) -> Hits {
    Hits { count, all: false, random: true }
}

fn upgrade_first(c: &mut Combat) {
    c.hand[0].upgrade();
}

#[test]
fn fixed_hits() {
    use CardId::*;
    let none = |_: &mut Combat| {};
    check(Character::Ironclad, "Jaw Worm", &[StrikeR], none, ONE);
    check(Character::Ironclad, "Jaw Worm", &[TwinStrike], none, hits(2));
    check(Character::Silent, "Jaw Worm", &[GlassKnife], none, hits(2));
    check(Character::Silent, "Jaw Worm", &[Eviscerate], none, hits(3));
    check(Character::Silent, "Jaw Worm", &[RiddleWithHoles], none, hits(5));
    check(Character::Watcher, "Jaw Worm", &[FlyingSleeves], none, hits(2));
    check(Character::Silent, "Small Slimes", &[DaggerSpray], none, all(2));
    check(Character::Ironclad, "Small Slimes", &[Cleave], none, all(1));
}

#[test]
fn hits_from_the_magic_number() {
    use CardId::*;
    let none = |_: &mut Combat| {};
    check(Character::Ironclad, "Jaw Worm", &[Pummel], none, hits(4));
    check(Character::Ironclad, "Jaw Worm", &[Pummel], upgrade_first, hits(5));
    check(Character::Watcher, "Jaw Worm", &[Tantrum], none, hits(3));
    check(Character::Watcher, "Jaw Worm", &[Tantrum], upgrade_first, hits(4));
    check(Character::Watcher, "Jaw Worm", &[Expunger], |c| (c.hand[0].magic = 3, c.hand[0].base_magic = 3).0, hits(3));
    // Random targets: the hits land on the monsters at random, all of them counted.
    check(Character::Ironclad, "Small Slimes", &[SwordBoomerang], none, random(3));
    check(Character::Ironclad, "Small Slimes", &[SwordBoomerang], upgrade_first, random(4));
    check(Character::Defect, "Small Slimes", &[RipAndTear], none, random(2));
    check(Character::Watcher, "Small Slimes", &[Ragnarok], none, random(5));
}

#[test]
fn x_cost_attacks_hit_once_per_energy() {
    use CardId::*;
    let energy = |n: i32| move |c: &mut Combat| c.energy = n;
    check(Character::Ironclad, "Small Slimes", &[Whirlwind], energy(3), all(3));
    check(Character::Silent, "Jaw Worm", &[Skewer], energy(2), hits(2));
    // Chemical X adds two.
    let chem = |c: &mut Combat| {
        c.energy = 3;
        c.player.relics.push(RelicState::new(RelicId::ChemicalX));
    };
    check(Character::Ironclad, "Small Slimes", &[Whirlwind], chem, all(5));
    let mut c = fight(Character::Silent, "Jaw Worm", &[Malaise, StrikeG]);
    c.energy = 2;
    assert_eq!((c.x_energy(&c.hand[0]), c.x_energy(&c.hand[1])), (2, 0));
    c.player.relics.push(RelicState::new(RelicId::ChemicalX));
    assert_eq!(c.x_energy(&c.hand[0]), 4);
}

#[test]
fn hits_from_the_fight() {
    use CardId::*;
    // Finisher: once per attack played before it this turn.
    let mut c = fight(Character::Silent, "Jaw Worm", &[StrikeG, StrikeG, Finisher]);
    let m = c.order[0];
    assert_eq!(c.card_hits(&c.hand[2].clone(), Some(m)), hits(0));
    for _ in 0..2 {
        c.step(&PlayerAction::PlayCard { hand_index: 0, target: Some(m) }).unwrap();
    }
    assert_eq!(c.card_hits(&c.hand[0].clone(), Some(m)), hits(2));
    assert_eq!(hits_landed(&mut c, 0), 2);
    // Flechettes: once per skill in hand.
    check(Character::Silent, "Jaw Worm", &[Flechettes, DefendG, DefendG, StrikeG], |_| {}, hits(2));
    // Barrage: once per orb channeled (filled slots).
    let orbs = |c: &mut Combat| c.player.orbs = vec![Some(Orb::new(OrbId::Lightning)), Some(Orb::new(OrbId::Frost)), None];
    check(Character::Defect, "Jaw Worm", &[Barrage], orbs, hits(2));
    // Thunder Strike: once per Lightning channeled this combat, at random.
    let channeled = |c: &mut Combat| c.orbs_channeled_this_combat = vec![OrbId::Lightning, OrbId::Frost, OrbId::Lightning];
    check(Character::Defect, "Small Slimes", &[ThunderStrike], channeled, random(2));
    // Bowling Bash: once per monster alive.
    check(Character::Watcher, "Small Slimes", &[BowlingBash], |_| {}, hits(2));
    check(Character::Watcher, "Jaw Worm", &[BowlingBash], |_| {}, hits(1));
    // Fiend Fire: once per other card in hand (each exhausted).
    check(Character::Ironclad, "Jaw Worm", &[FiendFire, DefendR, DefendR, StrikeR, Bash], |_| {}, hits(4));
}

#[test]
fn bane_hits_a_poisoned_monster_twice() {
    use CardId::*;
    check(Character::Silent, "Jaw Worm", &[Bane], |_| {}, hits(1));
    let poison = |c: &mut Combat| {
        let m = c.order[0];
        c.monsters[m].powers.push(Power::new(PowerId::Poison, 5));
    };
    check(Character::Silent, "Jaw Worm", &[Bane], poison, hits(2));
    // Not aimed at anyone, it counts the one sure hit.
    let mut c = fight(Character::Silent, "Jaw Worm", &[Bane]);
    poison(&mut c);
    assert_eq!(c.card_hits(&c.hand[0].clone(), None), hits(1));
}

#[test]
fn cards_that_are_not_attacks_have_no_hits() {
    use CardId::*;
    let c = fight(Character::Ironclad, "Jaw Worm", &[DefendR, Inflame, Wound]);
    for card in &c.hand {
        assert_eq!(c.card_hits(card, Some(c.order[0])), Hits::default(), "{:?}", card.id);
    }
}
