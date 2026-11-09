//! Relic effects outside combat, checked against what the real game did.

use sts_core::cards::{CardId, CardRarity};
use sts_core::relics::RelicId;
use sts_core::run::{Run, RunConfig};

/// Pandora's Box rolls one card per Strike and Defend, but the game's confirmation grid fills
/// with `addToBottom`, so the deck gets them last roll first (seen in a real-game playback of
/// seed 6ZWK, where the engine's order shuffled into different hands).
#[test]
fn pandoras_box_adds_cards_last_roll_first() {
    let mut run = Run::new(RunConfig::new(300005));
    let kept: Vec<CardId> =
        run.deck.iter().map(|c| c.id).filter(|id| !matches!(id, CardId::StrikeR | CardId::DefendR)).collect();
    let replaced = run.deck.len() - kept.len();
    assert!(replaced > 0);

    let mut list = Vec::new();
    for r in [CardRarity::Common, CardRarity::Uncommon, CardRarity::Rare] {
        list.extend(run.pools.src_pool(r));
    }
    let mut rng = run.rng.card_random.clone();
    let rolls: Vec<CardId> = (0..replaced).map(|_| list[rng.random(list.len() as i32 - 1) as usize]).collect();

    run.obtain_relic(RelicId::PandorasBox);
    let deck: Vec<CardId> = run.deck.iter().map(|c| c.id).collect();
    let expected: Vec<CardId> = kept.into_iter().chain(rolls.into_iter().rev()).collect();
    assert_eq!(deck, expected);
}

/// Prismatic Shard: card rewards from every color (`getAnyColorCard`, cardRng), and one orb
/// slot for anyone but the Defect (`onEquip` sets `masterMaxOrbs` to 1).
#[test]
fn prismatic_shard_rewards_any_color_and_gives_an_orb_slot() {
    use sts_core::cards::CardColor;
    use sts_core::character::Character;
    use sts_core::combat::{Combat, CombatConfig};
    let mut run = Run::new(RunConfig::new(3));
    run.obtain_relic(RelicId::PrismaticShard);
    let mut colors = std::collections::HashSet::new();
    for _ in 0..40 {
        for c in run.reward_cards(sts_core::run::pools::RewardRoom::Normal) {
            colors.insert(c.data().color);
        }
    }
    assert!(colors.len() >= 4, "{colors:?}");
    assert!(colors.contains(&CardColor::Purple) && colors.contains(&CardColor::Blue));

    let mut cfg = CombatConfig::new(Character::Watcher, 1, "Jaw Worm");
    cfg.relics.push(RelicId::PrismaticShard);
    assert_eq!(Combat::new(&cfg).unwrap().player.orbs.len(), 1);
    let mut cfg = CombatConfig::new(Character::Defect, 1, "Jaw Worm");
    cfg.relics.push(RelicId::PrismaticShard);
    assert_eq!(Combat::new(&cfg).unwrap().player.orbs.len(), 3);
    assert!(Combat::new(&CombatConfig::ironclad(1, "Jaw Worm")).unwrap().player.orbs.is_empty());
}
