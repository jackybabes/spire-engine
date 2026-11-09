//! Relic effects outside combat, one test per behaviour, checked against the decompiled game's
//! relic classes (`onEquip`, `onObtainCard`, `onEnterRoom`, `onChestOpen`, campfire options...)
//! and the rooms and screens that call them. The combat hooks are tested in `tests/relics.rs`.

use super::*;
use crate::cards::CardType;

fn run() -> Run {
    Run::new(RunConfig::new(1))
}

fn relic(r: &Run, id: RelicId) -> &RelicState {
    r.relics.iter().find(|x| x.id == id).unwrap()
}

fn count(r: &Run, id: CardId) -> usize {
    r.deck.iter().filter(|c| c.id == id).count()
}

fn rewards(r: &Run) -> Vec<Reward> {
    match &r.screen {
        Screen::Rewards { items } => items.clone(),
        s => panic!("not a reward screen: {}", screen_name(s)),
    }
}

fn relic_rewards(items: &[Reward]) -> usize {
    items.iter().filter(|i| matches!(i, Reward::Relic(_))).count()
}

/// Starts the fight `key` in a room of type `room` and wins it at once with a Strike.
fn win_fight(r: &mut Run, key: &str, room: RoomType) -> Vec<Reward> {
    r.start_fight(key, room);
    let c = r.combat.as_mut().unwrap();
    let m = c.order[0];
    c.monsters[m].hp = 1;
    let card = c.new_card(CardId::StrikeR);
    c.hand.push(card);
    let i = c.hand.len() - 1;
    r.step(&RunAction::Combat { action: PlayerAction::PlayCard { hand_index: i, target: Some(m) } }).unwrap();
    rewards(r)
}

// ---- onEquip ----

#[test]
fn max_hp_gold_and_potion_slot_relics() {
    let mut r = run();
    r.hp = 50;
    r.obtain_relic(RelicId::Mango);
    assert_eq!((r.hp, r.max_hp), (64, 94));
    r.obtain_relic(RelicId::Pear);
    assert_eq!((r.hp, r.max_hp), (74, 104));
    r.obtain_relic(RelicId::Strawberry);
    assert_eq!((r.hp, r.max_hp), (81, 111));
    // Lee's Waffle: 7 max HP, then a full heal.
    r.hp = 20;
    r.obtain_relic(RelicId::Waffle);
    assert_eq!((r.hp, r.max_hp), (118, 118));
    let gold = r.gold;
    r.obtain_relic(RelicId::OldCoin);
    assert_eq!(r.gold, gold + 300);
    let slots = r.potions.len();
    r.obtain_relic(RelicId::PotionBelt);
    assert_eq!(r.potions.len(), slots + 2);
    // Neow's Lament comes with three charges.
    r.obtain_relic(RelicId::NeowsLament);
    assert_eq!(relic(&r, RelicId::NeowsLament).counter, 3);
}

#[test]
fn starter_upgrades_take_the_starters_place_and_circlets_stack() {
    let mut r = run();
    r.obtain_relic(RelicId::Anchor);
    r.obtain_relic(RelicId::BlackBlood);
    assert_eq!(r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), [RelicId::BlackBlood, RelicId::Anchor]);
    r.obtain_relic(RelicId::Circlet);
    r.obtain_relic(RelicId::Circlet);
    assert_eq!(r.relics.iter().filter(|x| x.id == RelicId::Circlet).count(), 1);
    assert_eq!(relic(&r, RelicId::Circlet).counter, 2);
}

#[test]
fn upgrade_relics_on_equip() {
    let upgraded = |r: &Run, t: Option<CardType>| r.deck.iter().filter(|c| c.upgraded && t.map_or(true, |t| c.card_type() == t)).count();
    // War Paint: two random Skills; Whetstone: two random Attacks (one miscRng long each).
    let mut r = run();
    let misc = r.rng.misc.counter;
    r.obtain_relic(RelicId::WarPaint);
    assert_eq!((upgraded(&r, Some(CardType::Skill)), upgraded(&r, None)), (2, 2));
    r.obtain_relic(RelicId::Whetstone);
    assert_eq!((upgraded(&r, Some(CardType::Attack)), upgraded(&r, None)), (2, 4));
    assert_eq!(r.rng.misc.counter, misc + 2);
    // Which ones: `Collections.shuffle(upgradable, new Random(miscRng.randomLong()))`, first two.
    let mut r = run();
    let mut misc = r.rng.misc.clone();
    let mut skills: Vec<usize> = (0..r.deck.len()).filter(|&i| r.deck[i].card_type() == CardType::Skill).collect();
    java_shuffle(&mut skills, &mut JavaRandom::new(misc.random_long()));
    r.obtain_relic(RelicId::WarPaint);
    let mut got: Vec<usize> = (0..r.deck.len()).filter(|&i| r.deck[i].upgraded).collect();
    let mut want = skills[..2].to_vec();
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want);

    // Tiny House: one random upgrade, 5 max HP, and 50 gold, a potion and a card reward.
    let mut r = run();
    r.obtain_relic(RelicId::TinyHouse);
    assert_eq!(upgraded(&r, None), 1);
    assert_eq!(r.max_hp, 85);
    let items = rewards(&r);
    assert!(matches!(&items[..], [Reward::Gold(50), Reward::Potion(_), Reward::Cards(cs)] if cs.len() == 3));
}

#[test]
fn deck_choice_relics() {
    // Astrolabe: transform three cards; each comes back upgraded.
    let mut r = run();
    let n = r.deck.len();
    r.obtain_relic(RelicId::Astrolabe);
    assert!(matches!(r.screen, Screen::Grid { purpose: GridPurpose::Astrolabe, count: 3, .. }));
    r.step(&RunAction::Select { indices: vec![0, 1, 2] }).unwrap();
    assert_eq!(r.deck.len(), n);
    assert_eq!(r.deck[n - 3..].iter().filter(|c| c.upgraded || !c.can_upgrade()).count(), 3);
    assert!(matches!(r.screen, Screen::Neow { .. }));
    // Empty Cage: remove two cards.
    let mut r = run();
    r.obtain_relic(RelicId::EmptyCage);
    assert!(matches!(r.screen, Screen::Grid { purpose: GridPurpose::Remove, count: 2, .. }));
    r.step(&RunAction::Select { indices: vec![0, 1] }).unwrap();
    assert_eq!(r.deck.len(), n - 2);
    // Dolly's Mirror: a copy of a card, upgrades included.
    let mut r = run();
    let bash = r.deck.iter().position(|c| c.id == CardId::Bash).unwrap();
    r.deck[bash].upgrade();
    r.obtain_relic(RelicId::DollysMirror);
    r.step(&RunAction::Select { indices: vec![bash] }).unwrap();
    assert_eq!(count(&r, CardId::Bash), 2);
    assert!(r.deck.last().unwrap().upgraded);
}

#[test]
fn bottled_cards_start_in_the_opening_hand() {
    use RelicId::*;
    // (relic, the card type it bottles)
    for (b, t) in [(BottledFlame, CardType::Attack), (BottledLightning, CardType::Skill), (BottledTornado, CardType::Power)] {
        let mut r = run();
        let inflame = r.new_card(CardId::Inflame);
        r.deck.push(inflame);
        r.obtain_relic(b);
        let Screen::Grid { purpose: GridPurpose::Bottle(_), options, count: 1 } = r.screen.clone() else { panic!("{b:?}'s grid") };
        assert!(options.iter().all(|&i| r.deck[i].card_type() == t), "{b:?}");
        let pick = *options.last().unwrap();
        let uuid = r.deck[pick].uuid;
        r.step(&RunAction::Select { indices: vec![pick] }).unwrap();
        assert!(r.deck[pick].bottled);
        assert_eq!(relic(&r, b).card_uuid, Some(uuid));
        for seed in 0..5 {
            let mut f = r.clone();
            f.rng = crate::rngs::Rngs::new(seed);
            f.start_fight("Jaw Worm", RoomType::Monster);
            assert!(f.combat.as_ref().unwrap().hand.iter().any(|c| c.uuid == uuid), "{b:?} seed {seed}");
        }
    }
    // No card of the type: nothing to choose.
    let mut r = run();
    r.obtain_relic(BottledTornado);
    assert!(matches!(r.screen, Screen::Neow { .. }));
}

#[test]
fn reward_screen_relics() {
    // Cauldron: five potions.
    let mut r = run();
    r.obtain_relic(RelicId::Cauldron);
    let items = rewards(&r);
    assert!(items.len() == 5 && items.iter().all(|i| matches!(i, Reward::Potion(_))));
    // Orrery: five card rewards.
    let mut r = run();
    r.obtain_relic(RelicId::Orrery);
    let items = rewards(&r);
    assert!(items.len() == 5 && items.iter().all(|i| matches!(i, Reward::Cards(cs) if cs.len() == 3)));
    // Calling Bell: Curse of the Bell, and a common, an uncommon and a rare relic.
    let mut r = run();
    r.obtain_relic(RelicId::CallingBell);
    assert_eq!(count(&r, CardId::CurseOfTheBell), 1);
    let tiers: Vec<RelicTier> = rewards(&r)
        .iter()
        .map(|i| match i {
            Reward::Relic(x) => x.data().tier,
            _ => panic!("{i:?}"),
        })
        .collect();
    assert_eq!(tiers, [RelicTier::Common, RelicTier::Uncommon, RelicTier::Rare]);
}

// ---- Cards ----

#[test]
fn card_obtain_relics() {
    let mut r = run();
    r.obtain_relic(RelicId::CeramicFish);
    r.obtain_relic(RelicId::DarkstonePeriapt);
    r.obtain_relic(RelicId::DuVuDoll);
    assert_eq!(relic(&r, RelicId::DuVuDoll).counter, 0);
    let gold = r.gold;
    let c = r.new_card(CardId::Anger);
    r.obtain_card(c);
    assert_eq!((r.gold, r.max_hp), (gold + 9, 80));
    let c = r.new_card(CardId::Regret);
    r.obtain_card(c);
    assert_eq!((r.gold, r.max_hp), (gold + 18, 86));
    assert_eq!(relic(&r, RelicId::DuVuDoll).counter, 1);
    let i = r.deck.len() - 1;
    r.take_cards(&[i]);
    r.on_master_deck_change();
    assert_eq!(relic(&r, RelicId::DuVuDoll).counter, 0);
    // Omamori: the next two curses never arrive.
    let mut r = run();
    r.obtain_relic(RelicId::Omamori);
    for _ in 0..3 {
        let c = r.new_card(CardId::Regret);
        r.obtain_card(c);
    }
    assert_eq!(count(&r, CardId::Regret), 1);
    assert!(relic(&r, RelicId::Omamori).used_up);
}

#[test]
fn eggs_upgrade_their_card_type() {
    use RelicId::*;
    for (egg, card) in [(MoltenEgg2, CardId::Anger), (ToxicEgg2, CardId::ShrugItOff), (FrozenEgg2, CardId::Inflame)] {
        let mut r = run();
        // Cards already on the reward screen are upgraded too.
        let shown = r.new_card(card);
        r.screen = Screen::Rewards { items: vec![Reward::Cards(vec![shown])] };
        r.obtain_relic(egg);
        assert!(matches!(&rewards(&r)[0], Reward::Cards(cs) if cs[0].upgraded), "{egg:?}");
        let c = r.new_card(card);
        r.obtain_card(c);
        assert!(r.deck.last().unwrap().upgraded, "{egg:?}");
        let other = if card == CardId::Anger { CardId::ShrugItOff } else { CardId::Anger };
        let c = r.new_card(other);
        r.obtain_card(c);
        assert!(!r.deck.last().unwrap().upgraded, "{egg:?}");
        // And new card rewards show them upgraded.
        for _ in 0..20 {
            for c in r.reward_cards(RewardRoom::Normal) {
                if c.id == card {
                    assert!(c.upgraded);
                }
            }
        }
    }
}

#[test]
fn card_reward_relics() {
    let size = |relics: &[RelicId]| {
        let mut r = run();
        for &id in relics {
            r.obtain_relic(id);
        }
        r.reward_cards(RewardRoom::Normal).len()
    };
    assert_eq!(size(&[]), 3);
    assert_eq!(size(&[RelicId::QuestionCard]), 4);
    assert_eq!(size(&[RelicId::BustedCrown]), 1);
    assert_eq!(size(&[RelicId::QuestionCard, RelicId::BustedCrown]), 2);
    // N'loth's Gift: three times the rare chance (3% -> 9% in a normal fight).
    for gift in [false, true] {
        let mut r = run();
        if gift {
            r.obtain_relic(RelicId::NlothsGift);
        }
        let roll = r.rng.card.clone().random(99);
        r.card_blizz = 5 - roll;
        let rarity = r.roll_card_rarity(RewardRoom::Normal);
        assert_eq!(rarity, if gift { CardRarity::Rare } else { CardRarity::Uncommon });
    }
    // Singing Bowl: 2 max HP instead of a card.
    let mut r = run();
    r.obtain_relic(RelicId::SingingBowl);
    let cards = r.reward_cards(RewardRoom::Normal);
    r.screen = Screen::Rewards { items: vec![Reward::Cards(cards)] };
    r.step(&RunAction::TakeReward { index: 0 }).unwrap();
    assert!(r.legal_actions().contains(&RunAction::SingingBowl));
    r.step(&RunAction::SingingBowl).unwrap();
    assert_eq!(r.max_hp, 82);
    assert!(rewards(&r).is_empty());
}

#[test]
fn combat_reward_relics() {
    // Black Star: a second relic from elites.
    let mut r = run();
    assert_eq!(relic_rewards(&win_fight(&mut r, "Gremlin Nob", RoomType::Elite)), 1);
    let mut r = run();
    r.obtain_relic(RelicId::BlackStar);
    let items = win_fight(&mut r, "Gremlin Nob", RoomType::Elite);
    assert_eq!(relic_rewards(&items), 2);
    // Prayer Wheel: a second card reward from normal fights.
    let cards = |items: &[Reward]| items.iter().filter(|i| matches!(i, Reward::Cards(_))).count();
    let mut r = run();
    r.obtain_relic(RelicId::PrayerWheel);
    assert_eq!(cards(&win_fight(&mut r, "Jaw Worm", RoomType::Monster)), 2);
    assert_eq!(cards(&win_fight(&mut r, "Gremlin Nob", RoomType::Elite)), 1);
    // White Beast Statue: a potion every fight.
    let mut r = run();
    r.obtain_relic(RelicId::WhiteBeast);
    for _ in 0..5 {
        let items = win_fight(&mut r, "Jaw Worm", RoomType::Monster);
        assert!(items.iter().any(|i| matches!(i, Reward::Potion(_))));
        r.potions = vec![None; 3];
    }
}

// ---- Rooms ----

#[test]
fn room_entry_relics() {
    let mut r = run();
    for id in [RelicId::MawBank, RelicId::SsserpentHead, RelicId::MealTicket, RelicId::EternalFeather] {
        r.obtain_relic(id);
    }
    r.hp = 40;
    // Maw Bank: 12 gold per room; Ssserpent Head: 50 more in ? rooms.
    let gold = r.gold;
    r.relics_on_enter_room(RoomType::Monster);
    assert_eq!(r.gold, gold + 12);
    r.relics_on_enter_room(RoomType::Event);
    assert_eq!(r.gold, gold + 12 + 62);
    // Eternal Feather: 3 HP per 5 cards in the deck when entering a rest site (10 cards: 6).
    r.relics_on_enter_room(RoomType::Rest);
    assert_eq!(r.hp, 46);
    // Meal Ticket: 15 HP entering a shop.
    r.relics_just_entered_room(RoomType::Shop);
    assert_eq!(r.hp, 61);
    // Maw Bank stops once gold is spent.
    r.lose_gold(1);
    assert!(relic(&r, RelicId::MawBank).used_up);
    let gold = r.gold;
    r.relics_on_enter_room(RoomType::Monster);
    assert_eq!(r.gold, gold);
    // On the map: entering the first room pays Maw Bank.
    let mut r = run();
    r.obtain_relic(RelicId::MawBank);
    let gold = r.gold;
    let x = r.next_nodes()[0];
    r.enter_node(x);
    assert_eq!(r.gold, gold + 12);
}

#[test]
fn gold_relics() {
    let mut r = run();
    r.hp = 50;
    let gold = r.gold;
    r.obtain_relic(RelicId::BloodyIdol);
    r.gain_gold(10);
    assert_eq!((r.gold, r.hp), (gold + 10, 55));
    // Ectoplasm: no more gold.
    r.obtain_relic(RelicId::Ectoplasm);
    r.gain_gold(10);
    r.obtain_relic(RelicId::OldCoin);
    assert_eq!((r.gold, r.hp), (gold + 10, 55));
    // Golden Idol: 25% more gold from rewards (rounded), except in treasure rooms.
    let mut r = run();
    r.obtain_relic(RelicId::GoldenIdol);
    r.room = Some(RoomType::Monster);
    assert_eq!(r.gold_with_bonus(15), 19);
    r.room = Some(RoomType::Treasure);
    r.pos = Some((3, 8));
    assert_eq!(r.gold_with_bonus(15), 15);
    // The boss chest isn't a TreasureRoom: Tiny House's 50 gold gets the bonus.
    r.pos = Some((3, 15));
    assert_eq!(r.gold_with_bonus(50), 63);
}

#[test]
fn healing_relics_outside_combat() {
    let mut r = run();
    r.hp = 40;
    r.obtain_relic(RelicId::MarkOfTheBloom);
    r.heal(20);
    r.obtain_relic(RelicId::Waffle);
    assert_eq!(r.hp, 40);
    // Sacred Bark doubles potions outside combat too; Toy Ornithopter heals 5 per potion.
    let mut r = run();
    r.obtain_relic(RelicId::SacredBark);
    r.obtain_relic(RelicId::ToyOrnithopter);
    r.step(&RunAction::Neow { option: 1 }).unwrap();
    r.hp = 20;
    r.potions = vec![Some(PotionId::BloodPotion), Some(PotionId::FruitJuice), None];
    let max = r.max_hp;
    r.step(&RunAction::UsePotion { slot: 0 }).unwrap();
    assert_eq!(r.hp, 20 + (max as f32 * 0.4) as i32 + 5);
    let hp = r.hp;
    r.step(&RunAction::UsePotion { slot: 1 }).unwrap();
    assert_eq!((r.hp, r.max_hp), (hp + 10 + 5, max + 10));
}

#[test]
fn rest_site_relics() {
    let options = |relics: &[RelicId]| {
        let mut r = run();
        for &id in relics {
            r.obtain_relic(id);
        }
        r.campfire_options()
    };
    assert!(!options(&[RelicId::CoffeeDripper]).contains(&RunAction::Rest));
    assert!(!options(&[RelicId::FusionHammer]).contains(&RunAction::Smith));
    let o = options(&[RelicId::Girya, RelicId::PeacePipe, RelicId::Shovel]);
    for a in [RunAction::Rest, RunAction::Smith, RunAction::Lift, RunAction::Toke, RunAction::Dig] {
        assert!(o.contains(&a), "{a:?}");
    }
    // Girya: three lifts.
    let mut r = run();
    r.obtain_relic(RelicId::Girya);
    for _ in 0..3 {
        r.screen = Screen::Rest;
        r.step(&RunAction::Lift).unwrap();
    }
    assert_eq!(relic(&r, RelicId::Girya).counter, 3);
    assert!(!r.campfire_options().contains(&RunAction::Lift));
    // Peace Pipe: remove a card.
    let mut r = run();
    r.obtain_relic(RelicId::PeacePipe);
    r.screen = Screen::Rest;
    let n = r.deck.len();
    r.step(&RunAction::Toke).unwrap();
    // No cancel: the game's reopens the campfire, and the toke can't be left without a removal here.
    assert!(!r.legal_actions().contains(&RunAction::Proceed));
    assert!(r.step(&RunAction::Proceed).is_err());
    r.step(&RunAction::Select { indices: vec![0] }).unwrap();
    assert_eq!(r.deck.len(), n - 1);
    // Shovel: dig up a relic.
    let mut r = run();
    r.obtain_relic(RelicId::Shovel);
    r.screen = Screen::Rest;
    r.step(&RunAction::Dig).unwrap();
    assert!(matches!(&rewards(&r)[..], [Reward::Relic(_)]));
    // Regal Pillow: 15 more HP from resting; Dream Catcher: a card reward after.
    let mut r = run();
    r.obtain_relic(RelicId::RegalPillow);
    r.obtain_relic(RelicId::DreamCatcher);
    r.hp = 20;
    r.screen = Screen::Rest;
    r.step(&RunAction::Rest).unwrap();
    assert_eq!(r.hp, 20 + 24 + 15);
    assert!(matches!(&r.screen, Screen::CardReward { item: None, cards } if cards.len() == 3));
    // Ancient Tea Set: entering a rest site charges it for the next fight.
    let mut r = run();
    r.obtain_relic(RelicId::AncientTeaSet);
    r.relics_on_enter_rest_room();
    assert_eq!(relic(&r, RelicId::AncientTeaSet).counter, -2);
}

#[test]
fn chest_relics() {
    // Cursed Key: a curse with every chest.
    let mut r = run();
    r.obtain_relic(RelicId::CursedKey);
    let n = r.deck.len();
    r.open_chest();
    assert_eq!(r.deck.len(), n + 1);
    assert_eq!(r.deck.last().unwrap().card_type(), CardType::Curse);
    // Matryoshka: a second relic from the next two chests.
    let mut r = run();
    r.obtain_relic(RelicId::Matryoshka);
    for left in [1, -2] {
        r.open_chest();
        assert_eq!(relic_rewards(&rewards(&r)), 2);
        assert_eq!(relic(&r, RelicId::Matryoshka).counter, left);
    }
    r.open_chest();
    assert_eq!(relic_rewards(&rewards(&r)), 1);
    // N'loth's Hungry Face: the next chest is empty (its relic and the sapphire key go).
    let mut r = run();
    r.obtain_relic(RelicId::NlothsMask);
    r.open_chest();
    assert!(!rewards(&r).iter().any(|i| matches!(i, Reward::Relic(_) | Reward::SapphireKey)));
    assert!(relic(&r, RelicId::NlothsMask).used_up);
    r.open_chest();
    let items = rewards(&r);
    assert_eq!(relic_rewards(&items), 1);
    assert!(items.contains(&Reward::SapphireKey));
}

#[test]
fn map_relics() {
    // Juzu Bracelet: a ? room never turns into a fight.
    for juzu in [false, true] {
        let mut r = run();
        if juzu {
            r.obtain_relic(RelicId::JuzuBracelet);
        }
        r.event_chances = [0.0, 1.0, 0.0, 0.0];
        let room = r.roll_event_room(&mut StsRandom::new(5));
        assert_eq!(room, if juzu { RoomType::Event } else { RoomType::Monster });
    }
    // Tiny Chest: every fourth ? room is a treasure room.
    let mut r = run();
    r.obtain_relic(RelicId::TinyChest);
    let mut rng = StsRandom::new(5);
    let rooms: Vec<RoomType> = (0..8)
        .map(|_| {
            r.event_chances = [0.0; 4];
            r.roll_event_room(&mut rng)
        })
        .collect();
    let t = RoomType::Treasure;
    let e = RoomType::Event;
    assert_eq!(rooms, [e, e, e, t, e, e, e, t]);
    // Winged Greaves: any node of the next row, three times.
    let mut r = run();
    r.obtain_relic(RelicId::WingBoots);
    let x = r.next_nodes()[0];
    r.enter_node(x);
    r.combat = None;
    r.screen = Screen::Map;
    let next = r.next_nodes();
    let far = *r.reachable_nodes().iter().find(|x| !next.contains(x)).expect("an unconnected node");
    r.step(&RunAction::Path { x: far }).unwrap();
    assert_eq!(relic(&r, RelicId::WingBoots).counter, 2);
    assert_eq!(r.pos, Some((far, 1)));
}

#[test]
fn shop_relics() {
    let shop = |relics: &[RelicId]| {
        let mut r = run();
        for &id in relics {
            r.relics.push(RelicState::new(id));
        }
        r.room = Some(RoomType::Shop);
        r.open_shop();
        let Screen::Shop { shop } = r.screen.clone() else { panic!() };
        let prices: Vec<i32> = shop
            .cards
            .iter()
            .flatten()
            .map(|c| c.price)
            .chain(shop.relics.iter().flatten().map(|x| x.price))
            .chain(shop.potions.iter().flatten().map(|p| p.price))
            .collect();
        (prices, shop.purge_cost)
    };
    let half = |p: i32| map::java_round(p as f32 * 0.5);
    let fifth_off = |p: i32| map::java_round(p as f32 * 0.8);
    let (base, purge) = shop(&[]);
    // Membership Card: half price; The Courier: 20% off; both compound.
    let (mc, mc_purge) = shop(&[RelicId::MembershipCard]);
    assert_eq!(mc, base.iter().map(|&p| half(p)).collect::<Vec<_>>());
    assert_eq!(mc_purge, half(purge));
    let (co, co_purge) = shop(&[RelicId::Courier]);
    assert_eq!(co, base.iter().map(|&p| fifth_off(p)).collect::<Vec<_>>());
    assert_eq!(co_purge, fifth_off(purge));
    let (both, _) = shop(&[RelicId::Courier, RelicId::MembershipCard]);
    assert_eq!(both, base.iter().map(|&p| half(fifth_off(p))).collect::<Vec<_>>());
    // Smiling Mask: removal always costs 50.
    assert_eq!(shop(&[RelicId::SmilingMask]).1, 50);

    // The Courier restocks a bought card; without it the slot stays empty.
    for courier in [false, true] {
        let mut r = run();
        if courier {
            r.obtain_relic(RelicId::Courier);
        }
        r.obtain_relic(RelicId::MawBank);
        r.gold = 999;
        r.room = Some(RoomType::Shop);
        r.open_shop();
        r.step(&RunAction::BuyCard { index: 0 }).unwrap();
        let Screen::Shop { shop } = &r.screen else { panic!() };
        assert_eq!(shop.cards[0].is_some(), courier);
        // Spending gold stops Maw Bank.
        assert!(relic(&r, RelicId::MawBank).used_up);
    }
}

#[test]
fn sozu_refuses_potions() {
    let mut r = run();
    r.obtain_relic(RelicId::Sozu);
    r.gold = 999;
    r.room = Some(RoomType::Shop);
    r.open_shop();
    assert!(r.step(&RunAction::BuyPotion { index: 0 }).is_err());
    // A potion reward is taken but nothing arrives.
    r.screen = Screen::Rewards { items: vec![Reward::Potion(PotionId::BlockPotion)] };
    r.step(&RunAction::TakeReward { index: 0 }).unwrap();
    assert!(rewards(&r).is_empty());
    assert!(r.potions.iter().all(|p| p.is_none()));
}

/// Digging and events roll the act's relic chances: Acts 1-3 50/33/17, The Ending always uncommon
/// (soak 2026-10-27, run GBFJ: a dig on Act 4's rest site).
#[test]
fn act_4_random_relics_are_uncommon() {
    let mut run = Run::new(RunConfig::new(1));
    run.act = 4;
    for _ in 0..100 {
        assert_eq!(run.random_relic_tier(), RelicTier::Uncommon);
    }
    run.act = 2;
    let tiers: std::collections::HashSet<RelicTier> = (0..300).map(|_| run.random_relic_tier()).collect();
    assert_eq!(tiers.len(), 3, "Acts 1-3 roll all three tiers");
}
