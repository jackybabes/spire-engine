//! Every event, one test each, checked against its class in the decompiled game
//! (`events/exordium`, `city`, `beyond`, `shrines`): the options it offers, their costs at A0 and
//! A15, and what each one gives or takes. Fights an event starts are checked as far as the fight
//! (the monsters and rewards are covered by the monster tests and the recordings).

use super::*;
use crate::cards::{CardColor, CardType};
use crate::potions::PotionId;
use events::{Event, EventId};

fn run_at(asc: i32) -> Run {
    let mut cfg = RunConfig::new(1);
    cfg.ascension = asc;
    Run::new(cfg)
}

fn run() -> Run {
    run_at(0)
}

/// Opens `id` as entering a ? room would.
fn open(r: &mut Run, id: EventId) {
    r.screen = Screen::Map;
    r.room = Some(RoomType::Event);
    r.open_event(id);
}

fn event(r: &Run) -> &Event {
    match &r.screen {
        Screen::Event { event } => event,
        s => panic!("not an event: {}", screen_name(s)),
    }
}

fn pick(r: &mut Run, k: usize) {
    r.step(&RunAction::EventChoice { index: k }).unwrap_or_else(|e| panic!("option {k}: {e}"));
}

/// The event is over: its one option leaves to the map.
fn leave(r: &mut Run) {
    assert_eq!(event(r).labels, ["leave"]);
    pick(r, 0);
    assert!(matches!(r.screen, Screen::Map));
}

fn grid(r: &Run) -> (GridPurpose, usize, Vec<usize>) {
    match &r.screen {
        Screen::Grid { purpose, options, count } => (*purpose, *count, options.clone()),
        s => panic!("not a grid: {}", screen_name(s)),
    }
}

fn select(r: &mut Run, indices: &[usize]) {
    r.step(&RunAction::Select { indices: indices.to_vec() }).unwrap();
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

fn add(r: &mut Run, id: CardId) {
    let c = r.new_card(id);
    r.deck.push(c);
}

// ---- Exordium ----

#[test]
fn big_fish() {
    // Banana: heal a third of max HP; Donut: 5 max HP; Box: a relic and Regret.
    let mut r = run();
    r.hp = 40;
    open(&mut r, EventId::BigFish);
    pick(&mut r, 0);
    assert_eq!(r.hp, 40 + 80 / 3);
    leave(&mut r);
    let mut r = run();
    open(&mut r, EventId::BigFish);
    pick(&mut r, 1);
    assert_eq!((r.hp, r.max_hp), (85, 85));
    let mut r = run();
    open(&mut r, EventId::BigFish);
    pick(&mut r, 2);
    assert_eq!((r.relics.len(), count(&r, CardId::Regret)), (2, 1));
}

#[test]
fn the_cleric() {
    // Heal 25% for 35 gold; purify for 50 (75 from A15).
    let mut r = run();
    r.hp = 40;
    open(&mut r, EventId::Cleric);
    pick(&mut r, 0);
    assert_eq!((r.gold, r.hp), (99 - 35, 60));
    for (asc, cost) in [(0, 50), (15, 75)] {
        let mut r = run_at(asc);
        r.gold = 99;
        open(&mut r, EventId::Cleric);
        let n = r.deck.len();
        pick(&mut r, 1);
        assert_eq!((grid(&r).0, grid(&r).1), (GridPurpose::Remove, 1));
        // (Ascender's Bane, first in an A15 deck, can't be removed.)
        let i = grid(&r).2[0];
        select(&mut r, &[i]);
        assert_eq!((r.gold, r.deck.len()), (99 - cost, n - 1));
    }
    let mut r = run();
    r.gold = 30;
    open(&mut r, EventId::Cleric);
    assert_eq!(event(&r).options, [false, false, true]);
}

#[test]
fn dead_adventurer() {
    // 25% (35% from A15) to wake the elite, 25% more each search; rewards in a shuffled order.
    assert_eq!({
        let mut r = run_at(15);
        open(&mut r, EventId::DeadAdventurer);
        event(&r).n[0]
    }, 35);
    let (mut fought, mut found) = (false, false);
    for seed in 0..30 {
        let mut r = Run::new(RunConfig::new(seed));
        open(&mut r, EventId::DeadAdventurer);
        assert_eq!(event(&r).n[0], 25);
        let first = event(&r).list[0];
        let roll = r.rng.misc.clone().random_range(0, 99);
        let (gold, relics) = (r.gold, r.relics.len());
        pick(&mut r, 0);
        if roll < 25 {
            fought = true;
            assert_eq!(event(&r).labels, ["fight"]);
            pick(&mut r, 0);
            assert!(matches!(r.screen, Screen::Combat));
            assert!(r.combat.as_ref().unwrap().elite);
        } else {
            found = true;
            assert_eq!(event(&r).n[0], 50);
            match first {
                0 => assert_eq!(r.gold, gold + 30),
                2 => assert_eq!(r.relics.len(), relics + 1),
                _ => assert_eq!((r.gold, r.relics.len()), (gold, relics)),
            }
        }
    }
    assert!(fought && found);
}

#[test]
fn golden_idol() {
    for (asc, dmg, max) in [(0, 0.25, 0.08), (15, 0.35, 0.1)] {
        let take = |k: usize| {
            let mut r = run_at(asc);
            open(&mut r, EventId::GoldenIdol);
            pick(&mut r, 0);
            assert!(r.has_relic(RelicId::GoldenIdol));
            let (hp, max_hp) = (r.hp, r.max_hp);
            pick(&mut r, k);
            (r, hp, max_hp)
        };
        let (r, _, _) = take(0);
        assert_eq!(count(&r, CardId::Injury), 1);
        let (r, hp, max_hp) = take(1);
        assert_eq!(r.hp, hp - (max_hp as f32 * dmg) as i32, "A{asc}");
        let (r, _, max_hp) = take(2);
        assert_eq!(r.max_hp, max_hp - (max_hp as f32 * max) as i32, "A{asc}");
    }
    // A second idol is a Circlet.
    let mut r = run();
    r.obtain_relic(RelicId::GoldenIdol);
    open(&mut r, EventId::GoldenIdol);
    pick(&mut r, 0);
    assert!(r.has_relic(RelicId::Circlet));
}

#[test]
fn golden_wing() {
    // Pray: lose 7 HP and remove a card. Destroy needs an Attack with 10+ damage: 50-80 gold.
    let mut r = run();
    open(&mut r, EventId::GoldenWing);
    assert_eq!(event(&r).options, [true, false, true]);
    let n = r.deck.len();
    pick(&mut r, 0);
    assert_eq!(r.hp, 73);
    pick(&mut r, 0);
    assert_eq!(grid(&r).0, GridPurpose::Remove);
    select(&mut r, &[0]);
    assert_eq!(r.deck.len(), n - 1);
    let mut r = run();
    add(&mut r, CardId::Bludgeon);
    open(&mut r, EventId::GoldenWing);
    let g = r.rng.misc.clone().random_range(50, 80);
    pick(&mut r, 1);
    assert_eq!(r.gold, 99 + g);
    assert!((50..=80).contains(&g));
}

#[test]
fn world_of_goop() {
    let mut r = run();
    open(&mut r, EventId::WorldOfGoop);
    pick(&mut r, 0);
    assert_eq!((r.hp, r.gold), (69, 99 + 75));
    // Leaving loses 20-50 gold (35-75 from A15), no more than the player has.
    for (asc, lo, hi) in [(0, 20, 50), (15, 35, 75)] {
        for seed in 0..20 {
            let mut cfg = RunConfig::new(seed);
            cfg.ascension = asc;
            let mut r = Run::new(cfg);
            r.gold = 60;
            open(&mut r, EventId::WorldOfGoop);
            let loss = event(&r).n[0];
            assert!(loss <= 60 && (loss == 60 || (lo..=hi).contains(&loss)), "A{asc}: {loss}");
            pick(&mut r, 1);
            assert_eq!(r.gold, 60 - loss);
        }
    }
}

#[test]
fn liars_game() {
    for (asc, gold) in [(0, 175), (15, 150)] {
        let mut r = run_at(asc);
        open(&mut r, EventId::LiarsGame);
        pick(&mut r, 0);
        pick(&mut r, 0);
        assert_eq!((r.gold, count(&r, CardId::Doubt)), (99 + gold, 1));
        leave(&mut r);
    }
    let mut r = run();
    open(&mut r, EventId::LiarsGame);
    pick(&mut r, 1);
    assert_eq!((r.gold, r.deck.len()), (99, 10));
}

#[test]
fn living_wall() {
    for (k, purpose) in [(0, GridPurpose::Remove), (1, GridPurpose::EventTransform), (2, GridPurpose::Upgrade)] {
        let mut r = run();
        open(&mut r, EventId::LivingWall);
        pick(&mut r, k);
        assert_eq!((grid(&r).0, grid(&r).1), (purpose, 1));
        let i = grid(&r).2[0];
        select(&mut r, &[i]);
    }
}

#[test]
fn mushrooms() {
    // Eat: heal 25% and a Parasite. Stomp: the Mushroom Lair, Odd Mushroom and 20-30 gold.
    let mut r = run();
    r.hp = 40;
    open(&mut r, EventId::Mushrooms);
    pick(&mut r, 1);
    assert_eq!((r.hp, count(&r, CardId::Parasite)), (60, 1));
    let mut r = run();
    open(&mut r, EventId::Mushrooms);
    pick(&mut r, 0);
    pick(&mut r, 0);
    assert!(matches!(r.screen, Screen::Combat));
    assert_eq!(r.encounter.as_deref(), Some("The Mushroom Lair"));
    let pending = r.event_rewards_pending.clone().unwrap();
    assert!(matches!(pending[..], [Reward::Gold(20..=30), Reward::Relic(RelicId::OddMushroom)]));
}

#[test]
fn scrap_ooze() {
    // Reach in: 3 damage (5 from A15) for a 25% relic chance; each miss adds 1 damage and 10%.
    assert_eq!({
        let mut r = run_at(15);
        open(&mut r, EventId::ScrapOoze);
        event(&r).n[0]
    }, 5);
    let mut r = run();
    open(&mut r, EventId::ScrapOoze);
    for i in 0..20 {
        let (dmg, chance) = (event(&r).n[0], event(&r).n[1]);
        assert_eq!((dmg, chance), (3 + i, 25 + 10 * i));
        let roll = r.rng.misc.clone().random_range(0, 99);
        let (hp, relics) = (r.hp, r.relics.len());
        pick(&mut r, 0);
        assert_eq!(r.hp, hp - dmg);
        if roll >= 99 - chance {
            assert_eq!(r.relics.len(), relics + 1);
            leave(&mut r);
            return;
        }
    }
    panic!("never found the relic");
}

#[test]
fn shining_light() {
    for (asc, pct) in [(0, 0.2), (15, 0.3)] {
        let mut r = run_at(asc);
        open(&mut r, EventId::ShiningLight);
        let (hp, max) = (r.hp, r.max_hp);
        pick(&mut r, 0);
        assert_eq!(r.hp, hp - map::java_round(max as f32 * pct));
        assert_eq!(r.deck.iter().filter(|c| c.upgraded).count(), 2);
    }
}

#[test]
fn match_and_keep() {
    // Six pairs: a rare, an uncommon, a common, a colorless uncommon, a curse and the starter
    // card (from A15 a second curse instead of the colorless card).
    let mut r = run();
    open(&mut r, EventId::MatchAndKeep);
    let cards = event(&r).cards.clone();
    assert_eq!(cards.len(), 12);
    for c in &cards {
        assert_eq!(cards.iter().filter(|x| x.id == c.id).count(), 2, "{:?}", c.id);
    }
    assert!(cards.iter().any(|c| c.id == CardId::Bash));
    assert!(cards.iter().any(|c| c.data().color == CardColor::Colorless));
    assert_eq!(cards.iter().filter(|c| c.card_type() == CardType::Curse).count(), 2);
    let mut r15 = run_at(15);
    open(&mut r15, EventId::MatchAndKeep);
    let cards15 = &event(&r15).cards;
    assert_eq!(cards15.iter().filter(|c| c.card_type() == CardType::Curse).count(), 4);
    assert!(cards15.iter().any(|c| c.id == CardId::Bash) && !cards15.iter().any(|c| c.data().color == CardColor::Colorless));
    // Five tries; a match keeps the card.
    pick(&mut r, 0);
    pick(&mut r, 0);
    let first = 0;
    let pair = (1..cards.len()).find(|&j| cards[j].id == cards[0].id).unwrap();
    pick(&mut r, first);
    pick(&mut r, pair);
    assert_eq!(count(&r, cards[0].id), if cards[0].id == CardId::Bash { 2 } else { 1 });
    assert_eq!((event(&r).n[0], event(&r).cards.len()), (4, 10));
}

#[test]
fn golden_shrine() {
    for (asc, gold) in [(0, 100), (15, 50)] {
        let mut r = run_at(asc);
        open(&mut r, EventId::GoldenShrine);
        pick(&mut r, 0);
        assert_eq!(r.gold, 99 + gold);
    }
    let mut r = run();
    open(&mut r, EventId::GoldenShrine);
    pick(&mut r, 1);
    assert_eq!((r.gold, count(&r, CardId::Regret)), (99 + 275, 1));
}

#[test]
fn card_shrines() {
    // Transmogrifier, Purifier, Upgrade Shrine: one card through the grid.
    for (id, purpose) in [(EventId::Transmogrifier, GridPurpose::EventTransform), (EventId::Purifier, GridPurpose::Remove), (EventId::UpgradeShrine, GridPurpose::Upgrade)] {
        let mut r = run();
        open(&mut r, id);
        pick(&mut r, 0);
        assert_eq!((grid(&r).0, grid(&r).1), (purpose, 1), "{id:?}");
        let i = grid(&r).2[0];
        let before = r.deck[i].clone();
        select(&mut r, &[i]);
        match id {
            EventId::Purifier => assert_eq!(r.deck.len(), 9),
            EventId::UpgradeShrine => assert!(r.deck[i].upgraded),
            _ => assert_ne!(r.deck.last().unwrap().id, before.id),
        }
        leave(&mut r);
    }
}

#[test]
fn wheel_of_change() {
    // miscRng 0-5: 100 gold in Act 1, a relic, a full heal, Decay, a removal, or 10% max HP lost
    // (15% from A15).
    let mut seen = [false; 6];
    for seed in 0..60 {
        let mut r = Run::new(RunConfig::new(seed));
        r.hp = 40;
        open(&mut r, EventId::WheelOfChange);
        assert_eq!(event(&r).n[1], 10);
        let result = r.rng.misc.clone().random_range(0, 5);
        pick(&mut r, 0);
        assert_eq!(event(&r).n[0], result);
        let gold = r.gold;
        pick(&mut r, 0);
        match result {
            0 => assert_eq!(gold, 99 + 100),
            1 => assert!(matches!(rewards(&r)[..], [Reward::Relic(_)])),
            2 => assert_eq!(r.hp, 80),
            3 => assert_eq!(count(&r, CardId::Decay), 1),
            4 => assert_eq!(grid(&r).0, GridPurpose::Remove),
            _ => assert_eq!(r.hp, 40 - 8),
        }
        seen[result as usize] = true;
    }
    assert!(seen.iter().all(|&s| s));
    let mut r = run_at(15);
    open(&mut r, EventId::WheelOfChange);
    assert_eq!(event(&r).n[1], 15);
}

#[test]
fn accursed_blacksmith() {
    let mut r = run();
    open(&mut r, EventId::AccursedBlacksmith);
    pick(&mut r, 0);
    assert_eq!(grid(&r).0, GridPurpose::Upgrade);
    let mut r = run();
    open(&mut r, EventId::AccursedBlacksmith);
    pick(&mut r, 1);
    assert!(r.has_relic(RelicId::WarpedTongs));
    assert_eq!(count(&r, CardId::Pain), 1);
    leave(&mut r);
}

#[test]
fn bonfire_spirits() {
    // The offered card's rarity: Basic nothing, Common 5 HP, Uncommon full heal, Rare 10 max HP
    // and full heal, Curse Spirit Poop.
    let offer = |id: CardId| {
        let mut r = run();
        r.hp = 40;
        add(&mut r, id);
        open(&mut r, EventId::Bonfire);
        pick(&mut r, 0);
        pick(&mut r, 0);
        assert_eq!(grid(&r).0, GridPurpose::Bonfire);
        let i = r.deck.len() - 1;
        select(&mut r, &[i]);
        assert_eq!(r.deck.len(), 10);
        r
    };
    assert_eq!(offer(CardId::StrikeR).hp, 40);
    assert_eq!(offer(CardId::Anger).hp, 45);
    assert_eq!(offer(CardId::Uppercut).hp, 80);
    let r = offer(CardId::Bludgeon);
    assert_eq!((r.hp, r.max_hp), (90, 90));
    assert!(offer(CardId::Regret).has_relic(RelicId::SpiritPoop));
}

#[test]
fn fountain_of_cleansing() {
    let mut r = run();
    for id in [CardId::Regret, CardId::Doubt, CardId::AscendersBane, CardId::Necronomicurse, CardId::CurseOfTheBell] {
        add(&mut r, id);
    }
    open(&mut r, EventId::FountainOfCleansing);
    pick(&mut r, 0);
    let curses: Vec<CardId> = r.deck.iter().filter(|c| c.card_type() == CardType::Curse).map(|c| c.id).collect();
    assert_eq!(curses, [CardId::AscendersBane, CardId::Necronomicurse, CardId::CurseOfTheBell]);
}

#[test]
fn face_trader() {
    // Touch: 75 gold (50 from A15), damage a tenth of max HP. Trade: one of five faces.
    for (asc, gold) in [(0, 75), (15, 50)] {
        let mut r = run_at(asc);
        open(&mut r, EventId::FaceTrader);
        pick(&mut r, 0);
        let (hp, max) = (r.hp, r.max_hp);
        pick(&mut r, 0);
        assert_eq!((r.gold, r.hp), (99 + gold, hp - max / 10));
    }
    let mut r = run();
    open(&mut r, EventId::FaceTrader);
    pick(&mut r, 0);
    pick(&mut r, 1);
    let face = r.relics.last().unwrap().id;
    assert!(matches!(face, RelicId::CultistMask | RelicId::FaceOfCleric | RelicId::GremlinMask | RelicId::NlothsMask | RelicId::SsserpentHead));
}

#[test]
fn the_lab() {
    for (asc, n) in [(0, 3), (15, 2)] {
        let mut r = run_at(asc);
        open(&mut r, EventId::Lab);
        pick(&mut r, 0);
        let items = rewards(&r);
        assert!(items.len() == n && items.iter().all(|i| matches!(i, Reward::Potion(_))));
    }
}

#[test]
fn note_for_yourself() {
    // Take the stored card (Iron Wave on a fresh profile), then store one.
    let mut r = run();
    open(&mut r, EventId::NoteForYourself);
    pick(&mut r, 0);
    pick(&mut r, 0);
    assert_eq!(count(&r, CardId::IronWave), 1);
    assert_eq!(grid(&r).0, GridPurpose::Remove);
    select(&mut r, &[0]);
    assert_eq!(r.deck.len(), 10);
}

#[test]
fn we_meet_again() {
    let setup = || {
        let mut r = run();
        r.gold = 120;
        r.potions = vec![Some(PotionId::BlockPotion), None, Some(PotionId::FirePotion)];
        add(&mut r, CardId::Anger);
        open(&mut r, EventId::WeMeetAgain);
        r
    };
    // A potion, gold (50 to the player's gold, at most 150) or a non-basic card, for a relic.
    let mut r = setup();
    let slot = event(&r).potion_slot.unwrap();
    pick(&mut r, 0);
    assert!(r.potions[slot].is_none() && r.relics.len() == 2);
    let mut r = setup();
    let g = event(&r).n[0];
    assert!((50..=120).contains(&g));
    pick(&mut r, 1);
    assert_eq!((r.gold, r.relics.len()), (120 - g, 2));
    let mut r = setup();
    assert_eq!(event(&r).card_index, Some(10));
    pick(&mut r, 2);
    assert_eq!((count(&r, CardId::Anger), r.relics.len()), (0, 2));
    // Nothing to give: the options are off.
    let mut r = run();
    r.gold = 30;
    open(&mut r, EventId::WeMeetAgain);
    assert_eq!(event(&r).options, [false, false, false, true]);
}

#[test]
fn woman_in_blue() {
    for (k, cost) in [(0, 20), (1, 30), (2, 40)] {
        let mut r = run();
        open(&mut r, EventId::WomanInBlue);
        pick(&mut r, k);
        assert_eq!(r.gold, 99 - cost);
        assert_eq!(rewards(&r).len(), k + 1);
    }
    // Leaving costs 5% of max HP (rounded up) from A15.
    for (asc, loss) in [(0, 0), (15, 4)] {
        let mut r = run_at(asc);
        let hp = r.hp;
        open(&mut r, EventId::WomanInBlue);
        pick(&mut r, 3);
        assert_eq!(r.hp, hp - loss, "A{asc}");
    }
}

// ---- The City ----

#[test]
fn the_addict() {
    let mut r = run();
    open(&mut r, EventId::Addict);
    pick(&mut r, 0);
    assert_eq!((r.gold, r.relics.len()), (99 - 85, 2));
    let mut r = run();
    open(&mut r, EventId::Addict);
    pick(&mut r, 1);
    assert_eq!((r.relics.len(), count(&r, CardId::Shame)), (2, 1));
    let mut r = run();
    r.gold = 80;
    open(&mut r, EventId::Addict);
    assert_eq!(event(&r).options, [false, true, true]);
}

#[test]
fn back_to_basics() {
    let mut r = run();
    open(&mut r, EventId::BackToBasics);
    pick(&mut r, 1);
    for c in &r.deck {
        assert_eq!(c.upgraded, c.id != CardId::Bash, "{:?}", c.id);
    }
    let mut r = run();
    open(&mut r, EventId::BackToBasics);
    pick(&mut r, 0);
    assert_eq!(grid(&r).0, GridPurpose::Remove);
}

#[test]
fn the_beggar() {
    let mut r = run();
    open(&mut r, EventId::Beggar);
    pick(&mut r, 0);
    assert_eq!(r.gold, 24);
    pick(&mut r, 0);
    select(&mut r, &[0]);
    assert_eq!(r.deck.len(), 9);
    assert!(matches!(r.screen, Screen::Map));
    let mut r = run();
    r.gold = 70;
    open(&mut r, EventId::Beggar);
    assert_eq!(event(&r).options, [false, true]);
}

#[test]
fn the_colosseum() {
    let mut r = run();
    open(&mut r, EventId::Colosseum);
    pick(&mut r, 0);
    pick(&mut r, 0);
    assert!(matches!(r.screen, Screen::Combat));
    assert_eq!(r.encounter.as_deref(), Some("Colosseum Slavers"));
    assert!(r.event_resume.is_some());
}

#[test]
fn cursed_tome() {
    for (asc, last) in [(0, 10), (15, 15)] {
        let mut r = run_at(asc);
        r.hp = 60;
        open(&mut r, EventId::CursedTome);
        pick(&mut r, 0);
        for _ in 0..3 {
            pick(&mut r, 0);
        }
        assert_eq!(r.hp, 60 - 6);
        pick(&mut r, 0);
        assert_eq!(r.hp, 60 - 6 - last);
        let book = match rewards(&r)[..] {
            [Reward::Relic(b)] => b,
            _ => panic!("a book"),
        };
        assert!(matches!(book, RelicId::Necronomicon | RelicId::Enchiridion | RelicId::NilrysCodex));
    }
    let mut r = run();
    r.hp = 60;
    open(&mut r, EventId::CursedTome);
    for _ in 0..4 {
        pick(&mut r, 0);
    }
    pick(&mut r, 1);
    assert_eq!(r.hp, 60 - 6 - 3);
    leave(&mut r);
}

#[test]
fn drug_dealer() {
    let mut r = run();
    open(&mut r, EventId::DrugDealer);
    pick(&mut r, 0);
    assert_eq!(count(&r, CardId::JAX), 1);
    let mut r = run();
    open(&mut r, EventId::DrugDealer);
    pick(&mut r, 1);
    assert_eq!((grid(&r).0, grid(&r).1), (GridPurpose::EventTransform, 2));
    let mut r = run();
    open(&mut r, EventId::DrugDealer);
    pick(&mut r, 2);
    assert!(r.has_relic(RelicId::MutagenicStrength));
}

#[test]
fn forgotten_altar() {
    // Offer the Golden Idol: Bloody Idol in its place.
    let mut r = run();
    r.obtain_relic(RelicId::GoldenIdol);
    r.obtain_relic(RelicId::Anchor);
    open(&mut r, EventId::ForgottenAltar);
    pick(&mut r, 0);
    assert_eq!(r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), [RelicId::BurningBlood, RelicId::BloodyIdol, RelicId::Anchor]);
    // Sacrifice: 5 max HP, then 25% of max HP as damage (35% from A15).
    for (asc, pct) in [(0, 0.25), (15, 0.35)] {
        let mut r = run_at(asc);
        open(&mut r, EventId::ForgottenAltar);
        assert!(!event(&r).options[0]);
        let (hp, max) = (r.hp, r.max_hp);
        pick(&mut r, 1);
        assert_eq!((r.max_hp, r.hp), (max + 5, hp + 5 - map::java_round(max as f32 * pct)), "A{asc}");
    }
    let mut r = run();
    open(&mut r, EventId::ForgottenAltar);
    pick(&mut r, 2);
    assert_eq!(count(&r, CardId::Decay), 1);
}

#[test]
fn council_of_ghosts() {
    for (asc, n) in [(0, 5), (15, 3)] {
        let mut r = run_at(asc);
        open(&mut r, EventId::Ghosts);
        let max = r.max_hp;
        pick(&mut r, 0);
        assert_eq!(r.max_hp, max - (max as f32 * 0.5).ceil() as i32);
        assert_eq!(count(&r, CardId::Apparition), n, "A{asc}");
    }
}

#[test]
fn masked_bandits() {
    let mut r = run();
    open(&mut r, EventId::MaskedBandits);
    pick(&mut r, 0);
    assert_eq!(r.gold, 0);
    pick(&mut r, 0);
    pick(&mut r, 0);
    leave(&mut r);
    let mut r = run();
    open(&mut r, EventId::MaskedBandits);
    pick(&mut r, 1);
    assert_eq!(r.encounter.as_deref(), Some("Masked Bandits"));
    let pending = r.event_rewards_pending.clone().unwrap();
    assert!(matches!(pending[..], [Reward::Gold(25..=35), Reward::Relic(RelicId::RedMask)]));
}

#[test]
fn the_nest() {
    for (asc, gold) in [(0, 99), (15, 50)] {
        let mut r = run_at(asc);
        open(&mut r, EventId::Nest);
        pick(&mut r, 0);
        pick(&mut r, 0);
        assert_eq!(r.gold, 99 + gold);
    }
    let mut r = run();
    open(&mut r, EventId::Nest);
    pick(&mut r, 0);
    pick(&mut r, 1);
    assert_eq!((r.hp, count(&r, CardId::RitualDagger)), (74, 1));
}

#[test]
fn the_library() {
    // Sleep: heal 33% (20% from A15). Read: one of 20 different cards.
    for (asc, pct) in [(0, 0.33), (15, 0.2)] {
        let mut r = run_at(asc);
        r.hp = 30;
        open(&mut r, EventId::TheLibrary);
        pick(&mut r, 1);
        assert_eq!(r.hp, 30 + map::java_round(r.max_hp as f32 * pct));
    }
    let mut r = run();
    open(&mut r, EventId::TheLibrary);
    pick(&mut r, 0);
    let cards = event(&r).cards.clone();
    assert_eq!(cards.len(), 20);
    assert!(cards.iter().all(|c| cards.iter().filter(|x| x.id == c.id).count() == 1));
    pick(&mut r, 7);
    assert_eq!(r.deck.last().unwrap().id, cards[7].id);
    // In the game's grid order: the card generated last first (`CardGroup.addToBottom`). The first
    // card rolled (on the same rng) is the grid's last.
    let mut r = run();
    open(&mut r, EventId::TheLibrary);
    let first = r.library_card();
    let mut r = run();
    open(&mut r, EventId::TheLibrary);
    pick(&mut r, 0);
    assert_eq!(event(&r).cards[19].id, first);
}

#[test]
fn the_mausoleum() {
    // A relic; Writhe on a coin flip (always from A15).
    for seed in 0..10 {
        let mut r = Run::new(RunConfig::new(seed));
        open(&mut r, EventId::TheMausoleum);
        let cursed = r.rng.misc.clone().random_boolean();
        pick(&mut r, 0);
        assert_eq!((r.relics.len(), count(&r, CardId::Writhe)), (2, cursed as usize));
    }
    let mut r = run_at(15);
    open(&mut r, EventId::TheMausoleum);
    pick(&mut r, 0);
    assert_eq!(count(&r, CardId::Writhe), 1);
}

#[test]
fn vampires() {
    // Lose 30% of max HP (rounded up) or the Blood Vial; the Strikes become five Bites.
    let mut r = run();
    open(&mut r, EventId::Vampires);
    pick(&mut r, 0);
    assert_eq!(r.max_hp, 80 - 24);
    assert_eq!((count(&r, CardId::StrikeR), count(&r, CardId::Bite)), (0, 5));
    let mut r = run();
    r.obtain_relic(RelicId::BloodVial);
    open(&mut r, EventId::Vampires);
    pick(&mut r, 1);
    assert_eq!((r.max_hp, r.has_relic(RelicId::BloodVial), count(&r, CardId::Bite)), (80, false, 5));
}

#[test]
fn the_designer() {
    // Costs 40/60/90 and a 3 HP punch (50/75/110 and 5 from A15).
    for (asc, costs) in [(0, [40, 60, 90, 3]), (15, [50, 75, 110, 5])] {
        let mut r = run_at(asc);
        r.gold = 200;
        open(&mut r, EventId::Designer);
        pick(&mut r, 0);
        assert_eq!(event(&r).list, costs.map(|c| c as u8));
        let hp = r.hp;
        pick(&mut r, 3);
        assert_eq!(r.hp, hp - costs[3] as i32);
    }
    // Adjustments: one card by choice or two at random; clean up: remove one or transform two.
    for seed in 0..12 {
        let mut r = Run::new(RunConfig::new(seed));
        r.gold = 200;
        open(&mut r, EventId::Designer);
        let (one, removes) = (event(&r).n[0] == 1, event(&r).n[1] == 1);
        pick(&mut r, 0);
        let mut a = r.clone();
        pick(&mut a, 0);
        assert_eq!(a.gold, 160);
        if one {
            assert_eq!(grid(&a).0, GridPurpose::Upgrade);
        } else {
            assert_eq!(a.deck.iter().filter(|c| c.upgraded).count(), 2);
        }
        let mut b = r.clone();
        pick(&mut b, 1);
        assert_eq!((grid(&b).0, grid(&b).1), if removes { (GridPurpose::Remove, 1) } else { (GridPurpose::EventTransform, 2) });
        pick(&mut r, 2);
        assert_eq!((r.gold, grid(&r).0), (110, GridPurpose::RemoveAndUpgrade));
    }
}

#[test]
fn the_duplicator() {
    let mut r = run();
    open(&mut r, EventId::Duplicator);
    pick(&mut r, 0);
    let bash = r.deck.iter().position(|c| c.id == CardId::Bash).unwrap();
    select(&mut r, &[bash]);
    assert_eq!(count(&r, CardId::Bash), 2);
}

#[test]
fn knowing_skull() {
    // Each answer costs 6 HP, one more each time it is asked.
    let mut r = run();
    r.hp = 70;
    open(&mut r, EventId::KnowingSkull);
    pick(&mut r, 0);
    pick(&mut r, 0);
    assert!(r.potions[0].is_some());
    pick(&mut r, 0);
    pick(&mut r, 1);
    assert_eq!(r.gold, 99 + 90);
    pick(&mut r, 2);
    assert_eq!(r.deck.last().unwrap().data().color, CardColor::Colorless);
    assert_eq!(r.hp, 70 - 6 - 7 - 6 - 6);
    assert_eq!(event(&r).n, [8, 7, 7, 6]);
    pick(&mut r, 3);
    assert_eq!(r.hp, 70 - 25 - 6);
    leave(&mut r);
}

#[test]
fn nloth() {
    let mut r = run();
    r.obtain_relic(RelicId::Anchor);
    r.obtain_relic(RelicId::Vajra);
    open(&mut r, EventId::Nloth);
    let offered = event(&r).relics.clone();
    assert_eq!(offered.len(), 2);
    pick(&mut r, 1);
    assert!(!r.has_relic(offered[1]) && r.has_relic(offered[0]) && r.has_relic(RelicId::NlothsGift));
}

#[test]
fn the_joust() {
    // 50 gold on the murderer (100 back when he wins, 70%) or the owner (250, 30%).
    let mut paid = [false; 2];
    for seed in 0..20 {
        for bet in [0, 1] {
            let mut r = Run::new(RunConfig::new(seed));
            open(&mut r, EventId::TheJoust);
            pick(&mut r, 0);
            pick(&mut r, bet);
            assert_eq!(r.gold, 49);
            let owner_wins = r.rng.misc.clone().random_boolean_chance(0.3);
            pick(&mut r, 0);
            pick(&mut r, 0);
            let won = owner_wins == (bet == 1);
            assert_eq!(r.gold, 49 + if won { [100, 250][bet] } else { 0 });
            paid[bet] |= won;
        }
    }
    assert!(paid[0] && paid[1]);
}

// ---- The Beyond ----

#[test]
fn falling() {
    // Lose a random Skill, Power or Attack (miscRng picks each when the event opens).
    let mut r = run();
    add(&mut r, CardId::Inflame);
    open(&mut r, EventId::Falling);
    let [skill, power, attack, _] = event(&r).n;
    assert_eq!(r.deck[power as usize].id, CardId::Inflame);
    assert_eq!(r.deck[skill as usize].card_type(), CardType::Skill);
    assert_eq!(r.deck[attack as usize].card_type(), CardType::Attack);
    pick(&mut r, 0);
    pick(&mut r, 1);
    assert_eq!((count(&r, CardId::Inflame), r.deck.len()), (0, 10));
    // No Power: that option is off.
    let mut r = run();
    open(&mut r, EventId::Falling);
    pick(&mut r, 0);
    assert_eq!(event(&r).options, [true, false, true]);
}

#[test]
fn mind_bloom() {
    // War: an Act 1 boss for 50 gold (25 from A13) and a rare relic.
    let mut r = run();
    open(&mut r, EventId::MindBloom);
    pick(&mut r, 0);
    assert!(matches!(r.encounter.as_deref(), Some("The Guardian" | "Hexaghost" | "Slime Boss")));
    let pending = r.event_rewards_pending.clone().unwrap();
    assert!(matches!(pending[..], [Reward::Gold(50), Reward::Relic(_)]));
    // Awake: upgrade everything, Mark of the Bloom.
    let mut r = run();
    open(&mut r, EventId::MindBloom);
    pick(&mut r, 1);
    assert!(r.deck.iter().all(|c| c.upgraded) && r.has_relic(RelicId::MarkOfTheBloom));
    // Rich (floor 40 or less): 999 gold and two Normality; Healthy after: full heal and Doubt.
    let mut r = run();
    open(&mut r, EventId::MindBloom);
    pick(&mut r, 2);
    assert_eq!((r.gold, count(&r, CardId::Normality)), (99 + 999, 2));
    let mut r = run();
    r.floor = 45;
    r.hp = 20;
    open(&mut r, EventId::MindBloom);
    pick(&mut r, 2);
    assert_eq!((r.hp, count(&r, CardId::Doubt)), (80, 1));
}

#[test]
fn the_moai_head() {
    for (asc, pct) in [(0, 0.125), (15, 0.18)] {
        let mut r = run_at(asc);
        r.hp = 20;
        let max = r.max_hp;
        open(&mut r, EventId::MoaiHead);
        pick(&mut r, 0);
        let left = max - map::java_round(max as f32 * pct);
        assert_eq!((r.max_hp, r.hp), (left, left), "A{asc}");
    }
    let mut r = run();
    r.obtain_relic(RelicId::GoldenIdol);
    open(&mut r, EventId::MoaiHead);
    pick(&mut r, 1);
    assert_eq!((r.gold, r.has_relic(RelicId::GoldenIdol)), (99 + 333, false));
}

#[test]
fn mysterious_sphere() {
    let mut r = run();
    open(&mut r, EventId::MysteriousSphere);
    pick(&mut r, 0);
    pick(&mut r, 0);
    assert_eq!(r.encounter.as_deref(), Some("2 Orb Walkers"));
    let pending = r.event_rewards_pending.clone().unwrap();
    match pending[..] {
        [Reward::Gold(g), Reward::Relic(rel)] => assert!((45..=55).contains(&g) && rel.data().tier == RelicTier::Rare),
        _ => panic!("{pending:?}"),
    }
}

#[test]
fn secret_portal() {
    let mut r = run();
    open(&mut r, EventId::SecretPortal);
    pick(&mut r, 0);
    pick(&mut r, 0);
    assert_eq!(r.room, Some(RoomType::Boss));
    assert_eq!(r.encounter.as_deref(), Some(r.boss.as_str()));
}

#[test]
fn sensory_stone() {
    // 1, 2 or 3 colorless card rewards for 0, 5 or 10 HP.
    for k in 0..3 {
        let mut r = run();
        open(&mut r, EventId::SensoryStone);
        pick(&mut r, 0);
        pick(&mut r, k);
        let items = rewards(&r);
        assert_eq!(items.len(), k + 1);
        for i in &items {
            assert!(matches!(i, Reward::Cards(cs) if cs.iter().all(|c| c.data().color == CardColor::Colorless)));
        }
        assert_eq!(r.hp, 80 - 5 * k as i32);
    }
}

#[test]
fn tomb_of_lord_red_mask() {
    let mut r = run();
    open(&mut r, EventId::TombRedMask);
    assert_eq!(event(&r).options, [false, true, true]);
    pick(&mut r, 1);
    assert_eq!((r.gold, r.has_relic(RelicId::RedMask)), (0, true));
    let mut r = run();
    r.obtain_relic(RelicId::RedMask);
    open(&mut r, EventId::TombRedMask);
    pick(&mut r, 0);
    assert_eq!(r.gold, 99 + 222);
}

#[test]
fn winding_halls() {
    for (asc, hurt, heal) in [(0, 0.125, 0.25), (15, 0.18, 0.2)] {
        let max = run_at(asc).max_hp as f32;
        let mut r = run_at(asc);
        r.hp = 40;
        open(&mut r, EventId::WindingHalls);
        pick(&mut r, 0);
        pick(&mut r, 0);
        assert_eq!((r.hp, count(&r, CardId::Madness)), (40 - map::java_round(max * hurt), 2));
        let mut r = run_at(asc);
        r.hp = 40;
        open(&mut r, EventId::WindingHalls);
        pick(&mut r, 0);
        pick(&mut r, 1);
        assert_eq!((r.hp, count(&r, CardId::Writhe)), (40 + map::java_round(max * heal), 1));
        let mut r = run_at(asc);
        open(&mut r, EventId::WindingHalls);
        pick(&mut r, 0);
        pick(&mut r, 2);
        assert_eq!(r.max_hp, max as i32 - map::java_round(max * 0.05));
    }
}

// ---- Options the tests above don't take ----

/// Leaving changes nothing (each event's leave button).
#[test]
fn leaving_events_changes_nothing() {
    // (event, the leave option, whether it goes straight to the map)
    let cases = [
        (EventId::Beggar, 1, false),
        (EventId::CursedTome, 1, false),
        (EventId::Duplicator, 1, false),
        (EventId::FountainOfCleansing, 1, false),
        (EventId::MoaiHead, 2, false),
        (EventId::SecretPortal, 1, false),
        (EventId::TombRedMask, 2, true),
        (EventId::WeMeetAgain, 3, false),
        (EventId::Nloth, 2, false),
    ];
    for (id, k, straight) in cases {
        let mut r = run();
        add(&mut r, CardId::Regret);
        r.obtain_relic(RelicId::Anchor);
        r.potions = vec![Some(PotionId::BlockPotion), None, None];
        open(&mut r, id);
        let (hp, max, gold) = (r.hp, r.max_hp, r.gold);
        let deck: Vec<CardId> = r.deck.iter().map(|c| c.id).collect();
        let relics: Vec<RelicId> = r.relics.iter().map(|x| x.id).collect();
        pick(&mut r, k);
        if straight {
            assert!(matches!(r.screen, Screen::Map), "{id:?}");
        } else {
            leave(&mut r);
        }
        assert_eq!((r.hp, r.max_hp, r.gold), (hp, max, gold), "{id:?}");
        assert_eq!(r.deck.iter().map(|c| c.id).collect::<Vec<_>>(), deck, "{id:?}");
        assert_eq!(r.relics.iter().map(|x| x.id).collect::<Vec<_>>(), relics, "{id:?}");
        assert_eq!(r.potions[0], Some(PotionId::BlockPotion), "{id:?}");
    }
    // With the Red Mask, the tomb's leave is the second option.
    let mut r = run();
    r.obtain_relic(RelicId::RedMask);
    open(&mut r, EventId::TombRedMask);
    pick(&mut r, 1);
    assert!(matches!(r.screen, Screen::Map));
    assert_eq!(r.gold, 99);
    // Note For Yourself: leaving takes no card and stores none.
    let mut r = run();
    open(&mut r, EventId::NoteForYourself);
    pick(&mut r, 0);
    pick(&mut r, 1);
    assert_eq!((r.deck.len(), count(&r, CardId::IronWave)), (10, 0));
}

#[test]
fn falling_loses_the_chosen_kind() {
    // The skill (option 0) or the attack (option 2) picked when the event opened.
    for (k, t) in [(0, CardType::Skill), (2, CardType::Attack)] {
        let mut r = run();
        open(&mut r, EventId::Falling);
        let i = event(&r).n[k] as usize;
        let lost = r.deck[i].id;
        assert_eq!(r.deck[i].card_type(), t);
        let before = count(&r, lost);
        pick(&mut r, 0);
        pick(&mut r, k);
        assert_eq!((r.deck.len(), count(&r, lost)), (9, before - 1));
    }
}

#[test]
fn nloth_takes_either_relic() {
    let mut r = run();
    r.obtain_relic(RelicId::Anchor);
    r.obtain_relic(RelicId::Vajra);
    open(&mut r, EventId::Nloth);
    let offered = event(&r).relics.clone();
    pick(&mut r, 0);
    assert!(!r.has_relic(offered[0]) && r.has_relic(offered[1]) && r.has_relic(RelicId::NlothsGift));
    // A second gift is a Circlet, and nothing is taken.
    let mut r = run();
    r.obtain_relic(RelicId::NlothsGift);
    r.obtain_relic(RelicId::Anchor);
    open(&mut r, EventId::Nloth);
    pick(&mut r, 0);
    assert_eq!(r.relics.len(), 4);
    assert!(r.has_relic(RelicId::Circlet));
}

#[test]
fn vampires_with_the_blood_vial() {
    // Three options: accept (lose max HP, keep the vial), give the vial, refuse.
    let mut r = run();
    r.obtain_relic(RelicId::BloodVial);
    open(&mut r, EventId::Vampires);
    assert_eq!(event(&r).options.len(), 3);
    pick(&mut r, 0);
    assert_eq!((r.max_hp, r.has_relic(RelicId::BloodVial), count(&r, CardId::Bite)), (80 - 24, true, 5));
    let mut r = run();
    r.obtain_relic(RelicId::BloodVial);
    open(&mut r, EventId::Vampires);
    pick(&mut r, 2);
    assert_eq!((r.max_hp, r.has_relic(RelicId::BloodVial), count(&r, CardId::Bite)), (80, true, 0));
    leave(&mut r);
}

/// Wins the current fight at once: every monster at 1 HP, then a Strike each.
fn win_now(r: &mut Run) {
    loop {
        let c = r.combat.as_mut().unwrap();
        let Some(m) = c.order.iter().copied().find(|&m| !c.monsters[m].is_dead_or_escaped()) else { break };
        c.monsters[m].hp = 1;
        c.monsters[m].block = 0;
        c.energy = 3;
        let card = c.new_card(CardId::StrikeR);
        c.hand.push(card);
        let i = c.hand.len() - 1;
        r.step(&RunAction::Combat { action: PlayerAction::PlayCard { hand_index: i, target: Some(m) } }).unwrap();
        if !matches!(r.screen, Screen::Combat) {
            break;
        }
    }
}

#[test]
fn the_colosseum_after_the_first_fight() {
    // After the Slavers: cowardice leaves; victory fights the Nobs for a rare relic, an uncommon
    // relic and 100 gold.
    let start = || {
        let mut r = run();
        open(&mut r, EventId::Colosseum);
        pick(&mut r, 0);
        pick(&mut r, 0);
        win_now(&mut r);
        assert_eq!(event(&r).screen, 2);
        r
    };
    let mut r = start();
    pick(&mut r, 0);
    assert!(matches!(r.screen, Screen::Map));
    let mut r = start();
    pick(&mut r, 1);
    assert_eq!(r.encounter.as_deref(), Some("Colosseum Nobs"));
    assert!(r.combat.as_ref().unwrap().elite);
    let pending = r.event_rewards_pending.clone().unwrap();
    match pending[..] {
        [Reward::Relic(a), Reward::Relic(b), Reward::Gold(100)] => {
            assert_eq!((a.data().tier, b.data().tier), (RelicTier::Rare, RelicTier::Uncommon));
        }
        _ => panic!("{pending:?}"),
    }
}
