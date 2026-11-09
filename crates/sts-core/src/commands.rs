//! CommunicationMod commands for engine actions and seeds, for watching engine play in the real
//! game (`sts-cli run --save`, the `sts` module's recorded fights, `tools/trace/playback*.py`).

use serde_json::{json, Value};

use crate::combat::{ChoicePile, Combat};

use crate::relics::RelicId;
use crate::run::map::RoomType;
use crate::run::{GridPurpose, NeowDrawback, Run, RunAction, Screen};
use crate::PlayerAction;

/// `SeedHelper.getString`: the seed as the game shows it.
pub fn seed_string(seed: i64) -> String {
    const CHARS: &[u8] = b"0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";
    let mut n = seed as u64;
    let mut out = Vec::new();
    while n != 0 {
        out.push(CHARS[(n % 35) as usize]);
        n /= 35;
    }
    out.reverse();
    String::from_utf8(out).unwrap()
}

/// `SeedHelper.getLong`: parses a game seed string (or a plain number).
pub fn parse_seed(s: &str) -> Option<i64> {
    if let Ok(n) = s.parse::<i64>() {
        return Some(n);
    }
    const CHARS: &str = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";
    let mut total: i64 = 0;
    for ch in s.to_uppercase().replace('O', "0").chars() {
        let d = CHARS.find(ch)? as i64;
        total = total.wrapping_mul(35).wrapping_add(d);
    }
    Some(total)
}

/// The CommunicationMod command(s) that take engine action `a` in the game, from combat `c`
/// (call before `c.step(a)`). A trailing `?` marks a command to send only if the game offers it.
pub fn combat_commands(c: &Combat, a: &PlayerAction) -> Vec<String> {
    let pos = |m: usize| c.order.iter().position(|&o| o == m).unwrap_or(0);
    match a {
        PlayerAction::PlayCard { hand_index, target } => match target {
            Some(t) => vec![format!("play {} {}", hand_index + 1, pos(*t))],
            None => vec![format!("play {}", hand_index + 1)],
        },
        PlayerAction::EndTurn => vec!["end".into()],
        PlayerAction::UsePotion { slot, target } => match target {
            Some(t) => vec![format!("potion use {slot} {}", pos(*t))],
            None => vec![format!("potion use {slot}")],
        },
        PlayerAction::DiscardPotion { slot } => vec![format!("potion discard {slot}")],
        PlayerAction::Choose { indices } => {
            let Some(p) = &c.pending else { return Vec::new() };
            if indices.is_empty() && p.pile == ChoicePile::Offer {
                return vec!["skip".into()];
            }
            let positions: Vec<usize> = indices.iter().filter_map(|i| p.options.iter().position(|o| o == i)).collect();
            // The game keeps the cards in the order they're clicked (`selectedCards`), and the
            // engine takes them in `indices` order, so click them in that order. Picking from the
            // hand removes the card from the list the game indexes: each pick moves down past the
            // earlier ones to its left. The grid screens keep their list.
            let mut out: Vec<String> = positions
                .iter()
                .enumerate()
                .map(|(k, &pos)| {
                    let shift = if p.pile == ChoicePile::Hand { positions[..k].iter().filter(|&&q| q < pos).count() } else { 0 };
                    format!("choose {}", pos - shift)
                })
                .collect();
            // A pick from offered cards (Discovery, Toolbox, the card potions: the game's card
            // reward screen) closes on the pick. A confirm after it could reach the next selection
            // screen and close that with nothing picked (soak 2026-10-27, run GBFK: Toolbox's pick,
            // then Gambling Chip's, at the fast harness speed).
            if p.pile != ChoicePile::Offer {
                out.push("confirm?".into());
            }
            out
        }
    }
}

fn kind(s: &Screen) -> &'static str {
    match s {
        Screen::Neow { .. } => "neow",
        Screen::Map => "map",
        Screen::Combat => "combat",
        Screen::Rewards { .. } => "rewards",
        Screen::CardReward { .. } => "card_reward",
        Screen::Grid { .. } => "grid",
        Screen::Rest => "rest",
        Screen::BossRelic { .. } => "boss_relic",
        Screen::Shop { .. } => "shop",
        Screen::Event { .. } => "event",
        Screen::Unsupported { .. } => "other",
        Screen::Over => "over",
    }
}

/// One run decision as a playback step (`tools/trace/playback.py`): the screen kind, the action
/// and the game commands, or the map x / deck positions where the game's indices depend on its
/// own state. Call before `run.step(a)`.
pub fn run_step(run: &Run, a: &RunAction) -> Value {
    let k = kind(&run.screen);
    let cmds = |v: Vec<String>| json!({ "kind": k, "action": a, "cmds": v });
    match (&run.screen, a) {
        (_, RunAction::UsePotion { slot }) => cmds(vec![format!("potion use {slot}")]),
        (_, RunAction::DiscardPotion { slot }) => cmds(vec![format!("potion discard {slot}")]),
        // A blessing with a curse: the game rolls the curse (cardRng) in `NeowReward.update()` a
        // frame later. Leaving at once let it roll after other cardRng draws (a different curse,
        // 2026-10-16 soak), so give it time first.
        (Screen::Neow { options }, RunAction::Neow { option }) if options.get(*option).is_some_and(|o| o.drawback == NeowDrawback::Curse) => {
            cmds(vec![format!("choose {option}"), "wait 30".into()])
        }
        (Screen::Neow { .. }, RunAction::Neow { option }) => cmds(vec![format!("choose {option}")]),
        (Screen::Map, RunAction::Path { x }) => json!({ "kind": k, "action": a, "path_x": x }),
        (Screen::Combat, RunAction::Combat { action }) => cmds(run.combat.as_ref().map(|c| combat_commands(c, action)).unwrap_or_default()),
        // Cursed Key's curse from a chest is obtained when its card animation ends (0.5 s,
        // `ShowCardAndObtainEffect`), after the relic if playback takes it at once: Darkstone
        // Periapt from that chest then counted the curse (2026-10-18 retest at speed 2). The
        // engine obtains the curse first, so let the animation end.
        (Screen::Rewards { .. }, RunAction::TakeReward { index }) if run.room == Some(RoomType::Treasure) && run.has_relic(RelicId::CursedKey) => {
            cmds(vec!["wait 30".into(), format!("choose {index}")])
        }
        (Screen::Rewards { .. }, RunAction::TakeReward { index }) => cmds(vec![format!("choose {index}")]),
        // A labelled reward screen (Cauldron, Orrery, Tiny House) has no Proceed button; its
        // cancel button goes back to where it opened. CommunicationMod's proceed would run the
        // Proceed button's code anyway, which opens the map.
        (Screen::Rewards { .. }, RunAction::Proceed) if run.rewards_labelled() => cmds(vec!["skip".into()]),
        (Screen::Rewards { .. }, RunAction::Proceed) => cmds(vec!["proceed".into()]),
        (Screen::CardReward { .. }, RunAction::PickCard { index }) => cmds(vec![format!("choose {index}")]),
        (Screen::CardReward { .. }, RunAction::SkipCards) => cmds(vec!["skip".into()]),
        (Screen::CardReward { .. }, RunAction::SingingBowl) => cmds(vec!["choose bowl".into()]),
        (Screen::Grid { purpose, .. }, RunAction::Proceed) if *purpose == GridPurpose::Toke => cmds(vec!["cancel".into()]),
        (Screen::Grid { .. }, RunAction::Select { indices }) => json!({ "kind": k, "action": a, "grid_deck": indices }),
        (Screen::Rest, RunAction::Rest) => cmds(vec!["choose rest".into()]),
        (Screen::Rest, RunAction::Smith) => cmds(vec!["choose smith".into()]),
        (Screen::Rest, RunAction::Lift) => cmds(vec!["choose lift".into()]),
        (Screen::Rest, RunAction::Recall) => cmds(vec!["choose recall".into()]),
        (Screen::Rest, RunAction::Toke) => cmds(vec!["choose toke".into()]),
        (Screen::Rest, RunAction::Dig) => cmds(vec!["choose dig".into()]),
        (Screen::Rest, RunAction::Proceed) => cmds(vec!["proceed".into()]),
        (Screen::BossRelic { .. }, RunAction::BossRelic { index }) => cmds(vec![format!("choose {index}")]),
        // Skipping closes the boss chest again (`BossRelicSelectScreen` calls `chest.close()`), so
        // the chest room offers to open it once more; its Proceed goes on to the next act.
        (Screen::BossRelic { .. }, RunAction::Proceed) => cmds(vec!["skip".into(), "proceed".into()]),
        (Screen::Shop { .. }, RunAction::Proceed) => cmds(vec!["leave?".into(), "proceed".into()]),
        (Screen::Shop { shop }, act) => {
            // getAvailableShopItems: removal (if affordable), cards, relics, potions.
            let mut items = Vec::new();
            if shop.purge_available && run.gold >= shop.purge_cost {
                items.push(RunAction::Purge);
            }
            for (i, c) in shop.cards.iter().enumerate() {
                if c.as_ref().is_some_and(|c| c.price <= run.gold) {
                    items.push(RunAction::BuyCard { index: i });
                }
            }
            for (i, r) in shop.relics.iter().enumerate() {
                if r.as_ref().is_some_and(|r| r.price <= run.gold) {
                    items.push(RunAction::BuyRelic { index: i });
                }
            }
            for (i, p) in shop.potions.iter().enumerate() {
                if p.as_ref().is_some_and(|p| p.price <= run.gold) {
                    items.push(RunAction::BuyPotion { index: i });
                }
            }
            let k2 = items.iter().position(|x| x == act).unwrap_or(0);
            cmds(vec![format!("choose {k2}")])
        }
        // The Library's 20 cards are a grid in the game.
        (Screen::Event { event }, RunAction::EventChoice { index }) if event.id == crate::run::events::EventId::TheLibrary && event.screen == 1 => {
            json!({ "kind": "grid", "action": a, "cmds": [format!("choose {index}"), "confirm?"] })
        }
        // Wheel of Change's spin is two clicks in the game: the intro option hides the dialog, then
        // the wheel's own button (CommunicationMod's only choice, "spin") starts it.
        (Screen::Event { event }, RunAction::EventChoice { .. }) if event.id == crate::run::events::EventId::WheelOfChange && event.screen == 0 => {
            cmds(vec!["choose 0".into(), "choose 0".into()])
        }
        // Match and Keep's cards: CommunicationMod lists the face-down ones by board place, not in
        // the engine's (shuffled) order.
        (Screen::Event { event }, RunAction::EventChoice { index }) if event.id == crate::run::events::EventId::MatchAndKeep && event.screen == 2 && event.board.len() == event.cards.len() => {
            let at = event.board[*index];
            let before = (0..event.cards.len()).filter(|&j| event.options[j] && event.board[j] < at).count();
            cmds(vec![format!("choose {before}")])
        }
        (Screen::Event { event }, RunAction::EventChoice { index }) => {
            let enabled = event.options[..*index].iter().filter(|&&o| o).count();
            cmds(vec![format!("choose {enabled}")])
        }
        _ => cmds(vec!["proceed".into()]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relics::RelicId;
    use crate::{Combat, PlayerAction};
    use crate::run::events::{Event, EventId};
    use crate::run::RunConfig;

    fn commands(run: &Run, a: &RunAction) -> Vec<String> {
        serde_json::from_value(run_step(run, a)["cmds"].clone()).unwrap()
    }

    #[test]
    fn a_chest_with_cursed_key_waits_for_the_curse() {
        let mut run = Run::new(RunConfig::new(1));
        run.screen = Screen::Rewards { items: Vec::new() };
        run.room = Some(RoomType::Treasure);
        assert_eq!(commands(&run, &RunAction::TakeReward { index: 0 }), ["choose 0"]);
        run.obtain_relic(RelicId::CursedKey);
        assert_eq!(commands(&run, &RunAction::TakeReward { index: 0 }), ["wait 30", "choose 0"]);
        run.room = Some(RoomType::Monster);
        assert_eq!(commands(&run, &RunAction::TakeReward { index: 1 }), ["choose 1"]);
    }

    #[test]
    fn skipping_the_boss_relic_then_proceeds_from_the_closed_chest() {
        let mut run = Run::new(RunConfig::new(1));
        run.screen = Screen::BossRelic { relics: vec![RelicId::Astrolabe, RelicId::BlackStar, RelicId::CallingBell] };
        assert_eq!(commands(&run, &RunAction::Proceed), ["skip", "proceed"]);
        assert_eq!(commands(&run, &RunAction::BossRelic { index: 1 }), ["choose 1"]);
    }

    #[test]
    fn a_labelled_reward_screen_leaves_by_cancel() {
        for relic in [RelicId::Cauldron, RelicId::Orrery, RelicId::TinyHouse] {
            let mut run = Run::new(RunConfig::new(1));
            run.obtain_relic(relic);
            assert!(matches!(run.screen, Screen::Rewards { .. }), "{relic:?}");
            assert_eq!(commands(&run, &RunAction::Proceed), ["skip"], "{relic:?}");
            run.step(&RunAction::Proceed).unwrap();
            assert!(!run.rewards_labelled(), "{relic:?}");
        }
        // Calling Bell's screen is opened without a label: it keeps Proceed.
        let mut run = Run::new(RunConfig::new(1));
        run.obtain_relic(RelicId::CallingBell);
        assert_eq!(commands(&run, &RunAction::Proceed), ["proceed"]);
    }

    #[test]
    fn orrery_stays_labelled_across_its_card_rewards() {
        let mut run = Run::new(RunConfig::new(1));
        run.obtain_relic(RelicId::Orrery);
        run.step(&RunAction::TakeReward { index: 0 }).unwrap();
        run.step(&RunAction::SkipCards).unwrap();
        assert!(matches!(run.screen, Screen::Rewards { .. }));
        assert_eq!(commands(&run, &RunAction::Proceed), ["skip"]);
    }

    /// The game takes hand selections in click order: click in the engine's order, each index
    /// shifted past the earlier picks that already left the hand (soak run GBQ5: Gambling Chip's
    /// five discards went in reversed, and the next reshuffle dealt another hand).
    #[test]
    fn hand_selections_are_clicked_in_the_engines_order() {
        let mut cfg = crate::combat::CombatConfig::ironclad(1, "Lagavulin");
        cfg.relics.push(RelicId::GamblingChip);
        let c = Combat::new(&cfg).unwrap();
        let p = c.pending.as_ref().expect("Gambling Chip's choice");
        assert_eq!(p.pile, ChoicePile::Hand);
        let pick = |indices: Vec<usize>| combat_commands(&c, &PlayerAction::Choose { indices });
        assert_eq!(pick(vec![0, 1, 2, 3, 4]), ["choose 0", "choose 0", "choose 0", "choose 0", "choose 0", "confirm?"]);
        assert_eq!(pick(vec![3, 1]), ["choose 3", "choose 1", "confirm?"]);
        assert_eq!(pick(vec![1, 3]), ["choose 1", "choose 2", "confirm?"]);
    }

    /// Toolbox's pick (offered cards) takes no confirm: the next choice, Gambling Chip's, must stay
    /// open (soak run GBFK).
    #[test]
    fn a_pick_from_offered_cards_takes_no_confirm() {
        let mut cfg = crate::combat::CombatConfig::ironclad(1, "Lagavulin");
        cfg.relics.push(RelicId::Toolbox);
        cfg.relics.push(RelicId::GamblingChip);
        let c = Combat::new(&cfg).unwrap();
        let p = c.pending.as_ref().expect("Toolbox's choice");
        assert_eq!(p.pile, ChoicePile::Offer);
        assert_eq!(combat_commands(&c, &PlayerAction::Choose { indices: vec![p.options[0]] }), ["choose 0"]);
    }

    /// CommunicationMod lists Match and Keep's face-down cards by board place (column i % 4, row
    /// i % 3 of the card's index when the board was laid out), not in the engine's order.
    #[test]
    fn match_and_keep_cards_are_chosen_by_board_place() {
        let mut run = Run::new(RunConfig::new(1));
        let mut e = Event::new(EventId::MatchAndKeep);
        e.screen = 2;
        e.cards = (0..12).map(|_| run.new_card(crate::cards::CardId::StrikeR)).collect();
        e.board = (0..12).map(|i| (i % 4 + 4 * (i % 3)) as u8).collect();
        e.options = vec![true; 12];
        run.screen = Screen::Event { event: Box::new(e.clone()) };
        // Places of indices 0..11: [0, 5, 10, 3, 4, 9, 2, 7, 8, 1, 6, 11].
        assert_eq!(commands(&run, &RunAction::EventChoice { index: 1 }), ["choose 5"]);
        assert_eq!(commands(&run, &RunAction::EventChoice { index: 9 }), ["choose 1"]);
        // A turned card (index 9, place 1) leaves the list: places after it move down.
        e.options[9] = false;
        run.screen = Screen::Event { event: Box::new(e.clone()) };
        assert_eq!(commands(&run, &RunAction::EventChoice { index: 1 }), ["choose 4"]);
        // A matched pair (indices 0 and 4, places 0 and 4) leaves the board.
        e.options[9] = true;
        for k in [4, 0] {
            e.cards.remove(k);
            e.board.remove(k);
            e.options.remove(k);
        }
        run.screen = Screen::Event { event: Box::new(e) };
        assert_eq!(commands(&run, &RunAction::EventChoice { index: 0 }), ["choose 3"]);
    }

    #[test]
    fn wheel_of_change_spin_is_two_clicks() {
        let mut run = Run::new(RunConfig::new(1));
        let mut e = Event::new(EventId::WheelOfChange);
        e.options = vec![true];
        run.screen = Screen::Event { event: Box::new(e.clone()) };
        assert_eq!(commands(&run, &RunAction::EventChoice { index: 0 }), ["choose 0", "choose 0"]);
        e.screen = 1;
        run.screen = Screen::Event { event: Box::new(e) };
        assert_eq!(commands(&run, &RunAction::EventChoice { index: 0 }), ["choose 0"]);
    }
}
