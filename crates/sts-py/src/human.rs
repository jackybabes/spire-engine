//! Human decisions as network inputs: a decision record from `python/human/human_runs.py` (the state
//! before a choice and what the player took) becomes a run in that state, on that screen,
//! encoded as in training, with the action slot the player took. The run is built directly
//! (deck, relics, HP, gold, floor, act, room, boss), without running card or relic effects.

use serde_json::Value;
use sts_core::cards::{Card, CardId, CARD_DATA};
use sts_core::relics::{RelicId, RelicState, RELIC_DATA};
use sts_core::run::events::EventId;
use sts_core::run::map::RoomType;
use sts_core::run::{GridPurpose, Run, RunAction, RunConfig, Screen};

use crate::encode::{CONFIRM, PICK};
use crate::encode_run::{encode_run, options, OptAction};

fn card_id(java_id: &str) -> Option<CardId> {
    CARD_DATA.iter().find(|c| c.java_id == java_id).map(|c| c.id)
}

fn relic_id(java_id: &str) -> Option<RelicId> {
    RELIC_DATA.iter().find(|r| r.java_id == java_id).map(|r| r.id)
}

/// "Bash+1" -> Bash, upgraded once.
fn card(run: &mut Run, s: &str) -> Result<Card, String> {
    let (name, up) = match s.split_once('+') {
        Some((n, u)) => (n, u.parse::<u32>().unwrap_or(0)),
        None => (s, 0),
    };
    let id = card_id(name).ok_or_else(|| format!("unknown card {name}"))?;
    let mut c = run.new_card(id);
    for _ in 0..up {
        c.upgrade();
    }
    Ok(c)
}

fn room(code: &str) -> Option<RoomType> {
    Some(match code {
        "M" => RoomType::Monster,
        "E" => RoomType::Elite,
        "?" => RoomType::Event,
        "R" => RoomType::Rest,
        "$" => RoomType::Shop,
        "T" => RoomType::Treasure,
        "B" => RoomType::Boss,
        _ => return None,
    })
}

fn int(v: &Value, k: &str) -> Option<i32> {
    v[k].as_i64().or_else(|| v[k].as_f64().map(|f| f as i64)).map(|x| x as i32)
}

/// The run in the record's state, on the map (the screen is set per sample).
fn base(rec: &Value) -> Result<Run, String> {
    let mut config = RunConfig::new(1);
    config.ascension = int(rec, "ascension").unwrap_or(0);
    // The record's character (human_runs.py writes the dump's name, e.g. THE_SILENT); the Ironclad
    // for records from before it was written.
    if let Some(c) = rec["character"].as_str() {
        config.character = sts_core::character::Character::from_name(c).ok_or_else(|| format!("unknown character {c}"))?;
    }
    let mut run = Run::new(config);
    run.act = int(rec, "act").unwrap_or(1);
    run.floor = int(rec, "floor").unwrap_or(0);
    run.max_hp = int(rec, "max_hp").unwrap_or(80).max(1);
    run.hp = int(rec, "hp").unwrap_or(run.max_hp).clamp(1, run.max_hp);
    run.gold = int(rec, "gold").unwrap_or(0).max(0);
    run.room = rec["room"].as_str().and_then(room);
    if let Some(b) = rec["boss"].as_str() {
        run.boss = b.to_string();
    }
    run.keys.ruby = rec["ruby_key"].as_bool().unwrap_or(false);
    run.deck.clear();
    for c in rec["deck"].as_array().ok_or("no deck")? {
        let c = card(&mut run, c.as_str().ok_or("bad card")?)?;
        run.deck.push(c);
    }
    run.relics.clear();
    for r in rec["relics"].as_array().ok_or("no relics")? {
        let r = r.as_str().ok_or("bad relic")?;
        let id = relic_id(r).ok_or_else(|| format!("unknown relic {r}"))?;
        // A boss relic that replaces the starter takes its slot (records from before 2026-10-27
        // list both: human_runs.gain_relic kept the starter).
        match id.replaces_starter().and_then(|old| run.relics.iter().position(|x| x.id == old)) {
            Some(i) => run.relics[i] = RelicState::new(id),
            None => run.relics.push(RelicState::new(id)),
        }
    }
    Ok(run)
}

/// What the player took: an action, or a deck card picked on a grid.
#[derive(Clone, Debug)]
pub enum Target {
    Action(RunAction),
    Grid(usize),
}

/// The deck card a grid choice names (the least upgraded copy among the grid's options).
fn deck_index(run: &Run, options: &[usize], s: &str) -> Option<usize> {
    let name = s.split('+').next()?;
    let id = card_id(name)?;
    options.iter().copied().filter(|&d| run.deck[d].id == id).min_by_key(|&d| run.deck[d].times_upgraded)
}

/// The engine options a human event choice (the dump's `player_choice`) took, in order: each a
/// label prefix on the event's next screen with a real choice (screens with one option, such as
/// "continue", are stepped through), or `#k` for the k-th option. None for a choice the engine
/// can't follow (made after a fight, or not recorded: the Colosseum, Bonfire, Knowing Skull...).
fn event_path(id: EventId, taken: &str, rec: &Value) -> Option<Vec<String>> {
    use EventId::*;
    let p: &[&str] = match (id, taken) {
        (BigFish, "Banana") => &["banana"],
        (BigFish, "Donut") => &["donut"],
        (BigFish, "Box") => &["box"],
        (Cleric, "Healed") => &["heal"],
        (Cleric, "Card Removal") => &["purify"],
        (Cleric, "Leave") => &["leave"],
        (DeadAdventurer, "Searched '0' times") => &["leave"],
        (DeadAdventurer, t) if t.starts_with("Searched") => &["search"],
        (GoldenIdol, "Ignored") => &["leave"],
        (GoldenIdol, "Take Wound") => &["take", "outrun"],
        (GoldenIdol, "Take Damage") => &["take", "smash"],
        (GoldenIdol, "Lose Max HP") => &["take", "hide"],
        (GoldenWing, "Card Removal") => &["pray"],
        (GoldenWing, "Gained Gold") => &["destroy"],
        (GoldenWing, "Ignored") => &["leave"],
        (WorldOfGoop, "Gather Gold") => &["gather"],
        (WorldOfGoop, "Left Gold") => &["leave"],
        (LiarsGame, "AGREE") => &["agree"],
        (LiarsGame, "Ignored") => &["disagree"],
        (LivingWall, "Forget") => &["forget"],
        (LivingWall, "Change") => &["change"],
        (LivingWall, "Grow") => &["grow"],
        (Mushrooms, "Fought Mushrooms") => &["stomp"],
        (Mushrooms, "Healed and dodged fight") => &["eat"],
        (ScrapOoze, "Success") => &["reach"],
        (ShiningLight, "Entered Light") => &["enter"],
        (ShiningLight, "Ignored") => &["leave"],
        (GoldenShrine, "Pray") => &["pray"],
        (GoldenShrine, "Desecrate") => &["desecrate"],
        (GoldenShrine, "Ignored") => &["leave"],
        (Transmogrifier, "Transformed") => &["pray"],
        (Transmogrifier, "Ignored") => &["leave"],
        (Purifier, "Purged") => &["pray"],
        (Purifier, "Ignored") => &["leave"],
        (UpgradeShrine, "Upgraded") => &["pray"],
        (UpgradeShrine, "Ignored") => &["leave"],
        (AccursedBlacksmith, "Forge") => &["forge"],
        (AccursedBlacksmith, "Rummage") => &["rummage"],
        (AccursedBlacksmith, "Ignored") => &["leave"],
        (FountainOfCleansing, "Removed Curses") => &["drink"],
        (FountainOfCleansing, "Ignored") => &["leave"],
        (FaceTrader, "Trade") => &["trade"],
        (FaceTrader, "Touch") => &["touch"],
        (FaceTrader, "Leave") => &["leave"],
        (NoteForYourself, "Took Card") => &["take"],
        (NoteForYourself, "Ignored") => &["leave"],
        (WeMeetAgain, "Gave Potion") => &["give a potion"],
        (WeMeetAgain, "Paid Gold") => &["give gold"],
        (WeMeetAgain, "Gave Card") => &["give a card"],
        (WeMeetAgain, "Ignored") => &["attack"],
        (WomanInBlue, "Bought 0 Potions") => &["leave"],
        (WomanInBlue, "Bought 1 Potion") => &["20 gold"],
        (WomanInBlue, "Bought 2 Potions") => &["30 gold"],
        (WomanInBlue, "Bought 3 Potions") => &["40 gold"],
        // Act 2.
        (Addict, "Obtained Relic") => &["pay"],
        (Addict, "Stole Relic") => &["rob"],
        (Addict, "Ignored") => &["leave"],
        (BackToBasics, "Elegance") => &["elegance"],
        (BackToBasics, "Simplicity") => &["simplicity"],
        (Beggar, "Gave Gold") => &["give"],
        (Beggar, "Ignored") => &["leave"],
        (CursedTome, "Obtained Book") => &["read", "take"],
        (CursedTome, "Stopped") => &["read", "stop"],
        (CursedTome, "Ignored") => &["leave"],
        (DrugDealer, "Obtain J.A.X.") => &["obtain"],
        (DrugDealer, "Became Test Subject") => &["transform"],
        (DrugDealer, "Inject Mutagens") => &["Mutagenic"],
        (ForgottenAltar, "Gave Idol") => &["offer"],
        (ForgottenAltar, "Shed Blood") => &["sacrifice"],
        (ForgottenAltar, "Smashed Altar") => &["desecrate"],
        (Ghosts, "Became a Ghost") => &["accept"],
        (Ghosts, "Ignored") => &["refuse"],
        (MaskedBandits, "Paid Fearfully") => &["pay"],
        (MaskedBandits, "Fought Bandits") => &["fight"],
        (Nest, "Stole From Cult") => &["smash"],
        (Nest, "Joined the Cult") => &["stay"],
        (TheLibrary, "Read") => &["read"],
        (TheLibrary, "Heal") => &["sleep"],
        (TheMausoleum, "Opened") => &["open"],
        (TheMausoleum, "Ignored") => &["leave"],
        (Vampires, "Became a vampire") => &["accept"],
        (Vampires, "Became a vampire (Vial)") => &["give Blood Vial"],
        (Vampires, "Ignored") => &["refuse"],
        (Designer, "Upgrade") => &["adjustments: upgrade a card"],
        (Designer, "Upgraded Two") => &["adjustments: upgrade 2"],
        (Designer, "Single Remove") => &["clean up: remove"],
        (Designer, "Transformed Cards") => &["clean up: transform"],
        (Designer, "Upgrade and Remove") => &["full service"],
        (Designer, "Punched") => &["punch"],
        (Duplicator, "Copied") => &["pray"],
        (Duplicator, "Ignored") => &["leave"],
        (TheJoust, "Bet on Murderer") => &["bet on the murderer"],
        (TheJoust, "Bet on Owner") => &["bet on the owner"],
        (Nloth, "Ignored") => &["leave"],
        (Nloth, "Traded Relic") => {
            // The relic given, by its name on the option ("trade Shovel").
            let lost = rec["relics_lost"].get(0)?.as_str()?;
            return Some(vec![format!("trade {}", relic_id(lost)?.data().name)]);
        }
        // Act 3.
        (Falling, "Removed Skill") => &["#0"],
        (Falling, "Removed Power") => &["#1"],
        (Falling, "Removed Attack") => &["#2"],
        (MindBloom, "Fight") => &["I am War"],
        (MindBloom, "Upgrade") => &["I am Awake"],
        (MindBloom, "Gold") => &["I am Rich"],
        (MindBloom, "Heal") => &["I am Healthy"],
        (MoaiHead, "Heal") => &["jump in"],
        (MoaiHead, "Gave Idol") => &["offer"],
        (MoaiHead, "Ignored") => &["leave"],
        (MysteriousSphere, "Fight") => &["open"],
        (MysteriousSphere, "Ignored") => &["leave"],
        (SecretPortal, "Took Portal") => &["enter"],
        (SecretPortal, "Ignored") => &["leave"],
        (SensoryStone, "Memory 1") => &["recall: 1"],
        (SensoryStone, "Memory 2") => &["recall: 2"],
        (SensoryStone, "Memory 3") => &["recall: 3"],
        (TombRedMask, "Wore Mask") => &["don"],
        (TombRedMask, "Paid") => &["pay"],
        (TombRedMask, "Ignored") => &["leave"],
        (WindingHalls, "Embrace Madness") => &["embrace"],
        (WindingHalls, "Writhe") => &["retrace"],
        (WindingHalls, "Max HP") => &["press on"],
        _ => return None,
    };
    Some(p.iter().map(|s| s.to_string()).collect())
}

/// The samples of a human event choice: the run on each of the event's screens with a real choice
/// along `path`, and the option taken there (the engine is stepped through the event between them).
fn event_samples(mut run: Run, id: EventId, path: &[String]) -> Result<Vec<(Run, Target)>, String> {
    run.start_event(id);
    let mut out = Vec::new();
    for (i, want) in path.iter().enumerate() {
        // Screens with a single option ("continue", a page of a book) need no decision.
        let mut steps = 0;
        let (k, choice) = loop {
            let Screen::Event { event } = &run.screen else { return Err("the event ended before the choice".into()) };
            let open: Vec<usize> = (0..event.options.len()).filter(|&k| event.options[k]).collect();
            if open.len() == 1 && !event.labels[open[0]].starts_with(want.as_str()) {
                steps += 1;
                if steps > 8 {
                    return Err("too many screens without a choice".into());
                }
                run.step(&RunAction::EventChoice { index: open[0] })?;
                continue;
            }
            let k = match want.strip_prefix('#') {
                Some(n) => n.parse::<usize>().ok().filter(|&k| event.options.get(k) == Some(&true)),
                None => (0..event.labels.len()).find(|&k| event.options[k] && event.labels[k].starts_with(want.as_str())),
            };
            break (k.ok_or_else(|| format!("{}: no option {want:?} in {:?}", id.key(), event.labels))?, open.len() > 1);
        };
        if choice {
            out.push((run.clone(), Target::Action(RunAction::EventChoice { index: k })));
        }
        if i + 1 < path.len() {
            run.step(&RunAction::EventChoice { index: k })?;
        }
    }
    Ok(out)
}

/// The record's samples: the run on the decision's screen, and what the player took. A rest
/// site's smith or toke adds a second sample, the card chosen on its grid.
pub fn samples(rec: &Value) -> Result<Vec<(Run, Target)>, String> {
    let mut run = base(rec)?;
    let taken = rec["taken"].as_str().unwrap_or("");
    let mut out = Vec::new();
    match rec["kind"].as_str().unwrap_or("") {
        "card" => {
            let mut cards = Vec::new();
            for c in rec["offered"].as_array().ok_or("no offer")? {
                cards.push(card(&mut run, c.as_str().ok_or("bad card")?)?);
            }
            let target = match taken {
                "SKIP" => RunAction::SkipCards,
                "Singing Bowl" => RunAction::SingingBowl,
                t => {
                    let want = card(&mut run, t)?;
                    let i = cards.iter().position(|c| c.id == want.id && c.times_upgraded == want.times_upgraded)
                        .or_else(|| cards.iter().position(|c| c.id == want.id))
                        .ok_or("taken card not offered")?;
                    RunAction::PickCard { index: i }
                }
            };
            run.screen = Screen::CardReward { item: None, cards };
            out.push((run, Target::Action(target)));
        }
        "campfire" => {
            run.room = Some(RoomType::Rest);
            let action = match taken {
                "REST" => RunAction::Rest,
                "SMITH" => RunAction::Smith,
                "LIFT" => RunAction::Lift,
                "PURGE" => RunAction::Toke,
                "DIG" => RunAction::Dig,
                "RECALL" => RunAction::Recall,
                other => return Err(format!("unknown rest-site choice {other}")),
            };
            run.screen = Screen::Rest;
            let grid = rec["card"].as_str().map(str::to_string);
            let follow = run.clone();
            out.push((run, Target::Action(action.clone())));
            if let (Some(c), RunAction::Smith | RunAction::Toke) = (grid, &action) {
                let mut run = follow;
                let (purpose, opts): (GridPurpose, Vec<usize>) = if action == RunAction::Smith {
                    (GridPurpose::Smith, (0..run.deck.len()).filter(|&d| run.deck[d].can_upgrade()).collect())
                } else {
                    (GridPurpose::Toke, (0..run.deck.len()).filter(|&d| !run.deck[d].bottled).collect())
                };
                if let Some(d) = deck_index(&run, &opts, &c) {
                    run.screen = Screen::Grid { purpose, options: opts, count: 1 };
                    out.push((run, Target::Grid(d)));
                }
            }
        }
        "boss" => {
            run.room = Some(RoomType::Treasure);
            let mut relics = Vec::new();
            for r in rec["offered"].as_array().ok_or("no offer")? {
                let r = r.as_str().ok_or("bad relic")?;
                relics.push(relic_id(r).ok_or_else(|| format!("unknown relic {r}"))?);
            }
            let target = if taken == "SKIP" {
                RunAction::Proceed
            } else {
                let id = relic_id(taken).ok_or_else(|| format!("unknown relic {taken}"))?;
                RunAction::BossRelic { index: relics.iter().position(|&r| r == id).ok_or("taken relic not offered")? }
            };
            run.screen = Screen::BossRelic { relics };
            out.push((run, Target::Action(target)));
        }
        "purge" => {
            run.room = Some(RoomType::Shop);
            let opts: Vec<usize> = (0..run.deck.len()).filter(|&d| !run.deck[d].bottled).collect();
            let d = deck_index(&run, &opts, taken).ok_or("removed card not in deck")?;
            run.screen = Screen::Grid { purpose: GridPurpose::ShopPurge, options: opts, count: 1 };
            out.push((run, Target::Grid(d)));
        }
        "event" => {
            let name = rec["event"].as_str().ok_or("no event")?;
            let id = EventId::from_key(name).ok_or_else(|| format!("unknown event {name}"))?;
            let path = event_path(id, taken, rec).ok_or_else(|| format!("{name}: choice {taken:?} not followed"))?;
            out.extend(event_samples(run, id, &path)?);
        }
        other => return Err(format!("unknown decision kind {other}")),
    }
    Ok(out)
}

/// Encodes `run` into the row slices as `Envs.observe_arrays` does and returns the action slot
/// of `target`, or None if that isn't an allowed slot there (the rebuilt state doesn't allow it).
pub fn encode(run: &Run, target: &Target, ints: &mut [i32], floats: &mut [f32], mask: &mut [u8]) -> Option<usize> {
    let legal = run.legal_actions();
    encode_run(run, &legal, &[], &[], ints, floats, mask);
    let opts = options(run, &[], &[]);
    let slot = match target {
        Target::Action(RunAction::SkipCards | RunAction::Proceed) => CONFIRM,
        Target::Action(a) => PICK + opts.iter().position(|o| matches!(&o.action, OptAction::Run(x) if x == a))?,
        Target::Grid(d) => PICK + opts.iter().position(|o| matches!(o.action, OptAction::GridPick(x) if x == *d))?,
    };
    (mask.get(slot) == Some(&1)).then_some(slot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rec(event: &str, taken: &str, act: i32) -> Value {
        json!({"kind": "event", "event": event, "taken": taken, "character": "DEFECT", "ascension": 0, "act": act,
               "floor": 17 * act - 10, "hp": 60, "max_hp": 75, "gold": 120, "room": "?",
               "deck": ["Strike_B", "Strike_B", "Defend_B", "Zap", "Dualcast", "Coolheaded", "Ball Lightning+1"],
               "relics": ["Cracked Core", "Anchor", "Vajra"]})
    }

    fn picks(r: &Value) -> Vec<String> {
        samples(r).unwrap().iter().map(|(run, t)| match (&run.screen, t) {
            (Screen::Event { event }, Target::Action(RunAction::EventChoice { index })) => event.labels[*index].clone(),
            _ => panic!("not an event choice"),
        }).collect()
    }

    /// Each followed choice gives the screens with a real choice, through single-option screens;
    /// every sample's taken option is an allowed action slot.
    /// A boss relic that replaces the starter takes its slot (older records list both).
    #[test]
    fn a_replacing_boss_relic_takes_the_starters_slot() {
        let mut r = rec("Golden Idol", "Ignored", 1);
        r["relics"] = json!(["Cracked Core", "Anchor", "FrozenCore"]);
        let run = base(&r).unwrap();
        assert_eq!(run.relics.iter().map(|x| x.id).collect::<Vec<_>>(), [RelicId::FrozenCore, RelicId::Anchor]);
    }

    #[test]
    fn human_event_choices() {
        assert_eq!(picks(&rec("Golden Idol", "Take Wound", 1)), ["take the idol", "outrun: Injury"]);
        assert_eq!(picks(&rec("Golden Idol", "Ignored", 1)), ["leave"]);
        assert_eq!(picks(&rec("Cursed Tome", "Obtained Book", 2)).len(), 2);
        assert!(picks(&rec("Cursed Tome", "Stopped", 2))[1].starts_with("stop reading"));
        assert!(picks(&rec("Nest", "Stole From Cult", 2))[0].starts_with("smash and grab"));
        assert_eq!(picks(&rec("Masked Bandits", "Fought Bandits", 2)), ["fight"]);
        assert_eq!(picks(&rec("Mysterious Sphere", "Fight", 3)), ["open the sphere: fight"]);
        assert!(picks(&rec("Winding Halls", "Max HP", 3))[0].starts_with("press on"));
        assert_eq!(picks(&rec("FaceTrader", "Trade", 1)), ["trade: a face"]);
        assert_eq!(picks(&rec("Falling", "Removed Attack", 3)).len(), 1);
        let mut n = rec("N'loth", "Traded Relic", 2);
        n["relics_lost"] = json!(["Anchor"]);
        n["relics"] = json!(["Anchor", "Vajra"]);
        assert_eq!(picks(&n), ["trade Anchor"]);
        // Choices the engine can't follow, and unknown events, are errors (the record is left out).
        assert!(samples(&rec("Colosseum", "Fight", 2)).is_err());
        assert!(samples(&rec("No Such Event", "Leave", 1)).is_err());
        for r in [rec("Golden Idol", "Lose Max HP", 1), rec("World of Goop", "Gather Gold", 1), rec("The Library", "Read", 2)] {
            for (run, t) in samples(&r).unwrap() {
                let n = crate::encode_run::run_int_len() + crate::encode::int_len();
                let f = crate::encode_run::run_float_len() + crate::encode::float_len();
                let (mut i, mut fl, mut m) = (vec![0; n], vec![0.0; f], vec![0; crate::encode::ACTIONS]);
                assert!(encode(&run, &t, &mut i, &mut fl, &mut m).is_some());
            }
        }
    }
}
