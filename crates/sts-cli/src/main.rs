//! Command-line driver for the headless engine.
//!
//! Modes (`--character ironclad|silent|...` picks who plays; the Ironclad by default):
//!   sts-cli serve                       JSON-lines protocol on stdin/stdout (for training code)
//!   sts-cli play  [--seed N] [--encounter NAME] [--ascension A] [--character C]   interactive text mode
//!   sts-cli run   [--seed N|SEEDSTRING] [--ascension A] [--character C] [--mini-neow] [--save run.json]
//!                                       play a whole run interactively (--save: watch it
//!                                       later in the real game with tools/trace/playback.py)
//!   sts-cli bench-runs [--games N] [--seed N] [--character C]     random-agent runs benchmark
//!   sts-cli bench [--games N] [--seed N] [--encounter NAME|all] [--random-decks] [--character C]   random-agent benchmark
//!   sts-cli replay <trace.jsonl...>     diff recorded game traces against the engine
//!   sts-cli replay-run <run.jsonl...>   the same for whole recorded runs (tools/trace/record_run.py)
//!   sts-cli commands < runs.json > out.json   work out a saved run's (or a soak batch's) game
//!                                       commands again with the current engine
//!   sts-cli coverage <recordings, soak folders...>   what the recordings that match the game
//!                                       exercised, item by item (docs/checklist's real-game boxes)
//!   sts-cli encounters                  list encounter names

mod coverage;
mod replay;
mod run_replay;
mod runlog;

use std::io::{self, BufRead, Write};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};
use sts_core::cards::CardId;
use sts_core::character::Character;
use sts_core::encounters;
use sts_core::relics::RelicId as Relic;
use sts_core::rng::StsRandom;
use sts_core::run::{Run, RunAction, RunConfig};
use sts_core::{Combat, CombatConfig, PlayerAction};

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(|s| s.as_str()).unwrap_or("help");
    let result = match mode {
        "serve" => serve(),
        "play" => play(&args),
        "run" => run_cmd(&args),
        "bench-runs" => bench_runs(&args),
        "bench" => bench(&args),
        "replay" => replay_cmd(&args[1..]),
        "replay-run" => replay_run_cmd(&args[1..]),
        "commands" => commands_cmd(),
        "coverage" => coverage::coverage_cmd(&args[1..]),
        "encounters" => {
            for e in encounters::all() {
                println!("{e}");
            }
            Ok(())
        }
        _ => {
            eprintln!("usage: sts-cli <serve|play|run|bench|bench-runs|replay|encounters> [--seed N] [--encounter NAME] [--ascension A] [--games N]");
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `sts-cli commands`: reads a run saved with `run --save`, or a soak batch (`{"runs": [{"log":
/// run, ...}]}`, python/watch/soak_runs.py), from stdin, and writes it back with every step's game
/// commands, floor and HP worked out by the current engine (`runlog::recommand`), and each batch
/// run's `engine` result updated. Runs whose floor or HP changed since they were saved are
/// reported. A batch run whose decisions the engine can't take any more keeps its place (the
/// soak numbers runs by position) with an `engine_changed` note; a single run fails instead.
fn commands_cmd() -> Result<(), String> {
    let mut v: Value = serde_json::from_reader(io::stdin().lock()).map_err(|e| format!("stdin: {e}"))?;
    if let Some(runs) = v.get_mut("runs").and_then(|r| r.as_array_mut()) {
        let (mut changed, mut unplayable) = (0, 0);
        for (i, r) in runs.iter_mut().enumerate() {
            match runlog::recommand(&r["log"]) {
                Ok((log, end, diff)) => {
                    if let Some(d) = diff {
                        eprintln!("run {i} ({}): the engine plays it differently now, {d}", r["log"]["seed_string"]);
                        changed += 1;
                    }
                    r["log"] = log;
                    r["engine"] = end;
                }
                Err(e) => {
                    eprintln!("run {i} ({}): can't be played: {e}", r["log"]["seed_string"]);
                    r["engine_changed"] = json!(e);
                    unplayable += 1;
                }
            }
        }
        eprintln!("{} runs: {changed} played differently by the engine now, {unplayable} can't be played", runs.len());
    } else {
        let (log, _, diff) = runlog::recommand(&v)?;
        if let Some(d) = diff {
            eprintln!("the engine plays this run differently now, {d}");
        }
        v = log;
    }
    serde_json::to_writer(io::stdout().lock(), &v).map_err(|e| e.to_string())
}

/// `{"cmd": "reset", ...}` fields. Anything omitted falls back to the character's starting state
/// (the Ironclad's by default).
#[derive(Deserialize, Default)]
struct ResetArgs {
    /// "ironclad", "silent", "defect" or "watcher".
    character: Option<String>,
    seed: Option<i64>,
    encounter: Option<String>,
    ascension: Option<i32>,
    /// Card names or Java IDs; a trailing `+` (or `+N`) means upgraded.
    deck: Option<Vec<String>>,
    hp: Option<i32>,
    max_hp: Option<i32>,
    relics: Option<Vec<String>>,
    /// Potion ids or names, one per slot.
    potions: Option<Vec<String>>,
    floor: Option<i32>,
}

fn parse_card(spec: &str) -> Result<(CardId, i32), String> {
    let (name, ups) = match spec.rfind('+') {
        Some(i) => {
            let n = &spec[i + 1..];
            (&spec[..i], if n.is_empty() { 1 } else { n.parse().map_err(|_| format!("bad upgrade count in '{spec}'"))? })
        }
        None => (spec, 0),
    };
    let id = CardId::from_name(name.trim()).ok_or_else(|| format!("unknown card '{name}'"))?;
    Ok((id, ups))
}

fn character(name: Option<&str>) -> Result<Character, String> {
    match name {
        None => Ok(Character::Ironclad),
        Some(n) => Character::from_name(n).ok_or_else(|| format!("unknown character '{n}'")),
    }
}

fn config_from(r: ResetArgs) -> Result<CombatConfig, String> {
    let ch = character(r.character.as_deref())?;
    let mut c = CombatConfig::new(ch, r.seed.unwrap_or(0), r.encounter.as_deref().unwrap_or("Jaw Worm"));
    if let Some(a) = r.ascension {
        c.ascension = a;
    }
    if let Some(deck) = r.deck {
        c.deck = deck.iter().map(|s| parse_card(s)).collect::<Result<_, _>>()?;
    }
    if let Some(hp) = r.max_hp {
        c.player_max_hp = hp;
        c.player_hp = hp;
    }
    if let Some(hp) = r.hp {
        c.player_hp = hp;
    }
    if let Some(rs) = r.relics {
        c.relics = rs.iter().map(|s| Relic::from_name(s).ok_or(format!("unknown relic '{s}'"))).collect::<Result<_, _>>()?;
    }
    if let Some(ps) = r.potions {
        let mut slots: Vec<Option<sts_core::potions::PotionId>> =
            ps.iter().map(|p| sts_core::potions::PotionId::from_name(p).ok_or(format!("unknown potion '{p}'")).map(Some)).collect::<Result<_, _>>()?;
        slots.resize(slots.len().max(3), None);
        c.potions = slots;
    }
    if let Some(f) = r.floor {
        c.floor = f;
    }
    Ok(c)
}

/// JSON-lines protocol. Each input line is one command; each output line is one JSON object.
///
/// Commands:
///   {"cmd":"reset", "seed":1, "encounter":"Jaw Worm", "ascension":0, "deck":["Strike","Bash+"], "hp":80, "character":"ironclad"}
///   {"cmd":"new_run", "seed":1, "ascension":0, "neow_full":true, "character":"silent"}     a whole run instead of one fight
///   {"cmd":"step", "action": {"type":"play_card","hand_index":0,"target":0}}
///   {"cmd":"step", "index": 3}            index into the last observation's legal_actions
///   {"cmd":"observe"}
///   {"cmd":"encounters"}
/// Every reply to reset/new_run/step/observe is `{"ok":true, "obs":{...}, "done":bool}`; errors are
/// `{"ok":false,"error":"..."}`. In a run, `step` actions are `RunAction`s
/// (e.g. `{"type":"path","x":3}`, `{"type":"combat","action":{"type":"end_turn"}}`).
fn serve() -> Result<(), String> {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    let mut combat: Option<Session> = None;
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match handle(&line, &mut combat) {
            Ok(v) => v,
            Err(e) => json!({"ok": false, "error": e}),
        };
        writeln!(out, "{reply}").map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

enum Session {
    Combat(Box<Combat>),
    Run(Box<Run>),
}

fn obs_reply(c: &Combat) -> Value {
    json!({"ok": true, "obs": c.observe(), "done": c.is_over()})
}

fn session_reply(s: &Session) -> Value {
    match s {
        Session::Combat(c) => obs_reply(c),
        Session::Run(r) => json!({"ok": true, "obs": r.observe(), "done": r.is_over()}),
    }
}

#[derive(Deserialize, Default)]
struct RunArgs {
    character: Option<String>,
    seed: Option<i64>,
    ascension: Option<i32>,
    neow_full: Option<bool>,
    bosses_seen: Option<bool>,
    final_act: Option<bool>,
}

fn run_config(r: RunArgs) -> Result<RunConfig, String> {
    let mut c = RunConfig::new(r.seed.unwrap_or(0));
    c.character = character(r.character.as_deref())?;
    c.ascension = r.ascension.unwrap_or(0);
    c.neow_full = r.neow_full.unwrap_or(true);
    c.bosses_seen = r.bosses_seen.unwrap_or(true);
    c.final_act = r.final_act.unwrap_or(true);
    Ok(c)
}

fn handle(line: &str, session: &mut Option<Session>) -> Result<Value, String> {
    let v: Value = serde_json::from_str(line).map_err(|e| format!("bad json: {e}"))?;
    let cmd = v.get("cmd").and_then(|c| c.as_str()).ok_or("missing cmd")?;
    match cmd {
        "reset" => {
            let r: ResetArgs = serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
            let s = Session::Combat(Box::new(Combat::new(&config_from(r)?)?));
            let reply = session_reply(&s);
            *session = Some(s);
            Ok(reply)
        }
        "new_run" => {
            let r: RunArgs = serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
            let s = Session::Run(Box::new(Run::new(run_config(r)?)));
            let reply = session_reply(&s);
            *session = Some(s);
            Ok(reply)
        }
        "step" => {
            let s = session.as_mut().ok_or("send reset or new_run first")?;
            let index = v.get("index").and_then(|i| i.as_u64()).map(|i| i as usize);
            let action = v.get("action").cloned();
            match s {
                Session::Combat(c) => {
                    let a: PlayerAction = match index {
                        Some(i) => c.legal_actions().get(i).cloned().ok_or("index out of range")?,
                        None => serde_json::from_value(action.ok_or("missing action")?).map_err(|e| e.to_string())?,
                    };
                    c.step(&a)?;
                }
                Session::Run(r) => {
                    let a: RunAction = match index {
                        Some(i) => r.legal_actions().get(i).cloned().ok_or("index out of range")?,
                        None => serde_json::from_value(action.ok_or("missing action")?).map_err(|e| e.to_string())?,
                    };
                    r.step(&a)?;
                }
            }
            Ok(session_reply(s))
        }
        "observe" => Ok(session_reply(session.as_ref().ok_or("send reset or new_run first")?)),
        "encounters" => Ok(json!({"ok": true, "encounters": encounters::all()})),
        _ => Err(format!("unknown cmd '{cmd}'")),
    }
}

fn describe(c: &Combat) -> String {
    let o = c.observe();
    let mut s = String::new();
    s += &format!(
        "--- turn {} | HP {}/{} block {} energy {} | powers {:?}\n",
        o.turn,
        o.player.hp,
        o.player.max_hp,
        o.player.block,
        o.player.energy,
        o.player.powers.iter().map(|p| format!("{} {}", p.id, p.amount)).collect::<Vec<_>>()
    );
    for m in &o.monsters {
        let intent = if m.intent_damage >= 0 { format!("{:?} {}x{}", m.intent, m.intent_damage, m.intent_hits) } else { format!("{:?}", m.intent) };
        s += &format!(
            "  [{}] {} {}/{} block {} intent {} powers {:?}{}\n",
            m.id,
            m.name,
            m.hp,
            m.max_hp,
            m.block,
            intent,
            m.powers.iter().map(|p| format!("{} {}", p.id, p.amount)).collect::<Vec<_>>(),
            if m.alive { "" } else { " (dead)" }
        );
    }
    let hand: Vec<String> = o
        .hand
        .iter()
        .map(|c| format!("{} ({}){}", c.name, c.cost, if c.playable { "" } else { " [can't play]" }))
        .collect();
    s += &format!("  hand: {}\n", hand.join(", "));
    s += &format!("  draw {} | discard {} | exhaust {}\n", o.draw_pile.len(), o.discard_pile.len(), o.exhaust_pile.len());
    if let Some(ch) = &o.choice {
        s += &format!("  choose {} card(s) to {}:\n", ch.count, ch.kind);
    }
    for (i, a) in o.legal_actions.iter().enumerate() {
        let label = match a {
            PlayerAction::PlayCard { hand_index, target } => {
                let card = &o.hand[*hand_index];
                match target {
                    Some(t) => format!("play {} (cost {}) -> [{}]", card.name, card.cost, t),
                    None => format!("play {} (cost {})", card.name, card.cost),
                }
            }
            PlayerAction::EndTurn => "end turn".to_string(),
            PlayerAction::Choose { indices } => {
                let ch = o.choice.as_ref().unwrap();
                let names: Vec<String> = indices
                    .iter()
                    .map(|i| {
                        let pos = c.pending.as_ref().unwrap().options.iter().position(|o| o == i).unwrap();
                        ch.options[pos].name.clone()
                    })
                    .collect();
                format!("choose {}", names.join(", "))
            }
            PlayerAction::UsePotion { slot, target } => {
                let name = c.potions[*slot].map_or("?", |p| p.data().name);
                match target {
                    Some(t) => format!("drink {name} -> [{t}]"),
                    None => format!("drink {name}"),
                }
            }
            PlayerAction::DiscardPotion { slot } => format!("discard potion {slot}"),
        };
        s += &format!("  {i}: {label}\n");
    }
    s
}

fn play(args: &[String]) -> Result<(), String> {
    let r = ResetArgs {
        character: arg(args, "--character").map(String::from),
        seed: arg(args, "--seed").map(|s| s.parse().unwrap_or(0)),
        encounter: arg(args, "--encounter").map(String::from),
        ascension: arg(args, "--ascension").map(|s| s.parse().unwrap_or(0)),
        ..Default::default()
    };
    let mut c = Combat::new(&config_from(r)?)?;
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    while !c.is_over() {
        print!("{}> ", describe(&c));
        io::stdout().flush().ok();
        let Some(Ok(line)) = lines.next() else { break };
        let Ok(i) = line.trim().parse::<usize>() else {
            println!("enter an action number");
            continue;
        };
        let Some(a) = c.legal_actions().get(i).cloned() else {
            println!("no such action");
            continue;
        };
        if let Err(e) = c.step(&a) {
            println!("error: {e}");
        }
    }
    println!("{}", describe(&c));
    println!("outcome: {:?}", c.outcome);
    Ok(())
}

fn describe_run(r: &Run) -> String {
    use sts_core::run::Screen;
    let mut s = format!(
        "=== act {} floor {} | HP {}/{} gold {} | relics {:?} | deck {}\n",
        r.act,
        r.floor,
        r.hp,
        r.max_hp,
        r.gold,
        r.relics.iter().map(|x| x.id.data().name).collect::<Vec<_>>(),
        r.deck.len()
    );
    if let Screen::Combat = r.screen {
        if let Some(c) = &r.combat {
            return s + &describe(c);
        }
    }
    let legal = r.legal_actions();
    let label = |a: &RunAction| -> String {
        match (&r.screen, a) {
            (Screen::Neow { options }, RunAction::Neow { option }) => format!("{:?} / {:?}", options[*option].reward, options[*option].drawback),
            (Screen::Map, RunAction::Path { x }) => {
                let y = r.pos.map_or(0, |(_, y)| y + 1);
                let room = if y >= 15 { Some(sts_core::run::map::RoomType::Boss) } else { r.map.node(*x, y).room };
                format!("go to x={x} ({room:?})")
            }
            (Screen::Rewards { items }, RunAction::TakeReward { index }) => format!("take {:?}", items[*index]),
            (Screen::CardReward { cards, .. }, RunAction::PickCard { index }) => format!("pick {}", cards[*index].name()),
            (Screen::Grid { .. }, RunAction::Select { indices }) => {
                format!("select {}", indices.iter().map(|&i| r.deck[i].name()).collect::<Vec<_>>().join(", "))
            }
            (Screen::BossRelic { relics }, RunAction::BossRelic { index }) => format!("take {}", relics[*index].data().name),
            (Screen::Event { event }, RunAction::EventChoice { index }) => event.labels[*index].clone(),
            (Screen::Shop { shop }, RunAction::BuyCard { index }) => {
                let c = shop.cards[*index].as_ref().unwrap();
                format!("buy {} for {} gold", c.card.name(), c.price)
            }
            (Screen::Shop { shop }, RunAction::BuyRelic { index }) => {
                let x = shop.relics[*index].as_ref().unwrap();
                format!("buy {} for {} gold", x.relic.data().name, x.price)
            }
            (Screen::Shop { shop }, RunAction::BuyPotion { index }) => {
                let x = shop.potions[*index].as_ref().unwrap();
                format!("buy {} for {} gold", x.potion.data().name, x.price)
            }
            (Screen::Shop { shop }, RunAction::Purge) => format!("remove a card for {} gold", shop.purge_cost),
            (_, RunAction::UsePotion { slot }) => format!("drink {}", r.potions[*slot].map_or("?", |p| p.data().name)),
            (_, RunAction::DiscardPotion { slot }) => format!("discard {}", r.potions[*slot].map_or("?", |p| p.data().name)),
            (Screen::Rest, RunAction::Rest) => "rest (heal 30%)".into(),
            (Screen::Rest, RunAction::Smith) => "smith (upgrade a card)".into(),
            (Screen::Rest, RunAction::Recall) => "recall (take the ruby key)".into(),
            (_, RunAction::Proceed) => "proceed / leave".into(),
            (_, RunAction::SkipCards) => "skip".into(),
            _ => format!("{a:?}"),
        }
    };
    match &r.screen {
        Screen::Event { event } => {
            s += &format!("event: {}\n", event.id.key());
            for c in event.cards.iter() {
                if event.id != sts_core::run::events::EventId::MatchAndKeep {
                    s += &format!("  card on offer: {}\n", c.name());
                }
            }
        }
        Screen::Shop { .. } => s += "the merchant\n",
        Screen::Rewards { .. } => s += "rewards\n",
        Screen::CardReward { .. } => s += "choose a card\n",
        Screen::Grid { purpose, .. } => s += &format!("choose card(s): {purpose:?}\n"),
        Screen::Rest => s += "rest site\n",
        Screen::Neow { .. } => s += "Neow\n",
        Screen::BossRelic { .. } => s += "boss relic\n",
        _ => {}
    }
    if !r.potions.iter().all(|p| p.is_none()) {
        let pots: Vec<&str> = r.potions.iter().map(|p| p.map_or("-", |p| p.data().name)).collect();
        s += &format!("potions: {}\n", pots.join(", "));
    }
    if let Screen::Map = r.screen {
        s += &r.map.to_text();
        s += "\n";
    }
    for (i, a) in legal.iter().enumerate() {
        s += &format!("  [{i}] {}\n", label(a));
    }
    s
}

fn run_cmd(args: &[String]) -> Result<(), String> {
    let seed = match arg(args, "--seed") {
        Some(s) => runlog::parse_seed(s).ok_or(format!("bad seed '{s}'"))?,
        None => 0,
    };
    let save = arg(args, "--save");
    let mut cfg = RunConfig::new(seed);
    cfg.character = character(arg(args, "--character"))?;
    cfg.ascension = arg(args, "--ascension").and_then(|s| s.parse().ok()).unwrap_or(0);
    cfg.neow_full = !args.iter().any(|a| a == "--mini-neow");
    let mut r = Run::new(cfg);
    let mut log = runlog::RunLog::new(&r);
    println!("seed {} ({})", r.config.seed, log.seed_string);
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    while !r.is_over() {
        print!("{}> ", describe_run(&r));
        io::stdout().flush().ok();
        let Some(Ok(line)) = lines.next() else { break };
        let Ok(i) = line.trim().parse::<usize>() else {
            println!("enter an action number");
            continue;
        };
        let Some(a) = r.legal_actions().get(i).cloned() else {
            println!("no such action");
            continue;
        };
        let before = r.clone();
        match r.step(&a) {
            Ok(()) => {
                // Saved after every step, so a run that is abandoned part way can still be watched.
                log.record(&before, &a);
                if let Some(path) = save {
                    log.save(path)?;
                }
            }
            Err(e) => println!("error: {e}"),
        }
    }
    println!("outcome: {:?} on floor {}", r.outcome, r.floor);
    if let Some(path) = save {
        println!("saved {} steps to {path}; watch it with tools/trace/playback.py", log.steps.len());
    }
    Ok(())
}

fn bench_runs(args: &[String]) -> Result<(), String> {
    let games: u64 = arg(args, "--games").and_then(|s| s.parse().ok()).unwrap_or(1000);
    let seed: i64 = arg(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0);
    let ch = character(arg(args, "--character"))?;
    let mut picker = StsRandom::new(seed);
    let start = Instant::now();
    let (mut steps, mut floors, mut cleared) = (0u64, 0u64, 0u64);
    for g in 0..games {
        let mut cfg = RunConfig::new(seed.wrapping_add(g as i64));
        cfg.character = ch;
        let mut r = Run::new(cfg);
        while !r.is_over() {
            let legal = r.legal_actions();
            // Random, but only end the turn one time in ten while cards are playable.
            let playable: Vec<&RunAction> = legal.iter().filter(|a| !matches!(a, RunAction::Combat { action: PlayerAction::EndTurn })).collect();
            let a = if !playable.is_empty() && legal.len() != playable.len() && picker.random(9) != 0 {
                playable[picker.random(playable.len() as i32 - 1) as usize].clone()
            } else {
                legal[picker.random(legal.len() as i32 - 1) as usize].clone()
            };
            r.step(&a)?;
            steps += 1;
        }
        floors += r.floor as u64;
        if r.act > 1 {
            cleared += 1;
        }
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{games} runs, {cleared} cleared Act 1, mean floor {:.1}, {steps} decisions, {:.0} runs/s, {:.0} decisions/s",
        floors as f64 / games as f64,
        games as f64 / secs,
        steps as f64 / secs
    );
    Ok(())
}

fn bench(args: &[String]) -> Result<(), String> {
    let games: u64 = arg(args, "--games").and_then(|s| s.parse().ok()).unwrap_or(1000);
    let seed: i64 = arg(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or(0);
    let which = arg(args, "--encounter").unwrap_or("all");
    let list: Vec<String> = if which == "all" { encounters::all().iter().map(|s| s.to_string()).collect() } else { vec![which.to_string()] };
    let random_decks = args.iter().any(|a| a == "--random-decks");
    let ch = character(arg(args, "--character"))?;
    let pool = ch.library_order();
    let mut picker = StsRandom::new(seed);
    let start = Instant::now();
    let mut wins = 0u64;
    let mut steps = 0u64;
    let mut stalemates = 0u64;
    for g in 0..games {
        let enc = &list[(g as usize) % list.len()];
        let mut cfg = CombatConfig::new(ch, seed.wrapping_add(g as i64), enc);
        if random_decks {
            let n = 8 + picker.random(14);
            cfg.deck = (0..n)
                .map(|_| {
                    let id = pool[picker.random(pool.len() as i32 - 1) as usize];
                    (id, if picker.random_boolean() { 1 } else { 0 })
                })
                .collect();
        }
        let mut c = Combat::new(&cfg)?;
        let mut guard = 0;
        while !c.is_over() {
            let legal = c.legal_actions();
            let a = &legal[picker.random(legal.len() as i32 - 1) as usize];
            c.step(a)?;
            steps += 1;
            guard += 1;
            if guard > 20000 || c.turn > 300 {
                stalemates += 1;
                break;
            }
        }
        if c.outcome == Some(sts_core::Outcome::Victory) {
            wins += 1;
        }
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{games} games, {wins} wins ({:.1}%), {stalemates} unfinished after 300 turns, {steps} decisions, {:.0} games/s, {:.0} decisions/s",
        100.0 * wins as f64 / games as f64,
        games as f64 / secs,
        steps as f64 / secs
    );
    Ok(())
}

fn replay_cmd(files: &[String]) -> Result<(), String> {
    let verbose = files.iter().any(|f| f == "-v");
    let (mut ok, mut bad) = (0, 0);
    for f in files.iter().filter(|f| !f.starts_with('-')) {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        if text.trim().is_empty() {
            continue;
        }
        let r = replay::replay(&text);
        if r.ok {
            ok += 1;
            if verbose {
                println!("OK   {f} ({} steps)", r.steps);
            }
        } else {
            bad += 1;
            println!("DIFF {f}: {}", r.message);
        }
    }
    println!("{ok} traces match, {bad} differ");
    Ok(())
}

fn replay_run_cmd(files: &[String]) -> Result<(), String> {
    let verbose = files.iter().any(|f| f == "-v");
    let (mut ok, mut bad) = (0, 0);
    for f in files.iter().filter(|f| !f.starts_with('-')) {
        let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        if text.trim().is_empty() {
            continue;
        }
        let r = run_replay::replay_run(&text, verbose);
        if r.ok {
            ok += 1;
            println!("OK   {f} ({} steps; {})", r.steps, r.message);
        } else {
            bad += 1;
            println!("DIFF {f} ({} steps): {}", r.steps, r.message);
        }
    }
    println!("{ok} runs match, {bad} differ");
    Ok(())
}
