//! Plays whole runs with random legal actions to shake out panics and dead ends.

use sts_core::run::{Run, RunAction, RunConfig, RunOutcome, Screen};
use sts_core::PlayerAction;

/// A random policy that leans towards playing cards and resting when hurt, so runs get deeper.
fn pick(run: &Run, state: &mut u64) -> RunAction {
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

fn play_from(run: Run, seed: i64) -> Run {
    play_until(run, seed, |_| false)
}

fn play_until(mut run: Run, seed: i64, stop: fn(&Run) -> bool) -> Run {
    let mut state = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
    let mut steps = 0;
    while !run.is_over() && !stop(&run) && steps < 50_000 {
        let a = pick(&run, &mut state);
        run.step(&a).unwrap_or_else(|e| panic!("seed {seed}: {a:?} failed: {e}"));
        steps += 1;
    }
    assert!(run.is_over() || stop(&run), "seed {seed} stuck on {:?}", run.screen);
    run
}

fn play(seed: i64, full: bool) -> Run {
    let mut cfg = RunConfig::new(seed);
    cfg.neow_full = full;
    play_from(Run::new(cfg), seed)
}

#[test]
fn random_runs_finish() {
    let mut hist = std::collections::BTreeMap::new();
    for seed in 0..300 {
        let run = play(seed, seed % 3 != 0);
        *hist.entry(run.floor).or_insert(0) += 1;
    }
    eprintln!("floors reached: {hist:?}");
}

#[test]
fn runs_are_deterministic() {
    let a = serde_json::to_string(&play(42, true)).unwrap();
    let b = serde_json::to_string(&play(42, true)).unwrap();
    assert_eq!(a, b);
}

#[test]
fn tough_runs_clear_the_act() {
    let mut cleared = 0;
    for seed in 0..100 {
        let mut run = Run::new(RunConfig::new(seed));
        run.step(&RunAction::Neow { option: 1 }).unwrap();
        run.max_hp = 600;
        run.hp = 600;
        let run = play_until(run, seed, |r| r.act == 2);
        if run.act == 2 {
            cleared += 1;
            assert_eq!(run.floor, 17);
            assert!(matches!(run.screen, Screen::Map));
            assert_eq!(run.monster_list.len(), 15);
            assert_eq!(run.elite_list.len(), 10);
            assert!(["Automaton", "Collector", "Champ"].contains(&run.boss.as_str()));
        }
    }
    assert!(cleared > 70, "{cleared}");
}

#[test]
fn tough_runs_reach_later_acts() {
    let (mut later, mut won) = (0, 0);
    for seed in 0..30 {
        let mut run = Run::new(RunConfig::new(seed));
        run.step(&RunAction::Neow { option: 1 }).unwrap();
        run.max_hp = 600;
        run.hp = 600;
        let run = play_from(run, seed);
        if run.act > 1 {
            later += 1;
        }
        if run.outcome == Some(RunOutcome::Won) {
            won += 1;
            assert_eq!(run.act, 3);
        }
    }
    assert!(later > 20 && won > 0, "{later} {won}");
}

/// Many tough runs through every act, counting unported screens (`cargo test -- --ignored`).
#[test]
#[ignore]
fn fuzz_later_acts() {
    let mut unsupported = std::collections::BTreeMap::new();
    let (mut won, mut act4) = (0, 0);
    for seed in 0..400 {
        let mut cfg = RunConfig::new(seed);
        // A quarter of the runs at A20 (double boss at the end of Act 3).
        if seed % 4 == 0 {
            cfg.ascension = 20;
        }
        let mut run = Run::new(cfg);
        run.step(&RunAction::Neow { option: 1 }).unwrap();
        run.max_hp = 2000;
        run.hp = 2000;
        let mut state = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
        let mut steps = 0;
        while !run.is_over() && steps < 100_000 {
            if let Screen::Unsupported { what } = &run.screen {
                *unsupported.entry(what.clone()).or_insert(0) += 1;
            }
            let a = pick(&run, &mut state);
            run.step(&a).unwrap_or_else(|e| panic!("seed {seed}: {a:?} failed: {e}"));
            steps += 1;
        }
        if run.outcome == Some(RunOutcome::Won) {
            won += 1;
        }
        if run.act == 4 {
            act4 += 1;
        }
    }
    eprintln!("won {won}/400 ({act4} reached Act 4); unsupported: {unsupported:?}");
}
