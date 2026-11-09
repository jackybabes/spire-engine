//! A release network playing fixed runs greedily as its weights file's character, as
//! `train_runs.py --eval` does; one JSON line per run (seed, won, act, floor, hp, decisions). For checking the Rust network against PyTorch's
//! play (`python/release/net_play_check.py`) and timing it.
//!
//!     cargo run --release -p sts-net --example play -- web/models/ironclad-v1.safetensors 20 300000 32

use std::time::Instant;

use sts_core::character::Character;
use sts_core::run::{Run, RunConfig, RunOutcome};
use sts_net::{play, Net};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = Net::from_safetensors(&std::fs::read(&args[1]).expect("weights")).expect("network");
    let ascension: i32 = args[2].parse().unwrap();
    let first: i64 = args[3].parse().unwrap();
    let n: i64 = args[4].parse().unwrap();
    let (t0, mut decisions_all) = (Instant::now(), 0);
    for seed in first..first + n {
        let mut cfg = RunConfig::new(seed);
        cfg.ascension = ascension;
        // The weights file's character (the Ironclad's when it doesn't say).
        cfg.character = net.metadata.get("character").and_then(|c| Character::from_name(c)).unwrap_or(Character::Ironclad);
        let mut run = Run::new(cfg);
        let mut skipped = Vec::new();
        let mut decisions = 0;
        while !run.is_over() && decisions < 8000 {
            let legal = run.legal_actions();
            let a = play::choose(&net, &run, &legal, &skipped).unwrap();
            play::step(&mut run, &mut skipped, &a).unwrap();
            decisions += 1;
        }
        decisions_all += decisions;
        println!("{{\"seed\": {seed}, \"won\": {}, \"act\": {}, \"floor\": {}, \"hp\": {}, \"decisions\": {decisions}}}",
                 run.outcome == Some(RunOutcome::Won), run.act, run.floor, run.hp);
    }
    eprintln!("{decisions_all} decisions in {:.1} s ({:.2} ms each)", t0.elapsed().as_secs_f64(), 1000.0 * t0.elapsed().as_secs_f64() / decisions_all as f64);
}
