//! The network playing a run, as training plays it (`train_runs.py --eval`: the likeliest legal
//! action slot each time, a card selection one pick at a time), and its advice: the likeliest
//! decisions at a position with their probabilities.

use sts_core::run::{Run, RunAction};
use sts_encode::agent::{self, Skipped};
use sts_encode::encode;
use sts_encode::encode_run;

use crate::{probabilities, Net, NEG};

/// What the network makes of a position: each action slot's logit (`NEG` for slots not allowed)
/// and the run value.
pub fn logits(net: &Net, run: &Run, legal: &[RunAction], picked: &[usize], skipped: &Skipped) -> Result<(Vec<f32>, f32), String> {
    let mut ints = vec![0i32; encode::int_len() + encode_run::run_int_len()];
    let mut floats = vec![0f32; encode::float_len() + encode_run::run_float_len()];
    let mut mask = vec![0u8; encode::ACTIONS];
    encode_run::encode_run(run, legal, picked, skipped, &mut ints, &mut floats, &mut mask);
    if !mask.iter().any(|&m| m != 0) {
        return Err("no action slot is allowed here".into());
    }
    let (mut logits, value) = net.forward(&ints, &floats)?;
    for (l, &m) in logits.iter_mut().zip(&mask) {
        if m == 0 {
            *l = NEG;
        }
    }
    Ok((logits, value))
}

/// Each action slot's probability (0 for slots not allowed) and the run value.
pub fn evaluate(net: &Net, run: &Run, legal: &[RunAction], picked: &[usize], skipped: &Skipped) -> Result<(Vec<f32>, f32), String> {
    let (logits, value) = logits(net, run, legal, picked, skipped)?;
    let mask: Vec<u8> = logits.iter().map(|&l| (l > NEG) as u8).collect();
    Ok((probabilities(&logits, &mask), value))
}

/// The slot with the highest logit, the first of equal ones (as PyTorch's `argmax`: identical
/// cards in hand score exactly the same, and which copy is played moves the discard pile's
/// order, so later shuffles).
fn likeliest(logits: &[f32]) -> usize {
    let mut best = 0;
    for (a, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = a;
        }
    }
    best
}

/// The decision slot `first` leads to when the network makes the rest of a card selection
/// (its likeliest pick each time).
pub fn complete(net: &Net, run: &Run, legal: &[RunAction], skipped: &Skipped, first: usize) -> Result<RunAction, String> {
    let mut picked = Vec::new();
    let mut slot = first;
    for _ in 0..=encode::MAX_OPTIONS {
        if let Some(a) = agent::slot_action(run, legal, &mut picked, skipped, slot)? {
            return Ok(a);
        }
        slot = likeliest(&logits(net, run, legal, &picked, skipped)?.0);
    }
    Err("a card selection that never ends".into())
}

/// The network's own decision here (greedy, as its evaluations play).
pub fn choose(net: &Net, run: &Run, legal: &[RunAction], skipped: &Skipped) -> Result<RunAction, String> {
    let (logits, _) = logits(net, run, legal, &[], skipped)?;
    complete(net, run, legal, skipped, likeliest(&logits))
}

/// One piece of advice: a decision and the probability the network gives it (for a card
/// selection, its first pick's).
#[derive(Debug, Clone)]
pub struct Choice {
    pub action: RunAction,
    pub probability: f32,
}

/// The network's `k` likeliest decisions here, likeliest first, and the run value. A card
/// selection's first picks are each completed with the network's own later picks, and picks
/// that end in the same selection count together (among the 2k + 2 likeliest slots).
pub fn advise(net: &Net, run: &Run, legal: &[RunAction], skipped: &Skipped, k: usize) -> Result<(Vec<Choice>, f32), String> {
    let (probs, value) = evaluate(net, run, legal, &[], skipped)?;
    let mut slots: Vec<usize> = (0..probs.len()).filter(|&a| probs[a] > 0.0).collect();
    slots.sort_by(|&a, &b| probs[b].total_cmp(&probs[a])); // stable: equal ones keep slot order
    let mut out: Vec<Choice> = Vec::new();
    // The likeliest slots only: each card selection's completion costs more forward passes.
    for slot in slots.into_iter().take(2 * k + 2) {
        let action = complete(net, run, legal, skipped, slot)?;
        match out.iter().position(|c| c.action == action) {
            Some(i) => out[i].probability += probs[slot],
            None if out.len() < k => out.push(Choice { action, probability: probs[slot] }),
            None => {}
        }
    }
    out.sort_by(|a, b| b.probability.total_cmp(&a.probability));
    Ok((out, value))
}

/// Takes `a` in `run`, keeping the skipped card rewards as training does.
pub fn step(run: &mut Run, skipped: &mut Skipped, a: &RunAction) -> Result<(), String> {
    agent::before_step(skipped, run, a);
    run.step(a)?;
    agent::after_step(skipped, run);
    Ok(())
}
