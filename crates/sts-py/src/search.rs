//! Look-ahead for fights by playouts (`python/search/fight_search.py`): from a run waiting for a fight
//! move, one copy of the run for each candidate first move and each redraw of its unseen
//! randomness. Each copy takes its first move, then Python plays it on with the network's own
//! greedy moves until it is the player's turn again `turns` turns later, or the fight ends, and
//! scores where it ends with the network's value; a first move's score is its copies' average.
//!
//! The copies are redrawn ("determinized") so the search can't see what the player can't: the
//! draw pile is reshuffled (unless the player has Frozen Eye) and the fight's random streams are
//! reseeded, so draws, random targets and the monsters' next intents come out differently in
//! each copy. Every first move of a root sees the same redraws, so their scores differ by the
//! move, not by the luck of the draw.
//!
//! (An earlier version tried every line of the rest of the turn and scored a first move by its
//! best line per copy: the maximum over many noisy leaves favoured moves with many lines, and it
//! lost to the network alone.)

use sts_core::relics::RelicId;
use sts_core::rng::StsRandom;
use sts_core::run::{Run, RunAction, RunOutcome, Screen};

use crate::{encode_run, fight_slot_action, scenarios};

/// How a playout ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum End {
    /// Still in the run: Python scores it with the network's value.
    Leaf,
    Died,
    WonRun,
    /// A practice fight (or boss scenario) over, as `Slot::finish_step` scores it.
    Practice { won: bool, hp: i32, max_hp: i32, potions: u32, bosses_beaten: i32 },
}

impl End {
    pub fn code(&self) -> u8 {
        match self {
            End::Leaf => 0,
            End::Died => 1,
            End::WonRun => 2,
            End::Practice { .. } => 3,
        }
    }
}

/// Where a root's practice fight ends, if it is one: `None` for a run; for a practice fight,
/// `Some(None)`; for a boss scenario, `Some(Some((first floor, last floor)))`.
pub type PracticeEnd = Option<Option<(i32, i32)>>;

pub struct Playout {
    /// Index into the roots it was started from.
    pub root: u32,
    /// The action slot of its first move.
    pub first: u32,
    /// Which redraw.
    pub det: u32,
    pub run: Run,
    pub legal: Vec<RunAction>,
    pub picked: Vec<usize>,
    start_turn: i32,
    turns: i32,
    moves: u32,
    /// It stops after this many moves even if its turns aren't over (a leaf): the network can
    /// loop (with Blue Candle it plays a Necronomicurse, which comes back, for 1 HP each time).
    max_moves: u32,
    practice: PracticeEnd,
    pub end: Option<End>,
}

/// Redraws what the player can't know: the order of the draw pile and the fight's random streams.
pub fn determinize(r: &mut Run, seed: i64) {
    let Some(c) = r.combat.as_deref_mut() else { return };
    let mut k = 0i64;
    let mut next = || {
        k += 1;
        StsRandom::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(k))
    };
    c.rng.shuffle = next();
    c.rng.card_random = next();
    c.rng.ai = next();
    c.rng.misc = next();
    c.rng.monster_hp = next();
    c.rng.potion = next();
    c.rng.card = next();
    c.rng.unseeded = next();
    if !c.has_relic(RelicId::FrozenEye) {
        let mut rng = next();
        for i in (1..c.draw_pile.len()).rev() {
            let j = rng.random(i as i32) as usize;
            c.draw_pile.swap(i, j);
        }
    }
}

/// Whether `r` is waiting for a fight move that isn't part of a card selection (a root the
/// search can start from).
pub fn searchable(r: &Run) -> bool {
    matches!((&r.screen, r.combat.as_deref()), (Screen::Combat, Some(c)) if c.outcome.is_none() && c.pending.is_none())
}

impl Playout {
    /// Takes action slot `a`; a card selection is picked one slot at a time, as in
    /// `Slot::step_slot`. Then checks whether the playout is over.
    pub fn step(&mut self, a: usize) -> Result<(), String> {
        let c = self.run.combat.as_deref().ok_or("a playout out of its fight")?;
        let picked = &mut self.picked;
        let action = fight_slot_action(c, &encode_run::combat_legal(&self.legal), picked, a)?;
        self.moves += 1;
        if let Some(action) = action {
            self.run.step(&RunAction::Combat { action })?;
            self.legal = if self.run.is_over() { Vec::new() } else { self.run.legal_actions() };
            self.picked.clear();
        }
        self.settle();
        Ok(())
    }

    /// Ends the playout if it is over (see `End`).
    fn settle(&mut self) {
        let r = &self.run;
        if let Some(scenario) = self.practice {
            let over = match scenario {
                Some((_, last)) => scenarios::over(r, last),
                None => !matches!(r.screen, Screen::Combat),
            };
            if over || r.is_over() {
                let won = over && r.outcome != Some(RunOutcome::Died);
                let bosses_beaten = match scenario {
                    Some((first, last)) if won => last - first + 1,
                    Some((first, _)) => (r.floor - first).max(0),
                    None => 0,
                };
                self.end = Some(End::Practice { won, hp: if won { r.hp } else { 0 }, max_hp: r.max_hp,
                                                potions: r.potions.iter().filter(|p| p.is_some()).count() as u32, bosses_beaten });
                return;
            }
        }
        self.end = if r.is_over() {
            Some(if r.outcome == Some(RunOutcome::Won) { End::WonRun } else { End::Died })
        } else if !matches!(r.screen, Screen::Combat) || self.moves >= self.max_moves {
            Some(End::Leaf)
        } else {
            match r.combat.as_deref() {
                Some(c) if c.turn >= self.start_turn + self.turns => Some(End::Leaf),
                None => Some(End::Leaf),
                _ => None,
            }
        };
    }
}

/// How far each playout of a root goes: `turns` turns, at most `max_moves` moves.
#[derive(Clone, Copy)]
pub struct Horizon {
    pub turns: i32,
    pub max_moves: u32,
}

/// The playouts of one root: for each of `k` redraws, a copy for each first move in `firsts`
/// (action slots), with that move taken. `seed` picks the redraws. No playouts if `r`
/// isn't `searchable`; an error if a first move isn't legal.
pub fn start(root: u32, r: &Run, firsts: &[usize], k: usize, h: Horizon, seed: i64, practice: PracticeEnd) -> Result<Vec<Playout>, String> {
    if !searchable(r) {
        return Ok(Vec::new());
    }
    let start_turn = r.combat.as_ref().map_or(0, |c| c.turn);
    let mut out = Vec::with_capacity(k * firsts.len());
    for det in 0..k {
        let mut copy = r.clone();
        determinize(&mut copy, seed.wrapping_add(det as i64 * 7919));
        let legal = copy.legal_actions();
        for &a in firsts {
            let mut p = Playout { root, first: a as u32, det: det as u32, run: copy.clone(), legal: legal.clone(), picked: Vec::new(),
                                  start_turn, turns: h.turns, moves: 0, max_moves: h.max_moves, practice, end: None };
            p.step(a)?;
            out.push(p);
        }
    }
    Ok(out)
}
