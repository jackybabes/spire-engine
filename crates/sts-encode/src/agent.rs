//! From the network's action slot to the engine's action, as training plays it (`crates/sts-py`'s
//! `step_slot`): a card selection is made one pick at a time and sent to the engine when it is
//! full or confirmed, and the card rewards skipped on the current reward screen are remembered
//! (`encode_run::options` shows them).

use sts_core::run::{Run, RunAction, Screen};
use sts_core::{Combat, PlayerAction};

use crate::encode;
use crate::encode_run;

/// Card rewards skipped on the current reward screen (`encode_run::card_reward_key` each).
pub type Skipped = Vec<Vec<(i32, i32)>>;

/// The fight action for slot `a`, or `None` while a card selection is still being picked (the
/// pick is added to `picked`).
pub fn fight_slot_action(c: &Combat, legal: &[PlayerAction], picked: &mut Vec<usize>, a: usize) -> Result<Option<PlayerAction>, String> {
    let bad = || format!("action slot {a} is not allowed now");
    if let Some(p) = &c.pending {
        let (_, most) = encode::selection_size(p);
        if (encode::PICK..encode::CONFIRM).contains(&a) {
            let o = *p.options.get(a - encode::PICK).ok_or_else(bad)?;
            if picked.contains(&o) || picked.len() >= most {
                return Err(bad());
            }
            picked.push(o);
            if picked.len() < most {
                return Ok(None);
            }
        } else if a != encode::CONFIRM {
            return Err(bad());
        }
        let mut indices = std::mem::take(picked);
        indices.sort_unstable();
        return Ok(Some(PlayerAction::Choose { indices }));
    }
    let slots = encode::monster_slots(c);
    let slot_of = |m: usize| slots.iter().position(|&x| x == m);
    legal.iter().find(|l| encode::slot_for(l, &slot_of) == Some(a)).cloned().map(Some).ok_or_else(bad)
}

/// The run action for slot `a` (in a fight or between fights), or `None` while a card selection
/// is still being picked (the pick is added to `picked`).
pub fn slot_action(run: &Run, legal: &[RunAction], picked: &mut Vec<usize>, skipped: &[Vec<(i32, i32)>], a: usize) -> Result<Option<RunAction>, String> {
    if let (Screen::Combat, Some(c)) = (&run.screen, run.combat.as_ref()) {
        Ok(fight_slot_action(c, &encode_run::combat_legal(legal), picked, a)?.map(|action| RunAction::Combat { action }))
    } else {
        encode_run::resolve(run, legal, picked, skipped, a)
    }
}

/// Before `a` is taken from `run`: skipping a card reward remembers it.
pub fn before_step(skipped: &mut Skipped, run: &Run, a: &RunAction) {
    if let (RunAction::SkipCards, Screen::CardReward { cards, .. }) = (a, &run.screen) {
        skipped.push(encode_run::card_reward_key(cards));
    }
}

/// After an action, with `run` as it leaves it: leaving the reward screens forgets the skipped
/// card rewards.
pub fn after_step(skipped: &mut Skipped, run: &Run) {
    if !matches!(run.screen, Screen::Rewards { .. } | Screen::CardReward { .. }) {
        skipped.clear();
    }
}
