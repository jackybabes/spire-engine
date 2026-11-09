//! Headless Slay the Spire game logic, ported from the decompiled game.
//!
//! Milestone 1: Ironclad combat against every Act 1 encounter, with the game's seeded RNG.

pub mod action;
pub mod cards;
pub mod character;
pub mod combat;
pub mod commands;
pub mod encounters;
pub mod monsters;
pub mod potions;
pub mod powers;
pub mod relics;
pub mod rngs;
pub mod run;

pub use combat::{Combat, CombatConfig, Outcome, PlayerAction};
pub use sts_rng as rng;
