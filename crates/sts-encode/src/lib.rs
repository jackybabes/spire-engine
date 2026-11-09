//! What the run network sees and does, shared by the training environments (`crates/sts-py`) and
//! the browser engine (`crates/sts-wasm`), so a network reads the same numbers in both:
//! - `encode`: a fight as fixed-size arrays, and the fixed action slots;
//! - `encode_run`: a whole run's arrays (the fight's plus the run's own fields);
//! - `agent`: from the network's action slot to the engine's action.

pub mod agent;
pub mod encode;
pub mod encode_run;
