//! The named RNG streams on `AbstractDungeon`, with the game's seeding rules.

use serde::{Deserialize, Serialize};
use sts_rng::StsRandom;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rngs {
    pub seed: i64,
    pub monster: StsRandom,
    pub map: StsRandom,
    pub event: StsRandom,
    pub merchant: StsRandom,
    pub card: StsRandom,
    pub treasure: StsRandom,
    pub relic: StsRandom,
    pub potion: StsRandom,
    pub monster_hp: StsRandom,
    pub ai: StsRandom,
    pub shuffle: StsRandom,
    pub card_random: StsRandom,
    pub misc: StsRandom,
    /// Stand-in for the game's unseeded `MathUtils.random` / `Collections.shuffle(list)` calls,
    /// which are not reproducible in the real game. Seeded here so runs stay deterministic.
    pub unseeded: StsRandom,
    /// The Courier's shop restock picks its card unseeded in the game (`getCardFromPool(...,
    /// useRng = false)`). Its own stream here, which the harness patch `HarnessCourierPatch`
    /// gives the modded game too, so recorded runs that buy a restocked card can be checked.
    pub courier: StsRandom,
}

/// Seeds `Rngs::courier` (with the run seed); `HarnessCourierPatch` uses the same value.
pub const COURIER_SEED_XOR: i64 = 0x436F_7572_6965_7221; // "Courier!"

impl Rngs {
    /// `AbstractDungeon.generateSeeds()`; `mapRng` is set per act (`seed + actNum` in Act 1).
    pub fn new(seed: i64) -> Rngs {
        Rngs {
            seed,
            monster: StsRandom::new(seed),
            map: StsRandom::new(seed.wrapping_add(1)),
            event: StsRandom::new(seed),
            merchant: StsRandom::new(seed),
            card: StsRandom::new(seed),
            treasure: StsRandom::new(seed),
            relic: StsRandom::new(seed),
            potion: StsRandom::new(seed),
            monster_hp: StsRandom::new(seed),
            ai: StsRandom::new(seed),
            shuffle: StsRandom::new(seed),
            card_random: StsRandom::new(seed),
            misc: StsRandom::new(seed),
            unseeded: StsRandom::new(seed ^ 0x5DEE_CE66_D5DE_ECE6),
            courier: StsRandom::new(seed ^ COURIER_SEED_XOR),
        }
    }

    /// A fight's own streams reseeded from `seed` (training only, not the game: a replay of a lost
    /// practice fight with other draws, crates/sts-py `retry_losses`): monster HP, monster AI,
    /// shuffles, cards' random effects, misc, potions and the unseeded stand-in.
    pub fn reseed_fight(&mut self, seed: i64) {
        self.monster_hp = StsRandom::new(seed);
        self.ai = StsRandom::new(seed);
        self.shuffle = StsRandom::new(seed);
        self.card_random = StsRandom::new(seed);
        self.misc = StsRandom::new(seed);
        self.potion = StsRandom::new(seed);
        self.unseeded = StsRandom::new(seed ^ 0x5DEE_CE66_D5DE_ECE6);
    }

    /// Reseeding done in `nextRoomTransition` when entering a floor.
    pub fn enter_floor(&mut self, floor: i32) {
        let s = self.seed.wrapping_add(floor as i64);
        self.monster_hp = StsRandom::new(s);
        self.ai = StsRandom::new(s);
        self.shuffle = StsRandom::new(s);
        self.card_random = StsRandom::new(s);
        self.misc = StsRandom::new(s);
    }
}
