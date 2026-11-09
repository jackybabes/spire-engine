//! Bit-exact ports of the random number generators Slay the Spire uses.
//!
//! - [`Xs128`] is libGDX `RandomXS128` (xorshift128+ seeded via murmur3).
//! - [`StsRandom`] is `com.megacrit.cardcrawl.random.Random`, a counting wrapper
//!   around `Xs128`. Every gameplay stream (`monsterRng`, `aiRng`, `shuffleRng`, ...)
//!   is one of these.
//! - [`JavaRandom`] is `java.util.Random`, which the game uses for
//!   `Collections.shuffle(list, new java.util.Random(rng.randomLong()))`.

const NORM_DOUBLE: f64 = 1.0 / (1u64 << 53) as f64;
const NORM_FLOAT: f64 = 1.0 / (1u64 << 24) as f64;

fn murmur_hash3(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// libGDX `RandomXS128`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Xs128 {
    seed0: u64,
    seed1: u64,
}

impl Xs128 {
    pub fn new(seed: i64) -> Self {
        let seed = if seed == 0 { i64::MIN } else { seed } as u64;
        let seed0 = murmur_hash3(seed);
        Self { seed0, seed1: murmur_hash3(seed0) }
    }

    pub fn from_state(seed0: u64, seed1: u64) -> Self {
        Self { seed0, seed1 }
    }

    pub fn state(&self) -> (u64, u64) {
        (self.seed0, self.seed1)
    }

    pub fn next_long(&mut self) -> i64 {
        let mut s1 = self.seed0;
        let s0 = self.seed1;
        self.seed0 = s0;
        s1 ^= s1 << 23;
        self.seed1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        self.seed1.wrapping_add(s0) as i64
    }

    /// `nextLong(long n)`: rejection-sampled value in `[0, n)`.
    pub fn next_long_bounded(&mut self, n: i64) -> i64 {
        assert!(n > 0, "n must be positive");
        loop {
            let bits = ((self.next_long() as u64) >> 1) as i64;
            let value = bits % n;
            if bits.wrapping_sub(value).wrapping_add(n - 1) >= 0 {
                return value;
            }
        }
    }

    pub fn next_int_bounded(&mut self, n: i32) -> i32 {
        self.next_long_bounded(n as i64) as i32
    }

    pub fn next_double(&mut self) -> f64 {
        ((self.next_long() as u64) >> 11) as f64 * NORM_DOUBLE
    }

    pub fn next_float(&mut self) -> f32 {
        (((self.next_long() as u64) >> 40) as f64 * NORM_FLOAT) as f32
    }

    pub fn next_boolean(&mut self) -> bool {
        self.next_long() & 1 != 0
    }
}

/// `com.megacrit.cardcrawl.random.Random`. Method names mirror the Java overloads.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StsRandom {
    pub rng: Xs128,
    pub counter: i32,
}

impl StsRandom {
    pub fn new(seed: i64) -> Self {
        Self { rng: Xs128::new(seed), counter: 0 }
    }

    /// `Random(Long seed, int counter)`: advances by calling `random(999)` `counter` times.
    pub fn with_counter(seed: i64, counter: i32) -> Self {
        let mut r = Self::new(seed);
        for _ in 0..counter {
            r.random(999);
        }
        r
    }

    pub fn set_counter(&mut self, target: i32) {
        while self.counter < target {
            self.random_boolean();
        }
    }

    /// `random(int range)`: inclusive `[0, range]`.
    pub fn random(&mut self, range: i32) -> i32 {
        self.counter += 1;
        self.rng.next_int_bounded(range + 1)
    }

    /// `random(int start, int end)`: inclusive `[start, end]`.
    pub fn random_range(&mut self, start: i32, end: i32) -> i32 {
        self.counter += 1;
        start + self.rng.next_int_bounded(end - start + 1)
    }

    /// `random(long range)`: `[0, range)` via `nextDouble`.
    pub fn random_long_range(&mut self, range: i64) -> i64 {
        self.counter += 1;
        (self.rng.next_double() * range as f64) as i64
    }

    /// `random(long start, long end)`.
    pub fn random_long_between(&mut self, start: i64, end: i64) -> i64 {
        self.counter += 1;
        start + (self.rng.next_double() * (end - start) as f64) as i64
    }

    pub fn random_long(&mut self) -> i64 {
        self.counter += 1;
        self.rng.next_long()
    }

    pub fn random_boolean(&mut self) -> bool {
        self.counter += 1;
        self.rng.next_boolean()
    }

    pub fn random_boolean_chance(&mut self, chance: f32) -> bool {
        self.counter += 1;
        self.rng.next_float() < chance
    }

    /// `random()`: float in `[0, 1)`.
    pub fn random_float(&mut self) -> f32 {
        self.counter += 1;
        self.rng.next_float()
    }

    pub fn random_float_range(&mut self, range: f32) -> f32 {
        self.counter += 1;
        self.rng.next_float() * range
    }

    pub fn random_float_between(&mut self, start: f32, end: f32) -> f32 {
        self.counter += 1;
        start + self.rng.next_float() * (end - start)
    }
}

/// `java.util.Random` (48-bit LCG).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JavaRandom {
    seed: i64,
}

impl JavaRandom {
    const MULTIPLIER: i64 = 0x5_DEEC_E66D;
    const MASK: i64 = (1 << 48) - 1;

    pub fn new(seed: i64) -> Self {
        Self { seed: (seed ^ Self::MULTIPLIER) & Self::MASK }
    }

    fn next(&mut self, bits: u32) -> i32 {
        self.seed = (self.seed.wrapping_mul(Self::MULTIPLIER).wrapping_add(0xB)) & Self::MASK;
        (self.seed >> (48 - bits)) as i32
    }

    pub fn next_int(&mut self) -> i32 {
        self.next(32)
    }

    pub fn next_int_bounded(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive");
        if bound & -bound == bound {
            return ((bound as i64 * self.next(31) as i64) >> 31) as i32;
        }
        loop {
            let bits = self.next(31);
            let val = bits % bound;
            if bits.wrapping_sub(val).wrapping_add(bound - 1) >= 0 {
                return val;
            }
        }
    }

    pub fn next_long(&mut self) -> i64 {
        ((self.next(32) as i64) << 32).wrapping_add(self.next(32) as i64)
    }

    pub fn next_boolean(&mut self) -> bool {
        self.next(1) != 0
    }

    pub fn next_double(&mut self) -> f64 {
        (((self.next(26) as i64) << 27) + self.next(27) as i64) as f64 * NORM_DOUBLE
    }
}

/// Anything `Collections.shuffle` can draw from (`java.util.Random.nextInt(int)`).
pub trait NextInt {
    fn next_int(&mut self, bound: i32) -> i32;
}

impl NextInt for JavaRandom {
    fn next_int(&mut self, bound: i32) -> i32 {
        self.next_int_bounded(bound)
    }
}

/// `RandomXS128` overrides `nextInt(int)`; the map generator shuffles with it directly.
impl NextInt for Xs128 {
    fn next_int(&mut self, bound: i32) -> i32 {
        self.next_int_bounded(bound)
    }
}

/// `Collections.shuffle(list, rnd)` for a `RandomAccess` list such as `ArrayList`.
pub fn java_shuffle<T, R: NextInt>(items: &mut [T], rnd: &mut R) {
    for i in (2..=items.len()).rev() {
        let j = rnd.next_int(i as i32) as usize;
        items.swap(i - 1, j);
    }
}
