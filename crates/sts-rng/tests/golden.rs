//! Compares against values printed by the real game classes (tools/golden/RngGolden.java).

use sts_rng::{JavaRandom, StsRandom};

const GOLDEN: &str = include_str!("../../../tools/golden/rng_golden.txt");

fn field<'a>(line: &'a str, key: &str) -> &'a str {
    let prefix = format!("{key}=");
    line.split(' ')
        .find_map(|tok| tok.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("missing {key} in {line}"))
}

#[test]
fn matches_game_rng() {
    let mut checked = 0;
    for line in GOLDEN.lines().filter(|l| !l.trim().is_empty()) {
        let seed: i64 = line.split(' ').next().unwrap().parse().unwrap();
        let mut r = StsRandom::new(seed);
        assert_eq!(r.random_long().to_string(), field(line, "long"), "seed {seed}");
        assert_eq!(r.random(99).to_string(), field(line, "int99"), "seed {seed}");
        assert_eq!(r.random_range(5, 10).to_string(), field(line, "int5_10"), "seed {seed}");
        assert_eq!(r.random_boolean().to_string(), field(line, "bool"), "seed {seed}");
        assert_eq!(r.random_boolean_chance(0.5625).to_string(), field(line, "boolp"), "seed {seed}");
        assert_eq!(r.random_float().to_bits().to_string(), field(line, "float"), "seed {seed}");
        assert_eq!(r.random_long_range(1_000_000).to_string(), field(line, "longr"), "seed {seed}");
        assert_eq!(r.counter.to_string(), field(line, "counter"), "seed {seed}");

        let mut jr = JavaRandom::new(seed);
        let got = format!("{},{},{}", jr.next_int_bounded(10), jr.next_int_bounded(1000), jr.next_int());
        assert_eq!(got, field(line, "jutil"), "seed {seed}");
        checked += 1;
    }
    assert_eq!(checked, 5);
}
