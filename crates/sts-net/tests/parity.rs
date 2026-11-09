//! The Rust network against PyTorch's on the same positions (`python/release/net_fixture.py`): every
//! legal action's logit and the run value agree, and so does the likeliest action.
//!
//! The committed fixture is a tiny random network. `STS_NET_FIXTURE=/tmp/v1.json cargo test -p
//! sts-net` also checks a fixture made with a release's weights (`--weights`).

use std::path::{Path, PathBuf};

use serde_json::Value;
use sts_net::{Net, NEG};

fn check(fixture: &Path) {
    let f: Value = serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    let weights = PathBuf::from(f["weights"].as_str().unwrap());
    let weights = if weights.is_absolute() { weights } else { fixture.parent().unwrap().join(weights) };
    let net = Net::from_safetensors(&std::fs::read(&weights).unwrap()).unwrap();
    let (ni, nf) = (f["run_int_len"].as_u64().unwrap() as usize, f["run_float_len"].as_u64().unwrap() as usize);
    let (mut worst, mut n) = (0.0f32, 0);
    for (c, case) in f["cases"].as_array().unwrap().iter().enumerate() {
        let mut ints = vec![0i32; ni];
        for e in case["ints"].as_array().unwrap() {
            ints[e[0].as_u64().unwrap() as usize] = e[1].as_i64().unwrap() as i32;
        }
        let mut floats = vec![0f32; nf];
        for e in case["floats"].as_array().unwrap() {
            floats[e[0].as_u64().unwrap() as usize] = e[1].as_f64().unwrap() as f32;
        }
        let (logits, value) = net.forward(&ints, &floats).unwrap();
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 + 1e-4 * b.abs();
        let want_v = case["value"].as_f64().unwrap() as f32;
        assert!(close(value, want_v), "case {c}: value {value}, PyTorch {want_v}");
        let legal: Vec<usize> = case["legal"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let want: Vec<f32> = case["logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        for (&a, &w) in legal.iter().zip(&want) {
            assert!(logits[a] > NEG, "case {c}: legal action {a} has no score");
            assert!(close(logits[a], w), "case {c}: action {a} logit {}, PyTorch {w}", logits[a]);
            worst = worst.max((logits[a] - w).abs());
        }
        let best = |v: &dyn Fn(usize) -> f32| legal.iter().copied().max_by(|&a, &b| v(a).total_cmp(&v(b))).unwrap();
        let w_at = |a: usize| want[legal.iter().position(|&x| x == a).unwrap()];
        assert_eq!(best(&|a| logits[a]), best(&w_at), "case {c}: a different likeliest action");
        n += 1;
    }
    eprintln!("{}: {n} positions, largest logit difference {worst:.2e}", fixture.display());
}

#[test]
fn matches_pytorch_on_a_tiny_network() {
    check(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny.json"));
}

#[test]
fn matches_pytorch_on_a_release() {
    if let Ok(p) = std::env::var("STS_NET_FIXTURE") {
        check(Path::new(&p));
    }
}

#[test]
fn refuses_another_encoding() {
    let mut bytes = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny.safetensors")).unwrap();
    // Raise the stored encoding version by one digit, keeping the header's length.
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = String::from_utf8(bytes[8..8 + n].to_vec()).unwrap();
    let wrong = header.replacen(&format!("\"encoding_version\":\"{}\"", sts_encode::encode::ENCODING_VERSION), &format!("\"encoding_version\":\"{}\"", sts_encode::encode::ENCODING_VERSION + 1), 1);
    assert_ne!(wrong, header);
    assert_eq!(wrong.len(), header.len());
    bytes[8..8 + n].copy_from_slice(wrong.as_bytes());
    let err = Net::from_safetensors(&bytes).err().unwrap();
    assert!(err.contains("encoding_version"), "{err}");
}
