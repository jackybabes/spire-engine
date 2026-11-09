//! The run network's forward pass in plain Rust (no dependencies beyond JSON), for the browser:
//! the same computation as `python/core/run_model.py`'s `RunNet` in evaluation mode, from a release
//! weights file (`python/release/export_release.py`: safetensors, float32).
//!
//! One row at a time: the encoder's arrays for one decision (`sts_encode::encode_run`) give
//! the 171 action logits and the run value. Tokens that PyTorch pads out of attention (empty
//! hand, monster, potion, orb, deck and option slots) are left out here instead; the real tokens come
//! out the same, and the logits that would come from a padded token belong to actions the mask
//! never allows, so they come back as `NEG`.

pub mod play;

use std::collections::BTreeMap;

use serde_json::Value;
use sts_encode::encode::{self, ACTIONS, CONFIRM, END_TURN, MAX_HAND, MAX_MONSTERS, MAX_OPTIONS, MAX_ORBS, MAX_POTIONS, MOVES_SEEN, PICK, PLAY, POTION, TARGETS};
use sts_encode::encode_run;

/// The logit of an action no token scores (always masked out).
pub const NEG: f32 = -1e9;
/// The weights file's `format`.
pub const FORMAT: &str = "sts-run-net";

struct Lin {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    i: usize,
    o: usize,
}

impl Lin {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.i);
        (0..self.o)
            .map(|r| {
                let row = &self.w[r * self.i..(r + 1) * self.i];
                let mut s = self.b.as_ref().map_or(0.0, |b| b[r]);
                for (w, x) in row.iter().zip(x) {
                    s += w * x;
                }
                s
            })
            .collect()
    }

    fn apply1(&self, x: &[f32]) -> f32 {
        self.apply(x)[0]
    }
}

struct Emb {
    w: Vec<f32>,
    n: usize,
    d: usize,
}

impl Emb {
    fn row(&self, id: i32) -> Result<&[f32], String> {
        let i = usize::try_from(id).ok().filter(|&i| i < self.n).ok_or(format!("id {id} outside an embedding of {}", self.n))?;
        Ok(&self.w[i * self.d..(i + 1) * self.d])
    }
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        x.iter().zip(&self.w).zip(&self.b).map(|((v, w), b)| (v - mean) * inv * w + b).collect()
    }
}

/// `nn.TransformerEncoderLayer(d, heads, 2d, norm_first=True)`, ReLU, no dropout.
struct Layer {
    qkv: Lin,
    out: Lin,
    l1: Lin,
    l2: Lin,
    n1: Norm,
    n2: Norm,
}

/// Additive attention score of each of `a` against each of `b` (`fight_model.Scorer`).
struct Scorer {
    a: Lin,
    b: Lin,
    v: Lin,
    c: Option<Lin>,
}

/// Linear, ReLU, Linear.
struct Mlp {
    l0: Lin,
    l2: Lin,
}

impl Mlp {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let h: Vec<f32> = self.l0.apply(x).into_iter().map(|v| v.max(0.0)).collect();
        self.l2.apply(&h)
    }
}

/// Where each named field sits in a run row.
struct Fields(BTreeMap<&'static str, (usize, usize)>, BTreeMap<&'static str, (usize, usize)>);

impl Fields {
    fn new() -> Fields {
        let offsets = |lists: [&[(&'static str, usize)]; 2]| {
            let (mut m, mut at) = (BTreeMap::new(), 0);
            for list in lists {
                for &(name, n) in list {
                    m.insert(name, (at, at + n));
                    at += n;
                }
            }
            m
        };
        Fields(offsets([encode::INT_FIELDS, encode_run::RUN_INT_FIELDS]), offsets([encode::FLOAT_FIELDS, encode_run::RUN_FLOAT_FIELDS]))
    }
}

pub struct Net {
    d: usize,
    heads: usize,
    /// The weights file's metadata (architecture, encoder signature, source checkpoint...).
    pub metadata: BTreeMap<String, String>,
    f: Fields,
    // The fight network (`fight_model.FightNet`).
    card: Emb,
    power: Emb,
    monster: Emb,
    intent: Emb,
    relic: Emb,
    potion: Emb,
    kind: Emb,
    pile: Emb,
    character: Emb,
    stance: Emb,
    card_type: Emb,
    orb: Emb,
    orb_slot: Emb,
    moves: Emb,
    kinds: Emb,
    scalars: Lin,
    hand_feats: Lin,
    monster_feats: Lin,
    option_feats: Lin,
    orb_feats: Lin,
    moves_seen: Lin,
    upgraded: Vec<f32>,
    amount: Mlp,
    layers: Vec<Layer>,
    out_norm: Norm,
    no_target: Vec<f32>,
    play: Scorer,
    use_potion: Scorer,
    pick: Lin,
    end_turn: Lin,
    confirm: Lin,
    // The run's parts (`run_model.RunNet`).
    screen: Emb,
    encounter: Emb,
    room: Emb,
    event: Emb,
    event_screen: Emb,
    event_index: Emb,
    grid_purpose: Emb,
    neow: Emb,
    neow_drawback: Emb,
    option_kind: Emb,
    run_kinds: Emb,
    run_scalars: Lin,
    run_option_feats: Lin,
    run_pick: Lin,
    run_confirm: Lin,
    run_potion: Lin,
    run_value: Mlp,
}

/// Raw game numbers as two features each: symlog, then x / 100 (`fight_model.scale`).
fn scale(x: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = x.iter().map(|v| v.signum() * v.abs().ln_1p()).collect();
    out.extend(x.iter().map(|v| v / 100.0));
    // signum(0.0) is 1.0 in Rust, but 0 * ln_1p(0) is 0 either way.
    out
}

fn add(acc: &mut [f32], x: &[f32]) {
    for (a, b) in acc.iter_mut().zip(x) {
        *a += b;
    }
}

/// A weights file's tensors by name: shape and data.
pub type Tensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// The tensors of a safetensors file (float32 only) and its metadata.
pub fn read_safetensors(bytes: &[u8]) -> Result<(Tensors, BTreeMap<String, String>), String> {
    let n = u64::from_le_bytes(bytes.get(..8).ok_or("not a weights file")?.try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(bytes.get(8..8 + n).ok_or("truncated weights file")?).map_err(|e| format!("weights header: {e}"))?;
    let base = 8 + n;
    let (mut tensors, mut metadata) = (BTreeMap::new(), BTreeMap::new());
    for (name, t) in header.as_object().ok_or("weights header is not an object")? {
        if name == "__metadata__" {
            for (k, v) in t.as_object().ok_or("bad metadata")? {
                metadata.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
            }
            continue;
        }
        if t["dtype"] != "F32" {
            return Err(format!("{name} is {}, not F32", t["dtype"]));
        }
        let shape: Vec<usize> = t["shape"].as_array().ok_or("bad shape")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
        let off = t["data_offsets"].as_array().ok_or("bad offsets")?;
        let (lo, hi) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
        let raw = bytes.get(base + lo..base + hi).ok_or(format!("{name} runs past the end of the file"))?;
        if raw.len() != 4 * shape.iter().product::<usize>() {
            return Err(format!("{name}: {} bytes for shape {shape:?}", raw.len()));
        }
        let data = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        tensors.insert(name.clone(), (shape, data));
    }
    Ok((tensors, metadata))
}

fn take_tensor(t: &mut Tensors, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let (s, data) = t.remove(name).ok_or(format!("the weights have no {name}"))?;
    if s != shape {
        return Err(format!("{name} has shape {s:?}, expected {shape:?}"));
    }
    Ok(data)
}

/// This build's encoder signature, as a weights file stores it.
pub fn signature() -> [(&'static str, String); 4] {
    [
        ("run_int_len", (encode::int_len() + encode_run::run_int_len()).to_string()),
        ("run_float_len", (encode::float_len() + encode_run::run_float_len()).to_string()),
        ("actions", ACTIONS.to_string()),
        ("encoding_version", encode::ENCODING_VERSION.to_string()),
    ]
}

impl Net {
    /// The network in a release weights file. Refuses a file made for another encoding: a
    /// network trained on other inputs would read the wrong numbers without any error.
    pub fn from_safetensors(bytes: &[u8]) -> Result<Net, String> {
        let (mut t, metadata) = read_safetensors(bytes)?;
        if metadata.get("format").map(String::as_str) != Some(FORMAT) {
            return Err(format!("not a run network weights file (format {:?})", metadata.get("format")));
        }
        let wrong: Vec<String> = signature()
            .iter()
            .filter(|(k, v)| metadata.get(*k) != Some(v))
            .map(|(k, v)| format!("{k} {} (this build {v})", metadata.get(*k).map_or("missing", String::as_str)))
            .collect();
        if !wrong.is_empty() {
            return Err(format!("the weights were made for another encoding: {}", wrong.join(", ")));
        }
        let num = |k: &str| metadata.get(k).and_then(|v| v.parse::<usize>().ok()).ok_or(format!("metadata has no {k}"));
        let (d, heads, layers) = (num("d")?, num("heads")?, num("layers")?);
        if heads == 0 || d % heads != 0 {
            return Err(format!("{heads} heads don't divide d {d}"));
        }
        let emb_names = ["card", "power", "monster", "intent", "relic", "potion", "kind", "pile", "character", "stance", "card_type", "orb", "orb_slot", "move",
                         "kinds", "screen", "encounter", "room", "event", "event_screen", "event_index", "grid_purpose", "neow", "neow_drawback", "option_kind", "run_kinds"];
        let emb_tensors: Tensors =
            emb_names.iter().filter_map(|n| { let k = format!("{n}.weight"); t.remove(&k).map(|v| (k, v)) }).collect();
        let mut take = |name: &str, shape: &[usize]| take_tensor(&mut t, name, shape);
        macro_rules! lin {
            ($name:expr, $i:expr, $o:expr) => {
                Lin { w: take(&format!("{}.weight", $name), &[$o, $i])?, b: Some(take(&format!("{}.bias", $name), &[$o])?), i: $i, o: $o }
            };
            ($name:expr, $i:expr, $o:expr, nobias) => {
                Lin { w: take(&format!("{}.weight", $name), &[$o, $i])?, b: None, i: $i, o: $o }
            };
        }
        let emb = |name: &str| -> Result<Emb, String> {
            let key = format!("{name}.weight");
            let (s, w) = emb_tensors.get(&key).cloned().ok_or(format!("the weights have no {key}"))?;
            if s.len() != 2 || s[1] != d {
                return Err(format!("{key} has shape {s:?}"));
            }
            Ok(Emb { w, n: s[0], d })
        };
        let (card, power, monster, intent, relic, potion) = (emb("card")?, emb("power")?, emb("monster")?, emb("intent")?, emb("relic")?, emb("potion")?);
        let (kind, pile, character, kinds) = (emb("kind")?, emb("pile")?, emb("character")?, emb("kinds")?);
        let (stance, card_type, orb, orb_slot, moves) = (emb("stance")?, emb("card_type")?, emb("orb")?, emb("orb_slot")?, emb("move")?);
        let (screen, encounter, room, event, event_screen, event_index) = (emb("screen")?, emb("encounter")?, emb("room")?, emb("event")?, emb("event_screen")?, emb("event_index")?);
        let (grid_purpose, neow, neow_drawback, option_kind, run_kinds) = (emb("grid_purpose")?, emb("neow")?, emb("neow_drawback")?, emb("option_kind")?, emb("run_kinds")?);
        let f = Fields::new();
        let len =|m: &BTreeMap<&str, (usize, usize)>, name: &str| m[name].1 - m[name].0;
        let scalars_n = len(&f.1, "scalars");
        let hand_feats_n = len(&f.1, "hand") / MAX_HAND;
        let monster_feats_n = len(&f.1, "monster") / MAX_MONSTERS;
        let option_feats_n = len(&f.1, "option") / MAX_OPTIONS;
        let run_scalars_n = len(&f.1, "run_scalars");
        let opt_feats_n = len(&f.1, "opt") / MAX_OPTIONS;
        let mut layer_list = Vec::new();
        for l in 0..layers {
            let p = format!("encoder.layers.{l}");
            layer_list.push(Layer {
                qkv: Lin { w: take(&format!("{p}.self_attn.in_proj_weight"), &[3 * d, d])?, b: Some(take(&format!("{p}.self_attn.in_proj_bias"), &[3 * d])?), i: d, o: 3 * d },
                out: lin!(format!("{p}.self_attn.out_proj"), d, d),
                l1: lin!(format!("{p}.linear1"), d, 2 * d),
                l2: lin!(format!("{p}.linear2"), 2 * d, d),
                n1: Norm { w: take(&format!("{p}.norm1.weight"), &[d])?, b: take(&format!("{p}.norm1.bias"), &[d])? },
                n2: Norm { w: take(&format!("{p}.norm2.weight"), &[d])?, b: take(&format!("{p}.norm2.bias"), &[d])? },
            });
        }
        let pair = take("play.c.weight", &[d, PAIR_FEATS])?;
        let net = Net {
            d,
            heads,
            scalars: lin!("scalars", 2 * scalars_n, d),
            hand_feats: lin!("hand_feats", 2 * hand_feats_n, d),
            monster_feats: lin!("monster_feats", 2 * monster_feats_n, d),
            option_feats: lin!("option_feats", 2 * option_feats_n, d),
            orb_feats: lin!("orb_feats", 4, d),
            moves_seen: lin!("moves_seen", MOVES_SEEN * d, d, nobias),
            upgraded: take("upgraded.weight", &[d, 1])?,
            amount: Mlp { l0: lin!("amount.0", d + 2, d), l2: lin!("amount.2", d, d) },
            layers: layer_list,
            out_norm: Norm { w: take("out_norm.weight", &[d])?, b: take("out_norm.bias", &[d])? },
            no_target: take("no_target", &[d])?,
            play: Scorer { a: lin!("play.a", d, d), b: lin!("play.b", d, d, nobias), v: lin!("play.v", d, 1), c: Some(Lin { w: pair, b: None, i: PAIR_FEATS, o: d }) },
            use_potion: Scorer { a: lin!("use_potion.a", d, d), b: lin!("use_potion.b", d, d, nobias), v: lin!("use_potion.v", d, 1), c: None },
            pick: lin!("pick", d, 1),
            end_turn: lin!("end_turn", d, 1),
            confirm: lin!("confirm", d, 1),
            run_scalars: lin!("run_scalars", 2 * run_scalars_n, d),
            run_option_feats: lin!("run_option_feats", 2 * opt_feats_n, d),
            run_pick: lin!("run_pick", d, 1),
            run_confirm: lin!("run_confirm", d, 1),
            run_potion: lin!("run_potion", d, 2),
            run_value: Mlp { l0: lin!("run_value.0", d, d), l2: lin!("run_value.2", d, 1) },
            card, power, monster, intent, relic, potion, kind, pile, character, stance, card_type, orb, orb_slot, moves, kinds,
            screen, encounter, room, event, event_screen, event_index, grid_purpose, neow, neow_drawback, option_kind, run_kinds,
            metadata,
            f,
        };
        Ok(net)
    }

    fn ints<'a>(&self, ints: &'a [i32], name: &str) -> &'a [i32] {
        let (a, b) = self.f.0[name];
        &ints[a..b]
    }

    fn floats<'a>(&self, floats: &'a [f32], name: &str) -> &'a [f32] {
        let (a, b) = self.f.1[name];
        &floats[a..b]
    }

    /// Sum over a list of (id, amount) entities, empty ids skipped (`FightNet._amounts`).
    fn amounts(&self, emb: &Emb, ids: &[i32], amounts: &[f32]) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0; self.d];
        for (&id, &a) in ids.iter().zip(amounts) {
            if id > 0 {
                let mut x = emb.row(id)?.to_vec();
                x.extend(scale(&[a]));
                add(&mut out, &self.amount.apply(&x));
            }
        }
        Ok(out)
    }

    /// The mean of a pile's cards (`FightNet._pool_cards`).
    fn pool_cards(&self, ids: &[i32], upgrades: &[f32]) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0; self.d];
        let (mut n, mut ups) = (0.0f32, 0.0f32);
        for (&id, &u) in ids.iter().zip(upgrades) {
            if id > 0 {
                add(&mut out, self.card.row(id)?);
                n += 1.0;
                ups += u;
            }
        }
        for (o, w) in out.iter_mut().zip(&self.upgraded) {
            *o = (*o + ups * w) / n.max(1.0);
        }
        Ok(out)
    }

    /// The transformer over `tokens` (each `d` long), then the output norm.
    fn encode(&self, mut x: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let (d, h) = (self.d, self.heads);
        let dh = d / h;
        let n = x.len();
        let scale = 1.0 / (dh as f32).sqrt();
        for layer in &self.layers {
            let qkv: Vec<Vec<f32>> = x.iter().map(|t| layer.qkv.apply(&layer.n1.apply(t))).collect();
            let mut attn = vec![vec![0.0f32; d]; n];
            let mut scores = vec![0.0f32; n];
            for head in 0..h {
                let (q0, k0, v0) = (head * dh, d + head * dh, 2 * d + head * dh);
                for i in 0..n {
                    let q = &qkv[i][q0..q0 + dh];
                    let mut most = f32::NEG_INFINITY;
                    for j in 0..n {
                        let k = &qkv[j][k0..k0 + dh];
                        let s = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                        scores[j] = s;
                        most = most.max(s);
                    }
                    let mut total = 0.0;
                    for s in scores.iter_mut() {
                        *s = (*s - most).exp();
                        total += *s;
                    }
                    let out = &mut attn[i][q0..q0 + dh];
                    for (j, s) in scores.iter().enumerate() {
                        let w = s / total;
                        for (o, v) in out.iter_mut().zip(&qkv[j][v0..v0 + dh]) {
                            *o += w * v;
                        }
                    }
                }
            }
            for (t, a) in x.iter_mut().zip(&attn) {
                add(t, &layer.out.apply(a));
                let hidden: Vec<f32> = layer.l1.apply(&layer.n2.apply(t)).into_iter().map(|v| v.max(0.0)).collect();
                add(t, &layer.l2.apply(&hidden));
            }
        }
        x.iter().map(|t| self.out_norm.apply(t)).collect()
    }

    /// One decision's action logits (171, `NEG` for actions no token scores) and run value, from
    /// its encoded row (`encode_run::encode_run`'s ints and floats).
    pub fn forward(&self, ints: &[i32], floats: &[f32]) -> Result<(Vec<f32>, f32), String> {
        let (want_i, want_f) = (self.f.0.values().map(|r| r.1).max().unwrap_or(0), self.f.1.values().map(|r| r.1).max().unwrap_or(0));
        if ints.len() != want_i || floats.len() != want_f {
            return Err(format!("a row of {} ints and {} floats, expected {want_i} and {want_f}", ints.len(), floats.len()));
        }
        let combat = self.ints(ints, "screen")[0] as usize == 1 + encode_run::SCREENS.iter().position(|s| *s == "combat").unwrap();
        if combat {
            self.combat(ints, floats)
        } else {
            self.choices(ints, floats)
        }
    }

    fn run_token(&self, ints: &[i32], floats: &[f32]) -> Result<Vec<f32>, String> {
        let mut x = self.screen.row(self.ints(ints, "screen")[0])?.to_vec();
        add(&mut x, self.encounter.row(self.ints(ints, "boss")[0])?);
        add(&mut x, self.room.row(self.ints(ints, "room")[0])?);
        add(&mut x, self.event.row(self.ints(ints, "event")[0])?);
        add(&mut x, self.event_screen.row(self.ints(ints, "event_screen")[0])?);
        add(&mut x, self.grid_purpose.row(self.ints(ints, "grid_purpose")[0])?);
        add(&mut x, &self.run_scalars.apply(&scale(self.floats(floats, "run_scalars"))));
        add(&mut x, self.run_kinds.row(0)?);
        Ok(x)
    }

    fn combat(&self, ints: &[i32], floats: &[f32]) -> Result<(Vec<f32>, f32), String> {
        let d = self.d;
        let k = |i: i32| self.kinds.row(i);
        let scal = scale(self.floats(floats, "scalars"));
        let scal_x = self.scalars.apply(&scal);
        // The fight's tokens in PyTorch's order, each with whether it's real (not padding).
        let mut tokens: Vec<(Vec<f32>, bool)> = Vec::new();
        let mut glob = self.character.row(self.ints(ints, "character")[0])?.to_vec();
        add(&mut glob, &scal_x);
        add(&mut glob, self.kind.row(self.ints(ints, "choice_kind")[0])?);
        add(&mut glob, self.pile.row(self.ints(ints, "choice_pile")[0])?);
        add(&mut glob, self.card_type.row(self.ints(ints, "last_played_type")[0])?);
        add(&mut glob, k(0)?);
        tokens.push((glob, true));
        let mut player = self.amounts(&self.power, self.ints(ints, "player_power"), self.floats(floats, "player_power_amount"))?;
        add(&mut player, &scal_x);
        add(&mut player, self.stance.row(self.ints(ints, "stance")[0])?);
        add(&mut player, k(1)?);
        tokens.push((player, true));
        let mut relics = self.amounts(&self.relic, self.ints(ints, "relic"), self.floats(floats, "relic_counter"))?;
        add(&mut relics, k(2)?);
        tokens.push((relics, true));
        for p in ["draw_pile", "discard_pile", "exhaust_pile"] {
            let mut x = self.pool_cards(self.ints(ints, p), self.floats(floats, &format!("{p}_upgrades")))?;
            add(&mut x, k(3)?);
            tokens.push((x, true));
        }
        let hand_ids = self.ints(ints, "hand_card");
        let hand_f = self.floats(floats, "hand");
        let hf = hand_f.len() / MAX_HAND;
        for (h, &id) in hand_ids.iter().enumerate() {
            let mut x = self.card.row(id)?.to_vec();
            add(&mut x, &self.hand_feats.apply(&scale(&hand_f[h * hf..(h + 1) * hf])));
            add(&mut x, k(4)?);
            tokens.push((x, id > 0));
        }
        let mon_ids = self.ints(ints, "monster");
        let mon_f = self.floats(floats, "monster");
        let mf = mon_f.len() / MAX_MONSTERS;
        let (mon_pow, mon_amt) = (self.ints(ints, "monster_power"), self.floats(floats, "monster_power_amount"));
        let mp = mon_pow.len() / MAX_MONSTERS;
        let moves = self.ints(ints, "monster_moves");
        for (m, &id) in mon_ids.iter().enumerate() {
            let mut x = self.monster.row(id)?.to_vec();
            add(&mut x, self.intent.row(self.ints(ints, "monster_intent")[m])?);
            add(&mut x, &self.monster_feats.apply(&scale(&mon_f[m * mf..(m + 1) * mf])));
            add(&mut x, &self.amounts(&self.power, &mon_pow[m * mp..(m + 1) * mp], &mon_amt[m * mp..(m + 1) * mp])?);
            let mut seen = Vec::with_capacity(MOVES_SEEN * d);
            for &mv in &moves[m * MOVES_SEEN..(m + 1) * MOVES_SEEN] {
                seen.extend_from_slice(self.moves.row(mv)?);
            }
            add(&mut x, &self.moves_seen.apply(&seen));
            add(&mut x, k(5)?);
            tokens.push((x, id > 0));
        }
        for &id in self.ints(ints, "potion") {
            let mut x = self.potion.row(id)?.to_vec();
            add(&mut x, k(6)?);
            tokens.push((x, id > 0));
        }
        // The orb slots, front first, each with its position (only the slots there are).
        let (orb_ids, passive, evoke) = (self.ints(ints, "orb"), self.floats(floats, "orb_passive"), self.floats(floats, "orb_evoke"));
        for (o, &id) in orb_ids.iter().enumerate().take(MAX_ORBS).filter(|(_, &id)| id > 0) {
            let mut x = self.orb.row(id)?.to_vec();
            add(&mut x, &self.orb_feats.apply(&scale(&[passive[o], evoke[o]])));
            add(&mut x, self.orb_slot.row(o as i32)?);
            add(&mut x, k(8)?);
            tokens.push((x, true));
        }
        // The run token goes after the fight's fixed tokens and the orbs, before the options.
        let fixed = tokens.len();
        tokens.push((self.run_token(ints, floats)?, true));
        let opt_ids = self.ints(ints, "option_card");
        let opt_f = self.floats(floats, "option");
        let of = opt_f.len() / MAX_OPTIONS;
        // PyTorch keeps the first (number of options) option tokens (the options come first).
        let n_options = opt_ids.iter().filter(|&&i| i > 0).count();
        for (o, &id) in opt_ids.iter().enumerate().take(n_options) {
            let mut x = self.card.row(id)?.to_vec();
            add(&mut x, &self.option_feats.apply(&scale(&opt_f[o * of..(o + 1) * of])));
            add(&mut x, k(7)?);
            tokens.push((x, id > 0));
        }

        // Encode the real tokens only; `at[i]` is token i's place among them.
        let mut at = vec![None; tokens.len()];
        let mut real = Vec::new();
        for (i, (x, keep)) in tokens.into_iter().enumerate() {
            if keep {
                at[i] = Some(real.len());
                real.push(x);
            }
        }
        let x = self.encode(real);
        let tok = |i: usize| at.get(i).copied().flatten().map(|j| &x[j]);

        let value = self.run_value.apply(tok(fixed).unwrap())[0];
        let mut logits = vec![NEG; ACTIONS];
        let g = tok(0).unwrap();
        // Targets: no target, then each monster slot (6 + MAX_HAND is the first monster token).
        let mon0 = 6 + MAX_HAND;
        let pot0 = mon0 + MAX_MONSTERS;
        let targets: Vec<Option<Vec<f32>>> = std::iter::once(Some(self.no_target.clone()))
            .chain((0..MAX_MONSTERS).map(|m| tok(mon0 + m).cloned()))
            .collect();
        let pair = pair_features(self, floats);
        let play_b: Vec<Option<Vec<f32>>> = targets.iter().map(|t| t.as_ref().map(|t| self.play.b.apply(t))).collect();
        for h in 0..MAX_HAND {
            let Some(hx) = tok(6 + h) else { continue };
            let a = self.play.a.apply(hx);
            for (t, b) in play_b.iter().enumerate() {
                let Some(b) = b else { continue };
                let c = self.play.c.as_ref().unwrap().apply(&pair[h][t]);
                let s: Vec<f32> = (0..d).map(|i| (a[i] + b[i] + c[i]).tanh()).collect();
                logits[PLAY + h * TARGETS + t] = self.play.v.apply1(&s);
            }
        }
        let pot_b: Vec<Option<Vec<f32>>> = targets.iter().map(|t| t.as_ref().map(|t| self.use_potion.b.apply(t))).collect();
        for p in 0..MAX_POTIONS {
            let Some(px) = tok(pot0 + p) else { continue };
            let a = self.use_potion.a.apply(px);
            for (t, b) in pot_b.iter().enumerate() {
                let Some(b) = b else { continue };
                let s: Vec<f32> = (0..d).map(|i| (a[i] + b[i]).tanh()).collect();
                logits[POTION + p * TARGETS + t] = self.use_potion.v.apply1(&s);
            }
        }
        logits[END_TURN] = self.end_turn.apply1(g);
        for o in 0..MAX_OPTIONS {
            if let Some(ox) = tok(fixed + 1 + o) {
                logits[PICK + o] = self.pick.apply1(ox);
            }
        }
        logits[CONFIRM] = self.confirm.apply1(g);
        Ok((logits, value))
    }

    fn choices(&self, ints: &[i32], floats: &[f32]) -> Result<(Vec<f32>, f32), String> {
        let k = |i: i32| self.run_kinds.row(i);
        let mut tokens: Vec<(Vec<f32>, bool)> = vec![(self.run_token(ints, floats)?, true)];
        let mut relics = self.amounts(&self.relic, self.ints(ints, "run_relic"), self.floats(floats, "run_relic_counter"))?;
        add(&mut relics, k(1)?);
        tokens.push((relics, true));
        for &id in self.ints(ints, "run_potion") {
            let mut x = self.potion.row(id)?.to_vec();
            add(&mut x, k(2)?);
            tokens.push((x, id > 0));
        }
        let deck = self.ints(ints, "deck");
        let ups = self.floats(floats, "deck_upgrades");
        let n_deck = deck.iter().filter(|&&i| i > 0).count().max(1);
        for (i, &id) in deck.iter().enumerate().take(n_deck) {
            let mut x = self.card.row(id)?.to_vec();
            for (o, w) in x.iter_mut().zip(&self.upgraded) {
                *o += ups[i] * w;
            }
            add(&mut x, k(3)?);
            tokens.push((x, id > 0));
        }
        let kinds = self.ints(ints, "opt_kind");
        let n_opt = kinds.iter().filter(|&&i| i > 0).count().max(1);
        let event_option = 1 + encode_run::OPTION_KINDS.iter().position(|s| *s == "event_option").unwrap() as i32;
        let event = self.event.row(self.ints(ints, "event")[0])?;
        let feats = self.floats(floats, "opt");
        let of = feats.len() / MAX_OPTIONS;
        let first_opt = tokens.len();
        for (o, &kind) in kinds.iter().enumerate().take(n_opt) {
            let mut x = self.option_kind.row(kind)?.to_vec();
            add(&mut x, self.card.row(self.ints(ints, "opt_card")[o])?);
            add(&mut x, self.relic.row(self.ints(ints, "opt_relic")[o])?);
            add(&mut x, self.potion.row(self.ints(ints, "opt_potion")[o])?);
            add(&mut x, self.neow.row(self.ints(ints, "opt_neow")[o])?);
            add(&mut x, self.neow_drawback.row(self.ints(ints, "opt_drawback")[o])?);
            add(&mut x, self.room.row(self.ints(ints, "opt_room")[o])?);
            add(&mut x, self.event_index.row(self.ints(ints, "opt_index")[o])?);
            if kind == event_option {
                add(&mut x, event);
            }
            add(&mut x, &self.run_option_feats.apply(&scale(&feats[o * of..(o + 1) * of])));
            add(&mut x, k(4)?);
            tokens.push((x, kind > 0));
        }

        let mut at = vec![None; tokens.len()];
        let mut real = Vec::new();
        for (i, (x, keep)) in tokens.into_iter().enumerate() {
            if keep {
                at[i] = Some(real.len());
                real.push(x);
            }
        }
        let x = self.encode(real);
        let tok = |i: usize| at.get(i).copied().flatten().map(|j| &x[j]);

        let g = tok(0).unwrap();
        let mut logits = vec![NEG; ACTIONS];
        for o in 0..n_opt {
            if let Some(ox) = tok(first_opt + o) {
                logits[PICK + o] = self.run_pick.apply1(ox);
            }
        }
        logits[CONFIRM] = self.run_confirm.apply1(g);
        for p in 0..MAX_POTIONS {
            if let Some(px) = tok(2 + p) {
                let pot = self.run_potion.apply(px);
                logits[POTION + p * TARGETS] = pot[0];
                logits[POTION + p * TARGETS + 1] = pot[1];
            }
        }
        Ok((logits, self.run_value.apply(g)[0]))
    }
}

/// What playing hand card h at target t would do (`fight_model.pair_features`). For a monster
/// (t = 1-6): one hit's damage to it (scaled), whether that kills, the share of its HP and block it
/// removes, then the same for the whole play. For no target (t = 0), an attack on every enemy or on
/// random ones: zeros, then the damage that lands summed over the monsters (scaled), how many it
/// kills and whether it kills them all; all zeros for any other card.
const PAIR_FEATS: usize = 8;
/// Hand row columns: hits every enemy, hits random enemies (`encode::hand_row`).
const HAND_ALL: usize = 9;
const HAND_RANDOM: usize = 10;

fn pair_features(net: &Net, floats: &[f32]) -> Vec<Vec<[f32; PAIR_FEATS]>> {
    let dmg = net.floats(floats, "hand_target_damage");
    let total = net.floats(floats, "hand_target_total");
    let hand = net.floats(floats, "hand");
    let hf = hand.len() / MAX_HAND;
    let mon = net.floats(floats, "monster");
    let mf = mon.len() / MAX_MONSTERS;
    let ehp: Vec<f32> = (0..MAX_MONSTERS).map(|m| mon[m * mf] + mon[m * mf + 2]).collect();
    let hit = |d: f32, ehp: f32| {
        let kills = if d >= ehp && ehp > 0.0 && d > 0.0 { 1.0 } else { 0.0 };
        let share = if ehp > 0.0 { (d / ehp.max(1.0)).min(2.0) } else { 0.0 };
        let s = scale(&[d]);
        [s[0], s[1], kills, share]
    };
    let present = ehp.iter().filter(|&&e| e > 0.0).count() as f32;
    (0..MAX_HAND)
        .map(|h| {
            let mut none = [0.0; PAIR_FEATS];
            if hand[h * hf + HAND_ALL] > 0.0 || hand[h * hf + HAND_RANDOM] > 0.0 {
                let (mut landed, mut killed) = (0.0f32, 0.0f32);
                for (m, &e) in ehp.iter().enumerate() {
                    let t = total[h * MAX_MONSTERS + m];
                    if e > 0.0 {
                        landed += t.min(e);
                    }
                    if t >= e && e > 0.0 && t > 0.0 {
                        killed += 1.0;
                    }
                }
                let s = scale(&[landed]);
                let all_dead = if killed >= present && present > 0.0 { 1.0 } else { 0.0 };
                none[4..].copy_from_slice(&[s[0], s[1], killed, all_dead]);
            }
            std::iter::once(none)
                .chain((0..MAX_MONSTERS).map(|m| {
                    let mut f = [0.0; PAIR_FEATS];
                    f[..4].copy_from_slice(&hit(dmg[h * MAX_MONSTERS + m], ehp[m]));
                    f[4..].copy_from_slice(&hit(total[h * MAX_MONSTERS + m], ehp[m]));
                    f
                }))
                .collect()
        })
        .collect()
}

/// The legal actions' probabilities (a softmax over the logits the mask allows; 0 elsewhere).
pub fn probabilities(logits: &[f32], mask: &[u8]) -> Vec<f32> {
    let most = logits.iter().zip(mask).filter(|(_, &m)| m != 0).map(|(l, _)| *l).fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().zip(mask).map(|(l, &m)| if m != 0 { (l - most).exp() } else { 0.0 }).collect();
    let total: f32 = p.iter().sum();
    if total > 0.0 {
        p.iter_mut().for_each(|v| *v /= total);
    }
    p
}
