# ironclad-v1: a network that plays the Ironclad

The first release of the Ironclad network, 2026-10-21: `ironclad-v1.safetensors` on the
GitHub release `ironclad-v1` (the weights file is not in git). The browser viewer loads it.

## What it is

A transformer that plays whole Slay the Spire runs as the Ironclad, in this repository's Rust
port of the game (`crates/sts-core`), through the `sts` Python module's run encoding.

- `python/core/run_model.py`'s `RunNet`: the fight network's transformer (`python/core/fight_model.py`:
  tokens for the hand cards, the monsters, the player, the piles, the relics, the potions, a
  card selection's options) plus a run token and, outside combat, tokens for the relics, the
  potions, the deck and the screen's options. d 128, 6 layers, 4 heads, pre-norm, no dropout;
  1,074,188 parameters.
- Inputs: one row of 1,081 integers (ids into the network's embeddings) and 2,114 floats per
  decision (`sts.layout()`, `crates/sts-py/src/encode.rs` and `encode_run.rs`,
  `docs/network-inputs.md`). Outputs: logits over 171 fixed action slots (play a hand card at a
  target, drink a potion, end the turn, pick option k, confirm), masked to the legal moves, and a
  value: its estimate of how the run will go.
- The file: safetensors (8-byte header length, a JSON header, raw little-endian float32
  tensors), weights only, 4.5 MB. It holds 1,113,228 numbers: the card, power and token-kind
  embedding tables have grown to this build's vocabulary since ironclad-runs-habits-12 trained (the other
  characters' cards and powers, appended after the Ironclad's; `fight_model.grow_embeddings`),
  and those 39,040 new rows were never trained. The Ironclad never sees those ids. The header's metadata holds the architecture, the encoder
  signature (`run_int_len` 1081, `run_float_len` 2114, `actions` 171, `encoding_version` 1),
  the source checkpoint (`dh12`, since renamed `ironclad-runs-habits-12`, update 200, 52,428,800 steps of that run), the git commit it
  was exported at and the character.
- Written by `python/release/export_release.py <checkpoint> <out>`; `export_release.load_release(path)`
  builds the PyTorch network from it and refuses a file whose encoder signature doesn't match
  the `sts` module it runs with (`sts.ENCODING_VERSION` is raised whenever the inputs change).
  `python/tests/test_export.py` checks that the exported network's outputs equal the checkpoint's.

## How it was trained

PPO (`python/training/train_runs.py`) on one consumer GPU (an RTX 5070 Ti since 2026-10-18),
in a line of runs from 2026-10-14 to 2026-10-19, each starting from an earlier one's weights: a
fight network on random-deck fights, then whole runs (ironclad-runs-1 to 13), block and exhaust practice (block, wr), boss practice from winning
human A20 boss fights (ironclad-runs-boss-practice-1 to 7), then habit rewards, a win reward of 3, learning-rate annealing
and gamma 0.9995 (ironclad-runs-habits-1 to 12). Most of the A20 gain came from the habit rewards (deck size,
removing Strikes and curses, upgrading rather than resting at high HP) and the win reward.

ironclad-runs-habits-12 itself trained at ascension 20 only, from ironclad-runs-habits-11, for 300 updates of 2,048 games x 128 steps
(about 1.6 hours to update 200): rewards for progress (0.02 a floor, 0.1 an act) and a win (3),
small habit rewards and costs (deck size, resting, upgrading, potions, elites, unspent energy),
practice fights (30% of games; 60% of those are boss fights from winning human A20 runs) and a
small cross-entropy pull towards human A20 winners' choices between fights. Update 200 was its
best checkpoint at A20 (updates 250 and 300 were lower).

## Results

Greedy play (the likeliest legal move every time) on 1,024 fixed seeds per ascension
(`train_runs.py --eval <ckpt> --ascension N --eval-runs 1024`, the same seeds every time; about
±1.2 points at A20, less at A0), measured on 2026-10-21 on main `4850b5e`, the code being
shipped:

| Ascension | Won | Average floor reached | Reached Act 2 | Reached Act 3 |
|---|---|---|---|---|
| 0 | **93.1%** | 49.0 | 99% | 96% |
| 10 | **75.2%** | 45.6 | 96% | 83% |
| 20 | **21.9%** | 35.7 | 82% | 47% |

No run timed out. The A20 number matches the 22.0% measured on the same seeds on the code of
2026-10-19, before the other characters were added: the new code didn't change how it plays.

On the code of 2026-10-19/20 it also won 69.0% of the held-out
human A20 boss fights (`python/human/human_bosses.py`). With the fight look-ahead at play time
(`python/search/fight_search.py`: the network's likeliest fight moves played out on redrawn copies to its
next turn) it won 27.5% at A20 and 76.1% of those boss fights; the look-ahead is not part of this release and was not measured again.

## Limits

- The Ironclad only. Its inputs take the character, but it has only played the Ironclad; the
  Silent, the Defect and the Watcher have their own networks, [silent-v1](silent-v1.md),
  [defect-v1](defect-v1.md) and [watcher-v1](watcher-v1.md).
- It never takes the Sapphire key, so it never goes to Act 4: a win is beating the Act 3 boss
  (both bosses at A20), not the Heart.
- At ascension 20 it wins about one run in five; most of its losses there are boss fights.
- It plays this engine, not the game itself. The engine matches the real game step for step in
  every recording and soak checked, and its runs can be played back in the
  game (`tools/trace/watch.sh`), but it reads the encoding, not the screen.
- Greedy play is deterministic per seed; it can't be asked to explain a move beyond its move
  probabilities and its value.
- A network trained on another encoding version can't be loaded: when the inputs change, a
  network is retrained or carried over (`run_model.upgrade_state`) and released again.
