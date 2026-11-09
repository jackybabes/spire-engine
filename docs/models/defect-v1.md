# defect-v1: a network that plays the Defect

The first release of the Defect network, 2026-10-26: `defect-v1.safetensors` on the GitHub
release `defect-v1` (the weights file is not in git). The browser viewer offers it as
"Defect AI v1".

## What it is

The run network of [ironclad-v1](ironclad-v1.md) and [silent-v1](silent-v1.md)
(`python/core/run_model.py`'s `RunNet`: d 128, 4 heads, 171 action slots and a value), with 8
layers instead of 6, on encoding 2 (`docs/network-inputs.md`: the hand cards' displayed numbers,
the damage against each monster, orbs in slot order, stance, monsters' last two moves). It was
trained from scratch for the Defect: no Ironclad or Silent weights. The file is safetensors,
weights only, 6.2 MB (1,551,244 numbers); its metadata says `character: defect`, source
checkpoint `df10` (the run since renamed `defect-runs-10`) update 600 (157,286,400 steps of that run), and the encoder signature
(`encoding_version` 2).

Unlike the earlier two, it plays for the Heart: it goes for the three keys and on to Act 4, and
a win means beating the Heart.

## How it was trained

PPO (`python/training/train_runs.py --character defect`) in eleven runs from 2026-10-24 to
2026-10-26, each from the one before:

- defect-fights-0 to 1: fights only, from scratch, then boss fights (half of them on human winners' boss
  decks, from Mega Crit's 2020 metrics dump).
- defect-human-1: imitation of human Defect winners' choices between fights (5.7 million decisions).
- defect-runs-2 to 3: whole runs at A0-10, with imitation of human winners' event choices; the reward
  screen never leaves gold behind.
- defect-runs-4 to 7: the Heart: a reward for the Act 3 boss whether or not the run goes on, eight times
  as much for the Heart, and a bonus for each key held that only counts for runs that reach
  Act 4; ascensions widened to A10-20.
- defect-runs-8 to 10: for A20: practice on human winners' A15-20 boss and Heart fights (a fifth of
  games), imitation of A15-20 winners' choices (card rewards weighted three times), and in defect-runs-10
  two more layers.

Update 600 of defect-runs-10 is the release: on 512 fixed runs it was level with the best earlier
checkpoint (defect-runs-9 update 400) at A10 and a little ahead at A20.

## Results

Greedy play on fixed seeds (two sets of 512 at A10 and A20, `train_runs.py --eval` and
`python/release/win_calibration.py --heart`, pooled; one set of 512 at the others), measured
2026-10-26:

| Ascension | Beat the Heart | Won (the Heart or Act 3 without the keys) | Runs | Average floor | Reached Act 3 | Reached Act 4 |
|---|---|---|---|---|---|---|
| 0 | **27.5%** | | 512 | | | |
| 5 | 17.6% | | 512 | | | |
| 10 | **10.4%** | 26.4% | 1,024 | 43.8 | 70% | 35% |
| 15 | 5.1% | | 512 | | | |
| 20 | **0.9%** | 4.3% | 1,024 | 31.1 | 30% | 4% |

(Won, floors and acts from the `--eval` set only.) In practice on held-out human winners' boss
fights (A15-20, their decks, relics and HP) it won 77% of 256: Act 1 bosses 97%, Act 2 83%,
Act 3 78%, the Heart 48%.

The browser runs the same network: the Rust forward pass (`crates/sts-net`) matches PyTorch's
logits to 2e-5 on 152 positions, and 16 fixed A10 runs played by both end the same (2 won, one
of them beating the Heart). Its chance to beat the Heart comes from a logistic fit of its run
value, floor and ascension on 2,560 of its runs (held-out log loss 0.411 against 0.547 for the
base rate; by ascension within 6 points); the advice panel shows it as "Chance to beat the
Heart".

## Limits

- The Defect only.
- Weak at A20: it beats the Heart about once in a hundred runs, and most runs end at the Act 2
  boss (floor 33). Its decks are the limit there: it takes Go for the Eyes, Cold Snap and Chill
  far more often than human A15-20 winners do, and Hologram, Storm and Reinforced Body far
  less; forcing the winners' picks on it made its runs shorter, as its fights don't use those
  cards yet.
- It plays this engine, not the game itself, and reads the encoding, not the screen.
