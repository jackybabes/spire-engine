# silent-v1: a network that plays the Silent

The first release of the Silent network, 2026-10-24: `silent-v1.safetensors` on the GitHub
release `silent-v1` (the weights file is not in git). The browser viewer offers it as
"Silent AI v1".

## What it is

The same network as [ironclad-v1](ironclad-v1.md) (`python/core/run_model.py`'s `RunNet`: d 128,
6 layers, 4 heads; the same inputs, 171 action slots and value), trained to play the Silent. It
started from the Ironclad network's weights, so it knew the monsters, relics and fights before
it saw a Silent card. The file is safetensors, weights only, 4.5 MB (1,113,228 numbers); its
metadata says `character: silent`, source checkpoint `sl5` (the run since renamed `silent-runs-5`) update 1200 (314,572,800 steps of that
run), and the encoder signature (`encoding_version` 1).

## How it was trained

PPO (`python/training/train_runs.py --character silent`) in five runs from 2026-10-20 to
2026-10-21, each from the one before:

- silent-runs-1: from the Ironclad network, its rewards and settings, runs only, ascensions widening to
  A0-20.
- silent-runs-2 to 3: costs for keeping basic Strikes and curses and for upgrading Strikes, removing a
  Strike credited at once, and shop removal practice: human A20 winners keep about one Strike.
- silent-runs-4 to 5: imitation of human A20 Silent winners' choices between fights (Mega Crit's 2020
  metrics dump), and practice on every boss and elite fight of those winners' runs (their decks,
  relics, potions and HP; 30% of games). From update 601 it trained at A17-20, from update 990 on
  imitation data with rest sites' HP fixed, and from update 1060 with a fifth of its runs at
  A0-16 (so its play below A17 didn't drift).

Update 1200 is its best checkpoint at A20 (with update 1150; later ones were lower).

## Results

Greedy play on fixed seeds (two sets of 512, `train_runs.py --eval` and
`python/release/win_calibration.py`, pooled; 1,536 runs at A20), measured 2026-10-21 to 2026-10-24:

| Ascension | Won | Runs | Average floor reached | Reached Act 2 | Reached Act 3 |
|---|---|---|---|---|---|
| 0 | **75.8%** | 1,024 | 47.1 | 97% | 89% |
| 5 | 61.1% | 512 | | | |
| 10 | **47.9%** | 1,024 | 40.7 | 89% | 66% |
| 15 | 28.1% | 512 | | | |
| 20 | **2.9%** | 1,536 | 25.3 | 56% | 18% |

(Floors and acts from one set: 512 runs at A0 and A10, 1,024 at A20.) It also won 69.9% of 900
held-out boss and elite fights from human A20 Silent winners, played with their decks.

The browser runs the same network: the Rust forward pass (`crates/sts-net`) matches PyTorch's
logits to 2e-5 on 60 positions, and 16 fixed A10 runs played by both end the same (8 won each).
Its chance to win comes from a logistic fit of its run value, floor and ascension on 2,560 of its
runs (held-out log loss 0.454 against 0.688 for the base rate; by ascension within about 8 points).

## Limits

- The Silent only.
- Weak at A20: about one run in 35 wins. Its decks at the Act 1 boss are the main limit (with
  human winners' decks it beats that boss about three times in four).
- It never takes the Sapphire key: a win is beating the Act 3 boss, not the Heart.
- It plays this engine, not the game itself, and reads the encoding, not the screen.
