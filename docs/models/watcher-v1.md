# watcher-v1: a network that plays the Watcher

The first release of the Watcher network, 2026-10-29: `watcher-v1.safetensors` on the GitHub
release `watcher-v1` (the weights file is not in git). The browser viewer offers it as
"Watcher AI v1".

## What it is

The run network of the earlier releases (`python/core/run_model.py`'s `RunNet`: d 128, 4 heads,
171 action slots and a value) with 3 layers, on encoding 2 (`docs/network-inputs.md`: the hand
cards' displayed numbers, the damage against each monster, stance, orbs in slot order, monsters'
last two moves). It was trained from scratch for the Watcher: no other character's weights. The
file is safetensors, weights only, 3.6 MB (888,844 numbers); its metadata says `character:
watcher`, source checkpoint `watcher-runs-1-stance-seeds` update 3600 (943,718,400 steps of
that run), and the encoder signature (`encoding_version` 2).

Like [defect-v1](defect-v1.md), it plays for the Heart: it goes for the three keys and on to
Act 4, and a win means beating the Heart. It was trained at ascension 20 only.

## How it was trained

PPO (`python/training/train_runs.py --character watcher`) from 2026-10-27 to 2026-10-29, each
run from the one before:

- watcher-fights-1 and watcher-fights-2-human60: fights only, from scratch at A20, then 60% of
  them on human A20 winners' boss, elite and Heart fights (their decks, relics and HP, from Mega
  Crit's 2020 metrics dump).
- watcher-human-1: imitation of human A20 Heart winners' choices between fights (251,000
  decisions), and a separate head learning how human runs went from a position (it trains the
  network and plays no part in it: left out of the file).
- watcher-runs-1: whole A20 runs with the Heart: 0.02 a floor, 0.1 an act, 1 for the Act 3 boss,
  9 more for the Heart; habit rewards from the human winners (resting only when hurt, upgrading,
  elites, a deck kept small); a third of games practice fights; imitation of the winners' event
  and boss relic choices beside PPO. Then a lower learning rate.
- watcher-runs-1-keys: a bonus for each key held (0.15), which took runs to Act 4.
- watcher-runs-1-stance-seeds: a fifth of runs start with Mental Fortress+ and Rushdown, and stance
  cards added to them earn a little, so the stance engine gets played; 3,600 updates.

Update 3600 is the release: on 1,024 fixed A20 runs it beat the run's five other candidates from
update 1,800 on (6.8-7.6% Heart wins) and a side run that seeded Mantra cards (4.7%).

## Results

Greedy play on fixed seeds, measured 2026-10-29 (`python/release/win_calibration.py --heart`, 512
runs at each ascension, and `train_runs.py --eval`, 1,024 more at A10 and A20, pooled):

| Ascension | Beat the Heart | Won (the Heart or Act 3 without the keys) | Runs | Average floor | Reached Act 3 | Reached Act 4 |
|---|---|---|---|---|---|---|
| 0 | **43.8%** | | 512 | | | |
| 5 | 38.3% | | 512 | | | |
| 10 | **26.3%** | 44.3% | 1,536 | 47.0 | 81% | 44% |
| 15 | 18.9% | | 512 | | | |
| 20 | **8.0%** | 14.1% | 1,536 | 38.8 | 58% | 16% |

(Won, floors and acts from the `--eval` set only.)

In practice on 3,292 held-out human winners' A20 fights (their decks, relics and HP) it won 82.7%:
Act 1 bosses 87%, Act 2 85%, Act 3 41%, the Heart 41%, elites 91-93%.

**With the fight look-ahead** (`python/search/fight_search.py`, not part of this release: at each
fight move the network's four likeliest moves and ending the turn are played out on eight redrawn
copies by the network itself to its next turn, and its move is replaced where another clearly
scores better) the same 1,024 A20 runs went to 22.6% won against 15.1%, Act 3 reached in 71% and
Act 4 in 22%, and the human Heart fights to 61% against 41% (all human fights 87.8%). It is about
20 times slower per run.

The browser runs the same network: the Rust forward pass (`crates/sts-net`) matches PyTorch's
logits to 5e-5 on 152 positions, and of 16 fixed A10 runs played by both, 15 end the same; the
sixteenth goes another way at a choice between two map paths the network scores the same to
within 1e-5 (both copies beat the Heart). Its chance to beat the Heart comes from a logistic fit of
its run value, floor and ascension on 2,560 of its runs (held-out log loss 0.507 against 0.660 for
the base rate; by ascension within 8 points); the advice panel shows it as
"Chance to beat the Heart".

## Limits

- The Watcher only.
- At A20 it beats the Heart in about one run in twelve; most runs end in Act 2 or 3, and it wins
  only 41% of the human winners' Act 3 boss and Heart fights.
- Its decks lean on stances (Mental Fortress, Rushdown, Inner Peace, Tantrum) and skip much of
  what human A20 winners build around: Omniscience, Lesson Learned, Wish, Fasting, Blasphemy and
  Deva Form were almost never taken when offered at update 2,050 (human winners take them 22-54%
  of the time).
- It plays this engine, not the game itself, and reads the encoding, not the screen.
